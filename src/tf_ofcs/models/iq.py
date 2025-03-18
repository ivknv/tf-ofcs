# -*- coding: utf-8 -*-
# Copyright (C) 2025 Ivan Konovalov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Callable
import math
from typing import Any, Optional
import tensorflow as tf
import keras  # type: ignore[import-untyped]

from ..datatypes import Signal, Timing
from ..utils import bits_to_indices, indices_to_bits
from ..utils import dbm_to_w

from ..components import (
    Component, Composition, IQModulator, Laser, IQReceiver, Photodiode,
    BandPassFilter, SignalFromTensor, Lambda, RandomizeParameters,
    MapComponents, Gain, Fork, Identity, Reindex, Substitute,
    AnalogToDigitalConverter, DigitalToAnalogConverter, Downsample, Upsample
)
from ..communication_system import CommunicationSystem

__all__ = ["IQAutoencoder"]


def _clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)


def _to_iq(samples: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(samples)[0]
    sample_count = tf.shape(samples)[-1]
    shape = [batch_size, sample_count]

    return tf.reshape(samples[:, 0, :], shape), tf.reshape(samples[:, 1, :], shape)


def _from_iq_signal(signals: tuple[Signal, Signal]) -> tf.Tensor:
    i, q = signals  # shape=[batch_size, symbols_per_batch * n]
    shape = tf.shape(i.samples)
    batch_size, sample_count = shape[0], shape[1]

    return tf.reshape(tf.stack([i.samples, q.samples], axis=1), [batch_size, 2, sample_count])


# Slices the signal to select a single symbol (made up of `size` samples) at `index * size`
class _SelectSymbol(Component):
    index: tf.Variable  # shape=[]; dtype=tf.int32
    size: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        index: int | tf.Tensor = 0,
        size: int | tf.Tensor = 0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.index = tf.Variable(tf.constant(index), name=f"{self.name}__index")
        self.size = tf.Variable(tf.constant(size), name=f"{self.name}__size")

        self.local_parameters = ["index", "size"]

    def implementation(
        self,
        signal: Signal | Signal,
        /,
        index: tf.Tensor,
        size: tf.Tensor,
        **kwargs
    ) -> Any:
        Signal = signal.__class__

        return Signal(
            timing=Timing(
                sample_count=size,
                sampling_rate=signal.timing.sampling_rate
            ),
            samples=signal.samples[:, index * size:(index + 1) * size]
        )


class TransmissionLayer(keras.layers.Layer):
    def __init__(
        self,
        *,
        M: int,
        N: int,
        n: int,
        sampling_rate: float,
        encoder_network: Callable[[tf.Tensor], tf.Tensor],
        decoder_network: Callable[[tf.Tensor], tf.Tensor],
        enable_phase_noise: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, trainable=False)

        self._channel = Substitute(Identity())
        self._component_parameters: dict[str, Any] = {}
        self._parameter_distributions: dict[str, Any] = {}
        self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self._M = M
        self._N = N
        self._n = n
        self._sampling_rate = sampling_rate

        bandwidth = 32e9 # Hz

        lpf = BandPassFilter(min_frequency=[0.0], max_frequency=[bandwidth])

        adc_sampling_rate = dac_sampling_rate = 84e9
        dsp_sampling_rate = sampling_rate
        dac_max_power = 1.0
        dac_step = dac_max_power / 256.0
        adc_max_power = dbm_to_w(2.0)
        adc_step = adc_max_power / 256.0

        adc = AnalogToDigitalConverter(
            sampling_rate=adc_sampling_rate,
            step=adc_step,
            min_level=0.0,
            max_level=adc_max_power
        )

        dac = DigitalToAnalogConverter(
            sampling_rate=dac_sampling_rate,
            step=dac_step,
            min_level=0.0,
            max_level=dac_max_power
        )

        self._laser = Substitute(Laser(power=dbm_to_w([2.0]), enable_phase_noise=enable_phase_noise))
        self._local_oscillator = Substitute(
            Laser(
                power=dbm_to_w([-6.0]),
                enable_phase_noise=enable_phase_noise,
                name="local_oscillator"
            )
        )

        self._transmitter = Composition([
            MapComponents(
                Composition([
                    lpf,
                    Downsample(int(dsp_sampling_rate / dac_sampling_rate)),
                    dac,
                    Upsample(int(dsp_sampling_rate / dac_sampling_rate))
                ])
            ),
            Fork(shape=[2]),
            MapComponents([
                Composition([
                    Reindex(indices=(0,)),
                    self._laser
                ]),
                Identity()
            ]),
            IQModulator(),
        ], name="transmitter")

        self._receiver = Composition([
            IQReceiver(
                Photodiode(
                    maximum_power=dbm_to_w(tf.constant([3.0])),
                    dark_current=dbm_to_w(tf.constant([-20.0]))
                ),
                self._local_oscillator
            ),
            MapComponents(Composition([
                adc,
                Gain(gain=[1e5]),
                Upsample(int(dsp_sampling_rate / dac_sampling_rate)),
                lpf
            ])),
        ], name="receiver")

        self._communication_system = Composition([
            Lambda(_to_iq, name="to_iq"),
            MapComponents(SignalFromTensor(sampling_rate=sampling_rate)),
            self._transmitter,
            self._channel,
            self._receiver,
            MapComponents(_SelectSymbol(name="select_symbol")),
            Lambda(_from_iq_signal)
        ])

        self._system_with_randomization = RandomizeParameters(
            self._communication_system,
            shape=[1],
            distributions=self._parameter_distributions,
            rng=self.rng
        )

    def _set_channel(self, channel: Component) -> None:
        self._channel.substitute = channel

    def _precompute_lasers(self) -> None:
        """
            Replace laser by a function that returns a constant precomputed signal.
            This can speed up training a bit, but requires the assumption that
            laser signal doesn't change inbetween steps.
        """

        def laser_substitute(laser: Substitute) -> Callable:
            timing = Timing(sampling_rate=self._sampling_rate, sample_count=self._n // 2 * self._N)
            params = dict(self._component_parameters)
            params["batch_size"] = tf.constant(1)
            signal = laser.component(timing, **params)

            return lambda *args, **kwargs: Signal(
                samples=tf.tile(signal.samples, [kwargs["batch_size"], 1]),
                timing=signal.timing
            )

            return lambda *args, **kwargs: signal

        self._laser.substitute = laser_substitute(self._laser)
        self._local_oscillator.substitute = laser_substitute(self._local_oscillator)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        sample_count = tf.shape(inputs)[-1]

        adjacent_inputs = tf.gather(
            inputs,
            self.rng.uniform(((self._N - 1) * batch_size,), 0, batch_size, dtype=tf.int32)
        )  # shape=[batch_size * (N - 1), 2, n]

        # shape=[batch_size, 2, (N - 1) * n // 2]
        adjacent_inputs = tf.reshape(adjacent_inputs, [batch_size, 2, tf.size(adjacent_inputs) // batch_size // 2])
        symbol_index = self.rng.uniform([], 0, self._N, dtype=tf.int32)

        if self._N > 1:
            inputs = tf.concat([
                adjacent_inputs[:, :, :symbol_index * sample_count],
                inputs,
                adjacent_inputs[:, :, symbol_index * sample_count:]
            ], -1)

        parameters = dict(self._component_parameters)
        parameters["batch_size"] = batch_size

        return tf.reshape(
            self._system_with_randomization(
                inputs,
                randomize_parameters__shape=tf.expand_dims(batch_size, 0),
                select_symbol__index=symbol_index,
                select_symbol__size=sample_count,
                **parameters
            ),
            [batch_size, 2, sample_count]
        )

    def get_config(self) -> dict:
        return {}


class IQAutoencoderModel(keras.models.Sequential):
    """
        An autoencoder model with an IQ modulator.

        :ivar M: `int`, number of values a single message (symbol) carries
        :ivar n: `int`, number of samples per symbol produced by the transmitter
        :ivar N: `int`, number of adjacent symbols transmitted during training (must be odd)

        :ivar encoder_network: `Component`, transmitter layers of the neural network
        :ivar decoder_network: `Component`, receiver layers of the neural network
    """

    M: int
    n: int
    N: int
    bits_per_symbol: int

    encoder_network: Component
    decoder_network: Component

    def __init__(
        self,
        M: int,
        n: int,
        N: int,
        enable_phase_noise: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = int(math.log2(M))
        self._sampling_rate = 336e9

        #: Size of the input one-hot vector
        self.M = M

        #: Number of samples produced by the final transmitter layer
        self.n = n

        #: Number of ANN blocks
        self.N = N

        self._enable_phase_noise = enable_phase_noise

        if N % 2 != 1:
            raise ValueError("N must be odd")

        if n % 2 != 0:
            raise ValueError("n must be even")

        self._tx1_layer = keras.layers.Dense(2 * M, activation="relu", name="tx1")
        self._tx2_layer = keras.layers.Dense(2 * M, activation="relu", name="tx2")
        self._tx3_layer = keras.layers.Dense(n, activation=_clip, name="tx3")
        self._tx_reshape_layer = keras.layers.Reshape((2, n // 2))

        self.add(keras.Input(shape=(M,)))
        self.add(self._tx1_layer)
        self.add(self._tx2_layer)
        self.add(self._tx3_layer)
        self.add(self._tx_reshape_layer)

        self.encoder_network = Lambda(
            lambda x: self._tx_reshape_layer(self._tx3_layer(self._tx2_layer(self._tx1_layer(x)))),
            name="encoder_network"
        )
        self.decoder_network = Lambda(
            lambda x: self._rx3_layer(self._rx2_layer(self._rx1_layer(self._rx_reshape_layer(x)))),
            name="decoder_network"
        )

        self.encoder = Lambda(self.encode_input)
        self.decoder = Lambda(self.decode_signal)

        self._transmission_layer = TransmissionLayer(
            M=M,
            N=N,
            n=n,
            sampling_rate=self._sampling_rate,
            encoder_network=self.encoder_network,
            decoder_network=self.decoder_network,
            enable_phase_noise=enable_phase_noise,
            name="transmission"
        )

        self.add(self._transmission_layer)

        self._rx_reshape_layer = keras.layers.Reshape((n,), name="rx_reshape_layer")
        self._rx1_layer = keras.layers.Dense(2 * M, activation="relu", name="rx1")
        self._rx2_layer = keras.layers.Dense(2 * M, activation="relu", name="rx2")
        self._rx3_layer = keras.layers.Dense(M, activation="softmax", name="rx3")

        self.add(self._rx_reshape_layer)
        self.add(self._rx1_layer)
        self.add(self._rx2_layer)
        self.add(self._rx3_layer)

    def encode_input(self, bitseq: tf.Tensor) -> tuple[Signal, Signal]:
        bitseq_size = tf.size(bitseq)

        inputs = tf.one_hot(
            tf.reshape(
                bits_to_indices(bitseq, tf.constant(self.bits_per_symbol)),
                [bitseq_size // self.bits_per_symbol, 1]
            ),
            self.M
        )
        x = self.encoder_network(inputs)  # shape=[batch_size * symbols_per_batch, 2, n // 2]

        bits_per_batch = tf.shape(bitseq)[1]
        batch_size = tf.shape(bitseq)[0]
        symbols_per_batch = bits_per_batch // self.bits_per_symbol

        i_seq = x[:, 0, :]
        q_seq = x[:, 1, :]

        i_seq = tf.reshape(i_seq, [batch_size, self.n // 2 * symbols_per_batch])
        q_seq = tf.reshape(q_seq, [batch_size, self.n // 2 * symbols_per_batch])

        timing = Timing(sample_count=tf.shape(i_seq)[-1], sampling_rate=self._sampling_rate)

        return Signal(samples=i_seq, timing=timing), Signal(samples=q_seq, timing=timing)

    def decode_signal(self, input_signal: tuple[Signal, Signal]) -> tf.Tensor:
        samples = _from_iq_signal(input_signal)

        samples_per_batch = tf.shape(samples)[-1]
        batch_size = tf.shape(samples)[0]
        symbols_per_batch = samples_per_batch // (self.n // 2)

        i_seq = samples[:, 0, :]
        q_seq = samples[:, 1, :]

        i_seq = tf.reshape(i_seq, [batch_size * symbols_per_batch, self.n // 2])
        q_seq = tf.reshape(q_seq, [batch_size * symbols_per_batch, self.n // 2])

        samples = tf.stack([i_seq, q_seq], axis=1)

        x = self.decoder_network(samples)  # shape=[batch_size, M]
        bitseq = indices_to_bits(
            tf.cast(tf.argmax(x, axis=1), tf.int32),
            tf.constant(self.bits_per_symbol)
        )

        return tf.reshape(bitseq, (batch_size, tf.size(bitseq) // batch_size))

    def _setup(
        self,
        channel: Component,
        component_parameters: dict[str, Any],
        parameter_distributions: dict[str, Any]
    ) -> None:
        self._transmission_layer._set_channel(channel)

        self._transmission_layer._component_parameters.clear()
        self._transmission_layer._component_parameters.update(component_parameters)

        self._transmission_layer._parameter_distributions.clear()
        self._transmission_layer._parameter_distributions.update(parameter_distributions)

        if not self._enable_phase_noise:
            self._transmission_layer._precompute_lasers()


class IQAutoencoder(CommunicationSystem):
    def __init__(
        self,
        M: int,
        n: int,
        N: int,
        enable_phase_noise: bool = False,
        name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "iq_autoencoder_system"

        model = IQAutoencoderModel(
            M=M,
            n=n,
            N=N,
            enable_phase_noise=enable_phase_noise,
            name=f"{name}_model"
        )

        super().__init__(
            bits_per_symbol=model.bits_per_symbol,
            encoder=Lambda(model.encode_input),
            decoder=Lambda(model.decode_signal),
            transmitter=model._transmission_layer._transmitter,
            receiver=model._transmission_layer._receiver,
            model=model,
            name=name
        )

    def setup_model(
        self,
        channel: Component,
        component_parameters: dict[str, Any],
        parameter_distributions: dict[str, Any]
    ) -> None:
        assert isinstance(self.model, IQAutoencoderModel)
        self.model._setup(channel, component_parameters, parameter_distributions)
