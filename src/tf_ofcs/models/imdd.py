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
    Component, Composition, MachZehnderModulator, Laser, Photodiode,
    BandPassFilter, SignalFromTensor, Lambda, RandomizeParameters,
    Gain, ExtractSamples, Fork, Identity, MapComponents, Substitute,
    AnalogToDigitalConverter, DigitalToAnalogConverter, Downsample, Upsample,
    TruncateSignal
)

from ..communication_system import CommunicationSystem

__all__ = ["IMDDAutoencoder"]


def _clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)


class TransmissionLayer(keras.layers.Layer):
    def __init__(
        self,
        *,
        M: int,
        N_tx: int,
        N_rx: int,
        n: int,
        sampling_rate: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, trainable=False)

        self._N_tx = N_tx
        self._N_rx = N_rx
        self._M = M
        self._n = n
        self._sampling_rate = sampling_rate
        self._component_parameters: dict[str, Any] = {}
        self._parameter_distributions: dict[str, Any] = {}
        self._rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        bandwidth = 32e9 # Hz

        self._lpf = BandPassFilter(min_frequency=[0.0], max_frequency=[bandwidth])
        self._gain = Gain(gain=[1e5])
        self._laser = Substitute(Laser(power=dbm_to_w([2.0])))

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

        self._transmitter = Composition([
            self._lpf,
            Downsample(int(dsp_sampling_rate / dac_sampling_rate)),
            dac,
            Upsample(int(dsp_sampling_rate / dac_sampling_rate)),
            Fork(shape=[2]),
            MapComponents([self._laser, Identity()]),
            MachZehnderModulator()
        ], name="transmitter")

        self._channel = Substitute(Identity())

        self._receiver = Composition([
            Photodiode(
                maximum_power=dbm_to_w(tf.constant([3.0])),
                dark_current=dbm_to_w(tf.constant([-20.0]))
            ),
            adc,
            self._lpf,
            self._gain
        ], name="receiver")

        # Number of samples to truncate from the received signal.
        # These samples are used to induce intersymbol interference (ISI)
        # After the signal is received they have to be removed
        isi_samples = n * (N_tx - 1) // 8
        isi_samples_to_leave = n * (N_rx - 1) // 8
        isi_samples_to_truncate = isi_samples - isi_samples_to_leave

        self._communication_system = Composition([
            SignalFromTensor(sampling_rate=self._sampling_rate),
            self._transmitter,
            self._channel,
            self._receiver,
            TruncateSignal(left=isi_samples_to_truncate, right=isi_samples_to_truncate),
            ExtractSamples()
        ])

        self._system_with_randomization = RandomizeParameters(
            self._communication_system,
            shape=[1],
            distributions=self._parameter_distributions,
            rng=self._rng
        )

    def _set_channel(self, channel: Component) -> None:
        self._channel.substitute = channel

    def _precompute_laser(self) -> None:
        """
            Replace laser by a function that returns a constant precomputed signal.
            This can speed up training a bit, but requires the assumption that
            laser signal doesn't change inbetween steps.
        """

        def laser_substitute() -> Callable:
            timing = Timing(sampling_rate=self._sampling_rate, sample_count=self._n * self._N_tx)
            params = dict(self._component_parameters)
            params["batch_size"] = tf.constant(1)
            signal = self._laser.component(timing, **params)

            return lambda *args, **kwargs: Signal(
                samples=tf.tile(signal.samples, [kwargs["batch_size"], 1]),
                timing=signal.timing
            )

        self._laser.substitute = laser_substitute()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        sample_count = tf.shape(inputs)[1]

        adjacent_inputs = tf.gather(
            inputs,
            self._rng.uniform(((self._N_tx - 1) * batch_size,), 0, batch_size, dtype=tf.int32)
        )  # shape=[batch_size * (N - 1), n]

        # shape=[batch_size, (N - 1) * n]
        adjacent_inputs = tf.reshape(adjacent_inputs, [batch_size, tf.size(adjacent_inputs) // batch_size])
        central_index = (self._N_tx - 1) // 2 * sample_count

        if self._N_tx > 1:
            inputs = tf.concat([
                adjacent_inputs[:, :central_index],
                inputs,
                adjacent_inputs[:, central_index:]
            ], -1)

        parameters = dict(self._component_parameters)
        parameters["batch_size"] = batch_size

        return tf.reshape(
            self._system_with_randomization(
                inputs,
                randomize_parameters__shape=tf.expand_dims(batch_size, 0),
                **parameters
            ),
            (batch_size, sample_count * self._N_rx // 4)
        )

    def get_config(self) -> dict:
        return {}


class IMDDAutoencoderModel(keras.models.Sequential):
    """
        An IM/DD autoencoder model based on [1]. It consists of 3 sections:
        1) Transmitter;
        2) Channel (in this case referred to as transmission layer);
        3) Receiver.

        :ivar M: `int`, number of values a single message (symbol) carries
        :ivar n: `int`, number of samples per symbol produced by the transmitter
        :ivar N_tx: `int`, number of symbols transmitted during training (must be odd)
        :ivar N_rx: `int`, number of symbols processed by the receiver (must be odd)

        [1] 2018. Karanov et al. End-to-End Deep Learning of Optical Fiber Communications.
    """

    bits_per_symbol: int

    #: Size of the input one-hot vector
    M: int

    #: Number of samples produced by the final transmitter layer
    n: int

    #: Total number of symbols transmitted at once. All symbols adjacent to
    #: the center symbol are used to model ISI and are discarded afterwards
    N_tx: int

    #: Number of symbols processed by the receiveer at once. Only the central
    #: symbol is fully decoded, the adjacent symbols are used to help mitigate ISI
    N_rx: int

    def __init__(
        self,
        M: int,
        n: int,
        N_tx: int,
        N_rx: int = 5,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.bits_per_symbol = int(math.log2(M))
        self._sampling_rate = 336e9

        self.M = M
        self.n = n
        self.N_tx = N_tx
        self.N_rx = N_rx

        if N_tx % 2 != 1:
            raise ValueError("N must be odd")

        self._tx1_layer = keras.layers.Dense(2 * M, activation="relu", name="tx1")
        self._tx2_layer = keras.layers.Dense(2 * M, activation="relu", name="tx2")
        self._tx3_layer = keras.layers.Dense(n, activation=_clip, name="tx3")

        self.encoder_network = Lambda(
            lambda x: self._tx3_layer(self._tx2_layer(self._tx1_layer(x))),
            name="encoder_network"
        )
        self.decoder_network = Lambda(
            lambda x: self._rx3_layer(self._rx2_layer(self._rx1_layer(x))),
            name="decoder_network"
        )

        self._rx1_layer = keras.layers.Dense(2 * M, activation="relu", name="rx1")
        self._rx2_layer = keras.layers.Dense(2 * M, activation="relu", name="rx2")
        self._rx3_layer = keras.layers.Dense(M, activation="softmax", name="rx3")

        self._transmission_layer = TransmissionLayer(
            M=M,
            N_tx=N_tx,
            N_rx=N_rx,
            n=n,
            sampling_rate=self._sampling_rate,
            name="transmission"
        )

        self.add(keras.Input(shape=(M,)))
        self.add(self._tx1_layer)
        self.add(self._tx2_layer)
        self.add(self._tx3_layer)

        self.add(self._transmission_layer)

        self.add(self._rx1_layer)
        self.add(self._rx2_layer)
        self.add(self._rx3_layer)

    def encode_input(self, bitseq: tf.Tensor) -> Signal:
        batch_size = tf.shape(bitseq)[0]

        inputs = tf.one_hot(
            tf.expand_dims(
                bits_to_indices(bitseq, tf.constant(self.bits_per_symbol)),
                -1
            ),
            self.M
        )  # shape=[batch_size, bitseq_size // batch_size // self.bits_per_symbol, self.M]

        # Mirror inputs on both sides to add (N_rx - 1) // 2 adjacent symbols
        adjacent_symbols_count = (self.N_rx - 1) // 2
        inputs = tf.concat(
            [
                inputs[:, :adjacent_symbols_count],
                inputs,
                inputs[:, tf.shape(inputs)[1] - adjacent_symbols_count:]
            ],
            1
        )  # shape=[batch_size, bitseq_size // self.bits_per_symbol + N_rx - 1, self.M]
        inputs = tf.reshape(inputs, [tf.size(inputs) // self.M, self.M])

        samples = self.encoder_network(inputs)
        samples = tf.reshape(samples, [batch_size, tf.size(samples) // batch_size])

        return Signal(
            samples=samples,
            timing=Timing(
                sample_count=self.n * self.N_rx,
                sampling_rate=self._sampling_rate
            )
        )

    def decode_signal(self, input_signal: Signal) -> tf.Tensor:
        batch_size = tf.shape(input_signal.samples)[0]

        samples_per_batch = self.n // 4 * self.N_rx

        # shape=[batch_size, symbol_count, samples_per_batch]
        samples = tf.signal.frame(input_signal.samples, samples_per_batch, self.n // 4, axis=1)
        samples = tf.reshape(samples, (tf.size(samples) // samples_per_batch, samples_per_batch))

        x = self.decoder_network(samples)  # shape=[None, M]

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

        self._transmission_layer._precompute_laser()


class IMDDAutoencoder(CommunicationSystem):
    def __init__(
        self,
        M: int,
        n: int,
        N: int,
        N_rx: int = 1,
        name: Optional[str] = None
    ) -> None:
        if name is None:
            name = "imdd_autoencoder_system"

        model = IMDDAutoencoderModel(M=M, n=n, N_tx=N, N_rx=N_rx, name=f"{name}_model")

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
        assert isinstance(self.model, IMDDAutoencoderModel)
        self.model._setup(channel, component_parameters, parameter_distributions)
