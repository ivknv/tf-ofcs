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

from collections.abc import Mapping
import math
from typing import Any, Optional
import tensorflow as tf
import keras  # type: ignore[import-untyped]

from ..datatypes import ElectricalSignal, OpticalSignal, Timing
from ..utils import bits_to_indices, indices_to_bits
from ..utils import dbm_to_w

from ..components import (
    Component, Composition, OpticalFiber, OpticalAWGN, IQModulator, Laser,
    IQReceiver, Photodiode, BandPassFilter, ArbitraryWaveformGenerator, Lambda,
    RandomizeParameters, MapComponents, ElectricalGain, Fork, Identity, Reindex
)

__all__ = ["IQAutoencoder"]


@tf.function
def _clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)


def _to_iq(samples: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(samples)[0]
    sample_count = tf.shape(samples)[-1]
    shape = [batch_size, sample_count]

    return tf.reshape(samples[:, 0, :], shape), tf.reshape(samples[:, 1, :], shape)


def _from_iq_signal(signals: tuple[ElectricalSignal, ElectricalSignal]) -> tf.Tensor:
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
        signal: ElectricalSignal | OpticalSignal,
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
    #: Arbitrary additional parameters for the system
    parameters: dict[str, Any]

    def __init__(
        self,
        autoencoder: "IQAutoencoder",
        parameters: dict[str, Any],
        parameter_distributions: Mapping[str, Mapping[str, Any]],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, trainable=False)

        M = autoencoder.M
        N = autoencoder.N

        self.parameters = parameters

        self._communication_system = Composition([
            autoencoder.transmitter,
            autoencoder.channel,
            autoencoder.receiver,
            MapComponents(_SelectSymbol(name="select_symbol")),
            Lambda(_from_iq_signal)
        ])

        self._system_with_randomization = RandomizeParameters(
            self._communication_system,
            shape=[1],
            distributions=parameter_distributions
        )

        self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self._N = N
        self._M = M

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

        return tf.reshape(
            self._system_with_randomization(
                inputs,
                randomize_parameters__shape=tf.expand_dims(batch_size, 0),
                batch_size=1,
                parameter_dims=batch_size,
                select_symbol__index=symbol_index,
                select_symbol__size=sample_count,
                **self.parameters
            ),
            [batch_size, 2, sample_count]
        )

    def get_config(self) -> dict:
        return {}


class IQAutoencoder(keras.models.Sequential):
    """
        An autoencoder model with an IQ modulator.

        :ivar M: `int`, number of values a single message (symbol) carries
        :ivar n: `int`, number of samples per symbol produced by the transmitter
        :ivar N: `int`, number of adjacent symbols transmitted during training (must be odd)
        :ivar sampling_rate: `float`, sampling rate of the system in samples/second

        :ivar encoder: `Component`, component that encodes a bit sequence into a tensor
        :ivar encoder_network: `Component`, transmitter layers of the neural network
        :ivar transmitter: `Component`, component that converts `ElectricalSignal` into an `OpticalSignal`
        :ivar channel: `Component`, component that represents the channel
        :ivar receiver: `Component`, component that converts an `OpticalSignal` into `ElectricalSignal`
        :ivar decoder: `Component`, component that decodes the `ElectricalSignal` into a bit sequence
        :ivar decoder_network: `Component`, receiver layers of the neural network
    """

    M: int
    n: int
    N: int

    sampling_rate: float

    encoder_network: Component
    decoder_network: Component

    encoder: Component
    decoder: Component
    transmitter: Component
    channel: Component
    receiver: Component

    def __init__(
        self,
        M: int,
        n: int,
        N: int,
        transmission_layer_parameters: dict[str, Any],
        parameter_distributions: Mapping[str, Mapping[str, Any]],
        name: Optional[str] = None
    ) -> None:
        self._bits_per_symbol = int(math.log2(M))
        self.sampling_rate = 336e9

        #: Size of the input one-hot vector
        self.M = M

        #: Number of samples produced by the final transmitter layer
        self.n = n

        #: Number of ANN blocks
        self.N = N

        if N % 2 != 1:
            raise ValueError("N must be odd")

        if n % 2 != 0:
            raise ValueError("n must be even")

        super().__init__(name=name)

        self._tx1_layer = keras.layers.Dense(2 * M, activation="relu", name="tx1")
        self._tx2_layer = keras.layers.Dense(2 * M, activation="relu", name="tx2")
        self._tx3_layer = keras.layers.Dense(n, activation=_clip, name="tx3")
        self._tx_reshape_layer = keras.layers.Reshape((2, n // 2))

        self.add(keras.Input(shape=(M,)))
        self.add(self._tx1_layer)
        self.add(self._tx2_layer)
        self.add(self._tx3_layer)
        self.add(self._tx_reshape_layer)

        bandwidth = 32e9 # Hz
        central_frequency = 193.414e12

        lpf = MapComponents(
            BandPassFilter(min_frequency=[0.0], max_frequency=[bandwidth]),
            name="lpf"
        )

        awgn = OpticalAWGN(relative_db=[-33.0])
        self.channel = Composition([
            awgn,
            OpticalFiber(
                length=[40.0],
                attenuation=[0.22],
                chromatic_dispersion=[17.0],
                eff_non_linear_coeff=[0.0],
                step_count=1,
                central_frequency=central_frequency
            ),
            awgn
        ])

        self.transmitter = Composition([
            Lambda(_to_iq, name="to_iq"),
            MapComponents(
                ArbitraryWaveformGenerator(sampling_rate=self.sampling_rate),
                name="iq_waveform_generator"
            ),
            lpf,
            Fork(shape=[2]),
            MapComponents([
                Composition([
                    Reindex(indices=(0,)),
                    Laser(power=dbm_to_w(2.0))
                ]),
                Identity()
            ]),
            IQModulator(),
        ], name="transmitter")

        self.receiver = Composition([
            IQReceiver(
                Photodiode(
                    maximum_power=dbm_to_w(tf.constant([3.0])),
                    sensitivity=dbm_to_w(tf.constant([-20.0]))
                ),
                Laser(power=dbm_to_w(tf.constant([-6.0])), name="local_oscillator")
            ),
            lpf,
            MapComponents(ElectricalGain(gain=[1e6]))
        ], name="receiver")

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

        self.add(
            TransmissionLayer(
                autoencoder=self,
                parameters=transmission_layer_parameters,
                parameter_distributions=parameter_distributions,
                name="transmission"
            )
        )

        self._rx_reshape_layer = keras.layers.Reshape((n,), name="rx_reshape_layer")
        self._rx1_layer = keras.layers.Dense(2 * M, activation="relu", name="rx1")
        self._rx2_layer = keras.layers.Dense(2 * M, activation="relu", name="rx2")
        self._rx3_layer = keras.layers.Dense(M, activation="softmax", name="rx3")

        self.add(self._rx_reshape_layer)
        self.add(self._rx1_layer)
        self.add(self._rx2_layer)
        self.add(self._rx3_layer)

    def encode_input(self, bitseq: tf.Tensor) -> tf.Tensor:
        bitseq_size = tf.size(bitseq)

        inputs = tf.one_hot(
            tf.reshape(
                bits_to_indices(bitseq, self._bits_per_symbol),
                [bitseq_size // self._bits_per_symbol, 1]
            ),
            self.M
        )
        x = self.encoder_network(inputs)  # shape=[batch_size * symbols_per_batch, 2, n // 2]

        bits_per_batch = tf.shape(bitseq)[1]
        batch_size = tf.shape(bitseq)[0]
        symbols_per_batch = bits_per_batch // self._bits_per_symbol

        i_seq = x[:, 0, :]
        q_seq = x[:, 1, :]

        i_seq = tf.reshape(i_seq, [batch_size, self.n // 2 * symbols_per_batch])
        q_seq = tf.reshape(q_seq, [batch_size, self.n // 2 * symbols_per_batch])

        return tf.stack([i_seq, q_seq], axis=2)  # shape=[batch_size, 2, n * symbols_per_batch]

    def decode_signal(self, input_signal: tuple[ElectricalSignal, ElectricalSignal]) -> tf.Tensor:
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
        bitseq = indices_to_bits(tf.cast(tf.argmax(x, axis=1), tf.int32), self._bits_per_symbol)

        return tf.reshape(bitseq, (batch_size, tf.size(bitseq) // batch_size))
