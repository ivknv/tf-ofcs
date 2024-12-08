# -*- coding: utf-8 -*-
# Copyright (C) 2024 Ivan Konovalov

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

from collections.abc import Callable, Mapping
import math
from typing import Any, Optional
import tensorflow as tf
import keras

from .datatypes import ElectricalSignal, Timing
from .utils import bits_to_indices, indices_to_bits
from .utils import dbm_to_w

from .components import (
    Composition, OpticalFiber, OpticalAWGN, MachZehnderModulator, Laser,
    Photodiode, BandPassFilter, ArbitraryWaveformGenerator, Lambda, RandomizeParameters
)

__all__ = ["IMDDAutoencoder"]


@tf.function
def _clip(x: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(x, 0.0, 1.0)


def _slice_signal(signal: ElectricalSignal, start: int, end: int) -> ElectricalSignal:
    return ElectricalSignal(
        timing=Timing(sample_count=end - start, sampling_rate=signal.timing.sampling_rate),
        samples=signal.samples[:, :, start:end]
    )


class TransmissionLayer(keras.layers.Layer):
    #: Arbitrary additional parameters for the system
    parameters: dict[str, Any]

    def __init__(
        self,
        sampling_rate: float | tf.Tensor,
        M: int,
        N: int,
        n: int,
        encoder: Callable[[tf.Tensor], tf.Tensor],
        parameters: dict[str, Any],
        parameter_distributions: Mapping[str, Mapping[str, Any]],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name, trainable=False)

        self.parameters = parameters

        bandwidth = 32e9 # Hz
        central_frequency = 193.414e12

        lpf = BandPassFilter(min_frequency=0.0, max_frequency=bandwidth)

        laser = Laser(power=dbm_to_w(2.0))

        photodiode = Photodiode(
            maximum_power=dbm_to_w(3.0),
            sensitivity=dbm_to_w(-20.0)
        )

        awgn = OpticalAWGN(relative_db=-33.0)

        channel = Composition([
            awgn,
            OpticalFiber(
                length=40.0,
                attenuation=0.22,
                chromatic_dispersion=17.0,
                eff_non_linear_coeff=0.0,
                step_count=1,
                central_frequency=central_frequency
            ),
            awgn
        ])

        self.waveform_generator = ArbitraryWaveformGenerator(sampling_rate=sampling_rate)

        central_index = (N - 1) // 2

        self._communication_system = Composition([
            self.waveform_generator,
            lpf,
            MachZehnderModulator(carrier_source=laser),
            channel,
            photodiode,
            lpf,
            Lambda(function=lambda x: _slice_signal(x, n * central_index, n * (central_index + 1)))
        ])

        self._system_with_randomization = RandomizeParameters(
            self._communication_system,
            shape=[1],
            distributions=parameter_distributions
        )

        self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self._encoder = encoder
        self._N = N
        self._M = M

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        sample_count = tf.shape(inputs)[1]

        adjacent_inputs = self._encoder(
            tf.cast(
                tf.one_hot(
                    self.rng.uniform(((self._N - 1) * batch_size,), 0, self._M, dtype=tf.int32),
                    self._M
                ),
                tf.float32
            )
        )  # shape=[batch_size * (N - 1), n]

        # shape=[batch_size, (N - 1) * n]
        adjacent_inputs = tf.reshape(adjacent_inputs, [batch_size, tf.size(adjacent_inputs) // batch_size])
        central_index = (self._N - 1) // 2 * sample_count

        if self._N > 1:
            inputs = tf.concat([
                adjacent_inputs[:, :central_index],
                inputs,
                adjacent_inputs[:, central_index:]
            ], -1)

        inputs = tf.reshape(inputs, [1, batch_size, self._N * sample_count])

        return tf.reshape(
            self._system_with_randomization(
                inputs,
                randomize_parameters__shape=tf.expand_dims(batch_size, 0),
                batch_size=1,
                parameter_dims=batch_size,
                **self.parameters
            ).samples * 1e6,
            (batch_size, sample_count)
        )


class IMDDAutoencoder(keras.Sequential):
    """
        An IM/DD autoencoder model based on [1]. It consists of 3 sections:
        1) Transmitter;
        2) Channel (in this case referred to as transmission layer);
        3) Receiver.

        :ivar M: `int`, number of values a single message (symbol) carries
        :ivar n: `int`, number of samples per symbol produces by the transmitter
        :ivar N: `int`, number of adjacent symbols transmitted during training (must be odd)
        :ivar sampling_rate: `float`, sampling rate of the system in samples/second

        [1] 2018. Karanov et al. End-to-End Deep Learning of Optical Fiber Communications.
    """

    sampling_rate: float
    M: int
    n: int
    N: int

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

        super().__init__(name=name)

        self._tx1_layer = keras.layers.Dense(2 * M, activation="relu", name="tx1")
        self._tx2_layer = keras.layers.Dense(2 * M, activation="relu", name="tx2")
        self._tx3_layer = keras.layers.Dense(n, activation=_clip, name="tx3")

        self.add(keras.Input(shape=(M,)))
        self.add(self._tx1_layer)
        self.add(self._tx2_layer)
        self.add(self._tx3_layer)

        self.add(
            TransmissionLayer(
                sampling_rate=self.sampling_rate,
                N=N,
                M=M,
                n=n,
                encoder=lambda inp: self._tx3_layer(self._tx2_layer(self._tx1_layer(inp))),
                parameters=transmission_layer_parameters,
                parameter_distributions=parameter_distributions,
                name="transmission"
            )
        )

        self._rx1_layer = keras.layers.Dense(2 * M, activation="relu", name="rx1")
        self._rx2_layer = keras.layers.Dense(2 * M, activation="relu", name="rx2")
        self._rx3_layer = keras.layers.Dense(M, activation="softmax", name="rx3")

        self.add(self._rx1_layer)
        self.add(self._rx2_layer)
        self.add(self._rx3_layer)

    def encode_input(self, bitseq: tf.Tensor) -> ElectricalSignal:
        bitseq_size = tf.size(bitseq)

        inputs = tf.one_hot(
            tf.reshape(
                bits_to_indices(bitseq, self._bits_per_symbol),
                [bitseq_size // self._bits_per_symbol, 1]
            ),
            self.M
        )
        x = self._tx1_layer(inputs)
        x = self._tx2_layer(x)
        x = self._tx3_layer(x)

        bits_per_batch = tf.shape(bitseq)[1]
        batch_size = tf.shape(bitseq)[0]
        symbols_per_batch = bits_per_batch // self._bits_per_symbol
        x = tf.reshape(x, (batch_size, 1, self.n * symbols_per_batch))

        return ElectricalSignal(
            timing=Timing(sampling_rate=self.sampling_rate, sample_count=self.n * symbols_per_batch),
            samples=x
        )

    def decode_signal(self, input_signal: ElectricalSignal) -> tf.Tensor:
        sample_count = tf.size(input_signal.samples)
        batch_size = tf.shape(input_signal.samples)[0]
        parameter_dims = tf.shape(input_signal.samples)[1]

        samples = tf.reshape(input_signal.samples, (sample_count // self.n, self.n))

        x = self._rx1_layer(samples)
        x = self._rx2_layer(x)
        x = self._rx3_layer(x) # shape=[None, M]

        bitseq = indices_to_bits(tf.cast(tf.argmax(x, axis=1), tf.int32), self._bits_per_symbol)

        return tf.reshape(bitseq, (batch_size, parameter_dims, tf.size(bitseq) // batch_size // parameter_dims))
