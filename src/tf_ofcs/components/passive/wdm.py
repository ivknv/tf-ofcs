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

from typing import Optional
import tensorflow as tf

from ...utils import fftfreq, to_tensor
from ...datatypes import FrequencyResponseFunction, OpticalSignal, TensorCompatible
from ..component import Component

__all__ = ["OpticalMultiplexer", "OpticalDemultiplexer"]


class OpticalMultiplexer(Component):
    """
        Multiplexes multiple optical signals into one.
        Produces `batch_size // n` signals at the output.
    """

    #: Number of signals to multiplex (number of input ports)
    n: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(self, n: TensorCompatible, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = tf.Variable(to_tensor(n), name=f"{self.name}__n")
        self.local_parameters = ["n"]

    def implementation(self, input_signal: OpticalSignal, /, n: tf.Tensor) -> OpticalSignal:
        shape = tf.shape(input_signal.samples)
        batch_size = shape[0]

        tf.debugging.assert_equal(tf.math.floormod(batch_size, n), 0)

        output_samples = tf.math.reduce_sum(
            tf.reshape(
                input_signal.samples,
                tf.concat([[batch_size // n, n], shape[1:]], 0)
            ),
            axis=1
        ) / tf.math.sqrt(tf.complex(n, 0.0))

        return OpticalSignal(timing=input_signal.timing, samples=output_samples)


class OpticalDemultiplexer(Component):
    """
        Demultiplexes one optical signal into multiple based on frequencies.

        Works similar to `SerialToParallel` in a sense that it produces
        `batch_size * n` signals at the output.

        Every output signal corresponds to some specific frequency.
    """

    n_outputs: tf.Variable  # shape=[]; dtype=tf.int32

    #: Optical filters for each corresponding output port
    frequency_response_function: FrequencyResponseFunction

    def __init__(
        self,
        n_outputs: TensorCompatible,
        frequency_response_function: FrequencyResponseFunction,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.n_outputs = tf.Variable(to_tensor(n_outputs), name=f"{self.name}__n_outputs")
        self.frequency_response_function = frequency_response_function

        self.local_parameters = ["n_outputs"]

    def call(
        self,
        input_signal: OpticalSignal,
        /,
        **kwargs
    ) -> OpticalSignal:
        fft = tf.signal.fft(input_signal.samples)
        fft_freqs = fftfreq(tf.shape(fft)[-1], 1.0 / input_signal.timing.sampling_rate)

        n_outputs = self.get_component_parameter(kwargs, "n_outputs")
        batch_size = tf.shape(fft)[0]

        # We need override these parameters, so we drop them from kwargs
        kwargs.pop("batch_size", None)

        frequency_response = tf.tile(
            self.frequency_response_function(
                fft_freqs,
                batch_size=n_outputs,
                **kwargs
            ),
            [batch_size, 1]
        ) * tf.math.sqrt(tf.complex(n_outputs, 0.0))

        fft = tf.tile(fft, [n_outputs, 1])
        samples = tf.signal.ifft(frequency_response * fft)

        return OpticalSignal(timing=input_signal.timing, samples=samples)
