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

from ..datatypes import TensorCompatible, ElectricalSignal, Timing
from ..utils import to_tensor
from .component import Component

__all__ = [
    "ElectricalAdder",
    "ElectricalSubtractor",
    "ArbitraryWaveformGenerator",
    "Quantizer",
    "ElectricalGain"
]


class ElectricalAdder(Component):
    """
        Adds two electrical signals together.
        Takes a tuple of two electrical signals as input and returns a single
        `ElectricalSignal` with the sum of the two signals.
    """

    def implementation(self, input_signals: tuple[ElectricalSignal, ElectricalSignal], /) -> ElectricalSignal:
        return ElectricalSignal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples + input_signals[1].samples
        )


class ElectricalSubtractor(Component):
    """
        Subtracts two electrical signals.
        Takes a tuple of two electrical signals as input and returns a single
        `ElectricalSignal` with the difference of the two signals.
    """

    def implementation(self, input_signals: tuple[ElectricalSignal, ElectricalSignal], /) -> ElectricalSignal:
        return ElectricalSignal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples - input_signals[1].samples
        )


class ArbitraryWaveformGenerator(Component):
    """
        Generates an arbitrary waveform.
        Takes a tensor of samples as input and returns a `ElectricalSignal`
        with the given samples and sampling rate.

        :ivar sampling_rate: `shape=[], dtype=tf.float32`, sampling rate of the generated waveform in Hz
    """

    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        sampling_rate: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(to_tensor(sampling_rate), name=f"{self.name}__sampling_rate")
        self.local_parameters = ["sampling_rate"]

    def implementation(self, input_samples: tf.Tensor, /, sampling_rate: tf.Tensor) -> ElectricalSignal:
        return ElectricalSignal(
            timing=Timing(
                sampling_rate=sampling_rate,
                sample_count=tf.shape(input_samples)[-1]
            ),
            samples=input_samples
        )


class Quantizer(Component):
    """
        Quantizes the input electrical signal.
        Takes an `ElectricalSignal` as input and returns a `ElectricalSignal`
        with the samples quantized to the given step size.

        :ivar step: `shape=[None], dtype=tf.float32`, quantization step size
    """

    step: tf.Variable  # shape=[None]; dtype=float32

    def __init__(
        self,
        step: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.step = tf.Variable(to_tensor(step), name=f"{self.name}__step")
        self.local_parameters = ["step"]

    def implementation(self, signal: ElectricalSignal, /, step: tf.Tensor) -> ElectricalSignal:
        step = tf.expand_dims(step, -1)
        quantized_samples = tf.round(signal.samples / step) * step

        return ElectricalSignal(timing=signal.timing, samples=quantized_samples)


class ElectricalGain(Component):
    """
        Applies a gain to the input electrical signal.

        :ivar gain: `shape=[None], dtype=tf.float32`, gain to apply to the signal
    """

    gain: tf.Variable  # shape=[None]; dtype=float32

    def __init__(
        self,
        gain: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.gain = tf.Variable(to_tensor(gain), name=f"{self.name}__gain")
        self.local_parameters = ["gain"]

    def implementation(self, signal: ElectricalSignal, /, gain: tf.Tensor) -> ElectricalSignal:
        gain = tf.expand_dims(gain, -1)

        return ElectricalSignal(
            timing=signal.timing,
            samples=signal.samples * gain
        )
