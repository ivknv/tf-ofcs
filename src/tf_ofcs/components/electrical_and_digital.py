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

from ..datatypes import Signal, TensorCompatible, Timing
from ..utils import adjust_symbol_for_mzm, to_tensor
from .component import Component

__all__ = [
    "Adder",
    "Subtractor",
    "SignalFromTensor",
    "Gain",
    "DigitalToAnalogConverter",
    "AnalogToDigitalConverter",
    "MZMPreDistortion",
    "PhotodiodePreDistortion"
]


class Adder(Component):
    """
        Adds two electrical or digital signals together.
        Takes a tuple of two signals as input and returns a single
        `Signal` with the sum of the two signals.
    """

    def implementation(self, input_signals: tuple[Signal, Signal], /) -> Signal:
        assert input_signals[0].samples.dtype is tf.float32
        assert input_signals[1].samples.dtype is tf.float32

        return Signal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples + input_signals[1].samples
        )


class Subtractor(Component):
    """
        Subtracts two electrical or digital signals.
        Takes a tuple of two signals as input and returns a single
        `Signal` with the difference of the two signals.
    """

    def implementation(self, input_signals: tuple[Signal, Signal], /) -> Signal:
        assert input_signals[0].samples.dtype is tf.float32
        assert input_signals[1].samples.dtype is tf.float32

        return Signal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples - input_signals[1].samples
        )


class SignalFromTensor(Component):
    """
        Constructs a `Signal` instance from a tensor with specified sampling rate.

        :ivar sampling_rate: `shape=[], dtype=tf.float32`, sampling rate of the `Signal`
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

    def implementation(self, input_samples: tf.Tensor, /, sampling_rate: tf.Tensor) -> Signal:
        return Signal(
            timing=Timing(
                sampling_rate=sampling_rate,
                sample_count=tf.shape(input_samples)[-1]
            ),
            samples=input_samples
        )


class Gain(Component):
    """
        Multiplies the input electrical or digital signal by a `gain` factor.

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

    def implementation(self, signal: Signal, /, gain: tf.Tensor) -> Signal:
        assert signal.samples.dtype is tf.float32

        gain = tf.expand_dims(gain, -1)

        return Signal(
            timing=signal.timing,
            samples=signal.samples * gain
        )


class DigitalToAnalogConverter(Component):
    """
        Converts a digital signal to an analog electrical signal.
        Takes a digital `Signal` as input and returns an electrical `Signal`
        with the corresponding analog voltage levels.

        :ivar step: `shape=[None], dtype=tf.float32`, amplitude step for each digital level
        :ivar min_level: `shape=[None], dtype=tf.float32`, minimum signal level
        :ivar max_level: `shape=[None], dtype=tf.float32`, maximum signal level
        :ivar sampling_rate: `shape=[], dtype=tf.float32`, sampling rate of the output signal in Hz
    """

    step: tf.Variable  # shape=[None]; dtype=tf.float32
    min_level: tf.Variable  # shape=[None]; dtype=tf.float32
    max_level: tf.Variable  # shape=[None]; dtype=tf.float32
    sampling_rate: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        step: TensorCompatible,
        sampling_rate: TensorCompatible,
        min_level: TensorCompatible = 0.0,
        max_level: TensorCompatible = 1.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.step = tf.Variable(to_tensor(step), name=f"{self.name}__step")
        self.min_level = tf.Variable(to_tensor(min_level), name=f"{self.name}__min_level")
        self.max_level = tf.Variable(to_tensor(max_level), name=f"{self.name}__max_level")
        self.sampling_rate = tf.Variable(to_tensor(sampling_rate), name=f"{self.name}__sampling_rate")
        self.local_parameters = ["step", "min_level", "max_level", "sampling_rate"]

    def implementation(
        self,
        digital_signal: Signal,
        /,
        step: tf.Tensor,
        min_level: tf.Tensor,
        max_level: tf.Tensor,
        sampling_rate: tf.Tensor
    ) -> Signal:
        assert digital_signal.samples.dtype is tf.float32

        step = tf.expand_dims(step, -1)
        min_level = tf.expand_dims(min_level, -1)
        max_level = tf.expand_dims(max_level, -1)

        upsample_factor = tf.cast(
            sampling_rate / digital_signal.timing.sampling_rate,
            tf.int32
        )

        tf.debugging.assert_less(
            tf.abs(
                tf.subtract(
                    tf.cast(upsample_factor, tf.float32),
                    sampling_rate / digital_signal.timing.sampling_rate
                )
            ),
            1e-6,
            message="Got fractional upsampling factor"
        )

        analog_samples = tf.repeat(
            tf.clip_by_value(
                tf.round(digital_signal.samples / step) * step,
                min_level,
                max_level
            ),
            upsample_factor,
            axis=-1
        )

        return Signal(
            samples=analog_samples,
            timing=Timing(
                sample_count=tf.shape(analog_samples)[-1],
                sampling_rate=sampling_rate
            )
        )


class AnalogToDigitalConverter(Component):
    """
        Converts an analog electrical signal to a digital signal.
        Takes an electrical `Signal` as input and returns an digital `Signal`
        with the corresponding digital levels.

        :ivar step: `shape=[None], dtype=tf.float32`, amplitude step for each digital level
        :ivar min_level: `shape=[None], dtype=tf.float32`, minimum signal level
        :ivar max_level: `shape=[None], dtype=tf.float32`, maximum signal level
        :ivar sampling_rate: `shape=[], dtype=tf.float32`, sampling rate of the output digital signal in Hz
    """

    step: tf.Variable  # shape=[None]; dtype=tf.float32
    min_level: tf.Variable  # shape=[None]; dtype=tf.float32
    max_level: tf.Variable  # shape=[None]; dtype=tf.float32
    sampling_rate: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        step: TensorCompatible,
        sampling_rate: TensorCompatible,
        min_level: TensorCompatible = 0.0,
        max_level: TensorCompatible = 1.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.step = tf.Variable(to_tensor(step), name=f"{self.name}__step")
        self.min_level = tf.Variable(to_tensor(min_level), name=f"{self.name}__min_level")
        self.max_level = tf.Variable(to_tensor(max_level), name=f"{self.name}__max_level")
        self.sampling_rate = tf.Variable(to_tensor(sampling_rate), name=f"{self.name}__sampling_rate")
        self.local_parameters = ["step", "min_level", "max_level", "sampling_rate"]

    def implementation(
        self,
        analog_signal: Signal,
        /,
        step: tf.Tensor,
        min_level: tf.Tensor,
        max_level: tf.Tensor,
        sampling_rate: tf.Tensor
    ) -> Signal:
        assert analog_signal.samples.dtype is tf.float32

        step = tf.expand_dims(step, -1)
        min_level = tf.expand_dims(min_level, -1)
        max_level = tf.expand_dims(max_level, -1)

        downsample_factor = tf.cast(
            analog_signal.timing.sampling_rate / sampling_rate,
            tf.int32
        )

        tf.debugging.assert_less(
            tf.abs(
                tf.subtract(
                    tf.cast(downsample_factor, tf.float32),
                    analog_signal.timing.sampling_rate / sampling_rate
                )
            ),
            1e-6,
            message="Got fractional downsampling factor"
        )

        digital_samples = tf.round(
            tf.clip_by_value(
                analog_signal.samples[:, ::downsample_factor],
                min_level,
                max_level
            ) / step
        ) * step

        return Signal(
            samples=digital_samples,
            timing=Timing(
                sample_count=tf.shape(digital_samples)[-1],
                sampling_rate=sampling_rate
            )
        )


class MZMPreDistortion(Component):
    """
        Applies pre-distortion to the input digital signal to compensate for
        the non-linearity of the Mach-Zehnder Modulator (MZM).
    """

    def implementation(self, input_signal: Signal, /) -> Signal:
        assert input_signal.samples.dtype is tf.float32

        return Signal(
            samples=adjust_symbol_for_mzm(input_signal.samples),
            timing=input_signal.timing
        )


class PhotodiodePreDistortion(Component):
    """
        Applies pre-distortion to the input digital signal to compensate for
        the non-linearity of the photodiode.
    """

    def implementation(self, input_signal: Signal, /) -> Signal:
        assert input_signal.samples.dtype is tf.float32

        return Signal(
            samples=tf.sqrt(input_signal.samples),
            timing=input_signal.timing
        )
