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

from typing import Optional
import tensorflow as tf

from ...datatypes import Signal, TensorCompatible, Timing

from ..component import Component

__all__ = ["PadSignal", "UnpadSignal"]


class PadSignal(Component):
    """
        Pads signal with zeros from left and right sides.

        :ivar left_padding: `shape=[]; dtype=tf.int32 or tf.float32`, left padding factor (ratio to signal duration)
        :ivar right_padding: `shape=[]; dtype=tf.int32 or tf.float32`, right padding factor (ratio to signal duration)
    """

    #: Left padding factor (ratio to signal duration)
    left_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    #: Right padding factor (ratio to signal duration)
    right_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    def __init__(
        self,
        left_padding: TensorCompatible,
        right_padding: Optional[TensorCompatible] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if right_padding is None:
            right_padding = left_padding

        self.left_padding = tf.Variable(tf.constant(left_padding), name=f"{self.name}__left_padding")
        self.right_padding = tf.Variable(tf.constant(right_padding), name=f"{self.name}__right_padding")

        self.local_parameters = ["left_padding", "right_padding"]

    def implementation(
        self,
        signal: Signal,
        /,
        left_padding: tf.Tensor,
        right_padding: tf.Tensor
    ) -> Signal:
        shape = tf.shape(signal.samples)
        dtype = left_padding.dtype
        sample_count = tf.cast(shape[1], dtype)

        left_sample_count = tf.cast(sample_count * left_padding, tf.int32)
        right_sample_count = tf.cast(sample_count * right_padding, tf.int32)

        padded_samples = tf.pad(signal.samples, [[0, 0], [left_sample_count, right_sample_count]])
        padded_timing = Timing(
            sample_count=tf.shape(padded_samples)[1],
            sampling_rate=signal.timing.sampling_rate
        )

        return Signal(samples=padded_samples, timing=padded_timing)


class UnpadSignal(Component):
    """
        Unpads signal that was previously padded with zeros from left and right sides.

        :ivar left_padding: `shape=[]; dtype=tf.int32 or tf.float32`, left padding factor (ratio to signal duration)
        :ivar right_padding: `shape=[]; dtype=tf.int32 or tf.float32`, right padding factor (ratio to signal duration)
    """

    #: Left padding factor (ratio to signal duration)
    left_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    #: Right padding factor (ratio to signal duration)
    right_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    def __init__(
        self,
        left_padding: TensorCompatible,
        right_padding: Optional[TensorCompatible] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if right_padding is None:
            right_padding = left_padding

        self.left_padding = tf.Variable(tf.constant(left_padding), name=f"{self.name}__left_padding")
        self.right_padding = tf.Variable(tf.constant(right_padding), name=f"{self.name}__right_padding")

        self.local_parameters = ["left_padding", "right_padding"]

    def implementation(
        self,
        signal: Signal,
        /,
        left_padding: tf.Tensor,
        right_padding: tf.Tensor
    ) -> Signal:
        shape = tf.shape(signal.samples)
        dtype = left_padding.dtype
        sample_count = tf.cast(shape[1], dtype)

        # Figure out sample count of the signal before it was padded
        sample_count = sample_count // (tf.cast(1, dtype) + left_padding + right_padding)

        left_sample_count = tf.cast(sample_count * left_padding, tf.int32)
        right_sample_count = tf.cast(sample_count * right_padding, tf.int32)

        unpadded_samples = signal.samples[:, left_sample_count:shape[1] - right_sample_count]
        unpadded_timing = Timing(
            sample_count=tf.shape(unpadded_samples)[1],
            sampling_rate=signal.timing.sampling_rate
        )

        return Signal(samples=unpadded_samples, timing=unpadded_timing)
