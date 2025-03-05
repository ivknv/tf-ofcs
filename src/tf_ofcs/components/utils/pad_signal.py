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

from ...datatypes import ElectricalSignal, OpticalSignal, Timing

from ..component import Component

__all__ = ["PadSignal"]


class PadSignal(Component):
    """
        Pads signal with zeros from left and right sides.
        If `unpad` is set to True, the padding will be removed from the signal
        instead.
    """

    #: Left padding factor (ratio to signal duration)
    left_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    #: Right padding factor (ratio to signal duration)
    right_padding: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32

    #: If set to `True`, padding will be removed from the signal instead
    unpad: tf.Variable  # shape=[]; dtype=tf.bool

    def __init__(
        self,
        left_padding: float | tf.Tensor,
        right_padding: Optional[float | tf.Tensor] = None,
        unpad: bool | tf.Tensor = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if right_padding is None:
            right_padding = left_padding

        self.left_padding = tf.Variable(tf.constant(left_padding), name=f"{self.name}__left_padding")
        self.right_padding = tf.Variable(tf.constant(right_padding), name=f"{self.name}__right_padding")
        self.unpad = tf.Variable(tf.constant(unpad), name=f"{self.name}__unpad")

        self.local_parameters = ["left_padding", "right_padding", "unpad"]

    def implementation(
        self, signal: ElectricalSignal | OpticalSignal,
        /,
        left_padding: tf.Tensor,
        right_padding: tf.Tensor,
        unpad: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__
        shape = tf.shape(signal.samples)
        dtype = left_padding.dtype
        sample_count = tf.cast(shape[-1], dtype)

        sample_count = tf.cond(unpad,
            # Figure out sample count of the signal before it was padded
            lambda: sample_count // (tf.cast(1, dtype) + left_padding + right_padding),
            lambda: sample_count
        )

        left_sample_count = tf.cast(sample_count * left_padding, tf.int32)
        right_sample_count = tf.cast(sample_count * right_padding, tf.int32)

        padded_samples = signal.samples

        def unpad_case() -> tf.Tensor:
            return tf.cond(
                right_sample_count == 0,
                lambda: padded_samples[:, left_sample_count:],
                lambda: padded_samples[:, left_sample_count:-right_sample_count]
            )

        def pad_case() -> tf.Tensor:
            left_samples = tf.zeros(
                tf.concat([shape[:-1], [left_sample_count]], 0),
                dtype=signal.samples.dtype
            )

            right_samples = tf.zeros(
                tf.concat([shape[:-1], [right_sample_count]], 0),
                dtype=signal.samples.dtype
            )

            return tf.concat([left_samples, padded_samples, right_samples], -1)

        padded_samples = tf.cond(unpad, unpad_case, pad_case)
        padded_timing = Timing(
            sample_count=tf.shape(padded_samples)[-1],
            sampling_rate=signal.timing.sampling_rate
        )

        return Signal(samples=padded_samples, timing=padded_timing)
