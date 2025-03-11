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

from typing import Literal, Optional
import tensorflow as tf

from ..component import Component
from ...datatypes import TensorCompatible, Signal, Timing

__all__ = ["TruncateSignal"]


class TruncateSignal(Component):
    """
        Truncates the signal along the last dimension from both sides.

        :ivar left: `shape=[]; dtype=tf.int32`, number of samples or time to truncate from the start
        :ivar right: `shape=[]; dtype=tf.int32`, number of samples or time to truncate from the end
    """

    left: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32
    right: tf.Variable  # shape=[]; dtype=tf.int32 or tf.float32
    mode: Literal["samples"] | Literal["time"]

    def __init__(
        self,
        left: Optional[TensorCompatible] = None,
        right: Optional[TensorCompatible] = None,
        mode: Literal["samples"] | Literal["time"] = "samples",
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if left is None:
            left = 0 if mode == "samples" else 0.0

        if right is None:
            right = 0 if mode == "samples" else 0.0

        self.left = tf.Variable(tf.constant(left), name=f"{self.name}__left")
        self.right = tf.Variable(tf.constant(right), name=f"{self.name}__right")
        self.mode = mode

        self.local_parameters = ["left", "right"]

    def implementation(
        self,
        signal: Signal,
        /,
        left: tf.Tensor,
        right: tf.Tensor
    ) -> Signal:
        sample_count = tf.shape(signal.samples)[-1]

        if self.mode == "time":
            left = tf.cast(tf.round(left * signal.timing.sampling_rate), dtype=tf.int32)
            right = tf.cast(tf.round(right * signal.timing.sampling_rate), dtype=tf.int32)

        start = left
        end = tf.math.maximum(sample_count - right, 0)

        truncated = signal.samples[:, start:end]

        return Signal(
            timing=Timing(
                sample_count=tf.shape(truncated)[-1],
                sampling_rate=signal.timing.sampling_rate
            ),
            samples=truncated
        )
