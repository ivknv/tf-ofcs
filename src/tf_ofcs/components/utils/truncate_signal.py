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

from typing import Any, Optional
import tensorflow as tf

from ..component import Component
from ...datatypes import TensorCompatible, Signal, Timing

__all__ = ["TruncateSignal"]


class TruncateSignal(Component):
    """
        Truncates the signal along the last dimension from both sides.

        :ivar left: `shape=[]; dtype=tf.int32`, number of samples to truncate from the start
        :ivar right: `shape=[]; dtype=tf.int32`, number of samples to truncate from the end
    """

    #: Start index
    left: tf.Variable  # shape=[]; dtype=tf.int32

    #: End index
    right: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        left: TensorCompatible = 0,
        right: TensorCompatible = 0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.left = tf.Variable(tf.constant(left), name=f"{self.name}__left")
        self.right = tf.Variable(tf.constant(right), name=f"{self.name}__right")

        self.local_parameters = ["left", "right"]

    def implementation(
        self,
        signal: Signal,
        /,
        left: tf.Tensor,
        right: tf.Tensor,
        **kwargs
    ) -> Any:
        sample_count = tf.shape(signal.samples)[-1]
        truncated = signal.samples[:, left:tf.math.maximum(sample_count - right, 0)]

        return Signal(
            timing=Timing(
                sample_count=tf.shape(truncated)[-1],
                sampling_rate=signal.timing.sampling_rate
            ),
            samples=truncated
        )
