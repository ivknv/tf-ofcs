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
from ...datatypes import Signal, Timing

__all__ = ["SelectSamples"]


class SelectSamples(Component):
    """
        Slices the signal to select a block of samples (made up of `size` samples)
        at `index * size`.

        :ivar index: `shape=[]; dtype=tf.int32`, index of the sample block to select
        :ivar size: shape=[]; dtype=tf.int32`, size of the block (number of samples)
        :ivar count: shape=[]; dtype=tf.int32`, number of blocks to select
    """

    index: tf.Variable  # shape=[]; dtype=tf.int32
    size: tf.Variable  # shape=[]; dtype=tf.int32
    count: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        index: int | tf.Tensor = 0,
        size: int | tf.Tensor = 0,
        count: int | tf.Tensor = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.index = tf.Variable(tf.constant(index), name=f"{self.name}__index")
        self.size = tf.Variable(tf.constant(size), name=f"{self.name}__size")
        self.count = tf.Variable(tf.constant(count), name=f"{self.name}__count")

        self.local_parameters = ["index", "size", "count"]

    def implementation(
        self,
        signal: Signal,
        /,
        index: tf.Tensor,
        size: tf.Tensor,
        count: tf.Tensor
    ) -> Any:
        Signal = signal.__class__

        return Signal(
            timing=Timing(
                sample_count=size,
                sampling_rate=signal.timing.sampling_rate
            ),
            samples=signal.samples[:, index * size:(index + count) * size]
        )
