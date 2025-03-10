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

__all__ = ["SliceSignal"]


class SliceSignal(Component):
    """
        Slices the signal along the last dimension.

        :ivar start: `shape=[]; dtype=tf.int32`, start index of the slice
        :ivar end: `shape=[]; dtype=tf.int32`, end index of the slice
    """

    #: Start index
    start: tf.Variable  # shape=[]; dtype=tf.int32

    #: End index
    end: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        start: TensorCompatible = 0,
        end: TensorCompatible = 0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.start = tf.Variable(tf.constant(start), name=f"{self.name}__start")
        self.end = tf.Variable(tf.constant(end), name=f"{self.name}__end")

        self.local_parameters = ["start", "end"]

    def implementation(
        self,
        signal: Signal,
        /,
        start: tf.Tensor,
        end: tf.Tensor,
        **kwargs
    ) -> Any:
        end = tf.cond(end < 0, lambda: tf.shape(signal.samples)[-1] + end, lambda: end)

        return Signal(
            timing=Timing(
                sample_count=end - start,
                sampling_rate=signal.timing.sampling_rate
            ),
            samples=signal.samples[:, start:end]
        )
