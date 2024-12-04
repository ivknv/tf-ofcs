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

from ...datatypes import ElectricalSignal, OpticalSignal
from ..component import Component

__all__ = ["SliceSignal"]


class SliceSignal(Component):
    """
        Slices the input signal, returning
        `samples[a * batch_size // factor:b * batch_size // factor]`,
        where `a, b = slice[0], slice[1]`.
    """

    slice: tf.Variable  # shape=[2]; dtype=tf.int32
    factor: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        slice: tuple[int, int] | tf.Tensor,
        factor: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.slice = tf.Variable(tf.constant(slice), name=f"{self.name}__slice")
        self.factor = tf.Variable(tf.constant(factor), name=f"{self.name}__factor")
        self.local_parameters = ["slice", "factor"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        slice: tf.Tensor,
        factor: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__
        slice = slice * tf.shape(signal.samples)[0] // factor

        return Signal(samples=signal.samples[slice[0]:slice[1]], timing=signal.timing)
