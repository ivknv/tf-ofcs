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
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal

__all__ = ["RedistributeDimensions"]


class RedistributeDimensions(Component):
    """
        Reshapes the signal to [
            batch_size * factors[0] // factors[1],
            parameter_dims * factors[1] // factors[0],
            sample_count
        ]
    """

    factors: tf.Variable  # shape=[2]; dtype=int32

    def __init__(
        self,
        factors: tuple[int, int] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factors = tf.Variable(tf.constant(factors), name=f"{self.name}__factors")
        self.local_parameters = ["factors"]

    def implementation(
        self,
        signal: OpticalSignal | ElectricalSignal,
        /,
        factors: tf.Tensor
    ) -> OpticalSignal | ElectricalSignal:
        Signal = signal.__class__

        batch_size = tf.shape(signal.samples)[0]
        parameter_dims = tf.shape(signal.samples)[1]
        sample_count = tf.shape(signal.samples)[2]

        batch_size = batch_size * factors[0] // factors[1]
        parameter_dims = parameter_dims * factors[1] // factors[0]

        return Signal(
            timing=signal.timing,
            samples=tf.reshape(signal.samples, (batch_size, parameter_dims, sample_count))
        )
