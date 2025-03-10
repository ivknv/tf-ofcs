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
from ..datatypes import Signal, TensorCompatible
from ..utils import to_tensor
from .component import Component

__all__ = ["Photodiode"]


class Photodiode(Component):
    """
        A component that converts an optical signal into an electrical signal.

        The conversion is done by squaring the absolute value of the optical signal
        and then clipping it between a minimum and maximum power.

        :ivar sensitivity: `shape=[None]; dtype=float32`, minimum power that
                           the photodiode can detect in W.
        :ivar maximum_power: `shape=[None]; dtype=float32`, maximum optical power
                            that the photodiode can handle in W.
    """

    #: Minimum power (in watts) that can be converted into a current
    sensitivity: tf.Variable  # shape=[None]; dtype=float32

    #: Maximum power the diode can convert (anything above will be clipped)
    maximum_power: tf.Variable  # shape=[None]; dtype=float32

    def __init__(
        self,
        sensitivity: TensorCompatible = tf.constant([1e-5]),
        maximum_power: TensorCompatible = tf.constant([2e-3]),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sensitivity = tf.Variable(to_tensor(sensitivity), name=f"{self.name}__sensitivity")
        self.maximum_power = tf.Variable(to_tensor(maximum_power), name=f"{self.name}__maximum_power")

        self.local_parameters = ["sensitivity", "maximum_power"]

    def implementation(
        self,
        signal: Signal,
        /,
        sensitivity: tf.Tensor,
        maximum_power: tf.Tensor
    ) -> Signal:
        assert signal.samples.dtype is tf.complex64

        sensitivity = tf.expand_dims(sensitivity, -1)
        maximum_power = tf.expand_dims(maximum_power, -1)

        samples = tf.cast(
            tf.clip_by_value(
                tf.abs(signal.samples)**2,
                sensitivity,
                maximum_power
            ),
            tf.float32
        )

        return Signal(timing=signal.timing, samples=samples)
