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

        The conversion is done by squaring the absolute value of the optical
        signal, adding the dark current and then limiting it to be less than or
        equal to the maximum power.

        :ivar dark_current: `shape=[None]; dtype=float32`, minimum power that
                           the photodiode can detect in W.
        :ivar maximum_power: `shape=[None]; dtype=float32`, maximum optical power
                            that the photodiode can handle in W.
    """

    #: Photodiode's dark current (minimum power, effectively)
    dark_current: tf.Variable  # shape=[None]; dtype=float32

    #: Maximum power the diode can convert (anything above will be clipped)
    maximum_power: tf.Variable  # shape=[None]; dtype=float32

    def __init__(
        self,
        dark_current: TensorCompatible = tf.constant([1e-5]),
        maximum_power: TensorCompatible = tf.constant([2e-3]),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.dark_current = tf.Variable(to_tensor(dark_current), name=f"{self.name}__dark_current")
        self.maximum_power = tf.Variable(to_tensor(maximum_power), name=f"{self.name}__maximum_power")

        self.local_parameters = ["dark_current", "maximum_power"]

    def implementation(
        self,
        signal: Signal,
        /,
        dark_current: tf.Tensor,
        maximum_power: tf.Tensor
    ) -> Signal:
        assert signal.samples.dtype is tf.complex64

        dark_current = tf.expand_dims(dark_current, -1)
        maximum_power = tf.expand_dims(maximum_power, -1)

        samples = tf.math.minimum(
            tf.square(tf.abs(signal.samples)) + dark_current,
            maximum_power
        )

        return Signal(timing=signal.timing, samples=samples)
