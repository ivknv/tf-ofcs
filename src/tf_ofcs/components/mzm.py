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

from .. import constants
from .component import Component
from ..utils import to_tensor

from ..datatypes import Signal, TensorCompatible

__all__ = ["MachZehnderModulator"]


class MachZehnderModulator(Component):
    """
        Mach-Zehnder modulator (MZM) implementation.

        Expects a `tuple[Signal, Signal]` as input.
        The first tuple element is the carrier signal (optical signal),
        the second is the modulation signal (electrical signal).

        The modulation signal's samples must be in range `[0.0, 1.0]`.
        Value of `0` corresponds to a phase shift of `0`, while a value of `1`
        corresponds to a phase shift of `math.pi`.

        :ivar extinction_ratio: `tf.Variable` (`dtype=tf.float32`), modulator's
                                extinction ratio
    """

    extinction_ratio: tf.Variable  # shape=[None]; dtype=float32

    def __init__(
        self,
        *,
        extinction_ratio: TensorCompatible = tf.constant([float("inf")]),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.extinction_ratio = tf.Variable(
            to_tensor(extinction_ratio),
            name=f"{self.name}__extinction_ratio"
        )
        self.local_parameters = ["extinction_ratio"]

    def implementation(
        self,
        inputs: tuple[Signal, Signal],
        /,
        extinction_ratio: tf.Tensor
    ) -> Signal:
        carrier_signal, modulation_signal = inputs

        assert carrier_signal.samples.dtype is tf.complex64
        assert modulation_signal.samples.dtype is tf.float32

        k = tf.expand_dims(1.0 / extinction_ratio, -1)

        # Modulation signal is expected to be in range [0.0, 1.0]
        modulated_signal = tf.multiply(
            carrier_signal.samples,
            tf.complex(
                tf.math.cos(constants.pi * modulation_signal.samples),
                k * tf.math.sin(constants.pi * modulation_signal.samples)
            ),
        )

        return Signal(timing=carrier_signal.timing, samples=modulated_signal)
