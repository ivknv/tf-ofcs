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

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from .component import Component
from ..datatypes import OpticalSignal, optical_signal_spec

__all__ = ["OpticalGain"]

class OpticalGain(Component):
    #: Gain coefficient (unitless)
    gain: tf.Variable  # shape=[] or [None]; dtype=float32

    def __init__(self, gain: float | Sequence[float] | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.gain = tf.Variable(tf.constant(gain), name=f"{self.name}__gain")
        self.local_parameters = [("gain", 0)]

    @tf.function(input_signature=(
        optical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ))
    def implementation(self, input_signal: OpticalSignal, gain: tf.Tensor) -> OpticalSignal:
        gain = tf.expand_dims(gain, -1)

        return OpticalSignal(
            timing=input_signal.timing,
            samples=input_signal.samples * tf.cast(gain ** 0.5, tf.complex64)
        )
