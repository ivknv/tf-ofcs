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
from ...datatypes import DigitalSignal, ElectricalSignal, OpticalSignal, TensorCompatible, Timing
from ...utils import to_tensor

__all__ = ["Upsample"]


class Upsample(Component):
    #: Upsampling factor
    factor: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        factor: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factor = tf.Variable(to_tensor(factor), name=f"{self.name}__factor")
        self.local_parameters = ["factor"]

    def implementation(
        self,
        signal: ElectricalSignal | DigitalSignal | OpticalSignal,
        /,
        factor: tf.Tensor,
    ) -> ElectricalSignal | DigitalSignal | OpticalSignal:
        Signal = signal.__class__

        signal_samples = tf.repeat(signal.samples, factor, axis=1)

        return Signal(
            timing=Timing(
                sample_count=tf.shape(signal_samples)[-1],
                sampling_rate=signal.timing.sampling_rate * tf.cast(factor, tf.float32)
            ),
            samples=signal_samples
        )
