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
from ...datatypes import ElectricalSignal, OpticalSignal, TensorCompatible, Timing
from ...utils import to_tensor

__all__ = ["Downsample"]


class Downsample(Component):
    #: Downsampling factor
    factor: tf.Variable  # shape=[]; dtype=int32

    #: Shift signal in time before taking samples
    time_offset: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        factor: TensorCompatible,
        time_offset: TensorCompatible = 0.0,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factor = tf.Variable(to_tensor(factor), name=f"{self.name}__factor")
        self.time_offset = tf.Variable(to_tensor(time_offset), name=f"{self.name}__time_offset")
        self.local_parameters = ["factor", "time_offset"]

    def implementation(
        self,
        signal: ElectricalSignal | OpticalSignal,
        /,
        factor: tf.Tensor,
        time_offset: tf.Tensor
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signal.__class__

        shift = tf.cast(tf.round(signal.timing.sampling_rate * time_offset), tf.int32)

        signal_samples = signal.samples[:, shift::factor]

        return Signal(
            timing=Timing(
                sample_count=tf.shape(signal_samples)[-1],
                sampling_rate=signal.timing.sampling_rate / tf.cast(factor, tf.float32)
            ),
            samples=signal_samples
        )
