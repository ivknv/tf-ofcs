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

import tensorflow as tf

from typing import Optional
from ..datatypes import ElectricalSignal, TensorCompatible, Timing, OpticalSignal

from .component import Component
from ..utils import timing_to_samples, to_tensor
from .. import constants

__all__ = ["Laser"]


class Laser(Component):
    """Outputs a monochromatic optical signal."""

    #: Output power in watts
    power: tf.Variable  # shape=[None]; dtype=float32

    #: Laser frequency offset (relative to the central frequency) in Hz
    frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    #: The output signal's shape is `(batch_size, tf.size(time))`
    batch_size: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        power: TensorCompatible,
        frequency: TensorCompatible = tf.constant([0.0]),
        batch_size: TensorCompatible = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(to_tensor(power), name=f"{self.name}__power")
        self.frequency = tf.Variable(to_tensor(frequency), name=f"{self.name}__frequency")
        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")

        self.global_parameters = ["batch_size"]
        self.local_parameters = ["power", "frequency", "batch_size"]

    def implementation(
        self,
        timing_or_signal: Timing | ElectricalSignal | OpticalSignal,
        /,
        power: tf.Tensor,
        frequency: tf.Tensor,
        batch_size: tf.Tensor
    ) -> OpticalSignal:
        if isinstance(timing_or_signal, (ElectricalSignal, OpticalSignal)):
            timing = timing_or_signal.timing
        else:
            timing = timing_or_signal

        time = timing_to_samples(timing)

        power = tf.expand_dims(power, -1)  # shape=[None, 1]
        frequency = tf.expand_dims(frequency, -1)  # shape=[None, 1]

        samples = tf.math.exp(
            tf.complex(
                tf.math.log(tf.math.sqrt(power)),
                2.0 * constants.pi * frequency * time
            )
        )

        samples = tf.cond(
            tf.shape(samples)[0] == 1,
            lambda: tf.tile(samples, [batch_size, 1]),
            lambda: samples
        )

        return OpticalSignal(timing=timing, samples=samples)
