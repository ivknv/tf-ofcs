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

from ..datatypes import OpticalSignal, Timing
from .component import Component
from ..utils import timing_to_samples

__all__ = ["GaussianPulseGenerator"]


class GaussianPulseGenerator(Component):
    """Creates a single Gaussian pulse."""

    #: FWHM (full width at half maximum) pulse duration
    fwhm_duration: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Power (amplitude squared) of the pulse in watts
    power: tf.Variable  # shape=[] or [None]; dtype=float32

    #: The output signal has shape (batch_size, parameter_dims, sample_count)
    batch_size: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        fwhm_duration: float | Sequence[float] | tf.Tensor,
        power: float | Sequence[float] | tf.Tensor,
        batch_size: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.fwhm_duration = tf.Variable(tf.constant(fwhm_duration), name=f"{self.name}__fwhm_duration")
        self.power = tf.Variable(tf.constant(power), name=f"{self.name}__power")
        self.batch_size = tf.Variable(tf.constant(batch_size), name=f"{self.name}__batch_size")
        self.local_parameters = [("power", 0), ("fwhm_duration", 0), "batch_size"]
        self.global_parameters = ["batch_size"]

    def implementation(
        self,
        timing: Timing,
        /,
        fwhm_duration: tf.Tensor,
        power: tf.Tensor,
        batch_size: tf.Tensor
    ) -> OpticalSignal:
        fwhm_duration = tf.expand_dims(fwhm_duration, 1)
        power = tf.expand_dims(power, 1)

        time = timing_to_samples(timing)

        return OpticalSignal(
            timing=timing,
            samples=tf.tile(
                tf.expand_dims(
                    tf.cast(
                        tf.math.sqrt(power) * tf.exp(-0.5 * ((time - time[-1] * 0.5) / fwhm_duration) ** 2),
                        tf.complex64
                    ),
                    0
                ),
                [batch_size, 1, 1]
            )
        )
