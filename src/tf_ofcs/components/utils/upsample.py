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
from ...datatypes import Signal, Timing

__all__ = ["Upsample"]


class Upsample(Component):
    """
        Upsamples the input signal by a given factor.

        The upsampling is performed by repeating each sample of the input signal
        by the given factor. The sampling rate of the output signal is also
        multiplied by the same factor.

        :ivar factor: `int` or `tf.Tensor`, upsampling factor
    """

    factor: int | tf.Tensor

    def __init__(
        self,
        factor: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.factor = factor

    def implementation(self, signal: Signal, /) -> Signal:
        signal_samples = tf.repeat(signal.samples, self.factor, axis=1)

        return Signal(
            timing=Timing(
                sample_count=tf.shape(signal_samples)[-1],
                sampling_rate=signal.timing.sampling_rate * tf.cast(self.factor, tf.float32)
            ),
            samples=signal_samples
        )
