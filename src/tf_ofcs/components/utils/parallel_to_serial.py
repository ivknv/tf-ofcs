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
from ...datatypes import ElectricalSignal, OpticalSignal, Timing

__all__ = ["ParallelToSerial"]


class ParallelToSerial(Component):
    """
        Merges `n` parallel signals into one or more serial signals.
        A single parallel signal is considered to span `batch_size` // `n` batches
        of the signal.

        If the input signal has the shape `(batch_size, parameter_dims, sample_count)`,
        then the output shape is `(batch_size // n, parameter_dims, sample_count * n)`
    """

    #: Number of parallel signals to merge into one
    n: int

    def __init__(self, n: int, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = n

    def call(
        self,
        input_signal: OpticalSignal | ElectricalSignal,
        /,
        **kwargs
    ) -> ElectricalSignal | OpticalSignal:
        timing = input_signal.timing
        sampling_rate = timing.sampling_rate

        SignalClass = input_signal.__class__

        new_sample_count = timing.sample_count * self.n

        batch_size = tf.shape(input_signal.samples)[0]

        tf.debugging.assert_equal(tf.math.floormod(batch_size, self.n), 0)

        new_signal_samples = tf.vectorized_map(
            lambda i: tf.concat(
                [input_signal.samples[i * self.n + j] for j in range(self.n)],
                1
            ),
            tf.range(batch_size // self.n))

        return SignalClass(
            timing=Timing(
                sample_count=new_sample_count,
                sampling_rate=sampling_rate
            ),
            samples=new_signal_samples
        )
