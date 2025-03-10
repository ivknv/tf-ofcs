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

from ...datatypes import Signal
from ..component import Component

__all__ = ["MergeSignals"]


class MergeSignals(Component):
    """
        Merges a tuple of `n` signals with shape (batch_size, sample_count)
        into a signal with shape (batch_size * n, sample_count).

        Of course, the signals don't have to have the same `batch_size`, but
        they must be of the same type (e.g all optical or all electrical) and
        have the same `timing`.
    """

    def call(
        self,
        signals: tuple[Signal, ...],
        /,
        **kwargs
    ) -> Signal:
        timing = signals[0].timing

        merged = tf.concat([signal.samples for signal in signals], 0)

        return Signal(samples=merged, timing=timing)
