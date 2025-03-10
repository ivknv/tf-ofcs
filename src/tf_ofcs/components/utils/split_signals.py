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

from ...datatypes import Signal
from ..component import Component

__all__ = ["SplitSignals"]


class SplitSignals(Component):
    """
        Splits signal with shape (batch_size, sample_count)
        into `n` tuples of shape (batch_size // n, sample_count).

        :ivar n: `int`, split factor
    """

    n: int

    def __init__(self, n: int = 0, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.n = n

    def call(self, signal: Signal, /, **kwargs) -> tuple[Signal, ...]:
        batch_size = tf.shape(signal.samples)[0] // self.n

        return tuple(
            Signal(
                samples=signal.samples[i * batch_size:(i + 1) * batch_size],
                timing=signal.timing
            )
            for i in range(self.n)
        )
