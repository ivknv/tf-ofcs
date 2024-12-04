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
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal

__all__ = ["ExtractSamples"]


class ExtractSamples(Component):
    def call(self, signal: ElectricalSignal | OpticalSignal, /, **kwargs) -> tf.Tensor:
        assert isinstance(signal, (ElectricalSignal, OpticalSignal))
        return signal.samples
