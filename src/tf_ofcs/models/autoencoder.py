# -*- coding: utf-8 -*-
# Copyright (C) 2025 Ivan Konovalov

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

import keras  # type: ignore[import-untyped]

from typing import Protocol
from ..components import Component

__all__ = ["Autoencoder"]


class Autoencoder(Protocol):
    sampling_rate: float
    M: int
    n: int
    N: int

    encoder: Component
    decoder: Component

    encoder_network: Component
    decoder_network: Component
    transmitter: Component
    channel: Component
    receiver: Component


class AutoencoderModel(keras.Model, Autoencoder):
    ...
