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

from .composition import Composition
from .identity import Identity
from .reindex import Reindex
from .map_components import MapComponents
from .fork import Fork
from .downsample import Downsample
from .upsample import Upsample
from .extract_samples import ExtractSamples
from .lambda_ import Lambda
from .split_signals import SplitSignals
from .merge_signals import MergeSignals
from .tile_signals import TileSignals
from .randomize_parameters import RandomizeParameters
from .pad_signal import PadSignal, UnpadSignal
from .loop import Loop
from .substitute import Substitute
from .truncate_signal import TruncateSignal
from .select_samples import SelectSamples

__all__ = [
    "Composition",
    "MapComponents",
    "Identity",
    "Reindex",
    "Fork",
    "Downsample",
    "Upsample",
    "ExtractSamples",
    "Lambda",
    "SplitSignals",
    "MergeSignals",
    "TileSignals",
    "RandomizeParameters",
    "PadSignal",
    "UnpadSignal",
    "Loop",
    "Substitute",
    "TruncateSignal",
    "SelectSamples"
]
