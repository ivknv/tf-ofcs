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

from .laser import Laser
from .photodiode import Photodiode
from .mzm import MachZehnderModulator
from .pam import PAMSymbolEncoder, PAMSymbolDecoder, PAMEncoder, PAMDecoder
from .iq import (
    IQSymbolEncoder, IQSymbolDecoder, IQEncoder, IQModulator, IQReceiver, IQDecoder
)
from .noise import ElectricalAWGN, OpticalAWGN
from .passive import (
    OpticalFiber, OpticalPhaseShifter, OpticalCoupler1x2,
    OpticalCoupler2x2, OpticalCoupler2x1, OpticalMultiplexer, OpticalDemultiplexer
)
from .filters import (
    BandPassFilter, FFTFilter, BrickwallFrequencyResponse,
    WDMBrickwallFrequencyResponse, GaussianFrequencyResponse,
    BesselFrequencyResponse
)
from .ber import BERTester
from .utils import (
    Composition, MapComponents, Identity, Reindex, Fork, Downsample, Upsample,
    ExtractSamples, Lambda, SplitSignals, MergeSignals, TileSignals,
    RandomizeParameters, PadSignal, Loop
)
from .electrical import (
    ElectricalAdder, ElectricalSubtractor, ArbitraryWaveformGenerator,
    Quantizer, ElectricalGain
)
from .component import Component
from .optical_gain import OpticalGain
from .channel_loss_estimator import ChannelLossEstimator
from .gaussian_pulse_generator import GaussianPulseGenerator
from .optical_amplifier import OpticalAmplifier

__all__ = [
    "Laser",
    "Photodiode",
    "PAMSymbolEncoder",
    "PAMSymbolDecoder",
    "PAMEncoder",
    "PAMDecoder",
    "IQSymbolEncoder",
    "IQSymbolDecoder",
    "IQEncoder",
    "IQModulator",
    "IQReceiver",
    "IQDecoder",
    "MachZehnderModulator",
    "ElectricalAWGN",
    "OpticalAWGN",
    "OpticalFiber",
    "OpticalPhaseShifter",
    "OpticalCoupler1x2",
    "OpticalCoupler2x2",
    "OpticalCoupler2x1",
    "BandPassFilter",
    "FFTFilter",
    "BERTester",
    "BrickwallFrequencyResponse",
    "WDMBrickwallFrequencyResponse",
    "GaussianFrequencyResponse",
    "BesselFrequencyResponse",
    "Composition",
    "MapComponents",
    "Identity",
    "Reindex",
    "Fork",
    "ElectricalAdder",
    "ElectricalSubtractor",
    "ArbitraryWaveformGenerator",
    "Quantizer",
    "ElectricalGain",
    "Downsample",
    "Upsample",
    "ExtractSamples",
    "Lambda",
    "SplitSignals",
    "MergeSignals",
    "TileSignals",
    "RandomizeParameters",
    "PadSignal",
    "Loop",
    "Component",
    "OpticalGain",
    "ChannelLossEstimator",
    "GaussianPulseGenerator",
    "OpticalMultiplexer",
    "OpticalDemultiplexer",
    "OpticalAmplifier"
]
