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

from .laser import Laser
from .photodiode import Photodiode
from .mzm import MachZehnderModulator
from .pam import PAMSymbolEncoder, PAMSymbolDecoder
from .iq import (
    IQSymbolEncoder, IQSymbolDecoder, IQModulator, IQReceiver,
    IQToComplex, ComplexToIQ
)
from .noise import ElectricalAWGN, OpticalAWGN
from .passive import (
    OpticalFiber, OpticalPhaseShifter, OpticalCoupler1x2,
    OpticalCoupler2x2, OpticalCoupler2x1, OpticalMultiplexer, OpticalDemultiplexer
)
from .filters import (
    BandPassFilter, FFTFilter, FIRFilter, RaisedCosineFrequencyResponse,
    RootRaisedCosineFrequencyResponse, BrickwallFrequencyResponse,
    WDMBrickwallFrequencyResponse, GaussianFrequencyResponse, BesselFrequencyResponse
)
from .ber import BERTester
from .utils import (
    Composition, MapComponents, Identity, Reindex, Fork, Downsample, Upsample,
    ExtractSamples, Lambda, SplitSignals, MergeSignals, TileSignals,
    RandomizeParameters, PadSignal, UnpadSignal, Loop, Substitute, TruncateSignal,
    SelectSamples
)
from .electrical_and_digital import (
    Adder, Subtractor, SignalFromTensor,
    Gain, AnalogToDigitalConverter, DigitalToAnalogConverter,
    MZMPreDistortion, PhotodiodePreDistortion
)
from .component import Component
from .channel_loss_estimator import ChannelLossEstimator
from .gaussian_pulse_generator import GaussianPulseGenerator
from .optical_amplifier import OpticalAmplifier
from .bit_sequence_generator import BitSequenceGenerator
from .carrier_phase_recovery import DecisionDirectedPhaseRecovery, ViterbiViterbiPhaseRecovery
from .confusion_matrix_analyzer import ConfusionMatrixAnalyzer

__all__ = [
    "Laser",
    "Photodiode",
    "PAMSymbolEncoder",
    "PAMSymbolDecoder",
    "IQSymbolEncoder",
    "IQSymbolDecoder",
    "IQModulator",
    "IQReceiver",
    "IQToComplex",
    "ComplexToIQ",
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
    "FIRFilter",
    "BERTester",
    "RaisedCosineFrequencyResponse",
    "RootRaisedCosineFrequencyResponse",
    "BrickwallFrequencyResponse",
    "WDMBrickwallFrequencyResponse",
    "GaussianFrequencyResponse",
    "BesselFrequencyResponse",
    "Composition",
    "MapComponents",
    "Identity",
    "Reindex",
    "Fork",
    "Adder",
    "Subtractor",
    "SignalFromTensor",
    "Gain",
    "AnalogToDigitalConverter",
    "DigitalToAnalogConverter",
    "MZMPreDistortion",
    "PhotodiodePreDistortion",
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
    "SelectSamples",
    "Component",
    "ChannelLossEstimator",
    "GaussianPulseGenerator",
    "OpticalMultiplexer",
    "OpticalDemultiplexer",
    "OpticalAmplifier",
    "BitSequenceGenerator",
    "DecisionDirectedPhaseRecovery",
    "ViterbiViterbiPhaseRecovery",
    "ConfusionMatrixAnalyzer"
]
