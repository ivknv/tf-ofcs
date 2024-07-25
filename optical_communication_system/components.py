# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Any, Optional
import numpy.typing as npt
import numpy as np
import scipy.signal
import scipy as sp

import math

from dataclasses import dataclass
from .types import *

__all__ = [
    "Signal",

    "NRZEncoder",
    "PAMEncoder",
    "System",
    "Transmitter",
    "IntensityModulator",
    "Identity",
    "Fork",
    "SignalCombination",
    "AWGN",
    "OpticalChannel",
    "ChromaticDispersion",
    "OpticalLoss",
    "ElectricalLowPassFilter"
]


class Constants:
    #: Speed of light in vacuum in m/s
    c: float = 2.99792458E8


@dataclass(frozen=True)
class Timing:
    #: Time axis of the signal (in seconds)
    time: npt.NDArray[np.float32]

    #: Sampling rate in Hz
    sampling_rate: float

    #: Data rate in bits/second
    data_rate: float


@dataclass(frozen=True)
class Signal:
    #: Type of the signal
    type: "SignalType"

    #: Signal's timing information
    timing: Timing

    #: Signal's frequencies (in Hz), for every frequency there's an amplitude
    frequencies: npt.NDArray[np.float32]

    #: Signal's samples
    samples: npt.NDArray

    #: Power (maximum) of the optical signal in watts
    power: float = 0.0

    #: Central frequency of the of the emitted light in Hz
    central_frequency: float = 0.0

    @staticmethod
    def electrical(*, timing: Timing, samples: npt.NDArray) -> "Signal":
        return Signal(
            type="electrical",
            timing=timing,
            frequencies=np.array([]),
            samples=samples
        )

    @staticmethod
    def optical(
        *,
        timing: Timing,
        frequencies: npt.NDArray[np.float32],
        samples: npt.NDArray,
        central_frequency: float,
        power: float
    ) -> "Signal":
        return Signal(
            type="optical",
            timing=timing,
            frequencies=frequencies,
            samples=samples,
            central_frequency=central_frequency,
            power=power
        )


@dataclass(frozen=True)
class NRZEncoder:
    #: Sampling rate in Hz
    sampling_rate: float

    #: Data rate in bits/s
    data_rate: float

    def __call__(self, bitseq: "BitSequence") -> Signal:
        signal_duration = len(bitseq) / self.data_rate
        time = np.arange(0, signal_duration, 1 / self.sampling_rate, dtype=np.float32)

        samples = np.zeros(shape=time.shape, dtype=np.float32)

        # Oversample input signal if needed
        samples = np.repeat(
            bitseq.astype(np.float32),
            math.ceil(len(time) / len(bitseq))
        )[:len(time)]

        return Signal.electrical(
            timing=Timing(
                time=time,
                sampling_rate=self.sampling_rate,
                data_rate=self.data_rate
            ),
            samples=samples
        )


@dataclass(frozen=True)
class PAMEncoder:
    #: Sampling rate in Hz
    sampling_rate: float

    #: Data rate in symbols/second
    symbol_rate: float

    #: Modulation order (number of amplitude/power levels)
    order: int = 1

    #: Constellation used for encoding
    constellation: Optional[dict[tuple[int, ...], float]] = None

    def _map_to_pam(self, bitseq: "BitSequence") -> float:
        symbol = 0

        for i, bit in enumerate(bitseq):
            symbol += bit * self.order ** (len(bitseq) - 1 - i)

        return symbol

    def __call__(self, bitseq: "BitSequence") -> Signal:
        if len(bitseq) % self.order != 0:
            raise ValueError(f"bit sequence length must be a multiple of {self.order} bits")

        signal_duration = len(bitseq) / self.symbol_rate
        time = np.arange(0, signal_duration, 1 / self.sampling_rate, dtype=np.float32)

        pam_sequence = np.zeros(shape=(len(bitseq) // self.order,), dtype=np.float32)

        for i in range(len(pam_sequence)):
            input_symbol = bitseq[i * self.order:(i + 1) * self.order]

            if self.constellation is None:
                pam_sequence[i] = self._map_to_pam(input_symbol)
            else:
                pam_sequence[i] = self.constellation[tuple(input_symbol)]

        pam_sequence /= self.order

        return Signal.electrical(
            timing=Timing(
                time=time,
                sampling_rate=self.sampling_rate,
                data_rate=self.symbol_rate * self.order
            ),
            samples=np.repeat(
                pam_sequence,
                math.ceil(len(time) / len(pam_sequence))
            )[:len(time)]
        )


@dataclass(frozen=True)
class System:
    #: System components
    components: list["Callable"]

    def __call__(self, inp: Any = None) -> Any:
        result: Any = inp

        for component in self.components:
            if result is not None:
                result = component(result)
            else:
                result = component()

        return result


@dataclass(frozen=True)
class Transmitter:
    #: Output power in watts
    power: float = 2e-3

    #: Signal-to-noise ratio in dB
    snr: float = 1000

    #: Central frequency of the of the emitted light in Hz
    central_frequency: float = 193.414e12

    #: Line width of the signal, full width at half-maximum (FWHM), Hz
    linewidth: float = 0.0

    #: Number of frequency components in the signal
    frequency_count: int = 64

    def __call__(self, timing: Timing) -> Signal:
        signal = np.ones(
            shape=(len(timing.time), self.frequency_count),
            dtype=np.float32
        )
        signal *= self.power ** 0.5

        gauss_variance = 1 / (2 * math.log(2)) ** 0.5 * self.linewidth

        frequencies = np.linspace(
            self.central_frequency - self.linewidth * 10,
            self.central_frequency + self.linewidth * 10,
            self.frequency_count
        )

        distribution = gaussian_distribution(
            frequencies,
            self.central_frequency,
            gauss_variance
        )
        distribution /= np.sum(distribution)

        signal[:,] *= distribution**0.5

        return Signal.optical(
            timing=timing,
            frequencies=frequencies,
            samples=signal,
            central_frequency=self.central_frequency,
            power=self.power
        )


@dataclass(frozen=True)
class IntensityModulator:
    #: Carrier signal source
    carrier_source: Callable[[Timing], Signal]

    def __call__(self, modulation_signal: Signal) -> Signal:
        carrier_signal = self.carrier_source(modulation_signal.timing)

        modulated_signal = np.array(carrier_signal.samples)
        modulated_signal *= abs(
            modulation_signal.samples.repeat(
                len(carrier_signal.frequencies)
            ).reshape(modulated_signal.shape)
        ) ** 0.5

        return Signal.optical(
            timing=carrier_signal.timing,
            frequencies=carrier_signal.frequencies,
            samples=modulated_signal,
            central_frequency=carrier_signal.central_frequency,
            power=carrier_signal.power
        )


@dataclass(frozen=True)
class Identity:
    def __call__(self, inp: Any) -> Any:
        return inp


@dataclass(frozen=True)
class Fork:
    #: Components that will process forked output. Every list item will get its
    #: own copy of the forked output
    components: list[Callable]

    def __call__(self, inp: Any) -> tuple[Any, ...]:
        return tuple(component(inp) for component in self.components)


@dataclass(frozen=True)
class SignalCombination:
    def __call__(self, signals: tuple[Signal, ...]) -> Signal:
        first_signal = signals[0]
        combined_signal_samples = np.array(first_signal.samples)

        for signal in signals[1:]:
            combined_signal_samples += signal.samples

        return Signal(
            type=first_signal.type,
            timing=first_signal.timing,
            frequencies=first_signal.frequencies,
            samples=combined_signal_samples,
            power=first_signal.power,
            central_frequency=first_signal.central_frequency
        )


@dataclass(frozen=True)
class AWGN:
    #: Noise level in dB relative to the input signal
    level: float = 0.0

    #: Fixed amount of noise in watts that is independent of input signal
    fixed_power: float = 0.0

    def __call__(self, input_signal: Signal) -> Signal:
        noise_amplitude = self.fixed_power
        noise_amplitude += input_signal.power * 10 ** (self.level * 0.1)

        if len(input_signal.frequencies):
            noise_amplitude /= len(input_signal.frequencies)

        noise_amplitude **= 0.5

        noise = np.random.normal(
            0.0,
            noise_amplitude,
            input_signal.samples.shape
        )

        return Signal(
            type=input_signal.type,
            timing=input_signal.timing,
            frequencies=input_signal.frequencies,
            samples=noise + input_signal.samples,
            central_frequency=input_signal.central_frequency,
            power=input_signal.power
        )


def gaussian_distribution(x: npt.NDArray, mean: float, variance: float) -> npt.NDArray:
    return 1 / (variance * ((2 * math.pi) ** 0.5)) * math.e ** (-1/2 * ((x - mean) / variance) ** 2)


@dataclass(frozen=True)
class ChromaticDispersion:
    #: Dispersion in ps/nm
    amount: float

    def __call__(self, signal: Signal) -> Signal:
        wavelength_deltas = (Constants.c * 1e9 / signal.frequencies) - (Constants.c * 1e9 / signal.central_frequency)

        # Each wavelength (or frequency) will be shifted in time by some number
        # of samples
        timeshifts = (
            self.amount * 1e-12 * signal.timing.sampling_rate * wavelength_deltas
        ).astype(np.int32)

        min_shift = timeshifts.min()
        max_shift = timeshifts.max()

        # We have to pad samples from both sides
        left_pad = max(-min_shift, 0)
        right_pad = max(max_shift, 0)

        new_signal = np.zeros(
            (
                signal.samples.shape[0] + left_pad + right_pad,
                signal.samples.shape[1]
            ),
            dtype=signal.samples.dtype
        )

        for i, shift in enumerate(timeshifts):
            last_index = -right_pad + shift
            if last_index == 0:
                last_index = len(new_signal)

            new_signal[left_pad + shift:last_index,i] = signal.samples[:,i]

        # Cut off the previosuly added padding
        if right_pad != 0:
            samples = new_signal[left_pad:-right_pad]
        else:
            samples = new_signal[left_pad:]

        return Signal.optical(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=samples,
            central_frequency=signal.central_frequency,
            power=signal.power
        )


@dataclass(frozen=True)
class OpticalLoss:
    # Optical signal power loss in dB
    loss: float

    def __call__(self, signal: Signal) -> Signal:
        power_loss_coeff = 10 ** (self.loss * 0.1)
        samples = np.array(signal.samples)
        samples /= power_loss_coeff ** 0.5

        return Signal.optical(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=samples,
            central_frequency=signal.central_frequency,
            power=signal.power / power_loss_coeff
        )

@dataclass(frozen=True)
class OpticalChannel:
    #: Attenuation in the channel in dB/km
    attenuation: float

    #: Chromatic dispersion coefficient in ps/(nm * km)
    chromatic_dispersion: float

    #: Transmission distance in km
    length: float

    def __call__(self, signal: Signal) -> Signal:
        attenuated_signal = OpticalLoss(self.attenuation * self.length)(signal)

        if abs(self.chromatic_dispersion) >= 1e-6:
            return ChromaticDispersion(
                self.chromatic_dispersion * self.length
            )(attenuated_signal)

        return attenuated_signal


@dataclass(frozen=True)
class ElectricalLowPassFilter:
    #: Filter cutoff frequency in Hz
    cutoff_frequency: float

    #: Filter order
    order: int

    def __call__(self, signal: Signal) -> Signal:
        sos = sp.signal.bessel(
            self.order,
            self.cutoff_frequency,
            "low",
            fs=int(signal.timing.sampling_rate),
            output="sos",
            norm="delay"
        )

        filtered_amplitude = sp.signal.sosfilt(sos, signal.samples)
        filtered_samples = np.array(signal.samples)
        filtered_samples = filtered_amplitude

        return Signal.electrical(
            timing=signal.timing,
            samples=filtered_samples
        )
