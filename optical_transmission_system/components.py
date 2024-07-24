# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, ClassVar, Optional
import numpy.typing as npt
import numpy as np
import scipy.signal
import scipy as sp

import math

from dataclasses import dataclass

if TYPE_CHECKING:
    from .types import *

__all__ = [
    "Signal",
    "DigitalSignal",
    "ElectricalSignal",
    "OpticalSignal",

    "NRZEncoder",
    "System",
    "Transmitter",
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
class Signal:
    #: Data type (dtype) for the samples array
    dtype: ClassVar[npt.DTypeLike] = [("time", np.float32)]

    #: Type of the signal
    type: "SignalType"

    time: npt.NDArray[np.float32]

    #: Signal's frequencies (in Hz), for every frequency there's an amplitude
    frequencies: npt.NDArray[np.float32]

    #: Signal's samples
    samples: npt.NDArray

    #: Sampling rate in Hz
    sampling_rate: float

    #: Data rate in bits/second
    data_rate: float

    #: Central frequency of the of the emitted light in Hz
    central_frequency: float = 0.0

    #: Line width of the signal, full width at half-maximum (FWHM), Hz
    linewidth: float = 0.0


@dataclass(frozen=True)
class DigitalSignal(Signal):
    dtype: ClassVar[npt.DTypeLike] = np.bool

    def __init__(
        self,
        *,
        time: Optional[npt.NDArray[np.float32]] = None,
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
        central_frequency: float = 0,
        linewidth: float = 0
    ) -> None:
        if time is None:
            duration = len(samples) / data_rate
            time = np.arange(0, duration, 1 / sampling_rate, dtype="float32")

        super().__init__(
            type="digital",
            time=time,
            frequencies=np.array([]),
            samples=samples,
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            central_frequency=central_frequency,
            linewidth=linewidth
        )


class ElectricalSignal(Signal):
    dtype: ClassVar[npt.DTypeLike] = np.float32

    def __init__(
        self,
        *,
        time: npt.NDArray[np.float32],
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
    ) -> None:
        super().__init__(
            type="electrical",
            time=time,
            frequencies=np.array([]),
            samples=samples,
            sampling_rate=sampling_rate,
            data_rate=data_rate
        )


class OpticalSignal(Signal):
    dtype: ClassVar[npt.DTypeLike] = np.float32

    def __init__(
        self,
        *,
        time: npt.NDArray[np.float32],
        frequencies: Optional[npt.NDArray[np.float32]] = None,
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
        central_frequency: float,
        linewidth: float
    ) -> None:
        if frequencies is None:
            frequencies = np.array([central_frequency], dtype="float32")

        super().__init__(
            type="optical",
            time=time,
            frequencies=frequencies,
            samples=samples,
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            central_frequency=central_frequency,
            linewidth=linewidth
        )


@dataclass(frozen=True)
class NRZEncoder:
    #: Sampling rate in Hz
    sampling_rate: float

    def __call__(self, digital_signal: DigitalSignal) -> ElectricalSignal:
        signal_duration = len(digital_signal.samples) / digital_signal.data_rate
        time = np.arange(0, signal_duration, 1 / self.sampling_rate, dtype="float32")

        samples = samples=np.zeros(
            shape=(len(time),),
            dtype=ElectricalSignal.dtype
        )

        # Oversample input signal if needed
        samples = np.repeat(
            digital_signal.samples,
            math.ceil(len(time) / len(digital_signal.samples))
        )[:len(time)]

        return ElectricalSignal(
            time=time,
            samples=samples,
            sampling_rate=self.sampling_rate,
            data_rate=digital_signal.data_rate,
        )


@dataclass(frozen=True)
class System:
    #: System components
    components: list["Component"]

    def __call__(self, input_signal: DigitalSignal) -> Signal:
        result: Signal = input_signal

        for component in self.components:
            result = component(result)

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

    def __call__(self, electrical_signal: ElectricalSignal) -> OpticalSignal:
        # Produce carrier signal
        signal = np.ones(
            shape=(len(electrical_signal.samples), self.frequency_count),
            dtype=OpticalSignal.dtype
        )
        signal *= self.power ** 0.5

        # PAM modulation
        signal *= abs(
            electrical_signal.samples.repeat(
                self.frequency_count
            ).reshape(signal.shape)
        ) ** 0.5

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

        return OpticalSignal(
            time=electrical_signal.time,
            frequencies=frequencies,
            samples=signal,
            sampling_rate=electrical_signal.sampling_rate,
            data_rate=electrical_signal.data_rate,
            central_frequency=self.central_frequency,
            linewidth=self.linewidth
        )


@dataclass(frozen=True)
class AWGN:
    #: Signal's power
    power: float = 2e-5

    def __call__(self, optical_signal: OpticalSignal) -> OpticalSignal:
        noise_amplitude = self.power ** 0.5
        noise = np.random.normal(
            0.0,
            noise_amplitude,
            optical_signal.samples.shape
        )

        samples = np.array(optical_signal.samples)
        samples += noise

        return OpticalSignal(
            time=optical_signal.time,
            frequencies=optical_signal.frequencies,
            samples=samples,
            sampling_rate=optical_signal.sampling_rate,
            data_rate=optical_signal.data_rate,
            central_frequency=optical_signal.central_frequency,
            linewidth=optical_signal.linewidth
        )


def gaussian_distribution(x: npt.NDArray, mean: float, variance: float) -> npt.NDArray:
    return 1 / (variance * ((2 * math.pi) ** 0.5)) * math.e ** (-1/2 * ((x - mean) / variance) ** 2)


def linewidth_from_hz_to_nm(central_frequency_hz: float, linewidth_hz: float) -> float:
    lambda_hi = Constants.c / (central_frequency_hz - linewidth_hz * 0.5) * 1e9
    lambda_lo = Constants.c / (central_frequency_hz + linewidth_hz * 0.5) * 1e9

    return lambda_hi - lambda_lo


@dataclass(frozen=True)
class ChromaticDispersion:
    #: Dispersion in ps/nm
    amount: float

    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        wavelength_deltas = (Constants.c * 1e9 / signal.frequencies) - (Constants.c * 1e9 / signal.central_frequency)

        # Each wavelength (or frequency) will be shifted in time by some number
        # of samples
        timeshifts = (
            self.amount * 1e-12 * signal.sampling_rate * wavelength_deltas
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

        return OpticalSignal(
            time=signal.time,
            frequencies=signal.frequencies,
            samples=samples,
            sampling_rate=signal.sampling_rate,
            data_rate=signal.data_rate,
            central_frequency=signal.central_frequency,
            linewidth=signal.linewidth
        )


@dataclass(frozen=True)
class OpticalLoss:
    # Optical signal power loss in dB
    loss: float

    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        samples = np.array(signal.samples)
        samples /= 10 ** (self.loss * 0.1 * 0.5)

        return OpticalSignal(
            time=signal.time,
            frequencies=signal.frequencies,
            samples=samples,
            sampling_rate=signal.sampling_rate,
            data_rate=signal.data_rate,
            central_frequency=signal.central_frequency,
            linewidth=signal.linewidth
        )

@dataclass(frozen=True)
class OpticalChannel:
    #: Attenuation in the channel in dB/km
    attenuation: float

    #: Chromatic dispersion coefficient in ps/(nm * km)
    chromatic_dispersion: float

    #: Transmission distance in km
    length: float

    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
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

    def __call__(self, signal: ElectricalSignal) -> ElectricalSignal:
        sos = sp.signal.bessel(
            self.order,
            self.cutoff_frequency,
            "low",
            fs=int(signal.sampling_rate),
            output="sos",
            norm="delay"
        )

        filtered_amplitude = sp.signal.sosfilt(sos, signal.samples)
        filtered_samples = np.array(signal.samples)
        filtered_samples = filtered_amplitude

        return ElectricalSignal(
            time=signal.time,
            samples=filtered_samples,
            sampling_rate=signal.sampling_rate,
            data_rate=signal.data_rate,
        )
