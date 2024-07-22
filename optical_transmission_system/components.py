# -*- coding: utf-8 -*-

from typing import Optional, TYPE_CHECKING
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
    "ElectricalLowPassFilter"
]


class Constants:
    #: Speed of light in vacuum in m/s
    c: float = 2.99792458E8


@dataclass
class Signal:
    #: Type of the signal
    type: "SignalType"

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

    def __init__(
        self,
        *,
        type: "SignalType",
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
        central_frequency: float = 0.0,
        linewidth: float = 0.0
    ) -> None:
        self.type = type
        self.samples = samples
        self.sampling_rate = sampling_rate
        self.data_rate = data_rate
        self.central_frequency = central_frequency
        self.linewidth = linewidth

@dataclass
class DigitalSignal(Signal):
    def __init__(
        self,
        *,
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
        central_frequency: float = 0,
        linewidth: float = 0
    ) -> None:
        super().__init__(
            type="digital",
            samples=samples,
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            central_frequency=central_frequency,
            linewidth=linewidth
        )


class ElectricalSignal(Signal):
    def __init__(
        self,
        *,
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
        central_frequency: float = 0,
        linewidth: float = 0
    ) -> None:
        super().__init__(
            type="electrical",
            samples=samples,
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            central_frequency=central_frequency,
            linewidth=linewidth
        )


class OpticalSignal(Signal):
    def __init__(
        self,
        *,
        samples: npt.NDArray,
        sampling_rate: float,
        data_rate: float,
        central_frequency: float,
        linewidth: float
    ) -> None:
        super().__init__(
            type="optical",
            samples=samples,
            sampling_rate=sampling_rate,
            data_rate=data_rate,
            central_frequency=central_frequency,
            linewidth=linewidth
        )


@dataclass
class NRZEncoder:
    #: Sampling rate in Hz
    sampling_rate: float

    def __call__(self, digital_signal: DigitalSignal) -> ElectricalSignal:
        signal_duration = len(digital_signal.samples) / digital_signal.data_rate
        time = np.arange(0, signal_duration, 1 / self.sampling_rate, dtype="float32")

        samples = samples=np.zeros(
            shape=(len(time), 1),
            dtype=[("time", "float32"), ("amplitude", "float32")]
        )

        samples["time"][:,0] = time

        # Oversample input signal if needed
        samples["amplitude"][:,0] = np.repeat(
            digital_signal.samples,
            math.ceil(len(time) / len(digital_signal.samples))
        )[:len(time)]

        return ElectricalSignal(
            samples=samples,
            sampling_rate=self.sampling_rate,
            data_rate=digital_signal.data_rate,
            central_frequency=digital_signal.central_frequency,
            linewidth=digital_signal.linewidth
        )


@dataclass
class System:
    #: System components
    components: list["Component"]

    def __init__(self, components: Optional[list["Component"]] = None) -> None:
        self.components = components or []

    def add(self, component: "Component", /, *args, **kwargs) -> None:
        self.components.append(component)

    def __call__(self, input_signal: DigitalSignal) -> Signal:
        result: Signal = input_signal

        for component in self.components:
            result = component(result)

        return result


@dataclass
class Transmitter:
    #: Output power in watts
    power: float = 2e-3

    #: Signal-to-noise ratio in dB
    snr: float = 1000

    #: Central frequency of the of the emitted light in Hz
    central_frequency: float = 194.414e12

    #: Line width of the signal, full width at half-maximum (FWHM), Hz
    linewidth: float = 0.0

    def noise(self, signal: np.ndarray) -> np.ndarray:
        noise_amplitude = (self.power ** 0.5) / (10 ** (self.snr * 0.1))
        noise = np.random.normal(0.0, noise_amplitude, signal.shape)

        return noise

    def __call__(self, electrical_signal: ElectricalSignal) -> OpticalSignal:
        # Produce carrier signal
        signal = np.ones(
            shape=electrical_signal.samples.shape,
            dtype=[("time", "float32"), ("amplitude", "float32")]
        )
        signal["time"] = electrical_signal.samples["time"]
        signal["amplitude"] *= self.power ** 0.5

        # PAM modulation
        signal["amplitude"] *= abs(electrical_signal.samples["amplitude"]) ** 0.5

        return OpticalSignal(
            samples=signal,
            sampling_rate=electrical_signal.sampling_rate,
            data_rate=electrical_signal.data_rate,
            central_frequency=self.central_frequency,
            linewidth=self.linewidth
        )


@dataclass
class AWGN:
    #: Signal's power
    power: float = 2e-5

    def __call__(self, optical_signal: OpticalSignal) -> OpticalSignal:
        noise_amplitude = self.power ** 0.5
        noise = np.random.normal(
            0.0,
            noise_amplitude,
            optical_signal.samples["amplitude"][:,0].shape
        )

        samples = np.array(optical_signal.samples)
        samples["amplitude"][:,0] += noise

        return OpticalSignal(
            samples=samples,
            sampling_rate=optical_signal.sampling_rate,
            data_rate=optical_signal.data_rate,
            central_frequency=optical_signal.central_frequency,
            linewidth=optical_signal.linewidth
        )


def gaussian_distribution(x: npt.NDArray, mean: float, variance: float) -> npt.NDArray[np.float32]:
    return 1 / (variance * ((2 * math.pi) ** 0.5)) * math.e ** (-1/2 * ((x - mean) / variance) ** 2)


def linewidth_from_hz_to_nm(central_frequency_hz: float, linewidth_hz: float) -> float:
    lambda_hi = Constants.c / (central_frequency_hz - linewidth_hz * 0.5) * 1e9
    lambda_lo = Constants.c / (central_frequency_hz + linewidth_hz * 0.5) * 1e9

    return lambda_hi - lambda_lo


@dataclass
class ChromaticDispersion:
    #: Chromatic dispersion coefficient for a transmission length in ps/nm
    coefficient: float

    #: Wavelength range in nm, affects the number of samples in the kernel
    wavelength_range: float

    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        # This is the wavelength step for the kernel
        nm_per_sample = (1 / signal.sampling_rate) / (self.coefficient * 1e-12)

        kernel_sample_count = int(math.ceil(self.wavelength_range / nm_per_sample))

        kernel = np.ones(shape=(kernel_sample_count,))

        central_wavelength_nm = Constants.c / signal.central_frequency * 1e9
        linewidth_nm = linewidth_from_hz_to_nm(signal.central_frequency, signal.linewidth)

        # First we need to figure out the variance parameter of the gaussian distribution
        gauss_variance = linewidth_nm / (2 * math.log(2)) ** 0.5

        wavelengths = np.linspace(
            central_wavelength_nm - self.wavelength_range / 2,
            central_wavelength_nm + self.wavelength_range / 2,
            kernel_sample_count
        )

        distribution = gaussian_distribution(
            wavelengths,
            central_wavelength_nm,
            gauss_variance
        ) ** 0.5

        kernel *= distribution
        kernel /= np.sum(kernel)

        # Some of the signal's components are traveling faster than the central
        # wavelength, we have to shift the resulting signal back in time
        new_signal = np.array(signal.samples)
        new_signal["amplitude"][:,0] = np.convolve(
            new_signal["amplitude"][:,0],
            kernel
        )[len(kernel) // 2:][:len(signal.samples)]

        return OpticalSignal(
            samples=new_signal,
            sampling_rate=signal.sampling_rate,
            data_rate=signal.data_rate,
            central_frequency=signal.central_frequency,
            linewidth=signal.linewidth
        )


@dataclass
class OpticalChannel:
    #: Attenuation in the channel in dB/km
    attenuation: float

    #: Chromatic dispersion coefficient in ps/(nm * km)
    chromatic_dispersion: float

    #: Transmission distance in km
    length: float

    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        loss = self.attenuation * self.length

        samples = np.array(signal.samples)
        samples["amplitude"] /= 10 ** (loss * 0.1 * 0.5)

        attenuated_signal = OpticalSignal(
            samples=samples,
            sampling_rate=signal.sampling_rate,
            data_rate=signal.data_rate,
            central_frequency=signal.central_frequency,
            linewidth=signal.linewidth
        )

        if self.chromatic_dispersion >= 1e-6:
            dispersion = ChromaticDispersion(
                coefficient=self.chromatic_dispersion * self.length,
                wavelength_range=linewidth_from_hz_to_nm(
                    signal.central_frequency, signal.linewidth
                ) * 2
            )

            return dispersion(attenuated_signal)

        return attenuated_signal


@dataclass
class ElectricalLowPassFilter:
    #: Filter type
    type: "FilterType"

    #: Filter cutoff frequency in Hz
    cutoff_frequency: float

    #: Filter order
    order: int

    def __call__(self, signal: ElectricalSignal) -> ElectricalSignal:
        filters = {
            "butterworth": sp.signal.butter,
            "bessel": sp.signal.bessel,
            "chebyshev": sp.signal.cheby1
        }

        try:
            sos = filters[self.type](
                self.order,
                self.cutoff_frequency,
                "low",
                fs=int(signal.sampling_rate),
                output="sos"
            )
        except KeyError:
            raise ValueError(f"invalid filter type: {self.type}") from None

        filtered_power = sp.signal.sosfilt(sos, signal.samples["amplitude"][:,0])
        filtered_samples = np.array(signal.samples)
        filtered_samples["amplitude"][:,0] = filtered_power

        return ElectricalSignal(
            samples=filtered_samples,
            sampling_rate=signal.sampling_rate,
            data_rate=signal.data_rate,
            central_frequency=signal.central_frequency,
            linewidth=signal.linewidth
        )
