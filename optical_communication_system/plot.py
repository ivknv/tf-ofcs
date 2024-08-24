# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Optional
from .components.datatypes import OpticalSignal, ElectricalSignal
from .components import QAMConstellation

__all__ = [
    "plot_power",
    "plot_envelope",
    "plot_phase",
    "plot_bitseq",
    "plot_spectrum",
    "plot_iq_constellation",
    "plot_eye_diagram"
]


def _set_xlabel(ax: Optional[Axes], label: str) -> None:
    if ax is not None:
        ax.set_xlabel(label)
    else:
        plt.xlabel(label)


def _set_ylabel(ax: Optional[Axes], label: str) -> None:
    if ax is not None:
        ax.set_ylabel(label)
    else:
        plt.ylabel(label)


def _plot(ax: Optional[Axes], *args, **kwargs) -> None:
    plot = ax.plot if ax is not None else plt.plot
    plot(*args, **kwargs)


def _scatter(ax: Optional[Axes], *args, **kwargs) -> None:
    scatter = ax.scatter if ax is not None else plt.scatter
    scatter(*args, **kwargs)


def plot_power(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    time_units: tuple[float, str] = (1e9, "ns"),
    power_units: tuple[float, str] = (1e3, "mW"),
    label: str = "Signal power",
    **kwargs
) -> None:
    if isinstance(signal, OpticalSignal):
        _plot(
            ax,
            signal.timing.time.numpy() * time_units[0],
            np.sum(np.abs(signal.samples.numpy()) ** 2, axis=1) * power_units[0],
            label=label,
            **kwargs
        )
    elif isinstance(signal, ElectricalSignal):
        _plot(
            ax,
            signal.timing.time * time_units[0],
            signal.samples.numpy() ** 2 * power_units[0],
            label=label,
            **kwargs
        )
    else: # IQ signal case
        _plot(
            ax,
            signal[0].timing.time * time_units[0],
            signal[0].samples.numpy() ** 2 * power_units[0],
            label=f"{label} (I)",
            **kwargs
        )

        _plot(
            ax,
            signal[1].timing.time * time_units[0],
            signal[1].samples.numpy() ** 2 * power_units[0],
            label=f"{label} (Q)",
            **kwargs
        )

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, f"Power, {power_units[1]}")


def plot_envelope(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    time_units: tuple[float, str] = (1e9, "ns"),
    envelope_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal envelope",
    **kwargs
) -> None:
    if isinstance(signal, OpticalSignal):
        time = signal.timing.time.numpy() * time_units[0]
        samples = np.sum(signal.samples.numpy(), axis=1) * envelope_units[0]

        _plot(ax, time, np.real(samples), label=f"{label} (real)", **kwargs)
        _plot(ax, time, np.imag(samples), label=f"{label} (imaginary)", **kwargs)
    elif isinstance(signal, ElectricalSignal):
        _plot(
            ax,
            signal.timing.time * time_units[0],
            signal.samples.numpy() * envelope_units[0],
            label=label,
            **kwargs
        )
    else: # IQ signal case
        _plot(
            ax,
            signal[0].timing.time * time_units[0],
            signal[0].samples.numpy() * envelope_units[0],
            label=f"{label} (I)",
            **kwargs
        )

        _plot(
            ax,
            signal[1].timing.time * time_units[0],
            signal[1].samples.numpy() * envelope_units[0],
            label=f"{label} (Q)",
            **kwargs
        )

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, f"Envelope, {envelope_units[1]}")


def plot_phase(
    signal: OpticalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    time_units: tuple[float, str] = (1e9, "ns"),
    label: str = "Signal phase",
    **kwargs
) -> None:
    if isinstance(signal, OpticalSignal):
        time = signal.timing.time.numpy() * time_units[0]
        samples = np.sum(signal.samples.numpy(), axis=1)

        phase = np.arctan2(np.real(samples), np.imag(samples))
    else:
        time = signal[0].timing.time.numpy() * time_units[0]
        phase = np.arctan2(signal[0].samples.numpy(), signal[1].samples.numpy())

    phase /= np.pi / 180.0

    _plot(ax, time, phase, label=label, **kwargs)

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, "Phase, degrees")


def plot_bitseq(
    bitseq: tf.Tensor,
    ax: Optional[Axes] = None,
    label: str = "Bit sequence",
    **kwargs
) -> None:
    upsample_factor = 10

    seq = bitseq.numpy()
    indices = np.arange(0, seq.size, 1.0 / upsample_factor)

    _plot(ax, indices, seq.repeat(upsample_factor), label=label, **kwargs)

    _set_xlabel(ax, "Bit index")
    _set_ylabel(ax, "Bit value")

def _plot_spectrum(
    ax: Optional[Axes],
    signal: OpticalSignal | ElectricalSignal,
    frequency_scale: float,
    amplitude_scale: float,
    label: str,
    **kwargs
) -> None:
    sampling_rate = signal.timing.sampling_rate.numpy()

    if isinstance(signal, OpticalSignal):
        samples = np.sum(signal.samples.numpy(), axis=1)
    else:
        samples = signal.samples.numpy()

    fft = np.fft.fftshift(np.fft.fft(samples)) * amplitude_scale
    freqs = np.fft.fftshift(np.fft.fftfreq(fft.size, 1 / sampling_rate)) * frequency_scale

    if isinstance(signal, OpticalSignal):
        freqs += signal.central_frequency * frequency_scale

    _plot(ax, freqs, np.abs(fft), label=label, **kwargs)


def plot_spectrum(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    frequency_units: tuple[float, str] = (1e-9, "GHz"),
    amplitude_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal spectrum",
    **kwargs
) -> None:
    if isinstance(signal, (OpticalSignal, ElectricalSignal)):
        _plot_spectrum(ax, signal, frequency_units[0], amplitude_units[0], label, **kwargs)
    else: # IQ signal case
        _plot_spectrum(ax, signal[0], frequency_units[0], amplitude_units[0], f"{label} (I)", **kwargs)
        _plot_spectrum(ax, signal[1], frequency_units[0], amplitude_units[0], f"{label} (Q)", **kwargs)

    _set_xlabel(ax, f"Frequency, {frequency_units[1]}")
    _set_ylabel(ax, f"Amplitude, {amplitude_units[1]}")


def plot_iq_constellation(
    signal_or_constellation: tuple[ElectricalSignal, ElectricalSignal] | QAMConstellation,
    ax: Optional[Axes] = None,
    iq_units: tuple[float, str] = (1.0, "AU"),
    label: str = "IQ signal constellation",
    **kwargs
) -> None:
    if isinstance(signal_or_constellation, QAMConstellation):
        constellation = signal_or_constellation
        mapping = constellation.mapping.numpy()

        _scatter(ax, mapping[:,0], mapping[:,1], label=label, **kwargs)
    else:
        signal = signal_or_constellation
        _scatter(
            ax,
            signal[0].samples.numpy() * iq_units[0],
            signal[1].samples.numpy() * iq_units[0],
            label=label,
            **kwargs
        )

    _set_xlabel(ax, f"I, {iq_units[1]}")
    _set_ylabel(ax, f"Q, {iq_units[1]}")


def plot_eye_diagram(
    signal: ElectricalSignal,
    ax: Optional[Axes] = None,
    time_units: tuple[float, str] = (1e9, "ns"),
    label: str = "Signal eye diagram",
    **kwargs
) -> None:
    T_sym = int((signal.timing.sampling_rate / signal.timing.symbol_rate).numpy())

    pattern: list[np.ndarray] = []

    samples = signal.samples.numpy()

    for i in range(samples.size // (T_sym * 2)):
        wave = samples[i * T_sym * 2:(i + 1) * T_sym * 2]
        pattern.append(wave)

    _plot(ax, np.linspace(-1.0, 1.0, T_sym * 2), np.array(pattern).T, **kwargs)
