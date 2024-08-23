# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Optional
from .components.datatypes import OpticalSignal, ElectricalSignal

__all__ = ["plot_power", "plot_envelope", "plot_bitseq", "plot_spectrum"]


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


def plot_power(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    time_units: tuple[float, str] = (1e9, "ns"),
    power_units: tuple[float, str] = (1000.0, "mW"),
    label: str = "Signal power"
) -> Optional[Axes]:
    if isinstance(signal, OpticalSignal):
        _plot(
            ax,
            signal.timing.time.numpy() * time_units[0],
            np.sum(np.abs(signal.samples.numpy()), axis=1) ** 2 * power_units[0],
            label=label
        )
    elif isinstance(signal, ElectricalSignal):
        _plot(
            ax,
            signal.timing.time * time_units[0],
            signal.samples.numpy() ** 2 * power_units[0],
            label=label
        )
    else: # IQ signal case
        _plot(
            ax,
            signal[0].timing.time * time_units[0],
            signal[0].samples.numpy() ** 2 * power_units[0],
            label=f"{label} (I)"
        )

        _plot(
            ax,
            signal[1].timing.time * time_units[0],
            signal[1].samples.numpy() ** 2 * power_units[0],
            label=f"{label} (Q)"
        )

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, f"Power, {power_units[1]}")

    return ax


def plot_envelope(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    time_units: tuple[float, str] = (1e9, "ns"),
    envelope_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal envelope"
) -> Optional[Axes]:
    if isinstance(signal, OpticalSignal):
        time = signal.timing.time.numpy() * time_units[0]
        samples = np.sum(signal.samples.numpy(), axis=1) * envelope_units[0]

        _plot(ax, time, np.real(samples), label=f"{label} (real)")
        _plot(ax, time, np.imag(samples), label=f"{label} (imaginary)")
    elif isinstance(signal, ElectricalSignal):
        _plot(
            ax,
            signal.timing.time * time_units[0],
            signal.samples.numpy() * envelope_units[0],
            label=label
        )
    else: # IQ signal case
        _plot(
            ax,
            signal[0].timing.time * time_units[0],
            signal[0].samples.numpy() * envelope_units[0],
            label=f"{label} (I)"
        )

        _plot(
            ax,
            signal[1].timing.time * time_units[0],
            signal[1].samples.numpy() * envelope_units[0],
            label=f"{label} (Q)"
        )

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, f"Envelope, {envelope_units[1]}")

    return ax


def plot_bitseq(
    bitseq: tf.Tensor,
    ax: Optional[Axes] = None,
    label: str = "Bit sequence"
) -> Optional[Axes]:
    upsample_factor = 10

    seq = bitseq.numpy()
    indices = np.arange(0, seq.size, 1.0 / upsample_factor)

    _plot(ax, indices, seq.repeat(upsample_factor), label=label)

    _set_xlabel(ax, "Bit index")
    _set_ylabel(ax, "Bit value")

    return ax

def _plot_spectrum(
    ax: Optional[Axes],
    signal: OpticalSignal | ElectricalSignal,
    frequency_scale: float,
    amplitude_scale: float,
    label: str
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

    _plot(ax, freqs, np.abs(fft), label=label)


def plot_spectrum(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    frequency_units: tuple[float, str] = (1e-9, "GHz"),
    amplitude_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal spectrum"
) -> Optional[Axes]:
    if isinstance(signal, (OpticalSignal, ElectricalSignal)):
        _plot_spectrum(ax, signal, frequency_units[0], amplitude_units[0], label)
    else: # IQ signal case
        _plot_spectrum(ax, signal[0], frequency_units[0], amplitude_units[0], f"{label} (I)")
        _plot_spectrum(ax, signal[1], frequency_units[0], amplitude_units[0], f"{label} (Q)")

    _set_xlabel(ax, f"Frequency, {frequency_units[1]}")
    _set_ylabel(ax, f"Amplitude, {amplitude_units[1]}")

    return ax
