# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Optional

from . import constants
from .components.datatypes import OpticalSignal, ElectricalSignal
from .components.utils import timing_to_samples

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
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    power_units: tuple[float, str] = (1e3, "mW"),
    label: str = "Signal power",
    **kwargs
) -> None:
    if isinstance(signal, (ElectricalSignal, OpticalSignal)):
        time = timing_to_samples(signal.timing) * time_units[0]
        samples = tf.abs(signal.samples[index]) ** 2 * power_units[0]

        _plot(ax, time, samples, label=f"{label}", **kwargs)
    else: # IQ signal case
        time_i = timing_to_samples(signal[0].timing) * time_units[0]
        samples_i = tf.abs(signal[0].samples[index]) ** 2 * power_units[0]

        time_q = timing_to_samples(signal[1].timing) * time_units[0]
        samples_q = tf.abs(signal[1].samples[index]) ** 2 * power_units[0]

        _plot(ax, time_i, samples_i, label=f"{label} (I)", **kwargs)
        _plot(ax, time_q, samples_q, label=f"{label} (Q)", **kwargs)

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, f"Power, {power_units[1]}")


def plot_envelope(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    envelope_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal envelope",
    **kwargs
) -> None:
    if isinstance(signal, OpticalSignal):
        time = timing_to_samples(signal.timing) * time_units[0]
        samples = signal.samples[index] * envelope_units[0]

        _plot(ax, time, tf.real(samples), label=f"{label} (real)]", **kwargs)
        _plot(ax, time, tf.imag(samples), label=f"{label} (imaginary)", **kwargs)
    elif isinstance(signal, ElectricalSignal):
        time = timing_to_samples(signal.timing) * time_units[0]
        samples = signal.samples[index] * envelope_units[0]

        _plot(ax, time, samples, label=f"{label}", **kwargs)
    else: # IQ signal case
        time_i = timing_to_samples(signal[0].timing) * time_units[0]
        samples_i = signal[0].samples[index] * envelope_units[0]

        time_q = timing_to_samples(signal[1].timing) * time_units[0]
        samples_q = signal[1].samples[index] * envelope_units[0]

        _plot(ax, time_i, samples_i, label=f"{label} (I)", **kwargs)
        _plot(ax, time_q, samples_q, label=f"{label} (Q)", **kwargs)

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, f"Envelope, {envelope_units[1]}")


def plot_phase(
    signal: OpticalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    label: str = "Signal phase",
    **kwargs
) -> None:
    if isinstance(signal, OpticalSignal):
        timing = signal.timing
        samples = signal.samples

        phase = tf.math.atan2(tf.math.real(samples[index]), tf.math.imag(samples[index]))
    else:
        timing = signal[0].timing
        phase = tf.math.atan2(signal[0].samples[index], signal[1].samples[index])

    phase /= constants.pi / 180.0

    time = timing_to_samples(timing) * time_units[0]
    _plot(ax, time, phase, label=f"{label}", **kwargs)

    _set_xlabel(ax, f"Time, {time_units[1]}")
    _set_ylabel(ax, "Phase, degrees")


def plot_bitseq(
    bitseq: tf.Tensor,
    ax: Optional[Axes] = None,
    index: int | tuple[int, ...] = (0, 0),
    label: str = "Bit sequence",
    **kwargs
) -> None:
    upsample_factor = 10

    seq = bitseq[index]
    indices = tf.range(0, tf.size(seq), 1.0 / upsample_factor)

    _plot(ax, indices, tf.repeat(seq, upsample_factor), label=f"{label}", **kwargs)

    _set_xlabel(ax, "Bit index")
    _set_ylabel(ax, "Bit value")


def _plot_spectrum(
    ax: Optional[Axes],
    signal: OpticalSignal | ElectricalSignal,
    index: int | tuple[int, ...],
    frequency_scale: float,
    amplitude_scale: float,
    label: str,
    **kwargs
) -> None:
    sampling_rate = signal.timing.sampling_rate.numpy()

    samples = signal.samples[index].numpy()

    fft = np.fft.fftshift(np.fft.fft(samples)) * amplitude_scale
    freqs = np.fft.fftshift(np.fft.fftfreq(fft.size, 1 / sampling_rate)) * frequency_scale

    _plot(ax, freqs, np.abs(fft), label=f"{label}", **kwargs)


def plot_spectrum(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    ax: Optional[Axes] = None,
    index: int | tuple[int, ...] = (0, 0),
    frequency_units: tuple[float, str] = (1e-9, "GHz"),
    amplitude_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal spectrum",
    **kwargs
) -> None:
    if isinstance(signal, (OpticalSignal, ElectricalSignal)):
        _plot_spectrum(ax, signal, index, frequency_units[0], amplitude_units[0], label, **kwargs)
    else: # IQ signal case
        _plot_spectrum(ax, signal[0], index, frequency_units[0], amplitude_units[0], f"{label} (I)", **kwargs)
        _plot_spectrum(ax, signal[1], index, frequency_units[0], amplitude_units[0], f"{label} (Q)", **kwargs)

    _set_xlabel(ax, f"Frequency, {frequency_units[1]}")
    _set_ylabel(ax, f"Amplitude, {amplitude_units[1]}")


def plot_iq_constellation(
    signal_or_constellation: tuple[ElectricalSignal, ElectricalSignal] | tf.Tensor,
    ax: Optional[Axes] = None,
    index: int | tuple[int, ...] = (0, 0),
    iq_units: tuple[float, str] = (1.0, "AU"),
    label: str = "IQ signal constellation",
    **kwargs
) -> None:
    if isinstance(signal_or_constellation, tf.Tensor):
        constellation = signal_or_constellation
        iq_symbols = constellation
    else:
        signal = signal_or_constellation
        iq_symbols = tf.stack([signal[0].samples[index], signal[1].samples[index]], axis=-1)

    iq_symbols = tf.reshape(iq_symbols, (tf.size(iq_symbols) // 2, 2))
    iq_symbols *= iq_units[0]

    _scatter(ax, iq_symbols[:,0], iq_symbols[:,1], label=f"{label}", **kwargs)

    _set_xlabel(ax, f"I, {iq_units[1]}")
    _set_ylabel(ax, f"Q, {iq_units[1]}")


def plot_eye_diagram(
    signal: ElectricalSignal,
    symbol_rate: float | tf.Tensor,
    ax: Optional[Axes] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    label: str = "Signal eye diagram",
    **kwargs
) -> None:
    T_sym = int((signal.timing.sampling_rate / symbol_rate))

    samples = signal.samples[index][T_sym // 2:].numpy()

    pattern: list[np.ndarray] = []

    for i in range(samples.size // (T_sym * 2)):
        wave = samples[i * T_sym * 2:(i + 1) * T_sym * 2]
        pattern.append(wave)

    _plot(ax, np.linspace(0.0, 2.0, T_sym * 2), np.array(pattern).T, label=f"{label}", **kwargs)
