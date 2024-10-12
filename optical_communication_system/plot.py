# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import tensorflow as tf

from typing import Optional
from .components.datatypes import OpticalSignal, ElectricalSignal

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
    if isinstance(signal, (ElectricalSignal, OpticalSignal)):
        time = signal.timing.time.numpy() * time_units[0]
        samples = np.abs(signal.samples.numpy()) ** 2 * power_units[0]

        for batch in range(len(samples)):
            _plot(ax, time, samples[batch], label=f"{label} [{batch}]", **kwargs)
    else: # IQ signal case
        time_i = signal[0].timing.time.numpy() * time_units[0]
        samples_i = np.abs(signal[0].samples.numpy()) ** 2 * power_units[0]

        time_q = signal[1].timing.time.numpy() * time_units[0]
        samples_q = np.abs(signal[1].samples.numpy()) ** 2 * power_units[0]

        for batch in range(len(samples_i)):
            _plot(ax, time_i, samples_i[batch], label=f"{label} (I) [{batch}]", **kwargs)
            _plot(ax, time_q, samples_q[batch], label=f"{label} (Q) [{batch}]", **kwargs)

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
        samples = signal.samples.numpy() * envelope_units[0]

        for batch in range(len(samples)):
            _plot(ax, time, np.real(samples[batch]), label=f"{label} (real) [{batch}]", **kwargs)
            _plot(ax, time, np.imag(samples[batch]), label=f"{label} (imaginary) [{batch}]", **kwargs)
    elif isinstance(signal, ElectricalSignal):
        time = signal.timing.time.numpy() * time_units[0]
        samples = signal.samples.numpy() * envelope_units[0]

        for batch in range(len(samples)):
            _plot(ax, time, samples[batch], label=f"{label} [{batch}]", **kwargs)
    else: # IQ signal case
        time_i = signal[0].timing.time.numpy() * time_units[0]
        samples_i = signal[0].samples.numpy() * envelope_units[0]

        time_q = signal[1].timing.time.numpy() * time_units[0]
        samples_q = signal[1].samples.numpy() * envelope_units[0]

        for batch in range(len(samples_i)):
            _plot(ax, time_i, samples_i[batch], label=f"{label} (I) [{batch}]", **kwargs)

            _plot(ax, time_q, samples_q[batch], label=f"{label} (Q) [{batch}]", **kwargs)

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
        samples = signal.samples.numpy()

        phase = np.arctan2(np.real(samples), np.imag(samples))
    else:
        time = signal[0].timing.time.numpy() * time_units[0]
        phase = np.arctan2(signal[0].samples.numpy(), signal[1].samples.numpy())

    phase /= np.pi / 180.0

    for batch in range(len(phase)):
        _plot(ax, time, phase[batch], label=f"{label} [{batch}]", **kwargs)

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
    indices = np.arange(0, seq.shape[1], 1.0 / upsample_factor)

    for batch in range(len(seq)):
        _plot(ax, indices, seq[batch].repeat(upsample_factor), label=f"{label} [{batch}]", **kwargs)

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

    samples = signal.samples.numpy()

    fft = np.fft.fftshift(np.fft.fft(samples)) * amplitude_scale
    freqs = np.fft.fftshift(np.fft.fftfreq(fft.shape[1], 1 / sampling_rate)) * frequency_scale

    for batch in range(len(fft)):
        _plot(ax, freqs, np.abs(fft[batch]), label=f"{label} [{batch}]", **kwargs)


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
    signal_or_constellation: tuple[ElectricalSignal, ElectricalSignal] | tf.Tensor,
    ax: Optional[Axes] = None,
    iq_units: tuple[float, str] = (1.0, "AU"),
    label: str = "IQ signal constellation",
    **kwargs
) -> None:
    if isinstance(signal_or_constellation, tf.Tensor):
        constellation = signal_or_constellation
        iq_symbols = constellation
    else:
        signal = signal_or_constellation
        iq_symbols = tf.stack([signal[0].samples, signal[1].samples], axis=-1)

    iq_symbols *= iq_units[0]

    for batch in range(len(iq_symbols)):
        _scatter(
            ax,
            iq_symbols[batch,:,0].numpy(),
            iq_symbols[batch,:,1].numpy(),
            label=f"{label} [{batch}]",
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

    samples = signal.samples.numpy()[:,T_sym // 2:]

    for batch in range(len(signal.samples)):
        pattern: list[np.ndarray] = []

        for i in range(samples.size // (T_sym * 2)):
            wave = samples[batch, i * T_sym * 2:(i + 1) * T_sym * 2]
            pattern.append(wave)

        _plot(ax, np.linspace(0.0, 2.0, T_sym * 2), np.array(pattern).T, label=f"{label} [{batch}]", **kwargs)
