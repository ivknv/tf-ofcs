# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go

import tensorflow as tf

from typing import Optional

from . import constants
from .datatypes import OpticalSignal, ElectricalSignal
from .utils import timing_to_samples

__all__ = [
    "plot_power",
    "plot_envelope",
    "plot_phase",
    "plot_bitseq",
    "plot_spectrum",
    "plot_iq_constellation",
    "plot_eye_diagram"
]


def plot_power(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    power_units: tuple[float, str] = (1e3, "mW"),
    label: str = "Signal power",
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": f"Time, {time_units[1]}"},
            yaxis={"title": f"Power, {power_units[1]}"}
        ))

    if isinstance(signal, (ElectricalSignal, OpticalSignal)):
        time = timing_to_samples(signal.timing) * time_units[0]
        samples = tf.abs(signal.samples[index]) ** 2 * power_units[0]

        figure.add_trace(
            go.Scatter(x=time, y=samples, name=label, **kwargs),
            row=row,
            col=col
        )
    else: # IQ signal case
        time_i = timing_to_samples(signal[0].timing) * time_units[0]
        samples_i = tf.abs(signal[0].samples[index]) ** 2 * power_units[0]

        time_q = timing_to_samples(signal[1].timing) * time_units[0]
        samples_q = tf.abs(signal[1].samples[index]) ** 2 * power_units[0]

        figure.add_trace(
            go.Scatter(x=time_i, y=samples_i, name=f"{label} (I)", **kwargs),
            row=row,
            col=col
        )
        figure.add_trace(
            go.Scatter(x=time_q, y=samples_q, name=f"{label} (Q)", **kwargs),
            row=row,
            col=col
        )

    return figure


def plot_envelope(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    envelope_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal envelope",
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": f"Time, {time_units[1]}"},
            yaxis={"title": f"Envelope, {envelope_units[1]}"}
        ))

    if isinstance(signal, OpticalSignal):
        time = timing_to_samples(signal.timing) * time_units[0]
        samples = signal.samples[index] * envelope_units[0]

        figure.add_trace(
            go.Scatter(x=time, y=tf.math.real(samples), name=f"{label} (real)", **kwargs),
            row=row,
            col=col
        )
        figure.add_trace(
            go.Scatter(x=time, y=tf.math.imag(samples), name=f"{label} (imag)", **kwargs),
            row=row,
            col=col
        )
    elif isinstance(signal, ElectricalSignal):
        time = timing_to_samples(signal.timing) * time_units[0]
        samples = signal.samples[index] * envelope_units[0]

        figure.add_trace(
            go.Scatter(x=time, y=samples, name=label, **kwargs),
            row=row,
            col=col
        )
    else: # IQ signal case
        time_i = timing_to_samples(signal[0].timing) * time_units[0]
        samples_i = signal[0].samples[index] * envelope_units[0]

        time_q = timing_to_samples(signal[1].timing) * time_units[0]
        samples_q = signal[1].samples[index] * envelope_units[0]

        figure.add_trace(
            go.Scatter(x=time_i, y=samples_i, name=f"{label} (I)", **kwargs),
            row=row,
            col=col
        )
        figure.add_trace(
            go.Scatter(x=time_q, y=samples_q, name=f"{label} (Q)", **kwargs),
            row=row,
            col=col
        )

    return figure


def plot_phase(
    signal: OpticalSignal | tuple[ElectricalSignal, ElectricalSignal],
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    label: str = "Signal phase",
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": f"Time, {time_units[1]}"},
            yaxis={"title": "Phase, degrees", "range": [-180.0, 180.0]}
        ))

    if isinstance(signal, OpticalSignal):
        timing = signal.timing
        samples = signal.samples

        phase = tf.math.atan2(tf.math.real(samples[index]), tf.math.imag(samples[index]))
    else:
        timing = signal[0].timing
        phase = tf.math.atan2(signal[0].samples[index], signal[1].samples[index])

    phase /= constants.pi / 180.0

    time = timing_to_samples(timing) * time_units[0]
    figure.add_trace(
        go.Scatter(x=time, y=phase, name=label, **kwargs),
        row=row,
        col=col
    )

    return figure


def plot_bitseq(
    bitseq: tf.Tensor,
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    label: str = "Bit sequence",
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": "Bit index"},
            yaxis={"title": "Bit value"}
        ))

    upsample_factor = 10

    seq = bitseq[index]
    indices = tf.range(0, tf.size(seq), 1.0 / upsample_factor)

    figure.add_trace(
        go.Scatter(x=indices, y=tf.repeat(seq, upsample_factor), name=label, **kwargs),
        row=row,
        col=col
    )

    return figure


def _plot_spectrum(
    figure: go.Figure,
    signal: OpticalSignal | ElectricalSignal,
    index: int | tuple[int, ...],
    frequency_scale: float,
    amplitude_scale: float,
    label: str,
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    sampling_rate = signal.timing.sampling_rate.numpy()

    samples = signal.samples[index].numpy()

    fft = np.fft.fftshift(np.fft.fft(samples)) * amplitude_scale
    freqs = np.fft.fftshift(np.fft.fftfreq(fft.size, 1 / sampling_rate)) * frequency_scale

    figure.add_trace(
        go.Scatter(x=freqs, y=np.abs(fft), name=label, **kwargs),
        row=row,
        col=col
    )
    return figure


def plot_spectrum(
    signal: OpticalSignal | ElectricalSignal | tuple[ElectricalSignal, ElectricalSignal],
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    frequency_units: tuple[float, str] = (1e-9, "GHz"),
    amplitude_units: tuple[float, str] = (1.0, "AU"),
    label: str = "Signal spectrum",
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": f"Frequency, {frequency_units[1]}"},
            yaxis={"title": f"Amplitude, {amplitude_units[1]}"}
        ))

    if isinstance(signal, (OpticalSignal, ElectricalSignal)):
        _plot_spectrum(
            figure,
            signal,
            index,
            frequency_units[0],
            amplitude_units[0],
            label,
            row=row,
            col=col,
            **kwargs
        )
    else: # IQ signal case
        _plot_spectrum(
            figure,
            signal[0],
            index,
            frequency_units[0],
            amplitude_units[0],
            f"{label} (I)",
            row=row,
            col=col,
            **kwargs
        )
        _plot_spectrum(
            figure,
            signal[1],
            index,
            frequency_units[0],
            amplitude_units[0],
            f"{label} (Q)",
            row=row,
            col=col,
            **kwargs
        )

    return figure


def plot_iq_constellation(
    signal_or_constellation: tuple[ElectricalSignal, ElectricalSignal] | tf.Tensor,
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    iq_units: tuple[float, str] = (1.0, "AU"),
    label: str = "IQ signal constellation",
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": f"I, {iq_units[1]}"},
            yaxis={"title": f"Q, {iq_units[1]}"}
        ))

    if isinstance(signal_or_constellation, tf.Tensor):
        constellation = signal_or_constellation
        iq_symbols = constellation
    else:
        signal = signal_or_constellation
        iq_symbols = tf.stack([signal[0].samples[index], signal[1].samples[index]], axis=-1)

    iq_symbols = tf.reshape(iq_symbols, (tf.size(iq_symbols) // 2, 2))
    iq_symbols *= iq_units[0]

    kwargs.setdefault("mode", "markers")

    figure.add_trace(
        go.Scatter(x=iq_symbols[:,0], y=iq_symbols[:,1], name=label, **kwargs),
        row=row,
        col=col
    )
    return figure


def plot_eye_diagram(
    signal: ElectricalSignal,
    symbol_rate: float | tf.Tensor,
    figure: Optional[go.Figure] = None,
    index: int | tuple[int, ...] = (0, 0),
    time_units: tuple[float, str] = (1e9, "ns"),
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> go.Figure:
    if figure is None:
        figure = go.Figure(layout=go.Layout(
            xaxis={"title": f"Time, {time_units[1]}"},
            yaxis={"title": "Envelope (unitless)"},
            showlegend=False
        ))

    T_sym = int((signal.timing.sampling_rate / symbol_rate))

    samples = signal.samples[index][T_sym // 2:].numpy()

    time = np.linspace(0.0, 2.0, T_sym * 2)

    for i in range(samples.size // (T_sym * 2)):
        wave = samples[i * T_sym * 2:(i + 1) * T_sym * 2]

        figure.add_trace(
            go.Scatter(x=time, y=wave, **kwargs),
            row=row,
            col=col
        )

    return figure
