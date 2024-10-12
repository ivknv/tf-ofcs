# -*- coding: utf-8 -*-

import math
import tensorflow as tf
import numpy as np

from optical_communication_system.components import BandPassFilter, TimingGenerator
from optical_communication_system.components.datatypes import ElectricalSignal


def test_bandpass_filter() -> None:
    bandpass_filter = BandPassFilter(min_frequency=0.2, max_frequency=0.5)
    timing_generator = TimingGenerator(sampling_rate=2.0, symbol_rate=1.0)
    timing = timing_generator(tf.constant(200))

    input_signal = ElectricalSignal(
        timing=timing,
        samples=[
            tf.math.cos(2 * math.pi * 0.25 * timing.time) +
            tf.math.cos(2 * math.pi * 0.30 * timing.time) +
            tf.math.cos(2 * math.pi * 0.1  * timing.time) +
            tf.math.cos(2 * math.pi * 0.55 * timing.time)
        ]
    )

    filtered_signal = bandpass_filter(input_signal)
    fft = tf.signal.fft(tf.cast(filtered_signal.samples, tf.complex64))
    freqs = np.fft.fftfreq(tf.size(fft).numpy(), 1.0 / timing.sampling_rate.numpy())

    amplitudes = tf.abs(fft[0])
    expected_amplitude_pass_band = 200.0
    expected_amplitude_rejection_band = 0.0

    pass_band_mask = tf.logical_and(
        tf.math.greater(tf.abs(freqs), 0.2),
        tf.math.less(tf.abs(freqs), 0.5)
    )
    rejection_band_mask = tf.logical_or(
        tf.math.less(tf.abs(freqs), 0.2),
        tf.math.greater(tf.abs(freqs), 0.5)
    )

    assert isinstance(filtered_signal, ElectricalSignal)

    assert tf.reduce_all(
        tf.abs(
            tf.reduce_sum(tf.boolean_mask(amplitudes, pass_band_mask)) - 4 * expected_amplitude_pass_band
        ) < 1e-1
    )

    assert tf.reduce_all(
        tf.abs(
            tf.boolean_mask(amplitudes, rejection_band_mask) - expected_amplitude_rejection_band
        ) < 1e-3
    )
