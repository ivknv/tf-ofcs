# -*- coding: utf-8 -*-

import math
import tensorflow as tf
import numpy as np

from optical_communication_system.components import BandPassFilter
from optical_communication_system.datatypes import ElectricalSignal, Timing
from optical_communication_system.utils import timing_to_samples


def test_bandpass_filter() -> None:
    bandpass_filter = BandPassFilter(min_frequency=[0.2, 0.1], max_frequency=[0.5, 0.3])
    timing = Timing(sampling_rate=2.0, sample_count=400)

    time = timing_to_samples(timing)

    input_signal = ElectricalSignal(
        timing=timing,
        samples=[[
            tf.math.cos(2 * math.pi * 0.10 * time) +
            tf.math.cos(2 * math.pi * 0.25 * time) +
            tf.math.cos(2 * math.pi * 0.30 * time) +
            tf.math.cos(2 * math.pi * 0.55 * time)
        ] * 2]
    )

    filtered_signal = bandpass_filter(input_signal)

    assert isinstance(filtered_signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(filtered_signal.samples) == [1, 2, 400])  # type: ignore

    fft = tf.signal.fft(tf.cast(filtered_signal.samples, tf.complex64))
    freqs = np.fft.fftfreq(tf.shape(fft)[-1].numpy(), 1.0 / timing.sampling_rate.numpy())

    amplitudes1 = tf.abs(fft[0, 0])
    amplitudes2 = tf.abs(fft[0, 1])
    expected_amplitude_pass_band1 = 200.0
    expected_amplitude_pass_band2 = 300.0
    expected_amplitude_rejection_band = 0.0

    pass_band_mask1 = tf.logical_and(
        tf.math.greater_equal(tf.abs(freqs), 0.2),
        tf.math.less_equal(tf.abs(freqs), 0.5)
    )
    rejection_band_mask1 = tf.logical_or(
        tf.math.less(tf.abs(freqs), 0.2),
        tf.math.greater(tf.abs(freqs), 0.5)
    )

    pass_band_mask2 = tf.logical_and(
        tf.math.greater_equal(tf.abs(freqs), 0.1),
        tf.math.less_equal(tf.abs(freqs), 0.3)
    )
    rejection_band_mask2 = tf.logical_or(
        tf.math.less(tf.abs(freqs), 0.1),
        tf.math.greater(tf.abs(freqs), 0.3)
    )

    assert tf.math.reduce_all(
        tf.abs(
            tf.reduce_sum(tf.boolean_mask(amplitudes1, pass_band_mask1)) - 4 * expected_amplitude_pass_band1
        ) < 1e-1
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            tf.boolean_mask(amplitudes1, rejection_band_mask1) - expected_amplitude_rejection_band
        ) < 1e-3
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            tf.reduce_sum(tf.boolean_mask(amplitudes2, pass_band_mask2)) - 4 * expected_amplitude_pass_band2
        ) < 1e-1
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            tf.boolean_mask(amplitudes2, rejection_band_mask2) - expected_amplitude_rejection_band
        ) < 1e-3
    )  # type: ignore
