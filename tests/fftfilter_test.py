# -*- coding: utf-8 -*-

import tensorflow as tf

import math

from tf_ofcs.components import FFTFilter
from tf_ofcs.datatypes import ElectricalSignal, Timing
from tf_ofcs.utils import timing_to_samples


def test_fftfilter() -> None:
    def frequency_response(
        frequencies: tf.Tensor,
        /,
        batch_size,
        **kwargs
    ) -> tf.Tensor:
        return tf.fill(
            [batch_size, tf.size(frequencies)],
            tf.cast(0.5, tf.complex64)
        )

    fft_filter = FFTFilter(frequency_response_function=frequency_response)

    timing = Timing(sampling_rate=2.0, sample_count=400)
    time = timing_to_samples(timing)

    input_signal = ElectricalSignal(
        timing=timing,
        samples=[
            tf.math.cos(2 * math.pi * 0.10 * time) +
            tf.math.cos(2 * math.pi * 0.25 * time) +
            tf.math.cos(2 * math.pi * 0.30 * time) +
            tf.math.cos(2 * math.pi * 0.55 * time)
        ] * 2
    )

    filtered_signal = fft_filter(input_signal)

    assert isinstance(filtered_signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(filtered_signal.samples) == [2, 400])  # type: ignore[operator]

    expected_samples = input_signal.samples * 0.5

    error = tf.abs(filtered_signal.samples - expected_samples)
    assert tf.math.reduce_all(error < 1e-6)  # type: ignore[operator]
