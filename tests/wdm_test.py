# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import (
    OpticalMultiplexer, Laser, OpticalDemultiplexer, RedistributeDimensions
)
from tf_ofcs.datatypes import OpticalSignal, Timing


def test_optical_multiplexer() -> None:
    omux = OpticalMultiplexer(4)

    input_signal = OpticalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=8),
        samples=tf.cast(tf.reshape(tf.range(64), [8, 1, 8]), tf.complex64)
    )

    expected_output_signal = tf.cast([
        [[ 48,  52,  56,  60,  64,  68,  72,  76]],
        [[176, 180, 184, 188, 192, 196, 200, 204]]
    ], tf.complex64) / tf.cast(4 ** 0.5, tf.complex64)

    output_signal = omux(input_signal)

    assert isinstance(output_signal, OpticalSignal)
    assert tf.math.reduce_all(tf.shape(output_signal.samples) == [2, 1, 8])  # type: ignore[operator]

    error = tf.abs(output_signal.samples - expected_output_signal)

    assert tf.math.reduce_all(error < 1e-9)  # type: ignore[operator]


def test_optical_demultiplexer() -> None:
    freqs = [-1e9, 0.0, 1e9]

    laser = Laser(power=1e-3, frequency=freqs, parameter_dims=len(freqs), batch_size=2)
    odemux = OpticalDemultiplexer(frequencies=freqs, channel_width=0.5e9)
    omux = OpticalMultiplexer(3)

    timing = Timing(sampling_rate=10e9, sample_count=1000)
    signal_before_mux = RedistributeDimensions(factors=(3, 1))(laser(timing))

    assert tf.math.reduce_all(
        tf.shape(signal_before_mux.samples) == [6, 1, 1000]
    )  # type: ignore[operator]

    mux_signal = omux(signal_before_mux)

    assert isinstance(mux_signal, OpticalSignal)
    assert tf.math.reduce_all(tf.shape(mux_signal.samples) == [2, 1, 1000])  # type: ignore[operator]

    output_signal = odemux(mux_signal)

    assert isinstance(output_signal, OpticalSignal)
    assert tf.math.reduce_all(tf.shape(output_signal.samples) == [6, 1, 1000])  # type: ignore[operator]

    error = tf.abs((output_signal.samples - signal_before_mux.samples) / signal_before_mux.samples)
    assert tf.math.reduce_all(error < 0.001)  # type: ignore[operator]
