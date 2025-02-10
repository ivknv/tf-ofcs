# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import OpticalAmplifier
from tf_ofcs.datatypes import OpticalSignal, Timing


def test_optical_amplifier() -> None:
    rng = tf.random.Generator.from_seed(123, alg="philox")
    amplifier = OpticalAmplifier(gain=[10.0, 20.0], noise_figure=[4.0, 7.0], rng=rng)

    input_signal = OpticalSignal(
        timing=Timing(sampling_rate=1000.0, sample_count=1000),
        samples=tf.cast(tf.fill([2, 2, 1000], 1.0), tf.complex64)
    )

    amplified_signal = amplifier(input_signal)

    assert isinstance(amplified_signal, OpticalSignal)
    assert tf.math.reduce_all(tf.shape(amplified_signal.samples) == [2, 2, 1000])  # type: ignore[operator]

    amplification = tf.abs(amplified_signal.samples / input_signal.samples)
    error = tf.abs(tf.concat(
        [
            amplification[:, 0, :] - 10.0 ** 0.5,
            amplification[:, 1, :] - 20.0 ** 0.5
        ],
        1
    ))

    assert tf.math.reduce_all(tf.math.less(error, 0.0001))  # type: ignore[operator]
