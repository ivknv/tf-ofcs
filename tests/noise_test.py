# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import OpticalAWGN, Laser
from tf_ofcs.datatypes import OpticalSignal, Timing
from tf_ofcs.utils import from_db, dbm_to_w


def test_optical_awgn() -> None:
    rng = tf.random.Generator.from_seed(1234, alg="philox")
    awgn = OpticalAWGN(rng=rng)

    batch_size = tf.constant(3)

    laser = Laser(power=tf.repeat(dbm_to_w(0.0), 3))
    timing = Timing(sampling_rate=20.0, sample_count=20000)

    input_signal = laser(timing, batch_size=batch_size)

    assert tf.math.reduce_all(tf.shape(input_signal.samples) == (batch_size, 20000))  # type: ignore

    relative_noise = tf.constant([10.0, 0.0, -5.0])  # in dB
    absolute_noise = tf.constant([1.0, 5.0, -5.0])  # in dBm

    signal_with_relative_noise = awgn(
        input_signal,
        optical_awgn__relative_db=relative_noise
    )
    signal_with_absolute_noise = awgn(
        input_signal,
        optical_awgn__absolute_dbm=absolute_noise
    )

    assert isinstance(signal_with_relative_noise, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(signal_with_relative_noise.samples) == tf.shape(input_signal.samples)
    )  # type: ignore

    average_power1 = tf.math.reduce_mean(tf.abs(signal_with_relative_noise.samples) ** 2, axis=-1)
    expected_power1 = laser.power * (1.0 + from_db(relative_noise))

    average_power2 = tf.math.reduce_mean(tf.abs(signal_with_absolute_noise.samples) ** 2, axis=-1)
    expected_power2 = laser.power + dbm_to_w(absolute_noise)

    error1 = tf.abs(average_power1 - expected_power1) / expected_power1
    error2 = tf.abs(average_power2 - expected_power2) / expected_power2

    assert tf.math.reduce_all(error1 < 0.01)  # type: ignore
    assert tf.math.reduce_all(error2 < 0.01)  # type: ignore
