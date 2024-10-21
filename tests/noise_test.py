# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import OpticalAWGN, Laser, TimingGenerator
from optical_communication_system.components.datatypes import OpticalSignal
from optical_communication_system.utils import from_db, dbm_to_w


def test_optical_awgn() -> None:
    rng = tf.random.Generator.from_seed(1234, alg="philox")
    awgn = OpticalAWGN(rng=rng)

    parameter_dims = tf.constant(3)

    laser = Laser(power=dbm_to_w(0.0))
    timing_generator = TimingGenerator(sampling_rate=20.0, symbol_rate=1.0)

    timing = timing_generator(tf.constant(1000))
    input_signal = laser(timing, batch_size=2, parameter_dims=parameter_dims)

    assert tf.math.reduce_all(tf.shape(input_signal.samples) == (2, parameter_dims, 20000))  # type: ignore

    relative_noise = tf.constant([10.0, 0.0, -5.0])  # in dB
    absolute_noise = tf.constant([1.0, 5.0, -5.0])  # in dBm

    signal_with_relative_noise = awgn(
        input_signal,
        optical_awgn__relative_db=relative_noise,
        parameter_dims=parameter_dims
    )
    signal_with_absolute_noise = awgn(
        input_signal,
        optical_awgn__absolute_dbm=absolute_noise,
        parameter_dims=parameter_dims
    )

    assert isinstance(signal_with_relative_noise, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(signal_with_relative_noise.samples) == tf.shape(input_signal.samples)
    )  # type: ignore

    average_power1 = tf.math.reduce_mean(tf.abs(signal_with_relative_noise.samples) ** 2, axis=[0, -1])
    expected_power1 = laser.power * (1.0 + from_db(relative_noise))

    average_power2 = tf.math.reduce_mean(tf.abs(signal_with_absolute_noise.samples) ** 2, axis=[0, -1])
    expected_power2 = laser.power + dbm_to_w(absolute_noise)

    error1 = tf.abs(average_power1 - expected_power1) / expected_power1
    error2 = tf.abs(average_power2 - expected_power2) / expected_power2

    assert tf.math.reduce_all(error1 < 0.01)  # type: ignore
    assert tf.math.reduce_all(error2 < 0.01)  # type: ignore
