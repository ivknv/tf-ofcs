# -*- coding: utf-8 -*-

import tensorflow as tf

from optical_communication_system import constants
from optical_communication_system.components import Laser
from optical_communication_system.datatypes import Timing, OpticalSignal
from optical_communication_system.utils import timing_to_samples


def test_laser() -> None:
    batch_size = tf.constant(4)
    laser = Laser(power=tf.constant([2e-3, 1e-3]), name="test_laser")

    timing = Timing(sampling_rate=tf.constant(1.0), sample_count=tf.constant(16))

    laser_output = laser(timing, parameter_dims=tf.constant(2), batch_size=batch_size)

    assert isinstance(laser_output, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(laser_output.samples) == [batch_size, 2, timing.sample_count]
    )  # type: ignore[operator]

    signal_power = tf.math.reduce_mean(tf.abs(laser_output.samples) ** 2, axis=[0, 2])

    assert tf.math.reduce_all(tf.less(tf.abs(signal_power - laser.power), 1e-6))  # type: ignore[operator]


def test_laser_with_frequency_param() -> None:
    laser = Laser(power=1e-3, frequency=[-10e9, 0.0, 15e9], parameter_dims=3)

    timing = Timing(sampling_rate=tf.constant(30e9), sample_count=tf.constant(64))

    time = timing_to_samples(timing)

    amplitude = tf.complex(tf.sqrt(1e-3), 0.0)

    expected_signal1 = amplitude * tf.math.exp(
        tf.complex(0.0, -10e9 * 2.0 * constants.pi * time)
    )
    expected_signal2 = amplitude * tf.math.exp(tf.complex(0.0, 0.0 * time))
    expected_signal3 = amplitude * tf.math.exp(
        tf.complex(0.0, 15e9 * 2.0 * constants.pi * time)
    )

    laser_output = laser(timing)

    assert isinstance(laser_output, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(laser_output.samples) == [1, 3, timing.sample_count]
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(laser_output.samples[0, 0] - expected_signal1) < 1e-6
    )  # type: ignore[operator]

    assert tf.math.reduce_all(
        tf.abs(laser_output.samples[0, 1] - expected_signal2) < 1e-6
    )  # type: ignore[operator]

    assert tf.math.reduce_all(
        tf.abs(laser_output.samples[0, 2] - expected_signal3) < 1e-6
    )  # type: ignore[operator]
