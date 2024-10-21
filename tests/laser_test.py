# -*- coding: utf-8 -*-

import tensorflow as tf

from optical_communication_system.components import Laser
from optical_communication_system.components.datatypes import Timing, OpticalSignal


def test_laser() -> None:
    batch_size = tf.constant(4)
    laser = Laser(power=tf.constant([2e-3, 1e-3]), name="test_laser")

    timing = Timing(
        sampling_rate=tf.constant(1.0),
        symbol_rate=tf.constant(1.0),
        duration=tf.constant(16.0),
        sample_count=tf.constant(16)
    )

    laser_output = laser(timing, parameter_dims=tf.constant(2), batch_size=batch_size)

    assert isinstance(laser_output, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(laser_output.samples) == [batch_size, 2, timing.sample_count]
    )  # type: ignore

    signal_power = tf.math.reduce_mean(tf.abs(laser_output.samples) ** 2, axis=[0, 2])

    assert tf.math.reduce_all(tf.less(tf.abs(signal_power - laser.power), 1e-6))  # type: ignore
