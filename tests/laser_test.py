# -*- coding: utf-8 -*-

import tensorflow as tf

from optical_communication_system.components import Laser
from optical_communication_system.components.datatypes import Timing, OpticalSignal


def test_laser() -> None:
    batch_shape = (4, 3)
    laser = Laser(power=tf.constant([2e-3, 1e-3]), batch_shape=batch_shape, name="test_laser")

    timing = Timing(
        sampling_rate=tf.constant(1.0),
        symbol_rate=tf.constant(1.0),
        duration=tf.constant(16.0),
        sample_count=tf.constant(16)
    )

    laser_output = laser(timing)

    assert isinstance(laser_output, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(laser_output.samples) == [*batch_shape, 2, timing.sample_count]
    )  # type: ignore
    assert tf.math.reduce_all(
        tf.less(
            tf.abs(
                tf.reshape(
                    tf.math.reduce_mean(tf.abs(laser_output.samples) ** 2, axis=[0, 1, -1]),
                    tf.shape(laser.power)
                ) - laser.power
            ),
            1e-6
        )
    )  # type: ignore
