# -*- coding: utf-8 -*-

import tensorflow as tf

from optical_communication_system.components import Laser
from optical_communication_system.components.datatypes import Timing, OpticalSignal


def test_laser() -> None:
    laser = Laser(power=2e-3, batch_size=4, name="test_laser")

    timing = Timing(
        sampling_rate=tf.constant(1.0),
        symbol_rate=tf.constant(1.0),
        time=tf.range(0, 16, dtype=tf.float32)
    )

    laser_output = laser(timing)

    assert isinstance(laser_output, OpticalSignal)
    assert tf.reduce_all(tf.shape(laser_output.samples) == [laser.batch_size, tf.size(timing.time)])
    assert tf.reduce_all(
        tf.reduce_mean(tf.abs(laser_output.samples) ** 2, axis=1) == laser.power
    )
