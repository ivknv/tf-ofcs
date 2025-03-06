# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs import constants
from tf_ofcs.components import Laser
from tf_ofcs.datatypes import Timing, OpticalSignal
from tf_ofcs.utils import timing_to_samples

import pytest

@pytest.mark.parametrize("enable_phase_noise", [True, False])
def test_laser(enable_phase_noise: bool) -> None:
    batch_size = tf.constant(2)
    laser = Laser(power=tf.constant([2e-3, 1e-3]), enable_phase_noise=True, name="test_laser")

    timing = Timing(sampling_rate=tf.constant(1.0), sample_count=tf.constant(16))

    laser_output = laser(timing, batch_size=batch_size)

    assert isinstance(laser_output, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(laser_output.samples) == [batch_size, timing.sample_count]
    )  # type: ignore[operator]

    signal_power = tf.math.reduce_mean(tf.abs(laser_output.samples) ** 2, axis=-1)

    assert tf.math.reduce_all(tf.less(tf.abs(signal_power - laser.power), 1e-6))  # type: ignore[operator]


def test_laser_with_frequency_param() -> None:
    laser = Laser(power=[1e-3], frequency=[-10e9, 0.0, 15e9], batch_size=3)

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
        tf.shape(laser_output.samples) == [3, timing.sample_count]
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(laser_output.samples[0] - expected_signal1) < 1e-6
    )  # type: ignore[operator]

    assert tf.math.reduce_all(
        tf.abs(laser_output.samples[1] - expected_signal2) < 1e-6
    )  # type: ignore[operator]

    assert tf.math.reduce_all(
        tf.abs(laser_output.samples[2] - expected_signal3) < 1e-6
    )  # type: ignore[operator]
