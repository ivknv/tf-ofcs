# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import (
    ChannelLossEstimator, Laser, Photodiode, OpticalFiber
)
from optical_communication_system.utils import from_db


def test_channel_loss_estimator() -> None:
    laser = Laser(power=[1e-3, 2e-3], batch_size=3)
    photodiode = Photodiode()
    channel = OpticalFiber(
        attenuation=[0.22, 0.3],
        chromatic_dispersion=[0.0, 0.0],
        eff_non_linear_coeff=[0.0, 0.0],
        length=[20.0, 15.0],
        central_frequency=193.414e12,
        step_count=100
    )

    sampling_rate = 1e3

    loss_estimator = ChannelLossEstimator(
        laser=laser,
        channel=channel,
        photodiode=photodiode,
        sampling_rate=sampling_rate
    )

    expected_loss = from_db(channel.attenuation * channel.length)

    probe_duration = 30e-3

    losses = loss_estimator(probe_duration)

    assert isinstance(losses, tf.Tensor)
    assert tf.math.reduce_all(tf.shape(losses) == (3, 2))  # type: ignore

    actual_loss = tf.math.reduce_mean(loss_estimator(probe_duration), axis=[0])

    assert tf.math.reduce_all(tf.abs(actual_loss - expected_loss) < 1e-3)  # type: ignore
