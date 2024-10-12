# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import (
    ChannelLossEstimator, Laser, Photodiode, OpticalFiber
)


def test_channel_loss_estimator() -> None:
    laser = Laser(power=1e-3, batch_size=3)
    photodiode = Photodiode()
    channel = OpticalFiber(
        attenuation=0.22,
        chromatic_dispersion=0.0,
        eff_non_linear_coeff=0.0,
        length=20.0,
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

    expected_loss = tf.math.pow(10.0, 0.1 * channel.attenuation * channel.length)

    probe_duration = 30e-3

    actual_loss = tf.math.reduce_mean(loss_estimator(probe_duration))

    assert tf.abs(actual_loss - expected_loss) < 1e-3  # type: ignore
