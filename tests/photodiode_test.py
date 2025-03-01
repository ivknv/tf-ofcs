# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import Photodiode
from tf_ofcs.datatypes import (
    OpticalSignal, ElectricalSignal, Timing
)
from tf_ofcs.utils import timing_to_samples
from tf_ofcs.utils import dbm_to_w
from tf_ofcs import constants


def test_photodiode() -> None:
    photodiode = Photodiode(
        sensitivity=dbm_to_w(tf.constant([-10.0, -1.0, -10.0])),
        maximum_power=dbm_to_w(tf.constant([0.0, 0.0, -2.0]))
    )

    timing = Timing(sampling_rate=1.0, sample_count=10)

    power = dbm_to_w(-3.0)

    input_signal = OpticalSignal(
        timing=timing,
        samples=tf.cast(
            tf.tile(
                [tf.sqrt(power) * tf.math.cos(2 * constants.pi * timing_to_samples(timing))],
                [3, 1]
            ),
            tf.complex64
        )
    )

    received_signal = photodiode(input_signal)
    expected_signal_samples = tf.stack([
        tf.abs(input_signal.samples[0]) ** 2,
        tf.clip_by_value(
            tf.abs(input_signal.samples[1]) ** 2,
            photodiode.sensitivity[1],
            photodiode.maximum_power[1]
        ),
        tf.clip_by_value(
            tf.abs(input_signal.samples[2]) ** 2,
            photodiode.sensitivity[2],
            photodiode.maximum_power[2]
        )
    ])

    assert isinstance(received_signal, ElectricalSignal)
    assert tf.math.reduce_all(
        tf.shape(received_signal.samples) == tf.shape(expected_signal_samples)
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(received_signal.samples - expected_signal_samples) < 1e-5
    )  # type: ignore
