# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from tf_ofcs.components import Laser, MachZehnderModulator
from tf_ofcs.datatypes import ElectricalSignal, OpticalSignal, Timing
from tf_ofcs.utils import timing_to_samples

def test_mzm() -> None:
    laser = Laser(power=[1e-3, 2e-3], batch_size=2)
    mzm = MachZehnderModulator()

    timing = Timing(sampling_rate=10.0, sample_count=100)

    carrier_signal = laser(timing)
    modulation_signal = ElectricalSignal(
        timing=timing,
        samples=[
            (tf.math.cos(2 * math.pi * timing_to_samples(timing)) + 1.0) * 0.5
        ] * 2
    )

    modulated_signal = mzm((carrier_signal, modulation_signal))
    expected_signal: tf.Tensor = tf.math.cos(math.pi * modulation_signal.samples)

    assert isinstance(modulated_signal, OpticalSignal)
    assert tf.math.reduce_all(tf.shape(modulated_signal.samples) == tf.shape(expected_signal))  # type: ignore

    error = tf.math.abs(
        tf.subtract(
            tf.math.real(modulated_signal.samples) / tf.sqrt(tf.expand_dims(laser.power, -1)),
            expected_signal
        )
    )

    error /= tf.math.abs(expected_signal)  # type: ignore[operator]

    assert tf.reduce_all(tf.less(error, 0.0001))
