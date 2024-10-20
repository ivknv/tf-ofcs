# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from optical_communication_system.components import (
    Laser, MachZehnderModulator, TimingGenerator
)
from optical_communication_system.components.datatypes import ElectricalSignal, OpticalSignal
from optical_communication_system.components.utils import timing_to_samples

def test_mzm() -> None:
    laser = Laser(power=[1e-3, 2e-3], batch_size=1)
    mzm = MachZehnderModulator(carrier_source=laser)

    timing_generator = TimingGenerator(sampling_rate=10.0, symbol_rate=1.0)
    timing = timing_generator(tf.constant(10))

    modulation_signal = ElectricalSignal(
        timing=timing,
        samples=[
            [(tf.math.cos(2 * math.pi * timing_to_samples(timing)) + 1.0) * 0.5] * 2,
        ]
    )

    modulated_signal = mzm(modulation_signal)
    expected_signal = tf.math.cos(math.pi * modulation_signal.samples)

    assert isinstance(modulated_signal, OpticalSignal)
    assert tf.math.reduce_all(tf.shape(modulated_signal.samples) == tf.shape(expected_signal))  # type: ignore
    assert tf.reduce_all(
        tf.less(
            tf.abs(tf.subtract(
                tf.math.real(modulated_signal.samples) / tf.sqrt(tf.expand_dims(laser.power, -1)),
                expected_signal
            )),
            1e-7
        )
    )
