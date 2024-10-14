# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from optical_communication_system.components import (
    Laser, MachZehnderModulator, TimingGenerator
)
from optical_communication_system.components.datatypes import ElectricalSignal, OpticalSignal

def test_mzm() -> None:
    laser = Laser(power=1e-3, batch_shape=(1,))
    mzm = MachZehnderModulator(carrier_source=laser, extinction_ratio=float("inf"))

    timing_generator = TimingGenerator(sampling_rate=10.0, symbol_rate=1.0)
    timing = timing_generator(tf.constant(10))

    modulation_signal = ElectricalSignal(
        timing=timing,
        samples=[(tf.math.cos(2 * math.pi * timing.time) + 1.0) * 0.5]
    )

    modulated_signal = mzm(modulation_signal)
    expected_signal = tf.math.cos(math.pi * modulation_signal.samples)

    assert isinstance(modulated_signal, OpticalSignal)
    assert tf.reduce_all(
        tf.less(
            tf.abs(tf.subtract(
                tf.math.real(modulated_signal.samples) / tf.sqrt(laser.power),
                expected_signal
            )),
            1e-9
        )
    )
