# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import OpticalFiber
from optical_communication_system.components.datatypes import OpticalSignal, Timing


def test_optical_fiber_attenuation() -> None:
    fiber = OpticalFiber(
        attenuation=0.22,
        chromatic_dispersion=0.0,
        eff_non_linear_coeff=0.0,
        length=20.0,
        central_frequency=193.414e12,
        step_count=100
    )

    input_signal = OpticalSignal(
        timing=Timing(
            time=tf.range(0, 10, dtype=tf.float32),
            sampling_rate=1.0,
            symbol_rate=10.0
        ),
        samples=tf.cast(tf.tile([[0.1, 1.0]], [2, 5]), tf.complex64)
    )

    output_signal = fiber(input_signal)
    expected_loss = fiber.attenuation * fiber.length

    actual_loss = tf.math.reduce_mean(
        20.0 / tf.math.log(10.0) * tf.math.log(
            (tf.abs(input_signal.samples) / tf.abs(output_signal.samples))
        )
    )

    assert tf.abs(actual_loss - expected_loss) < 1e-3  # type: ignore
