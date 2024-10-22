# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import OpticalFiber
from optical_communication_system.components.datatypes import OpticalSignal, Timing


def test_optical_fiber_attenuation() -> None:
    fiber = OpticalFiber(
        attenuation=[0.0, 0.22, 0.1, 0.5],
        chromatic_dispersion=[0.0, 0.0, 0.0, 0.0],
        eff_non_linear_coeff=[0.0, 0.0, 0.0, 0.0],
        length=[20.0, 40.0, 10.0, 5.0],
        central_frequency=193.414e12,
        step_count=100
    )

    parameter_dims = tf.size(fiber.attenuation)
    batch_size = tf.constant(2)

    input_signal = OpticalSignal(
        timing=Timing(
            sample_count=tf.constant(10),
            sampling_rate=tf.constant(1.0),
            symbol_rate=tf.constant(10.0)
        ),
        samples=tf.cast(
            tf.tile([[[0.1, 1.0]]], [batch_size, parameter_dims, 5]),
            tf.complex64
        )
    )

    output_signal = fiber(input_signal)

    assert isinstance(output_signal, OpticalSignal)
    assert tf.math.reduce_all(
        tf.shape(output_signal.samples) == tf.shape(input_signal.samples)
    )  # type: ignore

    expected_loss = fiber.attenuation * fiber.length

    actual_loss = tf.math.reduce_mean(
        20.0 / tf.math.log(10.0) * tf.math.log(
            (tf.abs(input_signal.samples) / tf.abs(output_signal.samples))
        ),
        axis=[0, 2]
    )

    assert tf.math.reduce_all(tf.abs(actual_loss - expected_loss) < 1e-3)  # type: ignore
