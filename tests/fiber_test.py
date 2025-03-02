# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import OpticalFiber, GaussianPulseGenerator
from tf_ofcs.datatypes import OpticalSignal, Timing
from tf_ofcs import constants

import pytest


@pytest.mark.parametrize("enable_nonlinear_effects", [False, True])
def test_optical_fiber_attenuation(enable_nonlinear_effects: bool) -> None:
    fiber = OpticalFiber(
        attenuation=[0.0, 0.22, 0.1, 0.5],
        chromatic_dispersion=[0.0, 0.0, 0.0, 0.0],
        eff_non_linear_coeff=[0.0, 0.0, 0.0, 0.0],
        length=[20.0, 40.0, 10.0, 5.0],
        central_frequency=193.414e12,
        step_count=100,
        enable_nonlinear_effects=enable_nonlinear_effects
    )

    batch_size = tf.size(fiber.attenuation)

    input_signal = OpticalSignal(
        timing=Timing(
            sample_count=tf.constant(10),
            sampling_rate=tf.constant(1.0)
        ),
        samples=tf.cast(
            tf.tile([[0.1, 1.0]], [batch_size, 5]),
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
        axis=-1
    )

    assert tf.math.reduce_all(tf.abs(actual_loss - expected_loss) < 1e-3)  # type: ignore[operator]


@pytest.mark.parametrize("enable_nonlinear_effects", [False, True])
def test_optical_fiber_chromatic_dispersion(enable_nonlinear_effects: bool) -> None:
    fiber = OpticalFiber(
        attenuation=[0.0],
        chromatic_dispersion=[17.0],
        eff_non_linear_coeff=[0.0],
        length=[100.0],
        central_frequency=193.414e12,
        step_count=10,
        enable_nonlinear_effects=enable_nonlinear_effects
    )

    timing = Timing(sampling_rate=1e12, sample_count=1000)
    pulse_generator = GaussianPulseGenerator(
        fwhm_duration=[1 / 50e9],
        power=[1e-3],
        batch_size=1
    )

    pulse = pulse_generator(timing)
    pulse_after_fiber = fiber(pulse)

    def estimate_max_power(signal: OpticalSignal) -> tf.Tensor:
        return tf.math.reduce_max(tf.abs(signal.samples) ** 2, axis=-1)

    def estimate_pulse_duration(signal: OpticalSignal) -> tf.Tensor:
        max_power = estimate_max_power(signal)[0]
        indices = tf.where(tf.abs(signal.samples[0]) ** 2 >= 0.5 * max_power)

        return tf.cast(indices[-1] - indices[0], tf.float32) / tf.cast(timing.sampling_rate, tf.float32)

    # expected to be 3.3e-11
    pulse_duration = estimate_pulse_duration(pulse)

    # expected to be 1.83e-10
    pulse_duration_with_dispersion = estimate_pulse_duration(pulse_after_fiber)

    # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
    D = fiber.chromatic_dispersion * 1e-6  # s / m**2

    wavelength = constants.c / fiber.central_frequency
    beta_2 = wavelength ** 2 / (-2.0 * constants.pi * constants.c) * D

    # Computed value should be around 1.8217361e-10
    # See https://www.rp-photonics.com/chromatic_dispersion.html for the
    # original formula for estimating the duration
    expected_duration = tf.abs(
        4.0 * tf.math.log(2.0) * beta_2 * fiber.length * 1000.0 / pulse_duration
    )

    error = tf.abs(pulse_duration_with_dispersion - expected_duration) / expected_duration

    assert tf.math.reduce_all(pulse_duration_with_dispersion > pulse_duration)  # type: ignore[operator]
    assert tf.math.reduce_all(error < 0.005)  # type: ignore[operator]
