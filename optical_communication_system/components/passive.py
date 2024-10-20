# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import typing
import tensorflow as tf

from .datatypes import OpticalSignal, optical_signal_spec
from .. import constants
from .utils import fftfreq
from .component import Component

__all__ = [
    "OpticalPhaseShifter",
    "OpticalCoupler1x2",
    "OpticalCoupler2x2",
    "OpticalCoupler2x1"
]


class OpticalPhaseShifter(Component):
    phase_shift: tf.Variable  # shape=[None]; dtype=float32

    def __init__(self, *, phase_shift: float | tf.Tensor, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.phase_shift = tf.Variable(tf.constant(phase_shift))
        self.parameters = ["phase_shift"]

    @tf.function(input_signature=(optical_signal_spec, tf.TensorSpec(shape=[], dtype=tf.float32)))
    def implementation(
        self,
        input_signal: OpticalSignal,
        phase_shift: tf.Tensor
    ) -> OpticalSignal:
        shifted_samples = input_signal.samples * tf.math.exp(tf.complex(0.0, phase_shift))

        return OpticalSignal(
            timing=input_signal.timing,
            samples=shifted_samples
        )


class OpticalFiber(Component):
    #: Optical signal power loss in dB/km
    attenuation: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Chromatic dispersion coefficient (ps / (nm * km))
    chromatic_dispersion: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Effective non-linear coefficient of the fiber (rad / (W * m))
    eff_non_linear_coeff: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Fiber length in kilometers
    length: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Central frequency at which light travels at normal speed (without delay)
    central_frequency: tf.Variable  # shape=[]; dtype=float32

    #: Step count used for the split-step method (in meters)
    step_count: tf.Variable  # shape=[]; dtype=int32

    def __init__(
        self,
        attenuation: float | Sequence[float] | tf.Tensor,
        chromatic_dispersion: float | Sequence[float] | tf.Tensor,
        eff_non_linear_coeff: float | Sequence[float] | tf.Tensor,
        length: float | Sequence[float] | tf.Tensor,
        central_frequency: float | tf.Tensor,
        step_count: int | tf.Tensor = 50,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.attenuation = tf.Variable(tf.constant(attenuation))
        self.chromatic_dispersion = tf.Variable(tf.constant(chromatic_dispersion))
        self.eff_non_linear_coeff = tf.Variable(tf.constant(eff_non_linear_coeff))
        self.length = tf.Variable(tf.constant(length))
        self.central_frequency = tf.Variable(tf.constant(central_frequency))
        self.step_count = tf.Variable(tf.constant(step_count))

        self.parameters = [
            "attenuation",
            "chromatic_dispersion",
            "eff_non_linear_coeff",
            "length",
            "central_frequency",
            "step_count"
        ]

    @tf.function(input_signature=(
        optical_signal_spec,
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32),
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32),
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32),
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32),
        tf.TensorSpec([], tf.float32),
        tf.TensorSpec([], tf.int32)
    ))
    def implementation(
        self,
        signal: OpticalSignal,
        attenuation: tf.Tensor,
        chromatic_dispersion: tf.Tensor,
        eff_non_linear_coeff: tf.Tensor,
        length: tf.Tensor,
        central_frequency: tf.Tensor,
        step_count: tf.Tensor
    ) -> OpticalSignal:
        parameter_dims = tf.shape(signal.samples)[1]

        # shape=[parameter_dims, 1]
        attenuation = tf.expand_dims(tf.broadcast_to(attenuation, [parameter_dims]), -1)

        # shape=[parameter_dims, 1]
        chromatic_dispersion = tf.expand_dims(tf.broadcast_to(chromatic_dispersion, [parameter_dims]), -1)

        # shape=[parameter_dims, 1]
        eff_non_linear_coeff = tf.expand_dims(tf.broadcast_to(eff_non_linear_coeff, [parameter_dims]), -1)

        # shape=[parameter_dims, 1]
        length = tf.expand_dims(tf.broadcast_to(length, [parameter_dims]), -1)

        # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
        D = chromatic_dispersion * 1e-6  # s / m**2; shape=[parameter_dims, 1]

        wavelength = constants.c / central_frequency
        beta_2 = wavelength ** 2 / (-2.0 * constants.pi * constants.c) * D  # shape=[parameter_dims, 1]

        fs = signal.timing.sampling_rate

        samples = signal.samples

        fft = tf.signal.fft(samples)  # shape=[batch_size, parameter_dims, sample_count]
        freqs = 2 * constants.pi * tf.abs(fftfreq(tf.shape(fft)[-1], 1.0 / fs))  # shape=[sample_count]

        # Step length in kilometers
        step_length = length / tf.cast(step_count, tf.float32)  # shape=[parameter_dims, 1]
        half_step_length = step_length * 0.5  # shape=[parameter_dims, 1]

        # Phase shifts per step due to chromatic dispersion
        phase_shifts = tf.cast(
            beta_2 * 0.5 * half_step_length * 1e3 * freqs ** 2,
            tf.complex64
        )  # shape=[parameter_dims, sample_count]

        # Loss of signal power per step
        loss = tf.cast(
            tf.math.pow(10.0, attenuation * half_step_length * 0.1 * 0.5),
            tf.complex64
        )  # shape=[parameter_dims, 1]

        gamma_per_step = tf.cast(
            eff_non_linear_coeff * step_length * 1e-3,
            tf.complex64
        )  # shape=[parameter_dims, 1]

        frequency_domain_step = (1.0 / loss) * tf.math.exp(1j * phase_shifts)  # shape=[parameter_dims, sample_count]

        # Apply the symmetrized split-step Fourier method for solving the nonlinear Schrödinger equation
        for _ in tf.range(step_count):
            fft *= frequency_domain_step

            time_signal = tf.signal.ifft(fft)
            time_signal *= tf.math.exp(
                1j * gamma_per_step * tf.cast(tf.abs(time_signal) ** 2, tf.complex64)  # type: ignore[operator]
            )

            fft = tf.signal.fft(time_signal)
            fft *= frequency_domain_step

        return OpticalSignal(timing=signal.timing, samples=tf.signal.ifft(fft))


class OpticalCoupler1x2(Component):
    @tf.function(input_signature=(optical_signal_spec,))
    def implementation(self, input_signal: OpticalSignal) -> tuple[OpticalSignal, OpticalSignal]:
        split_samples = input_signal.samples * 0.5 ** 0.5

        return (
            OpticalSignal(
                timing=input_signal.timing,
                samples=split_samples
            ),
            OpticalSignal(
                timing=input_signal.timing,
                samples=split_samples
            )
        )


class OpticalCoupler2x2(Component):
    #: If true, the second output will contain the difference of the two input signals
    flip: tf.Variable  # shape=[]; dtype=bool

    def __init__(self, flip: bool | tf.Tensor = False, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.flip = tf.Variable(tf.constant(flip))
        self.parameters = ["flip"]

    @tf.function(input_signature=(
        (optical_signal_spec, optical_signal_spec),
        tf.TensorSpec([], tf.bool)
    ))
    def implementation(
        self,
        input_signals: tuple[OpticalSignal, OpticalSignal],
        flip: tf.Tensor
    ) -> tuple[OpticalSignal, OpticalSignal]:
        mixed_samples1 = (input_signals[0].samples + input_signals[1].samples) * 0.5 ** 0.5

        if flip:  # type: ignore
            mixed_samples2 = (input_signals[0].samples - input_signals[1].samples) * 0.5 ** 0.5
        else:
            mixed_samples2 = mixed_samples1

        timing = input_signals[0].timing

        return (
            OpticalSignal(
                timing=timing,
                samples=mixed_samples1
            ),
            OpticalSignal(
                timing=timing,
                samples=mixed_samples2
            )
        )


class OpticalCoupler2x1(Component):
    @tf.function(input_signature=((optical_signal_spec, optical_signal_spec),))
    def implementation(self, input_signals: tuple[OpticalSignal, OpticalSignal]) -> OpticalSignal:
        mixed_samples = (input_signals[0].samples + input_signals[1].samples) * 0.5 ** 0.5

        timing = input_signals[0].timing

        return OpticalSignal(
            timing=timing,
            samples=mixed_samples
        )
