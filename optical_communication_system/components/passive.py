# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .datatypes import OpticalSignal, optical_signal_spec
from . import constants
from .utils import fftfreq
from .component import Component

__all__ = [
    "OpticalPhaseShifter",
    "OpticalCoupler1x2",
    "OpticalCoupler2x2",
    "OpticalCoupler2x1"
]


class OpticalPhaseShifter(Component):
    phase_shift: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, *, phase_shift: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.phase_shift = tf.Variable(tf.constant(phase_shift))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, input_signal: OpticalSignal) -> OpticalSignal:
        shifted_samples = input_signal.samples * tf.math.exp(tf.complex(0.0, self.phase_shift))

        return OpticalSignal(
            timing=input_signal.timing,
            samples=shifted_samples
        )


class OpticalFiber(Component):
    #: Optical signal power loss in dB/km
    attenuation: tf.Variable  # shape=(None,); dtype=float32

    #: Chromatic dispersion coefficient (ps / (nm * km))
    chromatic_dispersion: tf.Variable  # shape=(None,); dtype=float32

    #: Effective non-linear coefficient of the fiber (rad / (W * m))
    eff_non_linear_coeff: tf.Variable  # shape=(None,); dtype=float32

    #: Fiber length in kilometers
    length: tf.Variable  # shape=(None,); dtype=float32

    #: Central frequency at which light travels at normal speed (without delay)
    central_frequency: tf.Variable  # shape=(None,); dtype=float32

    #: Step count used for the split-step method (in meters)
    step_count: tf.Variable  # shape=(None,); dtype=int32

    def __init__(
        self,
        attenuation: float,
        chromatic_dispersion: float,
        eff_non_linear_coeff: float,
        length: float,
        central_frequency: float,
        step_count: int = 50,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.attenuation = tf.Variable(tf.constant(attenuation))
        self.chromatic_dispersion = tf.Variable(tf.constant(chromatic_dispersion))
        self.eff_non_linear_coeff = tf.Variable(tf.constant(eff_non_linear_coeff))
        self.length = tf.Variable(tf.constant(length))
        self.central_frequency = tf.Variable(tf.constant(central_frequency))

        self.step_count = tf.Variable(tf.constant(step_count))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
        D = self.chromatic_dispersion * 1e-6  # s / m**2

        wavelength = constants.c / self.central_frequency
        beta_2 = wavelength ** 2 / (-2.0 * constants.pi * constants.c) * D

        fs = signal.timing.sampling_rate

        samples = signal.samples

        fft = tf.signal.fft(samples)
        freqs = 2 * constants.pi * tf.abs(fftfreq(tf.size(fft), 1.0 / fs))

        # Step length in kilometers
        step_length = self.length / tf.cast(self.step_count, tf.float32)
        half_step_length = step_length * 0.5

        # Phase shifts per step due to chromatic dispersion
        phase_shifts = tf.cast(beta_2 * 0.5 * half_step_length * 1e3 * freqs ** 2, tf.complex64)

        # Loss of signal power per step
        loss = tf.cast(
            tf.math.pow(10.0, self.attenuation * half_step_length * 0.1 * 0.5),
            tf.complex64
        )

        gamma_per_step = tf.cast(self.eff_non_linear_coeff * step_length * 1e-3, tf.complex64)

        dispersion_step = (1.0 / loss) * tf.math.exp(1j * phase_shifts)

        for _ in tf.range(self.step_count):
            fft *= dispersion_step

            time_signal = tf.signal.ifft(fft)
            time_signal *= tf.math.exp(
                1j * gamma_per_step * tf.cast(tf.abs(time_signal) ** 2, tf.complex64)  # type: ignore[operator]
            )

            fft = tf.signal.fft(time_signal)
            fft *= dispersion_step

        return OpticalSignal(timing=signal.timing, samples=tf.signal.ifft(fft))


class OpticalCoupler1x2(Component):
    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, input_signal: OpticalSignal) -> tuple[OpticalSignal, OpticalSignal]:
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
    flip: tf.Variable  # shape=(1,); dtype=bool

    def __init__(self, flip: bool = False, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.flip = tf.Variable(tf.constant(flip))

    @tf.function(input_signature=((optical_signal_spec, optical_signal_spec),))
    def __call__(self, input_signals: tuple[OpticalSignal, OpticalSignal]) -> tuple[OpticalSignal, OpticalSignal]:
        mixed_samples1 = (input_signals[0].samples + input_signals[1].samples) * 0.5 ** 0.5

        if self.flip:  # type: ignore
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
    def __call__(self, input_signals: tuple[OpticalSignal, OpticalSignal]) -> OpticalSignal:
        mixed_samples = (input_signals[0].samples + input_signals[1].samples) * 0.5 ** 0.5

        timing = input_signals[0].timing

        return OpticalSignal(
            timing=timing,
            samples=mixed_samples
        )
