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


class OpticalLoss(Component):
    #: Optical signal power loss in dB
    loss: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, *, loss: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.loss = tf.Variable(tf.constant(loss))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        power_loss_coeff = tf.math.pow(10.0, self.loss * 0.1)
        samples = tf.divide(
            signal.samples,
            tf.cast(tf.sqrt(power_loss_coeff), tf.complex64)
        )

        return OpticalSignal(
            timing=signal.timing,
            samples=samples
        )


class ChromaticDispersion(Component):
    #: Chromatic dispersion coefficient (ps / (nm * km))
    coefficient: tf.Variable  # shape=(1,); dtype=float32
    distance: tf.Variable  # shape=(1,); dtype=float32

    #: Central frequency at which light travels at normal speed (without delay)
    central_frequency: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        *,
        coefficient: float,
        distance: float,
        central_frequency: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.coefficient = tf.Variable(tf.constant(coefficient))
        self.distance = tf.Variable(tf.constant(distance))
        self.central_frequency = tf.Variable(tf.constant(central_frequency))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        if tf.abs(self.coefficient) <= 1e-6:  # type: ignore
            return signal

        # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
        D = self.coefficient * 1e-6  # s / m**2

        wavelength = constants.c / self.central_frequency
        beta_2 = wavelength ** 2 / (-2.0 * constants.pi * constants.c) * D

        fs = signal.timing.sampling_rate

        samples = signal.samples

        fft = tf.signal.fft(samples)
        freqs = 2 * constants.pi * tf.abs(fftfreq(tf.size(fft), 1.0 / fs))

        phase_shifts = beta_2 * 0.5 * self.distance * 1e3 * freqs ** 2

        new_signal = tf.signal.ifft(
            tf.multiply(
                fft,
                tf.math.exp(1j * tf.cast(phase_shifts, tf.complex64))  # type: ignore
            )
        )

        return OpticalSignal(
            timing=signal.timing,
            samples=new_signal
        )


class OpticalFiber(Component):
    def __init__(
        self,
        attenuation: float,
        chromatic_dispersion: float,
        length: float,
        central_frequency: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self._optical_loss = OpticalLoss(loss=attenuation * length)
        self._chromatic_dispersion = ChromaticDispersion(
            coefficient=chromatic_dispersion,
            distance=length,
            central_frequency=central_frequency
        )

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        return self._chromatic_dispersion(self._optical_loss(signal))


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
