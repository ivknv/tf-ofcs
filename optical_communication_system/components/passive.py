# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .datatypes import OpticalSignal, optical_signal_spec
from . import constants
from .utils import fftfreq

__all__ = ["OpticalPhaseShifter"]


class OpticalPhaseShifter(tf.Module):
    phase_shift: tf.Variable  # shape=(1,); dtype=float32

    def __init__(self, *, phase_shift: float, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.phase_shift = tf.Variable(tf.constant(phase_shift))

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, input_signal: OpticalSignal) -> OpticalSignal:
        shifted_samples = input_signal.samples * tf.math.exp(tf.complex(0.0, self.phase_shift))

        return OpticalSignal(
            timing=input_signal.timing,
            frequencies=input_signal.frequencies,
            samples=shifted_samples,
            central_frequency=input_signal.central_frequency,
            power=input_signal.power
        )


class OpticalLoss(tf.Module):
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
            frequencies=signal.frequencies,
            samples=samples,
            central_frequency=signal.central_frequency,
            power=signal.power / power_loss_coeff
        )


class ChromaticDispersion(tf.Module):
    #: Chromatic dispersion coefficient (ps / (nm * km))
    coefficient: tf.Variable  # shape=(1,); dtype=float32
    distance: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        *,
        coefficient: float,
        distance: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.coefficient = tf.Variable(tf.constant(coefficient))
        self.distance = tf.Variable(tf.constant(distance))

    @tf.function(input_signature=(optical_signal_spec,))
    def _apply_pulse_dispersion(self, signal: OpticalSignal) -> OpticalSignal:
        wavelengths = constants.c / signal.frequencies # m

        # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
        D = self.coefficient * 1e-6  # s / m**2

        beta_2 = wavelengths ** 2 / (-2.0 * constants.pi * constants.c) * D

        fs = signal.timing.sampling_rate

        samples = signal.samples

        fft = tf.signal.fft(tf.transpose(samples))  # shape=(F, T)
        freqs = 2 * constants.pi * tf.abs(fftfreq(tf.shape(fft)[1], 1.0 / fs))  # shape=(T,)

        phase_shifts = tf.reshape(
            tf.repeat(beta_2, tf.size(freqs)),
            [tf.size(beta_2), tf.size(freqs)]
        ) * 0.5 * self.distance * 1e3 * freqs ** 2

        new_signal = tf.transpose(
            tf.signal.ifft(
                tf.multiply(
                    fft,
                    tf.math.exp(1j * tf.cast(phase_shifts, tf.complex64))  # type: ignore
                )
            )
        )

        return OpticalSignal(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=new_signal,
            central_frequency=signal.central_frequency,
            power=signal.power
        )

    @tf.function(input_signature=(optical_signal_spec,))
    def _apply_linewidth_dispersion(self, signal: OpticalSignal) -> OpticalSignal:
        wavelength_deltas = (constants.c * 1e9 / signal.frequencies) - (constants.c * 1e9 / signal.central_frequency)

        amount = self.coefficient * self.distance

        # Each wavelength (or frequency) will be shifted in time by some number
        # of samples
        timeshifts = tf.cast(
            amount * 1e-12 * signal.timing.sampling_rate * wavelength_deltas,
            tf.int32
        )

        min_shift = tf.reduce_min(timeshifts)
        max_shift = tf.reduce_max(timeshifts)

        # We have to pad samples from both sides
        left_pad = tf.math.maximum(max_shift, 0)
        right_pad = tf.math.maximum(-min_shift, 0)

        padded_samples = tf.concat([
            tf.zeros(shape=[left_pad, tf.shape(signal.samples)[1]], dtype=tf.complex64),
            signal.samples,
            tf.zeros(shape=[right_pad, tf.shape(signal.samples)[1]], dtype=tf.complex64)
        ], 0)

        new_signal = tf.map_fn(
            lambda i: tf.map_fn(
                lambda j: padded_samples[left_pad + i - timeshifts[j], j],
                tf.range(tf.shape(signal.samples)[1]),
                fn_output_signature=tf.complex64
            ),
            tf.range(tf.shape(signal.samples)[0]),
            fn_output_signature=tf.complex64
        )

        return OpticalSignal(
            timing=signal.timing,
            frequencies=signal.frequencies,
            samples=new_signal,
            central_frequency=signal.central_frequency,
            power=signal.power
        )

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        if tf.abs(self.coefficient) <= 1e-6:  # type: ignore
            return signal

        signal = self._apply_pulse_dispersion(signal)
        signal = self._apply_linewidth_dispersion(signal)

        return signal


class OpticalFiber(tf.Module):
    def __init__(
        self,
        attenuation: float,
        chromatic_dispersion: float,
        length: float,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self._optical_loss = OpticalLoss(loss=attenuation * length)
        self._chromatic_dispersion = ChromaticDispersion(coefficient=chromatic_dispersion, distance=length)

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, signal: OpticalSignal) -> OpticalSignal:
        return self._chromatic_dispersion(self._optical_loss(signal))
