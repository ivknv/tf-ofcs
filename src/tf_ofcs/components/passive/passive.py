# -*- coding: utf-8 -*-
# Copyright (C) 2024 Ivan Konovalov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Optional
import tensorflow as tf

from ...datatypes import OpticalSignal, TensorCompatible
from ... import constants
from ...utils import fftfreq, to_tensor
from ..component import Component

__all__ = [
    "OpticalPhaseShifter",
    "OpticalCoupler1x2",
    "OpticalCoupler2x2",
    "OpticalCoupler2x1"
]


class OpticalPhaseShifter(Component):
    """
        Phase-shifts an optical signal.

        :ivar phase_shift: `shape=[None], dtype=tf.float32`, phase shift in radians
    """

    phase_shift: tf.Variable  # shape=[None]; dtype=float32

    def __init__(
        self,
        phase_shift: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.phase_shift = tf.Variable(to_tensor(phase_shift), name=f"{self.name}__phase_shift")
        self.local_parameters = ["phase_shift"]

    def implementation(
        self,
        input_signal: OpticalSignal,
        /,
        phase_shift: tf.Tensor
    ) -> OpticalSignal:
        phase_shift = tf.expand_dims(phase_shift, -1)
        shifted_samples = input_signal.samples * tf.math.exp(tf.complex(0.0, phase_shift))

        return OpticalSignal(
            timing=input_signal.timing,
            samples=shifted_samples
        )


class OpticalFiber(Component):
    """
        Simulates signal propagation through an optical fiber.
        Models effects of attenuation, chromatic dispersion and the Kerr effect.

        :ivar attenuation: `shape=[None], dtype=tf.float32`, optical signal power loss in dB/km

        :ivar chromatic_dispersion: `shape[None], dtype=tf.float32`, chromatic
                                    dispersion coefficient (ps / (nm * km))

        :ivar eff_non_linear_coeff: `shape[None], dtype=tf.float32`, effective
                                    non-linear coefficient of the fiber (rad / (W * m))

        :ivar length: `shape[None], dtype=tf.float32`, fiber length in kilometers

        :ivar central_frequency: `shape=[], dtype=tf.float32`, central
                                 frequency at which light travels at normal speed (without delay)

        :ivar step_count: `shape=[], dtype=tf.int32`, step count used for the split-step method

        :ivar enable_nonlinear_effects: `bool`, if true, enables symmetrized
                                        split-step method and nonlinear effects
    """

    #: Optical signal power loss in dB/km
    attenuation: tf.Variable  # shape=[None]; dtype=float32

    #: Chromatic dispersion coefficient (ps / (nm * km))
    chromatic_dispersion: tf.Variable  # shape=[None]; dtype=float32

    #: Effective non-linear coefficient of the fiber (rad / (W * m))
    eff_non_linear_coeff: tf.Variable  # shape=[None]; dtype=float32

    #: Fiber length in kilometers
    length: tf.Variable  # shape=[None]; dtype=float32

    #: Central frequency at which light travels at normal speed (without delay)
    central_frequency: tf.Variable  # shape=[]; dtype=float32

    #: Step count used for the split-step method
    step_count: tf.Variable  # shape=[]; dtype=int32

    #: If true, enables symmetrized split-step method and nonlinear effects
    enable_nonlinear_effects: bool

    def __init__(
        self,
        attenuation: TensorCompatible,
        chromatic_dispersion: TensorCompatible,
        eff_non_linear_coeff: TensorCompatible,
        length: TensorCompatible,
        central_frequency: TensorCompatible,
        step_count: int | tf.Tensor = 50,
        enable_nonlinear_effects: bool = True,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.attenuation = tf.Variable(to_tensor(attenuation), name=f"{self.name}__attenuation")
        self.chromatic_dispersion = tf.Variable(
            tf.constant(chromatic_dispersion),
            name=f"{self.name}__chromatic_dispersion"
        )
        self.eff_non_linear_coeff = tf.Variable(
            to_tensor(eff_non_linear_coeff),
            name=f"{self.name}__eff_non_linear_coeff"
        )
        self.length = tf.Variable(to_tensor(length), name=f"{self.name}__length")
        self.central_frequency = tf.Variable(
            to_tensor(central_frequency),
            name=f"{self.name}__central_frequency"
        )
        self.step_count = tf.Variable(to_tensor(step_count), name=f"{self.name}__step_count")
        self.enable_nonlinear_effects = enable_nonlinear_effects

        self.local_parameters = [
            "attenuation",
            "chromatic_dispersion",
            "eff_non_linear_coeff",
            "length",
            "central_frequency",
            "step_count"
        ]

    def implementation(
        self,
        signal: OpticalSignal,
        /,
        attenuation: tf.Tensor,
        chromatic_dispersion: tf.Tensor,
        eff_non_linear_coeff: tf.Tensor,
        length: tf.Tensor,
        central_frequency: tf.Tensor,
        step_count: tf.Tensor
    ) -> OpticalSignal:
        # shape=[None, 1]
        attenuation = tf.expand_dims(attenuation, -1)

        # shape=[None, 1]
        chromatic_dispersion = tf.expand_dims(chromatic_dispersion, -1)

        # shape=[None, 1]
        eff_non_linear_coeff = tf.expand_dims(eff_non_linear_coeff, -1)

        # shape=[None, 1]
        length = tf.expand_dims(length, -1)

        # Convert the dispersion coefficient from ps / (nm * km) to s / (m * m)
        D = chromatic_dispersion * 1e-6  # s / m**2; shape=[None, 1]

        wavelength = constants.c / central_frequency
        beta_2 = wavelength ** 2 / (-2.0 * constants.pi * constants.c) * D  # shape=[None, 1]

        fs = signal.timing.sampling_rate

        samples = signal.samples

        freqs = 2 * constants.pi * fftfreq(tf.shape(samples)[-1], 1.0 / fs)  # shape=[sample_count]

        phase_shifts_per_km = beta_2 * 0.5 * 1e3 * freqs ** 2  # shape=[None, sample_count]

        fft = tf.signal.fft(samples)  # shape=[batch_size, sample_count]

        if self.enable_nonlinear_effects:
            # Step length in kilometers
            step_length = length / tf.cast(step_count, tf.float32)  # shape=[None, 1]
            half_step_length = step_length * 0.5  # shape=[None, 1]

            # Phase shifts per step due to chromatic dispersion
            phase_shifts = phase_shifts_per_km * half_step_length

            # Loss of signal power per step
            loss = tf.math.pow(10.0, attenuation * half_step_length * 0.1 * 0.5)  # shape=[None, 1]

            gamma_per_step = eff_non_linear_coeff * step_length * 1e-3  # shape=[None, 1]

            frequency_domain_step = tf.multiply(
                tf.complex(1.0 / loss, 0.0),
                tf.math.exp(tf.complex(0.0, phase_shifts))
            )  # shape=[None, sample_count]

            # Apply the symmetrized split-step Fourier method for solving the nonlinear Schrödinger equation
            for _ in tf.range(step_count):
                fft *= frequency_domain_step

                time_signal = tf.signal.ifft(fft)
                time_signal *= tf.math.exp(
                    tf.complex(0.0, gamma_per_step * tf.abs(time_signal) ** 2)  # type: ignore[operator]
                )

                fft = tf.signal.fft(time_signal)
                fft *= frequency_domain_step
        else:
            # Apply only attenuation and chromatic dispersion
            loss = tf.math.pow(10.0, attenuation * length * 0.1 * 0.5)  # shape=[None, 1]

            fft *= tf.multiply(
                tf.complex(1.0 / loss, 0.0),
                tf.math.exp(tf.complex(0.0, phase_shifts_per_km * length))
            )

        return OpticalSignal(timing=signal.timing, samples=tf.signal.ifft(fft))


class OpticalCoupler1x2(Component):
    """
        Splits an optical signal into two equal parts.
        Returns a tuple containing the two `OpticalSignal` instances.
    """

    def implementation(self, input_signal: OpticalSignal, /) -> tuple[OpticalSignal, OpticalSignal]:
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
    """
        Mixes two optical signals.
        Takes and returns `tuple[OpticalSignal, OpticalSignal]`.

        :ivar flip: `shape=[], dtype=tf.bool`, if true, the second output will
                    contain the difference of the two input signals
    """

    #: If true, the second output will contain the difference of the two input signals
    flip: tf.Variable  # shape=[]; dtype=bool

    def __init__(self, flip: TensorCompatible = False, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.flip = tf.Variable(to_tensor(flip), name=f"{self.name}__flip")
        self.local_parameters = ["flip"]

    def implementation(
        self,
        input_signals: tuple[OpticalSignal, OpticalSignal],
        /,
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
    """
        Combines two optical signals into one.
        Takes `tuple[OpticalSignal, OpticalSignal]` as input and returns a single
        `OpticalSignal`.
    """

    def implementation(self, input_signals: tuple[OpticalSignal, OpticalSignal], /) -> OpticalSignal:
        mixed_samples = (input_signals[0].samples + input_signals[1].samples) * 0.5 ** 0.5

        timing = input_signals[0].timing

        return OpticalSignal(
            timing=timing,
            samples=mixed_samples
        )
