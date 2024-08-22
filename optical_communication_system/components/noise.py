# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .datatypes import (
    ElectricalSignal, electrical_signal_spec,
    OpticalSignal, optical_signal_spec
)

__all__ = ["ElectricalAWGN", "OpticalAWGN"]


class ElectricalAWGN(tf.Module):
    #: Noise level in dB relative to the input signal
    level: tf.Variable  # shape=(1,); dtype=float32

    #: Fixed amount of noise in watts that is independent of input signal
    fixed_power: tf.Variable  # shape=(1,); dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        level: float,
        fixed_power: float = 0.0,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.level = tf.Variable(tf.constant(level))
        self.fixed_power = tf.Variable(tf.constant(fixed_power))

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, input_signal: ElectricalSignal) -> ElectricalSignal:
        noise_amplitude = (tf.add(
            self.fixed_power,
            input_signal.power * tf.math.pow(10.0, self.level * 0.1)
        ) * 0.5) ** 0.5  # * 0.5 to account for two noise components

        noise = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        return ElectricalSignal(
            timing=input_signal.timing,
            samples=tf.add(noise, input_signal.samples)
        )


class OpticalAWGN(tf.Module):
    #: Noise level in dB relative to the input signal
    level: tf.Variable  # shape=(1,); dtype=float32

    #: Fixed amount of noise in watts that is independent of input signal
    fixed_power: tf.Variable  # shape=(1,); dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        level: float,
        fixed_power: float = 0.0,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.level = tf.Variable(tf.constant(level))
        self.fixed_power = tf.Variable(tf.constant(fixed_power))

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(optical_signal_spec,))
    def __call__(self, input_signal: OpticalSignal) -> OpticalSignal:
        noise_amplitude = tf.divide(
            tf.add(
                self.fixed_power,
                input_signal.power * tf.math.pow(10.0, self.level * 0.1)
            ) * 0.5,  # * 0.5 to account for two noise components
            tf.cast(tf.size(input_signal.frequencies), tf.float32)
        ) ** 0.5

        noise_real = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise_imag = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise = tf.reshape(
            tf.dtypes.complex(noise_real, noise_imag),
            tf.shape(input_signal.samples)
        )

        return OpticalSignal(
            timing=input_signal.timing,
            samples=tf.add(noise, input_signal.samples),
            frequencies=input_signal.frequencies,
            central_frequency=input_signal.central_frequency,
            power=input_signal.power
        )
