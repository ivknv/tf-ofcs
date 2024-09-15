# -*- coding: utf-8 -*-

from typing import Optional
import tensorflow as tf

from .datatypes import ElectricalSignal, electrical_signal_spec, OpticalSignal, optical_signal_spec
from .component import Component

__all__ = ["ElectricalAWGN", "OpticalAWGN"]


class ElectricalAWGN(Component):
    #: Fixed amount of noise in watts that is independent of input signal
    power: tf.Variable  # shape=[]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        power: float,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.parameters = ["power"]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(electrical_signal_spec, tf.TensorSpec([], tf.float32)))
    def implementation(self, input_signal: ElectricalSignal, power: tf.Tensor) -> ElectricalSignal:
        # * 0.5 to account for two noise components
        noise_amplitude = power ** 0.5

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


class OpticalAWGN(Component):
    #: Fixed amount of noise in watts that is independent of input signal
    power: tf.Variable  # shape=[]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        power: float,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(tf.constant(power))
        self.parameters = ["power"]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(optical_signal_spec, tf.TensorSpec([], tf.float32)))
    def implementation(self, input_signal: OpticalSignal, power: tf.Tensor) -> OpticalSignal:
        # * 0.5 to account for two noise components
        noise_amplitude = (power * 0.5) ** 0.5

        # We have to generate the double amount of noise samples since they must be complex
        noise_components = self.rng.normal(
            shape=[2, tf.size(input_signal.samples)],
            mean=0.0,
            stddev=noise_amplitude,
            dtype=tf.float32
        )

        noise = tf.dtypes.complex(noise_components[0], noise_components[1])

        return OpticalSignal(
            timing=input_signal.timing,
            samples=tf.add(noise, input_signal.samples)
        )
