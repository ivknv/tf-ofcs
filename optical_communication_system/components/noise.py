# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import typing
import tensorflow as tf

from .datatypes import ElectricalSignal, electrical_signal_spec, OpticalSignal, optical_signal_spec
from .component import Component

__all__ = ["ElectricalAWGN", "OpticalAWGN"]


class ElectricalAWGN(Component):
    #: Fixed amount of noise in dB that is independent of input signal
    fixed_db: tf.Variable  # shape=[]; dtype=float32

    #: Relative amount of noise in dB
    relative_db: tf.Variable  # shape=[]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        fixed_db: float | Sequence[float] | tf.Tensor = -float("inf"),
        relative_db: float | Sequence[float] | tf.Tensor = -float("inf"),
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.fixed_db = tf.Variable(tf.constant(fixed_db))
        self.relative_db = tf.Variable(tf.constant(relative_db))
        self.parameters = ["fixed_db", "relative_db"]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32, name="fixed_db"),
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32, name="relative_db")
    ))
    def implementation(
        self,
        input_signal: ElectricalSignal,
        fixed_db: tf.Tensor,
        relative_db: tf.Tensor
    ) -> ElectricalSignal:
        fixed_noise_amplitude = tf.constant(10.0) ** (fixed_db * 0.1 * 0.5)
        relative_noise_amplitude = tf.constant(10.0) ** (relative_db * 0.1 * 0.5)

        fixed_noise = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=fixed_noise_amplitude,
            dtype=tf.float32
        )

        relative_noise = self.rng.normal(
            shape=tf.shape(input_signal.samples),
            mean=0.0,
            stddev=relative_noise_amplitude,
            dtype=tf.float32
        )

        return ElectricalSignal(
            timing=input_signal.timing,
            samples=fixed_noise + input_signal.samples + tf.abs(input_signal.samples) * relative_noise
        )


class OpticalAWGN(Component):
    #: Fixed amount of noise in dBm that is independent of input signal
    fixed_dbm: tf.Variable  # shape=[]; dtype=float32

    #: Relative amount of noise in dB
    relative_db: tf.Variable  # shape=[]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        fixed_dbm: float | Sequence[float] | tf.Tensor = -float("inf"),
        relative_db: float | Sequence[float] | tf.Tensor = -float("inf"),
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.fixed_dbm = tf.Variable(tf.constant(fixed_dbm))
        self.relative_db = tf.Variable(tf.constant(relative_db))
        self.parameters = ["fixed_dbm", "relative_db"]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(
        optical_signal_spec,
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32, name="fixed_db"),
        tf.TensorSpec(typing.cast(tf.TensorShape, None), tf.float32, name="relative_db")
    ))
    def implementation(
        self,
        input_signal: OpticalSignal,
        fixed_dbm: tf.Tensor,
        relative_db: tf.Tensor
    ) -> OpticalSignal:
        # * 0.5 to account for two noise components
        fixed_noise_amplitude = (0.5 * 1e-3) ** 0.5 * tf.constant(10.0) ** (fixed_dbm * 0.1 * 0.5)
        relative_noise_amplitude = 0.5 ** 0.5 * tf.constant(10.0) ** (relative_db * 0.1 * 0.5)

        # We have to generate the double amount of noise samples since they must be complex
        fixed_noise_components = self.rng.normal(
            shape=tf.concat([[2], tf.shape(input_signal.samples)], 0),
            mean=0.0,
            stddev=fixed_noise_amplitude,
            dtype=tf.float32
        )

        relative_noise_components = self.rng.normal(
            shape=tf.concat([[2], tf.shape(input_signal.samples)], 0),
            mean=0.0,
            stddev=relative_noise_amplitude,
            dtype=tf.float32
        )

        fixed_noise = tf.dtypes.complex(fixed_noise_components[0], fixed_noise_components[1])
        relative_noise = tf.dtypes.complex(relative_noise_components[0], relative_noise_components[1])

        return OpticalSignal(
            timing=input_signal.timing,
            samples=(
                fixed_noise +
                input_signal.samples +
                tf.cast(tf.abs(input_signal.samples), tf.complex64) * relative_noise
            )
        )
