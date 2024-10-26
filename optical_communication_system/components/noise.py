# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from ..datatypes import ElectricalSignal, electrical_signal_spec, OpticalSignal, optical_signal_spec
from .component import Component

__all__ = ["ElectricalAWGN", "OpticalAWGN"]


class ElectricalAWGN(Component):
    #: Amount of absolute noise in dB that is independent of input signal
    absolute_db: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Relative amount of noise in dB
    relative_db: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        absolute_db: float | Sequence[float] | tf.Tensor = -float("inf"),
        relative_db: float | Sequence[float] | tf.Tensor = -float("inf"),
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.absolute_db = tf.Variable(tf.constant(absolute_db))
        self.relative_db = tf.Variable(tf.constant(relative_db))
        self.local_parameters = [("absolute_db", 0), ("relative_db", 0)]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="absolute_db"),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="relative_db")
    ))
    def implementation(
        self,
        input_signal: ElectricalSignal,
        absolute_db: tf.Tensor,
        relative_db: tf.Tensor
    ) -> ElectricalSignal:
        absolute_noise_amplitude = tf.constant(10.0) ** (absolute_db * 0.1 * 0.5)
        relative_noise_amplitude = tf.constant(10.0) ** (relative_db * 0.1 * 0.5)
        parameter_dims = tf.shape(input_signal.samples)[1]

        noise_shape = tf.shape(input_signal.samples)
        amplitude_shape = [1, parameter_dims, 1]

        absolute_noise_amplitude = tf.reshape(absolute_noise_amplitude, amplitude_shape)
        relative_noise_amplitude = tf.reshape(relative_noise_amplitude, amplitude_shape)

        absolute_noise = self.rng.normal(
            shape=noise_shape,
            mean=0.0,
            stddev=absolute_noise_amplitude,
            dtype=tf.float32
        )

        relative_noise = self.rng.normal(
            shape=noise_shape,
            mean=0.0,
            stddev=relative_noise_amplitude,
            dtype=tf.float32
        )

        return ElectricalSignal(
            timing=input_signal.timing,
            samples=absolute_noise + input_signal.samples + tf.abs(input_signal.samples) * relative_noise
        )


class OpticalAWGN(Component):
    #: Amount of absolute noise in dBm that is independent of input signal
    absolute_dbm: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Relative amount of noise in dB
    relative_db: tf.Variable  # shape=[] or [None]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        absolute_dbm: float | Sequence[float] | tf.Tensor = -float("inf"),
        relative_db: float | Sequence[float] | tf.Tensor = -float("inf"),
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.absolute_dbm = tf.Variable(tf.constant(absolute_dbm))
        self.relative_db = tf.Variable(tf.constant(relative_db))
        self.local_parameters = [("absolute_dbm", 0), ("relative_db", 0)]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    @tf.function(input_signature=(
        optical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="absolute_dbm"),
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="relative_db")
    ))
    def implementation(
        self,
        input_signal: OpticalSignal,
        absolute_dbm: tf.Tensor,
        relative_db: tf.Tensor
    ) -> OpticalSignal:
        # * 0.5 to account for two noise components
        absolute_noise_amplitude = (0.5 * 1e-3) ** 0.5 * tf.constant(10.0) ** (absolute_dbm * 0.1 * 0.5)
        relative_noise_amplitude = 0.5 ** 0.5 * tf.constant(10.0) ** (relative_db * 0.1 * 0.5)
        parameter_dims = tf.shape(input_signal.samples)[1]

        # We have to generate the double amount of noise samples since they must be complex
        noise_shape = tf.concat([[2], tf.shape(input_signal.samples)], 0)
        amplitude_shape = [1, 1, parameter_dims, 1]

        absolute_noise_amplitude = tf.reshape(absolute_noise_amplitude, amplitude_shape)
        relative_noise_amplitude = tf.reshape(relative_noise_amplitude, amplitude_shape)

        absolute_noise_components = self.rng.normal(
            shape=noise_shape,
            mean=0.0,
            stddev=absolute_noise_amplitude,
            dtype=tf.float32
        )

        relative_noise_components = self.rng.normal(
            shape=noise_shape,
            mean=0.0,
            stddev=relative_noise_amplitude,
            dtype=tf.float32
        )

        absolute_noise = tf.dtypes.complex(absolute_noise_components[0], absolute_noise_components[1])
        relative_noise = tf.dtypes.complex(relative_noise_components[0], relative_noise_components[1])

        return OpticalSignal(
            timing=input_signal.timing,
            samples=(
                absolute_noise +
                input_signal.samples +
                tf.cast(tf.abs(input_signal.samples), tf.complex64) * relative_noise
            )
        )
