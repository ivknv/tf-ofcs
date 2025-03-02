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

from ..utils import to_tensor
from ..datatypes import ElectricalSignal, OpticalSignal, TensorCompatible
from .component import Component

__all__ = ["ElectricalAWGN", "OpticalAWGN"]


class ElectricalAWGN(Component):
    """
        Adds additive white gaussian noise to an electrical signal.

        :ivar absolute_db: `shape=[None], dtype=tf.float32`, amount of absolute noise in dB
        :ivar relative_db: `shape=[None], dtype=tf.float32`, relative amount of noise in dB
        :ivar rng: `tf.random.Generator`, random number generator
    """

    #: Amount of absolute noise in dB that is independent of input signal
    absolute_db: tf.Variable  # shape=[None]; dtype=float32

    #: Relative amount of noise in dB
    relative_db: tf.Variable  # shape=[None]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        absolute_db: TensorCompatible = tf.constant([-float("inf")]),
        relative_db: TensorCompatible = tf.constant([-float("inf")]),
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.absolute_db = tf.Variable(to_tensor(absolute_db), name=f"{self.name}__absolute_db")
        self.relative_db = tf.Variable(to_tensor(relative_db), name=f"{self.name}__relative_db")
        self.local_parameters = ["absolute_db", "relative_db"]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    def implementation(
        self,
        input_signal: ElectricalSignal,
        /,
        absolute_db: tf.Tensor,
        relative_db: tf.Tensor
    ) -> ElectricalSignal:
        absolute_noise_amplitude = tf.constant(10.0) ** (absolute_db * 0.1 * 0.5)
        relative_noise_amplitude = tf.constant(10.0) ** (relative_db * 0.1 * 0.5)

        noise_shape = tf.shape(input_signal.samples)

        absolute_noise_amplitude = tf.expand_dims(absolute_noise_amplitude, -1)
        relative_noise_amplitude = tf.expand_dims(relative_noise_amplitude, -1)

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
    """
        Adds additive white gaussian noise to an optical signal.

        :ivar absolute_dbm: `shape=[None], dtype=tf.float32`, amount of absolute noise in dBm
        :ivar relative_db: `shape=[None], dtype=tf.float32`, relative amount of noise in dB
        :ivar rng: `tf.random.Generator`, random number generator
    """

    #: Amount of absolute noise in dBm that is independent of input signal
    absolute_dbm: tf.Variable  # shape=[None]; dtype=float32

    #: Relative amount of noise in dB
    relative_db: tf.Variable  # shape=[None]; dtype=float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        *,
        absolute_dbm: TensorCompatible = tf.constant([-float("inf")]),
        relative_db: TensorCompatible = tf.constant([-float("inf")]),
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.absolute_dbm = tf.Variable(to_tensor(absolute_dbm), name=f"{self.name}__absolute_dbm")
        self.relative_db = tf.Variable(to_tensor(relative_db), name=f"{self.name}__relative_db")
        self.local_parameters = ["absolute_dbm", "relative_db"]

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

    def implementation(
        self,
        input_signal: OpticalSignal,
        /,
        absolute_dbm: tf.Tensor,
        relative_db: tf.Tensor
    ) -> OpticalSignal:
        # * 0.5 to account for two noise components
        absolute_noise_amplitude = (0.5 * 1e-3) ** 0.5 * tf.constant(10.0) ** (absolute_dbm * 0.1 * 0.5)
        relative_noise_amplitude = 0.5 ** 0.5 * tf.constant(10.0) ** (relative_db * 0.1 * 0.5)

        # We have to generate the double amount of noise samples since they must be complex
        noise_shape = tf.concat([[2], tf.shape(input_signal.samples)], 0)

        absolute_noise_amplitude = tf.expand_dims(tf.expand_dims(absolute_noise_amplitude, 0), -1)
        relative_noise_amplitude = tf.expand_dims(tf.expand_dims(relative_noise_amplitude, 0), -1)

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

        absolute_noise = tf.complex(absolute_noise_components[0], absolute_noise_components[1])
        relative_noise = tf.complex(relative_noise_components[0], relative_noise_components[1])

        return OpticalSignal(
            timing=input_signal.timing,
            samples=(
                absolute_noise +
                input_signal.samples +
                tf.complex(tf.abs(input_signal.samples), 0.0) * relative_noise
            )
        )
