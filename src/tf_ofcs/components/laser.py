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

import tensorflow as tf

from typing import Optional
from ..datatypes import Signal, TensorCompatible, Timing

from .component import Component
from ..utils import timing_to_samples, to_tensor
from .. import constants

__all__ = ["Laser"]


class Laser(Component):
    """
        Models a laser. Takes a `Timing` or `Signal`
        as input. If the input is a signal rather than timing, then the timing
        will be extracted from the signal. Outputs an `OpticalSignal` instance.

        :ivar power: `shape=[None], dtype=tf.float32`, output power of the laser.
        :ivar frequency: `shape=[None], dtype=tf.float32`, laser frequency offset
                         (relative to the central frequency) in Hz
        :ivar linewidth: `shape=[None], dtype=tf.float32`, laser linewidth in Hz
        :ivar initial_phase: `shape=[None], dtype=tf.float32`, initial phase (in radians)
        :ivar batch_size: `shape=[], dtype=tf.int32`, determines the output
                          signal's shape is `(batch_size, tf.size(time))`
        :ivar enable_phase_noise: `bool`, enables/disables phase noise
        :ivar rng: `tf.random.Generator`, random number generator for phase noise.
    """

    #: Output power in watts
    power: tf.Variable  # shape=[None]; dtype=float32

    #: Laser frequency offset (relative to the central frequency) in Hz
    frequency: tf.Variable  # shape=[None]; dtype=tf.float32

    #: Linewidth in Hz
    linewidth: tf.Variable  # shape=[None]; dtype=float32

    #: Initial phase
    initial_phase: tf.Variable  # shape=[None]; dtype=tf.float32

    #: The output signal's shape is `(batch_size, tf.size(time))`
    batch_size: tf.Variable  # shape=[]; dtype=int32

    #: Enables/disables phase noise
    enable_phase_noise: bool

    rng: tf.random.Generator

    def __init__(
        self,
        power: TensorCompatible,
        frequency: TensorCompatible = tf.constant([0.0]),
        linewidth: TensorCompatible = tf.constant([0.0]),
        initial_phase: TensorCompatible = tf.constant([0.0]),
        batch_size: TensorCompatible = 1,
        enable_phase_noise: bool = False,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.power = tf.Variable(to_tensor(power), name=f"{self.name}__power")
        self.frequency = tf.Variable(to_tensor(frequency), name=f"{self.name}__frequency")
        self.linewidth = tf.Variable(to_tensor(linewidth), name=f"{self.name}__linewidth")
        self.initial_phase = tf.Variable(to_tensor(initial_phase), name=f"{self.name}__initial_phase")
        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")
        self.enable_phase_noise = enable_phase_noise

        if rng is None:
            rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self.rng = rng

        self.global_parameters = ["batch_size"]
        self.local_parameters = ["power", "frequency", "linewidth", "initial_phase", "batch_size"]

    def implementation(
        self,
        timing_or_signal: Timing | Signal,
        /,
        power: tf.Tensor,
        frequency: tf.Tensor,
        linewidth: tf.Tensor,
        initial_phase: tf.Tensor,
        batch_size: tf.Tensor
    ) -> Signal:
        if isinstance(timing_or_signal, Signal):
            timing = timing_or_signal.timing
        else:
            timing = timing_or_signal

        time = timing_to_samples(timing)

        power = tf.expand_dims(power, -1)  # shape=[None, 1]
        frequency = tf.expand_dims(frequency, -1)  # shape=[None, 1]
        initial_phase = tf.expand_dims(initial_phase, -1)  # shape=[None, 1]

        phase = 2.0 * constants.pi * frequency * time + initial_phase

        if self.enable_phase_noise:
            phase_noise_stddev = tf.sqrt(2.0 * constants.pi / timing.sampling_rate * linewidth)
            phase_noise = tf.math.cumsum(
                self.rng.normal(
                    shape=[tf.shape(linewidth)[0], tf.size(time)],
                    mean=0.0,
                    stddev=phase_noise_stddev,
                    dtype=tf.float32
                ),
                axis=-1
            )

            phase += phase_noise

        samples = tf.complex(tf.sqrt(power), 0.0) * tf.math.exp(tf.complex(0.0, phase))
        samples = tf.cond(
            tf.shape(samples)[0] == 1,
            lambda: tf.tile(samples, [batch_size, 1]),
            lambda: samples
        )

        return Signal(timing=timing, samples=samples)
