# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from .component import Component
from ..datatypes import OpticalSignal
from .. import constants

__all__ = ["EDFA"]


class EDFA(Component):
    """Erbium-Doped Fiber Amplifier."""

    #: Amplifier gain (unitless)
    gain: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Amplifier noise figure (unitless)
    noise_figure: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Carrier frequency in Hz
    central_frequency: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        gain: float | Sequence[float] | tf.Tensor = 4.0,
        noise_figure: float | Sequence[float] | tf.Tensor = 7.0,
        central_frequency: float | Sequence[float] | tf.Tensor = 193.414e12,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.gain = tf.Variable(tf.constant(gain), name=f"{self.name}__gain")
        self.noise_figure = tf.Variable(tf.constant(noise_figure), name=f"{self.name}__noise_figure")
        self.central_frequency = tf.Variable(tf.constant(central_frequency), name=f"{self.name}__central_frequency")

        if rng is None:
            rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self.rng = rng

        self.local_parameters = [
            ("gain", 0),
            ("noise_figure", 0),
            ("central_frequency", 0)
        ]

    def implementation(
        self,
        signal: OpticalSignal,
        /,
        gain: tf.Tensor,
        noise_figure: tf.Tensor,
        central_frequency: tf.Tensor
    ) -> OpticalSignal:
        # Spectral density of spontaneous emission in W/Hz
        S_sp = 0.5 * constants.h * noise_figure * gain * central_frequency

        # Noise power in watts
        # The 2.0 factor accounts for unpolarized noise (would be 1.0 for dual-polarized signal)
        P_se = 2.0 * S_sp * signal.timing.sampling_rate

        noise_stddev = tf.reshape(
            tf.sqrt(P_se * 0.5),
            [1, tf.size(P_se), 1]
        )

        def _noise() -> tf.Tensor:
            return self.rng.normal(
                tf.shape(signal.samples),
                0.0,
                noise_stddev,
                dtype=tf.float32
            )

        noise = tf.complex(_noise(), _noise())

        return OpticalSignal(
            timing=signal.timing,
            samples=tf.add(
                tf.multiply(
                    signal.samples,
                    tf.cast(
                        tf.reshape(tf.sqrt(gain), [tf.size(gain), 1]),
                        tf.complex64
                    )
                ),
                noise
            )
        )
