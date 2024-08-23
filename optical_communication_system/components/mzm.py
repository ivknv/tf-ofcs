# -*- coding: utf-8 -*-

from collections.abc import Callable
from typing import Optional
import tensorflow as tf

from . import constants

from .datatypes import (
    Timing, ElectricalSignal, electrical_signal_spec, OpticalSignal
)

__all__ = ["MachZehnderModulator"]


class MachZehnderModulator(tf.Module):
    carrier_source: Callable[[Timing], OpticalSignal]
    extinction_ratio: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        *,
        carrier_source: Callable[[Timing], OpticalSignal],
        extinction_ratio: float = float("inf"),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.carrier_source = carrier_source
        self.extinction_ratio = tf.Variable(tf.constant(extinction_ratio))

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, modulation_signal: ElectricalSignal) -> OpticalSignal:
        carrier_signal = self.carrier_source(modulation_signal.timing)

        k = 1.0 / self.extinction_ratio

        # Modulation signal is expected to be in range [0.0, 1.0]
        modulated_signal = tf.multiply(
            carrier_signal.samples,
            tf.reshape(
                tf.repeat(
                    tf.complex(
                        tf.math.cos(constants.pi * modulation_signal.samples),
                        k * tf.math.sin(constants.pi * modulation_signal.samples)
                    ),
                    tf.size(carrier_signal.frequencies)
                ),
                tf.shape(carrier_signal.samples)
            )
        )

        return OpticalSignal(
            timing=carrier_signal.timing,
            frequencies=carrier_signal.frequencies,
            samples=modulated_signal,
            central_frequency=carrier_signal.central_frequency,
            power=carrier_signal.power
        )
