# -*- coding: utf-8 -*-

from collections.abc import Callable, Sequence
from typing import Optional
import tensorflow as tf

from .. import constants
from .component import Component

from .datatypes import (
    Timing, ElectricalSignal, electrical_signal_spec, OpticalSignal, optical_signal_spec
)

__all__ = ["MachZehnderModulator"]


class MachZehnderModulator(Component):
    carrier_source: Callable[[Timing], OpticalSignal]
    extinction_ratio: tf.Variable  # shape=[None] or []; dtype=float32

    def __init__(
        self,
        *,
        carrier_source: Callable[[Timing], OpticalSignal],
        extinction_ratio: float | Sequence[float] | tf.Tensor = float("inf"),
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.carrier_source = carrier_source
        self.extinction_ratio = tf.Variable(tf.constant(extinction_ratio))

        self.local_parameters = [("extinction_ratio", 0)]

    def call(self, modulation_signal: ElectricalSignal, /, **kwargs) -> OpticalSignal:
        carrier_signal = self.carrier_source(modulation_signal.timing, **kwargs)

        extinction_ratio = self.get_local_kwarg(kwargs, "extinction_ratio")

        if extinction_ratio is None:
            extinction_ratio = self.extinction_ratio

        extinction_ratio = tf.broadcast_to(extinction_ratio, [tf.shape(carrier_signal.samples)[1]])

        return self._implementation(
            modulation_signal,
            carrier_signal,
            extinction_ratio=extinction_ratio
        )

    @tf.function(input_signature=(
        electrical_signal_spec,
        optical_signal_spec,
        tf.TensorSpec([None], tf.float32)
    ))
    def _implementation(
        self,
        modulation_signal: ElectricalSignal,
        carrier_signal: OpticalSignal,
        extinction_ratio: tf.Tensor
    ) -> OpticalSignal:
        k = tf.expand_dims(1.0 / extinction_ratio, -1)

        # Modulation signal is expected to be in range [0.0, 1.0]
        modulated_signal = tf.multiply(
            carrier_signal.samples,
            tf.complex(
                tf.math.cos(constants.pi * modulation_signal.samples),
                k * tf.math.sin(constants.pi * modulation_signal.samples)
            ),
        )

        return OpticalSignal(
            timing=carrier_signal.timing,
            samples=modulated_signal
        )
