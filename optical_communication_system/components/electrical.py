# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from .datatypes import ElectricalSignal, electrical_signal_spec
from .component import Component
from .utils import generate_timing

__all__ = [
    "ElectricalAdder",
    "ElectricalSubtractor",
    "ArbitraryWaveformGenerator",
    "Quantizer"
]


class ElectricalAdder(Component):
    @tf.function(input_signature=((electrical_signal_spec, electrical_signal_spec),))
    def implementation(self, input_signals: tuple[ElectricalSignal, ElectricalSignal]) -> ElectricalSignal:
        return ElectricalSignal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples + input_signals[1].samples
        )


class ElectricalSubtractor(Component):
    @tf.function(input_signature=((electrical_signal_spec, electrical_signal_spec),))
    def implementation(self, input_signals: tuple[ElectricalSignal, ElectricalSignal]) -> ElectricalSignal:
        return ElectricalSignal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples - input_signals[1].samples
        )


class ArbitraryWaveformGenerator(Component):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=[]; dtype=float32

    #: Symbol rate in symbols/second
    symbol_rate: tf.Variable  # shape=[]; dtype=float32

    def __init__(
        self,
        sampling_rate: float | tf.Tensor,
        symbol_rate: float | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.symbol_rate = tf.Variable(tf.constant(symbol_rate))

        self.local_parameters = ["sampling_rate", "symbol_rate"]

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="input_samples"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="sampling_rate"),
        tf.TensorSpec(shape=[], dtype=tf.float32, name="symbol_rate"),
    ))
    def implementation(
        self,
        input_samples: tf.Tensor,
        sampling_rate: tf.Tensor,
        symbol_rate: tf.Tensor
    ) -> ElectricalSignal:
        symbol_count = tf.cast(
            tf.round(tf.cast(tf.shape(input_samples)[-1], tf.float32) * symbol_rate / sampling_rate),
            tf.int32
        )

        return ElectricalSignal(
            timing=generate_timing(sampling_rate, symbol_rate, symbol_count),
            samples=input_samples
        )


class Quantizer(Component):
    step: tf.Variable  # shape=[] or [None]; dtype=tf.float32

    def __init__(
        self,
        step: float | Sequence[float] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.step = tf.Variable(tf.constant(step))

        self.local_parameters = [("step", 0)]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="step"),
    ))
    def implementation(self, signal: ElectricalSignal, step: tf.Tensor) -> ElectricalSignal:
        step = tf.expand_dims(step, -1)
        quantized_samples = tf.round(signal.samples / step) * step

        return ElectricalSignal(timing=signal.timing, samples=quantized_samples)
