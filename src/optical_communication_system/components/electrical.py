# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from ..datatypes import ElectricalSignal, electrical_signal_spec, Timing
from .component import Component

__all__ = [
    "ElectricalAdder",
    "ElectricalSubtractor",
    "ArbitraryWaveformGenerator",
    "Quantizer",
    "ElectricalGain"
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

    def __init__(
        self,
        sampling_rate: float | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate), name=f"{self.name}__sampling_rate")
        self.local_parameters = ["sampling_rate"]

    def implementation(self, input_samples: tf.Tensor, /, sampling_rate: tf.Tensor) -> ElectricalSignal:
        return ElectricalSignal(
            timing=Timing(
                sampling_rate=sampling_rate,
                sample_count=tf.shape(input_samples)[-1]
            ),
            samples=input_samples
        )


class Quantizer(Component):
    step: tf.Variable  # shape=[] or [None]; dtype=float32

    def __init__(
        self,
        step: float | Sequence[float] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.step = tf.Variable(tf.constant(step), name=f"{self.name}__step")
        self.local_parameters = [("step", 0)]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="step")
    ))
    def implementation(self, signal: ElectricalSignal, step: tf.Tensor) -> ElectricalSignal:
        step = tf.expand_dims(step, -1)
        quantized_samples = tf.round(signal.samples / step) * step

        return ElectricalSignal(timing=signal.timing, samples=quantized_samples)


class ElectricalGain(Component):
    gain: tf.Variable  # shape[] or [None]; dtype=float32

    def __init__(
        self,
        gain: float | Sequence[float] | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.gain = tf.Variable(tf.constant(gain), name=f"{self.name}__gain")
        self.local_parameters = [("gain", 0)]

    @tf.function(input_signature=(
        electrical_signal_spec,
        tf.TensorSpec(shape=[None], dtype=tf.float32, name="gain")
    ))
    def implementation(self, signal: ElectricalSignal, gain: tf.Tensor) -> ElectricalSignal:
        gain = tf.expand_dims(gain, -1)

        return ElectricalSignal(
            timing=signal.timing,
            samples=signal.samples * gain
        )
