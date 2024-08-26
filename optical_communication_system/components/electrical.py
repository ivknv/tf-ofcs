# -*- coding: utf-8 -*-

import tensorflow as tf

from .datatypes import ElectricalSignal, electrical_signal_spec

__all__ = ["ElectricalAdder", "ElectricalSubtractor"]


class ElectricalAdder(tf.Module):
    @tf.function(input_signature=((electrical_signal_spec, electrical_signal_spec),))
    def __call__(self, input_signals: tuple[ElectricalSignal, ElectricalSignal]) -> ElectricalSignal:
        return ElectricalSignal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples + input_signals[1].samples
        )


class ElectricalSubtractor(tf.Module):
    @tf.function(input_signature=((electrical_signal_spec, electrical_signal_spec),))
    def __call__(self, input_signals: tuple[ElectricalSignal, ElectricalSignal]) -> ElectricalSignal:
        return ElectricalSignal(
            timing=input_signals[0].timing,
            samples=input_signals[0].samples - input_signals[1].samples
        )
