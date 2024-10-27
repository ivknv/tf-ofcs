# -*- coding: utf-8 -*-

import tensorflow as tf
from ..component import Component
from ...datatypes import ElectricalSignal, OpticalSignal

__all__ = ["ExtractSamples"]


class ExtractSamples(Component):
    def call(self, signal: ElectricalSignal | OpticalSignal, /, **kwargs) -> tf.Tensor:
        assert isinstance(signal, (ElectricalSignal, OpticalSignal))
        return signal.samples
