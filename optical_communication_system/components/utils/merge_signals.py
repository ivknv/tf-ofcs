# -*- coding: utf-8 -*-

import tensorflow as tf

from ..datatypes import ElectricalSignal, OpticalSignal
from ..component import Component

__all__ = ["MergeSignals"]


class MergeSignals(Component):
    """
        Merges a tuple of `n` signals with shape (batch_size, parameter_dims, sample_count)
        into a signal with shape (batch_size * n, parameter_dims, sample_count).

        Of course, the signals don't have to have the same `batch_size`, but
        they must be of the same type (e.g all optical or all electrical) and
        have the same `timing`.

    """

    def call(
        self,
        signals: tuple[ElectricalSignal, ...] | tuple[OpticalSignal, ...],
        /,
        **kwargs
    ) -> ElectricalSignal | OpticalSignal:
        Signal = signals[0].__class__
        timing = signals[0].timing

        merged = tf.concat([signal.samples for signal in signals], 0)

        return Signal(samples=merged, timing=timing)
