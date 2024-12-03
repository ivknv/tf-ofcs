# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import RedistributeDimensions
from tf_ofcs.datatypes import ElectricalSignal, Timing


def test_redistribute_dimensions() -> None:
    signal = ElectricalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=20.0),
        samples=tf.fill((3, 4, 20), 1.0)
    )

    redistribute_dims = RedistributeDimensions(factors=(2, 1))
    reshaped_signal = redistribute_dims(signal)

    assert isinstance(signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(reshaped_signal.samples) == (6, 2, 20))  # type: ignore
    assert tf.math.reduce_all(
        tf.reshape(reshaped_signal.samples, (240,)) == tf.reshape(signal.samples, (240,))
    )  # type: ignore
