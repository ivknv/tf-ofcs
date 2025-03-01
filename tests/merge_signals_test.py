# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import MergeSignals
from tf_ofcs.datatypes import ElectricalSignal, Timing


def test_merge_signals() -> None:
    input_signals = (
        ElectricalSignal(
            timing=Timing(sampling_rate=1.0, sample_count=8),
            samples=tf.cast(tf.constant(
                [[ 0,  1,  2,  3,  4,  5,  6,  7], [ 8,  9, 10, 11, 12, 13, 14, 15]]
            ), tf.float32)
        ),
        ElectricalSignal(
            timing=Timing(sampling_rate=1.0, sample_count=8),
            samples=tf.cast(tf.constant(
                [[16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]
            ), tf.float32)
        ),
        ElectricalSignal(
            timing=Timing(sampling_rate=1.0, sample_count=8),
            samples=tf.cast(tf.constant(
                [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47]]
            ), tf.float32)
        )
    )

    merge_signals = MergeSignals()
    output_signal = merge_signals(input_signals)

    assert isinstance(output_signal, ElectricalSignal)

    assert tf.math.reduce_all(tf.shape(output_signal.samples) == (6, 8))  # type: ignore[operator]

    assert tf.math.reduce_all(output_signal.samples[0:2] == input_signals[0].samples)  # type: ignore[operator]
    assert tf.math.reduce_all(output_signal.samples[2:4] == input_signals[1].samples)  # type: ignore[operator]
    assert tf.math.reduce_all(output_signal.samples[4:] == input_signals[2].samples)  # type: ignore[operator]
