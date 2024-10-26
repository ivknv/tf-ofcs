# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import SplitSignals
from optical_communication_system.datatypes import ElectricalSignal, Timing


def test_split_signals() -> None:
    input_signal = ElectricalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=8),
        samples=tf.cast(
            tf.constant(
                [
                    [
                        [ 0,  1,  2,  3,  4,  5,  6,  7],
                        [ 8,  9, 10, 11, 12, 13, 14, 15]
                    ],
                    [
                        [16, 17, 18, 19, 20, 21, 22, 23],
                        [24, 25, 26, 27, 28, 29, 30, 31]
                    ],
                    [
                        [32, 33, 34, 35, 36, 37, 38, 39],
                        [40, 41, 42, 43, 44, 45, 46, 47]
                    ]
                ]
            ),
            tf.float32
        )
    )

    split_signals = SplitSignals(3)
    output_signals = split_signals(input_signal)

    assert isinstance(output_signals, tuple)
    assert len(output_signals) == 3

    assert tf.math.reduce_all(tf.shape(output_signals[0].samples) == (1, 2, 8))  # type: ignore[operator]
    assert tf.math.reduce_all(tf.shape(output_signals[1].samples) == (1, 2, 8))  # type: ignore[operator]
    assert tf.math.reduce_all(tf.shape(output_signals[2].samples) == (1, 2, 8))  # type: ignore[operator]

    assert tf.math.reduce_all(output_signals[0].samples == input_signal.samples[0])  # type: ignore[operator]
    assert tf.math.reduce_all(output_signals[1].samples == input_signal.samples[1])  # type: ignore[operator]
    assert tf.math.reduce_all(output_signals[2].samples == input_signal.samples[2])  # type: ignore[operator]
