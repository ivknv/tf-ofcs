# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import GatherSignals
from optical_communication_system.components.datatypes import ElectricalSignal, Timing


def test_gather_signals() -> None:
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

    gather_signals = GatherSignals([0, 2])
    output_signal = gather_signals(input_signal)

    expected_samples = tf.gather(input_signal.samples, [0, 2])

    assert isinstance(output_signal, ElectricalSignal)

    assert tf.math.reduce_all(tf.shape(output_signal.samples) == (2, 2, 8))  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.abs(output_signal.samples - expected_samples) < 1e-9
    )  # type: ignore[operator]
