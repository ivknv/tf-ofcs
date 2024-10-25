# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import SliceSignal
from optical_communication_system.components.datatypes import ElectricalSignal, Timing


def test_slice_signal() -> None:
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
                    ],
                    [
                        [48, 49, 50, 51, 52, 53, 54, 55],
                        [56, 57, 58, 59, 60, 61, 62, 63]
                    ]
                ]
            ),
            tf.float32
        )
    )

    slice_signal = SliceSignal(slice=(1, 2), factor=2)
    output_signal = slice_signal(input_signal)

    expected_samples = input_signal.samples[2:4]

    assert isinstance(output_signal, ElectricalSignal)

    assert tf.math.reduce_all(tf.shape(output_signal.samples) == (2, 2, 8))  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.abs(output_signal.samples - expected_samples) < 1e-9
    )  # type: ignore[operator]
