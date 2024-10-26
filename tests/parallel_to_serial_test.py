# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import ParallelToSerial
from optical_communication_system.datatypes import ElectricalSignal, Timing


def test_parallel_to_serial() -> None:
    parallel_to_serial = ParallelToSerial(2)

    input_signal = ElectricalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=8),
        samples=tf.cast(tf.constant([
            [[ 0,  1,  2,  3,  4,  5,  6,  7], [ 8,  9, 10, 11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]],
            [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47]],
            [[48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]]
        ]), tf.float32)
    )

    expected_serial_samples = tf.cast(
        tf.constant(
            [
                [
                    [ 0,  1,  2,  3,  4,  5,  6,  7, 16, 17, 18, 19, 20, 21, 22, 23],
                    [ 8,  9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
                ],
                [
                    [32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55],
                    [40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 60, 61, 62, 63]
                ]
            ]
        ),
        tf.float32
    )

    serial_signal = parallel_to_serial(input_signal)

    assert isinstance(serial_signal, ElectricalSignal)
    assert serial_signal.timing.sample_count == 16
    assert tf.math.reduce_all(tf.shape(serial_signal.samples) == (2, 2, 16))  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.abs(serial_signal.samples - expected_serial_samples) < 1e-6
    )  # type: ignore[operator]
