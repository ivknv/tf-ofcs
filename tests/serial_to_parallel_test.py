# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import SerialToParallel
from optical_communication_system.datatypes import ElectricalSignal, Timing


def test_serial_to_parallel() -> None:
    serial_to_parallel = SerialToParallel(2)

    input_signal = ElectricalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=8),
        samples=tf.cast(tf.constant([
            [[ 0,  1,  2,  3,  4,  5,  6,  7], [ 8,  9, 10, 11, 12, 13, 14, 15]],
            [[16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]],
            [[32, 33, 34, 35, 36, 37, 38, 39], [40, 41, 42, 43, 44, 45, 46, 47]],
            [[48, 49, 50, 51, 52, 53, 54, 55], [56, 57, 58, 59, 60, 61, 62, 63]]
        ]), tf.float32)
    )

    parallel_signals = serial_to_parallel(input_signal)
    expected_parallel_samples = tf.cast(tf.constant([
        [[ 0,  1,  2,  3], [ 8,  9, 10, 11]],
        [[ 4,  5,  6,  7], [12, 13, 14, 15]],
        [[16, 17, 18, 19], [24, 25, 26, 27]],
        [[20, 21, 22, 23], [28, 29, 30, 31]],
        [[32, 33, 34, 35], [40, 41, 42, 43]],
        [[36, 37, 38, 39], [44, 45, 46, 47]],
        [[48, 49, 50, 51], [56, 57, 58, 59]],
        [[52, 53, 54, 55], [60, 61, 62, 63]]
    ]), tf.float32)

    assert isinstance(parallel_signals, ElectricalSignal)
    assert parallel_signals.timing.sample_count == 4
    assert tf.math.reduce_all(tf.shape(parallel_signals.samples) == (8, 2, 4))  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.abs(parallel_signals.samples - expected_parallel_samples) < 1e-6
    )  # type: ignore[operator]
