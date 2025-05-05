# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs import components
from tf_ofcs.datatypes import Signal, Timing


def test_insert_pilot_symbols() -> None:
    signal = Signal(
        samples=tf.reshape(tf.cast(tf.range(100), tf.float32), [5, -1]),
        timing=Timing(sample_count=20, sampling_rate=1e9)
    )
    pilot_symbol = tf.cast([-1, -2, -3], tf.float32)

    insert_pilot_symbols = components.InsertPilotSymbols(pilot_symbol, pilot_interval=7)

    signal_with_pilot_symbols = insert_pilot_symbols(signal)

    expected_signal = tf.cast([
        [
            -1, -2, -3,
             0,  1,  2,  3,  4,  5,  6,
            -1, -2, -3,
             7,  8,  9, 10, 11, 12, 13,
            -1, -2, -3,
            14, 15, 16, 17, 18, 19
        ],
        [
            -1, -2, -3,
            20, 21, 22, 23, 24, 25, 26,
            -1, -2, -3,
            27, 28, 29, 30, 31, 32, 33,
            -1, -2, -3,
            34, 35, 36, 37, 38, 39
        ],
        [
            -1, -2, -3,
            40, 41, 42, 43, 44, 45, 46,
            -1, -2, -3,
            47, 48, 49, 50, 51, 52, 53,
            -1, -2, -3,
            54, 55, 56, 57, 58, 59
        ],
        [
            -1, -2, -3,
            60, 61, 62, 63, 64, 65, 66,
            -1, -2, -3,
            67, 68, 69, 70, 71, 72, 73,
            -1, -2, -3,
            74, 75, 76, 77, 78, 79
        ],
        [
            -1, -2, -3,
            80, 81, 82, 83, 84, 85, 86,
            -1, -2, -3,
            87, 88, 89, 90, 91, 92, 93,
            -1, -2, -3,
            94, 95, 96, 97, 98, 99
        ]
    ], tf.float32)

    assert tf.math.reduce_all(tf.shape(signal_with_pilot_symbols.samples) == [5, 29])  # type: ignore[operator]
    assert tf.math.reduce_all(signal_with_pilot_symbols.samples == expected_signal)  # type: ignore[operator]


def test_remove_pilot_symbols() -> None:
    signal = Signal(
        samples=tf.cast([
            [
                -1, -2, -3,
                 0,  1,  2,  3,  4,  5,  6,
                -1, -2, -3,
                 7,  8,  9, 10, 11, 12, 13,
                -1, -2, -3,
                14, 15, 16, 17, 18, 19
            ],
            [
                -1, -2, -3,
                20, 21, 22, 23, 24, 25, 26,
                -1, -2, -3,
                27, 28, 29, 30, 31, 32, 33,
                -1, -2, -3,
                34, 35, 36, 37, 38, 39
            ],
            [
                -1, -2, -3,
                40, 41, 42, 43, 44, 45, 46,
                -1, -2, -3,
                47, 48, 49, 50, 51, 52, 53,
                -1, -2, -3,
                54, 55, 56, 57, 58, 59
            ],
            [
                -1, -2, -3,
                60, 61, 62, 63, 64, 65, 66,
                -1, -2, -3,
                67, 68, 69, 70, 71, 72, 73,
                -1, -2, -3,
                74, 75, 76, 77, 78, 79
            ],
            [
                -1, -2, -3,
                80, 81, 82, 83, 84, 85, 86,
                -1, -2, -3,
                87, 88, 89, 90, 91, 92, 93,
                -1, -2, -3,
                94, 95, 96, 97, 98, 99
            ]
        ], tf.float32),
        timing=Timing(sample_count=29, sampling_rate=1e9)
    )

    expected_signal = tf.cast([
        [
             0,  1,  2,  3,  4,  5,  6,
             7,  8,  9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19
        ],
        [
            20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32, 33,
            34, 35, 36, 37, 38, 39
        ],
        [
            40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53,
            54, 55, 56, 57, 58, 59
        ],
        [
            60, 61, 62, 63, 64, 65, 66,
            67, 68, 69, 70, 71, 72, 73,
            74, 75, 76, 77, 78, 79
        ],
        [
            80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93,
            94, 95, 96, 97, 98, 99
        ]
    ], tf.float32)

    remove_pilot_symbols = components.RemovePilotSymbols(pilot_symbol_size=3, pilot_interval=7)
    signal_without_pilot_symbols = remove_pilot_symbols(signal)

    assert tf.math.reduce_all(tf.shape(signal_without_pilot_symbols.samples) == [5, 20])  # type: ignore[operator]
    assert tf.math.reduce_all(signal_without_pilot_symbols.samples == expected_signal)  # type: ignore[operator]
