# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import TileSignals
from tf_ofcs.datatypes import ElectricalSignal, Timing


def test_tile_signals() -> None:
    input_signal = ElectricalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=8),
        samples=tf.cast(
            tf.constant([
                [ 0,  1,  2,  3,  4,  5,  6,  7],
                [ 8,  9, 10, 11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20, 21, 22, 23],
                [24, 25, 26, 27, 28, 29, 30, 31],
                [32, 33, 34, 35, 36, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 46, 47]
            ]),
            tf.float32
        )
    )

    tile_signals = TileSignals(3)
    output_signal = tile_signals(input_signal)

    expected_samples = tf.tile(input_signal.samples, [3, 1])

    assert isinstance(output_signal, ElectricalSignal)

    assert tf.math.reduce_all(tf.shape(output_signal.samples) == (18, 8))  # type: ignore[operator]

    assert tf.math.reduce_all(
        tf.abs(output_signal.samples - expected_samples) < 1e-9
    )  # type: ignore[operator]
