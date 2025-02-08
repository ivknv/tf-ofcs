# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import PadSignal
from tf_ofcs.datatypes import ElectricalSignal, Timing


def test_pad_signal() -> None:
    input_signal = ElectricalSignal(
        timing=Timing(sample_count=10, sampling_rate=1.0),
        samples=tf.tile(
            [[tf.cast(tf.range(1, 11), tf.float32)]],
            [2, 3, 1]
        )
    )

    pad_signal = PadSignal(left_padding=2, right_padding=3)
    padded_signal = pad_signal(input_signal)

    assert isinstance(padded_signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(padded_signal.samples) == [2, 3, 60])  # type: ignore[operator]

    assert tf.math.reduce_all(padded_signal.samples[:, :, :20] == 0.0)  # type: ignore[operator]
    assert tf.math.reduce_all(padded_signal.samples[:, :, 30:] == 0.0)  # type: ignore[operator]
    assert tf.math.reduce_all(padded_signal.samples[:, :, 20:30] == input_signal.samples)  # type: ignore[operator]

    unpad_signal = PadSignal(left_padding=2, right_padding=3, unpad=True)
    unpadded_signal = unpad_signal(padded_signal)

    assert isinstance(unpadded_signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(unpadded_signal.samples) == [2, 3, 10])  # type: ignore[operator]
    assert tf.math.reduce_all(unpadded_signal.samples == input_signal.samples)  # type: ignore[operator]
