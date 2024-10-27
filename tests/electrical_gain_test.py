# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import ElectricalGain
from optical_communication_system.datatypes import ElectricalSignal, Timing


def test_electrical_gain() -> None:
    gain = ElectricalGain(gain=[0.0, 1.0, 0.5, 2.0])

    input_signal = ElectricalSignal(
        timing=Timing(sampling_rate=1.0, sample_count=100),
        samples=tf.tile(
            tf.reshape(tf.math.cos(tf.range(100, dtype=tf.float32)), [1, 1, 100]),
            [2, 4, 1]
        )
    )

    output_signal = gain(input_signal)

    assert isinstance(output_signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(output_signal.samples) == tf.shape(input_signal.samples))  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            output_signal.samples[:, 0, :] - input_signal.samples[:, 0, :] * 0.0
        ) < 1e-9
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            output_signal.samples[:, 1, :] - input_signal.samples[:, 1, :] * 1.0
        ) < 1e-9
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            output_signal.samples[:, 2, :] - input_signal.samples[:, 3, :] * 0.5
        ) < 1e-9
    )  # type: ignore

    assert tf.math.reduce_all(
        tf.abs(
            output_signal.samples[:, 3, :] - input_signal.samples[:, 3, :] * 2.0
        ) < 1e-9
    )  # type: ignore
