# -*- coding: utf-8 -*-

import tensorflow as tf

from tf_ofcs.components import FIRFilter
from tf_ofcs.datatypes import ElectricalSignal, Timing


def test_firfilter() -> None:
    timing = Timing(sample_count=1000, sampling_rate=1e9)

    # The order is reversed because the implementation actually uses cross-correlation
    delay_kernels = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0]
    ]

    fir_filter = FIRFilter(delay_kernels)

    input_signal = ElectricalSignal(
        samples=tf.reshape(
            tf.cast(tf.range(timing.sample_count * 2), tf.float32),
            [2, timing.sample_count]
        ),
        timing=timing
    )
    output_signal = fir_filter(input_signal)

    assert isinstance(output_signal, ElectricalSignal)
    assert tf.math.reduce_all(tf.shape(output_signal.samples) == [2, 1004])  # type: ignore[operator]

    assert tf.math.reduce_all(
        output_signal.samples[0][:4] == [0.0, 0.0, 0.0, 0.0]
    )  # type: ignore[operator]
    assert tf.math.reduce_all(
        output_signal.samples[0][4:] == input_signal.samples[0]
    )  # type: ignore[operator]

    assert tf.math.reduce_all(
        output_signal.samples[1][:4] == [0.0, 0.0, 0.0, input_signal.samples[1, 0]]
    )  # type: ignore[operator]

    assert tf.math.reduce_all(
        output_signal.samples[1][4:] == tf.concat([input_signal.samples[1][1:], [0.0]], 0)
    )  # type: ignore[operator]
