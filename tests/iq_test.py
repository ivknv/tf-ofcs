# -*- coding: utf-8 -*-

import itertools
import tensorflow as tf

from optical_communication_system.components.datatypes import Timing, ElectricalSignal
from optical_communication_system.components import (
    IQSymbolEncoder, IQSymbolDecoder, IQEncoder, IQDecoder
)

qam4_mapping = [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
qam16_mapping = [
    [-1.0,  1.0], [-1.0,  0.5], [-1.0, -0.5], [-1.0, -1.0],
    [-0.5, -1.0], [-0.5, -0.5], [-0.5,  0.5], [-0.5,  1.0],
    [ 0.5,  1.0], [ 0.5,  0.5], [ 0.5, -0.5], [ 0.5, -1.0],
    [ 1.0, -1.0], [ 1.0, -0.5], [ 1.0,  0.5], [ 1.0,  1.0]
]


def test_iq_symbol_encoder() -> None:
    qam_symbol_encoder = IQSymbolEncoder(
        bits_per_symbol=2,
        mapping=qam4_mapping,
        adjust_for_mzm=False
    )

    qam4_bits = tf.tile(
        tf.reshape(list(itertools.product([0, 1], repeat=2)), (1, 8)),
        [2, 1]
    )
    qam4_symbols = qam_symbol_encoder(qam4_bits)

    qam16_bits = tf.tile(
        tf.reshape(list(itertools.product([0, 1], repeat=4)), (1, 64)),
        [2, 1]
    )

    qam16_symbols = qam_symbol_encoder(
        qam16_bits,
        iq_symbol_encoder__mapping=tf.constant(qam16_mapping),
        iq_symbol_encoder__bits_per_symbol=tf.constant(4)
    )

    assert tf.math.reduce_all(tf.less(tf.abs(qam4_symbols - qam4_mapping), 1e-6))  # type: ignore
    assert tf.math.reduce_all(tf.less(tf.abs(qam16_symbols - qam16_mapping), 1e-6))  # type: ignore


def test_iq_symbol_decoder() -> None:
    qam_symbol_decoder = IQSymbolDecoder(
        bits_per_symbol=2,
        mapping=qam4_mapping
    )

    qam4_symbols = [
        [[ 1.0,  0.9], [-1.1,  0.7], [ 1.0, -0.8]],  # first batch
        [[-1.0,  0.9], [ 0.7,  1.0], [-0.6, -1.0]]   # second batch
    ]

    expected_qam4_bitseq = tf.constant([
        [[1, 1], [0, 1], [1, 0]],
        [[0, 1], [1, 1], [0, 0]]
    ])

    qam16_symbols = [
        [[ 1.0,  0.45], [ 0.5, -0.8], [ 0.3,  0.8], [-1.3,  0.1], [-1.0, -1.4],  [0.7,  1.0]],
        [[-1.0,  0.95], [ 0.2, -0.1], [-0.6,  1.2], [ 0.5,  0.2], [-0.7,  0.3],  [0.1,  0.1]]
    ]

    expected_qam16_bitseq = [
        [[1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0]],
        [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]]
    ]

    decoded_qam4_bitseq = qam_symbol_decoder(qam4_symbols)
    decoded_qam16_bitseq = qam_symbol_decoder(
        qam16_symbols,
        iq_symbol_decoder__mapping=tf.constant(qam16_mapping),
        iq_symbol_decoder__bits_per_symbol=tf.constant(4)
    )

    assert tf.math.reduce_all(decoded_qam4_bitseq == expected_qam4_bitseq)  # type: ignore
    assert tf.math.reduce_all(decoded_qam16_bitseq == expected_qam16_bitseq)  # type: ignore


def test_iq_encoder() -> None:
    qam_symbol_encoder = IQSymbolEncoder(
        bits_per_symbol=4,
        mapping=qam16_mapping,
        adjust_for_mzm=False
    )

    encoder = IQEncoder(sampling_rate=1.0, data_rate=1.0, symbol_encoder=qam_symbol_encoder)
    qam16_bitseq = tf.constant([
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    ])

    expected_i_samples = tf.reshape(
        tf.repeat(
            [
                [ 1.0, 0.5,  0.5, -1.0, -1.0, 0.5],
                [-1.0, 0.5, -0.5,  0.5, -0.5, 0.5]
            ],
            4
        ),
        (2, 24)
    )
    expected_q_samples = tf.reshape(
        tf.repeat(
            [
                [0.5, -1.0, 1.0, 0.5, -1.0, 1.0],
                [1.0, -0.5, 1.0, 0.5,  0.5, 0.5]
            ],
            4),
        (2, 24)
    )

    i_signal, q_signal = encoder(qam16_bitseq)

    assert tf.math.reduce_all(tf.abs(i_signal.samples - expected_i_samples) < 1e-6)  # type: ignore
    assert tf.math.reduce_all(tf.abs(q_signal.samples - expected_q_samples) < 1e-6)  # type: ignore

    assert tf.abs(i_signal.timing.symbol_rate - 0.25) < 1e-6
    assert tf.abs(q_signal.timing.symbol_rate - 0.25) < 1e-6


def test_iq_decoder() -> None:
    qam_symbol_decoder = IQSymbolDecoder(
        bits_per_symbol=4,
        mapping=qam16_mapping
    )

    decoder = IQDecoder(data_rate=1.0, symbol_decoder=qam_symbol_decoder)

    expected_qam16_bitseq = [
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    ]

    timing = Timing(
        duration=tf.constant(24.0),
        sample_count=tf.constant(24),
        sampling_rate=tf.constant(1.0),
        symbol_rate=tf.constant(0.25)
    )

    i_signal = ElectricalSignal(
        timing=timing,
        samples=tf.reshape(
            tf.repeat(
                [
                    [ 1.0, 0.5,  0.5, -1.0, -1.0, 0.5],
                    [-1.0, 0.5, -0.5,  0.5, -0.5, 0.5]
                ],
                4
            ),
            (2, 24)
        )
    )
    q_signal = ElectricalSignal(
        timing=timing,
        samples=tf.reshape(
            tf.repeat(
                [
                    [0.5, -1.0, 1.0, 0.5, -1.0, 1.0],
                    [1.0, -0.5, 1.0, 0.5,  0.5, 0.5]
                ],
                4),
            (2, 24)
        )
    )

    decoded_bitseq = decoder((i_signal, q_signal))

    assert tf.math.reduce_all(decoded_bitseq == expected_qam16_bitseq)  # type: ignore
