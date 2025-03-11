# -*- coding: utf-8 -*-

import itertools
import tensorflow as tf

from tf_ofcs.datatypes import Timing, Signal
from tf_ofcs.components import IQSymbolEncoder, IQSymbolDecoder

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
        symbol_rate=1e9
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

    assert isinstance(qam4_symbols, Signal)
    assert isinstance(qam16_symbols, Signal)

    assert tf.math.reduce_all(
        tf.less(
            tf.abs(
                tf.subtract(
                    tf.stack([
                        tf.math.real(qam4_symbols.samples),
                        tf.math.imag(qam4_symbols.samples)
                    ], -1),
                    tf.expand_dims(qam4_mapping, 0)
                ),
            ),
            1e-6
        )
    )  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.less(
            tf.abs(
                tf.subtract(
                    tf.stack([
                        tf.math.real(qam16_symbols.samples),
                        tf.math.imag(qam16_symbols.samples)
                    ], -1),
                    tf.expand_dims(qam16_mapping, 0)
                ),
            ),
            1e-6
        )
    )  # type: ignore[operator]


def test_iq_symbol_decoder() -> None:
    qam_symbol_decoder = IQSymbolDecoder(
        bits_per_symbol=2,
        mapping=qam4_mapping
    )

    qam4_symbols = Signal(
        samples=tf.complex(
            [[1.0, -1.1,  1.0], [-1.0, 0.7, -0.6]],
            [[0.9,  0.7, -0.8], [ 0.9, 1.0, -1.0]]
        ),
        timing=Timing(sample_count=3, sampling_rate=1e9)
    )

    expected_qam4_bitseq = tf.constant([
        [1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0]
    ])

    qam16_symbols = Signal(
        samples=tf.complex(
            [[ 1.0,  0.5,  0.3, -1.3, -1.0,  0.7], [-1.0,  0.2, -0.6,  0.5, -0.7,  0.1]],
            [[ 0.45, -0.8,  0.8,  0.1, -1.4,  1.0], [0.95, -0.1,  1.2,  0.2,  0.3,  0.1]]
        ),
        timing=Timing(sample_count=6, sampling_rate=1e9)
    )

    expected_qam16_bitseq = [
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1]
    ]

    decoded_qam4_bitseq = qam_symbol_decoder(qam4_symbols)
    decoded_qam16_bitseq = qam_symbol_decoder(
        qam16_symbols,
        iq_symbol_decoder__mapping=tf.constant(qam16_mapping),
        iq_symbol_decoder__bits_per_symbol=tf.constant(4)
    )

    assert tf.math.reduce_all(decoded_qam4_bitseq == expected_qam4_bitseq)  # type: ignore
    assert tf.math.reduce_all(decoded_qam16_bitseq == expected_qam16_bitseq)  # type: ignore
