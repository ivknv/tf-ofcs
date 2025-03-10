# -*- coding: utf-8 -*-

import tensorflow as tf
from tf_ofcs.components import PAMSymbolEncoder, PAMSymbolDecoder
from tf_ofcs.datatypes import Signal, Timing


def test_pam_symbol_encoder() -> None:
    pam2_encoder = PAMSymbolEncoder(bits_per_symbol=1, symbol_rate=1e3)
    pam4_encoder = PAMSymbolEncoder(bits_per_symbol=2, symbol_rate=1e3)
    pam8_encoder = PAMSymbolEncoder(bits_per_symbol=3, symbol_rate=1e3)

    pam_input = tf.constant([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    ])

    pam2_symbols = pam2_encoder(pam_input)
    expected_pam2_symbols = Signal(
        samples=tf.constant([
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        ]),
        timing=Timing(sample_count=12, sampling_rate=1e3)
    )

    pam4_symbols = pam4_encoder(pam_input)
    expected_pam4_symbols = Signal(
        samples=tf.constant([
            [2.0, 1.0, 0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 1.0, 2.0, 2.0]
        ]) / 3.0,
        timing=Timing(sample_count=6, sampling_rate=1e3)
    )

    pam8_symbols = pam8_encoder(pam_input)
    expected_pam8_symbols = Signal(
        samples=tf.constant([
            [4.0, 4.0, 2.0, 3.0],
            [0.0, 3.0, 3.0, 2.0]
        ]) / 7.0,
        timing=Timing(sample_count=4, sampling_rate=1e3)
    )

    assert isinstance(pam2_symbols, Signal)
    assert isinstance(pam4_symbols, Signal)
    assert isinstance(pam8_symbols, Signal)

    assert tf.math.reduce_all(
        tf.abs(pam2_symbols.samples - expected_pam2_symbols.samples) < 1e-9
    )  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.abs(pam4_symbols.samples - expected_pam4_symbols.samples) < 1e-9
    )  # type: ignore[operator]
    assert tf.math.reduce_all(
        tf.abs(pam8_symbols.samples - expected_pam8_symbols.samples) < 1e-9
    )  # type: ignore[operator]


def test_pam_symbol_decoder() -> None:
    pam2_decoder = PAMSymbolDecoder(bits_per_symbol=1)
    pam4_decoder = PAMSymbolDecoder(bits_per_symbol=2)
    pam8_decoder = PAMSymbolDecoder(bits_per_symbol=3)

    expected_bitseq = tf.constant([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    ])

    pam2_symbols = Signal(
        samples=tf.cast(expected_bitseq, tf.float32),
        timing=Timing(sample_count=12, sampling_rate=1e3)
    )
    pam4_symbols = Signal(
        samples=tf.constant([
            [2.0, 1.0, 0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 3.0, 1.0, 2.0, 2.0]
        ]) / 3.0,
        timing=Timing(sample_count=6, sampling_rate=1e3)
    )
    pam8_symbols = Signal(
        samples=tf.constant([
            [4.0, 4.0, 2.0, 3.0],
            [0.0, 3.0, 3.0, 2.0]
        ]) / 7.0,
        timing=Timing(sample_count=4, sampling_rate=1e3)
    )

    decoded_pam2_bitseq = pam2_decoder(pam2_symbols)
    decoded_pam4_bitseq = pam4_decoder(pam4_symbols)
    decoded_pam8_bitseq = pam8_decoder(pam8_symbols)

    assert isinstance(decoded_pam2_bitseq, tf.Tensor)
    assert isinstance(decoded_pam4_bitseq, tf.Tensor)
    assert isinstance(decoded_pam8_bitseq, tf.Tensor)

    assert tf.math.reduce_all(decoded_pam2_bitseq == expected_bitseq)  # type: ignore[operator]
    assert tf.math.reduce_all(decoded_pam4_bitseq == expected_bitseq)  # type: ignore[operator]
    assert tf.math.reduce_all(decoded_pam8_bitseq == expected_bitseq)  # type: ignore[operator]
