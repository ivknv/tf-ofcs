# -*- coding: utf-8 -*-

import tensorflow as tf
from optical_communication_system.components import (
    PAMSymbolEncoder, PAMSymbolDecoder, PAMEncoder, PAMDecoder
)


def test_pam_symbol_encoder() -> None:
    pam2_encoder = PAMSymbolEncoder(bits_per_symbol=1)
    pam4_encoder = PAMSymbolEncoder(bits_per_symbol=2)
    pam8_encoder = PAMSymbolEncoder(bits_per_symbol=3)

    pam_input = tf.constant([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    ])

    pam2_symbols = pam2_encoder(pam_input)
    expected_pam2_symbols = tf.cast(pam_input, tf.float32)

    pam4_symbols = pam4_encoder(pam_input)
    expected_pam4_symbols = tf.constant([
        [2.0, 1.0, 0.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 3.0, 1.0, 2.0, 2.0]
    ]) / 3.0

    pam8_symbols = pam8_encoder(pam_input)
    expected_pam8_symbols = tf.constant([
        [4.0, 4.0, 2.0, 3.0],
        [0.0, 3.0, 3.0, 2.0]
    ]) / 7.0

    assert tf.math.reduce_all(tf.abs(pam2_symbols - expected_pam2_symbols) < 1e-9)  # type: ignore
    assert tf.math.reduce_all(tf.abs(pam4_symbols - expected_pam4_symbols) < 1e-9)  # type: ignore
    assert tf.math.reduce_all(tf.abs(pam8_symbols - expected_pam8_symbols) < 1e-9)  # type: ignore


def test_pam_symbol_decoder() -> None:
    pam2_decoder = PAMSymbolDecoder(bits_per_symbol=1)
    pam4_decoder = PAMSymbolDecoder(bits_per_symbol=2)
    pam8_decoder = PAMSymbolDecoder(bits_per_symbol=3)

    expected_bitseq = tf.constant([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    ])

    expected_pam2_bitseq = tf.reshape(expected_bitseq, (2, 12, 1))
    expected_pam4_bitseq = tf.reshape(expected_bitseq, (2, 6, 2))
    expected_pam8_bitseq = tf.reshape(expected_bitseq, (2, 4, 3))

    pam2_symbols = tf.cast(expected_bitseq, tf.float32)
    pam4_symbols = tf.constant([
        [2.0, 1.0, 0.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 3.0, 1.0, 2.0, 2.0]
    ]) / 3.0
    pam8_symbols = tf.constant([
        [4.0, 4.0, 2.0, 3.0],
        [0.0, 3.0, 3.0, 2.0]
    ]) / 7.0

    decoded_pam2_bitseq = pam2_decoder(pam2_symbols)
    decoded_pam4_bitseq = pam4_decoder(pam4_symbols)
    decoded_pam8_bitseq = pam8_decoder(pam8_symbols)

    assert tf.math.reduce_all(decoded_pam2_bitseq == expected_pam2_bitseq)  # type: ignore
    assert tf.math.reduce_all(decoded_pam4_bitseq == expected_pam4_bitseq)  # type: ignore
    assert tf.math.reduce_all(decoded_pam8_bitseq == expected_pam8_bitseq)  # type: ignore


def test_pam_encoder_decoder() -> None:
    sampling_rate = 4.0
    data_rate = 1.0
    upsample_factor = int(sampling_rate / data_rate)

    pam2_encoder = PAMEncoder(
        sampling_rate=sampling_rate,
        data_rate=data_rate,
        symbol_encoder=PAMSymbolEncoder(bits_per_symbol=1)
    )
    pam4_encoder = PAMEncoder(
        sampling_rate=sampling_rate,
        data_rate=data_rate,
        symbol_encoder=PAMSymbolEncoder(bits_per_symbol=2)
    )
    pam8_encoder = PAMEncoder(
        sampling_rate=sampling_rate,
        data_rate=data_rate,
        symbol_encoder=PAMSymbolEncoder(bits_per_symbol=3)
    )

    pam_input = tf.constant([
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0]
    ])

    pam2_signal = pam2_encoder(pam_input)
    expected_pam2_symbols = tf.repeat(tf.cast(pam_input, tf.float32), upsample_factor, axis=1)

    pam4_signal = pam4_encoder(pam_input)
    expected_pam4_symbols = tf.repeat([
        [2.0, 1.0, 0.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 3.0, 1.0, 2.0, 2.0]
    ], 2 * upsample_factor, axis=1) / 3.0

    pam8_signal = pam8_encoder(pam_input)
    expected_pam8_symbols = tf.repeat([
        [4.0, 4.0, 2.0, 3.0],
        [0.0, 3.0, 3.0, 2.0]
    ], 3 * upsample_factor, axis=1) / 7.0

    assert tf.math.reduce_all(tf.abs(pam2_signal.samples - expected_pam2_symbols) < 1e-9)  # type: ignore
    assert tf.math.reduce_all(tf.abs(pam4_signal.samples - expected_pam4_symbols) < 1e-9)  # type: ignore
    assert tf.math.reduce_all(tf.abs(pam8_signal.samples - expected_pam8_symbols) < 1e-9)  # type: ignore

    assert tf.abs(pam2_signal.timing.symbol_rate - 1.0) < 1e-9
    assert tf.abs(pam4_signal.timing.symbol_rate - 0.5) < 1e-9
    assert tf.abs(pam8_signal.timing.symbol_rate - 1 / 3) < 1e-9

    pam2_decoder = PAMDecoder(
        data_rate=data_rate,
        symbol_decoder=PAMSymbolDecoder(bits_per_symbol=1)
    )
    pam4_decoder = PAMDecoder(
        data_rate=data_rate,
        symbol_decoder=PAMSymbolDecoder(bits_per_symbol=2)
    )
    pam8_decoder = PAMDecoder(
        data_rate=data_rate,
        symbol_decoder=PAMSymbolDecoder(bits_per_symbol=3)
    )

    decoded_pam2_bitseq = pam2_decoder(pam2_signal)
    decoded_pam4_bitseq = pam4_decoder(pam4_signal)
    decoded_pam8_bitseq = pam8_decoder(pam8_signal)

    assert tf.math.reduce_all(decoded_pam2_bitseq == pam_input)  # type: ignore
    assert tf.math.reduce_all(decoded_pam4_bitseq == pam_input)  # type: ignore
    assert tf.math.reduce_all(decoded_pam8_bitseq == pam_input)  # type: ignore
