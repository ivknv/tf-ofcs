# -*- coding: utf-8 -*-

from collections.abc import Callable, Sequence
import typing
from typing import Any, Optional
import tensorflow as tf

from .datatypes import (
    bit_sequence_spec, bit_dtype,
    Timing,
    ElectricalSignal, electrical_signal_spec,
    OpticalSignal
)

__all__ = [
    "PAMConstellation",
    "PAMEncoder",
    "PAMDecoder",
    "IntensityModulator",
    "AmplitudeModulator"
]


class PAMConstellation:
    order: tf.Variable  # shape=(1,); dtype=int32
    mapping: tf.Variable  # shape=(2 ** order,); dtype=float32

    def __init__(
        self,
        *,
        order: int,
        mapping: Optional[Sequence[float]] = None
    ) -> None:
        self.order = tf.Variable(tf.constant(order))

        if mapping is None:
            self.mapping = tf.Variable(
                typing.cast(tf.Tensor, self._default_pam_constellation())
            )
        else:
            self.mapping = tf.Variable(tf.constant(mapping))

    @tf.function
    def _default_pam_constellation(self) -> tf.Tensor:
        return tf.vectorized_map(
            fn=lambda i: tf.cast(i / (tf.math.pow(2, self.order) - 1), tf.float32),
            elems=tf.range(tf.math.pow(2, self.order))
        )

    @tf.function(input_signature=(bit_sequence_spec,))
    def encode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        symbol_index: Any = tf.reduce_sum(
            symbol * 2 ** (self.order - tf.range(self.order) - 1)
        )

        return self.mapping[symbol_index]

    @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.complex64),))
    def decode_symbol(self, symbol: tf.Tensor) -> tf.Tensor:
        best_index = tf.cast(tf.argmin(tf.abs(self.mapping - symbol)), tf.int32)

        return tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.math.floormod(best_index / 2 ** (self.order - i - 1), 2),
                bit_dtype
            ),
            elems=tf.range(self.order)
        )


class PAMEncoder(tf.Module):
    #: Sampling rate in Hz
    sampling_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: PAMConstellation

    #: Modulation order (number of bits per symbol)
    order: tf.Variable  # shape=(1,); dtype=int32

    def __init__(
        self,
        sampling_rate: float,
        data_rate: float,
        order: int = 1,
        constellation: Optional[PAMConstellation] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if constellation is None:
            constellation = PAMConstellation(order=order)

        self.sampling_rate = tf.Variable(tf.constant(sampling_rate))
        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.order = tf.Variable(tf.constant(order))
        self.constellation = constellation

    @tf.function(input_signature=(bit_sequence_spec,))
    def __call__(self, bitseq: tf.Tensor) -> ElectricalSignal:
        tf.debugging.assert_equal(tf.math.floormod(tf.size(bitseq), self.order), 0)

        signal_duration = tf.divide(
            tf.cast(tf.size(bitseq), tf.float32),
            self.data_rate
        )

        time = tf.linspace(
            0.0,
            signal_duration,
            tf.cast(tf.round(self.sampling_rate * signal_duration), tf.int32)
        )

        pam_sequence = tf.vectorized_map(
            fn=lambda i: self.constellation.encode_symbol(bitseq[i * self.order:(i + 1) * self.order]),
            elems=tf.range(tf.math.floordiv(tf.size(bitseq), self.order))
        )

        return ElectricalSignal(
            timing=Timing(
                time=time,
                sampling_rate=self.sampling_rate,
                symbol_rate=tf.divide(self.data_rate, tf.cast(self.order, tf.float32))
            ),
            samples=tf.repeat(
                pam_sequence,
                tf.cast(
                    tf.math.ceil(tf.divide(tf.size(time), tf.size(pam_sequence))),
                    tf.int32
                )
            )[:tf.size(time)]
        )


class PAMDecoder(tf.Module):
    #: Data rate in bits/second
    data_rate: tf.Variable  # shape=(1,); dtype=float32

    #: Constellation used for encoding
    constellation: PAMConstellation

    #: Modulation order (number of bits per symbol)
    order: tf.Variable  # shape=(1,); dtype=int32

    #: Minimum current (or power in watts) recognized as non-zero
    threshold: tf.Variable  # shape=(1,); dtype=float32

    #: Currents above this value will be clipped
    max_current: tf.Variable  # shape=(1,); dtype=float32

    def __init__(
        self,
        data_rate: float,
        order: int = 1,
        constellation: Optional[PAMConstellation] = None,
        threshold: float = 1e-5,
        max_current: float = 2e-3,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        if constellation is None:
            constellation = PAMConstellation(order=order)

        self.data_rate = tf.Variable(tf.constant(data_rate))
        self.order = tf.Variable(tf.constant(order))
        self.constellation = constellation
        self.threshold = tf.Variable(tf.constant(threshold))
        self.max_current = tf.Variable(tf.constant(max_current))

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, signal: ElectricalSignal) -> tf.Tensor:
        n = tf.cast(
            tf.round(signal.timing.sampling_rate / self.data_rate * tf.cast(self.order, tf.float32)),
            tf.int32
        )

        current = tf.clip_by_value(tf.abs(signal.samples), 0, self.max_current)
        max_current = tf.math.maximum(tf.reduce_max(current), self.threshold)

        symbols = tf.vectorized_map(
            fn=lambda i: tf.cast(
                tf.reduce_mean(current[i * n:(i + 1) * n]) / max_current,
                tf.complex64
            ),
            elems=tf.range(tf.cast(tf.size(current) / n, tf.int32))
        )

        return tf.reshape(
            tf.vectorized_map(
                fn=self.constellation.decode_symbol,
                elems=symbols
            ),
            (tf.size(symbols) * self.order,)
        )


class IntensityModulator(tf.Module):
    #: Carrier signal source
    carrier_source: Callable[[Timing], OpticalSignal]

    def __init__(self, carrier_source: Callable, name=None) -> None:
        super().__init__(name=name)

        self.carrier_source = carrier_source

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, modulation_signal: ElectricalSignal) -> OpticalSignal:
        carrier_signal = self.carrier_source(modulation_signal.timing)

        modulated_signal = tf.multiply(
            carrier_signal.samples,
            tf.cast(
                tf.abs(
                    tf.reshape(
                        tf.repeat(
                            modulation_signal.samples,
                            tf.size(carrier_signal.frequencies)
                        ),
                        tf.shape(carrier_signal.samples)
                    )
                ) ** 0.5,
                tf.complex64
            )
        )

        return OpticalSignal(
            timing=carrier_signal.timing,
            frequencies=carrier_signal.frequencies,
            samples=modulated_signal,
            central_frequency=carrier_signal.central_frequency,
            power=carrier_signal.power
        )


class AmplitudeModulator(tf.Module):
    #: Carrier signal source
    carrier_source: Callable[[Timing], OpticalSignal]

    def __init__(self, carrier_source: Callable, name=None) -> None:
        super().__init__(name=name)

        self.carrier_source = carrier_source

    @tf.function(input_signature=(electrical_signal_spec,))
    def __call__(self, modulation_signal: ElectricalSignal) -> OpticalSignal:
        carrier_signal = self.carrier_source(modulation_signal.timing)

        modulated_signal = tf.multiply(
            carrier_signal.samples,
            tf.reshape(
                tf.repeat(
                    tf.cast(modulation_signal.samples, tf.complex64),
                    tf.size(carrier_signal.frequencies)
                ),
                tf.shape(carrier_signal.samples)
            )
        )

        return OpticalSignal(
            timing=carrier_signal.timing,
            frequencies=carrier_signal.frequencies,
            samples=modulated_signal,
            central_frequency=carrier_signal.central_frequency,
            power=carrier_signal.power
        )
