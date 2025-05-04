# -*- coding: utf-8 -*-
# Copyright (C) 2025 Ivan Konovalov

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Sequence
from typing import Optional
import tensorflow as tf

from ..datatypes import Signal, TensorCompatible
from .component import Component
from ..utils import to_tensor

__all__ = ["DecisionDirectedPhaseRecovery", "ViterbiViterbiPhaseRecovery", "PilotAidedPhaseRecovery"]


class DecisionDirectedPhaseRecovery(Component):
    """
        Decision-directed carrier phase recovery. Updates phase error
        every `window_length` symbols. The bigger the window, the fewer
        loop iterations there are, but this comes at the cost of reduced
        phase error tolerance.

        :ivar window_length: `shape=[]; dtype=tf.int32`, length of the window
        :ivar mapping: `shape=[None]; dtype=tf.complex64` constellation mapping
                       used for symbol decisions
    """

    window_length: tf.Variable  # shape=[]; dtype=tf.int32
    mapping: tf.Variable  # shape=[None]; dtype=tf.complex64

    def __init__(
        self,
        window_length: int | tf.Tensor,
        mapping: TensorCompatible,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.window_length = tf.Variable(tf.constant(window_length), name=f"{self.name}__window_length")
        self.mapping = tf.Variable(to_tensor(mapping), name=f"{self.name}__mapping")
        self.local_parameters = ["window_length", "mapping"]

    def implementation(
        self,
        input_symbols: Signal,
        /,
        window_length: tf.Tensor,
        mapping: tf.Tensor
    ) -> Signal:
        assert input_symbols.samples.dtype is tf.complex64

        batch_size = tf.shape(input_symbols.samples)[0]
        sample_count = tf.shape(input_symbols.samples)[-1]

        # TODO: handle case where sample_count is not a multiple of window_length
        tf.debugging.assert_equal(sample_count % window_length, 0)

        # Slice the symbols into windows
        # This reduces the number of iterations of the loop at the expense of
        # reduced phase error tolerance
        sliced_symbols = tf.signal.frame(input_symbols.samples, window_length, window_length, axis=-1)

        phase_error = tf.zeros((batch_size,), dtype=tf.float32)
        phase_offsets = tf.TensorArray(
            dtype=tf.float32,
            size=sample_count // window_length
        )

        for i in tf.range(sample_count // window_length):
            symbols = sliced_symbols[:, i]  # shape=[batch_size, window_length]
            symbols *= tf.expand_dims(tf.math.exp(tf.complex(0.0, -phase_error)), -1)

            best_indices = tf.cast(
                tf.argmin(
                    tf.abs(mapping - tf.expand_dims(symbols, -1)),
                    axis=-1
                ),
                tf.int32
            )  # shape=[batch_size, window_length]

            # shape=[batch_size, window_length]
            expected_phase = tf.math.angle(tf.gather(mapping, best_indices))
            actual_phase = tf.math.angle(symbols)

            phase_error += tf.math.reduce_mean(actual_phase - expected_phase, axis=-1)
            phase_offsets = phase_offsets.write(i, phase_error)

        # Compute the resulting phase rotations for all input symbols
        rotation = tf.math.exp(
            tf.repeat(tf.complex(0.0, -tf.transpose(phase_offsets.stack())), window_length, axis=-1)
        )

        return Signal(
            samples=input_symbols.samples * rotation,
            timing=input_symbols.timing
        )


class ViterbiViterbiPhaseRecovery(Component):
    """
        Viterbi-Viterbi carrier phase recovery. Can only be used for PSK.

        :ivar window_length: `shape=[]; dtype=tf.int32`, length of the window
        :ivar M: `shape=[]; dtype=tf.int32`, PSK constellation order
        :ivar phase_offset: `shape=[]; dtype=tf.float32`, constellation phase offset
    """

    window_length: tf.Variable  # shape=[]; dtype=tf.int32
    M: tf.Variable  # shape=[]; dtype=tf.int32
    phase_offset: tf.Variable  # shape=[]; dtype=tf.float32

    def __init__(
        self,
        window_length: int | tf.Tensor,
        M: int | tf.Tensor,
        phase_offset: float | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.window_length = tf.Variable(tf.constant(window_length), name=f"{self.name}__window_length")
        self.M = tf.Variable(tf.constant(M), name=f"{self.name}__M")
        self.phase_offset = tf.Variable(tf.constant(phase_offset), name=f"{self.name}__phase_offset")
        self.local_parameters = ["window_length", "M", "phase_offset"]

    def implementation(
        self,
        input_symbols: Signal,
        /,
        window_length: tf.Tensor,
        M: tf.Tensor,
        phase_offset: tf.Tensor
    ) -> Signal:
        assert input_symbols.samples.dtype is tf.complex64

        sample_count = tf.shape(input_symbols.samples)[-1]

        # TODO: handle case where sample_count is not a multiple of window_length
        tf.debugging.assert_equal(sample_count % window_length, 0)

        # Slice the symbols into windows
        # shape=[batch_size, sample_count // window_length, window_length]
        symbols = tf.signal.frame(input_symbols.samples, window_length, window_length, axis=-1)

        M = tf.cast(M, tf.float32)
        phase_error = tf.subtract(
            tf.math.angle(
                tf.math.reduce_sum(
                    (symbols / tf.complex(tf.abs(symbols), 0.0)) ** tf.complex(M, 0.0),
                    axis=-1
                )
            ) / M,
            phase_offset
        )  # shape=[batch_size, sample_count // window_length]

        # Compute the resulting phase rotations for all input symbols
        rotation = tf.math.exp(
            tf.repeat(tf.complex(0.0, -phase_error), window_length, axis=-1)  # type: ignore[operator]
        )

        return Signal(
            samples=input_symbols.samples * rotation,
            timing=input_symbols.timing
        )


class PilotAidedPhaseRecovery(Component):
    """
        Pilot-aided carrier phase recovery. Estimates phase error based on known
        pilot symbols periodically inserted into the data stream.

        :param pilot_symbol_size: `shape=[], dtype=tf.int32`, pilot symbol size
        :param pilot_interval: `shape=[], dtype=tf.int32`, pilot symbol interval (number of samples)
        :param keep_pilot_symbols: `bool`, if `True`, the pilot symbols are kept in the output,
            otherwise they are discarded
    """

    pilot_symbol: tf.Variable  # shape=[None]; dtype=tf.complex64
    pilot_interval: tf.Variable  # shape=[]; dtype=tf.int32
    keep_pilot_symbols: bool

    def __init__(
        self,
        pilot_symbol: Sequence[complex] | tf.Tensor,
        pilot_interval: int | tf.Tensor,
        keep_pilot_symbols: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)
        self.pilot_symbol = tf.Variable(
            tf.constant(pilot_symbol),  # type: ignore[arg-type]
            name=f"{self.name}__pilot_symbol"
        )
        self.pilot_interval = tf.Variable(
            tf.constant(pilot_interval),
            name=f"{self.name}__pilot_interval"
        )
        self.keep_pilot_symbols = keep_pilot_symbols
        self.local_parameters = ["pilot_symbol", "pilot_interval"]

    def implementation(
        self,
        input_symbols: Signal,
        /,
        pilot_symbol: tf.Tensor,
        pilot_interval: tf.Tensor,
    ) -> Signal:
        assert input_symbols.samples.dtype is tf.complex64

        signal_shape = tf.shape(input_symbols.samples)
        batch_size = signal_shape[0]
        pilot_symbol_size = tf.size(pilot_symbol)

        # The idea here is to split the samples into groups of pilot_interval + pilot_symbol_size
        # and then slice each group to isolate the pilot symbols
        # We will then compute the phase error between the expected and the received pilot symbols

        # We might have to pad the samples to make it a multiple of pilot_interval + pilot_symbol_size
        n = pilot_interval + pilot_symbol_size
        padding_size = (n - (signal_shape[-1] % n)) % n

        samples = tf.pad(input_symbols.samples, [[0, 0], [0, padding_size]])
        samples = tf.reshape(samples, [batch_size, -1, n])

        extracted_pilot_symbols = samples[:, :, :pilot_symbol_size]

        if not self.keep_pilot_symbols:
            samples = samples[:, :, pilot_symbol_size:]

        samples = tf.reshape(samples, [batch_size, -1])
        samples = samples[:, :tf.shape(samples)[-1] - padding_size]

        phase_error = tf.repeat(
            tf.math.reduce_mean(
                tf.math.angle(extracted_pilot_symbols) - tf.math.angle(pilot_symbol)[None, None, :],
                axis=-1
            ),
            pilot_symbol_size * tf.shape(samples)[-1] // (tf.size(extracted_pilot_symbols) // batch_size),
            axis=-1
        )

        phase_corrected_symbols = samples * tf.complex(
            tf.cos(-phase_error),
            tf.math.sin(-phase_error)
        )

        return Signal(
            samples=phase_corrected_symbols,
            timing=input_symbols.timing
        )
