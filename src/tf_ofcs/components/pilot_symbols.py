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

from .component import Component

from ..datatypes import Signal, Timing

__all__ = ["InsertPilotSymbols", "RemovePilotSymbols"]


class InsertPilotSymbols(Component):
    """
        Inserts pilot symbols into a signal at regular intervals.

        :param pilot_symbol: `shape=[None], dtype=tf.float32 or tf.complex64`,
            pilot symbol to be inserted.
        :param pilot_interval: `shape=[], dtype=tf.int32`, interval (in samples)
            at which to insert the pilot symbols.
    """

    pilot_interval: tf.Variable  # shape=[]; dtype=tf.int32
    pilot_symbol: tf.Variable  # shape=[None]; dtype=tf.float32 or tf.complex64

    def __init__(
        self,
        pilot_symbol: Sequence[float] | Sequence[complex] | tf.Tensor,
        pilot_interval: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.pilot_symbol = tf.Variable(
            tf.convert_to_tensor(pilot_symbol),  # type: ignore[arg-type]
            name=f"{self.name}__pilot_symbol"
        )
        self.pilot_interval = tf.Variable(
            tf.convert_to_tensor(pilot_interval),
            name=f"{self.name}__pilot_interval"
        )

        self.local_parameters = ["pilot_symbol", "pilot_interval"]

    def implementation(
        self,
        input_signal: Signal,
        pilot_symbol: tf.Tensor,
        pilot_interval: tf.Tensor
    ) -> Signal:
        signal_shape = tf.shape(input_signal.samples)
        batch_size = signal_shape[0]
        sample_count = signal_shape[-1]
        pilot_sample_count = tf.shape(pilot_symbol)[-1]

        # Calculate number of pilot insertions needed
        num_pilot_insertions = (sample_count + pilot_interval - 1) // pilot_interval

        # Calculate new indices for original signal elements
        original_indices = tf.range(sample_count, dtype=tf.int32)
        chunk_numbers = original_indices // pilot_interval
        offset = pilot_sample_count * (chunk_numbers + 1)
        new_indices_original = original_indices + offset

        # Calculate indices for pilot symbols
        pilot_start_positions = (pilot_interval + pilot_sample_count) * tf.range(num_pilot_insertions, dtype=tf.int32)
        pilot_indices = tf.reshape(
            pilot_start_positions[:, None] + tf.range(pilot_sample_count, dtype=tf.int32),
            [-1]
        )

        # Create indices and values for scatter operation for original signal elements
        batch_indices_orig = tf.repeat(tf.range(batch_size), sample_count)
        elem_indices_orig = tf.tile(new_indices_original, [batch_size])
        indices_orig = tf.stack([batch_indices_orig, elem_indices_orig], axis=1)
        values_orig = tf.reshape(input_signal.samples, [-1])

        # For pilot symbols
        batch_indices_pilot = tf.repeat(tf.range(batch_size), pilot_sample_count * num_pilot_insertions)
        elem_indices_pilot = tf.tile(pilot_indices, [batch_size])
        indices_pilot = tf.stack([batch_indices_pilot, elem_indices_pilot], axis=1)
        values_pilot = tf.tile(pilot_symbol, [batch_size * num_pilot_insertions])

        # Combine indices and values
        all_indices = tf.concat([indices_orig, indices_pilot], axis=0)
        all_values = tf.concat([values_orig, values_pilot], axis=0)

        # Calculate final output shape and scatter values
        output_length = sample_count + pilot_sample_count * num_pilot_insertions
        samples_with_pilot_symbols = tf.scatter_nd(
            indices=all_indices,
            updates=all_values,
            shape=[batch_size, output_length]
        )

        return Signal(
            samples=samples_with_pilot_symbols,
            timing=Timing(
                sample_count=tf.shape(samples_with_pilot_symbols)[-1],
                sampling_rate=input_signal.timing.sampling_rate
            )
        )


class RemovePilotSymbols(Component):
    """
        Removes pilot symbols from a signal at regular intervals.

        :param pilot_symbol_size: `shape=[], dtype=tf.int32`,
            size of the pilot symbols.
        :param pilot_interval: `shape=[], dtype=tf.int32`, interval (in samples)
            at which the pilot symbols were previously inserted
    """

    pilot_interval: tf.Variable  # shape=[]; dtype=tf.int32
    pilot_symbol_size: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        pilot_symbol_size: int | tf.Tensor,
        pilot_interval: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.pilot_symbol_size = tf.Variable(
            tf.convert_to_tensor(pilot_symbol_size),
            name=f"{self.name}__pilot_symbol_size"
        )
        self.pilot_interval = tf.Variable(
            tf.convert_to_tensor(pilot_interval),
            name=f"{self.name}__pilot_interval"
        )

        self.local_parameters = ["pilot_symbol_size", "pilot_interval"]

    def implementation(
        self,
        input_signal: Signal,
        pilot_symbol_size: tf.Tensor,
        pilot_interval: tf.Tensor
    ) -> Signal:
        signal_shape = tf.shape(input_signal.samples)
        batch_size = signal_shape[0]

        # The idea here is to split the samples into groups of pilot_interval + pilot_symbol_size
        # and then slice each group to remove the pilot symbols

        # We might have to pad the samples to make it a multiple of pilot_interval + pilot_symbol_size
        n = pilot_interval + pilot_symbol_size
        padding_size = (n - (signal_shape[-1] % n)) % n

        samples = tf.pad(input_signal.samples, [[0, 0], [0, padding_size]])
        samples = tf.reshape(samples, [batch_size, -1, pilot_interval + pilot_symbol_size])
        samples = samples[:, :, pilot_symbol_size:]
        samples = tf.reshape(samples, [batch_size, -1])
        samples = samples[:, :tf.shape(samples)[-1] - padding_size]

        return Signal(
            samples=samples,
            timing=Timing(
                sample_count=tf.shape(samples)[-1],
                sampling_rate=input_signal.timing.sampling_rate
            )
        )
