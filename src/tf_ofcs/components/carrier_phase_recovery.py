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

from typing import Optional
import tensorflow as tf

from ..datatypes import Signal, TensorCompatible
from .component import Component
from ..utils import to_tensor

__all__ = ["DecisionDirectedPhaseRecovery"]


class DecisionDirectedPhaseRecovery(Component):
    """
        Decision-directed carrier phase recovery. Updates phase error
        every `window_length` symbols.

        :ivar window_length` `shape=[]; dtype=tf.int32`, length of the window
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

        sample_count = tf.shape(input_symbols.samples)[-1]

        # Slice the symbols
        sliced_symbols = tf.signal.frame(input_symbols.samples, window_length, window_length, axis=-1)

        phase_error = tf.constant(0.0)
        phase_offsets = tf.TensorArray(dtype=tf.float32, size=sample_count // window_length)

        for i in tf.range(sample_count // window_length):
            symbols = sliced_symbols[:, i]
            symbols *= tf.math.exp(tf.complex(0.0, -phase_error))

            best_indices = tf.cast(
                tf.argmin(
                    tf.abs(mapping - tf.expand_dims(symbols, -1)),
                    axis=-1
                ),
                tf.int32
            )

            expected_phase = tf.math.angle(tf.gather(mapping, best_indices))
            phase_error += tf.math.reduce_mean(tf.math.angle(symbols) - expected_phase)
            phase_offsets = phase_offsets.write(i, phase_error)

        return Signal(
            samples=input_symbols.samples * tf.math.exp(
                tf.repeat(tf.complex(0.0, -phase_offsets.stack()), window_length)
            ),
            timing=input_symbols.timing
        )
