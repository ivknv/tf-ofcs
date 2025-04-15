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
import numpy as np
from .component import Component
from ..datatypes import TensorCompatible
from ..utils import to_tensor, bits_to_indices

__all__ = ["ConfusionMatrixAnalyzer"]


class ConfusionMatrixAnalyzer(Component):
    #: System to test
    system: Component

    #: Random number generator
    rng: tf.random.Generator

    #: Batch size for test bit sequence
    batch_size: tf.Variable  # shape=[]; dtype=tf.int32

    #: Number of iterations for testing
    n_iterations: tf.Variable  # shape=[]; dtype=tf.int32

    #: Number of bits per symbol
    bits_per_symbol: int

    def __init__(
        self,
        system: Component,
        rng: Optional[tf.random.Generator] = None,
        batch_size: TensorCompatible = 100,
        n_iterations: TensorCompatible = 1,
        bits_per_symbol: int = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.system = system

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.bits_per_symbol = bits_per_symbol

        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")
        self.n_iterations = tf.Variable(to_tensor(n_iterations), name=f"{self.name}__n_iterations")
        self.local_parameters = ["batch_size", "n_iterations"]

    def call(self, sequence_length: tf.Tensor, /, tracing_level: int = 0, **kwargs) -> tf.Tensor:
        batch_size = self.get_component_parameter(kwargs, "batch_size")
        n_iterations = self.get_component_parameter(kwargs, "n_iterations")

        kwargs["batch_size"] = batch_size

        return self._loop(sequence_length, n_iterations=n_iterations, **kwargs)

    def _loop(
        self,
        sequence_length: tf.Tensor,
        /,
        n_iterations: tf.Tensor,
        batch_size: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        symbol_set_size = 2 ** self.bits_per_symbol
        confusion_matrix = np.zeros([symbol_set_size, symbol_set_size], dtype=np.int64)

        for _ in range(int(n_iterations)):
            inputs = self.rng.uniform(
                (batch_size, sequence_length,),
                minval=0,
                maxval=2,
                dtype=tf.int32
            )
            output = self.system(inputs, **kwargs)

            input_symbols = bits_to_indices(
                tf.reshape(inputs, [tf.size(inputs)]),
                tf.cast(self.bits_per_symbol, tf.int32)
            ).numpy()
            output_symbols = bits_to_indices(
                tf.reshape(output, [tf.size(output)]),
                tf.cast(self.bits_per_symbol, tf.int32)
            ).numpy()

            for i in range(input_symbols.size):
                input_symbol = input_symbols[i]
                output_symbol = output_symbols[i]

                confusion_matrix[input_symbol, output_symbol] += 1

        return tf.convert_to_tensor(confusion_matrix)
