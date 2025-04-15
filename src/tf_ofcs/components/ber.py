# -*- coding: utf-8 -*-
# Copyright (C) 2024 Ivan Konovalov

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

from typing import Literal, Optional

import tensorflow as tf
from .component import Component
from ..datatypes import TensorCompatible
from ..utils import to_tensor

__all__ = ["BERTester"]


class BERTester(Component):
    #: System to test
    system: Component

    #: Random number generator
    rng: tf.random.Generator

    #: Batch size for test bit sequence
    batch_size: tf.Variable  # shape=[]; dtype=tf.int32

    #: Number of iterations for testing
    n_iterations: tf.Variable  # shape=[]; dtype=tf.int32

    #: Block size to consider. If `block_size > 1`, then this component
    #: computes BLER (block error rate), instead of BER (bit error rate)
    block_size: int

    def __init__(
        self,
        system: Component,
        rng: Optional[tf.random.Generator] = None,
        batch_size: TensorCompatible = 100,
        n_iterations: TensorCompatible = 1,
        block_size: int = 1,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.system = system

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.block_size = block_size

        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")
        self.n_iterations = tf.Variable(to_tensor(n_iterations), name=f"{self.name}__n_iterations")
        self.local_parameters = ["batch_size", "n_iterations"]

    def call(self, sequence_length: tf.Tensor, /, tracing_level: int = 0, **kwargs) -> tf.Tensor:
        batch_size = self.get_component_parameter(kwargs, "batch_size")
        n_iterations = self.get_component_parameter(kwargs, "n_iterations")

        kwargs["batch_size"] = batch_size

        return self._loop(sequence_length, n_iterations=n_iterations, **kwargs)

    @tf.function
    def _loop(
        self,
        sequence_length: tf.Tensor,
        /,
        n_iterations: tf.Tensor,
        batch_size: tf.Tensor,
        **kwargs
    ) -> tf.Tensor:
        correct_count = tf.constant(0.0, dtype=tf.float32)

        for _ in tf.range(n_iterations):
            inputs = self.rng.uniform(
                (batch_size, sequence_length,),
                minval=0,
                maxval=2,
                dtype=tf.int32
            )

            output = self.system(inputs, **kwargs)

            correct_count += tf.cast(
                tf.math.count_nonzero(
                    tf.math.reduce_all(
                        tf.math.equal(  # Split input bits into blocks
                            tf.signal.frame(
                                inputs,
                                frame_length=self.block_size,
                                frame_step=self.block_size,
                                axis=-1
                            ),
                            tf.signal.frame(  # Split output bits into blocks
                                output,
                                frame_length=self.block_size,
                                frame_step=self.block_size,
                                axis=-1
                            )
                        ),
                        axis=-1
                    )
                ),
                tf.float32
            )

        total_blocks = (
            tf.cast(n_iterations, tf.float32) *
            tf.cast(batch_size, tf.float32) *
            tf.cast(sequence_length, tf.float32) /
            tf.cast(self.block_size, tf.float32)
        )

        return 1.0 - correct_count / total_blocks
