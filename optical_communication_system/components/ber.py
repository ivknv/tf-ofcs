# -*- coding: utf-8 -*-

from collections.abc import Callable, Sequence
from typing import Optional

import tensorflow as tf
from .datatypes import bit_sequence_spec

__all__ = ["BERAnalyzer", "BERTester"]


class BERAnalyzer(tf.Module):
    @tf.function(input_signature=(
        bit_sequence_spec,
        bit_sequence_spec
    ))
    def __call__(self, expected_output: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_equal(tf.size(output), tf.size(expected_output))

        correct_count = tf.cast(tf.math.count_nonzero(expected_output == output), tf.float32)
        total_bits = tf.cast(tf.size(output), tf.float32)

        return (total_bits - correct_count) / total_bits


class BERTester(tf.Module):
    #: System to test
    systems: list[Callable]

    #: Random number generator
    rng: tf.random.Generator

    def __init__(
        self,
        systems: Sequence[Callable],
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.systems = list(systems)

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.ber_analyzer = BERAnalyzer()

    @tf.function(input_signature=(
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.int32)
    ))
    def __call__(self, sequence_length: tf.Tensor, n_iterations: tf.Tensor) -> tf.Tensor:
        inputs = tf.map_fn(
            fn=lambda i: self.rng.uniform(shape=(sequence_length,), minval=0, maxval=2, dtype=tf.int32),
            elems=tf.range(n_iterations)
        )

        bers = tf.zeros(shape=(0,), dtype=tf.float32)

        for system in self.systems:
            def fn(system):
                return lambda inp: self.ber_analyzer(inp, system(inp))

            bers = tf.concat(
                [
                    bers,
                    [tf.reduce_sum(tf.vectorized_map(fn(system), inputs))]
                ],
                0
            )

        return bers / tf.cast(n_iterations, tf.float32)
