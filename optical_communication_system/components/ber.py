# -*- coding: utf-8 -*-

from collections.abc import Sequence
from typing import Optional

import tensorflow as tf
from .datatypes import bit_sequence_spec
from .component import Component

__all__ = ["BERTester"]


class BERTester(Component):
    #: Systems to test
    systems: list[Component]

    #: Random number generator
    rng: tf.random.Generator

    #: Number of iterations for testing
    n_iterations: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        systems: Sequence[Component],
        rng: Optional[tf.random.Generator] = None,
        n_iterations: int | tf.Tensor = 100,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.systems = list(systems)

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.n_iterations = tf.Variable(tf.constant(n_iterations))
        self.parameters = ["n_iterations"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        bit_sequence_spec
    ))
    def _compute_ber(self, expected_output: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        tf.debugging.assert_equal(tf.size(output), tf.size(expected_output))

        correct_count = tf.cast(tf.math.count_nonzero(expected_output == output), tf.float32)
        total_bits = tf.cast(tf.size(output), tf.float32)

        return (total_bits - correct_count) / total_bits

    @tf.function
    def __call__(self, sequence_length: tf.Tensor, **kwargs) -> tf.Tensor:
        n_iterations = self.get_component_parameter(kwargs, "n_iterations")

        inputs = self.rng.uniform(
            (n_iterations, sequence_length,),
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        bers = tf.zeros(shape=(0,), dtype=tf.float32)

        for system in self.systems:
            bers = tf.concat(
                [
                    bers,
                    [tf.reduce_sum(self._compute_ber(inputs, system(inputs, **kwargs)))]
                ],
                0
            )

        return bers / tf.cast(n_iterations, tf.float32)
