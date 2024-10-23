# -*- coding: utf-8 -*-

from typing import Optional

import tensorflow as tf
from .datatypes import bit_sequence_spec, bit_dtype
from .component import Component

__all__ = ["BERTester"]


class BERTester(Component):
    #: System to test
    system: Component

    #: Random number generator
    rng: tf.random.Generator

    #: Number of iterations for testing
    n_iterations: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        system: Component,
        rng: Optional[tf.random.Generator] = None,
        n_iterations: int | tf.Tensor = 100,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.system = system

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.n_iterations = tf.Variable(tf.constant(n_iterations))
        self.local_parameters = ["n_iterations"]

    @tf.function(input_signature=(
        bit_sequence_spec,
        tf.TensorSpec(shape=[None, None, None], dtype=bit_dtype, name="output")
    ))
    def _compute_ber(self, expected_output: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
        expected_output = tf.expand_dims(expected_output, 1)

        correct_counts = tf.cast(
            tf.math.count_nonzero(tf.math.equal(expected_output, output), axis=[0, -1]),
            tf.float32
        )  # shape=[parameter_dims]

        batch_size = tf.shape(output, out_type=tf.int64)[0]
        bit_count = tf.shape(output, out_type=tf.int64)[-1]

        total_bits = tf.cast(batch_size * bit_count, tf.float32)

        return (total_bits - correct_counts) / total_bits

    def call(self, sequence_length: tf.Tensor, /, tracing_level: int = 0, **kwargs) -> tf.Tensor:
        n_iterations = self.get_component_parameter(kwargs, "n_iterations")

        inputs = self.rng.uniform(
            (n_iterations, sequence_length,),
            minval=0,
            maxval=2,
            dtype=tf.int32
        )

        kwargs["batch_size"] = n_iterations

        return self._compute_ber(
            inputs,
            self.system(inputs, tracing_level=tracing_level + 1, **kwargs)
        )
