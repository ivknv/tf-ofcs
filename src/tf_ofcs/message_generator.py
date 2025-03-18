# -*- coding: utf-8
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


from collections.abc import Generator
from typing import Optional
import tensorflow as tf

__all__ = ["message_generator"]


def message_generator(
    M: int,
    batch_size: int,
    rng: Optional[tf.random.Generator] = None
) -> Generator[tuple[tf.Tensor, tf.Tensor], None, None]:
    """
        Generates batches of input and expected output messages.
        Output messages match the inputs.

        The output signature is
        `(tf.TensorSpec(shape=(batch_size, M), dtype=tf.float32),
         tf.TensorSpec(shape=(batch_size, M), dtype=tf.float32))`.

        :param M: `int`, alphabet size (determines the length of each one-hot vector)
        :param batch_size: `int`, batch size
        :param rng: `tf.random.Generator`, random number generator

        :returns: `Generator[tuple[tf.Tensor, tf.Tensor], None, None]`,
                  batches of one-hot input/output messages
    """

    if rng is None:
        rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

    while True:
        message_batch = tf.one_hot(
            rng.uniform([batch_size], minval=0, maxval=M, dtype=tf.int32),  # pyright: ignore[reportOptionalMemberAccess]
            M, dtype=tf.float32
        )

        yield message_batch, message_batch
