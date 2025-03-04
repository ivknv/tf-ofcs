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

from ..utils import to_tensor
from .component import Component

from ..datatypes import TensorCompatible

__all__ = ["BitSequenceGenerator"]


class BitSequenceGenerator(Component):
    """
        Generates a random bit sequence of a given length and batch size.

        The generated bit sequence is a tensor of shape `[batch_size, length]`
        with elements being either 0 or 1.

        The `seed` local parameter can be used to reset the random number generator
        to a known state, `self.rng` will be reset if `seed` is not `None`.

        :ivar rng: `tf.random.Generator`, random number generator to use
        :ivar length: `shape=[]; dtype=tf.int32`, length of the bit sequence
        :ivar batch_size: `shape=[]; dtype=tf.int32`, batch size of the bit sequence
    """

    rng: tf.random.Generator
    length: tf.Variable  # shape=[]; dtype=tf.int32
    batch_size: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        length: TensorCompatible,
        batch_size: TensorCompatible = 1,
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)

        if rng is None:
            rng = tf.random.Generator.from_non_deterministic_state(alg="philox")

        self.rng = rng
        self.length = tf.Variable(to_tensor(length), name=f"{self.name}__length")
        self.batch_size = tf.Variable(to_tensor(batch_size), name=f"{self.name}__batch_size")
        self.local_parameters = ["length", "batch_size", "seed"]
        self.global_parameters = ["batch_size"]

    def implementation(
        self,
        /,
        length: tf.Tensor,
        batch_size: tf.Tensor,
        seed: Optional[tf.Tensor]
    ) -> tf.Tensor:
        if seed is not None:
            self.rng.reset_from_seed(seed)

        return self.rng.uniform([batch_size, length], maxval=2, dtype=tf.int32)
