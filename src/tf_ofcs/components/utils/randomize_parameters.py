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

from collections.abc import Mapping, Sequence
from typing import Any, Optional
from ..component import Component

import tensorflow as tf

__all__ = ["RandomizeParameters"]


class RandomizeParameters(Component):
    shape: tf.Variable  # shape=[None]; dtype=tf.int32
    distributions: Mapping[str, Mapping[str, Any]]
    rng: tf.random.Generator

    def __init__(
        self,
        next_component: Component,
        shape: Sequence[int] | tf.Tensor,
        distributions: Mapping[str, Mapping[str, Any]],
        rng: Optional[tf.random.Generator] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.next_component = next_component
        self.shape = tf.Variable(tf.constant(shape), name=f"{self.name}__shape")
        self.distributions = distributions

        if rng is None:
            self.rng = tf.random.Generator.from_non_deterministic_state(alg="philox")
        else:
            self.rng = rng

        self.local_parameters = ["shape"]

    def call(self, inp: Any, /, tracing_level: int = 0, **kwargs) -> Any:
        shape = self.get_component_parameter(kwargs, "shape")

        for parameter, distribution in self.distributions.items():
            if distribution["type"] == "normal":
                    kwargs[parameter] = self.rng.normal(
                        shape,
                        stddev=distribution["stddev"],
                        mean=distribution["mean"],
                        dtype=distribution["dtype"]
                    )
            elif distribution["type"] == "uniform":
                kwargs[parameter] = self.rng.uniform(
                    shape,
                    minval=distribution["minval"],
                    maxval=distribution["maxval"],
                    dtype=distribution["dtype"]
                )
            else:
                 raise ValueError(f"Unknown distribution type: {distribution['type']}")

        return self.next_component(inp, tracing_level=tracing_level + 1, **kwargs)
