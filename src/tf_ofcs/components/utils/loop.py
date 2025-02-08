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

from typing import Any, Optional
import tensorflow as tf
from ..component import Component

__all__ = ["Loop"]


class Loop(Component):
    """Repeatedly passes an input through a specified component `n` times."""

    #: Component to run
    component: Component

    #: Number of loop iterations
    n: tf.Variable  # shape=[]; dtype=tf.int32

    def __init__(
        self,
        component: Component,
        n: int | tf.Tensor,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.component = component
        self.n = tf.Variable(tf.constant(n), name=f"{self.name}__n")
        self.local_parameters = ["n"]

    def call(self, inp: Any, /, **kwargs) -> Any:
        n = self.get_component_parameter(kwargs, "n")

        for _i in tf.range(n):
            inp = self.component(inp, **kwargs)

        return inp
