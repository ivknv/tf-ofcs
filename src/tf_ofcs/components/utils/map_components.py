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

from collections.abc import Sequence
from typing import Optional

from ..component import Component

__all__ = ["MapComponents"]


class MapComponents(Component):
    component: Component | Sequence[Component]

    def __init__(self, component: Component | Sequence[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.component = component

    def call(
        self,
        inputs: tuple,
        /,
        tracing_level: int = 0,
        **kwargs
    ) -> tuple:
        if isinstance(self.component, Sequence):
            return tuple(
                component(
                    input,
                    tracing_level=tracing_level + 1,
                    **kwargs
                )
                for component, input in zip(self.component, inputs, strict=True)
            )
        else:
            return tuple(
                self.component(
                    input,
                    tracing_level=tracing_level + 1,
                    **kwargs
                )
                for input in inputs
            )
