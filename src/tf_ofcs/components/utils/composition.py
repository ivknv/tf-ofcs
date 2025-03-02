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

from typing import Any, Optional
from ..component import Component

__all__ = ["Composition"]


class Composition(Component):
    #: System components
    components: list[Component]

    def __init__(self, components: list[Component], name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.components = components

    def call(self, *args, tracing_level: int = 0, **kwargs) -> Any:
        if not self.components:
            return

        result: Any = args[0] if args else None

        for component in self.components:
            result = component(result, tracing_level=tracing_level + 1, **kwargs)

        return result
