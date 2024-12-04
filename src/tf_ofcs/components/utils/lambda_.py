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

from collections.abc import Callable
from typing import Any, Optional
from ..component import Component

__all__ = ["Lambda"]


class Lambda(Component):
    #: `Callable` that will be invoked
    function: Callable

    #: If `True`, keyword arguments will be passed to `function`
    allow_kwargs: bool

    def __init__(
        self,
        function: Callable,
        allow_kwargs: bool = False,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.function = function
        self.allow_kwargs = allow_kwargs

    def call(self, inp: Any, /, **kwargs) -> Any:
        if self.allow_kwargs:
            return self.function(inp, **kwargs)

        return self.function(inp)
