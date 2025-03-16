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

from ..component import Component
from typing import Any, Callable, Optional

__all__ = ["Substitute"]


class Substitute(Component):
    """
        Takes a component as a parameter, allows to replace it with any callable.

        :ivar component: Component to substitute
        :ivar substitute: Callable to be used a substitution for the component (`component` by default)
    """

    component: Component
    substitute: Callable

    def __init__(
        self,
        component: Component,
        substitute: Optional[Callable] = None,
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.component = component

        if substitute is None:
            substitute = component.__call__

        self.substitute = substitute

    def __call__(self, *args, **kwargs) -> Any:
        return self.substitute(*args, **kwargs)
