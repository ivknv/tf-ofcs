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

__all__ = ["Reindex"]


class Reindex(Component):
    """
        Takes a `tuple` as an input, produces a new tuple with indices as specified
        in the `indices` parameter.

        :ivar indices: `list | tuple[int, ...]`, indices of the output tuple.
                       If indices is a tuple, it represents a single index.
                       If indices is a list, it will create a tuple of indices,
                       this works recursively.

        Example 1: specifying `indices=(0, 3, 2)` is the same as `inputs[0][3][2]`.
        Example 2: specifying `indices=[(0, 1), (1, 0)]` will produce `(inputs[0][1], inputs[1][0])`.
        Example 3: specifying `indices=[[(0, 0), (1, 0)], [(1, 0), (1, 1)]]` will
        produce `((inputs[0][0], inputs[1][0]), (inputs[0][1], inputs[1][1]))`.
    """

    indices: list | tuple[int, ...]

    def __init__(
        self,
        indices: list | tuple[int, ...],
        name: Optional[str] = None
    ) -> None:
        super().__init__(name=name)

        self.indices = indices

    def _get_tuple_index(self, inputs: tuple, index: tuple[int, ...]) -> Any:
        current = inputs

        for i in index:
            current = current[i]

        return current

    def _reindex(self, inputs: tuple, index: list | tuple[int, ...]) -> Any:
        if isinstance(index, tuple):
            return self._get_tuple_index(inputs, index)
        else:
            return tuple(self._reindex(inputs, i) for i in index)

    def call(self, inputs: tuple, /, **kwargs) -> Any:
        return self._reindex(inputs, self.indices)
