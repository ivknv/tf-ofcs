# -*- coding: utf-8 -*-

from typing import Any, Optional
from ..component import Component

__all__ = ["Reindex"]


class Reindex(Component):
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
