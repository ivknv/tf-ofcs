# -*- coding: utf-8 -*-

from typing import Any
import tensorflow as tf

__all__ = ["Component"]


class Component(tf.Module):
    output: Any

    def __call__(self, inp: Any) -> Any:
        raise NotImplementedError

    def capture(self, inp: Any) -> Any:
        self.output = self.__call__(inp)

        return self.output

    def find_child(self, name: str) -> "Component":
        for module in self.submodules:
            if module.name == name:
                if not isinstance(module, Component):
                    raise ValueError(f"Found {repr(name)}, but it's not an instance of Component")

                return module

        raise KeyError(f"No subcomponent with name {repr(name)}")
