# -*- coding: utf-8 -*-

from typing import Any, Optional
import tensorflow as tf

__all__ = ["Component"]


def _type_as_str(inp: Any) -> str:
    if isinstance(inp, tuple):
        return f"tuple[{', '.join(_type_as_str(i) for i in inp)}]"
    elif isinstance(inp, tf.Tensor):
        return f"tf.Tensor(shape={tf.shape(inp)}, dtype={repr(inp.dtype)})"
    else:
        return type(inp).__name__


class Component(tf.Module):
    output: Any
    trace_debug_enabled: bool

    trace_debug_enabled_by_default: bool = False

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.output = None
        self.trace_debug_enabled = self.trace_debug_enabled_by_default

    @tf.function
    def __call__(self, inp: Any) -> Any:
        raise NotImplementedError

    def _padding(self, level: int) -> str:
        return "|---" * level

    def trace_call(self, inp: Any, level: int = 0) -> None:
        if self.trace_debug_enabled:
            component = f"{self.__class__.__name__}(name={repr(self.name)})"
            padding = self._padding(level)

            print(f"{padding}Calling {component} with args {_type_as_str(inp)}")

    def trace_return_value(self, output: Any, level: int = 0) -> None:
        if self.trace_debug_enabled:
            padding = self._padding(level)

            print(f"{padding}Return {_type_as_str(output)}")

    def trace(self, inp: Any, level: int = 0) -> Any:
        self.trace_call(inp, level)
        self.output = self.__call__(inp)
        self.trace_return_value(self.output, level + 1)

        return self.output

    def find_child(self, name: str) -> "Component":
        for module in self.submodules:
            if module.name == name:
                if not isinstance(module, Component):
                    raise ValueError(f"Found {repr(name)}, but it's not an instance of Component")

                return module

        raise KeyError(f"No subcomponent with name {repr(name)}")
