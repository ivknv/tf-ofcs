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
    parameters: list[str]

    trace_debug_enabled_by_default: bool = False

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.output = None
        self.trace_debug_enabled = self.trace_debug_enabled_by_default
        self.parameters = []

    @tf.function
    def __call__(self, inp: Any, **kwargs) -> Any:
        component_kwargs = {}

        for parameter in self.parameters:
            component_kwargs[parameter] = self.get_component_parameter(kwargs, parameter)

        return self.implementation(inp, **component_kwargs)

    @tf.function
    def implementation(self, inp: Any, **kwargs) -> Any:
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

    def trace(self, inp: Any, level: int = 0, **kwargs) -> Any:
        self.trace_call(inp, level)
        self.output = self.__call__(inp, **kwargs)
        self.trace_return_value(self.output, level + 1)

        return self.output

    def find_child(self, name: str) -> "Component":
        for module in self.submodules:
            if module.name == name:
                if not isinstance(module, Component):
                    raise ValueError(f"Found {repr(name)}, but it's not an instance of Component")

                return module

        raise KeyError(f"No subcomponent with name {repr(name)}")

    def get_kwarg(
        self,
        kwargs: dict[str, Any],
        kwarg_name: str,
        default: Optional[Any] = None
    ) -> Any:
        return kwargs.get(f"{self.name}__{kwarg_name}", default)

    def get_component_parameter(self, kwargs: dict[str, Any], parameter: str):
        value = self.get_kwarg(kwargs, parameter)

        if value is not None:
            return value

        return getattr(self, parameter)
