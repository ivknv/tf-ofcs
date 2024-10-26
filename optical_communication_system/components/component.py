# -*- coding: utf-8 -*-

from typing import Any, Optional
import tensorflow as tf

from ..datatypes import ElectricalSignal, OpticalSignal

__all__ = ["Component"]


def _type_as_str(inp: Any) -> str:
    if isinstance(inp, tuple):
        return f"tuple[{', '.join(_type_as_str(i) for i in inp)}]"
    elif isinstance(inp, tf.Tensor):
        return f"tf.Tensor(shape={tf.shape(inp)}, dtype={repr(inp.dtype)})"
    elif isinstance(inp, (ElectricalSignal, OpticalSignal)):
        return f"{type(inp).__name__}{tf.shape(inp.samples)}"
    else:
        return type(inp).__name__


class Component(tf.Module):
    output: Any
    trace_debug_enabled: bool

    #: Parameters that can be passed as {component name}__{parameter name}.
    #: Each parameter in the list specifies the parameter name and optionally
    #: the minimum tensor rank (when specified as tuple).
    #: If the minimum rank is specified, the parameter may be automatically
    #: broadcasted to the desired shape based on the `parameter_dims` value.
    local_parameters: list[str | tuple[str, int]]

    #: Parameters that can be passed normally as ({parameter name})
    global_parameters: list[str | tuple[str, int]]

    trace_debug_enabled_by_default: bool = False

    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)

        self.output = None
        self.trace_debug_enabled = self.trace_debug_enabled_by_default
        self.local_parameters = []
        self.global_parameters = []

    def __call__(self, inp: Any, /, **kwargs) -> Any:
        if kwargs.get("tracing", False):
            tracing_level = kwargs.get("tracing_level", 0)

            self.trace_call(inp, tracing_level)
            self.output = self.call(inp, **kwargs)
            self.trace_return_value(self.output, tracing_level + 1)

            return self.output
        else:
            return self.call(inp, **kwargs)

    def call(self, inp: Any, /, **kwargs):
        component_kwargs = self.extract_component_parameters(kwargs)
        return self.implementation(inp, **component_kwargs)

    def implementation(self, inp: Any, /, *args, **kwargs) -> Any:
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

    def trace(self, inp: Any, /, **kwargs) -> Any:
        return self.__call__(inp, tracing=True, **kwargs)

    def find_child(self, name: str) -> "Component":
        for module in self.submodules:
            if module.name == name:
                if not isinstance(module, Component):
                    raise ValueError(f"Found {repr(name)}, but it's not an instance of Component")

                return module

        raise KeyError(f"No subcomponent with name {repr(name)}")

    def get_local_kwarg(
        self,
        kwargs: dict[str, Any],
        kwarg_name: str,
        default: Optional[Any] = None
    ) -> Any:
        return kwargs.get(f"{self.name}__{kwarg_name}", default)

    def get_component_parameter(
        self,
        kwargs: dict[str, Any],
        parameter: str,
        default: Optional[tf.Tensor] = None,
        global_parameter: bool = False,
        rank: int = -1,
        parameter_dims: tf.Tensor = tf.constant(1)
    ) -> Any:
        if global_parameter:
            value = kwargs.get(parameter, default)
        else:
            value = self.get_local_kwarg(kwargs, parameter, default)

        if value is None:
            value = getattr(self, parameter)

        if tf.rank(value) < tf.constant(rank + 1):
            value = tf.broadcast_to(value, tf.concat([[parameter_dims], tf.shape(value)], 0))

        return value

    def extract_component_parameters(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        parameter_dims = kwargs.get("parameter_dims", tf.constant(1))
        component_kwargs: dict[str, Any] = {}

        iterables = zip([True, False], [self.global_parameters, self.local_parameters], strict=True)

        for global_parameter, parameters in iterables:
            for parameter in parameters:
                if isinstance(parameter, tuple):
                    parameter, rank = parameter
                else:
                    parameter, rank = parameter, -1

                component_kwargs[parameter] = self.get_component_parameter(
                    kwargs,
                    parameter,
                    default=component_kwargs.get(parameter),
                    global_parameter=global_parameter,
                    rank=rank,
                    parameter_dims=parameter_dims
                )

        return component_kwargs
