"""
Module to do type checking on dataclasses and function calls.
(c) 2021 d373c7
"""


import inspect
import typing
from functools import wraps


def _find_type_origin(type_hint):
    if isinstance(type_hint, typing._SpecialForm):
        # case of typing.Any, typing.ClassVar, typing.Final, typing.Literal,
        # typing.NoReturn, typing.Optional, or typing.Union without parameters
        return

    actual_type = typing.get_origin(type_hint) or type_hint  # requires Python 3.8
    go = typing.get_args(type_hint)
    if isinstance(actual_type, typing._SpecialForm):
        # case of typing.Union[…] or typing.ClassVar[…] or …
        for origins in map(_find_type_origin, typing.get_args(type_hint)):
            yield from origins
    else:
        yield actual_type


def _check_types(parameters, hints):
    for name, value in parameters.items():
        type_hint = hints.get(name, typing.Any)
        actual_types = tuple(_find_type_origin(type_hint))
        if actual_types and not isinstance(value, actual_types):
            raise TypeError(
                    f"Expected type '{type_hint}' for argument '{name}'"
                    f" but received type '{type(value)}' instead"
            )
        if actual_types and actual_types[0] == list:
            # We have a list and can check the content of the list
            type_args = typing.get_args(type_hint)
            if len(value) > 0 and not isinstance(value[0], type_args):
                raise TypeError(
                    f"Expected List to contain type '{type_args}' for argument '{name}'"
                    f" but contained type '{type(value[0])}' instead"
                )


def enforce_types(a_callable):
    def decorate(func):
        hints = typing.get_type_hints(func)
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            parameters = dict(zip(signature.parameters, args))
            parameters.update(kwargs)
            _check_types(parameters, hints)

            return func(*args, **kwargs)
        return wrapper

    if inspect.isclass(a_callable):
        a_callable.__init__ = decorate(a_callable.__init__)
        return a_callable

    return decorate(a_callable)


def enforce_strict_types(a_callable):
    def decorate(func):
        hints = typing.get_type_hints(func)
        signature = inspect.signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = dict(zip(signature.parameters, bound.args))
            parameters.update(bound.kwargs)
            _check_types(parameters, hints)

            return func(*args, **kwargs)
        return wrapper

    if inspect.isclass(a_callable):
        a_callable.__init__ = decorate(a_callable.__init__)
        return a_callable

    return decorate(a_callable)
