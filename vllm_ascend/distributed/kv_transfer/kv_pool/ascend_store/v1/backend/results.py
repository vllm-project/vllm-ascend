"""Validate integer result shapes; each adapter interprets its native values."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any


class BackendResultError(RuntimeError):
    """A native operation did not return the v1 evidence shape it promises."""


def integer_results(operation: str, keys: list[str], results: Iterable[int] | None) -> tuple[int, ...]:
    try:
        raw_values = tuple(results) if results is not None else ()
    except (TypeError, ValueError) as error:
        raise BackendResultError(f"{operation} returned non-integer results") from error
    if len(raw_values) != len(keys):
        raise BackendResultError(f"{operation} returned {len(raw_values)} results for {len(keys)} keys")
    converted: list[int] | None = None
    for index, value in enumerate(raw_values):
        if type(value) is not int:
            if not is_integer_result(value):
                raise BackendResultError(f"{operation} returned non-integer results")
            if converted is None:
                converted = list(raw_values)
            converted[index] = int(value)
    return raw_values if converted is None else tuple(converted)


def integer_result(operation: str, value: Any) -> int:
    if not is_integer_result(value):
        raise BackendResultError(f"{operation} returned a non-integer result")
    return int(value)


def presence_results(operation: str, keys: list[str], results: Iterable[int] | None) -> tuple[bool, ...]:
    codes = integer_results(operation, keys, results)
    if any(code not in (0, 1) for code in codes):
        raise BackendResultError(f"{operation} returned object states other than 0 or 1")
    return tuple(code == 1 for code in codes)


def is_integer_result(value: Any) -> bool:
    return type(value) is int or (not isinstance(value, bool) and isinstance(value, Integral))
