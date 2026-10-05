"""Validate native Backend results before they enter the v1 runtime."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any


class BackendResultError(RuntimeError):
    """A native operation did not return the v1 evidence shape it promises."""


def result_codes(operation: str, keys: list[str], results: Iterable[int] | None) -> tuple[int, ...]:
    try:
        raw_values = tuple(results) if results is not None else ()
    except (TypeError, ValueError) as error:
        raise BackendResultError(f"{operation} returned non-integer results") from error
    if len(raw_values) != len(keys):
        raise BackendResultError(f"{operation} returned {len(raw_values)} results for {len(keys)} keys")
    if any(not is_integer_result(value) for value in raw_values):
        raise BackendResultError(f"{operation} returned non-integer results")
    return tuple(int(value) for value in raw_values)


def result_code(operation: str, value: Any) -> int:
    if not is_integer_result(value):
        raise BackendResultError(f"{operation} returned a non-integer result")
    return int(value)


def presence_results(operation: str, keys: list[str], results: Iterable[int] | None) -> tuple[bool, ...]:
    codes = result_codes(operation, keys, results)
    if any(code not in (0, 1) for code in codes):
        raise BackendResultError(f"{operation} returned object states other than 0 or 1")
    return tuple(code == 1 for code in codes)


def is_integer_result(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral)
