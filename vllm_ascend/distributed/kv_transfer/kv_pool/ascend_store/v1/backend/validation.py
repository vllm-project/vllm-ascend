"""Shared input validation before native object-session acquisition."""

from collections.abc import Collection


def validate_session_open(keys: list[str], object_sizes: list[int], owned: Collection[str]) -> None:
    """Reject ambiguous ownership inputs before a native operation can acquire anything."""
    if len(keys) != len(object_sizes) or len(set(keys)) != len(keys):
        raise ValueError("Session keys must be unique and aligned with object sizes")
    if any(size <= 0 for size in object_sizes):
        raise ValueError("Session object sizes must be positive")
    if any(key in owned for key in keys):
        raise RuntimeError("Previous Backend sessions have not ended for these keys")
