"""Shared coordinates for the AscendStore v1 domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenRange:
    """A half-open token range ``[start_token, end_token)``."""

    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.start_token < 0:
            raise ValueError(f"Token range start must be non-negative, got {self.start_token}")
        if self.end_token < self.start_token:
            raise ValueError(f"Token range end {self.end_token} precedes start {self.start_token}")
