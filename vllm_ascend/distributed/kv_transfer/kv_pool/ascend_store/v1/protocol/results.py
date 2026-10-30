"""Operation results independent of concrete Workers and execution resources."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LoadFailureLocation:
    group_id: int
    block_id: int


@dataclass(frozen=True, slots=True)
class LoadResult:
    completed_request_ids: frozenset[str]
    failed_request_ids: frozenset[str]
    failed_block_ids: frozenset[int]
    failed_locations: tuple[LoadFailureLocation, ...] = ()


@dataclass(frozen=True, slots=True)
class StoreResult:
    request_id: str
    store_job_id: int | None
    keys: tuple[str, ...] = ()
    result_codes: tuple[int | None, ...] = ()
    succeeded: bool = True
    source_release_confirmed: bool = True
    error: Exception | None = None


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Report whether all local memory access has ended and preserve any drain error.

    An error can coexist with safe release, such as a result-interpretation error
    after normal Backend return. Unknown memory access keeps safe_to_release false.
    """

    safe_to_release: bool
    error: BaseException | None = None
