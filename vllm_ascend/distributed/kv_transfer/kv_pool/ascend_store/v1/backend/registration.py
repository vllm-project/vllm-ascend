"""Own native buffer registrations for the AscendStore v1 runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class Registration(Protocol):
    """A retryable owner whose close confirms that every region was released."""

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BufferRegion:
    """One exact native memory registration owned by a v1 Backend."""

    address: int
    size: int


class BufferReleaseError(RuntimeError):
    """A registration still owns regions after a release attempt."""

    def __init__(self, failures: tuple[tuple[BufferRegion | None, BaseException], ...]) -> None:
        self.failures = failures
        owners = ", ".join(
            "pending registration" if region is None else f"({region.address}, {region.size})" for region, _ in failures
        )
        super().__init__(f"Failed to release {len(failures)} buffer registration owner(s): {owners}")


class BufferRegistration:
    """Idempotent owner for successfully registered native memory regions.

    A region is acquired only after its native registration succeeds. ``close``
    releases in reverse order and retains failed releases for an explicit retry.
    """

    def __init__(
        self,
        release_region: Callable[[int, int], None] | None = None,
        *,
        before_release: Callable[[], None] | None = None,
    ) -> None:
        self._release_region = release_region
        self._before_release = before_release
        self._before_release_complete = before_release is None
        self._regions: list[BufferRegion] = []
        self._accepting_acquisitions = True

    @property
    def regions(self) -> tuple[BufferRegion, ...]:
        return tuple(self._regions)

    @property
    def released(self) -> bool:
        return self._before_release_complete and not self._regions

    def acquire(self, address: int, size: int) -> None:
        if not self._accepting_acquisitions:
            raise RuntimeError("A closing buffer registration cannot acquire new regions")
        if self._release_region is None:
            raise RuntimeError("A no-op buffer registration cannot acquire native regions")
        if address < 0 or size <= 0:
            raise ValueError(f"Invalid buffer region: address={address}, size={size}")
        self._regions.append(BufferRegion(address, size))

    def close(self) -> None:
        self._accepting_acquisitions = False
        failures: list[tuple[BufferRegion | None, BaseException]] = []
        if not self._before_release_complete:
            assert self._before_release is not None
            try:
                self._before_release()
            except BaseException as error:
                failures.append((None, error))
            else:
                self._before_release_complete = True

        remaining: list[BufferRegion] = []
        if self._release_region is not None:
            for region in reversed(self._regions):
                try:
                    self._release_region(region.address, region.size)
                except BaseException as error:
                    failures.append((region, error))
                    remaining.append(region)
        elif self._regions:
            raise RuntimeError("Buffer registration owns regions without a release operation")
        self._regions = list(reversed(remaining))
        if failures:
            raise BufferReleaseError(tuple(failures))


class BufferRegistrationError(RuntimeError):
    """Registration failed and rollback left an owner that must be retried."""

    def __init__(
        self,
        operation_error: BaseException,
        rollback_error: BaseException,
        registration: Registration,
    ) -> None:
        self.operation_error = operation_error
        self.rollback_error = rollback_error
        self.registration = registration
        super().__init__(
            "Buffer registration failed and rollback is incomplete: "
            f"operation={type(operation_error).__name__}: {operation_error}; "
            f"rollback={type(rollback_error).__name__}: {rollback_error}"
        )


def rollback_buffer_registration(
    registration: BufferRegistration,
    operation_error: BaseException,
) -> None:
    """Rollback a partial registration while retaining failed ownership."""

    try:
        registration.close()
    except BaseException as rollback_error:
        raise BufferRegistrationError(operation_error, rollback_error, registration) from operation_error
