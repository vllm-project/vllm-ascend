"""Process-wide Mooncake TransferEngine and exact registration leases."""

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegisteredRegion:
    address: int
    size: int


class RegistrationReleaseError(RuntimeError):
    def __init__(self, failures: tuple[tuple[RegisteredRegion, BaseException], ...]) -> None:
        self.failures = failures
        regions = ", ".join(f"({region.address}, {region.size})" for region, _ in failures)
        super().__init__(f"Failed to release {len(failures)} Mooncake registration(s): {regions}")


class RegistrationLease:
    """Retryable owner for references in the process-wide registration table."""

    def __init__(self, release_region: Callable[[int, int], None]) -> None:
        self._release_region = release_region
        self._regions: list[RegisteredRegion] = []
        self._accepting_acquisitions = True

    @property
    def regions(self) -> tuple[RegisteredRegion, ...]:
        return tuple(self._regions)

    def acquire(self, address: int, size: int) -> None:
        if not self._accepting_acquisitions:
            raise RuntimeError("A closing Mooncake registration cannot acquire regions")
        self._regions.append(RegisteredRegion(address, size))

    def close(self) -> None:
        self._accepting_acquisitions = False
        failures: list[tuple[RegisteredRegion, BaseException]] = []
        remaining: list[RegisteredRegion] = []
        for region in reversed(self._regions):
            try:
                self._release_region(region.address, region.size)
            except BaseException as error:
                failures.append((region, error))
                remaining.append(region)
        self._regions = list(reversed(remaining))
        if failures:
            raise RegistrationReleaseError(tuple(failures))


class RegistrationAcquisitionError(RuntimeError):
    """Registration failed and rollback left a retryable shared lease."""

    def __init__(
        self,
        operation_error: BaseException,
        rollback_error: BaseException,
        registration: RegistrationLease,
    ) -> None:
        self.operation_error = operation_error
        self.rollback_error = rollback_error
        self.registration = registration
        super().__init__(
            "Mooncake registration failed and rollback is incomplete: "
            f"operation={type(operation_error).__name__}: {operation_error}; "
            f"rollback={type(rollback_error).__name__}: {rollback_error}"
        )


class GlobalTE:
    """Own the one process-wide engine and coordinate all memory users."""

    def __init__(self) -> None:
        self.transfer_engine = None
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.RLock()
        self._registered_regions: dict[int, tuple[int, int]] = {}
        self._classic_registration: RegistrationLease | None = None

    @property
    def is_register_buffer(self) -> bool:
        """Preserve the classic caller's first-registration observation."""

        return self._classic_registration is not None

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as error:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                            "to run vLLM with MooncakeConnector."
                        ) from error
                    transfer_engine = TransferEngine()
                    native_result = transfer_engine.initialize(
                        hostname,
                        "P2PHANDSHAKE",
                        "ascend",
                        device_name if device_name is not None else "",
                    )
                    if native_result != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {native_result}")
                    # Publish only a fully initialized process-wide engine.
                    self.transfer_engine = transfer_engine
        return self.transfer_engine

    def register_buffer(self, ptrs: list[int], sizes: list[int]) -> None:
        """Keep the legacy first registration alive for process lifetime."""

        with self.register_buffer_lock:
            if self._classic_registration is not None:
                return
            try:
                self._classic_registration = self.acquire_registration(ptrs, sizes)
            except RegistrationAcquisitionError as error:
                # Classic callers have no returned owner. Keep any region that
                # could not be rolled back process-owned instead of losing its
                # only retry/lifetime record while the source memory is live.
                self._classic_registration = error.registration
                raise

    def acquire_registration(self, ptrs: list[int], sizes: list[int]) -> RegistrationLease:
        """Acquire exact, reference-counted regions for an explicit owner."""

        if len(ptrs) != len(sizes):
            raise ValueError(f"ptrs and sizes must have the same length: {len(ptrs)} != {len(sizes)}")
        registration = RegistrationLease(self._release_region)
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            try:
                for ptr, size in zip(ptrs, sizes, strict=True):
                    self._acquire_region(ptr, size)
                    registration.acquire(ptr, size)
            except BaseException as operation_error:
                try:
                    registration.close()
                except BaseException as rollback_error:
                    raise RegistrationAcquisitionError(
                        operation_error,
                        rollback_error,
                        registration,
                    ) from operation_error
                raise
        return registration

    def _acquire_region(self, ptr: int, size: int) -> None:
        if ptr < 0 or size <= 0:
            raise ValueError(f"Invalid Mooncake region: address={ptr}, size={size}")
        current = self._registered_regions.get(ptr)
        if current is not None:
            registered_size, ref_count = current
            if registered_size != size:
                raise ValueError(f"Mooncake address {ptr} is registered with size {registered_size}, not {size}")
            self._registered_regions[ptr] = (registered_size, ref_count + 1)
            return
        transfer_engine = self.transfer_engine
        assert transfer_engine is not None, "Transfer engine must be initialized"
        native_result = transfer_engine.register_memory(ptr, size)
        if native_result != 0:
            raise RuntimeError(f"Mooncake memory registration failed: ptr={ptr}, size={size}, ret={native_result}")
        self._registered_regions[ptr] = (size, 1)

    def _release_region(self, ptr: int, size: int) -> None:
        with self.register_buffer_lock:
            current = self._registered_regions.get(ptr)
            if current is None:
                raise RuntimeError(f"Mooncake address {ptr} is not registered")
            registered_size, ref_count = current
            if registered_size != size:
                raise RuntimeError(f"Mooncake address {ptr} is registered with size {registered_size}, not {size}")
            if ref_count > 1:
                self._registered_regions[ptr] = (registered_size, ref_count - 1)
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            native_result = self.transfer_engine.unregister_memory(ptr)
            if native_result != 0:
                raise RuntimeError(
                    f"Mooncake memory deregistration failed: ptr={ptr}, size={size}, ret={native_result}"
                )
            del self._registered_regions[ptr]


global_te = GlobalTE()
