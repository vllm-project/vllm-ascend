import threading

_WILDCARD_LOCATION = "*"


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self._initialization_attempted = False
        self.is_register_buffer: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
        self._registered_buffer_locations: list[str] = []
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self._registered_regions: list[tuple[int, int, str]] = []

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    if self._initialization_attempted:
                        raise RuntimeError(
                            "TransferEngine initialization previously failed. "
                            "Restart the process before retrying to avoid leaking native resources."
                        )
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    # Mooncake 0.3.11.post1 leaks transport resources on failed
                    # initialization (kvcache-ai/Mooncake#2332). Mark the attempt
                    # before native construction, including exceptions. Revisit
                    # retries once the minimum supported version includes that
                    # fix and repeated-failure resource cleanup is verified.
                    self._initialization_attempted = True
                    engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"TransferEngine initialization failed with ret_value: {ret_value}. "
                            "Restart the process before retrying."
                        )
                    self.transfer_engine = engine
        return self.transfer_engine

    def _unregister_regions(
        self,
        regions: list[tuple[int, int, str]],
    ) -> tuple[list[tuple[int, int, str]], list[str]]:
        """Unregister regions in reverse order and return any failures."""
        assert self.transfer_engine is not None
        failed_regions: list[tuple[int, int, str]] = []
        failure_messages: list[str] = []
        for region in reversed(regions):
            ptr, _, _ = region
            try:
                ret_value = self.transfer_engine.unregister_memory(ptr)
            except Exception as exc:  # noqa: BLE001
                failed_regions.append(region)
                failure_messages.append(f"ptr={ptr}: {exc}")
                continue
            if ret_value != 0:
                failed_regions.append(region)
                failure_messages.append(f"ptr={ptr}: ret_value={ret_value}")

        failed_regions.reverse()
        return failed_regions, failure_messages

    def _register_regions(self, regions: list[tuple[int, int, str]]) -> None:
        """Register under register_buffer_lock, rolling back partial failures."""
        assert self.transfer_engine is not None, "Transfer engine must be initialized"
        registered_regions: list[tuple[int, int, str]] = []
        try:
            for ptr, size, location in regions:
                ret_value = self.transfer_engine.register_memory(ptr, size, location)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
                registered_regions.append((ptr, size, location))
        except Exception as exc:
            failed_regions, failure_messages = self._unregister_regions(registered_regions)
            self._registered_regions = failed_regions
            self.is_register_buffer = bool(failed_regions)
            if failure_messages:
                raise RuntimeError(
                    "Mooncake memory registration failed and rollback failed "
                    "for regions: " + "; ".join(failure_messages)
                ) from exc
            raise
        self._registered_regions = regions
        self.is_register_buffer = True

    def register_buffer(
        self,
        ptrs: list[int],
        sizes: list[int],
        locations: list[str] | None = None,
    ) -> None:
        normalized_locations = locations if locations is not None else [_WILDCARD_LOCATION] * len(ptrs)
        regions = list(zip(ptrs, sizes, normalized_locations))
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            self._register_regions(regions)
            # Keep the successful plan for main's sleep/wake re-registration.
            self.registered_buffers = [(ptr, size) for ptr, size, _ in regions]
            self._registered_buffer_locations = [location for _, _, location in regions]

    def unregister_buffer(self) -> None:
        """Unregister active regions while retaining the successful restore plan."""
        with self.register_buffer_lock:
            if not self._registered_regions:
                self.is_register_buffer = False
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            failed_regions, failure_messages = self._unregister_regions(self._registered_regions)
            self._registered_regions = failed_regions
            self.is_register_buffer = bool(failed_regions)
            if failure_messages:
                raise RuntimeError("Mooncake memory unregistration failed for regions: " + "; ".join(failure_messages))

    def reregister_buffer(self) -> None:
        """Restore the same buffers and locations after a successful unregister."""
        with self.register_buffer_lock:
            if self.is_register_buffer or not self.registered_buffers:
                return
            locations = self._registered_buffer_locations or [_WILDCARD_LOCATION] * len(self.registered_buffers)
            regions = [(ptr, size, location) for (ptr, size), location in zip(self.registered_buffers, locations)]
            self._register_regions(regions)


global_te = GlobalTE()
