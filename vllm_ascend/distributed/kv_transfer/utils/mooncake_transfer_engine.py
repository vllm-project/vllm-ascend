import threading
from importlib.metadata import version as get_version

from packaging.version import Version

_WILDCARD_LOCATION = "*"
_LOCATION_API_MIN_VERSION = Version("0.3.12")
_MOONCAKE_DISTRIBUTION = "mooncake-transfer-engine"


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()
        self._registered_regions: list[tuple[int, int, str]] = []

    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    try:
                        from mooncake.engine import TransferEngine  # type: ignore
                    except ImportError as e:
                        raise ImportError(
                            "Please install mooncake by following the instructions at "
                            "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                            "to run vLLM with MooncakeConnector."
                        ) from e
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
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

    def register_buffer(
        self,
        ptrs: list[int],
        sizes: list[int],
        locations: list[str] | None = None,
    ) -> None:
        if locations is not None:
            mooncake_version = get_version(_MOONCAKE_DISTRIBUTION)
            if Version(mooncake_version) < _LOCATION_API_MIN_VERSION:
                raise RuntimeError(
                    f"Mooncake {mooncake_version} does not support register_memory locations; "
                    f"{_LOCATION_API_MIN_VERSION} or newer is required"
                )

        normalized_locations = locations if locations is not None else [_WILDCARD_LOCATION] * len(ptrs)
        regions = list(zip(ptrs, sizes, normalized_locations))

        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return

            for ptr, size, location in regions:
                if locations is None:
                    ret_value = self.transfer_engine.register_memory(ptr, size)
                else:
                    ret_value = self.transfer_engine.register_memory(ptr, size, location)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self._registered_regions = regions
            self.is_register_buffer = True

    def unregister_buffer(self) -> None:
        """Unregister all buffers owned by this wrapper, if any."""
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


global_te = GlobalTE()
