import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self._initialization_attempted = False
        self.is_register_buffer: bool = False
        # Keep registration order across sleep/wake cycles.
        # False entries remain known, but must be registered again before use.
        self.registered_buffers: dict[tuple[int, int], bool] = {}
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

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

    def register_buffer(
        self,
        ptrs: list[int],
        sizes: list[int],
    ):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            # Register each buffer at most once, so lazily allocated buffers
            # (e.g. the NPU staging window) can register after the initial
            # kv_caches registration instead of being silently skipped.
            for ptr, size in zip(ptrs, sizes):
                key = (int(ptr), int(size))
                if self.registered_buffers.get(key, False):
                    continue
                self._register_memory(*key)
                self.registered_buffers[key] = True
                self.is_register_buffer = True

    def _register_memory(self, ptr: int, size: int):
        assert self.transfer_engine is not None, "Transfer engine must be initialized"
        ret_value = self.transfer_engine.register_memory(ptr, size)
        if ret_value != 0:
            raise RuntimeError(f"Mooncake memory registration failed for ptr={ptr:#x}, ret_value={ret_value}")

    def unregister_buffer(self):
        with self.register_buffer_lock:
            if not self.is_register_buffer or not self.registered_buffers:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            # Ascend Direct unregister_memory also disconnects peers.
            for key, registered in self.registered_buffers.items():
                if not registered:
                    continue
                ptr = key[0]
                ret_value = self.transfer_engine.unregister_memory(ptr)
                if ret_value != 0:
                    raise RuntimeError(f"Mooncake memory unregistration failed for ptr={ptr:#x}, ret_value={ret_value}")

                self.registered_buffers[key] = False
                self.is_register_buffer = any(self.registered_buffers.values())

    def reregister_buffer(self):
        with self.register_buffer_lock:
            if not self.registered_buffers:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            for key, registered in self.registered_buffers.items():
                if registered:
                    continue
                self._register_memory(*key)
                self.registered_buffers[key] = True
                self.is_register_buffer = True


global_te = GlobalTE()
