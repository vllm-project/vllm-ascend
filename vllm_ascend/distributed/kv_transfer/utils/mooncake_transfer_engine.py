import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self._initialization_attempted = False
        self.registered_buffers: list[tuple[int, int]] = []
        # Buffers unregistered since the last full (re)registration, so a
        # failure part-way through either direction can be resumed.
        self._unregistered_ptrs: set[int] = set()
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

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            # Register once per process. Once recorded, buffers dropped by
            # unregister_buffer() come back through reregister_buffer(), so a
            # stray call here cannot register a still-registered buffer twice.
            if self.registered_buffers:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.registered_buffers = list(zip(ptrs, sizes))

    def unregister_buffer(self):
        with self.register_buffer_lock:
            if len(self._unregistered_ptrs) == len(self.registered_buffers):
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            # Record each buffer as it goes, so that after a failure part-way
            # reregister_buffer() restores exactly the ones this call dropped.
            # Ascend Direct unregister_memory also disconnects peers.
            for ptr, _ in self.registered_buffers:
                if ptr in self._unregistered_ptrs:
                    continue
                ret_value = self.transfer_engine.unregister_memory(ptr)
                if ret_value != 0:
                    raise RuntimeError(f"Mooncake memory unregistration failed for ptr={ptr:#x}, ret_value={ret_value}")
                self._unregistered_ptrs.add(ptr)

    def reregister_buffer(self):
        with self.register_buffer_lock:
            if not self._unregistered_ptrs:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            for ptr, size in self.registered_buffers:
                if ptr not in self._unregistered_ptrs:
                    continue
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake memory re-registration failed for ptr={ptr:#x}, ret_value={ret_value}"
                    )
                self._unregistered_ptrs.discard(ptr)


global_te = GlobalTE()
