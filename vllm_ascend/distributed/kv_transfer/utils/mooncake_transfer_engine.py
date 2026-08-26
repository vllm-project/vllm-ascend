import threading


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

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

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.registered_buffers = list(zip(ptrs, sizes))
            self.is_register_buffer = True

    def unregister_buffer(self):
        with self.register_buffer_lock:
            if not self.is_register_buffer or not self.registered_buffers:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            unregistered_buffers: list[tuple[int, int]] = []
            for ptr, size in self.registered_buffers:
                ret_value = self.transfer_engine.unregister_memory(ptr)
                if ret_value != 0:
                    rollback_failures = []
                    for unregistered_ptr, unregistered_size in reversed(unregistered_buffers):
                        rollback_ret = self.transfer_engine.register_memory(unregistered_ptr, unregistered_size)
                        if rollback_ret != 0:
                            rollback_failures.append((unregistered_ptr, rollback_ret))
                    self.is_register_buffer = not rollback_failures
                    raise RuntimeError(
                        f"Mooncake memory unregistration failed for ptr={ptr:#x}, "
                        f"ret_value={ret_value}, rollback_failures={rollback_failures}"
                    )
                unregistered_buffers.append((ptr, size))

            self.is_register_buffer = False

    def reregister_buffer(self):
        with self.register_buffer_lock:
            if self.is_register_buffer or not self.registered_buffers:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            reregistered_buffers: list[tuple[int, int]] = []
            for ptr, size in self.registered_buffers:
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    rollback_failures = []
                    for reregistered_ptr, _ in reversed(reregistered_buffers):
                        rollback_ret = self.transfer_engine.unregister_memory(reregistered_ptr)
                        if rollback_ret != 0:
                            rollback_failures.append((reregistered_ptr, rollback_ret))
                    self.is_register_buffer = bool(rollback_failures)
                    raise RuntimeError(
                        f"Mooncake memory re-registration failed for ptr={ptr:#x}, "
                        f"ret_value={ret_value}, rollback_failures={rollback_failures}"
                    )
                reregistered_buffers.append((ptr, size))

            self.is_register_buffer = True


global_te = GlobalTE()
