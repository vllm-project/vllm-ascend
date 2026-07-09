import threading
from contextlib import suppress


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
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
            self.is_register_buffer = True

    def reset(self):
        """Drop cached TE. Caller must unregister_memory on old engine BEFORE this."""
        with self.transfer_engine_lock:
            old_engine = self.transfer_engine
            self.transfer_engine = None
        with self.register_buffer_lock:
            self.is_register_buffer = False

        if old_engine is None:
            return

        # TransferEngine 没有可用的 close/finalize；必须把最后一个 Python 引用清掉，
        # 并立刻触发析构，不能留给 GC 延后执行。
        try:
            del old_engine
            import gc
            gc.collect()
        except Exception as e:
            logger.warning("[snapshot] destroy old TransferEngine failed: %s", e)


        # """[snapshot] Drop the cached transfer engine so it can be re-initialized
        # on a new host IP after container snapshot restore (PD-disaggregated only).

        # The next get_transfer_engine() call will create a brand-new engine bound
        # to the new hostname. We best-effort release the old engine's RPC endpoint
        # first so the listening port can be reused.
        # """
        # with self.transfer_engine_lock:
        #     old_engine = self.transfer_engine
        #     self.transfer_engine = None
        # with self.register_buffer_lock:
        #     self.is_register_buffer = False
        # if old_engine is not None:
        #     for closer in ("close", "shutdown", "deinitialize", "finalize", "stop"):
        #         fn = getattr(old_engine, closer, None)
        #         if callable(fn):
        #             with suppress(Exception):
        #                 fn()
        #             break
        # # Drop the last reference; rely on GC/destructor to free native resources.
        # del old_engine


global_te = GlobalTE()
