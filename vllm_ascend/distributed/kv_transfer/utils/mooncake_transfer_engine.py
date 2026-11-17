import threading


class GlobalTE:
    def __init__(self, protocol: str = "ascend"):
        self.protocol = protocol
        self.transfer_engine = None
        self._initialization_attempted = False
        self.is_register_buffer: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
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
                    ret_value = engine.initialize(hostname, "P2PHANDSHAKE", self.protocol, device_name)
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

            # Ascend Direct unregister_memory also disconnects peers.
            for ptr, _ in self.registered_buffers:
                ret_value = self.transfer_engine.unregister_memory(ptr)
                if ret_value != 0:
                    raise RuntimeError(f"Mooncake memory unregistration failed for ptr={ptr:#x}, ret_value={ret_value}")

            self.is_register_buffer = False

    def reregister_buffer(self):
        with self.register_buffer_lock:
            if self.is_register_buffer or not self.registered_buffers:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            for ptr, size in self.registered_buffers:
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError(
                        f"Mooncake memory re-registration failed for ptr={ptr:#x}, ret_value={ret_value}"
                    )

            self.is_register_buffer = True


global_te = GlobalTE()


class _MemfabricPullEngine:
    def __init__(self, engine):
        self._engine = engine

    def get_rpc_port(self):
        return self._engine.get_rpc_port()

    def batch_transfer_sync_read(self, session_id, local_buffers, peer_buffers, lengths):
        return self._engine.batch_transfer_sync_read(
            session_id, local_buffers, peer_buffers, lengths, transport_hint="memfabric"
        )


class HiSparseMooncakeTE(GlobalTE):
    def __init__(self):
        super().__init__(protocol="memfabric")
        self._pull_engine = None

    def get_transfer_engine(self, hostname: str, device_name: str | None = None):
        from vllm_ascend import envs

        if envs.MC_USE_TENT != "1" or envs.MF_PROTOCOL != "device_urma":
            raise ValueError(
                "HiSparse Mooncake PD requires MC_USE_TENT=1 and MF_PROTOCOL=device_urma; SDMA is disabled"
            )
        engine = super().get_transfer_engine(hostname, device_name)
        with self.transfer_engine_lock:
            if self._pull_engine is None:
                self._pull_engine = _MemfabricPullEngine(engine)
            return self._pull_engine


global_hisparse_te = HiSparseMooncakeTE()
