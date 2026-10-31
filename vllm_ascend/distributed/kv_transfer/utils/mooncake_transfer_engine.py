import threading

# Mooncake transport used for PD KV transfer unless
# ``kv_connector_extra_config["transfer_protocol"]`` overrides it. ``ascend``
# installs the Ascend transport, which falls back to ``aclrtMemcpy`` staging
# when the machine exposes neither RoCE nor fabric memory; nodes that do have a
# device-direct path can select it per deployment instead of patching this
# module. Prefill and decode must be configured with the same value.
DEFAULT_TRANSFER_PROTOCOL = "ascend"


class GlobalTE:
    def __init__(self):
        self.transfer_engine = None
        self.transfer_protocol: str | None = None
        self._initialization_attempted = False
        self.is_register_buffer: bool = False
        self.registered_buffers: list[tuple[int, int]] = []
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(
        self,
        hostname: str,
        device_name: str | None,
        protocol: str | None = None,
    ):
        """``protocol=None`` means "whatever this process already uses", so the
        components that share this singleton (encoder cache transfer, the
        AscendStore Mooncake backend) keep calling it the way they always did.
        """
        resolved_protocol = protocol or DEFAULT_TRANSFER_PROTOCOL
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
                    ret_value = engine.initialize(hostname, "P2PHANDSHAKE", resolved_protocol, device_name)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"TransferEngine initialization failed with ret_value: {ret_value}. "
                            "Restart the process before retrying."
                        )
                    # Publish the protocol before the engine: readers check the
                    # pair, and an engine visible with protocol=None would look
                    # like a conflict to a lock-free caller.
                    self.transfer_protocol = resolved_protocol
                    self.transfer_engine = engine
        # The native engine is process-wide and cannot be re-initialized, so a
        # second caller asking for another transport would otherwise silently
        # run on the first one. This only catches an in-process mismatch;
        # prefill and decode are separate processes.
        if protocol is not None and resolved_protocol != self.transfer_protocol:
            raise RuntimeError(
                f"TransferEngine already initialized with protocol {self.transfer_protocol!r}; "
                f"cannot switch to {resolved_protocol!r} in the same process."
            )
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

    def _disconnect_all_peers(self):
        if self.transfer_engine is None:
            return
        disconnect_ret = self.transfer_engine.disconnect_all_peers()
        if disconnect_ret != 0:
            raise RuntimeError(f"Mooncake peer disconnection failed with ret_value: {disconnect_ret}")

    def disconnect_all_peers(self):
        with self.register_buffer_lock:
            self._disconnect_all_peers()

    def unregister_buffer(self):
        with self.register_buffer_lock:
            if not self.is_register_buffer or not self.registered_buffers:
                return
            assert self.transfer_engine is not None, "Transfer engine must be initialized"

            self._disconnect_all_peers()

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
