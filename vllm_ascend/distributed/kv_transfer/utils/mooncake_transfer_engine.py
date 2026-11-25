import threading

from vllm_ascend import envs


def _configure_local_comm_res():
    """Use an opt-in per-device descriptor when visibility hides remote peers."""
    import json
    import os
    from pathlib import Path

    resource_dir = envs.VLLM_ASCEND_HY4_MOONCAKE_LOCAL_COMM_RES_DIR
    if not resource_dir:
        return
    import torch
    from vllm.platforms import current_platform

    physical_id = current_platform.visible_device_id_to_physical_device_id(torch.npu.current_device())
    config_path = Path(resource_dir) / f"{physical_id}.json"
    config = json.loads(config_path.read_text())
    endpoints = config.get("endpoint_list")
    if config.get("version") != "1.3" or not isinstance(endpoints, list) or not endpoints:
        raise ValueError(f"Invalid per-device Mooncake resource descriptor: {config_path}")
    if any(endpoint.get("protocol") != "ub_ctp" or endpoint.get("placement") != "device" for endpoint in endpoints):
        raise ValueError(f"Expected original device ub_ctp endpoints: {config_path}")
    os.environ["ASCEND_LOCAL_COMM_RES"] = json.dumps(config)
    import logging

    logging.getLogger(__name__).info(
        "Using complete Mooncake device resources: physical_device=%s endpoints=%s path=%s",
        physical_id,
        len(endpoints),
        config_path,
    )


class GlobalTE:
    def __init__(self):
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
                    _configure_local_comm_res()
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
