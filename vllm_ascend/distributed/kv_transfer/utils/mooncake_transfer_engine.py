import logging
import os
import threading

logger = logging.getLogger(__name__)


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
                    # Warn if VLLM_HOST_IP is set — it gets inherited by all
                    # subprocesses and can cause workers on different nodes to
                    # bind the TransferEngine to a wrong IP.  Each worker should
                    # auto-detect or have VLLM_HOST_IP set per-machine.
                    vllm_host_ip = os.environ.get("VLLM_HOST_IP")
                    if vllm_host_ip is not None and vllm_host_ip != hostname:
                        logger.warning(
                            "[ADDR] VLLM_HOST_IP=%s differs from detected hostname=%s. "
                            "Make sure VLLM_HOST_IP is set correctly per node, "
                            "especially when using multiprocessing executor.",
                            vllm_host_ip, hostname,
                        )
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
                        raise RuntimeError(f"TransferEngine initialization failed with hostname={hostname}, "
                                           f"ret_value: {ret_value}. Check that VLLM_HOST_IP is set correctly per node.")
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


global_te = GlobalTE()
