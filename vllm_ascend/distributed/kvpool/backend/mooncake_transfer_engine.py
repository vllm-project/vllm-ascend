import ipaddress
import threading
from typing import Optional

from mooncake.engine import TransferEngine  # type: ignore


class GlobalTE():

    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    def get_transfer_engine(self, hostname: str, device_name: Optional[str]):
        try:
            ip = ipaddress.ip_address(hostname)
            if isinstance(ip, ipaddress.IPv6Address):
                raise RuntimeError(
                    "The backend of mooncake's Ascend Direct Xfer Library currently does not support IPv6."
                )
        except ValueError:
            pass
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    if TransferEngine is None:
                        raise RuntimeError("mooncake is not available")
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(
                        hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(
                            f"TransferEngine initialization failed with ret_value: {ret_value}"
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
            self.is_register_buffer = True


global_te = GlobalTE()
