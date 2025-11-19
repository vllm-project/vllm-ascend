import ipaddress
import threading
from typing import Optional

from mooncake.engine import TransferEngine  # type: ignore

_global_te = None
_global_te_lock = threading.Lock()


def get_global_te(hostname: str, device_name: Optional[str]):
    try:
        ip = ipaddress.ip_address(hostname)
        if isinstance(ip, ipaddress.IPv6Address):
            raise RuntimeError(
                "The backend of mooncake's Ascend Direct Xfer Library currently does not support IPv6."
            )
    except ValueError:
        pass

    global _global_te
    if _global_te is None:
        with _global_te_lock:
            # Double-Checked Locking
            if _global_te is None:
                if TransferEngine is None:
                    raise RuntimeError("mooncake is not available")
                transfer_engine = TransferEngine()
                device_name = device_name if device_name is not None else ""
                ret_value = transfer_engine.initialize(hostname,
                                                       "P2PHANDSHAKE",
                                                       "ascend", device_name)
                if ret_value != 0:
                    raise RuntimeError(
                        f"TransferEngine initialization failed with ret_value: {ret_value}"
                    )
                _global_te = transfer_engine
    return _global_te
