import torch

_IS_310P = None


def is_310p():
    global _IS_310P
    if _IS_310P is None:
        device_name = torch.npu.get_device_name()
        _IS_310P = "310" in device_name
    return _IS_310P
