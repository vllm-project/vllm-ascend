from enum import Enum


class AscendDeviceType(Enum):
    A2 = 0
    A3 = 1
    _310P = 2
    A5 = 3


_ascend_device_type = None


def _init_ascend_device_type():
    global _ascend_device_type
    from vllm_ascend import _build_info  # type: ignore

    _ascend_device_type = AscendDeviceType[_build_info.__device_type__]


def get_ascend_device_type():
    global _ascend_device_type
    if _ascend_device_type is None:
        _init_ascend_device_type()
    return _ascend_device_type
