from typing import Any

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_NUM_AICORE = -1
_NUM_VECTORCORE = -1
_extension_module = None

if HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module  # type: ignore
    except ImportError:
        _extension_module = None


def _resolve_triton_ascend_op(op_name: str):
    if not HAS_TRITON:
        raise RuntimeError(f"Triton op '{op_name}' cannot be resolved because HAS_TRITON is False")

    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


if HAS_TRITON:
    insert_slice = _resolve_triton_ascend_op("insert_slice")
    extract_slice = _resolve_triton_ascend_op("extract_slice")
    get_element = _resolve_triton_ascend_op("get_element")
else:
    insert_slice = None
    extract_slice = None
    get_element = None


def init_device_properties_triton():
    global _NUM_AICORE, _NUM_VECTORCORE
    if _NUM_AICORE == -1 and HAS_TRITON:
        device_properties: dict[str, Any] = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        _NUM_AICORE = device_properties.get("num_aicore", -1)
        _NUM_VECTORCORE = device_properties.get("num_vectorcore", -1)
        assert _NUM_AICORE > 0 and _NUM_VECTORCORE > 0, "Failed to detect device properties."


def get_aicore_num():
    global _NUM_AICORE
    assert _NUM_AICORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    assert _NUM_VECTORCORE > 0, "Device properties not initialized. Please call init_device_properties_triton() first."
    return _NUM_VECTORCORE
