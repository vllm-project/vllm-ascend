from typing import Any

import torch
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON, tl, triton

_NUM_AICORE = -1
_NUM_VECTORCORE = -1
_extension_module = None

if HAS_TRITON:
    try:
        import triton.language.extra.cann.extension as _extension_module  # type: ignore
    except ImportError:
        logger.warning(
            "[TritonOps] Failed to import "
            "triton.language.extra.cann.extension, "
            "falling back to triton.language for op resolution."
        )
        _extension_module = None


def _resolve_triton_ascend_op(op_name: str):
    if not HAS_TRITON:
        logger.error(
            f"[TritonOps] Failed to resolve Triton op '{op_name}' because HAS_TRITON is False."
        )
        raise RuntimeError(f"Triton op '{op_name}' cannot be resolved because HAS_TRITON is False")

    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    logger.error(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )
    raise RuntimeError(
        f"Failed to resolve Triton op '{op_name}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it."
    )


if HAS_TRITON:
    insert_slice = _resolve_triton_ascend_op("insert_slice")
    extract_slice = _resolve_triton_ascend_op("extract_slice")
    get_element = _resolve_triton_ascend_op("get_element")
    logger.debug("[TritonOps] Resolved triton ascend ops: insert_slice, extract_slice, get_element")
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
        if _NUM_AICORE <= 0 or _NUM_VECTORCORE <= 0:
            logger.error(
                "[TritonOps] Failed to detect device properties: "
                f"num_aicore={_NUM_AICORE}, num_vectorcore={_NUM_VECTORCORE}"
            )
            raise RuntimeError(
                f"Failed to detect device properties: num_aicore={_NUM_AICORE}, num_vectorcore={_NUM_VECTORCORE}"
            )


def get_aicore_num():
    global _NUM_AICORE
    if _NUM_AICORE <= 0:
        logger.error(
            "[TritonOps] Device properties not initialized "
            f"(num_aicore={_NUM_AICORE}). "
            "Call init_device_properties_triton() first."
        )
        raise RuntimeError(
            f"Device properties not initialized (num_aicore={_NUM_AICORE}). Call init_device_properties_triton() first."
        )
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE <= 0:
        logger.error(
            "[TritonOps] Device properties not initialized "
            f"(num_vectorcore={_NUM_VECTORCORE}). "
            "Call init_device_properties_triton() first."
        )
        raise RuntimeError(
            "Device properties not initialized "
            f"(num_vectorcore={_NUM_VECTORCORE}). "
            "Call init_device_properties_triton() first."
        )
    return _NUM_VECTORCORE
