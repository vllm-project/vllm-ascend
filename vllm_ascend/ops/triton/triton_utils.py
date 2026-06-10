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
        logger.error("[TritonOps] Failed to resolve Triton op '%s' because HAS_TRITON is False.", op_name)
        raise RuntimeError("[TritonOps] Failed to resolve Triton op '{}' because HAS_TRITON is False.".format(op_name))

    if _extension_module is not None:
        extension_op = getattr(_extension_module, op_name, None)
        if extension_op is not None:
            return extension_op

    tl_op = getattr(tl, op_name, None)
    if tl_op is not None:
        return tl_op

    logger.error(
        "[TritonOps] Failed to resolve Triton op '%s': "
        "neither triton.language.extra.cann.extension nor triton.language provides it.",
        op_name,
    )
    raise RuntimeError(
        "[TritonOps] Failed to resolve Triton op '{}': "
        "neither triton.language.extra.cann.extension nor triton.language provides it.".format(op_name)
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
                "[TritonOps] Failed to detect device properties: num_aicore=%s, num_vectorcore=%s",
                _NUM_AICORE,
                _NUM_VECTORCORE,
            )
            raise RuntimeError(
                "[TritonOps] Failed to detect device properties: num_aicore={}, num_vectorcore={}".format(
                    _NUM_AICORE, _NUM_VECTORCORE
                )
            )


def get_aicore_num():
    global _NUM_AICORE
    if _NUM_AICORE <= 0:
        logger.error(
            "[TritonOps] Device properties not initialized (num_aicore=%s). "
            "Call init_device_properties_triton() first.",
            _NUM_AICORE,
        )
        raise RuntimeError(
            "[TritonOps] Device properties not initialized "
            "(num_aicore={}). "
            "Call init_device_properties_triton() first.".format(_NUM_AICORE)
        )
    return _NUM_AICORE


def get_vectorcore_num():
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE <= 0:
        logger.error(
            "[TritonOps] Device properties not initialized "
            "num_vectorcore=%s). "
            "Call init_device_properties_triton() first.",
            _NUM_VECTORCORE,
        )
        raise RuntimeError(
            "[TritonOps] Device properties not initialized "
            "(num_vectorcore={}). "
            "Call init_device_properties_triton() first.".format(_NUM_VECTORCORE)
        )
    return _NUM_VECTORCORE
