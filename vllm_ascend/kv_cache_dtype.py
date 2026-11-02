# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Resolve native C8 operand types separately from common byte storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def native_c8_cache_dtype(cache_dtype: str, model_config: ModelConfig) -> torch.dtype:
    """Return the A5 native FP8 operand type without changing generic storage.

    Common vLLM represents FP8 packed storage with UInt8. Native Ascend C8
    quantization and attention instead take an E4M3FN operand dtype. Resolve
    only those logical E4M3 names here; other formats retain their original
    resolver, including the separate DeepSeek opaque-storage format.
    """
    if cache_dtype in ("fp8", "fp8_e4m3"):
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() == AscendDeviceType.A5:
            return torch.float8_e4m3fn
    return kv_cache_dtype_str_to_dtype(cache_dtype, model_config)


def prepare_native_hisparse_view_dtypes(forward_context, kv_caches) -> None:
    """Align native C8 view declarations before the first common cache bind.

    The raw backing remains byte storage. Only this unbound Runtime instance's
    typed-view declaration is aligned to its existing native Host/source spec.
    Common binding then constructs Resident/Hot aliases in that same dtype.
    Already-bound instances must already match, as during profiling rebind.
    """
    from vllm_ascend.ops.hisparse_c8 import C8_PAYLOAD_FORMAT

    pending = []
    for name, layer in forward_context.items():
        impl = getattr(layer, "impl", None)
        if getattr(impl, "hisparse_c8_payload_format", None) != C8_PAYLOAD_FORMAT:
            continue
        handle = impl.hisparse_cache
        runtime = handle.runtime
        source = kv_caches[name]
        if (
            runtime.row_width != 656
            or runtime.kv_dtype not in (torch.uint8, torch.float8_e4m3fn)
            or runtime.kv_dtype.itemsize != 1
            or not isinstance(source, torch.Tensor)
            or source.dtype != torch.float8_e4m3fn
            or source.element_size() != 1
            or source.device.type != "cpu"
            or not (source.ndim in (2, 3) or (source.ndim == 4 and source.shape[1] == 1))
            or source.shape[-1] != 656
        ):
            raise ValueError("Native HiSparse C8 requires matching one-byte E4M3FN source/view declarations.")
        if runtime.kv_dtype == torch.float8_e4m3fn:
            continue
        if runtime.cache_ops is not None or handle.view is not None or hasattr(runtime, "hot"):
            raise RuntimeError("Native HiSparse C8 view dtype must be declared before cache binding.")
        pending.append(runtime)
    for runtime in pending:
        runtime.kv_dtype = torch.float8_e4m3fn
