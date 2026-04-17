#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""QuaRot W4A4 schemes for fused lean_h_only serving."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import math
from typing import Any

import torch
import torch_npu
from safetensors import safe_open
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm_ascend.ops.fast_hadamard import (
    fast_hadamard_dynamic_quant_blockwise_last_dim,
    fast_hadamard_last_dim_custom_op,
)
from vllm_ascend.quantization.quarot_kv_cache import use_native_quarot_kv_cache

from .base import AscendAttentionScheme, AscendLinearScheme
from .registry import register_scheme

QUAROT_LINEAR_QUANT_TYPE = "W4A4_QUAROT_DYNAMIC"
QUAROT_ATTN_QUANT_TYPE = "QUAROT_ATTENTION"

HADAMARD_KERNEL_FAMILY_ENV = "VLLM_ASCEND_QUAROT_HADAMARD_KERNEL_FAMILY"
HADAMARD_LAUNCH_MODE_ENV = "VLLM_ASCEND_QUAROT_HADAMARD_LAUNCH_MODE"
HADAMARD_KERNEL_FAMILY_PYTHON = "python"
HADAMARD_KERNEL_FAMILY_PTO = "pto"
HADAMARD_LAUNCH_MODE_EAGER_JIT = "eager_jit"
HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP = "compile_custom_op"
_VALID_HADAMARD_KERNEL_FAMILIES = {
    HADAMARD_KERNEL_FAMILY_PYTHON,
    HADAMARD_KERNEL_FAMILY_PTO,
}
_VALID_HADAMARD_LAUNCH_MODES = {
    HADAMARD_LAUNCH_MODE_EAGER_JIT,
    HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP,
}
_PTO_HADAMARD_JIT_FUNC = None
_PTO_JIT_WARMUP_DONE = False
_NATIVE_QUAROT_INDEX_CACHE: dict[str, dict[str, str]] = {}
_NATIVE_QUAROT_TENSOR_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_DETERMINISTIC_WALSH_CPU_CACHE: dict[int, torch.Tensor] = {}
_DETERMINISTIC_WALSH_DEVICE_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}
_BLOCK_DIAG_WALSH_CACHE: dict[tuple[int, int, str, int | None, torch.dtype], torch.Tensor] = {}
_PTO_KERNEL_CPP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "ops",
    "fast_hadamard_pto-isa.cpp",
)
_MODELSLIM_WEIGHT_INDEX_FILENAME = "quant_model_weights.safetensors.index.json"
_LEAN_H_ONLY_QUAROT_CONTRACT = {
    "export_rotation_tensors": False,
    "h_mode": "deterministic_walsh",
    "deterministic_h_profile": True,
    "alpha_source": "rmsnorm",
    "matrix_free_h_runtime": True,
}
_LEAN_H_ONLY_SUPPORTED_Q_MODES = {"randomized_hadamard", "identity"}
_LEAN_H_ONLY_SUPPORTED_CONTRACT_VERSIONS = {"2.1.0"}
_FFN_HADAMARD_LAYOUT = "pow2_last_dim"
_LEAN_ATTENTION_CONTRACT = "lean_h_only"
_QUAROT_DEBUG_VALUE_PATH_ENV = "VLLM_ASCEND_QUAROT_DEBUG_VALUE_PATH"
_QUAROT_DEBUG_ATTN_COMPARE_ENV = "VLLM_ASCEND_QUAROT_DEBUG_ATTN_COMPARE"
_QUAROT_TRACE_QKV_REF_ENV = "VLLM_ASCEND_QUAROT_TRACE_QKV_REF"
_HADAMARD_BASE_12 = torch.tensor(
    [
        [1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1],
        [1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1, -1],
        [1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1],
        [1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1],
        [1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1],
        [1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1],
        [1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1],
        [1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1],
        [1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1, -1],
        [1, -1, 1, -1, -1, -1, 1, 1, 1, -1, 1, 1],
    ],
    dtype=torch.float32,
)
logger = init_logger(__name__)

@dataclass(frozen=True)
class HadamardDispatch:
    kernel_family: str
    launch_mode: str | None


def _layer_prefix(layer: torch.nn.Module) -> str:
    return getattr(layer, "prefix", None) or getattr(layer, "layer_name", "unknown")



def _quarot_debug_value_path_enabled() -> bool:
    return os.getenv(_QUAROT_DEBUG_VALUE_PATH_ENV) == "1"


def _quarot_debug_attn_compare_enabled() -> bool:
    return os.getenv(_QUAROT_DEBUG_ATTN_COMPARE_ENV) == "1"


def _quarot_trace_qkv_ref_enabled() -> bool:
    return os.getenv(_QUAROT_TRACE_QKV_REF_ENV) == "1"


def _tensor_debug_summary(tensor: torch.Tensor) -> dict[str, object]:
    if tensor.numel() == 0:
        return {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype), "empty": True}
    flat = tensor.detach().reshape(-1).to(torch.float32).cpu()
    sample = flat[:16].tolist()
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "mean": float(flat.mean().item()),
        "sample": sample,
    }


def _maybe_log_value_path(layer: torch.nn.Module, stage: str, tensor: torch.Tensor) -> None:
    if not _quarot_debug_value_path_enabled():
        return
    prefix = _layer_prefix(layer).lower()
    allowed_prefixes = {
        f"model.layers.{idx}.self_attn.qkv_proj" for idx in range(4)
    } | {
        f"model.layers.{idx}.self_attn.o_proj" for idx in range(4)
    } | {"lm_head"}
    if prefix not in allowed_prefixes:
        return
    if tensor.ndim == 0 or tensor.shape[0] > 16:
        return
    logger.info(
        "[quarot-value-path] layer=%s stage=%s tensor=%s",
        _layer_prefix(layer),
        stage,
        _tensor_debug_summary(tensor),
    )


def _maybe_log_o_proj_basis_views(
    layer: torch.nn.Module,
    *,
    linear_input: torch.Tensor,
    linear_output: torch.Tensor,
) -> None:
    if not _quarot_debug_value_path_enabled():
        return
    prefix = _layer_prefix(layer).lower()
    if "self_attn.o_proj" not in prefix and "self_attn.out_proj" not in prefix:
        return
    if linear_input.ndim == 0 or linear_input.shape[0] > 16:
        return
    num_heads, head_dim = _get_quarot_shape(layer)
    if num_heads > 0 and head_dim > 0:
        linear_input_unh = _apply_headwise_hadamard(linear_input, num_heads, head_dim)
        logger.info(
            "[quarot-value-path] layer=%s stage=%s tensor=%s",
            _layer_prefix(layer),
            "linear_input_unh",
            _tensor_debug_summary(linear_input_unh),
        )
    linear_output_unq = _apply_exact_deterministic_walsh_last_dim(linear_output)
    logger.info(
        "[quarot-value-path] layer=%s stage=%s tensor=%s",
        _layer_prefix(layer),
        "linear_output_unq",
        _tensor_debug_summary(linear_output_unq),
    )


def _maybe_log_attention_compare(
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    vendor_output: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    scale: float | torch.Tensor | None,
) -> None:
    if not _quarot_debug_attn_compare_enabled():
        return
    prefix = _layer_prefix(layer)
    if "model.layers.0.self_attn" not in prefix:
        return
    fallback_output = _softmax_attention_fallback(query, key, value, num_heads, head_dim, scale)
    logger.info(
        "[quarot-attn-compare] layer=%s fallback=%s vendor=%s",
        prefix,
        _tensor_debug_summary(fallback_output),
        _tensor_debug_summary(vendor_output),
    )


def _maybe_log_attention_result(
    layer: torch.nn.Module,
    stage: str,
    tensor: torch.Tensor,
) -> None:
    if not (_quarot_debug_attn_compare_enabled() or _quarot_debug_value_path_enabled()):
        return
    prefix = _layer_prefix(layer)
    if prefix != "model.layers.0.self_attn.attn":
        return
    logger.info(
        "[quarot-attn-result] layer=%s stage=%s tensor=%s",
        prefix,
        stage,
        _tensor_debug_summary(tensor),
    )


def _maybe_log_attention_inputs(
    layer: torch.nn.Module,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    if not _quarot_debug_attn_compare_enabled():
        return
    prefix = _layer_prefix(layer)
    if "model.layers.0.self_attn" not in prefix:
        return
    logger.info(
        "[quarot-attn-inputs] layer=%s query=%s key=%s value=%s",
        prefix,
        _tensor_debug_summary(query),
        _tensor_debug_summary(key),
        _tensor_debug_summary(value),
    )


def _maybe_log_qkv_reference_trace(
    layer: torch.nn.Module,
    *,
    use_reference: bool,
    x: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if not _quarot_trace_qkv_ref_enabled():
        return
    prefix = _layer_prefix(layer)
    if prefix != "model.layers.0.self_attn.qkv_proj":
        return
    logger.info(
        "[quarot-qkv-ref] layer=%s use_reference=%s input=%s output=%s",
        prefix,
        use_reference,
        _tensor_debug_summary(x),
        _tensor_debug_summary(output),
    )
    if not use_reference:
        return
    ref_output = _apply_reference_qkv_linear(layer, x, output_dtype=output.dtype)
    diff = (output.to(torch.float32) - ref_output.to(torch.float32)).abs()
    logger.info(
        "[quarot-qkv-ref-compare] layer=%s max_abs=%.9f mean_abs=%.9f ref=%s",
        prefix,
        float(diff.max().item()),
        float(diff.mean().item()),
        _tensor_debug_summary(ref_output),
    )


def _is_quarot_enabled(layer: torch.nn.Module) -> bool:
    config = getattr(layer, "quarot_config", None)
    if isinstance(config, bool):
        return config
    if isinstance(config, dict):
        return bool(config.get("enabled", False))
    return False


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _is_float_dtype(dtype: torch.dtype) -> bool:
    return dtype in (torch.float16, torch.bfloat16, torch.float32)


def _is_int4_repr_dtype(dtype: torch.dtype) -> bool:
    allowed = {torch.int8, torch.uint8, torch.int32}
    # quint4x2 exists on quantized builds; guard for portability in unit tests.
    quint4x2 = getattr(torch, "quint4x2", None)
    if quint4x2 is not None:
        allowed.add(quint4x2)
    return dtype in allowed


def _largest_power_of_two_factor(n: int) -> int:
    if n <= 0:
        return 1
    factor = 1
    while n % 2 == 0:
        factor *= 2
        n //= 2
    return factor


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _apply_normalized_hadamard_last_dim(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 1:
        return x
    if (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be power-of-two, got {n}")
    y = x
    stages = int(math.log2(n))
    for _ in range(stages):
        even = y[..., 0::2]
        odd = y[..., 1::2]
        y = torch.cat((even + odd, even - odd), dim=-1)
    return y / math.sqrt(float(n))


def _get_hadamard_special_factor(n: int, transpose: bool = False) -> tuple[torch.Tensor | None, int]:
    if n % 12 == 0 and _is_power_of_two(n // 12):
        mat = _HADAMARD_BASE_12.T if transpose else _HADAMARD_BASE_12
        return mat, 12
    if not _is_power_of_two(n):
        raise ValueError(f"Cannot construct deterministic Walsh matrix for size {n}")
    return None, 1


def _matmul_had_u(x: torch.Tensor, transpose: bool = False) -> torch.Tensor:
    n = x.shape[-1]
    had_k, k = _get_hadamard_special_factor(n, transpose=transpose)
    input_tensor = x.clone().reshape(-1, n, 1)
    output_tensor = input_tensor.clone()
    while input_tensor.shape[1] > k:
        input_tensor = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1] // 2, 2, input_tensor.shape[2])
        output_tensor = output_tensor.reshape(input_tensor.shape)
        output_tensor[:, :, 0, :] = input_tensor[:, :, 0, :] + input_tensor[:, :, 1, :]
        output_tensor[:, :, 1, :] = input_tensor[:, :, 0, :] - input_tensor[:, :, 1, :]
        output_tensor = output_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        input_tensor, output_tensor = output_tensor, input_tensor

    if k > 1 and had_k is not None:
        input_tensor = had_k.view(1, k, k).to(input_tensor) @ input_tensor

    return input_tensor.reshape(x.shape) / math.sqrt(float(n))


def _get_deterministic_walsh_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    device_key = (size, device.type, device.index, dtype)
    cached_device = _DETERMINISTIC_WALSH_DEVICE_CACHE.get(device_key)
    if cached_device is not None:
        return cached_device

    cached = _DETERMINISTIC_WALSH_CPU_CACHE.get(size)
    if cached is None:
        cached = _matmul_had_u(torch.eye(size, dtype=torch.float32))
        _DETERMINISTIC_WALSH_CPU_CACHE[size] = cached
    device_tensor = cached.to(device=device, dtype=dtype)
    _DETERMINISTIC_WALSH_DEVICE_CACHE[device_key] = device_tensor
    return device_tensor


def _largest_power_of_two_factor(n: int) -> int:
    if n <= 0:
        raise ValueError(f"Expected positive dimension, got {n}")
    return n & -n


def _get_ffn_hadamard_layout(config: dict[str, Any]) -> str:
    layout = config.get("ffn_hadamard_layout", _FFN_HADAMARD_LAYOUT)
    if layout != _FFN_HADAMARD_LAYOUT:
        raise ValueError(
            f"Unsupported FFN Hadamard layout {layout!r}; expected {_FFN_HADAMARD_LAYOUT!r}"
        )
    return layout


def _get_ffn_hadamard_dims(n: int, layout: str) -> tuple[int, int]:
    if layout != _FFN_HADAMARD_LAYOUT:
        raise ValueError(f"Unsupported FFN Hadamard layout {layout!r}; expected {_FFN_HADAMARD_LAYOUT!r}")
    last_dim = _largest_power_of_two_factor(n)
    return n // last_dim, last_dim


def _get_block_diag_walsh_matrix(
    size: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if block_size <= 0 or block_size >= size:
        return _get_deterministic_walsh_matrix(size, dtype, device)
    cache_key = (size, block_size, device.type, device.index, dtype)
    cached = _BLOCK_DIAG_WALSH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if size % block_size != 0:
        raise ValueError(f"Cannot build block-diagonal Walsh matrix: size={size}, block_size={block_size}")
    block = _get_deterministic_walsh_matrix(block_size, dtype, device)
    block_diag = torch.block_diag(*([block] * (size // block_size)))
    _BLOCK_DIAG_WALSH_CACHE[cache_key] = block_diag
    return block_diag


def _apply_matrix_free_down_proj_rotation(
    x: torch.Tensor,
    block_size: int | None = None,
    *,
    layout: str = _FFN_HADAMARD_LAYOUT,
) -> torch.Tensor:
    _, n = _get_ffn_hadamard_dims(x.shape[-1], layout)
    init_shape = x.shape
    # Active FFN contract: flatten non-Hadamard groups into rows so the last
    # dimension can go straight through the power-of-two PTO FHT path.
    if _is_torch_compile_mode() and not x.is_contiguous():
        # Compile mode may lower reshape on a strided FFN activation into
        # an invalid view. Materialize the row-major boundary first.
        x = x.contiguous()
    rows = x.reshape(-1, n)
    rotated_rows = _apply_exact_deterministic_walsh_last_dim(rows)
    return rotated_rows.reshape(init_shape)


def _normalize_hadamard_kernel_family(kernel_family: str) -> str:
    name = kernel_family.strip().lower()
    if name not in _VALID_HADAMARD_KERNEL_FAMILIES:
        raise ValueError(
            f"Unknown Hadamard kernel family '{kernel_family}'. Valid: {sorted(_VALID_HADAMARD_KERNEL_FAMILIES)}"
        )
    return name


def _normalize_hadamard_launch_mode(launch_mode: str) -> str:
    name = launch_mode.strip().lower()
    if name not in _VALID_HADAMARD_LAUNCH_MODES:
        raise ValueError(
            f"Unknown Hadamard launch mode '{launch_mode}'. Valid: {sorted(_VALID_HADAMARD_LAUNCH_MODES)}"
        )
    return name


def _resolve_hadamard_dispatch(kernel_family: str | None) -> HadamardDispatch:
    configured_kernel_family = os.getenv(HADAMARD_KERNEL_FAMILY_ENV)
    launch_mode = os.getenv(HADAMARD_LAUNCH_MODE_ENV)
    if kernel_family is not None:
        normalized_family = _normalize_hadamard_kernel_family(kernel_family)
    elif configured_kernel_family:
        normalized_family = _normalize_hadamard_kernel_family(configured_kernel_family)
    else:
        normalized_family = HADAMARD_KERNEL_FAMILY_PTO

    if normalized_family == HADAMARD_KERNEL_FAMILY_PYTHON:
        return HadamardDispatch(HADAMARD_KERNEL_FAMILY_PYTHON, None)
    if launch_mode:
        normalized_launch_mode = _normalize_hadamard_launch_mode(launch_mode)
    else:
        normalized_launch_mode = (
            HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP
            if _is_torch_compile_mode()
            else HADAMARD_LAUNCH_MODE_EAGER_JIT
        )
    return HadamardDispatch(normalized_family, normalized_launch_mode)


def _get_quarot_config(layer: torch.nn.Module) -> dict[str, Any]:
    config = getattr(layer, "quarot_config", None)
    if isinstance(config, dict):
        return config
    return {}


def _is_lean_h_only_quarot_contract(config: dict[str, Any]) -> bool:
    if config.get("attention_contract") != _LEAN_ATTENTION_CONTRACT:
        return False
    for key, expected in _LEAN_H_ONLY_QUAROT_CONTRACT.items():
        if config.get(key) != expected:
            return False
    if config.get("q_mode") not in _LEAN_H_ONLY_SUPPORTED_Q_MODES:
        return False
    contract_version = config.get("contract_version")
    if contract_version not in _LEAN_H_ONLY_SUPPORTED_CONTRACT_VERSIONS:
        return False
    if config.get("allow_runtime_shift_permutation") not in (None, False):
        return False
    if _get_ffn_hadamard_layout(config) != _FFN_HADAMARD_LAYOUT:
        return False
    runtime_h_partition = config.get("runtime_h_partition")
    return runtime_h_partition == "full"


def _is_supported_fused_quarot_contract(config: dict[str, Any]) -> bool:
    if config.get("attention_contract") != _LEAN_ATTENTION_CONTRACT:
        return False
    for key, expected in _LEAN_H_ONLY_QUAROT_CONTRACT.items():
        if config.get(key) != expected:
            return False
    if config.get("q_mode") not in _LEAN_H_ONLY_SUPPORTED_Q_MODES:
        return False
    contract_version = config.get("contract_version")
    if contract_version not in _LEAN_H_ONLY_SUPPORTED_CONTRACT_VERSIONS:
        return False
    if config.get("allow_runtime_shift_permutation") not in (None, False):
        return False
    runtime_h_partition = config.get("runtime_h_partition")
    if runtime_h_partition != "full":
        return False
    _get_ffn_hadamard_layout(config)
    return True


def _validate_fused_quarot_contract(layer: torch.nn.Module) -> None:
    if not _is_quarot_enabled(layer):
        return
    config = _get_quarot_config(layer)
    if _is_supported_fused_quarot_contract(config):
        return
    bad_items = [f"{key}={config.get(key)!r}" for key in sorted(_LEAN_H_ONLY_QUAROT_CONTRACT)]
    bad_items.append(f"q_mode={config.get('q_mode')!r}")
    bad_items.append(f"contract_version={config.get('contract_version')!r}")
    bad_items.append(f"attention_contract={config.get('attention_contract')!r}")
    if config.get("allow_runtime_shift_permutation") not in (None, False):
        bad_items.append(f"allow_runtime_shift_permutation={config.get('allow_runtime_shift_permutation')!r}")
    bad_items.append(f"runtime_h_partition={config.get('runtime_h_partition')!r}")
    bad_items.append(f"ffn_hadamard_layout={config.get('ffn_hadamard_layout', _FFN_HADAMARD_LAYOUT)!r}")
    raise RuntimeError(
        "QuaRot fused mode only supports the deterministic full-runtime contract "
        f"(layer={_layer_prefix(layer)}; observed: {', '.join(bad_items)})"
    )


def _get_quarot_fold_type(config: dict[str, Any], prefix: str) -> str | None:
    fold_types = config.get("fold_types")
    if isinstance(fold_types, dict):
        for candidate in (prefix, prefix.lower()):
            if candidate in fold_types:
                return fold_types[candidate]
        suffix = prefix.split(".")[-1].lower()
        if suffix == "qkv_proj":
            base_prefix = prefix.rsplit(".", 1)[0]
            sibling_values = []
            for sibling in ("q_proj", "k_proj", "v_proj"):
                for candidate in (
                    f"{base_prefix}.{sibling}",
                    f"{base_prefix}.{sibling}".lower(),
                ):
                    if candidate in fold_types:
                        sibling_values.append(fold_types[candidate])
                        break
            if sibling_values and all(value == "QT_ALPHA_W" for value in sibling_values):
                return "QT_ALPHA_W"
        if suffix == "gate_up_proj":
            base_prefix = prefix.rsplit(".", 1)[0]
            sibling_values = []
            for sibling in ("gate_proj", "up_proj"):
                for candidate in (
                    f"{base_prefix}.{sibling}",
                    f"{base_prefix}.{sibling}".lower(),
                ):
                    if candidate in fold_types:
                        sibling_values.append(fold_types[candidate])
                        break
            if sibling_values and all(value == "QT_ALPHA_W" for value in sibling_values):
                return "QT_ALPHA_W"

    suffix = prefix.split(".")[-1].lower()
    if not _is_lean_h_only_quarot_contract(config):
        return None
    if suffix in {"q_proj", "k_proj", "v_proj", "qkv_proj", "gate_proj", "up_proj", "gate_up_proj"}:
        return "QT_ALPHA_W"
    if suffix in {"o_proj", "out_proj", "down_proj"}:
        return "HWQ"
    return None


def _lean_runtime_input_fold_is_embedded(layer: torch.nn.Module) -> bool:
    config = _get_quarot_config(layer)
    prefix = _layer_prefix(layer)
    fold_type = _get_quarot_fold_type(config, prefix)
    if fold_type != "QT_ALPHA_W":
        return False
    return True


def _lean_attention_requires_runtime_value_rotation(layer: torch.nn.Module) -> bool:
    return False


def _use_reference_linear_path(layer: torch.nn.Module) -> bool:
    prefix = _layer_prefix(layer).lower()
    return "qkv_proj" in prefix or "self_attn.o_proj" in prefix or "self_attn.out_proj" in prefix


def _use_perchannel_native_override_reference_path(layer: torch.nn.Module, group_size: int) -> bool:
    if group_size > 0:
        return False
    return _use_reference_linear_path(layer)


def _grouped_native_override_reference_path(layer: torch.nn.Module) -> bool:
    prefix = _layer_prefix(layer).lower()
    return "qkv_proj" in prefix or "self_attn.o_proj" in prefix or "self_attn.out_proj" in prefix


def _use_grouped_native_linear_path(layer: torch.nn.Module, group_size: int) -> bool:
    if group_size <= 0:
        return False
    if _use_reference_linear_path(layer) and not _grouped_native_override_reference_path(layer):
        return False
    scale = getattr(layer, "weight_scale", None)
    if not isinstance(scale, torch.Tensor) or scale.ndim < 2 or scale.shape[-1] <= 1:
        return False
    offset = getattr(layer, "weight_offset", None)
    if isinstance(offset, torch.Tensor) and offset.numel() > 0:
        if torch.count_nonzero(offset).item() != 0:
            return False
    return True


def _use_grouped_reference_linear_path(layer: torch.nn.Module, group_size: int) -> bool:
    return group_size > 0 and not _use_grouped_native_linear_path(layer, group_size)


def _apply_reference_qkv_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    ref_weight = getattr(layer, "quarot_ref_weight_int8", None)
    if not isinstance(ref_weight, torch.Tensor):
        raise RuntimeError(f"QuaRot qkv reference path missing raw weight for layer={_layer_prefix(layer)}")
    ref_weight = ref_weight.to(device=x.device, dtype=torch.float32)
    scale = layer.weight_scale.data.to(device=x.device, dtype=torch.float32)
    offset = layer.weight_offset.data.to(device=x.device, dtype=torch.float32)
    if scale.ndim == 1:
        scale = scale.view(-1, 1)
    if offset.ndim == 1:
        offset = offset.view(-1, 1)
    if scale.shape[-1] == 1:
        dequant_weight = (ref_weight - offset.view(-1, 1)) * scale.view(-1, 1)
    else:
        out_features, input_size = ref_weight.shape
        num_groups = scale.shape[-1]
        group_size = (input_size + num_groups - 1) // num_groups
        dequant_weight = torch.empty_like(ref_weight, dtype=torch.float32)
        for group_idx in range(num_groups):
            start = group_idx * group_size
            end = min(input_size, start + group_size)
            dequant_weight[:, start:end] = (
                ref_weight[:, start:end] - offset[:, group_idx].unsqueeze(-1)
            ) * scale[:, group_idx].unsqueeze(-1)
    return torch.matmul(x.to(torch.float32), dequant_weight.transpose(0, 1)).to(output_dtype)


def _apply_reference_groupwise_linear(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    x_q, x_scale = _quantize_int4_symmetric(x)
    x_qdq = _dequantize_int4_symmetric(x_q, x_scale, torch.float32)
    return _apply_reference_qkv_linear(layer, x_qdq, output_dtype=output_dtype)


def _apply_grouped_native_weight_quant_batchmatmul(
    layer: torch.nn.Module,
    x: torch.Tensor,
    *,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    x_q, x_scale = _quantize_int4_symmetric(x)
    x_qdq = _dequantize_int4_symmetric(x_q, x_scale, output_dtype)
    kwargs = {
        "x": x_qdq,
        "weight": layer.weight.data,
        "antiquant_scale": layer.weight_scale.data.to(device=x.device, dtype=output_dtype),
        "antiquant_group_size": group_size,
        "bias": None,
    }
    offset = getattr(layer, "weight_offset", None)
    if isinstance(offset, torch.Tensor) and offset.numel() > 0 and torch.count_nonzero(offset).item() != 0:
        kwargs["antiquant_offset"] = offset.data.to(device=x.device, dtype=output_dtype)
    return torch_npu.npu_weight_quant_batchmatmul(
        **kwargs,
    )


def _reshape_qkv_for_attention(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if x.ndim >= 2 and x.shape[-2] == num_heads and x.shape[-1] == head_dim:
        return x
    if x.shape[-1] == num_heads * head_dim:
        return x.reshape(*x.shape[:-1], num_heads, head_dim)
    raise ValueError(
        f"Cannot reshape tensor of shape {tuple(x.shape)} into (..., {num_heads}, {head_dim}) for attention."
    )


def _flatten_attention_output(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 2:
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
    return x


def _softmax_attention_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_heads: int,
    head_dim: int,
    scale: float | torch.Tensor | None,
) -> torch.Tensor:
    q = _reshape_qkv_for_attention(query, num_heads, head_dim).to(torch.float32)
    k = _reshape_qkv_for_attention(key, num_heads, head_dim).to(torch.float32)
    v = _reshape_qkv_for_attention(value, num_heads, head_dim).to(torch.float32)

    # Fallback for unit tests and non-runtime contexts where layer.impl is absent.
    if q.ndim == 3:
        scores = torch.einsum("thd,shd->hts", q, k)
        if scale is not None:
            scores = scores * float(scale)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hts,shd->thd", attn, v)
        return out.to(query.dtype)
    if q.ndim == 4:
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        if scale is not None:
            scores = scores * float(scale)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, v)
        return out.to(query.dtype)
    raise ValueError(f"Unsupported query rank for fallback attention: {q.ndim}")


def _run_attention(
    layer: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache,
    attn_metadata,
    scale,
    output: torch.Tensor | None,
    num_heads: int,
    head_dim: int,
    *,
    require_impl: bool = False,
) -> torch.Tensor:
    impl = getattr(layer, "impl", None)
    if impl is not None and hasattr(impl, "forward"):
        if output is None:
            output = torch.empty(query.shape[0], num_heads * head_dim, dtype=query.dtype, device=query.device)
        return impl.forward(layer, query, key, value, kv_cache, attn_metadata, output)
    if require_impl:
        prefix = _layer_prefix(layer)
        raise RuntimeError(
            f"QuaRot fused attention requires layer.impl.forward for {prefix}; "
            "python attention fallback is debug-only"
        )

    out = _softmax_attention_fallback(query, key, value, num_heads, head_dim, scale)
    out_flat = _flatten_attention_output(out)
    if output is None:
        return out_flat
    output.copy_(out_flat.to(output.dtype))
    return output


def _is_torch_compile_mode() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            try:
                if bool(is_compiling()):
                    return True
            except Exception:
                pass
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        is_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_compiling):
            try:
                if bool(is_compiling()):
                    return True
            except Exception:
                pass
    return False


def _is_pto_hadamard_supported(x: torch.Tensor, kernel_family: str) -> bool:
    dispatch = _resolve_hadamard_dispatch(kernel_family)
    if dispatch.kernel_family != HADAMARD_KERNEL_FAMILY_PTO:
        return False
    if x.device.type != "npu":
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    n = x.shape[-1]
    return n > 0 and (n & (n - 1)) == 0


def _get_pto_hadamard_jit_func():
    global _PTO_HADAMARD_JIT_FUNC
    if _PTO_HADAMARD_JIT_FUNC is not None:
        return _PTO_HADAMARD_JIT_FUNC

    from vllm_ascend.ops.fast_hadamard import jit_compile

    _PTO_HADAMARD_JIT_FUNC = jit_compile(_PTO_KERNEL_CPP_PATH, verbose=False, clean_up=False)
    return _PTO_HADAMARD_JIT_FUNC


def _maybe_warmup_pto_for_process() -> None:
    global _PTO_JIT_WARMUP_DONE
    if _PTO_JIT_WARMUP_DONE:
        return
    dispatch = _resolve_hadamard_dispatch(None)
    if dispatch.kernel_family != HADAMARD_KERNEL_FAMILY_PTO:
        return
    try:
        from vllm_ascend.ops.fast_hadamard import (
            _get_fast_hadamard_dynamic_quant_jit_func,
            ensure_fast_hadamard_dynamic_quant_shared_object,
            ensure_fast_hadamard_shared_object,
        )

        if dispatch.launch_mode == HADAMARD_LAUNCH_MODE_EAGER_JIT:
            _ = ensure_fast_hadamard_shared_object()
            _ = _get_pto_hadamard_jit_func()
        _ = ensure_fast_hadamard_dynamic_quant_shared_object()
        _ = _get_fast_hadamard_dynamic_quant_jit_func()
        _PTO_JIT_WARMUP_DONE = True
    except Exception:
        # Keep runtime robust: fallback path remains available.
        return


def _apply_hadamard_last_dim_pto(x: torch.Tensor, launch_mode: str) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 1:
        return x
    log2_n = int(math.log2(n))
    if launch_mode == HADAMARD_LAUNCH_MODE_EAGER_JIT:
        y = x.clone()
        hadamard_func = _get_pto_hadamard_jit_func()
        hadamard_func(y, y.shape[0], n, log2_n)
        return y / math.sqrt(float(n))

    if launch_mode == HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP:
        return fast_hadamard_last_dim_custom_op(x)

    raise ValueError(f"Unsupported PTO launch mode: {launch_mode}")


def _apply_exact_deterministic_walsh_last_dim(
    x: torch.Tensor,
    kernel_family: str | None = None,
) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 1:
        return x

    dispatch = _resolve_hadamard_dispatch(kernel_family)
    if _is_power_of_two(n):
        rows = x.reshape(-1, n).contiguous()
        if dispatch.kernel_family == HADAMARD_KERNEL_FAMILY_PTO and _is_pto_hadamard_supported(rows, dispatch.kernel_family):
            if dispatch.launch_mode == HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP:
                return fast_hadamard_last_dim_custom_op(rows).reshape_as(x)
            if dispatch.launch_mode == HADAMARD_LAUNCH_MODE_EAGER_JIT:
                pto_rows = rows
                needs_cast_back = False
                if rows.dtype == torch.bfloat16:
                    pto_rows = rows.to(torch.float16)
                    needs_cast_back = True
                try:
                    transformed_rows = _apply_hadamard_last_dim_pto(pto_rows, dispatch.launch_mode)
                    if needs_cast_back:
                        transformed_rows = transformed_rows.to(rows.dtype)
                    return transformed_rows.reshape_as(x)
                except Exception:
                    pass
        return _apply_normalized_hadamard_last_dim(x)

    return _matmul_had_u(x)


def _apply_exact_deterministic_walsh_packed_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"Expected tensor with at least 3 dims, got shape {tuple(x.shape)}")

    original_shape = x.shape
    rows = math.prod(original_shape[:-2])
    n = original_shape[-2]
    channels = original_shape[-1]
    if n <= 1:
        return x

    had_k, k = _get_hadamard_special_factor(n)
    input_tensor = x.reshape(rows, n, channels)
    while input_tensor.shape[1] > k:
        stage_rows, stage_n, stage_channels = input_tensor.shape
        paired = input_tensor.reshape(stage_rows, stage_n // 2, 2, stage_channels)
        even = paired[:, :, 0, :]
        odd = paired[:, :, 1, :]
        output_tensor = torch.empty(
            (stage_rows, stage_n // 2, stage_channels * 2),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        output_tensor[:, :, :stage_channels] = even + odd
        output_tensor[:, :, stage_channels:] = even - odd
        input_tensor = output_tensor

    if k > 1 and had_k is not None:
        input_tensor = had_k.view(1, k, k).to(input_tensor) @ input_tensor

    return input_tensor.reshape(original_shape) / math.sqrt(float(n))


def _apply_exact_deterministic_walsh(
    x: torch.Tensor,
    axis: int = -1,
    kernel_family: str | None = None,
) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Invalid axis {axis} for tensor with {x.ndim} dims")

    moved = x if axis == x.ndim - 1 else x.movedim(axis, -1)
    if not moved.is_contiguous():
        moved = moved.contiguous()
    transformed = _apply_exact_deterministic_walsh_last_dim(moved, kernel_family=kernel_family)
    return transformed if axis == x.ndim - 1 else transformed.movedim(-1, axis)


def _apply_hadamard_family_python(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    if x.numel() == 0:
        return x
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Invalid axis {axis} for tensor with {x.ndim} dims")

    moved = x.movedim(axis, -1)
    d = moved.shape[-1]
    p = _largest_power_of_two_factor(d)
    if p <= 1:
        return x

    groups = d // p
    reshaped = moved.reshape(*moved.shape[:-1], groups, p)
    transformed = _apply_normalized_hadamard_last_dim(reshaped)
    restored = transformed.reshape(*moved.shape[:-1], d).movedim(-1, axis)
    return restored


def _apply_hadamard_family(x: torch.Tensor, axis: int = -1, kernel_family: str | None = None) -> torch.Tensor:
    dispatch = _resolve_hadamard_dispatch(kernel_family)
    if dispatch.kernel_family == HADAMARD_KERNEL_FAMILY_PYTHON:
        return _apply_hadamard_family_python(x, axis=axis)

    if x.numel() == 0:
        return x
    if axis < 0:
        axis += x.ndim
    if axis < 0 or axis >= x.ndim:
        raise ValueError(f"Invalid axis {axis} for tensor with {x.ndim} dims")

    moved = x.movedim(axis, -1)
    if dispatch.launch_mode == HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP and not moved.is_contiguous():
        # Compile mode may lower reshape after movedim into an invalid view on a
        # non-contiguous tensor. Materialize the last-dim layout boundary before
        # flattening into PTO rows.
        moved = moved.contiguous()
    d = moved.shape[-1]
    p = _largest_power_of_two_factor(d)
    if p <= 1:
        return x

    groups = d // p
    grouped = moved.reshape(*moved.shape[:-1], groups, p)
    rows = grouped.reshape(-1, p).contiguous()

    if not _is_pto_hadamard_supported(rows, dispatch.kernel_family):
        return _apply_hadamard_family_python(x, axis=axis)

    if dispatch.launch_mode == HADAMARD_LAUNCH_MODE_COMPILE_CUSTOM_OP:
        transformed_rows = fast_hadamard_last_dim_custom_op(rows)
        transformed = transformed_rows.reshape(*moved.shape[:-1], groups, p)
        restored = transformed.reshape(*moved.shape[:-1], d).movedim(-1, axis)
        return restored

    pto_rows = rows
    needs_cast_back = False
    if dispatch.launch_mode == HADAMARD_LAUNCH_MODE_EAGER_JIT and rows.dtype == torch.bfloat16:
        pto_rows = rows.to(torch.float16)
        needs_cast_back = True

    if dispatch.launch_mode != HADAMARD_LAUNCH_MODE_EAGER_JIT:
        return _apply_hadamard_family_python(x, axis=axis)

    try:
        transformed_rows = _apply_hadamard_last_dim_pto(pto_rows, dispatch.launch_mode)
    except Exception:
        return _apply_hadamard_family_python(x, axis=axis)

    if needs_cast_back:
        transformed_rows = transformed_rows.to(rows.dtype)

    transformed = transformed_rows.reshape(*moved.shape[:-1], groups, p)
    restored = transformed.reshape(*moved.shape[:-1], d).movedim(-1, axis)
    return restored


def _quantize_int4_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    max_abs = x_f.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs / 7.0, min=1e-6)
    quant = torch.round(x_f / scale).clamp(-8, 7).to(torch.int8)
    return quant, scale


def _dequantize_int4_symmetric(quant: torch.Tensor, scale: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    return (quant.to(torch.float32) * scale).to(out_dtype)


def _dynamic_quant_normalize_no_round(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    max_abs = x_f.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs, min=1e-6)
    return (x_f / scale).to(x.dtype), scale.to(x.dtype)


def _quantize_int4_asymmetric_groupwise(
    x: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if x.ndim == 0:
        raise ValueError("Group-wise KV quantization expects tensor rank >= 1")

    original_shape = x.shape
    d = original_shape[-1]
    x_f = x.to(torch.float32).reshape(-1, d)
    rows = x_f.shape[0]
    num_groups = (d + group_size - 1) // group_size

    quant = torch.empty((rows, d), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, num_groups), dtype=torch.float32, device=x.device)
    zeros = torch.empty((rows, num_groups), dtype=torch.float32, device=x.device)

    for g in range(num_groups):
        start = g * group_size
        end = min(d, start + group_size)
        chunk = x_f[:, start:end]
        chunk_min = chunk.amin(dim=-1, keepdim=True)
        chunk_max = chunk.amax(dim=-1, keepdim=True)
        scale = torch.clamp((chunk_max - chunk_min) / 15.0, min=1e-6)
        zero = torch.round(-chunk_min / scale)
        q = torch.round(chunk / scale + zero).clamp(0, 15).to(torch.int8)

        quant[:, start:end] = q
        scales[:, g] = scale.squeeze(-1)
        zeros[:, g] = zero.squeeze(-1)

    return quant.reshape(original_shape), scales, zeros


def _dequantize_int4_asymmetric_groupwise(
    quant: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    original_shape: torch.Size,
    out_dtype: torch.dtype,
    group_size: int = 128,
) -> torch.Tensor:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    if len(original_shape) == 0:
        raise ValueError("Group-wise KV dequantization expects tensor rank >= 1")

    d = original_shape[-1]
    quant_f = quant.to(torch.float32).reshape(-1, d)
    rows = quant_f.shape[0]
    num_groups = (d + group_size - 1) // group_size
    if scales.shape != (rows, num_groups) or zeros.shape != (rows, num_groups):
        raise ValueError(
            "Invalid group-wise KV quant metadata shape: "
            f"scales={tuple(scales.shape)}, zeros={tuple(zeros.shape)}, expected={(rows, num_groups)}"
        )

    out = torch.empty((rows, d), dtype=torch.float32, device=quant.device)
    for g in range(num_groups):
        start = g * group_size
        end = min(d, start + group_size)
        scale = scales[:, g].unsqueeze(-1)
        zero = zeros[:, g].unsqueeze(-1)
        out[:, start:end] = (quant_f[:, start:end] - zero) * scale

    return out.reshape(original_shape).to(out_dtype)


def _get_quarot_shape(layer: torch.nn.Module) -> tuple[int, int]:
    config = getattr(layer, "quarot_config", None)
    if not isinstance(config, dict):
        return (0, 0)
    num_heads = int(config.get("num_heads", 0) or 0)
    head_dim = int(config.get("head_dim", 0) or 0)
    return num_heads, head_dim


def _get_quarot_max_tp_size(layer: torch.nn.Module) -> int:
    config = getattr(layer, "quarot_config", None)
    if not isinstance(config, dict):
        return -1
    max_tp_size = int(config.get("max_tp_size", -1) or -1)
    return max_tp_size if max_tp_size > 0 else -1


def _get_quarot_runtime_h_partition(layer: torch.nn.Module) -> str:
    config = getattr(layer, "quarot_config", None)
    if not isinstance(config, dict):
        return "tp_blocked"
    runtime_h_partition = config.get("runtime_h_partition")
    return "full" if runtime_h_partition == "full" else "tp_blocked"


def _get_quarot_runtime_block_size(layer: torch.nn.Module) -> int | None:
    if _get_quarot_runtime_h_partition(layer) == "full":
        return None
    return _get_quarot_max_tp_size(layer)


def _get_kv_group_size(layer: torch.nn.Module) -> int:
    config = getattr(layer, "quarot_config", None)
    if not isinstance(config, dict):
        return 1
    group_size = config.get("kv_group_size", 1)
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"QuaRot kv_group_size must be a positive integer, got {group_size!r}")
    return group_size


def _apply_headwise_hadamard(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if num_heads <= 0 or head_dim <= 0:
        return _apply_hadamard_family(x, axis=-1)
    if x.ndim >= 2 and x.shape[-2] == num_heads and x.shape[-1] == head_dim:
        return _apply_hadamard_family(x, axis=-1)
    if x.shape[-1] == num_heads * head_dim:
        y = x.reshape(*x.shape[:-1], num_heads, head_dim)
        y = _apply_hadamard_family(y, axis=-1)
        return y.reshape(*x.shape[:-1], num_heads * head_dim)
    if x.shape[-1] == head_dim:
        return _apply_hadamard_family(x, axis=-1)
    return _apply_hadamard_family(x, axis=-1)


def _apply_heads_mixing_hadamard(
    x: torch.Tensor,
    num_heads: int,
    head_dim: int,
    block_size: int | None = None,
) -> torch.Tensor:
    if num_heads <= 0 or head_dim <= 0:
        return _apply_hadamard_family(x, axis=-1)
    if block_size is not None and block_size > 0 and block_size < num_heads:
        rot = _get_block_diag_walsh_matrix(num_heads, block_size, x.dtype, x.device)
        return _apply_heads_rotation_tensor(x, rot)
    if x.ndim >= 2 and x.shape[-2] == num_heads and x.shape[-1] == head_dim:
        return _apply_hadamard_family(x, axis=-2)
    if x.shape[-1] == num_heads * head_dim:
        y = x.reshape(*x.shape[:-1], num_heads, head_dim)
        y = _apply_hadamard_family(y, axis=-2)
        return y.reshape(*x.shape[:-1], num_heads * head_dim)
    return _apply_hadamard_family(x, axis=-1)


def _use_perchannel_fused_down_proj_dynamic_quant(layer: torch.nn.Module, group_size: int, input_width: int) -> bool:
    if os.getenv("VLLM_ASCEND_QUAROT_DISABLE_FUSED_HADAMARD_QUANT") == "1":
        return False
    if group_size > 0 or not _is_quarot_enabled(layer):
        return False
    prefix = _layer_prefix(layer).lower()
    if "down_proj" not in prefix:
        return False
    _, hadamard_n = _get_ffn_hadamard_dims(
        input_width,
        _get_ffn_hadamard_layout(_get_quarot_config(layer)),
    )
    return hadamard_n < input_width


def _apply_heads_rotation_tensor(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(f"Invalid heads rotation shape {tuple(rotation.shape)}; expected square matrix.")
    num_heads = int(rotation.shape[0])
    if x.shape[-1] % num_heads != 0:
        raise ValueError(
            f"Cannot apply heads rotation with num_heads={num_heads} to activation shape {tuple(x.shape)}."
        )
    init_shape = x.shape
    head_dim = x.shape[-1] // num_heads
    reshaped = x.reshape(-1, num_heads, head_dim)
    rot = rotation if rotation.dtype == x.dtype else rotation.to(dtype=x.dtype)
    rotated = torch.matmul(rot.T, reshaped)
    return rotated.reshape(init_shape)


def _apply_kronecker_rotation_tensors(
    x: torch.Tensor,
    rotation_m: torch.Tensor,
    rotation_n: torch.Tensor,
) -> torch.Tensor:
    if rotation_m.ndim != 2 or rotation_m.shape[0] != rotation_m.shape[1]:
        raise ValueError(f"Invalid kronecker_rotation_m shape {tuple(rotation_m.shape)}; expected square matrix.")
    if rotation_n.ndim != 2 or rotation_n.shape[0] != rotation_n.shape[1]:
        raise ValueError(f"Invalid kronecker_rotation_n shape {tuple(rotation_n.shape)}; expected square matrix.")
    m = int(rotation_m.shape[0])
    n = int(rotation_n.shape[0])
    if x.shape[-1] != m * n:
        raise ValueError(
            f"Cannot apply Kronecker rotation with ({m}, {n}) to activation shape {tuple(x.shape)}."
        )
    init_shape = x.shape
    reshaped = x.reshape(-1, m, n)
    rot_m = rotation_m if rotation_m.dtype == x.dtype else rotation_m.to(dtype=x.dtype)
    rot_n = rotation_n if rotation_n.dtype == x.dtype else rotation_n.to(dtype=x.dtype)
    rotated = torch.matmul(reshaped, rot_n)
    rotated = torch.matmul(rot_m.T, rotated)
    return rotated.reshape(init_shape)


def _move_quarot_rotation_buffers_to_layer_device(layer: torch.nn.Module) -> None:
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return
    target_device = weight.device
    if target_device.type == "cpu":
        return

    for name in (
        "quarot_heads_rotation",
        "quarot_kronecker_rotation_m",
        "quarot_kronecker_rotation_n",
    ):
        buf = getattr(layer, name, None)
        if not isinstance(buf, torch.Tensor):
            continue
        if buf.device == target_device:
            continue
        _register_or_update_buffer(layer, name, buf.to(device=target_device))


def _get_model_dir_for_native_quarot() -> str | None:
    try:
        vllm_config = get_current_vllm_config()
    except Exception:
        return None
    model_path = getattr(vllm_config.model_config, "model", None)
    if not isinstance(model_path, str):
        return None
    if not os.path.isdir(model_path):
        return None
    return model_path


def _load_native_quarot_index(model_dir: str) -> dict[str, str]:
    cached = _NATIVE_QUAROT_INDEX_CACHE.get(model_dir)
    if cached is not None:
        return cached
    index_path = os.path.join(model_dir, _MODELSLIM_WEIGHT_INDEX_FILENAME)
    if not os.path.isfile(index_path):
        _NATIVE_QUAROT_INDEX_CACHE[model_dir] = {}
        return {}
    with open(index_path, encoding="utf-8") as f:
        data = json.load(f)
    weight_map = data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        weight_map = {}
    index = {str(k): str(v) for k, v in weight_map.items()}
    _NATIVE_QUAROT_INDEX_CACHE[model_dir] = index
    return index


def _load_native_quarot_tensor(model_dir: str, tensor_key: str) -> torch.Tensor | None:
    cache_key = (model_dir, tensor_key)
    cached = _NATIVE_QUAROT_TENSOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    weight_map = _load_native_quarot_index(model_dir)
    shard_name = weight_map.get(tensor_key)
    if shard_name is None:
        return None
    shard_path = os.path.join(model_dir, shard_name)
    if not os.path.isfile(shard_path):
        return None

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        if tensor_key not in f.keys():
            return None
        tensor = f.get_tensor(tensor_key).detach().clone()
    _NATIVE_QUAROT_TENSOR_CACHE[cache_key] = tensor
    return tensor


def _register_or_update_buffer(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if hasattr(layer, name):
        setattr(layer, name, value)
    else:
        layer.register_buffer(name, value, persistent=False)


def _attach_native_quarot_rotation_tensors(layer: torch.nn.Module) -> None:
    if not _is_quarot_enabled(layer):
        return
    config = _get_quarot_config(layer)
    export_rotation_tensors = config.get("export_rotation_tensors")
    _validate_fused_quarot_contract(layer)
    if export_rotation_tensors:
        raise RuntimeError(
            f"QuaRot fused mode rejects dense rotation tensors (layer={_layer_prefix(layer)})."
        )
    if export_rotation_tensors is False:
        return
    model_dir = _get_model_dir_for_native_quarot()
    if model_dir is None:
        return
    prefix = _layer_prefix(layer)
    lower_prefix = prefix.lower()
    if "self_attn.o_proj" in lower_prefix or "self_attn.out_proj" in lower_prefix:
        heads_rotation = _load_native_quarot_tensor(model_dir, f"{prefix}.heads_rotation")
        if heads_rotation is not None:
            _register_or_update_buffer(layer, "quarot_heads_rotation", heads_rotation)
    if "mlp.down_proj" in lower_prefix:
        rotation_m = _load_native_quarot_tensor(model_dir, f"{prefix}.kronecker_rotation_m")
        rotation_n = _load_native_quarot_tensor(model_dir, f"{prefix}.kronecker_rotation_n")
        if rotation_m is not None and rotation_n is not None:
            _register_or_update_buffer(layer, "quarot_kronecker_rotation_m", rotation_m)
            _register_or_update_buffer(layer, "quarot_kronecker_rotation_n", rotation_n)


@register_scheme(QUAROT_LINEAR_QUANT_TYPE, "linear")
class AscendW4A4QuaRotDynamicLinearMethod(AscendLinearScheme):
    """W4A4 dynamic linear scheme for lean_h_only fused serving."""

    def __init__(self, *, float_weight: bool = False):
        self.transpose_weight = True
        self.float_weight = float_weight
        try:
            vllm_config = get_current_vllm_config()
            self.group_size = int(vllm_config.quant_config.quant_description.get("group_size", 0) or 0)
        except Exception:
            self.group_size = 0
        if self.float_weight:
            self.group_size = 0

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if self.float_weight:
            return {"weight": torch.empty(output_size, input_size, dtype=params_dtype)}
        return {"weight": torch.empty(output_size, input_size, dtype=torch.int8)}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        if self.float_weight:
            return {}
        if self.group_size > 0:
            return {}
        return {
            "weight_scale": torch.empty(output_size, 1, dtype=torch.float32),
            "weight_offset": torch.empty(output_size, 1, dtype=torch.float32),
        }

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        if self.float_weight:
            return {}
        if self.group_size <= 0:
            return {}
        num_groups = (input_size + self.group_size - 1) // self.group_size
        return {
            "weight_scale": torch.empty(output_size, num_groups, dtype=torch.float32),
            "weight_offset": torch.empty(output_size, num_groups, dtype=torch.float32),
        }

    def maybe_apply_quarot_transform(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        prefix = _layer_prefix(layer).lower()
        heads_rotation = getattr(layer, "quarot_heads_rotation", None)
        if isinstance(heads_rotation, torch.Tensor):
            return _apply_heads_rotation_tensor(x, heads_rotation)

        rotation_m = getattr(layer, "quarot_kronecker_rotation_m", None)
        rotation_n = getattr(layer, "quarot_kronecker_rotation_n", None)
        if isinstance(rotation_m, torch.Tensor) and isinstance(rotation_n, torch.Tensor):
            return _apply_kronecker_rotation_tensors(x, rotation_m, rotation_n)

        if _is_quarot_enabled(layer) and _is_float_dtype(x.dtype):
            runtime_block_size = _get_quarot_runtime_block_size(layer)
            if "down_proj" in prefix:
                return _apply_matrix_free_down_proj_rotation(
                    x,
                    block_size=runtime_block_size,
                    layout=_get_ffn_hadamard_layout(_get_quarot_config(layer)),
                )
            if "o_proj" in prefix or "out_proj" in prefix:
                return x

        if "down_proj" in prefix:
            return _apply_matrix_free_down_proj_rotation(
                x,
                block_size=_get_quarot_runtime_block_size(layer),
                layout=_get_ffn_hadamard_layout(_get_quarot_config(layer)),
            )
        if "o_proj" in prefix or "out_proj" in prefix:
            return x
        return x

    def _apply_float_weight_linear(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        *,
        pertoken_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight.data
        if x.dtype != weight.dtype:
            raise TypeError(
                f"QuaRot float-weight linear expects activation dtype to "
                f"match weight dtype, got activation={x.dtype}, weight={weight.dtype} "
                f"for layer {_layer_prefix(layer)}"
            )
        output = torch.matmul(x, weight.transpose(0, 1))
        if pertoken_scale is not None:
            if pertoken_scale.dtype != output.dtype:
                raise TypeError(
                    f"QuaRot float-weight linear expects per-token scale dtype to "
                    f"match output dtype, got scale={pertoken_scale.dtype}, output={output.dtype} "
                    f"for layer {_layer_prefix(layer)}"
                )
            output = output * pertoken_scale.reshape(*output.shape[:-1], 1)
        return output

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        _validate_fused_quarot_contract(layer)
        prefix = _layer_prefix(layer)
        original_dtype = x.dtype
        use_reference_linear = _use_reference_linear_path(layer)
        use_perchannel_native_linear = _use_perchannel_native_override_reference_path(layer, self.group_size)
        use_grouped_native_linear = _use_grouped_native_linear_path(layer, self.group_size)
        use_grouped_reference_linear = _use_grouped_reference_linear_path(layer, self.group_size)
        use_fused_down_proj_dynamic_quant = (
            not use_grouped_native_linear
            and not use_grouped_reference_linear
            and not use_reference_linear
            and _use_perchannel_fused_down_proj_dynamic_quant(layer, self.group_size, x.shape[-1])
        )

        if not use_fused_down_proj_dynamic_quant:
            x = self.maybe_apply_quarot_transform(layer, x)
        _maybe_log_value_path(layer, "linear_input", x)
        input_for_basis_log = x

        if use_grouped_native_linear:
            output = _apply_grouped_native_weight_quant_batchmatmul(
                layer,
                x,
                group_size=self.group_size,
                output_dtype=original_dtype,
            )
        elif use_grouped_reference_linear:
            output = _apply_reference_groupwise_linear(layer, x, output_dtype=original_dtype)
        elif use_reference_linear and not use_perchannel_native_linear:
            output = _apply_reference_qkv_linear(layer, x, output_dtype=original_dtype)
        elif self.float_weight:
            if use_fused_down_proj_dynamic_quant:
                x_for_linear = _apply_matrix_free_down_proj_rotation(
                    x,
                    block_size=_get_quarot_runtime_block_size(layer),
                    layout=_get_ffn_hadamard_layout(_get_quarot_config(layer)),
                )
            else:
                x_for_linear = x
            x_for_linear, pertoken_scale = _dynamic_quant_normalize_no_round(x_for_linear)
            output = self._apply_float_weight_linear(
                layer,
                x_for_linear,
                pertoken_scale=pertoken_scale,
            )
        else:
            if use_fused_down_proj_dynamic_quant:
                _, hadamard_n = _get_ffn_hadamard_dims(
                    x.shape[-1],
                    _get_ffn_hadamard_layout(_get_quarot_config(layer)),
                )
                quant_x, pertoken_scale = fast_hadamard_dynamic_quant_blockwise_last_dim(x, hadamard_n)
            else:
                quant_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
            pertoken_scale = pertoken_scale.reshape(-1).to(torch.float32)

            output = torch_npu.npu_quant_matmul(
                quant_x,
                layer.weight.data,
                scale=layer.weight_scale.data.view(-1).to(torch.float32),
                pertoken_scale=pertoken_scale,
                bias=None,
                output_dtype=original_dtype,
            )
        _maybe_log_qkv_reference_trace(
            layer,
            use_reference=(use_reference_linear or use_grouped_reference_linear),
            x=x,
            output=output,
        )
        if bias is not None:
            output = output + bias.to(original_dtype)
        _maybe_log_value_path(layer, "linear_output", output)
        _maybe_log_o_proj_basis_views(layer, linear_input=input_for_basis_log, linear_output=output)
        return output

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _maybe_warmup_pto_for_process()
        _validate_fused_quarot_contract(layer)
        if self.float_weight:
            _attach_native_quarot_rotation_tensors(layer)
            _move_quarot_rotation_buffers_to_layer_device(layer)
            return
        _attach_native_quarot_rotation_tensors(layer)
        layer.weight_scale.data = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.to(torch.float32)
        prefix = _layer_prefix(layer).lower()
        use_perchannel_native_linear = _use_perchannel_native_override_reference_path(layer, self.group_size)
        use_grouped_native_linear = _use_grouped_native_linear_path(layer, self.group_size)
        if (
            _use_grouped_reference_linear_path(layer, self.group_size)
            or ("qkv_proj" in prefix and not use_grouped_native_linear and not use_perchannel_native_linear)
            or (
                ("self_attn.o_proj" in prefix or "self_attn.out_proj" in prefix)
                and not use_grouped_native_linear
                and not use_perchannel_native_linear
            )
        ):
            _register_or_update_buffer(layer, "quarot_ref_weight_int8", layer.weight.data.to(torch.int8).clone())
        if use_grouped_native_linear:
            logical_weight_kn = layer.weight.data.to(torch.int32).transpose(0, 1).contiguous()
            layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(logical_weight_kn).contiguous()
            layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1).contiguous()
            layer.weight_offset.data = layer.weight_offset.data.transpose(0, 1).contiguous()
        else:
            layer.weight.data = torch_npu.npu_convert_weight_to_int4pack(layer.weight.data.to(torch.int32))
        if self.transpose_weight and not use_grouped_native_linear:
            layer.weight.data = layer.weight.data.transpose(-1, -2)
        _move_quarot_rotation_buffers_to_layer_device(layer)


class _AscendQuaRotAttentionBase(AscendAttentionScheme):
    """Shared QuaRot attention implementation for fused lean_h_only serving."""

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Keep parity with BaseKVCacheMethod behavior expected by attention layers.
        if not hasattr(layer, "q_scale"):
            layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
            layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
            layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
            layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        _maybe_warmup_pto_for_process()

    def _apply_fused(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        scale,
        output,
        *,
        num_heads: int,
        head_dim: int,
        require_impl: bool = False,
    ) -> torch.Tensor:
        _validate_fused_quarot_contract(layer)
        _maybe_log_value_path(layer, "attention_query_input", query)
        _maybe_log_value_path(layer, "attention_key_input", key)
        _maybe_log_value_path(layer, "attention_value_input", value)
        if not use_native_quarot_kv_cache():
            query = _apply_headwise_hadamard(query, num_heads, head_dim)
            key = _apply_headwise_hadamard(key, num_heads, head_dim)
        if _lean_attention_requires_runtime_value_rotation(layer):
            value = _apply_headwise_hadamard(value, num_heads, head_dim)
        _maybe_log_value_path(layer, "attention_query_runtime_h", query)
        _maybe_log_value_path(layer, "attention_key_runtime_h", key)
        _maybe_log_value_path(layer, "attention_value_runtime_h", value)
        _maybe_log_attention_inputs(layer, query=query, key=key, value=value)
        result = _run_attention(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            scale,
            output,
            num_heads=num_heads,
            head_dim=head_dim,
            require_impl=require_impl,
        )
        _maybe_log_attention_result(layer, "post_run_attention", result)
        _maybe_log_attention_compare(
            layer,
            query,
            key,
            value,
            result,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
        )
        return result

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        num_heads, head_dim = _get_quarot_shape(layer)
        return self._apply_fused(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            scale,
            output,
            num_heads=num_heads,
            head_dim=head_dim,
        )


@register_scheme(QUAROT_ATTN_QUANT_TYPE, "attention")
class AscendQuaRotAttentionMethod(_AscendQuaRotAttentionBase):
    """Production QuaRot attention scheme for fused serving."""

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        num_heads, head_dim = _get_quarot_shape(layer)
        return self._apply_fused(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            scale,
            output,
            num_heads=num_heads,
            head_dim=head_dim,
            require_impl=True,
        )
