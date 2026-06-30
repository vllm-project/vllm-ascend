#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any, NoReturn, cast

import torch
import torch_npu
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.rotary_embedding import get_rope_cache

_UNSUPPORTED_C_ASCEND_OPS: set[str] = set()


try:
    from torch._dynamo import disable as _dynamo_disable  # type: ignore[attr-defined]
except Exception:  # pragma: no cover

    def _dynamo_disable(fn: Callable):  # type: ignore[misc]
        return fn


def _dynamo_disable_preserve_cache(fn: Callable):
    disabled = _dynamo_disable(fn)
    for attr in ("cache_clear", "cache_info", "cache_parameters"):
        if hasattr(fn, attr) and not hasattr(disabled, attr):
            setattr(disabled, attr, getattr(fn, attr))
    return disabled


def clear_rope_cache_op_capability_cache() -> None:
    _ensure_c_ascend_custom_ops_loaded.cache_clear()
    _get_torch_npu_op.cache_clear()
    _get_c_ascend_op.cache_clear()
    _get_c_ascend_composite_op.cache_clear()
    _c_aclnn_api_available.cache_clear()
    cast(Any, _get_vllm_op).cache_clear()
    _get_triton_rope_forward_siso.cache_clear()
    _get_triton_interleave_rope_by_cache.cache_clear()
    _get_triton_kv_rmsnorm_rope_cache_by_cache.cache_clear()
    _UNSUPPORTED_C_ASCEND_OPS.clear()


def _mark_c_ascend_op_unsupported(name: str) -> None:
    _UNSUPPORTED_C_ASCEND_OPS.add(name)
    _get_c_ascend_op.cache_clear()


@lru_cache(maxsize=1)
def _ensure_c_ascend_custom_ops_loaded() -> bool:
    try:
        from vllm_ascend.utils import enable_custom_op

        return bool(enable_custom_op())
    except Exception:
        return False


@_dynamo_disable_preserve_cache
@cache
def _get_torch_npu_op(name: str):
    op = getattr(torch_npu, name, None)
    return op if callable(op) else None


@_dynamo_disable
def _has_dispatch_kernel(op, dispatch_key: str) -> bool:
    overload = getattr(op, "default", op)
    has_kernel = getattr(overload, "has_kernel_for_dispatch_key", None)
    if has_kernel is None:
        return True
    try:
        return bool(has_kernel(dispatch_key))
    except RuntimeError:
        return False


@_dynamo_disable
def _get_c_ascend_attr(name: str):
    c_ascend = getattr(torch.ops, "_C_ascend", None)
    if c_ascend is None:
        return None
    try:
        return getattr(c_ascend, name)
    except AttributeError:
        return None


@_dynamo_disable_preserve_cache
@cache
def _get_c_ascend_op(name: str):
    if name in _UNSUPPORTED_C_ASCEND_OPS:
        return None
    op = _get_c_ascend_attr(name)
    if op is None:
        _ensure_c_ascend_custom_ops_loaded()
        op = _get_c_ascend_attr(name)
    if op is None:
        return None
    if not callable(op) or not _has_dispatch_kernel(op, "PrivateUse1"):
        return None
    return op


@_dynamo_disable_preserve_cache
@cache
def _get_c_ascend_composite_op(name: str):
    op = _get_c_ascend_attr(name)
    if op is None:
        _ensure_c_ascend_custom_ops_loaded()
        op = _get_c_ascend_attr(name)
    if op is None:
        return None
    return op if callable(op) else None


@_dynamo_disable_preserve_cache
@cache
def _c_aclnn_api_available(api_name: str) -> bool:
    capability_op = _get_c_ascend_composite_op("aclnn_api_available")
    if capability_op is None:
        return False
    try:
        return bool(capability_op(api_name))
    except (RuntimeError, TypeError):
        return False


def _torch_compiler_is_compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _get_vllm_op_unchecked(name: str):
    vllm_namespace = getattr(torch.ops, "vllm", None)
    if vllm_namespace is None:
        return None
    try:
        op = getattr(vllm_namespace, name)
    except AttributeError:
        return None
    return op if callable(op) else None


@cache
def _get_vllm_op_checked(name: str):
    op = _get_vllm_op_unchecked(name)
    if op is None or not _has_dispatch_kernel(op, "PrivateUse1"):
        return None
    return op


def _get_vllm_op(name: str):
    if _torch_compiler_is_compiling():
        return _get_vllm_op_unchecked(name)
    return _get_vllm_op_checked(name)


for _cache_attr in ("cache_clear", "cache_info", "cache_parameters"):
    setattr(_get_vllm_op, _cache_attr, getattr(_get_vllm_op_checked, _cache_attr))


def has_mla_preprocess_by_cache_kernel() -> bool:
    return _get_c_ascend_op("mla_preprocess_by_cache") is not None


def has_mla_preprocess_by_cache_backend() -> bool:
    return has_mla_preprocess_by_cache_kernel()


def has_mla_prolog_v2_by_cache_kernel() -> bool:
    return _get_torch_npu_op("npu_mla_prolog_v2_by_cache") is not None


def has_mla_prolog_v3_by_cache_kernel() -> bool:
    return _get_torch_npu_op("npu_mla_prolog_v3_by_cache") is not None


def has_inplace_partial_rotary_mul_by_cache_kernel() -> bool:
    return _c_aclnn_api_available("aclnnInplacePartialRotaryMulByCache")


def has_compressor_by_cache_kernel() -> bool:
    return _c_aclnn_api_available("aclnnCompressorByCache")


def has_split_qkv_tp_rmsnorm_rope_by_cache_backend() -> bool:
    return _get_vllm_op("split_qkv_tp_rmsnorm_rope_by_cache") is not None


def has_split_qkv_rmsnorm_mrope_by_cache_backend() -> bool:
    return _get_vllm_op("triton_split_qkv_rmsnorm_mrope_by_cache") is not None


@lru_cache(maxsize=1)
def _get_triton_rope_forward_siso():
    if not HAS_TRITON:
        return None
    try:
        from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

        init_device_properties_triton()
    except Exception:
        return None
    return rope_forward_triton_siso


@lru_cache(maxsize=1)
def _get_triton_interleave_rope_by_cache():
    if not HAS_TRITON:
        return None
    try:
        from vllm_ascend.ops.triton.rope import interleave_rope_by_cache_triton
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

        init_device_properties_triton()
    except Exception:
        return None
    return interleave_rope_by_cache_triton


@lru_cache(maxsize=1)
def _get_triton_kv_rmsnorm_rope_cache_by_cache():
    if not HAS_TRITON:
        return None
    try:
        from vllm_ascend.ops.triton.kv_rmsnorm_rope_cache import kv_rmsnorm_rope_cache_by_cache_triton
        from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

        init_device_properties_triton()
    except Exception:
        return None
    return kv_rmsnorm_rope_cache_by_cache_triton


def _rope_dim(rotary_emb, ref_tensor: torch.Tensor) -> int:
    return int(getattr(rotary_emb, "rotary_dim", ref_tensor.shape[-1]))


def _is_neox_style(rotary_emb) -> bool:
    return bool(getattr(rotary_emb, "is_neox_style", True))


def _validate_positions_1d(positions: torch.Tensor, expected_num_tokens: int, op_name: str) -> None:
    if positions.dim() != 1:
        raise ValueError(f"{op_name} expects 1D positions, got shape {tuple(positions.shape)}")
    if positions.shape[0] != expected_num_tokens:
        raise ValueError(
            f"{op_name} positions length must match token count: "
            f"got {positions.shape[0]}, expected {expected_num_tokens}"
        )
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{op_name} positions must be int32 or int64, got {positions.dtype}")


def _normalize_positions_1d(positions: torch.Tensor, expected_num_tokens: int, op_name: str) -> torch.Tensor:
    _validate_positions_1d(positions, expected_num_tokens, op_name)
    return positions.contiguous()


def _raise_missing_true_by_cache_backend(op_name: str) -> NoReturn:
    raise RuntimeError(
        f"{op_name} requires a true by-cache backend that reads rope cache by positions inside the kernel. "
        "No native or Triton by-cache backend is available; refusing to materialize sin/cos tensors."
    )


def _resolve_rope_mode(rotary_emb, rotary_mode: str | None) -> bool | None:
    if rotary_mode is None:
        return _is_neox_style(rotary_emb)
    if rotary_mode == "half":
        return True
    if rotary_mode == "interleave":
        return False
    return None


def _try_triton_rotary_siso_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    *,
    rotary_mode: str | None = None,
    partial_slice: list[int] | None = None,
    layout: str = "T11D",
    rope_dim_offset: int = 0,
    inverse: bool = False,
    fp32_compute: bool = False,
    inplace: bool = False,
) -> torch.Tensor | None:
    rope_forward = _get_triton_rope_forward_siso()
    if rope_forward is None:
        return None
    if layout != "T11D" or positions.dim() != 1:
        return None

    is_neox_style = _resolve_rope_mode(rotary_emb, rotary_mode)
    if is_neox_style is None:
        return None

    x_for_rope = x.to(torch.float32) if fp32_compute else x
    if x_for_rope.dim() == 4:
        if x_for_rope.shape[2] != 1:
            return None
        qk = x_for_rope.reshape(x_for_rope.shape[0], x_for_rope.shape[1], x_for_rope.shape[3])
    elif x_for_rope.dim() == 3:
        qk = x_for_rope
    else:
        return None

    if positions.shape[0] != qk.shape[0]:
        return None

    rope_dim = _rope_dim(rotary_emb, x_for_rope)
    if partial_slice is not None:
        if len(partial_slice) != 2:
            return None
        start, end = partial_slice
        if start != 0:
            return None
        rope_dim = end

    if rope_dim <= 0 or rope_dim > qk.shape[-1] or rope_dim_offset < 0 or rope_dim_offset % 2 != 0:
        return None

    qk_work = qk if inplace and qk.is_contiguous() else qk.contiguous().clone()
    output = rope_forward(
        qk_work,
        cos_sin_cache=get_rope_cache(rotary_emb, qk_work),
        positions=positions,
        rope_dim=rope_dim,
        rope_dim_offset=rope_dim_offset,
        is_neox_style=is_neox_style,
        inverse=inverse,
    ).reshape(x_for_rope.shape)
    if fp32_compute:
        output = output.to(x.dtype)
    return output


def _try_triton_interleave_rope_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor | None:
    interleave_rope = _get_triton_interleave_rope_by_cache()
    if interleave_rope is None or positions.dim() != 1:
        return None
    if x.dim() != 4 or x.shape[2] != 1:
        return None

    rope_dim = _rope_dim(rotary_emb, x)
    if rope_dim != x.shape[-1] or rope_dim % 2 != 0:
        return None
    if positions.shape[0] != x.shape[0]:
        return None
    if positions.dtype not in (torch.int32, torch.int64):
        return None
    positions = positions.contiguous()

    qk = x.squeeze(2)
    output = interleave_rope(
        qk,
        cos_sin_cache=get_rope_cache(rotary_emb, qk),
        positions=positions,
        rope_dim=rope_dim,
        is_neox_style=_is_neox_style(rotary_emb),
    )
    return output.reshape(x.shape)


def _try_c_ascend_interleave_rope_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor | None:
    native_op = _get_c_ascend_op("interleave_rope_by_cache")
    if native_op is None or x.dim() != 4 or x.shape[2] != 1:
        return None

    rope_dim = _rope_dim(rotary_emb, x)
    if rope_dim != x.shape[-1] or rope_dim % 32 != 0:
        return None
    if positions.dim() != 1 or positions.shape[0] != x.shape[0]:
        return None
    if positions.dtype not in (torch.int32, torch.int64):
        return None

    qk = x.squeeze(2)
    if qk.stride(-1) != 1:
        return None

    cos_sin_cache = get_rope_cache(rotary_emb, qk)
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] < rope_dim or cos_sin_cache.stride(-1) != 1:
        return None

    output = native_op(
        qk,
        _normalize_positions_1d(positions, qk.shape[0], "interleave_rope_by_cache"),
        cos_sin_cache,
        rope_dim,
        _is_neox_style(rotary_emb),
    )
    return output.reshape(x.shape)


def _try_triton_kv_rmsnorm_rope_cache_by_cache(
    kv_no_split: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    slots: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    *,
    c_kv_scale=None,
    epsilon=None,
    cache_mode: str = "PA",
    is_output_kv: bool = False,
    rope_dim: int | None = None,
    is_neox_style: bool | None = None,
):
    triton_op = _get_triton_kv_rmsnorm_rope_cache_by_cache()
    if triton_op is None or cache_mode not in ("PA", "PA_NZ") or c_kv_scale is not None or epsilon is None:
        return None
    if kv_no_split.dim() != 4 or kv_no_split.shape[1] != 1 or kv_no_split.shape[2] != 1:
        return None
    if weight.dim() != 1 or positions.dim() != 1 or slots.dim() != 1:
        return None
    if positions.shape[0] != kv_no_split.shape[0] or slots.shape[0] != kv_no_split.shape[0]:
        return None
    if positions.dtype not in (torch.int32, torch.int64):
        return None
    if slots.dtype not in (torch.int32, torch.int64):
        return None
    if kv_cache_rope.dim() != 4 or kv_cache_nope.dim() != 4:
        return None
    if kv_cache_rope.shape[:3] != kv_cache_nope.shape[:3]:
        return None
    if kv_cache_rope.shape[1] <= 0 or kv_cache_rope.shape[2] != 1:
        return None
    if kv_cache_nope.shape[2] != 1:
        return None

    resolved_rope_dim = int(rope_dim if rope_dim is not None else kv_cache_rope.shape[-1])
    nope_dim = int(weight.shape[0])
    if resolved_rope_dim <= 0 or resolved_rope_dim % 2 != 0:
        return None
    if kv_cache_rope.shape[-1] != resolved_rope_dim or kv_cache_nope.shape[-1] != nope_dim:
        return None
    if kv_no_split.shape[-1] != nope_dim + resolved_rope_dim:
        return None

    return triton_op(
        kv_no_split,
        weight,
        positions,
        get_rope_cache(rotary_emb, kv_no_split),
        slots,
        kv_cache_rope,
        kv_cache_nope,
        epsilon=float(epsilon),
        rope_dim=resolved_rope_dim,
        is_neox_style=_is_neox_style(rotary_emb) if is_neox_style is None else bool(is_neox_style),
        is_output_kv=is_output_kv,
        cache_mode_is_nz=cache_mode == "PA_NZ",
    )


def _try_c_ascend_kv_rmsnorm_rope_cache_by_cache(
    kv_no_split: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    slots: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    *,
    c_kv_scale=None,
    epsilon=None,
    cache_mode: str = "PA",
    is_output_kv: bool = False,
    rope_dim: int | None = None,
    is_neox_style: bool | None = None,
):
    native_op = _get_c_ascend_op("kv_rmsnorm_rope_cache_by_cache")
    if native_op is None or cache_mode not in ("PA", "PA_NZ") or c_kv_scale is not None or epsilon is None:
        return None
    if kv_no_split.dim() != 4 or kv_no_split.shape[1] != 1 or kv_no_split.shape[2] != 1:
        return None
    if weight.dim() != 1 or positions.dim() != 1 or slots.dim() != 1:
        return None
    if positions.shape[0] != kv_no_split.shape[0] or slots.shape[0] != kv_no_split.shape[0]:
        return None
    if positions.dtype not in (torch.int32, torch.int64) or slots.dtype not in (torch.int32, torch.int64):
        return None
    if kv_cache_rope.dim() != 4 or kv_cache_nope.dim() != 4:
        return None
    if kv_cache_rope.shape[:3] != kv_cache_nope.shape[:3]:
        return None
    if kv_cache_rope.shape[1] <= 0 or kv_cache_rope.shape[2] != 1 or kv_cache_nope.shape[2] != 1:
        return None

    resolved_rope_dim = int(rope_dim if rope_dim is not None else kv_cache_rope.shape[-1])
    nope_dim = int(weight.shape[0])
    if resolved_rope_dim <= 0 or resolved_rope_dim % 32 != 0 or nope_dim <= 0 or nope_dim % 16 != 0:
        return None
    if kv_cache_rope.shape[-1] != resolved_rope_dim or kv_cache_nope.shape[-1] != nope_dim:
        return None
    if kv_no_split.shape[-1] != nope_dim + resolved_rope_dim:
        return None

    kv_no_split = kv_no_split if kv_no_split.stride(-1) == 1 else kv_no_split.contiguous()
    weight = weight.contiguous()
    positions = positions.contiguous()
    slots = slots.contiguous()
    cos_sin_cache = get_rope_cache(rotary_emb, kv_no_split)
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] < resolved_rope_dim or cos_sin_cache.stride(-1) != 1:
        return None
    if not kv_cache_rope.is_contiguous() or not kv_cache_nope.is_contiguous():
        return None

    return native_op(
        kv_no_split,
        weight,
        positions,
        cos_sin_cache,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        float(epsilon),
        resolved_rope_dim,
        _is_neox_style(rotary_emb) if is_neox_style is None else bool(is_neox_style),
        is_output_kv,
        cache_mode == "PA_NZ",
    )


def _try_c_ascend_kv_rmsnorm_rope_cache_and_interleave_by_cache(
    q: torch.Tensor,
    kv_no_split: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    slots: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    *,
    c_kv_scale=None,
    epsilon=None,
    cache_mode: str = "PA",
    is_output_kv: bool = False,
    rope_dim: int | None = None,
    is_neox_style: bool | None = None,
):
    native_op = _get_c_ascend_op("kv_rmsnorm_rope_cache_and_interleave_by_cache")
    if native_op is None or cache_mode not in ("PA", "PA_NZ") or c_kv_scale is not None or epsilon is None:
        return None
    if kv_no_split.dim() != 4 or kv_no_split.shape[1] != 1 or kv_no_split.shape[2] != 1:
        return None
    q_shape = q.shape
    if q.dim() == 4 and q.shape[2] == 1:
        q = q.squeeze(2)
    elif q.dim() != 3:
        return None
    if weight.dim() != 1 or positions.dim() != 1 or slots.dim() != 1:
        return None
    if positions.shape[0] != kv_no_split.shape[0] or slots.shape[0] != kv_no_split.shape[0]:
        return None
    if q.shape[0] != kv_no_split.shape[0]:
        return None
    if positions.dtype not in (torch.int32, torch.int64) or slots.dtype not in (torch.int32, torch.int64):
        return None
    if q.dtype != kv_no_split.dtype:
        return None
    if kv_cache_rope.dim() != 4 or kv_cache_nope.dim() != 4:
        return None
    if kv_cache_rope.shape[:3] != kv_cache_nope.shape[:3]:
        return None
    if kv_cache_rope.shape[1] <= 0 or kv_cache_rope.shape[2] != 1 or kv_cache_nope.shape[2] != 1:
        return None

    resolved_rope_dim = int(rope_dim if rope_dim is not None else kv_cache_rope.shape[-1])
    nope_dim = int(weight.shape[0])
    if resolved_rope_dim <= 0 or resolved_rope_dim % 32 != 0 or nope_dim <= 0 or nope_dim % 16 != 0:
        return None
    if q.shape[-1] != resolved_rope_dim:
        return None
    if kv_cache_rope.shape[-1] != resolved_rope_dim or kv_cache_nope.shape[-1] != nope_dim:
        return None
    if kv_no_split.shape[-1] != nope_dim + resolved_rope_dim:
        return None

    kv_no_split = kv_no_split if kv_no_split.stride(-1) == 1 else kv_no_split.contiguous()
    weight = weight.contiguous()
    positions = positions.contiguous()
    slots = slots.contiguous()
    if q.stride(-1) != 1:
        return None
    cos_sin_cache = get_rope_cache(rotary_emb, kv_no_split)
    if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] < resolved_rope_dim or cos_sin_cache.stride(-1) != 1:
        return None
    if not kv_cache_rope.is_contiguous() or not kv_cache_nope.is_contiguous():
        return None

    q_out, cache_rope, cache_nope, out_rope, out_nope = native_op(
        kv_no_split,
        weight,
        q,
        positions,
        cos_sin_cache,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        float(epsilon),
        resolved_rope_dim,
        _is_neox_style(rotary_emb) if is_neox_style is None else bool(is_neox_style),
        is_output_kv,
        cache_mode == "PA_NZ",
    )
    if len(q_shape) == 4:
        q_out = q_out.reshape(q_shape)
    return q_out, cache_rope, cache_nope, out_rope, out_nope


def _is_inplace_partial_rotary_mul_unsupported(exc: RuntimeError) -> bool:
    message = str(exc)
    if "InplacePartialRotaryMul" not in message:
        return False
    return "does not support opType" in message or "not in libopapi.so" in message


def _match_cache_to_ref_tensor(cos_sin_cache: torch.Tensor, ref_tensor: torch.Tensor | None = None) -> torch.Tensor:
    if ref_tensor is None:
        return cos_sin_cache
    if cos_sin_cache.device == ref_tensor.device and cos_sin_cache.dtype == ref_tensor.dtype:
        return cos_sin_cache
    return cos_sin_cache.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def interleave_rope_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    positions = _normalize_positions_1d(positions, x.shape[0], "interleave_rope_by_cache")
    is_neox_style = _is_neox_style(rotary_emb)
    torch_npu_op = _get_torch_npu_op("npu_interleave_rope_by_cache") if is_neox_style else None
    if torch_npu_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, x)
        return torch_npu_op(
            x,
            positions,
            cos_sin_cache,
            rope_dim=_rope_dim(rotary_emb, x),
            is_neox_style=is_neox_style,
        )

    native_output = _try_c_ascend_interleave_rope_by_cache(
        x,
        positions,
        rotary_emb,
    )
    if native_output is not None:
        return native_output

    triton_output = _try_triton_interleave_rope_by_cache(
        x,
        positions,
        rotary_emb,
    )
    if triton_output is not None:
        return triton_output

    torch_npu_op = None if is_neox_style else _get_torch_npu_op("npu_interleave_rope_by_cache")
    if torch_npu_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, x)
        return torch_npu_op(
            x,
            positions,
            cos_sin_cache,
            rope_dim=_rope_dim(rotary_emb, x),
            is_neox_style=is_neox_style,
        )

    _raise_missing_true_by_cache_backend("interleave_rope_by_cache")


def rotary_mul_materialized(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_mode: str | None = None,
    inverse: bool = False,
    fp32_compute: bool = False,
) -> torch.Tensor:
    """Compatibility helper for upstream APIs that receive materialized cos/sin.

    This is intentionally limited to public vLLM interfaces such as
    ApplyRotaryEmb. New NPU hot paths must use the by-cache adapters above so
    the backend reads positions and cos_sin_cache directly.
    """
    origin_dtype = x.dtype
    x_for_rope = x.to(torch.float32) if fp32_compute else x
    sin_for_rope = -sin if inverse else sin
    kwargs = {}
    if rotary_mode is not None:
        kwargs["rotary_mode"] = rotary_mode
    output = torch_npu.npu_rotary_mul(x_for_rope, cos, sin_for_rope, **kwargs)
    if fp32_compute:
        output = output.to(origin_dtype)
    return output


def rotary_mul_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    *,
    rotary_mode: str | None = None,
    partial_slice: list[int] | None = None,
    inplace: bool = False,
    layout: str = "T11D",
    rope_dim_offset: int = 0,
    inverse: bool = False,
    fp32_compute: bool = False,
) -> torch.Tensor | None:
    positions = _normalize_positions_1d(positions, x.shape[0], "rotary_mul_by_cache")
    native_op = _get_torch_npu_op("npu_rotary_mul_by_cache")
    if native_op is not None:
        origin_dtype = x.dtype
        x_for_rope = x.to(torch.float32) if fp32_compute else x
        cos_sin_cache = get_rope_cache(rotary_emb, x_for_rope)
        is_neox_style = _resolve_rope_mode(rotary_emb, rotary_mode)
        if is_neox_style is None:
            raise ValueError(f"Unsupported rotary_mode {rotary_mode!r}")
        kwargs: dict[str, Any] = {
            "rotary_mode": rotary_mode,
            "partial_slice": partial_slice,
            "rope_dim": _rope_dim(rotary_emb, x_for_rope),
            "rope_dim_offset": rope_dim_offset,
            "inverse": inverse,
            "is_neox_style": is_neox_style,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        output = native_op(x_for_rope, positions, cos_sin_cache, **kwargs)
        if output is not None and fp32_compute:
            output = output.to(origin_dtype)
        if inplace:
            if output is not None:
                x.copy_(output)
            return None
        return output

    triton_output = _try_triton_rotary_siso_by_cache(
        x,
        positions,
        rotary_emb,
        rotary_mode=rotary_mode,
        partial_slice=partial_slice,
        layout=layout,
        rope_dim_offset=rope_dim_offset,
        inverse=inverse,
        fp32_compute=fp32_compute,
        inplace=inplace,
    )
    if triton_output is not None:
        if inplace:
            x.copy_(triton_output)
            return None
        return triton_output

    _raise_missing_true_by_cache_backend("rotary_mul_by_cache")


def rotary_siso_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    *,
    rotary_mode: str | None = None,
    rope_dim: int | None = None,
    rope_dim_offset: int = 0,
    inverse: bool = False,
    inplace: bool = False,
) -> torch.Tensor | None:
    positions = _normalize_positions_1d(positions, x.shape[0], "rotary_siso_by_cache")
    rope_forward = _get_triton_rope_forward_siso()
    is_neox_style = _resolve_rope_mode(rotary_emb, rotary_mode)
    resolved_rope_dim = rope_dim if rope_dim is not None else _rope_dim(rotary_emb, x)
    if (
        rope_forward is not None
        and is_neox_style is not None
        and x.dim() == 3
        and resolved_rope_dim > 0
        and resolved_rope_dim <= x.shape[-1]
        and rope_dim_offset >= 0
        and rope_dim_offset % 2 == 0
    ):
        qk = x if inplace else x.clone()
        if not qk.is_contiguous():
            qk = qk.contiguous()
        output = rope_forward(
            qk,
            cos_sin_cache=get_rope_cache(rotary_emb, qk),
            positions=positions,
            rope_dim=resolved_rope_dim,
            rope_dim_offset=rope_dim_offset,
            is_neox_style=is_neox_style,
            inverse=inverse,
        )
        if inplace:
            if output is not x:
                x.copy_(output)
            return None
        return output

    partial_slice = [0, resolved_rope_dim] if resolved_rope_dim is not None else None
    return rotary_mul_by_cache(
        x,
        positions,
        rotary_emb,
        rotary_mode=rotary_mode,
        partial_slice=partial_slice,
        rope_dim_offset=rope_dim_offset,
        inverse=inverse,
        inplace=inplace,
    )


def kv_rmsnorm_rope_cache_by_cache(
    kv_no_split: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    slots: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    *,
    c_kv_scale=None,
    epsilon=None,
    cache_mode: str = "PA",
    is_output_kv: bool = False,
    rope_dim: int | None = None,
    is_neox_style: bool | None = None,
    allow_negative_slots: bool = False,
):
    # Negative slots are used by CP prefill as a "compute outputs, skip cache
    # write" sentinel. Keep the torch_npu path off until that contract is
    # proven there; the C Ascend backend explicitly skips slot < 0 writes.
    positions = _normalize_positions_1d(positions, kv_no_split.shape[0], "kv_rmsnorm_rope_cache_by_cache")
    c_ascend_output = _try_c_ascend_kv_rmsnorm_rope_cache_by_cache(
        kv_no_split,
        weight,
        positions,
        rotary_emb,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        c_kv_scale=c_kv_scale,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
        rope_dim=rope_dim,
        is_neox_style=is_neox_style,
    )
    if c_ascend_output is not None:
        return c_ascend_output

    triton_output = _try_triton_kv_rmsnorm_rope_cache_by_cache(
        kv_no_split,
        weight,
        positions,
        rotary_emb,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        c_kv_scale=c_kv_scale,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
        rope_dim=rope_dim,
        is_neox_style=is_neox_style,
    )
    if triton_output is not None:
        return triton_output

    torch_npu_op = None if allow_negative_slots else _get_torch_npu_op("npu_kv_rmsnorm_rope_cache_by_cache")
    if torch_npu_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, kv_no_split)
        kwargs = {
            "c_kv_scale": c_kv_scale,
            "epsilon": epsilon,
            "cache_mode": cache_mode,
            "is_output_kv": is_output_kv,
            "rope_dim": rope_dim if rope_dim is not None else _rope_dim(rotary_emb, kv_no_split),
            "is_neox_style": is_neox_style if is_neox_style is not None else _is_neox_style(rotary_emb),
        }
        return torch_npu_op(
            kv_no_split,
            weight,
            positions,
            cos_sin_cache,
            slots.to(torch.int64).contiguous(),
            kv_cache_rope,
            kv_cache_nope,
            **kwargs,
        )

    _raise_missing_true_by_cache_backend("kv_rmsnorm_rope_cache_by_cache")


def kv_rmsnorm_rope_cache_and_interleave_by_cache(
    q: torch.Tensor,
    kv_no_split: torch.Tensor,
    weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    slots: torch.Tensor,
    kv_cache_rope: torch.Tensor,
    kv_cache_nope: torch.Tensor,
    *,
    c_kv_scale=None,
    epsilon=None,
    cache_mode: str = "PA",
    is_output_kv: bool = False,
    rope_dim: int | None = None,
    is_neox_style: bool | None = None,
    allow_negative_slots: bool = False,
):
    if allow_negative_slots:
        return None
    positions = _normalize_positions_1d(
        positions,
        kv_no_split.shape[0],
        "kv_rmsnorm_rope_cache_and_interleave_by_cache",
    )
    return _try_c_ascend_kv_rmsnorm_rope_cache_and_interleave_by_cache(
        q,
        kv_no_split,
        weight,
        positions,
        rotary_emb,
        slots,
        kv_cache_rope,
        kv_cache_nope,
        c_kv_scale=c_kv_scale,
        epsilon=epsilon,
        cache_mode=cache_mode,
        is_output_kv=is_output_kv,
        rope_dim=rope_dim,
        is_neox_style=is_neox_style,
    )


def split_qkv_tp_rmsnorm_rope_by_cache(
    *,
    input: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    tp_world: int,
    positions: torch.Tensor,
    rotary_emb,
    layout: str = "1T1D",
    **kwargs,
):
    native_op = _get_vllm_op("split_qkv_tp_rmsnorm_rope_by_cache")
    if native_op is not None and _is_neox_style(rotary_emb):
        cos_sin_cache = get_rope_cache(rotary_emb, input)
        return native_op(
            input=input,
            q_weight=q_weight,
            k_weight=k_weight,
            q_hidden_size=q_hidden_size,
            kv_hidden_size=kv_hidden_size,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            eps=eps,
            tp_world=tp_world,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            **kwargs,
        )

    _raise_missing_true_by_cache_backend("split_qkv_tp_rmsnorm_rope_by_cache")


def mla_preprocess_by_cache(
    hidden_states: torch.Tensor,
    wd_qkv: torch.Tensor,
    deq_scale_qkv: torch.Tensor,
    gamma1: torch.Tensor,
    beta1: torch.Tensor | None,
    wu_q: torch.Tensor,
    qb_deq_scl: torch.Tensor,
    gamma2: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    W_UK_T: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    layout: str = "T11D",
    **kwargs,
) -> None:
    raw_q_out = kwargs.pop("raw_q_out", None)
    enable_raw_q_out = kwargs.pop("enable_raw_q_out", raw_q_out is not None)
    if enable_raw_q_out and raw_q_out is None:
        raise ValueError("mla_preprocess_by_cache raw_q_out must be provided when enable_raw_q_out=True")
    if raw_q_out is None:
        raw_q_out = torch.empty(0, dtype=hidden_states.dtype, device=hidden_states.device)

    native_op = _get_c_ascend_op("mla_preprocess_by_cache") if has_mla_preprocess_by_cache_kernel() else None
    if native_op is not None:
        positions = _normalize_positions_1d(positions, hidden_states.shape[0], "mla_preprocess_by_cache")
        cos_sin_cache = get_rope_cache(rotary_emb, hidden_states)
        native_op(
            hidden_states,
            wd_qkv,
            deq_scale_qkv,
            gamma1,
            beta1,
            wu_q,
            qb_deq_scl,
            gamma2,
            positions,
            cos_sin_cache,
            W_UK_T,
            k_nope,
            k_pe,
            slot_mapping,
            is_neox_style=_is_neox_style(rotary_emb),
            enable_raw_q_out=enable_raw_q_out,
            raw_q_out=raw_q_out,
            **kwargs,
        )
        return None

    _raise_missing_true_by_cache_backend("mla_preprocess_by_cache")
    return None


def mla_prolog_v2_by_cache(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    cache_index: torch.Tensor,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    *,
    ref_tensor: torch.Tensor,
    **kwargs,
):
    native_op = _get_torch_npu_op("npu_mla_prolog_v2_by_cache")
    if native_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, ref_tensor)
        return native_op(
            token_x,
            weight_dq,
            weight_uq_qr,
            weight_uk,
            weight_dkv_kr,
            rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv,
            positions,
            cos_sin_cache,
            cache_index,
            kv_cache,
            kr_cache,
            **kwargs,
        )

    _raise_missing_true_by_cache_backend("mla_prolog_v2_by_cache")


def mla_prolog_v3_by_cache(
    token_x: torch.Tensor,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_uk: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    kv_cache: torch.Tensor,
    kr_cache: torch.Tensor,
    cache_index: torch.Tensor,
    ref_tensor: torch.Tensor,
    **kwargs,
):
    native_op = _get_torch_npu_op("npu_mla_prolog_v3_by_cache")
    if native_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, ref_tensor)
        return native_op(
            token_x=token_x,
            weight_dq=weight_dq,
            weight_uq_qr=weight_uq_qr,
            weight_uk=weight_uk,
            weight_dkv_kr=weight_dkv_kr,
            rmsnorm_gamma_cq=rmsnorm_gamma_cq,
            rmsnorm_gamma_ckv=rmsnorm_gamma_ckv,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            kr_cache=kr_cache,
            cache_index=cache_index,
            **kwargs,
        )

    _raise_missing_true_by_cache_backend("mla_prolog_v3_by_cache")


def inplace_partial_rotary_mul_by_cache(
    x: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    *,
    rotary_mode: str,
    partial_slice: list[int],
    layout: str = "T11D",
    rope_dim_offset: int = 0,
    inverse: bool = False,
) -> None:
    native_op = (
        _get_c_ascend_op("inplace_partial_rotary_mul_by_cache")
        if has_inplace_partial_rotary_mul_by_cache_kernel()
        else None
    )
    if native_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, x)
        try:
            native_op(
                x,
                positions,
                cos_sin_cache,
                rotary_mode=rotary_mode,
                partial_slice=partial_slice,
                rope_dim=_rope_dim(rotary_emb, x),
                is_neox_style=_is_neox_style(rotary_emb),
                rope_dim_offset=rope_dim_offset,
                inverse=inverse,
            )
            return None
        except RuntimeError as exc:
            if not _is_inplace_partial_rotary_mul_unsupported(exc):
                raise
            _mark_c_ascend_op_unsupported("inplace_partial_rotary_mul_by_cache")

    triton_output = _try_triton_rotary_siso_by_cache(
        x,
        positions,
        rotary_emb,
        rotary_mode=rotary_mode,
        partial_slice=partial_slice,
        layout=layout,
        rope_dim_offset=rope_dim_offset,
        inverse=inverse,
    )
    if triton_output is not None:
        x.copy_(triton_output)
        return None

    _raise_missing_true_by_cache_backend("inplace_partial_rotary_mul_by_cache")
    return None


def compressor_by_cache(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    norm_weight: torch.Tensor,
    compress_positions: torch.Tensor,
    rotary_emb,
    *,
    ref_tensor: torch.Tensor,
    layout: str = "TD",
    **kwargs,
):
    native_op = _get_c_ascend_op("compressor_by_cache") if has_compressor_by_cache_kernel() else None
    if native_op is not None:
        cos_sin_cache = get_rope_cache(rotary_emb, ref_tensor)
        return native_op(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            norm_weight,
            compress_positions,
            cos_sin_cache,
            is_neox_style=_is_neox_style(rotary_emb),
            **kwargs,
        )

    _raise_missing_true_by_cache_backend("compressor_by_cache")


def split_qkv_rmsnorm_mrope_by_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_op = _get_vllm_op("triton_split_qkv_rmsnorm_mrope_by_cache")
    if fused_op is None:
        _raise_missing_true_by_cache_backend("split_qkv_rmsnorm_mrope_by_cache")

    return fused_op(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        positions=positions,
        cos_sin_cache=get_rope_cache(rotary_emb, qkv),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        rope_dim=rope_dim,
        q_bias=q_bias,
        k_bias=k_bias,
        has_gate=has_gate,
    )
