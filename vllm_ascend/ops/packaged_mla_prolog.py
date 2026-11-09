# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatcher boundary for the packaged A5 NoPE MLA Prolog.

Like packaged_attention, fake tracing never imports the external DSL package.
Unlike the attention kernels, Prolog mutates KV cache. Keep that mutation opaque
and explicit: functionalization must not clone/copy back the persistent cache.
MX scales cross this boundary as byte views, not E8M0 tensors (the vendor graph
entry uses the same convention). The public vendor API owns capture dispatch.
"""

from functools import cache
from importlib import import_module

import torch

MX_GROUP_SIZE = 32
KV_LORA_RANK = 512
CACHE_GROUP_SIZE = 128
PACKED_CACHE_BYTES = KV_LORA_RANK + KV_LORA_RANK // CACHE_GROUP_SIZE * 4


@cache
def _get_dsl_op():
    try:
        module = import_module("ops.quant_mla_prolog")
    except ImportError as exc:
        raise ImportError(
            "A5 NoPE MLA Prolog requires matching CANNBotDSL and quant_mla_prolog "
            "operator wheels for Linux aarch64 / Python 3.12. Artifact installation "
            "is pending; DeepSeek's older operator wheel is not a substitute."
        ) from exc
    operation = getattr(module, "quant_mla_prolog", None)
    if not callable(operation):
        raise ImportError("The installed operator wheel must expose ops.quant_mla_prolog.quant_mla_prolog")
    return operation


torch.library.define(
    "vllm_ascend::quant_mla_prolog",
    "(Tensor x, Tensor wqa, Tensor wqb, Tensor wkva, Tensor wkb, Tensor descale_x, "
    "Tensor descale_wqa, Tensor descale_wqb, Tensor descale_wkva, Tensor norm_weight_qa, "
    "Tensor norm_weight_kva, Tensor(a!) kv_cache, Tensor cache_index, *, float norm_eps, "
    "int kv_cache_quant_mode=0, bool return_qr=False) -> (Tensor, Tensor, Tensor, Tensor)",
)


@torch.library.register_fake("vllm_ascend::quant_mla_prolog")
def _quant_mla_prolog_fake(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    *,
    norm_eps,
    kv_cache_quant_mode=0,
    return_qr=False,
):
    tokens = x.shape[0]
    rank = norm_weight_qa.shape[0]
    return (
        x.new_empty((tokens, wkb.shape[0], KV_LORA_RANK), dtype=torch.bfloat16),
        x.new_empty((0,), dtype=torch.float32),
        x.new_empty((tokens, rank) if return_qr else (0,), dtype=torch.float8_e4m3fn),
        x.new_empty((tokens, rank // MX_GROUP_SIZE) if return_qr else (0,), dtype=torch.uint8),
    )


def _cache_view(kv_cache, mode):
    if mode not in (0, 3):
        raise ValueError("quant_mla_prolog supports BF16 or packed FP8 C8 cache only")
    if kv_cache.ndim not in (3, 4) or not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous PA_BSND storage; copies are not permitted")
    width = KV_LORA_RANK if mode == 0 else PACKED_CACHE_BYTES
    expected = (*kv_cache.shape[:2], width) if kv_cache.ndim == 3 else (*kv_cache.shape[:2], 1, width)
    if tuple(kv_cache.shape) != expected:
        raise ValueError(f"kv_cache shape must be {expected}")
    dtypes = (torch.bfloat16,) if mode == 0 else (torch.uint8, torch.int8, torch.float8_e4m3fn)
    if kv_cache.dtype not in dtypes:
        raise TypeError("kv_cache dtype does not match kv_cache_quant_mode")
    if kv_cache.device.type in ("npu", "privateuseone"):
        import torch_npu

        if torch_npu.get_npu_format(kv_cache) not in (0, 2):
            raise ValueError("kv_cache must use linear ND storage")
    result = kv_cache.view(*kv_cache.shape[:2], 1, width)
    return result if mode == 0 else result.view(torch.float8_e4m3fn)


def _quant_mla_prolog_impl(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    *,
    norm_eps,
    kv_cache_quant_mode=0,
    return_qr=False,
):
    # The original cache crosses the mutable dispatcher boundary. Only then make
    # storage-preserving views, including nonzero storage offsets and C8 bytes.
    cache_view = _cache_view(kv_cache, kv_cache_quant_mode)
    scales = (descale_x, descale_wqa, descale_wqb, descale_wkva)
    if any(value.dtype != torch.uint8 for value in scales):
        raise TypeError("quant_mla_prolog dispatcher MX scales must be uint8 byte views")
    if x.shape[0] == 0:
        return _quant_mla_prolog_fake(
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            *scales,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            norm_eps=norm_eps,
            kv_cache_quant_mode=kv_cache_quant_mode,
            return_qr=return_qr,
        )
    operation = _get_dsl_op()
    # Do not call a private/native entry or retry after a potentially mutating
    # failure. A compatible public API detects stream capture itself. During
    # capture/replay, the caller must provide in-range, unique nonnegative slots
    # (negative slots are padding), just as for native cache-writing operators.
    q, q_scale, qr, qr_scale = operation(
        x=x,
        wqa=wqa,
        wqb=wqb,
        wkva=wkva,
        wkb=wkb,
        descale_x=descale_x.view(torch.float8_e8m0fnu),
        descale_wqa=descale_wqa.view(torch.float8_e8m0fnu),
        descale_wqb=descale_wqb.view(torch.float8_e8m0fnu),
        descale_wkva=descale_wkva.view(torch.float8_e8m0fnu),
        norm_weight_qa=norm_weight_qa,
        norm_weight_kva=norm_weight_kva,
        kv_cache=cache_view,
        cache_index=cache_index,
        quant_scale_kv=None,
        rope_sin=None,
        rope_cos=None,
        norm_eps=norm_eps,
        w_qr_quant_mode=1,
        qkv_quant_mode=0 if kv_cache_quant_mode == 0 else 1,
        cache_mode=0,
        qr_scale=1.0,
        kvr_scale=1.0,
        rope_mode=0,
        return_qr=return_qr,
    )
    if qr_scale.dtype != torch.float8_e8m0fnu:
        raise TypeError("CANNBot query-norm scales must be E8M0")
    return q, q_scale, qr, qr_scale.view(torch.uint8)


@torch.library.impl("vllm_ascend::quant_mla_prolog", "Functionalize")
def _reject_functionalization(*args, **kwargs):
    raise RuntimeError(
        "quant_mla_prolog requires storage-preserving mutable-cache execution; "
        "functionalization with cache clone/copy-back is unsupported. "
        "Keep the cache writer behind the existing opaque MLA attention boundary."
    )


torch.library.impl("vllm_ascend::quant_mla_prolog", "PrivateUse1", _quant_mla_prolog_impl)

quant_mla_prolog = torch.ops.vllm_ascend.quant_mla_prolog.default
