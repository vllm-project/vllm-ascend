# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Central MLA Prolog V3 boundary with a packaged A5 Cannbot adapter.

The default backend forwards the CANN contract unchanged. Cannbot accepts only
already-quantized MXFP8 input, NoPE D=256/K=512 and BF16 query output. Its public
Python API owns eager/capture dispatch behind a mutable Torch operator.
The caller owns capability gates and the lifetime of prepared weights.
Recreate the prepared object after any weight reload; it is an independent snapshot.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch

MX_GROUP_SIZE = 32
SCALE_PAIR_SIZE = 2
PROJECTION_ALIGNMENT = 256
MIN_HIDDEN_SIZE = 1024
MAX_HIDDEN_SIZE = 8192
SUPPORTED_Q_RANKS = (640, 1024, 1536, 2048)
NOPE_HEAD_DIM = 256
KV_LORA_RANK = 512
MAX_HEADS = 128
CACHE_GROUP_SIZE = 128
PACKED_CACHE_BYTES = KV_LORA_RANK + (KV_LORA_RANK // CACHE_GROUP_SIZE) * 4
MIN_CACHE_BLOCK_SIZE = 16
MAX_CACHE_BLOCK_SIZE = 1024


@dataclass(frozen=True)
class PreparedMlaProlog:
    """Owned ND buffers, prepared once per loaded layer; no captured callable."""

    wqa: torch.Tensor
    wqb: torch.Tensor
    wkva: torch.Tensor
    wkb: torch.Tensor
    descale_wqa: torch.Tensor
    descale_wqb: torch.Tensor
    descale_wkva: torch.Tensor
    norm_weight_qa: torch.Tensor
    norm_weight_kva: torch.Tensor
    norm_eps: float
    hidden_size: int
    q_lora_rank: int
    num_heads: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    dtypes: tuple[torch.dtype, ...],
    device: torch.device,
    *,
    contiguous: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    _require(tuple(value.shape) == shape, f"{name} shape must be {shape}, got {tuple(value.shape)}")
    if value.dtype not in dtypes:
        raise TypeError(f"{name} dtype must be {dtypes}, got {value.dtype}")
    _require(value.device == device, f"{name} must be on {device}")
    if contiguous:
        _require(value.is_contiguous(), f"{name} must be contiguous; copies are not permitted")


def _scale_dtypes() -> tuple[torch.dtype, ...]:
    return (torch.uint8, torch.float8_e8m0fnu)


def _epsilon(cq: float, ckv: float) -> float:
    _require(type(cq) in (int, float) and type(ckv) in (int, float), "RMSNorm epsilons must be host scalars")
    _require(isfinite(cq) and cq >= 0, "RMSNorm epsilon must be finite and nonnegative")
    _require(cq == ckv, "Cannbot requires equal RMSNorm epsilons")
    return cq


def _owned_nd(value: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    # Allocate explicitly rather than clone/contiguous: an NPU source may be NZ,
    # and an already-contiguous source must still get independent storage.
    result = torch.empty(tuple(value.shape), dtype=dtype or value.dtype, device=value.device)
    result.copy_(value.detach())
    return result


def prepare_mla_prolog(
    *,
    weight_dq: torch.Tensor,
    weight_uq_qr: torch.Tensor,
    weight_dkv_kr: torch.Tensor,
    weight_uk: torch.Tensor,
    dequant_scale_w_dq: torch.Tensor,
    dequant_scale_w_uq_qr: torch.Tensor,
    dequant_scale_w_dkv_kr: torch.Tensor,
    rmsnorm_gamma_cq: torch.Tensor,
    rmsnorm_gamma_ckv: torch.Tensor,
    rmsnorm_epsilon_cq: float,
    rmsnorm_epsilon_ckv: float,
) -> PreparedMlaProlog:
    """Snapshot loaded (in,out) weights and (in/64,out,2) MX scales.

    Noncontiguous slices of fused weights/scales are supported. Scale carriers
    are reinterpreted, never numerically cast. Gamma is copied to FP32; the
    absorption weight must already be BF16. No original parameter is modified.
    The optional operator package is resolved lazily by the NPU dispatcher, not
    during preparation or fake tracing.
    """
    _require(isinstance(weight_dq, torch.Tensor) and weight_dq.ndim == 2, "weight_dq must have loaded (in,out) shape")
    hidden_size, q_rank = weight_dq.shape
    _require(
        MIN_HIDDEN_SIZE <= hidden_size <= MAX_HIDDEN_SIZE and hidden_size % PROJECTION_ALIGNMENT == 0,
        "hidden size must be 1024..8192 and 256-aligned",
    )
    _require(q_rank in SUPPORTED_Q_RANKS, f"q_lora_rank must be one of {SUPPORTED_Q_RANKS}")
    _require(isinstance(weight_uk, torch.Tensor) and weight_uk.ndim == 3, "weight_uk must have (heads,256,512) shape")
    heads = weight_uk.shape[0]
    _require(1 <= heads <= MAX_HEADS, "head count must be 1..128")
    device = weight_dq.device
    weights = (weight_dq, weight_uq_qr, weight_dkv_kr)
    scales = (dequant_scale_w_dq, dequant_scale_w_uq_qr, dequant_scale_w_dkv_kr)
    names = ("weight_dq", "weight_uq_qr", "weight_dkv_kr")
    dimensions = ((hidden_size, q_rank), (q_rank, heads * NOPE_HEAD_DIM), (hidden_size, KV_LORA_RANK))
    for name, weight, scale, (reduction, outputs) in zip(names, weights, scales, dimensions):
        _tensor(name, weight, (reduction, outputs), (torch.float8_e4m3fn,), device)
        _tensor(
            name + " scale",
            scale,
            (reduction // (MX_GROUP_SIZE * SCALE_PAIR_SIZE), outputs, SCALE_PAIR_SIZE),
            _scale_dtypes(),
            device,
        )
    _tensor("weight_uk", weight_uk, (heads, NOPE_HEAD_DIM, KV_LORA_RANK), (torch.bfloat16,), device)
    for name, gamma, width in (
        ("rmsnorm_gamma_cq", rmsnorm_gamma_cq, q_rank),
        ("rmsnorm_gamma_ckv", rmsnorm_gamma_ckv, KV_LORA_RANK),
    ):
        _tensor(name, gamma, (width,), (torch.bfloat16, torch.float32), device)
    eps = _epsilon(rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv)
    wqa, wqb, wkva = (_owned_nd(weight.transpose(0, 1)) for weight in weights)
    ds_wqa, ds_wqb, ds_wkva = (_owned_nd(scale.transpose(0, 1)).view(torch.float8_e8m0fnu) for scale in scales)
    return PreparedMlaProlog(
        wqa,
        wqb,
        wkva,
        _owned_nd(weight_uk),
        ds_wqa,
        ds_wqb,
        ds_wkva,
        _owned_nd(rmsnorm_gamma_cq, torch.float32),
        _owned_nd(rmsnorm_gamma_ckv, torch.float32),
        eps,
        hidden_size,
        q_rank,
        heads,
    )


def _validate_cache(cache: torch.Tensor, mode: int, device: torch.device) -> None:
    _require(
        isinstance(cache, torch.Tensor) and cache.ndim in (3, 4),
        "kv_cache must have (blocks,block_size[,1],width) shape",
    )
    width = KV_LORA_RANK if mode == 0 else PACKED_CACHE_BYTES
    blocks, block_size = cache.shape[:2]
    _require(
        blocks > 0
        and MIN_CACHE_BLOCK_SIZE <= block_size <= MAX_CACHE_BLOCK_SIZE
        and block_size % MIN_CACHE_BLOCK_SIZE == 0,
        "cache requires positive blocks and block_size 16..1024, 16-aligned",
    )
    shape = (blocks, block_size, width) if cache.ndim == 3 else (blocks, block_size, 1, width)
    dtypes = (torch.bfloat16,) if mode == 0 else (torch.float8_e4m3fn, torch.uint8, torch.int8)
    _tensor("kv_cache", cache, shape, dtypes, device, contiguous=True)
    # ND-format validation and storage-preserving views belong inside the
    # mutable dispatcher boundary, not in tracing of the canonical wrapper.


def _cannbot_forward(
    *,
    prepared: PreparedMlaProlog,
    token_x: torch.Tensor,
    dequant_scale_x: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    kr_cache: torch.Tensor | None = None,
    rope_sin: torch.Tensor | None = None,
    rope_cos: torch.Tensor | None = None,
    weight_dq: torch.Tensor | None = None,
    weight_uq_qr: torch.Tensor | None = None,
    weight_dkv_kr: torch.Tensor | None = None,
    weight_uk: torch.Tensor | None = None,
    dequant_scale_w_dq: torch.Tensor | None = None,
    dequant_scale_w_uq_qr: torch.Tensor | None = None,
    dequant_scale_w_dkv_kr: torch.Tensor | None = None,
    rmsnorm_gamma_cq: torch.Tensor | None = None,
    rmsnorm_gamma_ckv: torch.Tensor | None = None,
    rmsnorm_epsilon_cq: float | None = None,
    rmsnorm_epsilon_ckv: float | None = None,
    weight_quant_mode: int = 3,
    query_quant_mode: int = 0,
    kv_cache_quant_mode: int = 0,
    query_norm_flag: bool = False,
    cache_mode: str = "PA_BSND",
    qc_qr_scale: float = 1.0,
    kc_scale: float = 1.0,
    ckvkr_repo_mode: int = 0,
    quant_scale_repo_mode: int = 0,
    tile_size: int = CACHE_GROUP_SIZE,
    k_nope_clip_alpha: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(
        type(weight_quant_mode) is int and weight_quant_mode == 3, "Cannbot supports only MXFP8 weight_quant_mode=3"
    )
    _require(type(query_quant_mode) is int and query_quant_mode == 0, "Cannbot supports only BF16 query_quant_mode=0")
    _require(
        type(kv_cache_quant_mode) is int and kv_cache_quant_mode in (0, 3),
        "Cannbot supports kv_cache_quant_mode=0 or per-group FP8 mode=3",
    )
    _require(cache_mode == "PA_BSND", "Cannbot supports only PA_BSND cache layout")
    _require(type(query_norm_flag) is bool, "query_norm_flag must be bool")
    _require(
        type(qc_qr_scale) in (int, float) and type(kc_scale) in (int, float) and qc_qr_scale == 1.0 and kc_scale == 1.0,
        "NoPE query/key scales must be host scalars equal to 1",
    )
    _require(
        k_nope_clip_alpha is None or (type(k_nope_clip_alpha) in (int, float) and k_nope_clip_alpha == 1.0),
        "Cannbot requires k_nope_clip_alpha=None or scalar 1.0; tensor clipping is unsupported",
    )
    repo_mode = 1 if kv_cache_quant_mode == 3 else 0
    _require(
        type(ckvkr_repo_mode) is int
        and type(quant_scale_repo_mode) is int
        and ckvkr_repo_mode == repo_mode
        and quant_scale_repo_mode == repo_mode,
        "cache repository modes do not match kv_cache_quant_mode",
    )
    _require(type(tile_size) is int and tile_size == CACHE_GROUP_SIZE, "FP8 cache group size must be 128")
    cq_eps = prepared.norm_eps if rmsnorm_epsilon_cq is None else rmsnorm_epsilon_cq
    ckv_eps = prepared.norm_eps if rmsnorm_epsilon_ckv is None else rmsnorm_epsilon_ckv
    _require(_epsilon(cq_eps, ckv_eps) == prepared.norm_eps, "RMSNorm epsilon changed after prepare")
    for name, value in (("rope_sin", rope_sin), ("rope_cos", rope_cos), ("kr_cache", kr_cache)):
        _require(
            value is None or (isinstance(value, torch.Tensor) and value.numel() == 0),
            f"{name} must be absent or empty for Dr=0",
        )
    device = prepared.wqa.device
    # Check any canonical static operands supplied by the caller, without
    # repacking or touching their data. The prepared snapshot remains authoritative.
    for name, value, expected in (
        ("weight_dq", weight_dq, prepared.wqa.transpose(0, 1)),
        ("weight_uq_qr", weight_uq_qr, prepared.wqb.transpose(0, 1)),
        ("weight_dkv_kr", weight_dkv_kr, prepared.wkva.transpose(0, 1)),
        ("weight_uk", weight_uk, prepared.wkb),
    ):
        if value is not None:
            _tensor(name, value, tuple(expected.shape), (expected.dtype,), device)
    for name, value, expected in (
        ("dequant_scale_w_dq", dequant_scale_w_dq, prepared.descale_wqa),
        ("dequant_scale_w_uq_qr", dequant_scale_w_uq_qr, prepared.descale_wqb),
        ("dequant_scale_w_dkv_kr", dequant_scale_w_dkv_kr, prepared.descale_wkva),
    ):
        if value is not None:
            _tensor(name, value, (expected.shape[1], expected.shape[0], SCALE_PAIR_SIZE), _scale_dtypes(), device)
    for name, value, width in (
        ("rmsnorm_gamma_cq", rmsnorm_gamma_cq, prepared.q_lora_rank),
        ("rmsnorm_gamma_ckv", rmsnorm_gamma_ckv, KV_LORA_RANK),
    ):
        if value is not None:
            _tensor(name, value, (width,), (torch.bfloat16, torch.float32), device)
    _require(isinstance(token_x, torch.Tensor) and token_x.ndim == 2, "token_x must have (T,H) shape")
    tokens = token_x.shape[0]
    _tensor("token_x", token_x, (tokens, prepared.hidden_size), (torch.float8_e4m3fn,), device, contiguous=True)
    _tensor(
        "dequant_scale_x",
        dequant_scale_x,
        (tokens, prepared.hidden_size // MX_GROUP_SIZE),
        _scale_dtypes(),
        device,
        contiguous=True,
    )
    _tensor("cache_index", cache_index, (tokens,), (torch.int64,), device, contiguous=True)
    # Validate now, but pass the ORIGINAL cache through the mutable dispatcher.
    _validate_cache(kv_cache, kv_cache_quant_mode, device)
    q_shape = (tokens, prepared.num_heads, KV_LORA_RANK)
    qr_shape = (tokens, prepared.q_lora_rank) if query_norm_flag else (0,)
    ds_shape = (tokens, prepared.q_lora_rank // MX_GROUP_SIZE) if query_norm_flag else (0,)
    empty_rope = torch.empty((tokens, prepared.num_heads, 0), dtype=torch.bfloat16, device=device)
    if tokens == 0:
        return (
            torch.empty(q_shape, dtype=torch.bfloat16, device=device),
            empty_rope,
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty(qr_shape, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(ds_shape, dtype=torch.float8_e8m0fnu, device=device),
        )
    from vllm_ascend.ops.packaged_mla_prolog import quant_mla_prolog

    q, descale_q, qr, descale_qr_bytes = quant_mla_prolog(
        token_x,
        prepared.wqa,
        prepared.wqb,
        prepared.wkva,
        prepared.wkb,
        dequant_scale_x.view(torch.uint8),
        prepared.descale_wqa.view(torch.uint8),
        prepared.descale_wqb.view(torch.uint8),
        prepared.descale_wkva.view(torch.uint8),
        prepared.norm_weight_qa,
        prepared.norm_weight_kva,
        kv_cache,
        cache_index,
        norm_eps=prepared.norm_eps,
        kv_cache_quant_mode=kv_cache_quant_mode,
        return_qr=query_norm_flag,
    )
    descale_qr = descale_qr_bytes.view(torch.float8_e8m0fnu)
    _tensor("query output", q, q_shape, (torch.bfloat16,), device)
    _tensor("query scale output", descale_q, (0,), (torch.float32,), device)
    _tensor("query norm output", qr, qr_shape, (torch.float8_e4m3fn,), device)
    _tensor("query norm scale output", descale_qr, ds_shape, (torch.float8_e8m0fnu,), device)
    return q, empty_rope, descale_q, qr, descale_qr


def mla_prolog(
    *,
    backend: str = "torch_npu",
    prepared: PreparedMlaProlog | None = None,
    torch_npu_prolog: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch canonical CANN kwargs and preserve the public five outputs.

    Cannbot execution always crosses the packaged Torch dispatcher boundary.
    The default branch intentionally does not apply Cannbot restrictions or
    normalize outputs. The caller selects the backend from the device adaptor.
    """
    if backend == "torch_npu":
        if torch_npu_prolog is None:
            import torch_npu

            torch_npu_prolog = torch_npu.npu_mla_prolog_v3
        return torch_npu_prolog(**kwargs)
    _require(backend == "cannbot", f"Unsupported MLA Prolog backend: {backend}")
    if not isinstance(prepared, PreparedMlaProlog):
        raise ValueError("Cannbot requires prepare_mla_prolog first")
    return _cannbot_forward(prepared=prepared, **kwargs)
