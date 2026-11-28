# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 split-Q MLA provider boundary for flash-attention-npu's 910 release.

The existing FlashMLA batch builder owns buffers and stream ordering. This
adapter translates its normalized inputs without imposing the A5 cache ABI.
Each packed Q segment is wholly live or wholly padding; the builder sets
padding KV lengths to zero before either operator consumes them.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from packaging.version import Version

ROPE_DIM = 64
LATENT_DIM = 512
QK_DIM = ROPE_DIM + LATENT_DIM
PAGE_SIZE = 128
OPERATOR_MODULE = "flash_attn_npu_3"


def supports_a3_mla_window() -> bool:
    # Older 910 wheels export the same Python APIs but cannot execute split Q/Qv.
    try:
        return Version(version("flash-attn-npu")) >= Version("0.5.1.post1")
    except PackageNotFoundError:
        return False


def get_a3_mla_ops():
    """Select before cache allocation, following the A5 package availability policy."""
    try:
        module = import_module(OPERATOR_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name == OPERATOR_MODULE:
            return None
        raise
    attention = getattr(module, "flash_attn_with_kvcache", None)
    metadata = getattr(module, "get_scheduler_metadata", None)
    if callable(attention) and callable(metadata):
        return attention, metadata
    return None


@dataclass(frozen=True)
class A3MLAConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    max_seqlen_k: int = PAGE_SIZE
    max_seqlen_q: int = 1
    dtype: torch.dtype = torch.bfloat16
    sliding_window: int | None = None

    def window_kwargs(self) -> dict[str, tuple[int, int]]:
        if self.sliding_window is None:
            return {}
        return {"window_size": (self.sliding_window - 1, 0)}

    def __post_init__(self):
        if self.num_heads <= 0 or self.mask_mode not in (0, 3, 4):
            raise ValueError("A3 MLA requires positive heads and a supported mask mode")
        if (self.mask_mode == 4) != (self.sliding_window is not None) or (
            self.sliding_window is not None and not 1 <= self.sliding_window <= 2**31
        ):
            raise ValueError("A3 windowed MLA requires mode 4 and a positive window with an int32 left bound")
        if not math.isfinite(self.softmax_scale):
            raise ValueError("softmax_scale must be finite")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("A3 MLA supports FP16/BF16")
        if self.max_seqlen_k <= 0 or self.max_seqlen_k % PAGE_SIZE:
            raise ValueError("A3 MLA logical KV capacity must be positive and page-aligned")
        if self.max_seqlen_q < 0:
            raise ValueError("query capacity must be nonnegative")


def supports_a3_mla(
    impl, *, is_a3: bool, is_draft: bool, supports_pd_layout: Callable[[Any], bool], dflash_draft_window: bool = False
) -> bool:
    """Static selection before cache allocation; no tensor-content reads."""
    if not is_a3 or not is_draft:
        return False
    cfg = impl.vllm_config
    if cfg.model_config.hf_config.model_type != "kimi_k31":
        return False
    spec = cfg.speculative_config
    if spec is None:
        return False
    if spec.method == "dflash":
        if not dflash_draft_window or impl.use_dcp or not supports_a3_mla_window():
            return False
    elif spec.method != "dspark":
        return False
    if impl.num_kv_heads != 1 or impl.kv_lora_rank != LATENT_DIM or impl.qk_rope_head_dim != ROPE_DIM:
        return False
    if impl.fa_quant_layer or impl.dtype not in (torch.float16, torch.bfloat16) or impl.enable_kv_nz:
        return False
    # Split DCP must understand the provider's separate Q/KV components and
    # normalize empty local rows before merging their LSE contributions.
    if impl.use_dcp and not getattr(impl, "supports_split_query_dcp", False):
        return False
    if impl.use_pcp or cfg.parallel_config.pipeline_parallel_size != 1:
        return False
    # The caller shares A5's component-major cache transfer gate. Keep this
    # provider boundary independent of vLLM's cache/connector imports.
    if not supports_pd_layout(cfg):
        return False
    # AscendMLABackend exposes 128-token kernel pages. Manager blocks (e.g.
    # K3.1's 1536) are expanded by the existing block-table/cache-view layer;
    # they are not the operator's page_size argument.
    return cfg.cache_config.block_size > 0 and cfg.cache_config.block_size % PAGE_SIZE == 0


@dataclass(frozen=True)
class A3MLAAdapter:
    config: A3MLAConfig
    attention_op: Any
    metadata_op: Any

    def allocate_cache(self, pages: int, *, device):
        """Keep both A3 components packed inside each page, including DCP KV."""
        shape = (pages, PAGE_SIZE, 1)
        return (
            torch.empty((*shape, LATENT_DIM), dtype=self.config.dtype, device=device),
            torch.empty((*shape, ROPE_DIM), dtype=self.config.dtype, device=device),
        )

    @classmethod
    def load(cls, config: A3MLAConfig):
        ops = get_a3_mla_ops()
        if ops is None:
            raise ImportError(
                "A3 MLA requires a validated build with flash_attn_with_kvcache and get_scheduler_metadata in "
                + OPERATOR_MODULE
            )
        return cls(config, *ops)

    def for_capacity(self, columns: int, tokens: int):
        return replace(self, config=replace(self.config, max_seqlen_k=columns * PAGE_SIZE, max_seqlen_q=tokens))

    def _check_lengths(self, cache_seqlens, cu_seqlens_q, seqused_q):
        batch = cache_seqlens.numel()
        if batch == 0:
            raise ValueError("A3 MLA requires a nonempty batch")
        if cache_seqlens.shape != (batch,) or cache_seqlens.dtype != torch.int32:
            raise ValueError("cache_seqlens must be int32 [B]")
        if cu_seqlens_q.shape != (batch + 1,) or cu_seqlens_q.dtype != torch.int32:
            raise ValueError("cu_seqlens_q must be int32 [B+1]")
        if not cache_seqlens.is_contiguous() or not cu_seqlens_q.is_contiguous():
            raise ValueError("A3 lengths must be contiguous")
        if cache_seqlens.device != cu_seqlens_q.device:
            raise ValueError("A3 lengths must share a device")
        if seqused_q is not None:
            if (
                seqused_q.shape != (batch,)
                or seqused_q.dtype != torch.int32
                or seqused_q.device != cache_seqlens.device
            ):
                raise ValueError("seqused_q must be same-device int32 [B]")
            # The builder derives used_q from cu differences, or sets a whole
            # dummy segment to zero. Partial segments are not representable by
            # this public operator ABI. Do not read device values here:
            # aten::_assert_async falls back to CPU on torch_npu 2.10.
        if cache_seqlens.device.type != "npu":
            raise ValueError("native A3 MLA requires NPU tensors")
        return batch

    def build_metadata(self, cache_seqlens, cu_seqlens_q, seqused_q=None):
        # FlashMLAMetadataBuilder zeros dummy-request lengths once per batch.
        # Both operators consume that same tensor; do not remask it per layer.
        batch = self._check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        cfg = self.config
        metadata = self.metadata_op(
            batch,
            cfg.max_seqlen_q,
            cfg.max_seqlen_k,
            cfg.num_heads,
            1,
            ROPE_DIM,
            cache_seqlens,
            qkv_dtype=cfg.dtype,
            headdim_v=LATENT_DIM,
            cu_seqlens_q=cu_seqlens_q,
            page_size=PAGE_SIZE,
            causal=cfg.mask_mode != 0,
            **cfg.window_kwargs(),
            num_splits=1,
        )
        if (
            not isinstance(metadata, torch.Tensor)
            or metadata.dtype != torch.uint8
            or metadata.ndim != 1
            or not metadata.is_contiguous()
        ):
            raise ValueError("A3 provider metadata must be a contiguous opaque uint8 vector")
        if not metadata.numel() or metadata.device != cache_seqlens.device:
            raise ValueError("A3 provider metadata must be nonempty and remain on the input device")
        return metadata

    def build_output_mask(self, cache_seqlens, cu_seqlens_q, num_tokens):
        """Rows without visible KV, shared across layers for one refreshed batch."""
        token = torch.arange(num_tokens, device=cache_seqlens.device, dtype=torch.int32)
        request = torch.searchsorted(cu_seqlens_q[1:].contiguous(), token, right=True).long()
        lengths = cache_seqlens[request]
        invalid = lengths <= 0
        if self.config.mask_mode != 0:
            query_start = cu_seqlens_q[request]
            query_lengths = cu_seqlens_q[request + 1] - query_start
            invalid |= lengths - query_lengths + token - query_start < 0
        return invalid

    def attention(
        self,
        q,
        k_cache,
        *,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        metadata,
        output_mask,
        seqused_q=None,
        attn_mask=None,
        normalize_output=True,
        return_softmax_lse=True,
    ):
        del attn_mask  # Mask semantics are explicit in cfg.mask_mode, never inferred from a tensor.
        # The batch builder validated these length buffers when preparing metadata.
        batch = cache_seqlens.numel()
        cfg = self.config
        if isinstance(q, tuple):
            qv, q_rope = q
            if qv.ndim != 3 or qv.shape[1:] != (cfg.num_heads, LATENT_DIM):
                raise ValueError("A3 content Query must be TND [T,H,512]")
            if q_rope.shape != (*qv.shape[:2], ROPE_DIM) or q_rope.device != qv.device:
                raise ValueError("A3 RoPE Query must be same-device TND [T,H,64]")
        else:
            if q.ndim != 3 or q.shape[1:] != (cfg.num_heads, QK_DIM):
                raise ValueError("normalized A3 Query must be TND [T,H,576]")
            qv, q_rope = q[..., :LATENT_DIM], q[..., LATENT_DIM:]
        if qv.dtype != cfg.dtype or q_rope.dtype != cfg.dtype:
            raise ValueError("A3 Query must have the configured dtype")
        tokens, device = qv.shape[0], qv.device
        if output_mask.shape != (tokens,) or output_mask.dtype != torch.bool or output_mask.device != device:
            raise ValueError("A3 output mask must be same-device bool [T] prepared with the metadata")
        if not isinstance(k_cache, tuple) or len(k_cache) != 2:
            raise ValueError("A3 MLA requires (content_cache, rope_cache)")
        v_cache, rope_cache = k_cache
        if v_cache.ndim != 4 or v_cache.shape[1:] != (PAGE_SIZE, 1, LATENT_DIM):
            raise ValueError("A3 content cache must be ND [pages,128,1,512]")
        if rope_cache.shape != (*v_cache.shape[:3], ROPE_DIM):
            raise ValueError("A3 RoPE cache shape does not match the content cache")
        if block_table.shape != (batch, cfg.max_seqlen_k // PAGE_SIZE) or block_table.dtype != torch.int32:
            raise ValueError("A3 page table does not match the prepared metadata capacity")
        for tensor in (v_cache, rope_cache, block_table, cache_seqlens, metadata):
            if tensor.device != device:
                raise ValueError("A3 attention inputs must share one device")
        if v_cache.dtype != cfg.dtype or rope_cache.dtype != cfg.dtype:
            raise ValueError("A3 Query and KV dtype must match")
        result = self.attention_op(
            q_rope.contiguous(),
            rope_cache,
            v_cache,
            qv=qv.contiguous(),
            cache_seqlens=cache_seqlens,
            page_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=cfg.max_seqlen_q,
            softmax_scale=cfg.softmax_scale,
            causal=cfg.mask_mode != 0,
            **cfg.window_kwargs(),
            scheduler_metadata=metadata,
            num_splits=1,
            return_softmax_lse=return_softmax_lse,
        )
        if return_softmax_lse:
            if not isinstance(result, (tuple, list)) or len(result) != 4:
                raise ValueError("A3 MLA must return (out, lse, out_accum, lse_accum)")
            output, lse, _, _ = result
            if lse.shape != (cfg.num_heads, tokens) or lse.dtype != torch.float32 or lse.device != device:
                raise ValueError("A3 provider LSE must be FP32 [H,T] on the Query device")
        else:
            output, lse = result, None
        expected_output = (tokens, cfg.num_heads, LATENT_DIM)
        if output.shape != expected_output or output.dtype != cfg.dtype or output.device != device:
            raise ValueError("A3 provider output must be TND and match Query dtype/device")
        if normalize_output and tokens:
            # The provider's empty/all-masked rows may carry +inf LSE.
            # Ordinary MLA needs normalized rows before V-up. DCP defers this
            # to its existing pack/combine kernels using the same output mask.
            output.masked_fill_(output_mask[:, None, None], 0)
            if lse is not None:
                lse.masked_fill_(output_mask[None, :], -torch.inf)
        # Existing MLA V-up and DCP adaptation consume NTD output and NT LSE.
        return output.transpose(0, 1), lse
