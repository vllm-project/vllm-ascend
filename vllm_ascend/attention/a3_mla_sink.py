# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 split-Q MLA provider boundary for flash-attention-npu's 910 release.

The existing FlashMLA batch builder owns buffers and stream ordering. This
adapter translates its normalized inputs without imposing the A5 cache ABI.
Each packed Q segment is wholly live or wholly padding; the builder sets
padding KV lengths to zero before either operator consumes them.
"""

import math
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any

import torch

ROPE_DIM = 64
LATENT_DIM = 512
QK_DIM = ROPE_DIM + LATENT_DIM
PAGE_SIZE = 128
OPERATOR_MODULE = "flash_attn_npu_3.flash_attn_npu_interface"


def get_a3_mla_ops():
    """Select before cache allocation, following the A5 package availability policy."""
    try:
        module = import_module(OPERATOR_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in ("flash_attn_npu_3", OPERATOR_MODULE):
            return None
        raise
    attention = getattr(module, "flash_attn_with_kvcache", None)
    metadata = getattr(module, "get_scheduler_metadata", None)
    return (attention, metadata) if callable(attention) and callable(metadata) else None


@dataclass(frozen=True)
class A3MLAConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    page_size: int = PAGE_SIZE
    max_seqlen_k: int = PAGE_SIZE
    max_seqlen_q: int = 1
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.num_heads <= 0 or self.mask_mode not in (0, 3):
            raise ValueError("A3 MLA requires positive heads, plain/causal mask and ND paged cache")
        if not math.isfinite(self.softmax_scale):
            raise ValueError("softmax_scale must be finite")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("A3 MLA supports FP16/BF16")
        if not 16 <= self.page_size <= 1024 or self.page_size % 16:
            raise ValueError("A3 MLA page size must be a multiple of 16 in [16, 1024]")
        if self.max_seqlen_k <= 0 or self.max_seqlen_k % self.page_size:
            raise ValueError("A3 MLA logical KV capacity must be positive and page-aligned")
        if self.max_seqlen_q < 0:
            raise ValueError("query capacity must be nonnegative")


def supports_a3_mla(impl, *, is_a3: bool, is_draft: bool, pd_supported: bool = False) -> bool:
    """Static selection before cache allocation; no tensor-content reads."""
    if not is_a3 or not is_draft:
        return False
    cfg = impl.vllm_config
    if cfg.model_config.hf_config.model_type != "kimi_k31":
        return False
    spec = cfg.speculative_config
    if not getattr(cfg, "use_v2_model_runner", False) or spec is None or spec.method != "dspark":
        return False
    if impl.num_kv_heads != 1 or impl.kv_lora_rank != LATENT_DIM or impl.qk_rope_head_dim != ROPE_DIM:
        return False
    if impl.fa_quant_layer or impl.dtype not in (torch.float16, torch.bfloat16) or impl.enable_kv_nz:
        return False
    if impl.use_pcp or cfg.parallel_config.pipeline_parallel_size != 1:
        return False
    # The caller shares A5's component-major cache transfer gate. Keep this
    # provider boundary independent of vLLM's cache/connector imports.
    if cfg.kv_transfer_config is not None and not pd_supported:
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
        shape = (pages, self.config.page_size, 1)
        return (
            torch.empty((*shape, LATENT_DIM), dtype=self.config.dtype, device=device),
            torch.empty((*shape, ROPE_DIM), dtype=self.config.dtype, device=device),
        )

    @classmethod
    def load(cls, config: A3MLAConfig):
        ops = get_a3_mla_ops()
        if ops is None:
            raise ImportError(
                "A3 MLA requires flash_attn_with_kvcache and get_scheduler_metadata in " + OPERATOR_MODULE
            )
        return cls(config, *ops)

    def for_capacity(self, columns: int, tokens: int):
        return replace(
            self, config=replace(self.config, max_seqlen_k=columns * self.config.page_size, max_seqlen_q=tokens)
        )

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
            page_size=cfg.page_size,
            causal=cfg.mask_mode == 3,
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

    def attention(
        self, q, k_cache, *, block_table, cache_seqlens, cu_seqlens_q, metadata, seqused_q=None, attn_mask=None
    ):
        del attn_mask  # Mask semantics are explicit in cfg.mask_mode, never inferred from a tensor.
        batch = self._check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        cfg = self.config
        if q.ndim != 3 or q.shape[1:] != (cfg.num_heads, QK_DIM) or q.dtype != cfg.dtype:
            raise ValueError("normalized A3 Query must be TND [T,H,576] with the configured dtype")
        if not isinstance(k_cache, tuple) or len(k_cache) != 2:
            raise ValueError("A3 MLA requires (content_cache, rope_cache)")
        v_cache, rope_cache = k_cache
        if v_cache.ndim != 4 or v_cache.shape[1:] != (cfg.page_size, 1, LATENT_DIM):
            raise ValueError("A3 content cache must be ND [pages,128,1,512]")
        if rope_cache.shape != (*v_cache.shape[:3], ROPE_DIM):
            raise ValueError("A3 RoPE cache shape does not match the content cache")
        if block_table.shape != (batch, cfg.max_seqlen_k // cfg.page_size) or block_table.dtype != torch.int32:
            raise ValueError("A3 page table does not match the prepared metadata capacity")
        for tensor in (v_cache, rope_cache, block_table, cache_seqlens, metadata):
            if tensor.device != q.device:
                raise ValueError("A3 attention inputs must share one device")
        if v_cache.dtype != cfg.dtype or rope_cache.dtype != cfg.dtype:
            raise ValueError("A3 Query and KV dtype must match")
        result = self.attention_op(
            q[..., LATENT_DIM:].contiguous(),
            rope_cache,
            v_cache,
            qv=q[..., :LATENT_DIM].contiguous(),
            cache_seqlens=cache_seqlens,
            page_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=cfg.max_seqlen_q,
            softmax_scale=cfg.softmax_scale,
            causal=cfg.mask_mode == 3,
            scheduler_metadata=metadata,
            num_splits=1,
            return_softmax_lse=True,
        )
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise ValueError("A3 MLA must return (out, lse, out_accum, lse_accum)")
        output, lse, _, _ = result
        expected_output = (q.shape[0], cfg.num_heads, LATENT_DIM)
        if output.shape != expected_output or output.dtype != cfg.dtype or output.device != q.device:
            raise ValueError("A3 provider output must be TND and match Query dtype/device")
        if lse.shape != (cfg.num_heads, q.shape[0]) or lse.dtype != torch.float32 or lse.device != q.device:
            raise ValueError("A3 provider LSE must be FP32 [H,T] on the Query device")
        if q.shape[0]:
            token = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
            request = torch.searchsorted(cu_seqlens_q[1:].contiguous(), token, right=True).long()
            live = cache_seqlens[request] > 0
            if cfg.mask_mode == 3:
                # Right-aligned causal attention can mask an entire query row
                # when Q is longer than KV. Normalize the provider's +inf LSE
                # sentinel so that DCP merging gives this row zero weight.
                query_lengths = cu_seqlens_q[request + 1] - cu_seqlens_q[request]
                offset = token - cu_seqlens_q[request]
                live &= cache_seqlens[request] - query_lengths + offset >= 0
            output = output.masked_fill(~live[:, None, None], 0)
            lse = lse.masked_fill(~live[None, :], -torch.inf)
        # Existing MLA V-up and DCP adaptation consume NTD output and NT LSE.
        return output.transpose(0, 1), lse


def a3_target_device_lengths_ready(model_state, attn_groups, draft_layer_names) -> bool:
    """Conservative consumer contract: unknown target builders keep the barrier.

    Native A5 TargetDeviceMetadata remains handled by the original policy.
    A3 builders can opt in through supports_spec_device_lengths(), after their
    exact-host-length consumers have been removed and tested. No user flag may
    substitute for these initialized consumer declarations.
    """
    state_support = getattr(model_state, "supports_spec_device_lengths", None)
    if not callable(state_support) or state_support() is not True:
        return False
    seen = False
    for groups in attn_groups:
        for group in groups:
            if not any(name not in draft_layer_names for name in group.layer_names):
                continue
            seen = True
            supported = getattr(group.get_metadata_builder(0), "supports_spec_device_lengths", None)
            if not callable(supported) or supported() is not True:
                return False
    return seen
