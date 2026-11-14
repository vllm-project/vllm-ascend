# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 split-Q MLA provider boundary, aligned with flash-attention-npu #338.

The existing FlashMLA batch builder owns buffers and stream ordering. This
adapter translates its normalized inputs without imposing the A5 cache ABI.
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
MAX_CAPACITY = 16384
DEFAULT_MODULE = "flash_attn_npu_3.flash_attn_npu_interface"
MOCK_MODULE = "vllm_ascend.attention.a3_mla_mock"


@dataclass(frozen=True)
class A3MLAConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    layout_kv: str = "PA_BBND"
    return_softmax_lse: bool = False
    page_size: int = PAGE_SIZE
    max_seqlen_k: int = PAGE_SIZE
    max_seqlen_q: int = 1
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if self.num_heads <= 0 or self.mask_mode not in (0, 3) or self.layout_kv != "PA_BBND":
            raise ValueError("A3 MLA requires positive heads, plain/causal mask and ND paged cache")
        if not math.isfinite(self.softmax_scale):
            raise ValueError("softmax_scale must be finite")
        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("A3 MLA supports FP16/BF16")
        if not 16 <= self.page_size <= 1024 or self.page_size % 16:
            raise ValueError("A3 MLA page size must be a multiple of 16 in [16, 1024]")
        if not 0 < self.max_seqlen_k <= MAX_CAPACITY or self.max_seqlen_k % self.page_size:
            raise ValueError("A3 MLA logical KV capacity must be page-aligned and at most 16384")
        if self.max_seqlen_q < 0:
            raise ValueError("query capacity must be nonnegative")


def validate_sink_selection(impl, *, mode: str, is_a3: bool, is_draft: bool, allow_graph: bool) -> bool:
    """Static selection before cache allocation; no tensor-content reads."""
    if mode not in ("disabled", "mock", "external"):
        raise ValueError("a3_mla_sink_mode must be disabled, mock or external")
    if mode == "disabled" or not is_a3 or not is_draft:
        return False
    cfg = impl.vllm_config
    if cfg.model_config.hf_config.model_type != "kimi_k31":
        return False
    spec = cfg.speculative_config
    if not getattr(cfg, "use_v2_model_runner", False) or spec is None or spec.method != "dspark":
        raise ValueError("A3 MLA sink requires K3.1 DSpark on Model Runner V2")
    if impl.num_kv_heads != 1 or impl.kv_lora_rank != LATENT_DIM or impl.qk_rope_head_dim != ROPE_DIM:
        raise ValueError("A3 MLA sink requires compressed MLA 64/512 and one KV head")
    if impl.fa_quant_layer or impl.dtype not in (torch.float16, torch.bfloat16) or impl.enable_kv_nz:
        raise ValueError("A3 MLA sink requires unquantized FP16/BF16 ND KV")
    if impl.use_pcp or cfg.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("A3 MLA sink does not yet support PCP or PP")
    if cfg.model_config.max_model_len > MAX_CAPACITY:
        raise ValueError("PR #338 FP16/BF16 contract is limited to 16384 logical KV tokens")
    # AscendMLABackend exposes 128-token kernel pages. Manager blocks (e.g.
    # K3.1's 1536) are expanded by the existing block-table/cache-view layer;
    # they are not the operator's page_size argument.
    if cfg.cache_config.block_size <= 0 or cfg.cache_config.block_size % PAGE_SIZE:
        raise ValueError("A3 MLA integration requires 128-aligned manager blocks")
    if not (cfg.model_config.enforce_eager or spec.enforce_eager or allow_graph):
        raise ValueError("A3 MLA graph use requires explicit a3_mla_sink_allow_graph validation opt-in")
    return True


@dataclass(frozen=True)
class A3MLAAdapter:
    config: A3MLAConfig
    attention_op: Any
    metadata_op: Any
    is_mock: bool = False
    # Native #338 metadata cannot be capacity-queried with device='meta'.
    # Bootstrap it from valid zero-length device tensors once per graph capacity.
    metadata_requires_device_inputs: bool = True

    @classmethod
    def load(cls, config: A3MLAConfig, mode: str, module_name: str = DEFAULT_MODULE):
        if mode not in ("mock", "external"):
            raise ValueError("an A3 provider must explicitly be mock or external")
        module = import_module(MOCK_MODULE if mode == "mock" else module_name)
        attention = getattr(module, "flash_attn_with_kvcache", None)
        metadata = getattr(module, "get_scheduler_metadata", None)
        if not callable(attention) or not callable(metadata):
            raise ImportError("A3 provider must export flash_attn_with_kvcache and get_scheduler_metadata")
        return cls(config, attention, metadata, is_mock=mode == "mock")

    def for_capacity(self, columns: int, tokens: int):
        return replace(
            self, config=replace(self.config, max_seqlen_k=columns * self.config.page_size, max_seqlen_q=tokens)
        )

    def _check_lengths(self, cache_seqlens, cu_seqlens_q, seqused_q):
        batch = cache_seqlens.numel()
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
            allocated = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            # The native contract has no Q-used argument. Whole dummy segments
            # are representable; partially used allocated segments are not.
            torch._assert_async(
                ((seqused_q == 0) | (seqused_q == allocated)).all(),
                "A3 MLA does not support partially used Query segments",
            )
        if not self.is_mock and cache_seqlens.device.type != "npu":
            raise ValueError("native A3 MLA requires NPU tensors")
        return batch

    @staticmethod
    def _effective_lengths(cache_seqlens, seqused_q):
        if seqused_q is None:
            return cache_seqlens
        return torch.where(seqused_q > 0, cache_seqlens, torch.zeros_like(cache_seqlens))

    def build_metadata(self, cache_seqlens, cu_seqlens_q, seqused_q=None):
        batch = self._check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        cfg = self.config
        lengths = self._effective_lengths(cache_seqlens, seqused_q)
        metadata = self.metadata_op(
            batch,
            cfg.max_seqlen_q,
            cfg.max_seqlen_k,
            cfg.num_heads,
            1,
            ROPE_DIM,
            lengths,
            qkv_dtype=cfg.dtype,
            headdim_v=LATENT_DIM,
            cu_seqlens_q=cu_seqlens_q,
            page_size=cfg.page_size,
            causal=cfg.mask_mode == 3,
            num_splits=1,
        )
        if not isinstance(metadata, torch.Tensor) or metadata.dtype != torch.int32 or metadata.ndim != 1:
            raise ValueError("A3 provider metadata must be an opaque int32 vector")
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
        if isinstance(k_cache, tuple) and len(k_cache) == 2:
            v_cache, rope_cache = k_cache
        elif isinstance(k_cache, torch.Tensor) and k_cache.ndim == 4 and k_cache.shape[-1] == QK_DIM:
            v_cache, rope_cache = k_cache[..., :LATENT_DIM], k_cache[..., LATENT_DIM:]
        else:
            raise ValueError("A3 MLA requires (content_cache, rope_cache) or a fused 576-wide cache")
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
        lengths = self._effective_lengths(cache_seqlens, seqused_q)
        result = self.attention_op(
            q[..., LATENT_DIM:].contiguous(),
            rope_cache,
            v_cache,
            qv=q[..., :LATENT_DIM].contiguous(),
            cache_seqlens=lengths,
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
            raise ValueError("PR #338 MLA must return (out, lse, out_accum, lse_accum)")
        output, lse, _, _ = result
        expected_output = (q.shape[0], cfg.num_heads, LATENT_DIM)
        if output.shape != expected_output or output.dtype != cfg.dtype or output.device != q.device:
            raise ValueError("A3 provider output must be TND and match Query dtype/device")
        if lse.shape != (cfg.num_heads, q.shape[0]) or lse.dtype != torch.float32 or lse.device != q.device:
            raise ValueError("A3 provider LSE must be FP32 [H,T] on the Query device")
        if seqused_q is not None and q.shape[0]:
            token = torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
            request = torch.searchsorted(cu_seqlens_q[1:].contiguous(), token, right=True).long()
            live = (token - cu_seqlens_q[request] < seqused_q[request]) & (lengths[request] > 0)
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
