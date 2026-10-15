# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External, unquantized FlashMLA contract; no cache allocation or routing.

Load this adapter after selecting the worker device. Importing the module does
not load cann_ops_transformer. Callers own metadata regeneration, stream ordering
and cache writes. Tensor-content checks belong in preflight tests, not here: the
decode hot path must not copy lengths or block tables to the CPU.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm_ascend.attention.utils import MLA_FLASH_SUPPORTED_Q_HEADS, get_flashmla_ops

FLASHMLA_QK_DIM = 576
FLASHMLA_V_DIM = 512
FLASHMLA_BLOCK_SIZE = 128
FLASHMLA_MASK_SIZE = 2048
FLASHMLA_MAX_BATCH_SIZE = 65535


def flashmla_prefill_flags(
    batch_state, draft_tokens: Mapping[str, Sequence[int]], *, allow_short_extends: bool, attn_groups=()
) -> np.ndarray:
    """Choose the attention route without changing logical prompt progress.

    Main supplies its exact decode-like prompt mask. The legacy lane has no
    mask, so mirror the one-new-token-over-context rule using original scheduler
    draft counts, before adaptive verification compacts those counts.
    """
    logical_prefill = batch_state.is_prefilling_np
    # A mixed model must use an ordering compatible with every MLA group.
    # Native DCP MLA still classifies logical prompt extensions as prefill.
    native_dcp = any(
        getattr(builder, "dcp_enabled", False) and getattr(builder, "flashmla_state", None) is None
        for groups in attn_groups
        for group in groups
        for builder in (group.get_metadata_builder(0),)
    )
    if not allow_short_extends or native_dcp:
        return logical_prefill.copy()
    if "prefill_runs_as_decode_np" in batch_state._fields:
        runs_as_decode = batch_state.prefill_runs_as_decode_np
        if runs_as_decode is None:
            return logical_prefill.copy()
    else:
        num_drafts = np.fromiter(
            (len(draft_tokens.get(req_id, ())) for req_id in batch_state.req_ids),
            dtype=np.int32,
            count=len(batch_state.req_ids),
        )
        runs_as_decode = (batch_state.num_scheduled_tokens - num_drafts == 1) & (
            batch_state.num_computed_prefill_tokens_np > 0
        )
    return logical_prefill & ~runs_as_decode


def split_flashmla_requests(common) -> tuple[int, int, int, int]:
    """Split attention routes after the runner's stable phase ordering.

    The external builder receives execution-phase flags, including eligible
    short prompt extensions. Logical prompt progress remains in the input batch.
    """
    flags = common.is_prefilling
    offsets = common.query_start_loc_cpu[: common.num_reqs + 1]
    if flags is None or flags.device.type != "cpu" or offsets.device.type != "cpu":
        raise RuntimeError("FlashMLA requires CPU is_prefilling and query boundaries from the runner")
    boundaries = offsets.clamp_max(common.num_actual_tokens).tolist()
    stages = flags[: common.num_reqs].tolist()
    if len(stages) < common.num_reqs:
        if any(boundaries[i + 1] != boundaries[i] for i in range(len(stages), common.num_reqs)):
            raise RuntimeError("FlashMLA is_prefilling does not cover the active request batch")
        stages.extend([False] * (common.num_reqs - len(stages)))
    # FIA may append a dummy request to own graph padding. Exclude trailing
    # empty rows so they cannot change our captured buffer capacity or route.
    request_count = common.num_reqs
    while request_count and boundaries[request_count] == boundaries[request_count - 1]:
        request_count -= 1
    first_prefill = request_count
    for index, is_prefill in enumerate(stages[:request_count]):
        if boundaries[index + 1] == boundaries[index]:
            continue
        if is_prefill:
            first_prefill = min(first_prefill, index)
        elif first_prefill != request_count:
            raise RuntimeError("FlashMLA requires real decode requests before prefill requests; check runner ordering")
    decode_tokens = boundaries[first_prefill]
    return first_prefill, request_count - first_prefill, decode_tokens, common.num_actual_tokens - decode_tokens


@dataclass(frozen=True)
class FlashMLAConfig:
    num_heads: int
    softmax_scale: float
    mask_mode: int = 3
    layout_kv: str = "PA_BBND"
    return_softmax_lse: bool = False

    def __post_init__(self) -> None:
        if self.num_heads not in MLA_FLASH_SUPPORTED_Q_HEADS:
            raise ValueError(f"FlashMLA requires local Q heads in {MLA_FLASH_SUPPORTED_Q_HEADS}, got {self.num_heads}")
        if self.mask_mode not in (0, 3):
            raise ValueError("FlashMLA mask_mode must be 0 or 3")
        if self.layout_kv != "PA_BBND":
            raise ValueError("FlashMLA layout_kv must be PA_BBND")


def _check_tensor(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    if tensor.shape != shape or tensor.dtype != dtype:
        raise ValueError(f"FlashMLA {name} requires shape={shape}, dtype={dtype}; got {tensor.shape}, {tensor.dtype}")


def _check_lengths(
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_q: torch.Tensor | None,
) -> int:
    if cache_seqlens.ndim != 1 or not 0 < cache_seqlens.numel() <= FLASHMLA_MAX_BATCH_SIZE:
        raise ValueError("FlashMLA cache_seqlens must be a nonempty vector with fewer than 65536 requests")
    batch_size = cache_seqlens.numel()
    _check_tensor("cache_seqlens", cache_seqlens, (batch_size,), torch.int32)
    _check_tensor("cu_seqlens_q", cu_seqlens_q, (batch_size + 1,), torch.int32)
    if cu_seqlens_q.device != cache_seqlens.device:
        raise ValueError("FlashMLA query and KV lengths must be on the same device")
    if seqused_q is not None:
        _check_tensor("seqused_q", seqused_q, (batch_size,), torch.int32)
        if seqused_q.device != cache_seqlens.device:
            raise ValueError("FlashMLA seqused_q must be on the same device as cache_seqlens")
    return batch_size


@dataclass(frozen=True)
class FlashMLAAdapter:
    """Keep metadata and attention attributes identical, preserving input storage.

    The installed package remains responsible for its hardware/stride support
    and metadata capacity. Accepting a view here is not evidence that the NPU
    binary supports it. No exception from an operator call is silently retried
    with a contiguous cache or a different attention backend.
    """

    config: FlashMLAConfig
    attention_op: Callable
    metadata_op: Callable

    @classmethod
    def load(cls, config: FlashMLAConfig) -> "FlashMLAAdapter":
        available_ops = get_flashmla_ops()
        if available_ops is None:
            raise RuntimeError(
                "FlashMLA requires cann_ops_transformer.ops.flash_mla "
                "and flash_mla_metadata. Install a package matching the worker's "
                "CANN and torch_npu versions."
            )
        attention_op, metadata_op = available_ops
        return cls(config, attention_op, metadata_op)

    def build_metadata(
        self,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate metadata for these device lengths, or query capacity on Meta.

        The package Meta implementation may query NPU core counts; select the
        worker device first. Do not replace -1 with CPU maximum-length bounds.
        """
        _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        return self.metadata_op(
            cache_seqlens,
            num_heads_q=self.config.num_heads,
            num_heads_kv=1,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            head_dim_qk=FLASHMLA_QK_DIM,
            head_dim_v=FLASHMLA_V_DIM,
            mask_mode=self.config.mask_mode,
            layout_q="TND",
        )

    def attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        *,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Consume caller-owned paged KV and return latent output in NTD order.

        Metadata must have been generated from these same lengths and this
        config. Regenerate it when lengths change, including graph replay.
        """
        self._check_inputs(q, k_cache, block_table, cache_seqlens, cu_seqlens_q, metadata, seqused_q, attn_mask)
        output, lse = self.attention_op(
            q,
            k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=seqused_q,
            attn_mask=attn_mask,
            metadata=metadata,
            head_dim_v=FLASHMLA_V_DIM,
            softmax_scale=self.config.softmax_scale,
            mask_mode=self.config.mask_mode,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q="TND",
            layout_kv=self.config.layout_kv,
            layout_out="NTD",
            return_softmax_lse=self.config.return_softmax_lse,
        )
        if self.config.return_softmax_lse:
            # TND queries return NTD latent output and NT FP32 statistics.
            # DCP must not guess an installed package's LSE layout or precision.
            _check_tensor("output", output, (self.config.num_heads, q.shape[0], FLASHMLA_V_DIM), q.dtype)
            _check_tensor("softmax_lse", lse, (self.config.num_heads, q.shape[0]), torch.float32)
        return output, lse

    def _check_inputs(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        metadata: torch.Tensor,
        seqused_q: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> None:
        batch_size = _check_lengths(cache_seqlens, cu_seqlens_q, seqused_q)
        if q.ndim != 3 or q.shape[0] == 0 or q.shape[1:] != (self.config.num_heads, FLASHMLA_QK_DIM):
            raise ValueError("FlashMLA q must be nonempty TND with configured local heads and head dimension 576")
        if q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FlashMLA q must be BF16 or FP16")
        cache_tail = (FLASHMLA_BLOCK_SIZE, 1, FLASHMLA_QK_DIM)
        if k_cache.ndim != len(cache_tail) + 1 or k_cache.shape[1:] != cache_tail or k_cache.shape[0] == 0:
            raise ValueError(f"FlashMLA {self.config.layout_kv} cache must have shape (pages, {cache_tail})")
        if k_cache.dtype != q.dtype:
            raise ValueError("FlashMLA q and k_cache must have the same dtype")
        if block_table.ndim != 2 or block_table.shape[0] != batch_size or block_table.shape[1] == 0:
            raise ValueError("FlashMLA block_table must have shape (batch_size, positive_table_capacity)")
        if block_table.dtype != torch.int32:
            raise ValueError("FlashMLA block_table must be int32")
        if metadata.ndim != 1 or metadata.numel() == 0 or metadata.dtype != torch.int32:
            raise ValueError("FlashMLA metadata must be a nonempty int32 vector from the metadata operator")
        if self.config.mask_mode == 0 and attn_mask is not None:
            raise ValueError("FlashMLA mask_mode=0 requires attn_mask=None")
        if self.config.mask_mode == 3:
            if attn_mask is None:
                raise ValueError("FlashMLA mask_mode=3 requires the upper-triangular int8 mask")
            _check_tensor("attn_mask", attn_mask, (FLASHMLA_MASK_SIZE, FLASHMLA_MASK_SIZE), torch.int8)
        for tensor in (k_cache, block_table, cache_seqlens, metadata, attn_mask):
            if tensor is not None and tensor.device != q.device:
                raise ValueError("FlashMLA inputs must be on the same device")
