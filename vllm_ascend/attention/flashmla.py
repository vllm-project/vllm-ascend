# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""External, unquantized FlashMLA implementation and metadata builder.

Call FlashMLAImpl.load after selecting the worker device. Importing this module
does not load cann_ops_transformer. FlashMLAImpl wraps the operator contract, while
FlashMLAMetadataBuilder owns per-batch inputs, stable buffers and per-build refreshes.
Callers own stream ordering and persistent KV writes. Tensor-content checks belong
in preflight tests: the decode hot path must not copy lengths or tables to the CPU.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.utils import MLA_FLASH_SUPPORTED_Q_HEADS, get_flashmla_ops
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import BatchReqState

FLASHMLA_QK_DIM = 576
FLASHMLA_V_DIM = 512
FLASHMLA_BLOCK_SIZE = 128
FLASHMLA_MASK_SIZE = 2048
FLASHMLA_MAX_BATCH_SIZE = 65535


def prepare_flashmla_batch_req_state(
    batch_state: "BatchReqState", uniform_token_count: int | None
) -> tuple["BatchReqState", int | None]:
    """Order real decode before FIA prefill before inputs and graph dispatch.

    Reorder the release request-state fields together so positions, KV slots
    and sampling indices inherit the same order. A one-token prompt remains
    FIA prefill and cannot reuse a graph captured for FlashMLA decode.
    """
    order = np.argsort(batch_state.is_prefilling_np, kind="stable")
    reordered_fields: dict[str, Any] = dict(
        req_ids=[batch_state.req_ids[index] for index in order],
        num_scheduled_tokens=batch_state.num_scheduled_tokens[order],
        idx_mapping_np=batch_state.idx_mapping_np[order],
        prefill_len_np=batch_state.prefill_len_np[order],
        num_computed_prefill_tokens_np=batch_state.num_computed_prefill_tokens_np[order],
        is_prefilling_np=batch_state.is_prefilling_np[order],
    )
    # Main carries additional per-request arrays; release lanes may omit them.
    for field in ("num_draft_tokens_np", "prefill_runs_as_decode_np", "max_seq_len_np"):
        if field in batch_state._fields:
            values = getattr(batch_state, field)
            reordered_fields[field] = values[order] if values is not None else None
    batch_state = batch_state._replace(**reordered_fields)
    if batch_state.has_prefill:
        if "decode_graph_eligible" in batch_state._fields:
            batch_state = batch_state._replace(decode_graph_eligible=False)
        uniform_token_count = None
    elif "decode_graph_eligible" in batch_state._fields and not batch_state.decode_graph_eligible:
        # Retain upstream's restrictive graph gate after PD reclassification.
        uniform_token_count = None
    return batch_state, uniform_token_count


def split_flashmla_requests(common) -> tuple[int, int, int, int]:
    """Split real generation stages after the runner's stable phase ordering.

    Read only CPU scheduler metadata. A one-token prompt suffix is still FIA
    prefill; query length alone must never opt a request into FlashMLA.
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
class FlashMLAImpl:
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
    def load(cls, config: FlashMLAConfig) -> "FlashMLAImpl":
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


@dataclass
class FlashMLACurrent:
    """Replicated current chunk, independent of persistent KV and prefix pages."""

    adapter: FlashMLAImpl
    query: torch.Tensor
    schedule: torch.Tensor
    cache: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    attn_mask: torch.Tensor


@dataclass
class FlashMLADecode:
    adapter: FlashMLAImpl
    query: torch.Tensor
    schedule: torch.Tensor
    cu: torch.Tensor
    used_q: torch.Tensor
    cache_lens: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    positions: torch.Tensor
    token_live: torch.Tensor
    live_boundaries: torch.Tensor
    attn_mask: torch.Tensor | None
    cos: torch.Tensor | None
    sin: torch.Tensor | None
    current: FlashMLACurrent | None = None


class FlashMLAMetadataBuilder:
    def __init__(self, impl, device: torch.device, max_num_reqs: int, attn_mask: torch.Tensor):
        self.dcp_size = getattr(impl, "dcp_size", 1)
        self.dcp_rank = getattr(impl, "dcp_rank", 0)
        # Materialize once: an integer rank would trigger a blocking H2D copy
        # inside get_dcp_local_seq_lens on every metadata refresh.
        self.dcp_rank_tensor = (
            torch.tensor(self.dcp_rank, dtype=torch.int32, device=device) if self.dcp_size > 1 else None
        )
        # Keep the DCP execution structure fixed across capture and replay.
        # Ordinary decode reads current KV from its owning rank's cache;
        # speculative causal queries need the replicated current chunk.
        self.use_spec_decode = impl.vllm_config.speculative_config is not None
        self.cp_interleave = impl.vllm_config.parallel_config.cp_kv_cache_interleave_size if self.dcp_size > 1 else 1
        adapter = FlashMLAImpl.load(FlashMLAConfig(impl.num_heads, impl.scale, return_softmax_lse=self.dcp_size > 1))
        self.current_adapter = adapter
        history_adapter = replace(
            adapter,
            config=replace(adapter.config, num_heads=impl.num_heads * self.dcp_size, mask_mode=0),
        )
        self.adapters = {True: history_adapter if self.dcp_size > 1 else adapter, False: history_adapter}
        self.device = device
        self.dtype = impl.dtype
        self.use_rope = impl.use_mla_rope
        self.max_num_reqs = max_num_reqs
        self.buffers: dict[tuple, FlashMLADecode] = {}
        self.attn_mask = attn_mask

    def _allocate(self, tokens: int, rows: int, columns: int, causal: bool) -> FlashMLADecode:
        ints = {"dtype": torch.int32, "device": self.device}
        adapter = self.adapters[causal]
        cu = torch.zeros(rows + 1, **ints)
        used = torch.zeros(rows, **ints)
        lengths = torch.zeros(rows, **ints)
        schedule_meta = adapter.build_metadata(
            torch.empty_like(lengths, device="meta"),
            torch.empty_like(cu, device="meta"),
            torch.empty_like(used, device="meta"),
        )
        rope_shape = (tokens, 1, 1, FLASHMLA_QK_DIM - FLASHMLA_V_DIM)
        flash = FlashMLADecode(
            adapter=adapter,
            query=torch.empty(
                (tokens, adapter.config.num_heads, FLASHMLA_QK_DIM), dtype=self.dtype, device=self.device
            ),
            schedule=torch.empty_like(schedule_meta, device=self.device),
            cu=cu,
            used_q=used,
            cache_lens=lengths,
            block_table=torch.zeros((rows, columns), **ints),
            slots=torch.full((tokens,), -1, dtype=torch.int64, device=self.device),
            positions=torch.zeros(tokens, dtype=torch.int64, device=self.device),
            token_live=torch.zeros(tokens, dtype=torch.bool, device=self.device),
            live_boundaries=torch.zeros(tokens + 1, **ints),
            attn_mask=self.attn_mask if adapter.config.mask_mode == 3 else None,
            cos=torch.empty(rope_shape, dtype=self.dtype, device=self.device) if self.use_rope else None,
            sin=torch.empty(rope_shape, dtype=self.dtype, device=self.device) if self.use_rope else None,
        )
        if self.dcp_size > 1 and causal and self.use_spec_decode:
            # As in split DCP attention, each rank retains every current row.
            # Only persistent history writes use the runner's rank-owned slots.
            block_columns = (tokens + FLASHMLA_BLOCK_SIZE - 1) // FLASHMLA_BLOCK_SIZE
            current_schedule = self.current_adapter.build_metadata(
                torch.empty_like(used, device="meta"),
                torch.empty_like(cu, device="meta"),
                torch.empty_like(used, device="meta"),
            )
            flash.current = FlashMLACurrent(
                adapter=self.current_adapter,
                query=torch.empty(
                    (tokens, self.current_adapter.config.num_heads, FLASHMLA_QK_DIM),
                    dtype=self.dtype,
                    device=self.device,
                ),
                schedule=torch.empty_like(current_schedule, device=self.device),
                cache=torch.empty(
                    (block_columns + rows, FLASHMLA_BLOCK_SIZE, 1, FLASHMLA_QK_DIM),
                    dtype=self.dtype,
                    device=self.device,
                ),
                block_table=torch.zeros((rows, block_columns), **ints),
                slots=torch.full_like(flash.slots, -1),
                attn_mask=self.attn_mask,
            )
        return flash

    def build(
        self, common, num_decodes: int, num_decode_tokens: int, has_prefill: bool, *, retain_for_graph: bool = False
    ) -> FlashMLADecode:
        # Mixed batches run outside FULL graph. Do not retain every prefill
        # shape. Retain only explicitly captured decode capacities: retaining
        # every eager token count grows total query storage quadratically.
        # Replay (and eager with that capacity) reuses the captured addresses.
        tokens = num_decode_tokens if has_prefill else max(common.num_actual_tokens, common.num_input_tokens)
        rows = num_decodes + 1 if has_prefill else max(num_decodes, min(tokens, self.max_num_reqs)) + 1
        columns = common.block_table_tensor.shape[1]
        key = tokens, rows, columns, common.causal
        if has_prefill:
            if retain_for_graph:
                raise ValueError("FlashMLA graph buffers require a decode-only batch")
            flash = self._allocate(tokens, rows, columns, common.causal)
        else:
            cached_flash = self.buffers.get(key)
            if cached_flash is None:
                flash = self._allocate(tokens, rows, columns, common.causal)
                if retain_for_graph:
                    self.buffers[key] = flash
            else:
                flash = cached_flash

        def refresh():
            # The extra zero-used request owns physical graph/SP padding.
            flash.cu.fill_(tokens)
            flash.cu[: num_decodes + 1].copy_(common.query_start_loc[: num_decodes + 1].clamp_max(num_decode_tokens))
            flash.used_q.zero_()
            flash.used_q[:num_decodes].copy_(flash.cu[1 : num_decodes + 1] - flash.cu[:num_decodes])
            flash.used_q[:num_decodes].masked_fill_(common.seq_lens[:num_decodes] <= 0, 0)
            flash.cache_lens.zero_()
            lengths = common.seq_lens[:num_decodes]
            if flash.current is not None:
                # Remove the query globally before interleave-aware sharding.
                # Ordinary decode and noncausal draft see the entire local sequence.
                lengths = (lengths - flash.used_q[:num_decodes]).clamp_min(0)
            if self.dcp_size > 1:
                lengths = get_dcp_local_seq_lens(
                    lengths,
                    dcp_size=self.dcp_size,
                    dcp_rank=self.dcp_rank_tensor,
                    cp_kv_cache_interleave_size=self.cp_interleave,
                )
            flash.cache_lens[:num_decodes].copy_(lengths)
            flash.cache_lens.masked_fill_(flash.used_q == 0, 0)
            flash.block_table.zero_()
            flash.block_table[:num_decodes].copy_(common.block_table_tensor[:num_decodes])
            flash.slots.fill_(-1)
            slots = common.slot_mapping[:num_decode_tokens]
            flash.slots[: slots.numel()].copy_(slots)
            flash.live_boundaries.zero_()
            live = (flash.used_q > 0).to(torch.int32)
            flash.live_boundaries.scatter_add_(0, flash.cu[:-1].long(), live)
            flash.live_boundaries.scatter_add_(0, (flash.cu[:-1] + flash.used_q).long(), -live)
            flash.token_live.copy_(flash.live_boundaries.cumsum(0)[:tokens] > 0)
            flash.slots.masked_fill_(~flash.token_live, -1)
            flash.positions.zero_()
            positions = common.positions[:num_decode_tokens]
            flash.positions[: positions.numel()].copy_(positions)
            if self.use_rope:
                assert flash.cos is not None and flash.sin is not None
                cos, sin = get_cos_and_sin_mla(flash.positions)
                flash.cos.copy_(cos)
                flash.sin.copy_(sin)
            flash.schedule.copy_(flash.adapter.build_metadata(flash.cache_lens, flash.cu, flash.used_q))
            if flash.current is not None:
                current = flash.current
                page_counts = (flash.used_q + FLASHMLA_BLOCK_SIZE - 1) // FLASHMLA_BLOCK_SIZE
                page_starts = page_counts.cumsum(0) - page_counts
                columns = torch.arange(current.block_table.shape[1], device=self.device)
                current.block_table.copy_(page_starts[:, None] + columns)
                current.block_table.masked_fill_(columns >= page_counts[:, None], 0)
                indices = torch.arange(tokens, device=self.device)
                requests = torch.searchsorted(flash.cu[1:], indices.to(torch.int32), right=True)
                current.slots.copy_(page_starts[requests] * FLASHMLA_BLOCK_SIZE + indices - flash.cu[requests])
                current.slots.masked_fill_(~flash.token_live, -1)
                current.schedule.copy_(current.adapter.build_metadata(flash.used_q, flash.cu, flash.used_q))

        # Refresh on the caller's stream before return; captured buffer caching
        # preserves the addresses used by graph replay.
        refresh()
        return flash
