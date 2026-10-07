# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Builder-owned FlashMLA buffers and device-metadata tasks.

Follows the reference's stable-buffer lifecycle, using the external package's
Meta implementation and the existing executor. It never owns persistent KV.
"""

from dataclasses import dataclass, replace

import torch
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

from vllm_ascend.attention.flashmla import (
    FLASHMLA_BLOCK_SIZE,
    FLASHMLA_QK_DIM,
    FLASHMLA_V_DIM,
    FlashMLAAdapter,
    FlashMLAConfig,
)
from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla
from vllm_ascend.worker.device_metadata import DeviceMetadataExecutor, DeviceMetadataStage, DeviceMetadataTask


@dataclass
class FlashMLACurrent:
    """Replicated current chunk, independent of persistent KV and prefix pages."""

    adapter: FlashMLAAdapter
    query: torch.Tensor
    schedule: torch.Tensor
    cache: torch.Tensor
    block_table: torch.Tensor
    slots: torch.Tensor
    attn_mask: torch.Tensor


@dataclass
class FlashMLADecode:
    adapter: FlashMLAAdapter
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
        self.cp_interleave = impl.vllm_config.parallel_config.cp_kv_cache_interleave_size if self.dcp_size > 1 else 1
        adapter = FlashMLAAdapter.load(FlashMLAConfig(impl.num_heads, impl.scale, return_softmax_lse=self.dcp_size > 1))
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
        self.defer = False
        self.executor: DeviceMetadataExecutor | None = None
        self.tasks: tuple[DeviceMetadataTask, ...] = ()

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
        if self.dcp_size > 1 and causal:
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
            flash = self.buffers.get(key)
            if flash is None:
                flash = self._allocate(tokens, rows, columns, common.causal)
                if retain_for_graph:
                    self.buffers[key] = flash

        def refresh():
            # All refreshes belong to the executor task, after its reuse fence.
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
                # Noncausal draft queries instead see the entire local sequence.
                lengths = (lengths - flash.used_q[:num_decodes]).clamp_min(0)
            if self.dcp_size > 1:
                lengths = get_dcp_local_seq_lens(
                    lengths,
                    dcp_size=self.dcp_size,
                    dcp_rank=self.dcp_rank,
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

        self.tasks = (DeviceMetadataTask(DeviceMetadataStage.ATTENTION, refresh, id(flash.schedule)),)
        if not self.defer:
            refresh()
            self.tasks = ()
        return flash

    def take_tasks(self) -> tuple[DeviceMetadataTask, ...]:
        tasks, self.tasks = self.tasks, ()
        return tasks
