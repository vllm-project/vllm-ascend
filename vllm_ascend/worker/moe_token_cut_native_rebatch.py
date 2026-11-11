# SPDX-License-Identifier: Apache-2.0
"""Rank-local persistent storage for the supplied BF16 rebatch_and_update.

Rebatch combine saves dropped rows. MoeTokenCut.RestoreDiscarded appends the
scheduled cache subset; rebatch combine then takes its no-cut fast path and
updates Attention metadata only. Rebatch dispatch is never used for refill.
Each store owns one physical layer, exposed as a one-row Attention bucket.
"""

from dataclasses import dataclass

import torch

from vllm_ascend.ops.fused_moe.moe_token_cut_native import (
    MOE_TOKEN_CUT_SCHEMA,
    REBATCH_AND_UPDATE_SCHEMA,
    invoke_native_op,
    validate_native_schema,
)

# CombineRebatch uses posBuf[100] and 8-KiB row/top-k UB buffers.
REBATCH_CAPACITY = 96
REBATCH_ROW_BYTES = 8192
INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class NativeRestoreContext:
    """Reuse the EP resources of the successful cut that created the cache."""

    group_ep: str
    rank_id: int
    rank_num: int
    expert_num: int
    send_aiv_num: int = 8
    recv_aiv_num: int = 8

    def __post_init__(self):
        if (
            not self.group_ep
            or not 0 < self.rank_num <= 384
            or not 0 <= self.rank_id < self.rank_num
            or not 0 < self.expert_num <= 512
            or self.expert_num % self.rank_num
            or self.send_aiv_num <= 0
            or self.recv_aiv_num <= 0
        ):
            raise ValueError("native restore requires valid EP topology and AIV counts")


def validate_rebatch_operator():
    # v0.27.1 loads the extension lazily. Runner construction is after
    # torch.npu.set_device, so registration is safe here and precedes weights.
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("native rebatch_backend requires the custom operator extension to load")
    try:
        op = torch.ops._C_ascend.rebatch_and_update.default
    except AttributeError as error:
        raise RuntimeError("native rebatch_backend requires the installed rebatch_and_update operator") from error
    validate_native_schema(op._schema, REBATCH_AND_UPDATE_SCHEMA)
    try:
        cut_op = torch.ops._C_ascend.moe_token_cut.default
    except AttributeError as error:
        raise RuntimeError("native rebatch_backend requires the installed moe_token_cut restore operator") from error
    validate_native_schema(cut_op._schema, MOE_TOKEN_CUT_SCHEMA)
    return str(op._schema)


def checked_int32(value: torch.Tensor) -> torch.Tensor:
    if value.dtype not in (torch.int32, torch.int64):
        raise ValueError("rebatch indices and Attention metadata must be int32/int64")
    result = value.to(torch.int32)
    # Eager-only correctness check: never silently truncate 64-bit KV addresses.
    if value.dtype == torch.int64 and not torch.equal(value, result.to(torch.int64)):
        raise ValueError("rebatch int32 index/slot/block-table overflow")
    return result


@dataclass
class NativeRebatchRows:
    hidden: torch.Tensor
    residual: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    def select(self, indices: torch.Tensor) -> "NativeRebatchRows":
        return NativeRebatchRows(**{name: value.index_select(0, indices) for name, value in vars(self).items()})


@dataclass
class NativeResumeCheckpoint:
    """A stable request handle, not a tensor view invalidated by cache packing."""

    layer_idx: int
    request_id: str
    store: "NativeRebatchStore"
    seq_len_cpu: int

    def _row(self, name: str) -> torch.Tensor:
        return self.store.row(self.request_id, name)

    @property
    def hidden_state(self):
        return self._row("hidden")

    @property
    def residual(self):
        return self._row("residual")

    @property
    def position(self):
        return self._row("positions")

    @property
    def topk_ids(self):
        return self._row("topk_ids")

    @property
    def topk_weights(self):
        return self._row("topk_weights")

    @property
    def slot_mapping(self):
        return self._row("slots")

    @property
    def seq_len(self):
        return self._row("seq_lens")

    @property
    def block_table(self):
        return self._row("block_tables")


class NativeRebatchStore:
    def __init__(self, example: NativeRebatchRows, *, restore_context: NativeRestoreContext | None = None):
        self._validate(example)
        self.device = example.hidden.device
        self.pending: tuple[str, ...] = ()
        self._serial = 0
        self._native_ids: dict[str, int] = {}
        self.rows = NativeRebatchRows(
            **{name: value.new_zeros((REBATCH_CAPACITY, *value.shape[1:])) for name, value in vars(example).items()}
        )
        self.rows.topk_ids = self.rows.topk_ids.to(torch.int32)
        for name in ("slots", "seq_lens", "block_tables"):
            setattr(self.rows, name, getattr(self.rows, name).to(torch.int32))
        self.moe_ids = torch.full((REBATCH_CAPACITY,), -1, dtype=torch.int32, device=self.device)
        self.attn_ids = self.moe_ids.clone().unsqueeze(0)
        self.attn_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.moe_count = torch.zeros_like(self.attn_count)
        self.save_calls = 0
        self.restore_calls = 0
        self.token_restore_calls = 0
        self.restore_context = restore_context

    def _restore_payload(self, values, work, cache, cache_ids, cache_count):
        context = self.restore_context
        if context is None:
            raise RuntimeError("native token restore requires the EP context of the originating cut")
        if work.topk_ids.shape[1] > context.expert_num:
            raise ValueError("native token restore topk must not exceed expert_num")
        scalar = lambda value: torch.tensor([value], dtype=torch.int32, device=self.device)
        # The supplied kernel restores BEFORE testing enable_cut. With cut=0
        # it performs only local copies/SyncAll, no counts/window exchange.
        # Do not advance the cross-rank cut mailbox generation for this call.
        cut_values = dict(
            topk_ids=work.topk_ids,
            topk_weights=work.topk_weights,
            hidden=work.hidden,
            cutidx=values["cut_idx"],
            new_topk=torch.zeros_like(work.topk_ids),
            new_topkweight=torch.zeros_like(work.topk_weights),
            new_hidden=torch.zeros_like(work.hidden),
            new_residuals=torch.zeros_like(work.residual),
            actual_bs=values["actual_bs"],
            prefix_index=values["prefix_index"],
            no_cut_rank_num=scalar(1),
            active_mask=torch.zeros(REBATCH_CAPACITY, dtype=torch.bool, device=self.device),
            local_cut_count=scalar(0),
            batch_size=values["running_bs"],
            enable_cut=scalar(0),
            residuals=work.residual,
            discard_label=values["discard_label"],
            discard_tokens=cache.hidden,
            discard_moe_req_id=cache_ids,
            req_id_list=values["req_id_list"],
            discard_residuals=cache.residual,
            discard_topk=cache.topk_ids,
            discard_topk_weight=cache.topk_weights,
            moe_cnt=cache_count,
            history_bs=values["history_bs"],
            original_bs=values["original_bs"],
            group_ep=context.group_ep,
            topk=work.topk_ids.shape[1],
            rank_id=context.rank_id,
            layer_idx=0,
            rank_num=context.rank_num,
            expert_num=context.expert_num,
            expert_per_rank=context.expert_num // context.rank_num,
            level_limit=0,
            cut_count_threshold=REBATCH_CAPACITY,
            small_expert_token_threshold=0,
            send_aiv_num=context.send_aiv_num,
            recv_aiv_num=context.recv_aiv_num,
        )
        invoke_native_op(MOE_TOKEN_CUT_SCHEMA, cut_values)
        self.token_restore_calls += 1
        # Consume the kernel's identity/compacted output, not a Python copy of
        # the cache. Metadata is still restored by the following fused call.
        work.hidden = cut_values["new_hidden"]
        work.residual = cut_values["new_residuals"]
        work.topk_ids = cut_values["new_topk"]
        work.topk_weights = cut_values["new_topkweight"]
        values.update(
            batch_before_dispatch=work.hidden,
            residuals_before_dispatch=work.residual,
            running_topk=work.topk_ids,
            running_topk_weight=work.topk_weights,
        )

    @staticmethod
    def _validate(rows: NativeRebatchRows):
        n = rows.hidden.shape[0]
        if rows.hidden.ndim != 2 or rows.hidden.dtype != torch.bfloat16:
            raise ValueError("native rebatch requires 2-D BF16 hidden states")
        if not 0 < rows.hidden.shape[1] * rows.hidden.element_size() <= REBATCH_ROW_BYTES:
            raise ValueError("native rebatch hidden row exceeds the kernel's 8-KiB buffer")
        if rows.residual.shape != rows.hidden.shape or rows.residual.dtype != torch.bfloat16:
            raise ValueError("native rebatch requires matching BF16 residuals")
        if rows.topk_ids.ndim != 2 or rows.topk_weights.shape != rows.topk_ids.shape:
            raise ValueError("native rebatch requires matching 2-D routing tensors")
        if rows.topk_weights.dtype != torch.bfloat16:
            raise ValueError("native rebatch requires BF16 routing weights; implicit rounding is not allowed")
        if not 0 < rows.topk_ids.shape[1] * 4 <= REBATCH_ROW_BYTES:
            raise ValueError("native rebatch routing row exceeds the kernel's 8-KiB buffer")
        if rows.positions.dtype != torch.int64 or rows.positions.ndim != 1:
            raise ValueError("native rebatch requires int64 vector positions")
        if rows.slots.ndim != 1 or rows.seq_lens.ndim != 1 or rows.block_tables.ndim != 2:
            raise ValueError("native rebatch requires decode Attention metadata")
        if not 0 < rows.block_tables.shape[1] * 4 <= REBATCH_ROW_BYTES:
            raise ValueError("native rebatch requires a non-empty block-table row within 8 KiB")
        if not 0 <= n <= 64:
            raise ValueError(f"native rebatch requires at most 64 request rows, got {n}")
        for name, value in vars(rows).items():
            if value.shape[0] != n:
                raise ValueError(f"native rebatch {name} has {value.shape[0]} rows, expected {n}")
            if value.device != rows.hidden.device:
                raise ValueError(f"native rebatch {name} is on {value.device}, expected {rows.hidden.device}")
        for name in ("topk_ids", "slots", "seq_lens", "block_tables"):
            checked_int32(getattr(rows, name))

    def row(self, request_id: str, field: str) -> torch.Tensor:
        return getattr(self.rows, field)[self.pending.index(request_id)]

    def _padded(self, rows: NativeRebatchRows) -> NativeRebatchRows:
        self._validate(rows)
        result = {}
        for name, source in vars(rows).items():
            target_example = getattr(self.rows, name)
            if source.shape[1:] != target_example.shape[1:] or source.device != self.device:
                raise ValueError("native rebatch layer shape/device changed while caches exist")
            value = target_example.new_zeros(target_example.shape)
            value[: source.shape[0]].copy_(source)
            result[name] = value
        return NativeRebatchRows(**result)

    def _ids(self, request_ids: tuple[str, ...]) -> list[int]:
        if self._serial + len(request_ids) > INT32_MAX:
            raise RuntimeError("native rebatch request-ID counter exhausted")
        result = list(range(self._serial, self._serial + len(request_ids)))
        self._serial += len(request_ids)
        return result

    def _invoke(self, rows, *, count, original_count, ids, keep, restore_rows=None):
        work = self._padded(rows)
        cap = REBATCH_CAPACITY
        labels = torch.zeros(cap, dtype=torch.int32, device=self.device)
        labels[:count] = 1
        req_ids = torch.full_like(labels, -1)
        if ids:
            req_ids[:count] = torch.tensor(ids, dtype=torch.int32, device=self.device)
        cut_idx = torch.zeros(cap, dtype=torch.int8, device=self.device)
        cut_idx[:count] = keep.to(torch.int8)
        prefix = cut_idx.to(torch.int32).cumsum(0, dtype=torch.int32)
        scalar = lambda value: torch.tensor([value], dtype=torch.int32, device=self.device)
        output_positions = torch.zeros_like(work.positions)
        output_slots = torch.full_like(work.slots, -1)
        output_seq = torch.zeros_like(work.seq_lens)
        output_blocks = torch.zeros_like(work.block_tables)
        # Token-cut restore consumes ALL supplied cache rows. Pass only the
        # scheduled subset; retain the full Attention bucket for ID lookup.
        cache = self.rows if restore_rows is None else restore_rows[0]
        cache_ids = self.moe_ids if restore_rows is None else restore_rows[1]
        cache_count = self.moe_count if restore_rows is None else scalar(restore_rows[2])
        values = dict(
            batch_before_dispatch=work.hidden,
            residuals_before_dispatch=work.residual,
            cut_idx=cut_idx,
            prefix_index=prefix,
            discard_label=labels,
            discard_tokens=cache.hidden,
            discard_moe_req_id_tensor=cache_ids,
            req_id_list=req_ids,
            discard_residuals=cache.residual,
            running_topk=work.topk_ids,
            discard_topk=cache.topk_ids,
            running_topk_weight=work.topk_weights,
            discard_topk_weight=cache.topk_weights,
            moe_cnt_input=cache_count,
            actual_bs=keep.sum(dtype=torch.int32).reshape(1),
            history_bs=scalar(0),
            running_bs=scalar(count),
            original_bs=scalar(original_count),
            positions=work.positions,
            slot_mapping=work.slots,
            block_table=work.block_tables,
            seq_lens=work.seq_lens,
            discard_positions=self.rows.positions.unsqueeze(0),
            discard_slot_mapping=self.rows.slots.unsqueeze(0),
            discard_context_lens=self.rows.seq_lens.unsqueeze(0),
            discard_block_tables=self.rows.block_tables.unsqueeze(0),
            discard_attn_cnt=self.attn_count,
            discard_req_ids=self.attn_ids,
            # On restore actual_bs == running_bs, so CombineRebatch returns
            # without touching payload. Attention still consumes history/IDs.
            is_combine_rebatch=1,
            max_bs=cap,
            k=work.topk_ids.shape[1],
            layer_idx=0,
            # This kernel never writes rebatchResiduals. Do not read this output.
            out_rebatch_residuals=torch.zeros_like(work.residual),
            out_positions=output_positions,
            out_slots=output_slots,
            out_seq_lens=output_seq,
            out_block_table=output_blocks,
        )
        if restore_rows is not None:
            self._restore_payload(values, work, cache, cache_ids, cache_count)
        print("rebatch_call before", flush = True)
        invoke_native_op(REBATCH_AND_UPDATE_SCHEMA, values)
        print("rebatch_call after", flush = True)
        work.positions, work.slots = output_positions, output_slots
        work.seq_lens, work.block_tables = output_seq, output_blocks
        return work

    def park(self, request_ids, rows, keep, dropped_ids):
        self._validate(rows)
        if len(request_ids) != rows.hidden.shape[0] or len(set(request_ids)) != len(request_ids):
            raise ValueError("native rebatch requires unique row-aligned request IDs")
        if keep.shape != (len(request_ids),) or keep.dtype != torch.bool or keep.device != self.device:
            raise ValueError("native rebatch keep mask must match the current rows/device")
        if len(set(dropped_ids)) != len(dropped_ids) or not set(dropped_ids).issubset(request_ids):
            raise ValueError("native rebatch dropped IDs must be a unique subset of current requests")
        if len(self.pending) + len(dropped_ids) > REBATCH_CAPACITY:
            raise RuntimeError("native rebatch cache capacity exhausted; drain parked requests first")
        if set(request_ids).intersection(self.pending):
            raise RuntimeError("cannot save a request that is still in the native rebatch cache")
        native_ids = self._ids(request_ids)
        output = self._invoke(rows, count=len(request_ids), original_count=len(request_ids), ids=native_ids, keep=keep)
        by_id = dict(zip(request_ids, native_ids))
        self._native_ids.update((rid, by_id[rid]) for rid in dropped_ids)
        self.pending += tuple(dropped_ids)
        self.save_calls += 1
        return output

    def restore(self, current_ids, rows, restore_ids):
        if not restore_ids or len(set(restore_ids)) != len(restore_ids):
            raise ValueError("native rebatch restore requires a non-empty unique selection")
        if set(current_ids).intersection(restore_ids):
            raise ValueError("a request cannot be both running and parked")
        n = len(current_ids)
        if len(set(current_ids)) != n or rows.hidden.shape[0] != n:
            raise ValueError("native rebatch requires unique row-aligned current request IDs")
        # A dummy -1 original slot avoids zero-sized TBuf allocation in the
        # supplied Attention kernel when every scheduled row was parked.
        original_count = max(n, 1)
        if original_count + len(restore_ids) > REBATCH_CAPACITY or n + len(restore_ids) > 64:
            raise ValueError("native rebatch restore exceeds the decode/cached-label capacity")
        indices = torch.tensor([self.pending.index(rid) for rid in restore_ids], device=self.device)
        selected = self._padded(self.rows.select(indices))
        native_ids = torch.full_like(self.moe_ids, -1)
        native_ids[: len(restore_ids)] = torch.tensor(
            [self._native_ids[rid] for rid in restore_ids], dtype=torch.int32, device=self.device
        )
        keep = torch.ones(n, dtype=torch.bool, device=self.device)
        output = self._invoke(
            rows,
            count=n,
            original_count=original_count,
            ids=self._ids(current_ids),
            keep=keep,
            restore_rows=(selected, native_ids, len(restore_ids)),
        )
        self.remove(set(restore_ids))
        self.restore_calls += 1
        return output

    def remove(self, request_ids: set[str]):
        """Cancel or retire selected cache entries, retaining all other requests."""
        remaining = tuple(rid for rid in self.pending if rid not in request_ids)
        if remaining == self.pending:
            return
        indices = torch.tensor([self.pending.index(rid) for rid in remaining], dtype=torch.long, device=self.device)
        count = len(remaining)
        # Attention and MoE slots are kept dense in the same host-known order.
        # Native restore erases selected Attention IDs before these copies.
        for value in vars(self.rows).values():
            value[:count].copy_(value.index_select(0, indices))
        self._native_ids = {rid: self._native_ids[rid] for rid in remaining}
        self.moe_ids.fill_(-1)
        self.attn_ids.fill_(-1)
        if remaining:
            ids = torch.tensor(list(self._native_ids.values()), dtype=torch.int32, device=self.device)
            self.moe_ids[:count].copy_(ids)
            self.attn_ids[0, :count].copy_(ids)
        self.moe_count.fill_(count)
        self.attn_count.fill_(count)
        self.pending = remaining