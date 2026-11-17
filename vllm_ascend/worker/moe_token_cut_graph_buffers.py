# SPDX-License-Identifier: Apache-2.0
"""Serial native graph buffers ported from the modified v0.13 decode path.

Preserve its original/running/history counters, request labels, layer-local
discard buckets, Tensor cut/refill switches and in-place capture cleanup.
This is an isolated operator-graph building block, NOT a full-model graph
backend. In particular, it does not replace scheduler checkpoint ownership.
"""

import torch

from vllm_ascend.ops.fused_moe.moe_token_cut import MAX_MOE_TOKEN_CUT_BATCH_SIZE, MoETokenCutConfig
from vllm_ascend.ops.fused_moe.moe_token_cut_native import (
    MOE_TOKEN_CUT_SCHEMA,
    REBATCH_AND_UPDATE_SCHEMA,
    invoke_native_op,
)
from vllm_ascend.worker.moe_token_cut_native_rebatch import (
    INT32_MAX,
    REBATCH_CAPACITY,
    NativeRebatchRows,
    NativeRebatchStore,
    NativeRestoreContext,
)


class MoETokenCutGraphBuffers:
    """One layer's fixed-address buffers, shared by capture and every replay.

    Unlike the eager adapter, binding arguments are assembled once. ``run``
    unconditionally submits restore+cut, then rebatch; runtime controls remain
    device Tensors. Python input preparation and result checks stay OUTSIDE
    capture/replay. Calls must be serialized on the owning stream.

    Cache capacity (96) follows the supplied rebatch ABI; live rows are still
    limited to 64. Do not copy v0.13's batch-dependent allocation size here.
    Output views are borrowed until the next prepare/replay/reset.
    """

    def __init__(self, example: NativeRebatchRows, context: NativeRestoreContext, policy: MoETokenCutConfig):
        NativeRebatchStore._validate(example)
        if example.topk_ids.shape[1] > context.expert_num:
            raise ValueError("graph topk must not exceed expert_num")
        self.context = context
        self.device = example.hidden.device
        # NativeRebatchStore allocates the same per-layer discard buckets used
        # by eager mode. Its Python pending/ID bookkeeping is NOT used here.
        self.cache = NativeRebatchStore(example, restore_context=context)
        self.inputs = NativeRebatchRows(
            **{name: torch.zeros_like(value) for name, value in vars(self.cache.rows).items()}
        )
        self.outputs = NativeRebatchRows(
            **{name: torch.zeros_like(value) for name, value in vars(self.cache.rows).items()}
        )
        scalar = lambda: torch.zeros(1, dtype=torch.int32, device=self.device)
        self.controls = {
            name: scalar()
            for name in (
                "original_bs",
                "running_bs",
                "history_bs",
                "actual_bs_temp",
                "enable_cut",
                "en_pad",
                "local_cut_count_temp",
            )
        }
        self.controls.update(
            req_id_list=torch.full((REBATCH_CAPACITY,), -1, dtype=torch.int32, device=self.device),
            discard_label=torch.zeros(REBATCH_CAPACITY, dtype=torch.int32, device=self.device),
            cut_idx_temp=torch.zeros(REBATCH_CAPACITY, dtype=torch.int8, device=self.device),
            prefix_index_temp=torch.zeros(REBATCH_CAPACITY, dtype=torch.int32, device=self.device),
            active_mask=torch.zeros(REBATCH_CAPACITY, dtype=torch.bool, device=self.device),
        )
        self.req_ids_cpu = torch.empty(
            REBATCH_CAPACITY, dtype=torch.int32, device="cpu", pin_memory=self.device.type == "npu"
        )
        self._local_cut_count = policy.local_cut_count
        self._unused_rebatch_residual = torch.zeros_like(self.inputs.residual)
        c, i, o, cache = self.controls, self.inputs, self.outputs, self.cache
        self.cut_arguments = dict(
            topk_ids=i.topk_ids,
            topk_weights=i.topk_weights,
            hidden=i.hidden,
            cutidx=c["cut_idx_temp"],
            new_topk=o.topk_ids,
            new_topkweight=o.topk_weights,
            new_hidden=o.hidden,
            new_residuals=o.residual,
            actual_bs=c["actual_bs_temp"],
            prefix_index=c["prefix_index_temp"],
            no_cut_rank_num=c["en_pad"],
            active_mask=c["active_mask"],
            local_cut_count=c["local_cut_count_temp"],
            batch_size=c["running_bs"],
            enable_cut=c["enable_cut"],
            residuals=i.residual,
            discard_label=c["discard_label"],
            discard_tokens=cache.rows.hidden,
            discard_moe_req_id=cache.moe_ids,
            req_id_list=c["req_id_list"],
            discard_residuals=cache.rows.residual,
            discard_topk=cache.rows.topk_ids,
            discard_topk_weight=cache.rows.topk_weights,
            moe_cnt=cache.moe_count,
            history_bs=c["history_bs"],
            original_bs=c["original_bs"],
            group_ep=context.group_ep,
            topk=i.topk_ids.shape[1],
            rank_id=context.rank_id,
            # Initial capture value only. MoETokenCutGraphReplay task-updates
            # the integer generation before each replay; Tensor controls alone
            # cannot change a captured integer attribute.
            layer_idx=0,
            rank_num=context.rank_num,
            expert_num=context.expert_num,
            expert_per_rank=context.expert_num // context.rank_num,
            level_limit=policy.level_limit,
            cut_count_threshold=policy.cut_count_threshold,
            small_expert_token_threshold=policy.small_expert_token_threshold,
            send_aiv_num=context.send_aiv_num,
            recv_aiv_num=context.recv_aiv_num,
        )
        self.rebatch_arguments = dict(
            batch_before_dispatch=i.hidden,
            residuals_before_dispatch=i.residual,
            cut_idx=c["cut_idx_temp"],
            prefix_index=c["prefix_index_temp"],
            discard_label=c["discard_label"],
            discard_tokens=cache.rows.hidden,
            discard_moe_req_id_tensor=cache.moe_ids,
            req_id_list=c["req_id_list"],
            discard_residuals=cache.rows.residual,
            running_topk=i.topk_ids,
            discard_topk=cache.rows.topk_ids,
            running_topk_weight=i.topk_weights,
            discard_topk_weight=cache.rows.topk_weights,
            moe_cnt_input=cache.moe_count,
            actual_bs=c["actual_bs_temp"],
            history_bs=c["history_bs"],
            running_bs=c["running_bs"],
            original_bs=c["original_bs"],
            positions=i.positions,
            slot_mapping=i.slots,
            block_table=i.block_tables,
            seq_lens=i.seq_lens,
            discard_positions=cache.rows.positions.unsqueeze(0),
            discard_slot_mapping=cache.rows.slots.unsqueeze(0),
            discard_context_lens=cache.rows.seq_lens.unsqueeze(0),
            discard_block_tables=cache.rows.block_tables.unsqueeze(0),
            discard_attn_cnt=cache.attn_count,
            discard_req_ids=cache.attn_ids,
            is_combine_rebatch=1,
            max_bs=REBATCH_CAPACITY,
            k=i.topk_ids.shape[1],
            layer_idx=0,
            # Supplied kernel never writes this result. Use token-cut residuals.
            out_rebatch_residuals=self._unused_rebatch_residual,
            out_positions=o.positions,
            out_slots=o.slots,
            out_seq_lens=o.seq_lens,
            out_block_table=o.block_tables,
        )
        self.reset_after_capture()

    def tensor_addresses(self):
        """Diagnostic fingerprint; call outside capture, never in a hot path."""
        return {
            (op, name): (value.data_ptr(), tuple(value.shape), value.dtype)
            for op, arguments in (("cut", self.cut_arguments), ("rebatch", self.rebatch_arguments))
            for name, value in arguments.items()
            if isinstance(value, torch.Tensor)
        }

    def _clear_step(self):
        for tensor in (*vars(self.inputs).values(), *vars(self.outputs).values(), *self.controls.values()):
            tensor.zero_()
        self.inputs.slots.fill_(-1)
        self.outputs.slots.fill_(-1)
        self.controls["req_id_list"].fill_(-1)
        self.controls["local_cut_count_temp"].fill_(self._local_cut_count)

    def prepare(self, rows, request_ids, *, allow_cut, allow_restore):
        """Host-side staging before a synchronized capture/replay.

        Cache-count/ID reads below intentionally synchronize outside the graph:
        reject overflowing or duplicate restores before entering native code.
        A future model integration must use scheduler-owned counts/IDs instead.
        Do not call on an in-flight graph or clear caches between real replays.
        """
        NativeRebatchStore._validate(rows)
        n = rows.hidden.shape[0]
        ids = tuple(request_ids)
        if len(ids) != n or len(set(ids)) != n or any(type(rid) is not int or not 0 <= rid < INT32_MAX for rid in ids):
            raise ValueError("graph request IDs must be unique nonnegative int32 IDs aligned with rows")
        if type(allow_cut) is not bool or type(allow_restore) is not bool:
            raise ValueError("graph controls must be booleans")
        for name, value in vars(rows).items():
            target = getattr(self.inputs, name)
            if value.device != target.device or value.shape[1:] != target.shape[1:]:
                raise ValueError("graph shape/device changed; allocate and capture a new buffer set")
        pending = int(self.cache.moe_count.cpu())
        if not 0 <= pending <= REBATCH_CAPACITY:
            raise RuntimeError("graph native cache count is invalid")
        cached_ids = self.cache.moe_ids[:pending].cpu().tolist()
        if set(ids).intersection(cached_ids):
            raise ValueError("running graph request is already parked")
        # Capacity must also allow saving every live row if policy cuts them.
        capacity = MAX_MOE_TOKEN_CUT_BATCH_SIZE if allow_restore else REBATCH_CAPACITY
        if n + pending > capacity:
            raise ValueError("graph rows plus cached rows exceed the native capacity")
        self._clear_step()
        for name, value in vars(rows).items():
            getattr(self.inputs, name)[:n].copy_(value)
        self.req_ids_cpu.fill_(-1)
        if ids:
            self.req_ids_cpu[:n] = torch.tensor(ids, dtype=torch.int32)
        # Same staging rule as v0.13: complete before reusing the host storage.
        self.controls["req_id_list"].copy_(self.req_ids_cpu, non_blocking=False)
        self.controls["discard_label"][:n].fill_(1)
        # Metadata kernel needs a nonzero original-slot extent for all-restored
        # batches. The dummy slot is -1/label=0 and is never a real request.
        self.controls["original_bs"].fill_(max(n, 1))
        self.controls["running_bs"].fill_(n)
        self.controls["actual_bs_temp"].fill_(n)
        self.controls["enable_cut"].fill_(int(allow_cut))
        self.controls["en_pad"].fill_(int(allow_restore))

    def run(self):
        """Capture these two serial calls only; no host-dependent branches."""
        invoke_native_op(MOE_TOKEN_CUT_SCHEMA, self.cut_arguments)
        invoke_native_op(REBATCH_AND_UPDATE_SCHEMA, self.rebatch_arguments)

    def reset_after_capture(self):
        """Clear dummy data in place after synchronization, preserving addresses.

        Only use on isolated warmup/capture state, before admitting real work.
        Unlike the old lazy-capture reset, never apply this to live model caches.
        """
        self._clear_step()
        for value in vars(self.cache.rows).values():
            value.zero_()
        self.cache.moe_ids.fill_(-1)
        self.cache.attn_ids.fill_(-1)
        self.cache.moe_count.zero_()
        self.cache.attn_count.zero_()
