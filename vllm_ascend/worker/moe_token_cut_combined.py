# SPDX-License-Identifier: Apache-2.0
"""One restore/cut invocation, then independent rebatch and dispatch buffers.

The model reserves restored row positions before routing, without copying their
payload. Native restore fills those rows immediately before MC2 dispatch. Host
request bookkeeping still synchronizes cut_idx once before launching rebatch.
"""

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from vllm_ascend.ops.fused_moe.moe_token_cut import (
    MAX_MOE_TOKEN_CUT_BATCH_SIZE,
    MoETokenRestoreInput,
    apply_moe_token_cut,
)
from vllm_ascend.worker.moe_token_cut_collective import build_local_deferred_mask
from vllm_ascend.worker.moe_token_cut_native_rebatch import (
    REBATCH_CAPACITY,
    REBATCH_ROW_BYTES,
    NativeRebatchRows,
    NativeRebatchStore,
    NativeRestoreContext,
    NativeResumeCheckpoint,
    checked_int32,
)


def agree_round(*, requested, cut_requested, compatible, device, process_group):
    # All EP ranks agree both on entering the kernel and on its communication
    # branch. A restore-only round has enable_cut=0 on EVERY participant.
    flags = torch.tensor([not compatible, requested, cut_requested], dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=process_group)
    incompatible, requested, cut = flags.cpu().tolist()
    if incompatible and requested:
        raise RuntimeError("combined token-cut has incompatible inputs on an EP rank; refusing partial restoration")
    return bool(requested), bool(cut)


def stage_restore(state, layer, hidden, residual, positions, resume_ids):
    """Reserve shape only. Cached hidden/residual stay in the native store."""
    if layer in state._native_staged_restores:
        raise RuntimeError("native restore was staged twice at one layer")
    n = len(state.current_request_ids)
    ids = tuple(resume_ids)
    state._native_staged_restores[layer] = (n, ids)
    attention = state._layer_attention_snapshots.setdefault(layer, {})
    restored_positions = []
    for rid in ids:
        cp = state._resume_checkpoints[rid]
        attention[rid] = (cp.slot_mapping, cp.seq_len, cp.block_table, cp.seq_len_cpu)
        restored_positions.append(cp.position)
        # Existing Router hooks exclude these reserved rows and reuse cached
        # routes. No gate/Attention is computed on placeholder hidden states.
        state._restored_request_ids.add(rid)
    state._current_request_ids += ids
    return (
        torch.cat((hidden, hidden.new_zeros((len(ids), hidden.shape[1])))),
        torch.cat((residual, residual.new_zeros((len(ids), residual.shape[1])))),
        torch.cat((positions, torch.stack(restored_positions))),
    )


def row_tensors(rows):
    return tuple(vars(rows).values())


def store_tensors(store):
    return (*row_tensors(store.rows), store.moe_ids, store.attn_ids, store.moe_count, store.attn_count)


@dataclass
class RebatchWork:
    output: NativeRebatchRows
    store: NativeRebatchStore
    rows: NativeRebatchRows
    keep: torch.Tensor
    kept_ids: tuple[str, ...]
    event: object = None
    references: tuple = ()

    def wait(self):
        if self.event is not None:
            stream = torch.npu.current_stream()
            stream.wait_event(self.event)
            # Outputs/cache were allocated/written on the rebatch stream and
            # will be consumed/repacked on the caller stream after this wait.
            for tensor in (*row_tensors(self.output), *store_tensors(self.store)):
                tensor.record_stream(stream)
            self.event = None


def launch_rebatch(state, store, rows, keep, request_ids, dropped):
    def run():
        if dropped:
            return store.park(request_ids, rows, keep, dropped)
        return store._invoke(
            rows, count=len(request_ids), original_count=len(request_ids), ids=store._ids(request_ids), keep=keep
        )

    refs = (*row_tensors(rows), keep, *store_tensors(store))
    done = None
    if state.config.rebatch_overlap:
        if rows.hidden.device.type != "npu":
            raise RuntimeError("rebatch_overlap requires an NPU stream runtime")
        main = torch.npu.current_stream()
        if state._native_rebatch_stream is None:
            state._native_rebatch_stream = torch.npu.Stream(device=rows.hidden.device)
        side = state._native_rebatch_stream
        ready = main.record_event()
        side.wait_event(ready)
        for tensor in refs:
            tensor.record_stream(side)
        try:
            with torch.npu.stream(side):
                output = run()
                done = side.record_event()
        except Exception:
            # Any partially submitted cache writes must precede cleanup/reuse.
            main.wait_event(side.record_event())
            raise
        refs += (ready,)
        state.native_overlap_launches += 1
    else:
        output = run()
    return RebatchWork(output, store, rows, keep, tuple(rid for rid in request_ids if rid not in dropped), done, refs)


def run_combined(
    state, dispatch, *, group_ep, rank_id, rank_num, expert_num, process_group, apply_op=apply_moe_token_cut
):
    layer = state._pending_layer_idx
    staged = state._native_staged_restores.get(layer)
    cut = state.active and state.allow_cut_this_forward
    requested = cut or staged is not None
    if not state.config.enabled:
        return None
    n = len(state.current_request_ids) if state.cross_layer_active else 0
    compatible = (
        dispatch.routing.pertoken_scale is None
        and dispatch.hidden_states.shape[0] <= MAX_MOE_TOKEN_CUT_BATCH_SIZE
        and dispatch.hidden_states.dtype == torch.bfloat16
        and dispatch.topk_weights.dtype == torch.bfloat16
        and dispatch.hidden_states.shape[1] * dispatch.hidden_states.element_size() <= REBATCH_ROW_BYTES
        and rank_num > 0
        and expert_num % rank_num == 0
        and n <= dispatch.hidden_states.shape[0]
    )
    store = state._native_rebatch_stores.get(layer)
    # Metadata kernel allocates first-fit before erasing restored entries.
    # This adapter retires selected rows before saving, with a conservative
    # upper bound checked on every rank BEFORE the collective kernel.
    pending = len(store.pending) if store is not None else 0
    restoring = len(staged[1]) if staged else 0
    compatible = compatible and pending - restoring + n <= REBATCH_CAPACITY
    if state.data_parallel_size > 1:
        if process_group is None:
            raise RuntimeError("combined DP token-cut requires the MC2 process group")
        enter, communicate = agree_round(
            requested=requested,
            cut_requested=cut,
            compatible=compatible,
            device=dispatch.hidden_states.device,
            process_group=process_group,
        )
    else:
        if requested and not compatible:
            raise RuntimeError("combined token-cut has unsupported rows/dtypes/cache capacity")
        enter, communicate = requested, cut
    if not enter:
        return None

    if state.cross_layer_active:
        # Native combined has a rank-local contiguous current-request prefix.
        # In particular, reserved restore rows must not remain masked out by
        # an Attention-era mask computed before shape reservation.
        mask = torch.arange(dispatch.hidden_states.shape[0], device=dispatch.hidden_states.device) < n
        dispatch = replace(dispatch, routing=replace(dispatch.routing, mc2_mask=mask))

    restore = None
    if staged:
        fresh, ids = staged
        indices = torch.tensor([store.pending.index(rid) for rid in ids], device=store.device)
        saved = store.rows.select(indices)
        restore = MoETokenRestoreInput(
            fresh,
            saved.hidden,
            saved.residual,
            saved.topk_ids,
            saved.topk_weights,
            store.moe_ids.index_select(0, indices),
        )
    inputs = state._layer_inputs.get(layer)
    residual = (
        None
        if inputs is None
        else state._align_layer_tensor(
            inputs[1], target_rows=dispatch.hidden_states.shape[0], rank_id=rank_id, rank_num=rank_num
        )
    )
    generation = state._native_invocation_id
    # Restore-only rounds do not touch HCCL mailboxes and do not consume their
    # generation; communicating rounds advance it on every EP participant.
    state._native_invocation_id += int(communicate)
    output = apply_op(
        dispatch,
        group_ep=group_ep,
        rank_id=rank_id,
        rank_num=rank_num,
        expert_num=expert_num,
        expert_per_rank=expert_num // rank_num,
        config=state.policy,
        layer_idx=generation,
        residuals=residual,
        allow_cut=cut,
        restore=restore,
        enable_cut=communicate,
    )
    state.native_combined_calls += 1
    if not requested or not n:
        return None
    if output.inputs is None or inputs is None:
        raise RuntimeError("combined native call did not expose restored input scratch")
    # One host read at the ownership boundary. No per-row .item() loop and no
    # synchronization after the rebatch side-stream launch before dispatch.
    keep = output.cut_idx[:n].bool()
    live_flags = keep.cpu().tolist()
    ids = state.current_request_ids
    dropped = tuple(rid for rid, live in zip(ids, live_flags) if not live)
    state.capture_layer_routing(layer, dispatch, rank_id=rank_id, rank_num=rank_num)
    rows = state._native_rows(
        layer, output.inputs.hidden[:n], output.inputs.residual[:n], state._native_layer_positions[layer]
    )
    rows.topk_ids, rows.topk_weights = output.inputs.topk_ids[:n], output.inputs.topk_weights[:n]
    for name in ("topk_ids", "slots", "seq_lens", "block_tables"):
        setattr(rows, name, checked_int32(getattr(rows, name)))
    context = NativeRestoreContext(
        group_ep, rank_id, rank_num, expert_num, state.policy.send_aiv_num, state.policy.recv_aiv_num
    )
    if store is None:
        store = NativeRebatchStore(rows, restore_context=context)
        state._native_rebatch_stores[layer] = store
    elif store.restore_context != context:
        raise RuntimeError("native rebatch EP context changed")
    state._native_restore_contexts[layer] = context
    if staged:
        # Snapshot metadata before repacking invalidates cache row views.
        from vllm_ascend.worker.moe_token_cut_state import MoETokenResumeCheckpoint

        for i, rid in enumerate(ids):
            if rid not in staged[1]:
                continue
            cp = state._resume_checkpoints[rid]
            state._resume_checkpoints[rid] = MoETokenResumeCheckpoint(
                layer,
                rows.hidden[i],
                rows.residual[i],
                rows.positions[i],
                rows.topk_ids[i],
                rows.topk_weights[i],
                rows.slots[i],
                rows.seq_lens[i],
                cp.seq_len_cpu,
                rows.block_tables[i],
            )
        store.remove(set(staged[1]))
        store.token_restore_calls += 1
        store.restore_calls += 1
    # Normalize restored metadata to this invocation's complete row layout.
    # The fused rebatch kernel now saves/compacts it like current metadata;
    # no second native restore or rebatch dispatch branch is used.
    work = launch_rebatch(state, store, rows, keep, ids, dropped)
    state._native_combined_work[layer] = work
    for rid in dropped:
        length = state._layer_attention_snapshots[layer][rid][3]
        state._resume_checkpoints[rid] = NativeResumeCheckpoint(layer, rid, store, length)
        state._restored_request_ids.discard(rid)
    mask = build_local_deferred_mask(state._request_ids, ids, output.cut_idx[:n])
    if state._global_deferred_mask is None:
        state._global_deferred_mask = mask
    else:
        state._global_deferred_mask.logical_or_(mask)
    state.last_output = output
    state.stats.moe_layers += 1
    state.stats.last_skip_reason = None
    state._native_staged_restores.pop(layer, None)
    return output


def finish_combined(state, layer, moe_output):
    work = state._native_combined_work.pop(layer)
    work.wait()
    kept = torch.nonzero(work.keep, as_tuple=False).flatten()
    state.publish_native_attention_result(layer, work.kept_ids, work.output)
    state._current_request_ids = work.kept_ids
    state._layer_attention_snapshots.pop(layer, None)
    state._layer_routing_snapshots.pop(layer, None)
    return (
        moe_output.index_select(0, kept),
        work.rows.residual.index_select(0, kept),
        work.output.positions[: len(work.kept_ids)],
    )
