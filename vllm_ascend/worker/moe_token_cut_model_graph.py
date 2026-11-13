# SPDX-License-Identifier: Apache-2.0
"""Experimental fixed-capacity Qwen decode graph, including native cut/refill.

ACLGraphWrapper owns capture/replay; this adapter supplies buffers and updates.
Host scheduling and cache selection surround the graph, never run within it.
Unlike the operator smoke, this captures embedding, every Attention, routing,
cut/rebatch, MC2 dispatch/GMM/combine and the final normalization together.
"""

from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from itertools import islice

import torch
import torch.distributed as dist
import torch_npu

from vllm_ascend.ops.fused_moe.moe_token_cut import MoETokenCutOutput
from vllm_ascend.ops.fused_moe.moe_token_cut_native import (
    MOE_TOKEN_CUT_SCHEMA,
    REBATCH_AND_UPDATE_SCHEMA,
    invoke_native_op,
)
from vllm_ascend.worker.moe_token_cut_graph_buffers import MoETokenCutGraphBuffers
from vllm_ascend.worker.moe_token_cut_native_rebatch import (
    INT32_MAX,
    REBATCH_CAPACITY,
    NativeRebatchRows,
    NativeRebatchStore,
    NativeRestoreContext,
    NativeResumeCheckpoint,
    checked_int32,
)


def compact_request_ids(ids, labels, capacity):
    """Fixed-shape device compaction; no nonzero(), host count or dynamic view."""
    live = labels.eq(1) & ids.ge(0)
    indices = (live.cumsum(0) - 1).clamp(0, capacity - 1).long()
    values = torch.where(live, ids + 1, 0)
    return torch.zeros(capacity, dtype=ids.dtype, device=ids.device).scatter_add_(0, indices, values) - 1


class CapturedReadyWait:
    """Replay only an external-event wait/reset, never any model operations.

    Value-based ExternalEvent rejects a second host wait unless host record
    was called again. A record node replayed by the model graph does not
    re-arm that host flag. Capture the consumer once as well, so both sides
    of the handshake repeat on device without issuing another host wait.
    """

    def __init__(self, ready, stream):
        self.ready = ready  # Keep the event storage alive with the graph.
        self.stream = stream
        self.graph = torch.npu.NPUGraph()
        # Startup only, after the dummy model has completed and all gates
        # are released. Seed the wait so capture cannot block the update stream.
        with torch.npu.stream(stream):
            ready.record(stream)
        with torch.npu.graph(self.graph, stream=stream):
            ready.wait(stream)
            ready.reset(stream)
        stream.synchronize()
        # Drain a known signal through the captured consumer. This also leaves
        # no dummy-capture readiness for the first real replay to consume,
        # whether this runtime executes nodes while recording or only on replay.
        with torch.npu.stream(stream):
            ready.record(stream)
            self.graph.replay()
        stream.synchronize()

    def wait(self):
        # No record here: only the current model layer may signal readiness.
        with torch.npu.stream(self.stream):
            self.graph.replay()


def drain_startup_gates(events, stream):
    """Leave external gates unsignaled after all capture work has completed.

    Value-based ExternalEvent.reset() is a no-op. Capture-time gate records
    may survive capture without being consumed by model wait nodes. Overwrite
    each signal with a known record, then consume it through a real wait.
    A new host record also rearms the single-host-wait guard. Never call this
    with a live model replay: it would release that model's gate prematurely.
    """
    with torch.npu.stream(stream):
        for event in events:
            event.record(stream)
            event.wait(stream)
            event.reset(stream)
    stream.synchronize()


class ModelGraphFrame:
    """Fixed buffers and captured layer operations, independent of graph launch."""

    def __init__(self, state, capacity, device, block_columns):
        self.state, self.capacity, self.device = state, capacity, device
        self.input_ids = torch.zeros(capacity, dtype=torch.long, device=device)
        self.positions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.ids = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.slots = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.lengths = torch.zeros(capacity, dtype=torch.int32, device=device)
        self.blocks = torch.zeros((capacity, block_columns), dtype=torch.int32, device=device)
        self.row_numbers = torch.arange(capacity, device=device)
        self.layers = {}
        self.selected = {}
        self.request_numbers = {}
        self.next_request_number = 0
        self.handles = {}
        self.events = {}
        self.capturing = False
        self.current_layer = None
        self.current_rows = None
        self.current_ids = self.ids
        self.current_count = self.count
        self.final_ids = self.ids
        self.final_count = self.count
        self.generation = 0
        self.allow_cut = False
        self.communicate = False
        self.metadata_templates = {}
        self.attention_tasks = {}
        self.attention_workspaces = {}
        self.last_cuts = ()
        self.last_restored = ()

    def prepare_inputs(self, input_ids, positions, metadata):
        """Host boundary. Exclude parked requests from Attention on entry."""
        state = self.state
        ids = state._request_ids
        if len(ids) > self.capacity or len(set(ids)) != len(ids):
            raise RuntimeError("model graph requires unique scheduled requests within capacity")
        for rid in ids:
            if rid not in self.request_numbers:
                if self.next_request_number >= INT32_MAX - 1:
                    raise RuntimeError("model graph request ID space exhausted")
                self.request_numbers[rid] = self.next_request_number
                self.next_request_number += 1
        fresh = [i for i, rid in enumerate(ids) if rid not in state._resume_checkpoints]
        n = len(fresh)
        for value in (self.input_ids, self.positions, self.lengths, self.blocks):
            value.zero_()
        self.ids.fill_(-1)
        self.slots.fill_(-1)
        self.count.fill_(n)
        if n:
            if metadata is None:
                raise RuntimeError("scheduled decode graph has no Attention metadata")
            indices = torch.tensor(fresh, dtype=torch.long, device=self.device)
            self.input_ids[:n].copy_(input_ids.index_select(0, indices))
            self.positions[:n].copy_(positions.index_select(0, indices))
            self.ids[:n].copy_(
                torch.tensor([self.request_numbers[ids[i]] for i in fresh], dtype=torch.int32, device=self.device)
            )
            for target, source in (
                (self.slots, metadata.slot_mapping),
                (self.lengths, metadata.seq_lens),
                (self.blocks, metadata.block_tables),
            ):
                selected = source.index_select(0, indices.to(source.device))
                selected = checked_int32(selected).to(self.device)
                if selected.shape[1:] != target.shape[1:]:
                    raise RuntimeError("model graph KV layout changed")
                target[:n].copy_(selected)

    def prepare_caches(self, *, communicate, allow_cut, allow_restore):
        """Stage only this step's eligible cache subset; durable stores survive."""
        self.communicate, self.allow_cut = communicate, allow_cut
        scheduled = set(self.state._request_ids)
        for layer, buffers in self.layers.items():
            buffers.reset_after_capture()  # Scratch only, never the durable store.
            store = self.state._native_rebatch_stores.get(layer)
            selected = tuple(rid for rid in (store.pending if store else ()) if rid in scheduled and allow_restore)
            self.selected[layer] = selected
            if selected:
                indices = torch.tensor(
                    [store.pending.index(rid) for rid in selected], dtype=torch.long, device=self.device
                )
                rows = store.rows.select(indices)
                for name, value in vars(rows).items():
                    getattr(buffers.cache.rows, name)[: len(selected)].copy_(value)
                numbers = torch.tensor(
                    [self.request_numbers[rid] for rid in selected], dtype=torch.int32, device=self.device
                )
                buffers.cache.moe_ids[: len(selected)].copy_(numbers)
                buffers.cache.attn_ids[0, : len(selected)].copy_(numbers)
                buffers.cache.moe_count.fill_(len(selected))
                buffers.cache.attn_count.fill_(len(selected))
            buffers.controls["enable_cut"].fill_(int(communicate))
            buffers.controls["en_pad"].fill_(int(allow_restore))

    def dispatch(self, dispatch, *, group_ep, rank_id, rank_num, expert_num):
        """Captured body: one restore+cut, one metadata rebatch per sparse layer."""
        layer = self.current_layer
        rows = self.current_rows
        if rows is None or layer is None:
            raise RuntimeError("model graph dispatch has no layer inputs")
        if dispatch.routing.pertoken_scale is not None or dispatch.topk_weights.dtype != torch.bfloat16:
            raise RuntimeError("model graph requires BF16 routes and activations before MC2 quantization")
        context = NativeRestoreContext(
            group_ep, rank_id, rank_num, expert_num, self.state.policy.send_aiv_num, self.state.policy.recv_aiv_num
        )
        if layer not in self.layers:
            if self.capturing:
                raise RuntimeError("model graph layer was not warmed up")
            example = replace(rows, topk_ids=dispatch.topk_ids, topk_weights=dispatch.topk_weights)
            self.layers[layer] = MoETokenCutGraphBuffers(example, context, self.state.policy)
        b = self.layers[layer]
        if b.context != context:
            raise RuntimeError("model graph EP topology changed")
        c = b.controls
        # Clear padding on every execution, including replay. Outputs of the
        # supplied kernels need not initialize their inactive suffix.
        for value in vars(b.inputs).values():
            value.zero_()
        for value in vars(b.outputs).values():
            value.zero_()
        b.inputs.slots.fill_(-1)
        for name, value in vars(rows).items():
            if name not in ("topk_ids", "topk_weights"):
                getattr(b.inputs, name)[: self.capacity].copy_(value)
        b.inputs.topk_ids[: self.capacity].copy_(dispatch.topk_ids)
        b.inputs.topk_weights[: self.capacity].copy_(dispatch.topk_weights)
        c["req_id_list"].fill_(-1)
        c["req_id_list"][: self.capacity].copy_(self.current_ids)
        c["discard_label"].zero_()
        c["discard_label"][: self.capacity].copy_(self.row_numbers < self.current_count)
        c["running_bs"].copy_(self.current_count)
        c["original_bs"].copy_(self.current_count.clamp_min(1))
        c["history_bs"].zero_()
        c["local_cut_count_temp"].fill_(self.state.policy.local_cut_count)
        if self.capturing:
            stream = torch.npu.current_stream()
            event = torch.npu.ExternalEvent()
            self.events[layer] = event
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            invoke_native_op(MOE_TOKEN_CUT_SCHEMA, self.cut_arguments(layer))
            self.handles[layer] = torch.npu.graph_task_group_end(stream)
        else:
            invoke_native_op(MOE_TOKEN_CUT_SCHEMA, self.cut_arguments(layer))
        invoke_native_op(REBATCH_AND_UPDATE_SCHEMA, b.rebatch_arguments)
        self.current_ids = compact_request_ids(c["req_id_list"], c["discard_label"], self.capacity)
        self.current_count = c["actual_bs_temp"]
        mask = self.row_numbers < self.current_count
        output = replace(
            dispatch,
            hidden_states=b.outputs.hidden[: self.capacity],
            topk_ids=b.outputs.topk_ids[: self.capacity],
            topk_weights=b.outputs.topk_weights[: self.capacity],
            routing=replace(dispatch.routing, mc2_mask=mask),
        )
        return MoETokenCutOutput(output, c["cut_idx_temp"], c["actual_bs_temp"], c["prefix_index_temp"], c["en_pad"])

    def cut_arguments(self, layer):
        # Rank-local no-delete is an integer threshold. All ranks still join
        # the communicating branch selected at the graph-entry agreement.
        return dict(
            self.layers[layer].cut_arguments,
            layer_idx=self.generation + list(self.layers).index(layer),
            cut_count_threshold=self.state.policy.cut_count_threshold if self.allow_cut else REBATCH_CAPACITY,
        )

    def forward(self, model, context, input_ids, positions):
        """Actual Qwen body captured as one graph. No per-layer host readback."""
        self.current_ids, self.current_count = self.ids, self.count
        hidden = model.embed_input_ids(input_ids)
        residual = None
        slots, lengths, blocks = self.slots, self.lengths, self.blocks
        original_metadata = context.attn_metadata
        try:
            for layer_idx, layer in enumerate(
                islice(model.layers, model.start_layer, model.end_layer), start=model.start_layer
            ):
                self.current_layer = layer_idx
                mask = self.row_numbers < self.current_count
                meta = copy(self.metadata_templates[layer.self_attn.attn.layer_name])
                meta.slot_mapping = torch.where(mask, slots, -1)
                meta.seq_lens = torch.where(mask, lengths, 0)
                meta.block_tables = torch.where(mask[:, None], blocks, 0)
                meta.moe_token_cut_model_graph = True
                context.attn_metadata = {layer.self_attn.attn.layer_name: meta}
                positions = torch.where(mask, positions, 0)
                if residual is None:
                    residual = hidden
                    hidden = layer.input_layernorm(hidden)
                else:
                    hidden, residual = layer.input_layernorm(hidden, residual)
                hidden = layer.self_attn(positions=positions, hidden_states=hidden)
                hidden, residual = layer.post_attention_layernorm(hidden, residual)
                hidden = torch.where(mask[:, None], hidden, 0)
                residual = torch.where(mask[:, None], residual, 0)
                self.current_rows = NativeRebatchRows(
                    hidden, residual, None, None, positions, meta.slot_mapping, meta.seq_lens, blocks
                )
                context.mc2_mask = mask
                hidden = layer.mlp(hidden)
                if hasattr(layer.mlp, "experts"):
                    if layer_idx not in self.layers:
                        raise RuntimeError("model graph MoE bypassed the MC2 token-cut hook")
                    rows = self.layers[layer_idx].outputs
                    residual = rows.residual[: self.capacity]
                    positions = rows.positions[: self.capacity]
                    slots, lengths, blocks = (
                        rows.slots[: self.capacity],
                        rows.seq_lens[: self.capacity],
                        rows.block_tables[: self.capacity],
                    )
            hidden, _ = model.norm(hidden, residual)
            self.final_ids, self.final_count = self.current_ids, self.current_count
            return torch.where((self.row_numbers < self.final_count)[:, None], hidden, 0)
        finally:
            context.attn_metadata = original_metadata

    def attention_workspace(self, arguments, max_seq_len):
        """Reuse scratch for identical FIA signatures on this graph's stream.

        Each layer still owns its Attention inputs/outputs and update handle.
        Only workspace is shared: FIA executions are ordered on the same
        stream, and the next replay waits for the preceding replay to finish.
        Different layouts retain separate allocations; capture never resizes.
        """
        signature = []
        for name, value in sorted(arguments.items()):
            if isinstance(value, torch.Tensor):
                value = (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
            elif isinstance(value, list):
                value = tuple(value)
            signature.append((name, value))
        key = (max_seq_len, tuple(signature))
        if key not in self.attention_workspaces:
            if self.capturing:
                raise RuntimeError("model graph FIA workspace signature was not warmed up")
            try:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    **arguments, actual_seq_lengths_kv=[max_seq_len] * self.capacity
                )
            except torch.OutOfMemoryError as exc:
                raise torch.OutOfMemoryError(
                    f"token-cut FIA workspace allocation failed at layer={self.current_layer}, "
                    f"capacity={self.capacity}, max_seq_len={max_seq_len}, "
                    f"cached_workspace_signatures={len(self.attention_workspaces)}. "
                    "Leave more memory for graph buffers by lowering gpu_memory_utilization "
                    "(or an explicit kv_cache_memory_bytes budget). "
                    f"Original error: {exc}"
                ) from exc
            self.attention_workspaces[key] = workspace
        return self.attention_workspaces[key]

    def attention(self, impl, query, metadata, output):
        """Capture FIA with a host-refreshed length task, as in v0.13.

        The public FIA lengths are host arrays. A device Tensor supplied to
        that argument is not a device-side dynamic-length contract. During
        replay a ready/gate pair pauses THIS layer until its rebatch lengths
        have been copied and its task has been refreshed on the update stream.
        """
        layer = self.current_layer
        blocks, block_size, _, _ = impl.key_cache.shape
        arguments = dict(
            query=query,
            key=impl.key_cache.view(blocks, block_size, -1),
            value=impl.value_cache.view(blocks, block_size, -1),
            block_table=metadata.block_tables,
            block_size=block_size,
            input_layout="TND",
            actual_seq_lengths=list(range(1, self.capacity + 1)),
            num_heads=impl.num_heads,
            num_key_value_heads=impl.num_kv_heads,
            scale=impl.scale,
            sparse_mode=0,
        )
        arguments["workspace"] = self.attention_workspace(arguments, impl.vllm_config.model_config.max_model_len)
        arguments["out"] = [output, torch.empty(1, dtype=query.dtype, device=query.device)]
        if not self.capturing:
            lengths = metadata.seq_lens.to(device="cpu").tolist()
            arguments["actual_seq_lengths_kv"] = [max(length, 1) for length in lengths]
            self.attention_tasks[layer] = dict(arguments=arguments)
            torch_npu.npu_fused_infer_attention_score.out(**arguments)
            return output
        # Dummy capture uses warmup lengths. Real replay updates after ready.
        arguments["actual_seq_lengths_kv"] = self.attention_tasks[layer]["arguments"]["actual_seq_lengths_kv"]
        ready, gate = torch.npu.ExternalEvent(), torch.npu.ExternalEvent()
        stream = torch.npu.current_stream()
        ready.record(stream)
        gate.wait(stream)
        gate.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(**arguments)
        handle = torch.npu.graph_task_group_end(stream)
        self.attention_tasks[layer] = dict(
            arguments=arguments,
            ready=ready,
            gate=gate,
            handle=handle,
            device_lengths=metadata.seq_lens,
            host_lengths=torch.empty(self.capacity, dtype=torch.int32, device="cpu", pin_memory=True),
        )
        return output

    def finish(self, output, output_rows):
        """Read ownership after graph completion, then hand off to Scheduler."""
        count = int(self.final_count.cpu())
        if not 0 <= count <= self.capacity:
            raise RuntimeError("model graph returned invalid live count")
        reverse = {number: rid for rid, number in self.request_numbers.items()}
        completed_numbers = self.final_ids[:count].cpu().tolist()
        completed = tuple(reverse[number] for number in completed_numbers)
        scheduled = set(self.state._request_ids)
        if len(set(completed)) != count or not set(completed) <= scheduled:
            raise RuntimeError("model graph returned duplicate or unscheduled rows")
        new_caches = {}
        for layer, b in self.layers.items():
            pending = int(b.cache.moe_count.cpu())
            if not 0 <= pending <= self.capacity:
                raise RuntimeError("model graph returned invalid parked count")
            numbers = b.cache.moe_ids[:pending].cpu().tolist()
            attn_ids = b.cache.attn_ids[0].cpu().tolist()
            requests = tuple(reverse[number] for number in numbers)
            if len(set(requests)) != pending or not set(requests) <= scheduled:
                raise RuntimeError("model graph parked duplicate or unscheduled requests")
            metadata_rows = torch.tensor(
                [attn_ids.index(number) for number in numbers], dtype=torch.long, device=self.device
            )
            saved = NativeRebatchRows(
                **{
                    name: value[:pending].clone()
                    if name in ("hidden", "residual", "topk_ids", "topk_weights")
                    else value.index_select(0, metadata_rows)
                    for name, value in vars(b.cache.rows).items()
                }
            )
            new_caches[layer] = requests, saved
        parked = [rid for requests, _ in new_caches.values() for rid in requests]
        if len(set(parked)) != len(parked) or set(parked).intersection(completed):
            raise RuntimeError("model graph ownership is not unique across layers")
        selected = {rid for ids in self.selected.values() for rid in ids}
        expected = scheduled - (set(self.state._resume_checkpoints) - selected)
        if set(parked) | set(completed) != expected:
            raise RuntimeError("model graph lost a scheduled request")
        # Validate every layer before committing host ownership.
        for layer, (requests, _) in new_caches.items():
            store = self.state._native_rebatch_stores.get(layer)
            retained = len(store.pending) - len(self.selected.get(layer, ())) if store else 0
            if retained + len(requests) > REBATCH_CAPACITY:
                raise RuntimeError("model graph durable cache overflow")
        for rid in selected:
            self.state._resume_checkpoints.pop(rid)
        for layer, (requests, rows) in new_caches.items():
            b = self.layers[layer]
            store = self.state._native_rebatch_stores.get(layer)
            if store is None:
                example = NativeRebatchRows(**{name: value[: self.capacity] for name, value in vars(b.inputs).items()})
                store = NativeRebatchStore(example, restore_context=b.context)
                self.state._native_rebatch_stores[layer] = store
            store.remove(set(self.selected.get(layer, ())))
            offset = len(store.pending)
            for name, value in vars(rows).items():
                getattr(store.rows, name)[offset : offset + len(requests)].copy_(value)
            store.pending += requests
            store._native_ids = {rid: self.request_numbers[rid] for rid in store.pending}
            numbers = torch.tensor(list(store._native_ids.values()), dtype=torch.int32, device=self.device)
            store.moe_ids.fill_(-1)
            store.attn_ids.fill_(-1)
            store.moe_ids[: len(store.pending)].copy_(numbers)
            store.attn_ids[0, : len(store.pending)].copy_(numbers)
            store.moe_count.fill_(len(store.pending))
            store.attn_count.fill_(len(store.pending))
            lengths = rows.seq_lens.cpu().tolist()
            for rid, length in zip(requests, lengths):
                self.state._resume_checkpoints[rid] = NativeResumeCheckpoint(layer, rid, store, length)
            store.save_calls += bool(requests)
            store.restore_calls += bool(self.selected.get(layer))
            store.token_restore_calls += bool(self.selected.get(layer))
        indices = torch.tensor(
            [self.state._request_ids.index(rid) for rid in completed], dtype=torch.long, device=self.device
        )
        result = output.new_zeros((output_rows, output.shape[1]))
        result.index_copy_(0, indices, output[:count])
        self.state._global_deferred_mask = torch.tensor(
            [rid not in completed for rid in self.state._request_ids], dtype=torch.bool, device=self.device
        )
        self.state._current_request_ids = completed
        self.last_cuts = tuple(parked)
        self.last_restored = tuple(sorted(selected))
        self.state.native_combined_calls += len(self.layers)
        self.state.stats.moe_layers += len(self.layers)
        # Keep IDs for live caches only. IDs remain monotonic, never recycled.
        self.request_numbers = {
            rid: number for rid, number in self.request_numbers.items() if rid in self.state._resume_checkpoints
        }
        return result


class MoETokenCutModelGraph:
    """Runner-owned graph lifecycle, with one EP agreement per model forward."""

    def __init__(self, state, capacity):
        self.state, self.capacity = state, capacity
        self.frame = None
        self.startup_capture = False
        self.update_stream = None
        self.failed = False
        self.captures = 0
        self.replays = 0
        self.observe_step = None

    @staticmethod
    def validate_runtime():
        required = (
            "NPUGraph",
            "graph",
            "ExternalEvent",
            "graph_task_group_begin",
            "graph_task_group_end",
            "graph_task_update_begin",
            "graph_task_update_end",
        )
        missing = [name for name in required if not hasattr(torch.npu, name)]
        if not hasattr(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace"):
            missing.append("_npu_fused_infer_attention_score_get_max_workspace")
        if not hasattr(getattr(torch_npu, "npu_fused_infer_attention_score", None), "out"):
            missing.append("npu_fused_infer_attention_score.out")
        if missing:
            raise RuntimeError("token-cut model graph runtime lacks: " + ", ".join(missing))

    @contextmanager
    def activate(self, context):
        previous = {
            name: getattr(context, name, None)
            for name in (
                "num_tokens",
                "padded_num_tokens",
                "mc2_mask",
                "capturing",
                "cudagraph_runtime_mode",
                "batch_descriptor",
                "moe_token_cut_graph_manager",
                "moe_layer_index",
            )
        }
        self.state.model_graph_frame = self.frame
        context.num_tokens = context.padded_num_tokens = self.capacity
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor

        context.capturing = False
        context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        context.batch_descriptor = BatchDescriptor(self.capacity, num_reqs=self.capacity, uniform=True)
        context.moe_token_cut_graph_manager = self
        try:
            yield
        finally:
            self.state.model_graph_frame = None
            for name, value in previous.items():
                setattr(context, name, value)

    def update_graph_params(self, phase):
        """Called through the runner's update_full_graph_params entry."""
        if phase == "before_replay":
            self.update_tasks()
        elif phase == "capture":
            self.update_tasks()
            self.update_attention(capture=True)
        elif phase == "after_launch":
            self.update_attention()
        else:
            raise ValueError(f"unknown token-cut graph update phase: {phase}")

    def update_tasks(self):
        frame = self.frame
        if frame.generation + len(frame.layers) >= INT32_MAX:
            raise RuntimeError("model graph communication generation exhausted")
        with torch.npu.stream(self.update_stream):
            for layer, handle in frame.handles.items():
                torch.npu.graph_task_update_begin(self.update_stream, handle)
                try:
                    invoke_native_op(MOE_TOKEN_CUT_SCHEMA, frame.cut_arguments(layer))
                finally:
                    torch.npu.graph_task_update_end(self.update_stream)
                frame.events[layer].record(self.update_stream)

    def update_attention(self, *, capture=False):
        """Service graph gates in layer order; never synchronize the blocked main stream."""
        for task in self.frame.attention_tasks.values():
            with torch.npu.stream(self.update_stream):
                arguments = dict(task["arguments"])
                if not capture:
                    task["ready_wait"].wait()
                    task["host_lengths"].copy_(task["device_lengths"], non_blocking=True)
                    self.update_stream.synchronize()
                    lengths = task["host_lengths"].tolist()
                    # Inactive rows read a harmless one-token dummy block; KV
                    # writes use slot=-1 and their results are masked in Qwen.
                    arguments["actual_seq_lengths_kv"] = [max(length, 1) for length in lengths]
                torch.npu.graph_task_update_begin(self.update_stream, task["handle"])
                try:
                    torch_npu.npu_fused_infer_attention_score.out(**arguments)
                finally:
                    torch.npu.graph_task_update_end(self.update_stream)
                # Keep the exact submitted host arrays alive with this task.
                # Capture-time arguments stay separate so diagnostics can
                # distinguish dummy lengths from this replay's update.
                task["last_update_arguments"] = arguments
                task["last_update_was_capture"] = capture
                task["runtime_update_count"] = task.get("runtime_update_count", 0) + int(not capture)
                task["gate"].record(self.update_stream)

    def run(self, runner, context, model_inputs, output_rows):
        if self.failed:
            raise RuntimeError("model graph failed previously; worker restart required")
        try:
            return self._run(runner, context, model_inputs, output_rows)
        except Exception:
            self.failed = True
            raise

    def _run(self, runner, context, model_inputs, output_rows):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
        from vllm_ascend.distributed.parallel_state import get_mc2_group

        # Profile/dummy warmup must not enter runtime collectives or allocate
        # graphs before KV cache initialization. Real idle DP runs do enter.
        if context.in_profile_run:
            return None
        state = self.state
        if self.startup_capture:
            # All ranks enter through capture_model after KV-cache creation.
            # No scheduler requests, runtime agreement or cache commit here.
            if state._request_ids or state._resume_checkpoints:
                raise RuntimeError("token-cut startup capture requires an empty request state")
            communicate = False
        else:
            local_eligible = (state.active or not state._request_ids) and (
                getattr(context.moe_comm_type, "name", None) == "MC2"
            )
            flags = torch.tensor(
                [not local_eligible, state.allow_cut_this_forward, state.active],
                dtype=torch.int32,
                device=runner.device,
            )
            dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=get_mc2_group().device_group)
            incompatible, communicate, any_active = flags.cpu().tolist()
            if incompatible or not any_active:
                # Retain eager handling of pending rows, but create no new cuts in
                # this globally non-decode round. Never mix eager and graph stores.
                state.allow_cut_this_forward = False
                state.allow_restore_this_forward = False
                state.apply_compaction_this_forward = False
                return None
        if any(model_inputs.get(name) is not None for name in ("inputs_embeds", "intermediate_tensors")):
            raise RuntimeError("model graph requires token IDs with TP=PP=1")
        metadata = context.attn_metadata
        source = next(iter(metadata.values())) if metadata else None
        if source is not None:
            for candidate in metadata.values():
                if candidate.block_tables.data_ptr() != source.block_tables.data_ptr():
                    raise RuntimeError("model graph currently requires one shared KV block-table layout")
        if self.frame is None:
            block_columns = (
                runner.max_model_len + runner.cache_config.block_size - 1
            ) // runner.cache_config.block_size
            if source is not None:
                block_columns = source.block_tables.shape[1]
            self.frame = ModelGraphFrame(state, self.capacity, runner.device, block_columns)
            self.update_stream = runner.update_stream
            model = runner.model.model
            for layer in islice(model.layers, model.start_layer, model.end_layer):
                impl = layer.self_attn.attn.impl
                if (
                    type(impl).__name__ != "AscendAttentionBackendImpl"
                    or impl.sliding_window is not None
                    or impl.sinks is not None
                    or impl.kv_sharing_target_layer_name is not None
                ):
                    raise RuntimeError("model graph requires standard BF16 Attention without sliding window/C8")
                self.frame.metadata_templates[layer.self_attn.attn.layer_name] = AscendMetadata(
                    attn_state=AscendAttentionState.DecodeOnly,
                    num_actual_tokens=self.capacity,
                    num_decode_tokens=self.capacity,
                    num_decodes=self.capacity,
                    max_query_len=1,
                    query_start_loc=torch.arange(self.capacity + 1, dtype=torch.int32, device=runner.device),
                    actual_seq_lengths_q=list(range(1, self.capacity + 1)),
                )
        frame = self.frame
        frame.prepare_inputs(model_inputs["input_ids"], model_inputs["positions"], source)
        from vllm.config import CUDAGraphMode

        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        if type(runner.model) is not ACLGraphWrapper:
            raise RuntimeError("token-cut full_decode requires the standard ACLGraphWrapper")
        def run_model():
            # v0.27 resolves "from_forward_context" MoE ops with a per-forward
            # cursor. Two warmups and capture share this context, but each is
            # a complete model invocation and must start at the first MoE.
            # Reset once per invocation, never between individual MoE layers.
            context.moe_layer_index = 0
            # Positional inputs enable the wrapper's Tensor address checks.
            return runner.model(frame.input_ids, frame.positions)
        with self.activate(context):
            entry = runner.model.concrete_aclgraph_entries.get(context.batch_descriptor)
            captured = entry is not None and entry.aclgraph is not None
            if self.startup_capture:
                if captured:
                    raise RuntimeError("token-cut model graph was already captured")
                # Warm the actual fixed-capacity body through the wrapper's
                # NONE bypass, then let the wrapper capture its FULL entry.
                context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                for _ in range(2):
                    frame.prepare_caches(communicate=False, allow_cut=False, allow_restore=False)
                    run_model()
                    torch.npu.current_stream().synchronize()
                context.cudagraph_runtime_mode = CUDAGraphMode.FULL
                frame.prepare_caches(communicate=False, allow_cut=False, allow_restore=False)
                frame.capturing = True
                try:
                    output = run_model()
                finally:
                    frame.capturing = False
                    # The standard wrapper leaves this flag set to suppress
                    # ordinary post-capture updates. This adapter must release
                    # its captured event gates before synchronizing instead.
                    context.capturing = False
                runner._update_full_graph_params_if_needed(context, self.capacity, token_cut_phase="capture")
                torch.npu.current_stream().synchronize()
                self.update_stream.synchronize()
                for task in frame.attention_tasks.values():
                    task["ready_wait"] = CapturedReadyWait(task["ready"], self.update_stream)
                with torch.npu.stream(self.update_stream):
                    for task in frame.attention_tasks.values():
                        task["gate"].reset(self.update_stream)
                self.update_stream.synchronize()
                # Both native and FIA gates received startup records. A bare
                # reset cannot drain value-events and lets the first replay
                # execute a stale task before its runtime update arrives.
                drain_startup_gates(
                    [*frame.events.values(), *(task["gate"] for task in frame.attention_tasks.values())],
                    self.update_stream,
                )
                frame.prepare_caches(communicate=False, allow_cut=False, allow_restore=False)
                self.captures += 1
                return output[:output_rows]
            if not captured:
                raise RuntimeError("token-cut model graph missing; capture_model must run before inference")
            # Do not reuse the native no-ack HCCL mailbox until all ranks
            # have completed the preceding invocation.
            torch.npu.current_stream().synchronize()
            dist.barrier(group=get_mc2_group().device_group)
            frame.prepare_caches(
                communicate=bool(communicate),
                allow_cut=state.allow_cut_this_forward,
                allow_restore=state.allow_restore_this_forward,
            )
            frame.generation += len(frame.layers)
            runner._update_full_graph_params_if_needed(context, self.capacity, token_cut_phase="before_replay")
            output = run_model()  # ACLGraphWrapper selects and replays the cached entry.
            runner._update_full_graph_params_if_needed(context, self.capacity)
            torch.npu.current_stream().synchronize()
            self.replays += 1
            result = frame.finish(output, output_rows)
            if self.observe_step is not None:
                self.observe_step(frame)
            return result