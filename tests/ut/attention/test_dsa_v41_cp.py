# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 CP contracts between the runner, cache metadata and attention.

Metadata tests use real builders for all four cache planes. Forward tests
exercise the inherited orchestration with CPU projections and cache writes;
only distributed communication and device operators are replaced. Graph
tests check persistent addresses and state controls, not NPU graph replay.
"""

import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import CircularBufferSpec

from vllm_ascend.attention import dsa_v41
from vllm_ascend.attention.context_parallel import dsa_cp, dsa_v41_cp
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v41.compressor import DeepseekV41Compressor
from vllm_ascend.models.deepseek_v41.indexer import DeepseekV41Indexer
from vllm_ascend.ops import rope_dsv4
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DSDynamicLinearMethod
from vllm_ascend.weight_switch import WeightSwitchConfig
from vllm_ascend.worker.v2.pcp_manager import AscendPCPAttentionContext

LAYER = "model.layers.0.attn"
PREFIXES = {
    "swa": "model.layers.0.swa_cache",
    "long_kv": "model.layers.0.long_kv_cache",
    "index_k": "model.layers.0.indexer.k_cache",
    "compressor_state": "model.layers.0.compressor.state_cache",
}
# Scheduler token IDs owned by each rank, in DualChunkSwap order. Repeated
# decode rows are added separately; the oracle does not call PCP helpers.
PREFILL_ROWS = {
    2: [[0, 1, 6, 7, 8, 9, 14, 15], [2, 3, 4, 5, 10, 11, 12, 13]],
    4: [[0, 7, 8, 15], [1, 6, 9, 14], [2, 5, 10, 13], [3, 4, 11, 12]],
}


def _linear(inputs, outputs):
    layer = torch.nn.Linear(inputs, outputs, bias=False)
    layer.quant_method = AscendUnquantizedLinearMethod()
    with torch.no_grad():
        layer.weight.copy_(torch.eye(outputs, inputs))
    return layer


def _projector(monkeypatch, world, rank, events, *, sharding, full_b=None):
    """Real PCP projection and weight switching; only device/collective boundaries are replaced."""
    projector = dsa_cp.AscendDSAPCPImpl.__new__(dsa_cp.AscendDSAPCPImpl)
    width = 4 // world
    full_a = torch.eye(width).repeat(world, 1, 1)
    full_b = torch.eye(4) if full_b is None else full_b
    projector.wo_a = torch.nn.Module()
    projector.wo_a.weight = torch.nn.Parameter(full_a[rank : rank + 1].clone() if sharding else full_a.clone())
    projector.wo_a.input_size = projector.wo_a.input_size_per_partition = width
    projector.wo_a.output_size = 4
    projector.wo_a.output_size_per_partition = width
    projector.wo_a.quant_method = AscendUnquantizedLinearMethod()
    projector.wo_b = _linear(width if sharding else 4, 4)
    with torch.no_grad():
        projector.wo_b.weight.copy_(full_b[:, rank * width : (rank + 1) * width] if sharding else full_b)
    projector.wo_b.input_size = 4
    projector.wo_b.input_size_per_partition = width
    projector.wo_b.output_size = projector.wo_b.output_size_per_partition = 4
    projector.wo_b.reduce_results = True
    projector.wo_b.skip_bias_add = False
    projector.n_local_groups = world
    projector.support_fp8_attention = False
    projector.enable_pcp_o_proj_weight_sharding = sharding
    projector._pcp_o_proj_use_full_weight = False
    projector._pcp_o_proj_weight_switches = None
    projector.wo_a_pcp_weight_method = projector.wo_a.quant_method
    projector.wo_b_pcp_weight_method = projector.wo_b.quant_method
    projector.pcp_o_proj_weight_switch_config = WeightSwitchConfig(
        group=SimpleNamespace(world_size=world), world_size=world, rank=rank
    )
    monkeypatch.setattr(dsa_cp.AscendDSAPCPImpl, "o_proj_full_pools", {})

    def matmul(value, weight, **kwargs):
        events.append("project")
        return torch.bmm(value.transpose(0, 1), weight).transpose(0, 1)

    def gather_weight(value, group, *, output, async_op):
        events.append("weights")
        assert async_op
        expected = full_a if value.ndim == 3 else full_b.T
        output.copy_(expected)
        return output, SimpleNamespace(wait=lambda: events.append("weight_wait"))

    def reduce(value):
        events.append("reduce")
        expected = torch.zeros_like(projector.reference_input)
        expected[:, rank * width : (rank + 1) * width] = projector.reference_input[:, rank * width : (rank + 1) * width]
        torch.testing.assert_close(value, expected)
        return projector.reference_input.clone()

    monkeypatch.setattr(dsa_cp.torch_npu, "npu_transpose_batchmatmul", matmul, raising=False)
    # Match the runtime import even when another UT replaces sys.modules.
    monkeypatch.setattr(importlib.import_module("vllm_ascend.distributed.utils"), "all_gather_async", gather_weight)
    monkeypatch.setattr(
        dsa_cp, "get_pcp_group", lambda: SimpleNamespace(world_size=world, rank_in_group=rank, all_reduce=reduce)
    )
    monkeypatch.setattr(dsa_cp.dsa_v1, "oproj_tp_enable", lambda: False)
    monkeypatch.setattr(torch.ops.vllm, "unquantized_gemm", torch.nn.functional.linear, raising=False)
    return projector


@pytest.fixture
def builders(monkeypatch):
    """Allocate real CPU builders with visible RoPE values and small caches."""
    full_cos = torch.arange(128, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 1, 1, 2).contiguous()
    full_sin = -full_cos
    rope_state = rope_dsv4.RopeGlobalState()
    rope_state.full_rope_cache["cp-test"] = (full_cos, full_sin)
    rope_state.registry_summary["cp-test"] = {"default"}
    rope_state.layer_info[LAYER] = ("cp-test", ["default"])
    monkeypatch.setattr(rope_dsv4, "_ROPE_STATE", rope_state)
    for module in (dsa_v41, dsa_v41_cp):
        monkeypatch.setattr(module, "get_full_cos_and_sin_dsa_for_layer", lambda name: (full_cos, full_sin))

    def make(world=2, rank=0, *, max_seqs=2, graph_sizes=(12, 24), dcp=1, legacy=False):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                tensor_parallel_size=world if legacy else 1,
                prefill_context_parallel_size=1 if legacy else world,
                decode_context_parallel_size=dcp,
            ),
            scheduler_config=SimpleNamespace(max_num_seqs=max_seqs, max_num_batched_tokens=64),
            speculative_config=SimpleNamespace(num_speculative_tokens=5),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=graph_sizes),
            cache_config=SimpleNamespace(block_size=8),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(
                    sliding_window=4,
                    num_attention_heads=8 if legacy else 2,
                    head_dim=2,
                    qk_rope_head_dim=2,
                    index_topk=3,
                )
            ),
        )
        common = dict(block_size=8, num_kv_heads=1, head_size=2, dtype=torch.float32)
        specs = {
            "swa": AscendSlidingWindowMLASpec(**common, sliding_window=4),
            "long_kv": AscendMLAAttentionSpec(**common, tokens_per_state=2),
            "index_k": AscendMLAAttentionSpec(**common, tokens_per_state=2, scale_dim=1),
            "compressor_state": CircularBufferSpec(**common, tokens_per_state=2),
        }
        monkeypatch.setattr(dsa_v41_cp, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=rank))
        monkeypatch.setattr(dsa_cp, "get_tp_group", lambda: SimpleNamespace(world_size=world, rank_in_group=rank))
        builder_class = (
            dsa_v41_cp.AscendDSAV41CPMetadataBuilder if legacy else dsa_v41_cp.AscendDSAV41PCPMetadataBuilder
        )
        result = {
            kind: builder_class(spec, [PREFIXES[kind]], config, torch.device("cpu")) for kind, spec in specs.items()
        }
        for builder in result.values():
            builder.prepare_source_rope()
        return result

    return make


def _batch(world, rank, scenario):
    """The runner's local view and independent canonical global context."""
    if scenario == "prefill":
        rows = PREFILL_ROWS[world]
        lengths, computed, flags = [8, 8], [4, 8], [True, True]
        local_lengths = [8 // (2 * world)] * 4
    elif scenario == "mixed":
        rows = [row[: len(row) // 2] + list(range(8, 14)) for row in PREFILL_ROWS[world]]
        lengths, computed, flags = [8, 6], [4, 8], [True, False]
        local_lengths = [8 // (2 * world)] * 2 + [6]
    elif scenario == "empty":
        rows = [[0]] + [[] for _ in range(world - 1)]
        lengths, computed, flags = [1], [3], [True]
        local_lengths = [1, 0] if rank == 0 else [0, 0]
    else:
        assert scenario in {"decode", "dummy"}
        rows = [list(range(6)) for _ in range(world)]
        lengths, computed, flags = [6], [8], [False]
        local_lengths = [6, 0, 0, 0]

    graph = scenario in {"decode", "dummy"}
    positions = torch.cat([torch.arange(start, start + length) for start, length in zip(computed, lengths)])
    num_tokens = len(positions)
    num_reqs = len(lengths)
    global_reqs = 4 if graph else num_reqs
    global_padded = 24 if graph else num_tokens + 2
    local_padded = 24 if graph else max(map(len, rows)) + 2
    offsets = np.array([0, *np.cumsum(lengths), *([num_tokens] * (global_reqs - num_reqs))], dtype=np.int32)
    seq_lens = torch.tensor(
        [start + length for start, length in zip(computed, lengths)] + [0] * (global_reqs - num_reqs), dtype=torch.int32
    )
    padded_positions = torch.cat([positions, torch.zeros(global_padded - num_tokens, dtype=torch.int64)])
    tables = torch.tensor([[2 + 2 * req, 3 + 2 * req] for req in range(global_reqs)], dtype=torch.int32)
    slots = torch.cat(
        [
            16 * (req + 1) + torch.arange(start, start + length)
            for req, (start, length) in enumerate(zip(computed, lengths))
        ]
    )
    global_slots = torch.cat([slots, torch.full((global_padded - num_tokens,), -1)])
    global_batch = SimpleNamespace(
        num_tokens=num_tokens,
        num_tokens_after_padding=global_padded,
        num_reqs=num_reqs,
        num_reqs_after_padding=global_reqs,
        query_start_loc=torch.from_numpy(offsets.copy()),
        query_start_loc_np=offsets.copy(),
        seq_lens=seq_lens,
        seq_lens_np=seq_lens.numpy().copy(),
        seq_lens_cpu_upper_bound=seq_lens + 1000,
        num_computed_tokens_np=np.array(computed, dtype=np.int32),
        num_scheduled_tokens=np.array(lengths, dtype=np.int32),
        is_prefilling_np=np.array(flags),
        positions=padded_positions,
        dcp_local_seq_lens=None,
        attn_state=scenario,
        is_dummy=scenario == "dummy",
    )
    # First occurrence selects a single copy of replicated decode tokens.
    gathered_rows = torch.full((world, local_padded), -1, dtype=torch.int64)
    for peer, ids in enumerate(rows):
        gathered_rows[peer, : len(ids)] = torch.tensor(ids, dtype=torch.int64)
    restore = torch.tensor(
        [int((gathered_rows.flatten() == token).nonzero()[0]) for token in range(num_tokens)], dtype=torch.int64
    )
    restore = torch.cat([restore, torch.full((global_padded - num_tokens,), -99)])
    gathered_slots = torch.full_like(gathered_rows, -1)
    for peer, ids in enumerate(rows):
        gathered_slots[peer, : len(ids)] = slots[ids]
    context = AscendPCPAttentionContext(
        global_batch=global_batch,
        global_block_tables=(tables, tables + 16),
        global_slot_mappings=torch.stack([global_slots, torch.where(global_slots >= 0, global_slots + 128, -1)]),
        hidden_restore_idx=restore,
    )
    local_ids = rows[rank]
    local_positions = torch.cat([positions[local_ids], torch.zeros(local_padded - len(local_ids), dtype=torch.int64)])
    local_offsets = torch.tensor([0, *np.cumsum(local_lengths)], dtype=torch.int32)
    local_seqs = torch.tensor(
        [
            int(local_positions[end - 1]) + 1 if end > start else 0
            for start, end in zip(local_offsets[:-1], local_offsets[1:])
        ],
        dtype=torch.int32,
    )
    local_flags = [False] if graph else ([True, True, False] if scenario == "mixed" else [True] * len(local_lengths))
    local = AscendCommonAttentionMetadata(
        query_start_loc=local_offsets,
        query_start_loc_cpu=local_offsets.clone(),
        seq_lens=local_seqs,
        seq_lens_cpu=local_seqs.clone(),
        seq_lens_cpu_upper_bound=local_seqs + 1000,
        num_reqs=len(local_lengths),
        num_actual_tokens=len(local_ids),
        num_input_tokens=local_padded,
        max_query_len=max(local_lengths),
        max_seq_len=int(seq_lens.max()),
        block_table_tensor=tables.repeat_interleave(2, dim=0)[: len(local_lengths)],
        slot_mapping=gathered_slots.flatten().clone(),
        positions=local_positions,
        is_prefilling=torch.tensor(local_flags),
        attn_state=scenario,
    )
    return SimpleNamespace(
        local=local,
        context=context,
        rows=rows,
        positions=positions,
        slots=slots,
        local_ids=local_ids,
        graph=graph,
        lengths=lengths,
        computed=computed,
    )


def _build_planes(builders, batch):
    # Like the runner: batch metadata spans cache groups, physical mappings do
    # not. Global and local namespaces must stay separate within both scopes.
    shared_batch: dict[str, Any] = {}
    result: dict[str, Any] = {}
    for kind, builder in builders.items():
        group = int(kind in {"long_kv", "index_k"})
        slots = batch.local.slot_mapping.clone()
        if group:
            slots = torch.where(slots >= 0, slots + 128, -1)
        result[PREFIXES[kind]] = builder.build(
            0,
            batch.local.replace(slot_mapping=slots),
            pcp_context=batch.context,
            pcp_cache_group_idx=group,
            num_actual_reqs=1 if batch.graph else batch.local.num_reqs,
            full_graph_mode=batch.graph,
            common_v41_metadata={},
            common_v41_batch_metadata=shared_batch,
        )
    # The target adapter must ignore independent draft cache metadata.
    result["draft.attn"] = object()
    return result


class TestLegacyMetadata:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("last_rank", [False, True])
    @pytest.mark.parametrize("causal", [False, True])
    def test_query_intersection_uses_device_lengths_and_global_rope(
        self, builders, monkeypatch, world, last_rank, causal
    ):
        rank = world - 1 if last_rank else 0
        batch = _batch(world, rank, "prefill")
        global_batch = batch.context.global_batch
        # Only pinned staging is device-specific. The legacy metadata builder
        # and its request-intersection calculation execute unmocked.
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self)
        common = AscendCommonAttentionMetadata(
            query_start_loc=global_batch.query_start_loc,
            query_start_loc_cpu=global_batch.query_start_loc.clone(),
            seq_lens=global_batch.seq_lens,
            seq_lens_cpu=global_batch.seq_lens + 1000,
            num_reqs=2,
            num_actual_tokens=16,
            num_input_tokens=18,
            max_query_len=8,
            max_seq_len=16,
            block_table_tensor=batch.context.global_block_tables[0],
            slot_mapping=batch.context.global_slot_mappings[0],
            positions=global_batch.positions,
            is_prefilling=torch.tensor([True, True]),
            causal=causal,
        )
        local = builders(world, rank, legacy=True)["swa"].build(
            0, common, common_v41_metadata={}, common_v41_batch_metadata={}
        )
        # Independent examples include partial requests and rank-tail padding.
        expected = {
            (2, 0): ([0, 8, 9], [12, 9], 0, 9),
            (2, 1): ([0, 0, 7], [0, 16], 9, 16),
            (4, 0): ([0, 5, 5], [9, 0], 0, 5),
            (4, 3): ([0, 0, 1], [0, 16], 15, 16),
        }
        offsets, lengths, start, end = expected[world, rank]
        if not causal:
            lengths = [
                length if right > left else 0 for length, left, right in zip([12, 16], offsets[:-1], offsets[1:])
            ]
        torch.testing.assert_close(local.query_start_loc, torch.tensor(offsets, dtype=torch.int32))
        torch.testing.assert_close(local.seq_lens, torch.tensor(lengths, dtype=torch.int32))
        assert local.num_actual_tokens == end - start
        assert local.seq_lens.data_ptr() != local.global_metadata.seq_lens.data_ptr()
        torch.testing.assert_close(local.global_metadata.seq_lens, torch.tensor([12, 16], dtype=torch.int32))
        torch.testing.assert_close(local.positions, batch.positions[start:end])
        torch.testing.assert_close(local.cos[LAYER], local.global_metadata.cos[LAYER][start:end])
        torch.testing.assert_close(local.global_metadata.cos[LAYER][:16, 0, 0, 0], batch.positions.float())


class TestPCPMetadata:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("last_rank", [False, True])
    @pytest.mark.parametrize("scenario", ["prefill", "mixed", "decode", "empty", "dummy"])
    def test_local_queries_and_global_cache_planes(self, builders, world, last_rank, scenario):
        rank = world - 1 if last_rank else 0
        batch = _batch(world, rank, scenario)
        metadata = _build_planes(builders(world, rank), batch)
        for kind, prefix in PREFIXES.items():
            local = metadata[prefix]
            has_prefill = scenario not in {"decode", "dummy"}
            global_meta = local.global_metadata if has_prefill else local
            assert isinstance(local, dsa_v41_cp.AscendDSAV41PCPMetadata) == has_prefill
            if not has_prefill:
                assert local.global_metadata is None
            assert local.num_actual_tokens == len(batch.local_ids)
            if has_prefill:
                assert local.local_num_tokens_after_padding == batch.local.num_input_tokens
            assert global_meta.num_actual_tokens == len(batch.positions)
            assert global_meta.num_actual_reqs == len(batch.lengths)
            torch.testing.assert_close(local.positions[: len(batch.local_ids)], batch.positions[batch.local_ids])
            torch.testing.assert_close(global_meta.positions[: len(batch.positions)], batch.positions)
            if has_prefill:
                assert torch.all(local.hidden_restore_idx[len(batch.positions) :] == 0)
            if batch.graph:
                expected_offsets = torch.arange(0, 25, 6, dtype=torch.int32)
                torch.testing.assert_close(local.query_start_loc, expected_offsets)
                torch.testing.assert_close(global_meta.query_start_loc, expected_offsets)
                assert torch.count_nonzero(local.seq_lens[1:]) == 0
                assert torch.count_nonzero(global_meta.seq_lens[1:]) == 0
            if kind == "compressor_state":
                if has_prefill:
                    assert local.c2_ring_metadata is None
                used = torch.zeros_like(global_meta.c2_ring_metadata[1])
                if scenario != "dummy":
                    used[: len(batch.lengths)] = torch.tensor(batch.lengths, dtype=used.dtype)
                torch.testing.assert_close(global_meta.c2_ring_metadata[1], used)
                complete = torch.zeros_like(global_meta.c2_complete_mask)
                if scenario != "dummy":
                    complete[: len(batch.positions)] = batch.positions.remainder(2) == 1
                torch.testing.assert_close(global_meta.c2_complete_mask, complete)
                assert torch.all(global_meta.slot_mapping == -1)
                continue
            expected = batch.slots + (128 if kind in {"long_kv", "index_k"} else 0)
            if kind in {"long_kv", "index_k"}:
                expected = torch.tensor([slot // 2 if int(slot) % 2 else -1 for slot in expected])
            if scenario == "dummy":
                expected.fill_(-1)
            block_size = 4 if kind in {"long_kv", "index_k"} else 8
            assert global_meta.storage_block_size == local.storage_block_size == block_size
            expected_slots = torch.stack((expected // block_size, expected % block_size), dim=1)
            expected_slots[expected < 0] = -1
            torch.testing.assert_close(global_meta.slot_mapping[: len(expected)], expected_slots.to(torch.int32))
            assert torch.all(global_meta.slot_mapping[len(expected) :] == -1)
            torch.testing.assert_close(
                local.slot_mapping[: len(batch.local_ids)], expected_slots[batch.local_ids].to(torch.int32)
            )
            assert torch.all(local.slot_mapping[len(batch.local_ids) :] == -1)
            if kind == "swa" and batch.local_ids and has_prefill:
                assert local.cos[LAYER].data_ptr() != global_meta.cos[LAYER].data_ptr()
                torch.testing.assert_close(
                    local.cos[LAYER][: len(batch.local_ids), 0, 0, 0], batch.positions[batch.local_ids].float()
                )
                torch.testing.assert_close(
                    global_meta.cos[LAYER][: len(batch.positions), 0, 0, 0], batch.positions.float()
                )

    @pytest.mark.parametrize("max_seqs,graph_sizes", [(2, (12, 24)), (16, ())])
    def test_capacity_and_buffer_lifetime_across_batches(self, builders, max_seqs, graph_sizes):
        group = builders(max_seqs=max_seqs, graph_sizes=graph_sizes)
        for builder in group.values():
            assert builder._seq_lens.numel() >= max(2 * max_seqs, max(graph_sizes, default=0)) + 1
            assert builder._global_builder._seq_lens.numel() >= max(max_seqs, max(graph_sizes, default=0)) + 1

        def addresses():
            buffers: list[int] = []
            for builder in group.values():
                for owner in (builder, builder._global_builder):
                    buffers.extend(
                        getattr(owner, name).data_ptr()
                        for name in ("_seq_lens", "_c2_ring_metadata", "_c2_complete_mask", "_c2_source_positions")
                    )
                buffers.append(builder._hidden_restore_idx_buffer.data_ptr())
                for rope in builder._pcp_rope_buffers.values():
                    buffers.extend(t.data_ptr() for t in rope)
            return tuple(buffers)

        initial = addresses()
        for scenario in ("prefill", "decode", "dummy", "empty", "mixed", "decode", "prefill"):
            batch = _batch(2, 1, scenario)
            metadata = _build_planes(group, batch)
            swa = metadata[PREFIXES["swa"]]
            assert addresses() == initial
            if isinstance(swa, dsa_v41_cp.AscendDSAV41PCPMetadata):
                assert torch.all(swa.hidden_restore_idx[len(batch.positions) :] == 0)
            if scenario != "empty":
                local_rope = swa.cos[LAYER].clone()
                group["swa"]._build_pcp_rope_views(batch.local, global_cache=True)
                torch.testing.assert_close(swa.cos[LAYER], local_rope)

    @pytest.mark.parametrize("asynchronous", [False, True])
    @pytest.mark.parametrize("stage", list(dsa_v41.DeviceMetadataStage))
    def test_metadata_tasks_cover_both_owners_and_are_drained_once(self, builders, asynchronous, stage):
        builder = builders()["swa"]
        if asynchronous:
            builder.enable_device_metadata()
        calls = []
        for name, owner in (("global", builder._global_builder), ("local", builder)):
            assert owner._device_metadata_enabled == asynchronous
            shared: dict[str, Any] = {}
            buffer = torch.zeros(1)
            run = lambda name=name: calls.append(name)
            assert owner._publish_task(shared, "task", buffer, stage, run) is buffer
            assert owner._publish_task(shared, "task", torch.ones(1), stage, run) is buffer
        tasks = builder.take_device_metadata_tasks()
        assert len(tasks) == (2 if asynchronous else 0)
        assert calls == ([] if asynchronous else ["global", "local"])
        for task in tasks:
            assert task.stage == stage
            task.run()
        assert calls == ["global", "local"]
        assert builder.take_device_metadata_tasks() == ()

    def test_rejects_unsupported_context(self, builders):
        with pytest.raises(NotImplementedError, match="DCP=1"):
            builders(dcp=2)
        builder = builders()["swa"]
        with pytest.raises(AssertionError, match="global batch context"):
            builder.build(0, _batch(2, 0, "prefill").local)


@pytest.fixture
def runtime(builders, monkeypatch):
    def make(*, sharding=True, overlap=True, rank=1):
        events: list[str] = []
        owners = builders(2, rank)
        projector = _projector(monkeypatch, 2, rank, events, sharding=sharding)
        attn = SimpleNamespace(
            rotary_emb=SimpleNamespace(layername=LAYER),
            n_heads=2,
            n_local_heads=2,
            head_dim=2,
            nope_head_dim=0,
            attn_sink=None,
            softmax_scale=1.0,
            wq_a=_linear(4, 4),
            wq_b=_linear(4, 4),
            wkv=_linear(4, 2),
            q_norm=torch.nn.Identity(),
            kv_norm=torch.nn.Identity(),
            dsa_attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=[torch.zeros(256, 2)]),
                dsa_attn=SimpleNamespace(impl=projector),
            ),
        )
        projector.cv_wq_a, projector.cv_wq_b, projector.cv_wkv = map(CVLinearWrapper, (attn.wq_a, attn.wq_b, attn.wkv))
        monkeypatch.setattr(dsa_v41, "get_ascend_config", lambda: SimpleNamespace(multistream_dsv4_dsa_overlap=overlap))
        impl = dsa_v41_cp.AscendDSAV41PCPImpl(
            "model.layers.0",
            SimpleNamespace(is_kv_source=False, has_long_context=False, compress_ratio=0),
            None,
            None,
            None,
        )
        context = SimpleNamespace(attn_metadata=None)
        for module in (dsa_v41, dsa_v41_cp):
            monkeypatch.setattr(module, "get_forward_context", lambda: context)

        def scatter(cache, slots, values):
            events.append("cache")
            valid = slots[:, 0] >= 0
            indices = slots[valid, 0].long() * 8 + slots[valid, 1]
            cache[indices] = values[valid]

        def attention(q, **kwargs):
            events.append("attention")
            return q.clone(), None

        monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter, raising=False)
        monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", attention, raising=False)
        monkeypatch.setattr(
            torch.ops._C_ascend, "inplace_partial_rotary_mul", lambda *args, **kwargs: None, raising=False
        )

        def run(scenario):
            batch = _batch(2, rank, scenario)
            metadata = _build_planes(owners, batch)
            context.attn_metadata = metadata
            canonical = torch.arange(len(batch.positions) * 4, dtype=torch.float32).view(-1, 4) + 1
            peers = []
            for ids in batch.rows:
                peer = torch.full((batch.local.num_input_tokens, 4), 9999.0)
                peer[: len(ids)] = canonical[ids]
                peers.append(peer)

            def gather(value, dim):
                events.append("hidden")
                return torch.cat(peers, dim=dim)

            monkeypatch.setattr(dsa_v41_cp, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather))
            projector.reference_input = torch.zeros_like(peers[rank])
            projector.reference_input[: len(batch.local_ids)] = canonical[batch.local_ids]
            output = torch.full_like(peers[rank], -777)
            assert impl.forward(attn, None, peers[rank], output) is output
            torch.testing.assert_close(output, projector.reference_input)
            return batch, metadata

        return SimpleNamespace(impl=impl, attn=attn, projector=projector, context=context, events=events, run=run)

    return make


class TestPCPBatchLifecycle:
    @pytest.mark.parametrize("sharding,overlap", [(False, False), (False, True), (True, False), (True, True)])
    @pytest.mark.parametrize("rank", [0, 1])
    def test_profile_prefill_decode_and_verification_share_one_weight_lifecycle(self, runtime, sharding, overlap, rank):
        state = runtime(sharding=sharding, overlap=overlap, rank=rank)
        profile_output = torch.empty(4, 4)
        state.impl.forward(state.attn, None, torch.zeros_like(profile_output), profile_output)
        assert torch.count_nonzero(profile_output) == 0
        assert not state.events
        switches = state.projector._pcp_o_proj_weight_switches or ()
        addresses = [
            (layer.weight.data_ptr(), part.gather_output.data_ptr())
            for layer, _, weight in switches
            for part in weight.gather_parts.values()
        ]
        for scenario in ("prefill", "decode", "dummy", "mixed", "decode", "empty", "prefill"):
            state.events.clear()
            state.run(scenario)
            has_prefill = scenario not in {"decode", "dummy"}
            assert state.events.count("hidden") == int(has_prefill)
            assert state.events.count("weights") == 2 * int(sharding and has_prefill)
            assert state.events.count("reduce") == int(sharding and not has_prefill)
            assert state.events.count("cache") == 1
            if sharding and has_prefill:
                assert state.events.index("hidden") < state.events.index("weights") < state.events.index("cache")
            assert addresses == [
                (layer.weight.data_ptr(), part.gather_output.data_ptr())
                for layer, _, weight in switches
                for part in weight.gather_parts.values()
            ]
            assert all(not weight.handles for _, _, weight in switches)
            assert not state.projector._pcp_o_proj_use_full_weight

    @pytest.mark.parametrize(
        "stage", ["query", "attention", "weight_gather", "weight_wait", "weight_switch", "projection"]
    )
    def test_failed_prefill_drains_both_weight_gathers_and_restores_decode(self, runtime, monkeypatch, stage):
        state = runtime()
        projector = state.projector
        with projector.pcp_o_proj_batch(has_prefill=False):
            switches = projector._get_pcp_o_proj_weight_switches()
        addresses = [layer.weight.data_ptr() for layer, _, _ in switches]

        def fail(*args, **kwargs):
            raise RuntimeError("injected failure")

        with monkeypatch.context() as failure:
            if stage == "query":
                original_gemm = torch.ops.vllm.unquantized_gemm

                def gemm(value, weight, bias=None):
                    if weight.data_ptr() == state.attn.wq_a.weight.data_ptr():
                        fail()
                    return original_gemm(value, weight, bias)

                failure.setattr(torch.ops.vllm, "unquantized_gemm", gemm)
            elif stage == "attention":
                failure.setattr(torch.ops._C_ascend, "npu_sparse_flash_mla", fail)
            elif stage == "projection":
                failure.setattr(dsa_cp.torch_npu, "npu_transpose_batchmatmul", fail)
            elif stage == "weight_gather":
                failure.setattr(switches[1][1], "all_gather_weight", fail)
            elif stage == "weight_wait":
                original = switches[0][1].wait_weight_all_gather

                def wait(weight):
                    original(weight)
                    fail()

                failure.setattr(switches[0][1], "wait_weight_all_gather", wait)
            else:
                original = switches[0][1].switch_weight

                def switch(layer, weight, *, use_full_weight):
                    original(layer, weight, use_full_weight=use_full_weight)
                    if use_full_weight:
                        fail()

                failure.setattr(switches[0][1], "switch_weight", switch)
            with pytest.raises(RuntimeError, match="injected failure"):
                state.run("prefill")
        assert all(not weight.handles for _, _, weight in switches)
        assert [layer.weight.data_ptr() for layer, _, _ in switches] == addresses
        assert not projector._pcp_o_proj_use_full_weight
        state.run("decode")
        state.run("prefill")

    @pytest.mark.parametrize("overlap", [False, True])
    def test_c2_source_updates_ring_index_and_long_kv_once_per_batch(self, runtime, monkeypatch, overlap):
        state = runtime(overlap=overlap)
        impl, attn = state.impl, state.attn
        impl.role = SimpleNamespace(is_kv_source=True, has_long_context=True, is_index_source=False, compress_ratio=2)
        impl.long_kv_source_prefix = PREFIXES["long_kv"]
        impl.index_k_source_prefix = PREFIXES["index_k"]
        impl.compressor_state_prefix = PREFIXES["compressor_state"]
        impl.topology = SimpleNamespace(index_topk=3)
        attn.shared_state = SimpleNamespace(topk_indices=torch.zeros(64, 3, dtype=torch.int32))
        attn.long_kv_cache = SimpleNamespace(kv_cache=[torch.zeros(256, 2)])
        k_cache, scale_cache = torch.zeros(256, 2), torch.zeros(256, 1)
        indexer = DeepseekV41Indexer.__new__(DeepseekV41Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.owns_k, indexer.width, indexer.rope_width = True, 2, 2
        indexer.wk, indexer.k_norm = _linear(2, 2), torch.nn.Identity()
        indexer.k_cache = SimpleNamespace(kv_cache=[(k_cache, scale_cache)])
        attn.indexer = indexer
        compressor = DeepseekV41Compressor.__new__(DeepseekV41Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.wkv, compressor.wgate, compressor.norm = _linear(4, 2), _linear(4, 2), torch.nn.Identity()
        compressor.state_cache = SimpleNamespace(kv_cache=[torch.zeros(256, 2)])
        compressor._ring_pooled, compressor._ring_num_cores = torch.empty(64, 2), 1
        attn.compressor = compressor
        state.context.no_compile_layers = {PREFIXES["long_kv"]: attn.long_kv_cache}
        ring_steps, stores = [], []

        def pool(kv, scores, cache, metadata, output, **kwargs):
            ring_steps.append(int(metadata[1].sum()))
            output.copy_(kv)
            return output

        def scatter(cache, slots, values):
            block_size = 8 if cache is attn.dsa_attn.swa_cache_layer.kv_cache[0] else 4
            valid = slots[:, 0] >= 0
            stores.append(int(valid.sum()))
            indices = slots[valid, 0].long() * block_size + slots[valid, 1]
            cache[indices] = values[valid].to(cache.dtype)

        monkeypatch.setattr("vllm_ascend.ops.triton.compressor.compressor_triton.compressor_from_projected", pool)
        monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter)
        monkeypatch.setattr(
            dsa_cp.torch_npu, "npu_dynamic_quant", lambda x, **kwargs: (x, torch.ones(x.shape[0])), raising=False
        )
        for scenario in ("prefill", "decode", "dummy", "mixed", "empty", "decode"):
            stores.clear()
            batch, _ = state.run(scenario)
            expected_live = 0 if scenario == "dummy" else len(batch.positions)
            expected_completed = 0 if scenario == "dummy" else int((batch.positions % 2 == 1).sum())
            assert ring_steps[-1] == expected_live
            assert stores == [expected_live, expected_completed, expected_completed, expected_completed]
            assert compressor._ring_pooled.shape == (64, 2)

    def test_decode_build_never_uses_global_owner_or_old_tasks(self, builders, monkeypatch):
        owners = builders()
        for builder in owners.values():
            monkeypatch.setattr(
                builder._global_builder, "build", lambda *args, **kwargs: pytest.fail("global decode build")
            )
            builder._global_builder._device_metadata_tasks = (object(),)
        metadata = _build_planes(owners, _batch(2, 0, "decode"))
        assert all(not builder.take_device_metadata_tasks() for builder in owners.values())
        state = metadata[PREFIXES["compressor_state"]]
        assert state.global_metadata is None
        torch.testing.assert_close(state.c2_ring_metadata[1], torch.tensor([6, 0, 0, 0], dtype=torch.int32))


class TestPreprocessingStreams:
    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("partitioned", [False, True])
    def test_q_kv_quantization_and_cube_dependencies(self, monkeypatch, overlap, partitioned):
        trace: list[tuple[str, str, torch.Tensor | tuple[str, int] | str | int | None]] = []
        current = ["main"]

        class Stream:
            def __init__(self, name):
                self.name = name

            def record_event(self):
                event = (self.name, len(trace))
                trace.append(("record", self.name, event))
                return event

            def wait_event(self, event):
                trace.append(("wait", self.name, event))

            def wait_stream(self, other):
                trace.append(("join", self.name, other.name))

        main, aux = Stream("main"), Stream("aux")

        @contextmanager
        def stream(target):
            old, current[0] = current[0], target.name
            try:
                yield
            finally:
                current[0] = old

        monkeypatch.setattr(torch.npu, "current_stream", lambda: main)
        monkeypatch.setattr(torch.npu, "stream", stream)
        monkeypatch.setattr(dsa_v41, "dsv4_dsa_overlap_stream", lambda: aux)

        class QuantLinear(torch.nn.Module):
            def __init__(self, name, inputs, outputs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.eye(inputs, outputs))
                self.weight_scale = torch.ones(outputs)
                self.bias = None
                self.quant_method = AscendW8A8DynamicLinearMethod()
                self.name = name

            def forward(self, value):
                return self.quant_method.apply(self, value)

        qa, qb, kv = QuantLinear("qa", 4, 4), QuantLinear("qb", 4, 4), QuantLinear("kv", 4, 2)
        names = {layer.weight.data_ptr(): layer.name for layer in (qa, qb, kv)}

        def quantize(value, **kwargs):
            trace.append(("quant", current[0], value.clone()))
            return value, torch.ones(value.shape[0])

        def matmul(value, weight, scale, **kwargs):
            trace.append(("cube", current[0], names[weight.data_ptr()]))
            return value @ weight

        def scatter(*args):
            trace.append(("store", current[0], None))

        def rope(value, *args, **kwargs):
            trace.append(("rope", current[0], value.shape[2]))

        monkeypatch.setattr(dsa_cp.torch_npu, "npu_dynamic_quant", quantize, raising=False)
        monkeypatch.setattr(dsa_cp.torch_npu, "npu_quant_matmul", matmul, raising=False)
        monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter, raising=False)
        monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rope, raising=False)
        attn = SimpleNamespace(
            wq_a=qa,
            wq_b=qb,
            wkv=kv,
            q_norm=torch.nn.Identity(),
            kv_norm=torch.nn.Identity(),
            head_dim=2,
            nope_head_dim=0,
            rotary_emb=SimpleNamespace(layername=LAYER),
            dsa_attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=[torch.zeros(16, 2)]),
                dsa_attn=SimpleNamespace(
                    impl=SimpleNamespace(
                        cv_wq_a=CVLinearWrapper(qa), cv_wq_b=CVLinearWrapper(qb), cv_wkv=CVLinearWrapper(kv)
                    )
                ),
            ),
        )
        hidden = torch.arange(16, dtype=torch.float32).view(4, 4)
        query = hidden[2:] if partitioned else hidden

        def metadata(count):
            return SimpleNamespace(
                swa=SimpleNamespace(num_actual_tokens=count, slot_mapping=torch.zeros(count, 2, dtype=torch.int32)),
                rope=lambda name, rows: (torch.ones(rows, 1, 1, 2), torch.zeros(rows, 1, 1, 2)),
            )

        cache_metadata = metadata(4)
        query_metadata = metadata(2) if partitioned else cache_metadata
        impl = dsa_v41.AscendDSAV41Impl.__new__(dsa_v41.AscendDSAV41Impl)
        impl.multistream_dsv4_dsa_overlap = overlap
        impl.role = SimpleNamespace(is_kv_source=False)
        q, qr = impl._preprocess(attn, query, hidden, query_metadata, cache_metadata)
        torch.testing.assert_close(q.flatten(1), query)
        torch.testing.assert_close(qr, query)
        cubes = [(stream, name) for kind, stream, name in trace if kind == "cube"]
        assert cubes == (
            [("main", "qa"), ("aux", "kv"), ("main", "qb")]
            if overlap
            else [("main", "qa"), ("main", "qb"), ("main", "kv")]
        )
        quantizations = [(stream, value) for kind, stream, value in trace if kind == "quant"]
        assert len(quantizations) == (2 if overlap and not partitioned else 3)
        if overlap:
            assert ("join", "main", "aux") in trace
            store = next(i for i, (kind, _, _) in enumerate(trace) if kind == "store")
            join = next(i for i, (kind, _, _) in enumerate(trace) if kind == "join")
            assert store < join < len(trace) - 1
            # Each cross-stream Cube handoff waits for the producer's event.
            for consumer, producer in (("aux", "main"), ("main", "aux")):
                assert any(
                    kind == "wait" and stream == consumer and isinstance(event, tuple) and event[0] == producer
                    for kind, stream, event in trace
                )
            if partitioned:
                torch.testing.assert_close(quantizations[1][1], hidden)
        else:
            assert all(stream == "main" for _, stream, _ in trace)
            assert not any(kind in {"join", "record", "wait"} for kind, _, _ in trace)


class TestPCPProjectionPartition:
    @pytest.mark.parametrize("tp_size,pcp_size", [(1, 2), (2, 2), (2, 4)])
    @pytest.mark.parametrize("quantized", [False, True])
    def test_checkpoint_shards_and_quantization_tensors_keep_the_forward_tp_group(
        self, monkeypatch, tp_size, pcp_size, quantized
    ):
        tp_rank, pcp_rank = tp_size - 1, pcp_size - 1
        group = SimpleNamespace(world_size=pcp_size, rank_in_group=pcp_rank)
        tp_group = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank)
        monkeypatch.setattr(dsa_cp, "get_pcp_group", lambda: group)
        monkeypatch.setattr(dsa_cp, "get_tp_group", lambda: tp_group)
        impl = dsa_cp.AscendDSAPCPImpl.__new__(dsa_cp.AscendDSAPCPImpl)
        impl.n_local_groups = pcp_size
        impl.wo_a, impl.wo_b = torch.nn.Module(), torch.nn.Module()
        total = tp_size * pcp_size * 2
        checkpoints = []
        for name, layer, axis in (("wo_a", impl.wo_a, 0), ("wo_b", impl.wo_b, 1)):
            layer.tp_size, layer.tp_rank, layer.tp_group = tp_size, tp_rank, tp_group
            layer.input_size = total if axis == 1 else 4
            layer.input_size_per_partition = layer.input_size // (tp_size if axis == 1 else 1)
            layer.output_size = total if axis == 0 else 4
            layer.output_size_per_partition = layer.output_size // (tp_size if axis == 0 else 1)
            layer.output_partition_sizes = [layer.output_size_per_partition]
            layer.n_local_groups = pcp_size
            layer.quant_method = AscendW8A8DynamicLinearMethod() if quantized else AscendUnquantizedLinearMethod()
            full = (
                torch.arange(total * 4, dtype=torch.float32).reshape(total, 4)
                if axis == 0
                else torch.arange(total * 4, dtype=torch.float32).reshape(4, total)
            )
            shape = list(full.shape)
            shape[axis] //= tp_size
            parameter = torch.nn.Parameter(torch.empty(shape), requires_grad=False)
            setattr(parameter, "output_dim" if axis == 0 else "input_dim", axis)
            parameter.weight_loader = lambda parameter, value: parameter.data.copy_(value)
            layer.register_parameter("weight", parameter)
            checkpoints.append((layer, parameter, full, axis))
            if quantized and axis == 0:
                for attr in ("weight_scale", "weight_offset"):
                    full = torch.arange(total, dtype=torch.float32)
                    parameter = torch.nn.Parameter(torch.empty(total // tp_size), requires_grad=False)
                    parameter.output_dim = 0
                    parameter.weight_loader = lambda parameter, value: parameter.data.copy_(value)
                    layer.register_parameter(attr, parameter)
                    checkpoints.append((layer, parameter, full, 0))
        impl._prepare_pcp_o_proj_weight_shards()
        composed_rank = tp_rank * pcp_size + pcp_rank
        for layer, parameter, full, axis in checkpoints:
            parameter.weight_loader(parameter, full)
            expected = full.narrow(axis, composed_rank * 2, 2)
            torch.testing.assert_close(parameter, expected)
            assert layer.tp_size == tp_size and layer.tp_rank == tp_rank
            assert layer.tp_group is tp_group
        assert impl.n_local_groups == pcp_size
        assert impl.wo_a.n_local_groups == 1
        assert impl.pcp_o_proj_weight_switch_config.group is group
        impl.wo_a.quant_method.supports_weight_switch = False
        with pytest.raises(RuntimeError, match="weight-switch capable"):
            impl._get_pcp_weight_switch_method(impl.wo_a)

    def test_prefill_projection_preserves_distinct_rank_token_values(self, monkeypatch):
        outputs = []
        for rank, values in enumerate(([1.0, 2.0, 0.0, 0.0], [10.0, 20.0, 0.0, 0.0])):
            events: list[str] = []
            projector = _projector(monkeypatch, 2, rank, events, sharding=True, full_b=torch.ones(4, 4))
            # The old replicated-decode path sums contributions from different tokens.
            monkeypatch.setattr(
                dsa_cp,
                "get_pcp_group",
                lambda rank=rank: SimpleNamespace(
                    world_size=2, rank_in_group=rank, all_reduce=lambda value: torch.full_like(value, 21.0)
                ),
            )
            attn = SimpleNamespace(dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=projector)))
            hidden = torch.tensor([values])
            output = torch.empty_like(hidden)
            impl = dsa_v41_cp.AscendDSAV41PCPImpl.__new__(dsa_v41_cp.AscendDSAV41PCPImpl)
            with projector.pcp_o_proj_batch(has_prefill=True):
                projector._maybe_all_gather_pcp_o_proj_weights()
                impl._project_output(attn, hidden.view(1, 2, 2), hidden, None, projected=output)
            outputs.append(output[0, 0].item())
        assert outputs == [3.0, 30.0]

    def test_fp8_projection_switches_packed_scale_and_weight_together(self, monkeypatch):
        events: list[str] = []
        projector = _projector(monkeypatch, 2, 0, events, sharding=True)
        projector.support_fp8_attention = True
        method = AscendW8A8MXFP8DSDynamicLinearMethod.__new__(AscendW8A8MXFP8DSDynamicLinearMethod)
        projector.wo_a.quant_method = projector.wo_a_pcp_weight_method = method
        projector.wo_a.weight_scale = torch.nn.Parameter(torch.ones(1, 2, 1, 2, dtype=torch.uint8), requires_grad=False)
        projector.reference_input = torch.tensor([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]])
        full_weight = torch.eye(2).repeat(2, 1, 1)
        full_scale = torch.cat((projector.wo_a.weight_scale, projector.wo_a.weight_scale + 1))
        expected_parts = {}
        with projector.pcp_o_proj_batch(has_prefill=False):
            switches = projector._get_pcp_o_proj_weight_switches()
            for layer, _, weight in switches:
                for name, part in weight.gather_parts.items():
                    full = (
                        full_weight
                        if layer is projector.wo_a and name == "weight"
                        else full_scale
                        if name == "weight_scale"
                        else torch.eye(4).T
                    )
                    expected_parts[part.gather_output.data_ptr()] = full

        def gather(value, group, *, output, async_op):
            output.copy_(expected_parts[output.data_ptr()])
            return output, SimpleNamespace(wait=lambda: None)

        monkeypatch.setattr(importlib.import_module("vllm_ascend.distributed.utils"), "all_gather_async", gather)
        monkeypatch.setattr(
            dsa_cp.torch_npu,
            "npu_dynamic_mx_quant",
            lambda value, **kwargs: (value, torch.ones(value.shape[:-1], dtype=torch.uint8)),
            raising=False,
        )
        monkeypatch.setattr(
            dsa_cp.torch_npu,
            "npu_transpose_quant_batchmatmul",
            lambda value, weight, **kwargs: torch.bmm(value.transpose(0, 1), weight).transpose(0, 1),
            raising=False,
        )
        addresses = [getattr(layer, name).data_ptr() for layer, _, weight in switches for name in weight.gather_parts]
        for has_prefill in (True, False, True):
            with projector.pcp_o_proj_batch(has_prefill=has_prefill):
                projector._maybe_all_gather_pcp_o_proj_weights()
                output = torch.empty_like(projector.reference_input)
                projector._forward_o_proj(projector.reference_input.view(2, 2, 2), output)
                torch.testing.assert_close(output, projector.reference_input)
            assert addresses == [
                getattr(layer, name).data_ptr() for layer, _, weight in switches for name in weight.gather_parts
            ]
            assert all(not weight.handles for _, _, weight in switches)


class TestForwardContract:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("scenario", ["prefill", "mixed", "decode", "empty", "dummy"])
    @pytest.mark.parametrize("sharding,overlap", [(False, False), (False, True), (True, False), (True, True)])
    @pytest.mark.parametrize("kv_source", [False, True])
    @pytest.mark.parametrize("index_role", ["reuse", "index_source", "candidate_source", "filtered_source"])
    def test_pcp_updates_global_caches_once_and_keeps_queries_local(
        self, builders, monkeypatch, world, scenario, kv_source, index_role, sharding, overlap
    ):
        """Exercise source/consumer roles through real metadata and inherited forward.

        CPU kernel stand-ins check canonical cache inputs and local query order;
        persistent TopK/candidate buffers must keep their untouched padded rows.
        """
        rank = world - 1
        batch = _batch(world, rank, scenario)
        metadata = _build_planes(builders(world, rank), batch)
        impl = dsa_v41_cp.AscendDSAV41PCPImpl.__new__(dsa_v41_cp.AscendDSAV41PCPImpl)
        index_source = index_role != "reuse"
        candidate_source = index_role == "candidate_source"
        filtered_source = index_role == "filtered_source"
        impl.role = SimpleNamespace(
            is_kv_source=kv_source,
            has_long_context=True,
            is_index_source=index_source,
            is_candidate_source=candidate_source,
            uses_candidate_filter=filtered_source,
        )
        impl.multistream_dsv4_dsa_overlap = overlap
        impl.topology = SimpleNamespace(candidate_topk_blocks=2, candidate_block_size=8)
        impl.swa_prefix, impl.long_kv_source_prefix, impl.index_k_source_prefix, impl.compressor_state_prefix = (
            PREFIXES.values()
        )
        events = []
        canonical = torch.arange(len(batch.positions) * 4, dtype=torch.float32).view(-1, 4) + 1
        peers = []
        for ids in batch.rows:
            peer = torch.full((batch.local.num_input_tokens, 4), 9999.0)
            peer[: len(ids)] = canonical[ids]
            peers.append(peer)
        hidden = peers[rank]

        def gather(value, dim):
            events.append("gather")
            torch.testing.assert_close(value, hidden)
            return torch.cat(peers, dim=dim)

        cache = torch.full((256, 2), -1.0)

        def write(cache_tensor, slots, values):
            events.append("swa")
            torch.testing.assert_close(values, canonical[:, :2])
            expected_slots = torch.full_like(batch.slots, -1) if scenario == "dummy" else batch.slots
            physical = torch.where(slots[:, 0] >= 0, slots[:, 0].long() * 8 + slots[:, 1], -1)
            torch.testing.assert_close(physical, expected_slots)
            valid = physical >= 0
            cache_tensor[physical[valid]] = values[valid]

        def rotary(value, cos, sin, **kwargs):
            events.append("rotary")
            expected = batch.positions if value.shape[0] == len(batch.positions) else batch.positions[batch.local_ids]
            torch.testing.assert_close(cos[:, 0, 0, 0], expected.float())
            return value

        def compressed(attn, values, positions, cos, sin, meta, **kwargs):
            events.append("compressed")
            torch.testing.assert_close(values, canonical)
            assert meta.swa is (metadata[PREFIXES["swa"]].global_metadata or metadata[PREFIXES["swa"]])
            assert meta.compressor.state is (
                metadata[PREFIXES["compressor_state"]].global_metadata or metadata[PREFIXES["compressor_state"]]
            )
            if scenario == "dummy":
                assert torch.count_nonzero(meta.compressor.state.c2_ring_metadata[1]) == 0
                assert torch.count_nonzero(meta.compressor.state.c2_complete_mask) == 0

        def attention(attn, q, meta, indices):
            events.append("attention")
            assert events.count("swa") == 1
            expected_cache = torch.full_like(canonical[:, :2], -1) if scenario == "dummy" else canonical[:, :2]
            torch.testing.assert_close(cache[batch.slots], expected_cache)
            torch.testing.assert_close(indices, attn.shared_state.topk_indices[: len(batch.local_ids)])
            return q.clone()

        index_cache = (object(), object())

        def select(values, qr, positions, cos, sin, source_cache, cache_metadata, **kwargs):
            events.append("select")
            assert events.count("swa") == 1
            expected = canonical[batch.local_ids]
            torch.testing.assert_close(values, expected)
            torch.testing.assert_close(qr, expected)
            torch.testing.assert_close(cos[:, 0, 0, 0], batch.positions[batch.local_ids].float())
            torch.testing.assert_close(positions, batch.positions[batch.local_ids])
            assert cache_metadata is metadata[PREFIXES["index_k"]]
            assert source_cache is index_cache
            assert kwargs["is_candidate_source"] == candidate_source
            assert kwargs["uses_candidate_filter"] == filtered_source
            torch.testing.assert_close(kwargs["candidates"], candidates_before[: len(batch.local_ids)])
            assert kwargs["output_indices"].data_ptr() == topk.data_ptr()
            selected = kwargs["output_indices"].fill_(11)
            candidates = torch.full_like(kwargs["candidates"], 23) if candidate_source else kwargs["candidates"]
            return selected, candidates

        projector = _projector(monkeypatch, world, rank, events, sharding=sharding)
        projector.reference_input = torch.zeros_like(hidden)
        projector.reference_input[: len(batch.local_ids)] = canonical[batch.local_ids]
        wq_a, wq_b, kv = _linear(4, 4), _linear(4, 4), _linear(4, 2)
        projector.cv_wq_a, projector.cv_wq_b, projector.cv_wkv = map(CVLinearWrapper, (wq_a, wq_b, kv))
        topk = torch.arange(batch.local.num_input_tokens * 3, dtype=torch.int32).view(-1, 3)
        candidates = torch.full((batch.local.num_input_tokens, 1, 2), 7, dtype=torch.int32)
        attn = SimpleNamespace(
            rotary_emb=SimpleNamespace(layername=LAYER),
            n_heads=2,
            n_local_heads=2,
            head_dim=2,
            nope_head_dim=0,
            wq_a=wq_a,
            q_norm=torch.nn.Identity(),
            wq_b=wq_b,
            wkv=kv,
            kv_norm=torch.nn.Identity(),
            dsa_attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=[cache]),
                dsa_attn=SimpleNamespace(impl=projector),
            ),
            indexer=SimpleNamespace(select=select),
            shared_state=SimpleNamespace(
                topk_indices=topk, candidates=candidates, candidate_lengths=None, topk_lengths=None
            ),
        )
        monkeypatch.setattr(dsa_v41_cp, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather))
        monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", write, raising=False)
        monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rotary, raising=False)
        monkeypatch.setattr(
            dsa_v41,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=metadata,
                no_compile_layers={
                    PREFIXES["index_k"]: SimpleNamespace(kv_cache=[index_cache]),
                },
            ),
        )
        monkeypatch.setattr(dsa_v41_cp, "get_forward_context", dsa_v41.get_forward_context)
        monkeypatch.setattr(impl, "_write_compressed_source", compressed)
        monkeypatch.setattr(impl, "_forward_attention", attention)
        output = torch.full_like(hidden, -777.0)
        topk_before = topk.clone()
        candidates_before = candidates.clone()
        assert impl.forward(attn, None, hidden, output) is output
        expected = torch.zeros_like(hidden)
        expected[: len(batch.local_ids)] = canonical[batch.local_ids]
        torch.testing.assert_close(output, expected)
        if index_source:
            topk_before[: len(batch.local_ids)] = 11
        if candidate_source:
            candidates_before[: len(batch.local_ids)] = 23
        torch.testing.assert_close(topk, topk_before)
        torch.testing.assert_close(candidates, candidates_before)
        assert events.count("gather") == int(scenario not in {"decode", "dummy"})
        assert events.count("swa") == events.count("project") == 1
        assert events.count("weights") == 2 * int(sharding and scenario not in {"decode", "dummy"})
        assert events.count("reduce") == int(sharding and scenario in {"decode", "dummy"})
        assert not projector._pcp_o_proj_use_full_weight
        if sharding:
            for layer, _, state in projector._get_pcp_o_proj_weight_switches():
                assert not state.handles
                assert layer.weight.data_ptr() == state.gather_parts["weight"].local_tensor.data_ptr()
        assert events.count("compressed") == int(kv_source)
        assert events.count("attention") == int(bool(batch.local_ids))
        assert events.count("select") == int(index_source and bool(batch.local_ids))
        if scenario == "dummy":
            assert torch.all(cache == -1)

    def test_legacy_cp_compressor_uses_global_rope_and_query_slice(self, monkeypatch):
        impl = dsa_v41_cp.AscendDSAV41CPImpl.__new__(dsa_v41_cp.AscendDSAV41CPImpl)
        impl.role = SimpleNamespace(
            is_kv_source=True,
            has_long_context=True,
            is_index_source=True,
            is_candidate_source=False,
            uses_candidate_filter=False,
        )
        impl.topology = SimpleNamespace(candidate_topk_blocks=2, candidate_block_size=8)
        impl.index_k_source_prefix = PREFIXES["index_k"]
        hidden = torch.arange(24, dtype=torch.float32).view(6, 4)
        local = SimpleNamespace(
            swa=SimpleNamespace(cp_token_range=(2, 4, 2, 0), num_actual_tokens=2),
            indexer=SimpleNamespace(cache=object()),
        )
        cos, sin = torch.ones(4, 2), torch.zeros(4, 2)
        global_metadata = SimpleNamespace(
            swa=SimpleNamespace(num_actual_tokens=4, slot_mapping=torch.zeros(4, 2, dtype=torch.int32)),
            positions=torch.arange(4),
            rope=lambda name, count: (cos[:count], sin[:count]),
        )
        impl.multistream_dsv4_dsa_overlap = True
        local.rope = lambda name, count: (cos[2 : 2 + count], sin[2 : 2 + count])
        calls, writes, selections = [], [], []
        kv = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            kv.weight.copy_(torch.eye(4)[:2])

        def wrapper(projection):
            return SimpleNamespace(
                _quant_method=object(),
                _has_communication=False,
                quantize=lambda values: (values, None),
                matmul=lambda values, scale, **kwargs: projection(values),
            )

        def select(values, qr, positions, local_cos, local_sin, cache, metadata, **kwargs):
            selections.append(values)
            torch.testing.assert_close(values, hidden[2:4])
            torch.testing.assert_close(qr, hidden[2:4])
            return kwargs["output_indices"], None

        attn = SimpleNamespace(
            rotary_emb=SimpleNamespace(layername=LAYER),
            n_heads=2,
            head_dim=2,
            nope_head_dim=0,
            wq_a=SimpleNamespace(bias=None),
            wkv=SimpleNamespace(bias=None),
            wq_b=SimpleNamespace(bias=None),
            q_norm=torch.nn.Identity(),
            kv_norm=torch.nn.Identity(),
            indexer=SimpleNamespace(select=select),
            shared_state=SimpleNamespace(topk_indices=torch.zeros(6, 3), candidates=torch.zeros(6, 1, 2)),
            dsa_attn=SimpleNamespace(
                swa_cache_layer=SimpleNamespace(kv_cache=[torch.zeros(256, 2)]),
                dsa_attn=SimpleNamespace(
                    impl=SimpleNamespace(
                        cv_wq_a=wrapper(torch.nn.Identity()), cv_wkv=wrapper(kv), cv_wq_b=wrapper(torch.nn.Identity())
                    )
                ),
            ),
        )
        monkeypatch.setattr(impl, "_global_layer_metadata", lambda metadata: global_metadata)
        monkeypatch.setattr(dsa_v41_cp, "get_forward_context", lambda: SimpleNamespace(attn_metadata=object()))

        def compress(*args, **kwargs):
            calls.append(args)
            cos.fill_(9999)

        def rotate(value, rope_cos, rope_sin, **kwargs):
            if value.shape[2] == 2:
                torch.testing.assert_close(rope_cos, torch.ones(2, 2))
                assert not calls

        monkeypatch.setattr(impl, "_write_compressed_source", compress)
        monkeypatch.setattr(
            torch.ops._C_ascend,
            "npu_scatter_nd_update_sk",
            lambda cache, slots, values: writes.append(values),
            raising=False,
        )
        monkeypatch.setattr(torch.ops._C_ascend, "inplace_partial_rotary_mul", rotate, raising=False)
        monkeypatch.setattr(
            dsa_v41,
            "get_forward_context",
            lambda: SimpleNamespace(no_compile_layers={PREFIXES["index_k"]: SimpleNamespace(kv_cache=[object()])}),
        )
        q, qr = impl._prepare_inputs_and_caches(attn, hidden, local, {})
        torch.testing.assert_close(q.flatten(1), hidden[2:4])
        assert len(writes) == 1
        torch.testing.assert_close(writes[0], hidden[:4, :2])
        impl._select_sparse_indices(attn, hidden, qr, torch.arange(2, 4), cos[2:4], sin[2:4], local)
        assert len(selections) == 1
        assert len(calls) == 1
        torch.testing.assert_close(calls[0][1], hidden[:4])
        assert calls[0][3].data_ptr() == cos.data_ptr()
        assert calls[0][4].data_ptr() == sin.data_ptr()
        assert calls[0][5] is global_metadata


class TestProjectionLayout:
    @pytest.mark.parametrize("world", [2, 4])
    @pytest.mark.parametrize("rank", [0, -1])
    @pytest.mark.parametrize("live_tokens,extra_padding", [(5, 0), (5, 1), (5, 3), (0, 2)])
    def test_legacy_cp_preserves_tp_heads_and_flashcomm_output_extent(
        self, monkeypatch, world, rank, live_tokens, extra_padding
    ):
        rank %= world
        per_rank = (live_tokens + world - 1) // world
        token_extent = world * (per_rank + extra_padding)
        heads, width = 2 * world, 2
        canonical = torch.arange(live_tokens * heads * width, dtype=torch.float32).view(live_tokens, heads, width) + 1
        peers = torch.zeros(world * per_rank, heads, width)
        peers[:live_tokens] = canonical
        local = peers[rank * per_rank : min((rank + 1) * per_rank, live_tokens)]
        group = SimpleNamespace(world_size=world, device_group=object())

        def exchange(recv, send, *, group):
            # Emulate peers' all-to-all sends, while running restore_tp_heads
            # itself unmocked to test the token/head coordinate change.
            expected_send = peers[rank * per_rank : (rank + 1) * per_rank]
            expected_send = torch.cat(expected_send.split(2, dim=1), dim=0)
            torch.testing.assert_close(send, expected_send)
            recv.copy_(peers[:, rank * 2 : (rank + 1) * 2])

        def project(value, output):
            expected = torch.zeros(token_extent, 2, width)
            expected[:live_tokens] = canonical[:, rank * 2 : (rank + 1) * 2]
            torch.testing.assert_close(value, expected)
            # FlashComm reduce-scatter keeps a TP-local token buffer.
            output.copy_(value.flatten(1).chunk(world, dim=0)[rank])

        monkeypatch.setattr(dsa_v41_cp, "get_tp_group", lambda: group)
        monkeypatch.setattr(dsa_cp.dist, "all_to_all_single", exchange)
        impl = dsa_v41_cp.AscendDSAV41CPImpl.__new__(dsa_v41_cp.AscendDSAV41CPImpl)
        attn = SimpleNamespace(
            dsa_attn=SimpleNamespace(dsa_attn=SimpleNamespace(impl=SimpleNamespace(_forward_o_proj=project)))
        )
        output = torch.full((token_extent // world, 4), -777.0)
        metadata = SimpleNamespace(swa=SimpleNamespace(cp_token_range=(0, 0, per_rank, 0)))
        assert impl._project_output(attn, local, torch.empty(token_extent, 4), metadata, projected=output) is output
        full = torch.zeros(token_extent, 4)
        full[:live_tokens] = canonical[:, rank * 2 : (rank + 1) * 2].flatten(1)
        torch.testing.assert_close(output, full.chunk(world)[rank])

    @pytest.mark.parametrize("impl_class", [dsa_v41.AscendDSAV41Impl, dsa_v41_cp.AscendDSAV41PCPImpl])
    @pytest.mark.parametrize("live,padded", [(4, 4), (3, 8), (0, 8)])
    def test_ordinary_and_pcp_preserve_local_output_buffer(self, impl_class, live, padded):
        values = torch.arange(live * 4, dtype=torch.float32).view(live, 2, 2) + 1
        output = torch.full((padded, 4), -777.0)
        attn = SimpleNamespace(
            dsa_attn=SimpleNamespace(
                dsa_attn=SimpleNamespace(
                    impl=SimpleNamespace(_forward_o_proj=lambda value, out: out.copy_(value.flatten(1)))
                )
            )
        )
        impl = impl_class.__new__(impl_class)
        assert impl._project_output(attn, values, torch.empty(padded, 4), None, projected=output) is output
        expected = torch.zeros_like(output)
        expected[:live] = values.flatten(1)
        torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("legacy,pcp", [(False, False), (True, False), (False, True), (True, True)])
def test_backend_selection_follows_current_target_or_draft_config(monkeypatch, legacy, pcp):
    config = {"legacy": legacy, "pcp": pcp}
    monkeypatch.setattr(dsa_v41_cp, "enable_dsa_cp", lambda: config["legacy"])
    monkeypatch.setattr(dsa_v41_cp, "enable_pcp", lambda: config["pcp"])
    if legacy and pcp:
        with pytest.raises(ValueError, match="cannot be enabled"):
            dsa_v41_cp.get_v41_cp_classes()
        return
    expected = (
        (dsa_v41_cp.AscendDSAV41CPMetadataBuilder, dsa_v41_cp.AscendDSAV41CPImpl)
        if legacy
        else (dsa_v41_cp.AscendDSAV41PCPMetadataBuilder, dsa_v41_cp.AscendDSAV41PCPImpl)
        if pcp
        else (dsa_v41.AscendDSAV41MetadataBuilder, dsa_v41.AscendDSAV41Impl)
    )
    assert dsa_v41_cp.get_v41_cp_classes() == expected
    # DSpark constructs its replicated draft under PCP1 in the same process.
    config.update(legacy=False, pcp=False)
    assert dsa_v41_cp.get_v41_cp_classes() == (dsa_v41.AscendDSAV41MetadataBuilder, dsa_v41.AscendDSAV41Impl)
