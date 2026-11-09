# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Window-only capture/read ownership over the real MRV2 helper and builder."""

from contextlib import nullcontext
from dataclasses import replace
from types import MethodType
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
)
from vllm_ascend.worker.v2.spec_decode.dflash import capture
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
from vllm_ascend.worker.v2.spec_decode.draft_window import graph as source
from vllm_ascend.worker.v2.spec_decode.draft_window.backends import DraftWindowGroupRoute, DraftWindowProviderStatus
from vllm_ascend.worker.v2.spec_decode.draft_window.graph import DraftWindowGraphBindingProvider


@pytest.fixture
def capture_inputs(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda value: value)
    monkeypatch.setattr(cudagraph, "build_attn_metadata", capture.build_attn_metadata)
    buffers = object.__new__(InputBuffers)
    buffers.device = torch.device("cpu")
    buffers.seq_lens = torch.zeros(2, dtype=torch.int32)
    buffers.query_start_loc = torch.zeros(3, dtype=torch.int32)
    buffers.input_ids = torch.zeros(16, dtype=torch.int32)
    buffers.positions = torch.zeros(16, dtype=torch.int64)
    buffers.is_padding = torch.zeros(16, dtype=torch.bool)
    tables = NS(
        input_block_tables=[torch.full((2, 32), 77, dtype=torch.int32)],
        num_blocks=NS(cpu=torch.full((1, 2), 32, dtype=torch.int32)),
        slot_mappings=torch.full((1, 16), 91, dtype=torch.int64),
        cp_size=1,
        _num_dirty_rows=0,
    )
    for name in ("get_dummy_block_tables", "get_dummy_slot_mappings"):
        setattr(tables, name, MethodType(getattr(BlockTables, name), tables))
    spec = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    builder = object.__new__(AscendAttentionMetadataBuilder)
    builder.device, builder.use_pcp = torch.device("cpu"), False
    builder.kv_cache_spec, builder.speculative_config = spec, NS(parallel_drafting=True, use_dspark=lambda: False)
    builder.model_config = NS(runner_type="generate", hf_text_config=NS(model_type="qwen3"))
    builder.vllm_config = NS(use_v2_model_runner=True, model_config=builder.model_config)
    builder.attn_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
    builder.decode_threshold, builder.metadata_cls = 8, AscendMetadata
    group = AttentionGroup(AscendAttentionBackend, ["draft.a", "draft.b"], spec, 0, metadata_builders=[builder])
    cache = NS(kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=group.layer_names)])
    route = DraftWindowGroupRoute(0, tuple(group.layer_names), 128, DraftWindowProviderStatus.DIRECT_DENSE)
    provider = DraftWindowGraphBindingProvider(policy=1024, group=route, block_tables=tables)
    bank = provider.register_capture_identity(
        descriptor=BatchExecutionDescriptor(CUDAGraphMode.FULL, 16, 2, uniform_token_count=8)
    )
    return NS(
        buffers=buffers,
        tables=tables,
        builder=builder,
        groups=[[group]],
        cache=cache,
        provider=provider,
        bank=bank,
        owner=NS(_draft_window_graph_binding_provider=provider),
    )


def prepare(c, *, skip=False, causal=False):
    return cudagraph._prepare_dflash_inputs_to_capture(
        2, 16, c.buffers, c.tables, c.groups, c.cache, 4096, skip, causal
    )


@pytest.mark.parametrize("causal", [True, False, {0: False}])
def test_capture_changes_only_draft_reads(capture_inputs, causal):
    c = capture_inputs
    off = prepare(c, causal=causal)
    full = (
        c.buffers.seq_lens,
        c.buffers.query_start_loc,
        c.tables.slot_mappings,
        c.buffers.positions,
        c.tables.input_block_tables[0],
    )
    before = [value.clone() for value in full]
    raw = cudagraph.build_attn_metadata
    with capture.draft_window_dflash_capture_scope(c.owner):
        on = prepare(c, causal=causal)
    assert cudagraph.build_attn_metadata is raw
    layer = on.attn_metadata["draft.a"]
    assert layer is on.attn_metadata["draft.b"] and layer is not off.attn_metadata["draft.a"]
    assert layer.block_tables.data_ptr() == c.bank.window_block_table.data_ptr()
    assert layer.seq_lens_list == [1151, 1151] and layer.actual_seq_lengths_q == [8, 16]
    assert layer.seq_lens_gpu.data_ptr() == c.buffers.seq_lens.data_ptr()
    assert layer.slot_mapping.data_ptr() == c.tables.slot_mappings.data_ptr()
    assert layer.attn_mask is off.attn_metadata["draft.a"].attn_mask
    for actual, expected in zip(full, before):
        torch.testing.assert_close(actual, expected)


def test_capture_skip_and_exception_keep_the_original_helper(capture_inputs, monkeypatch):
    c = capture_inputs
    original = cudagraph.build_attn_metadata
    before = c.bank.window_block_table.clone(), c.bank.cpu_window_exact.clone()
    with capture.draft_window_dflash_capture_scope(c.owner):
        assert prepare(c, skip=True).attn_metadata is None
    for value, expected in zip((c.bank.window_block_table, c.bank.cpu_window_exact), before):
        torch.testing.assert_close(value, expected)
    with capture.draft_window_dflash_capture_scope(NS()):
        assert cudagraph.build_attn_metadata is original

    def fail(*args, **kwargs):
        raise RuntimeError("builder failed")

    monkeypatch.setattr(c.builder, "build", fail)
    with pytest.raises(RuntimeError, match="builder failed"), capture.draft_window_dflash_capture_scope(c.owner):
        prepare(c)
    assert cudagraph.build_attn_metadata is original


@pytest.mark.parametrize(
    "cp,sharded,draft_dcp,accepted",
    [(1, True, 1, True), (16, False, 1, True), (16, True, 1, False), (16, False, 16, False)],
)
def test_shared_target_cp_uses_actual_draft_group(capture_inputs, cp, sharded, draft_dcp, accepted):
    c = capture_inputs
    c.tables.cp_size, c.tables.block_sizes, c.tables.kernel_block_sizes = cp, [128], [128]
    spec = replace(c.cache.kv_cache_groups[0].kv_cache_spec, dcp_sharded=sharded)
    c.cache.kv_cache_groups[0].kv_cache_spec = c.builder.kv_cache_spec = spec
    c.groups[0][0] = AttentionGroup(
        AscendAttentionBackend, ["draft.a", "draft.b"], spec, 0, metadata_builders=[c.builder]
    )
    impl = object.__new__(AscendAttentionBackendImpl)
    impl.use_bnsd_kv_cache, impl.sinks, impl.sliding_window = False, None, None
    config = NS(
        parallel_config=NS(prefill_context_parallel_size=1, decode_context_parallel_size=draft_dcp),
        compilation_config=NS(static_forward_context={name: NS(impl=impl) for name in ("draft.a", "draft.b")}),
    )
    owner = NS(
        _draft_window_policy=1024,
        block_tables=c.tables,
        kv_cache_config=c.cache,
        attn_vllm_config=config,
        attn_groups=c.groups,
        draft_attn_layer_names={"draft.a", "draft.b"},
    )
    if accepted:
        assert source.create_draft_window_graph_provider(owner).table is c.tables.input_block_tables[0]
    else:
        with pytest.raises(NotImplementedError):
            source.create_draft_window_graph_provider(owner)


def test_ar_banks_survive_together_and_failed_submission_quarantines(capture_inputs):
    provider = capture_inputs.provider
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 2, 2, uniform_token_count=1)
    banks = [provider.register_capture_identity(descriptor=desc, consumer_ordinal=i) for i in range(2)]
    assert banks[0].window_block_table.data_ptr() != banks[1].window_block_table.data_ptr()
    with pytest.raises(RuntimeError, match="uncertain submission"), provider.proposal_scope((False, False)):
        for i, bank in enumerate(banks):
            provider.prepare(
                bank,
                cpu_full_exact=torch.tensor([2400 + i, 1300 + i], dtype=torch.int32),
                idx_mapping_np=np.array([0, 1]),
            )
            provider.bind_current(desc)
        assert len(provider.get(None)) == 2
        assert provider.get(None)[0] is not provider.get(None)[1]
        raise RuntimeError("uncertain submission")
    assert provider.quarantined
    with pytest.raises(RuntimeError, match="quarantined"):
        provider.require_reusable()


@pytest.mark.parametrize("has_hook", [True, False])
@pytest.mark.parametrize("second_generation", [False, True])
def test_fresh_dflash_load_aligns_exactly_once(monkeypatch, has_hook, second_generation):
    calls = []
    model = NS(model=NS(use_aux_hidden_state=True, fc=NS(weight=torch.ones(2, 2))))
    if has_hook:
        model.post_process = lambda config: calls.append(("post_process", config))
    monkeypatch.setattr(dflash.DFlashSpeculator, "load_draft_model", lambda *args: model)
    monkeypatch.setattr(dflash, "disable_profiling_chunk_for_draft", lambda config: nullcontext())
    monkeypatch.setattr(dflash, "get_rotation_path", lambda config: "quarot")
    from vllm_ascend.models import qwen3_dspark

    monkeypatch.setattr(qwen3_dspark, "align_draft_weights", lambda actual, fc, config: calls.append(("align", config)))
    from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import AscendDFlash2Speculator

    owner = object.__new__(AscendDFlash2Speculator if second_generation else dflash.AscendDFlashSpeculator)
    owner.vllm_config = object()
    assert owner.load_draft_model(object(), set()) is model
    assert calls == [("post_process" if has_hook else "align", owner.vllm_config)]
    if second_generation:
        assert owner._dflash2_full_kv_retention is True


@pytest.fixture
def eagle(capture_inputs):
    from vllm_ascend.worker.v2.spec_decode.draft_window.eagle import EagleDraftWindow
    from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator

    c = capture_inputs
    owner = object.__new__(AscendEagleSpeculator)
    owner.method, owner.num_speculative_steps = "eagle3", 7
    owner.vllm_config = NS(model_config=NS(max_model_len=4096))
    owner.replicated_pcp = False
    owner.draft_attn_layer_names = {"draft.a", "draft.b"}
    owner.input_buffers, owner.block_tables = c.buffers, c.tables
    owner.input_batch = NS(
        num_reqs=2,
        num_tokens_after_padding=16,
        query_start_loc_np=np.array([0, 8, 16]),
        seq_lens_np=np.array([2400, 1300]),
        idx_mapping_np=np.array([1, 0]),
        idx_mapping=torch.tensor([1, 0]),
        has_prefill=True,
        prefill_len_np=np.array([2400, 1300]),
        num_computed_prefill_tokens_np=np.array([2392, 1292]),
        num_scheduled_tokens=np.array([8, 8]),
        num_draft_tokens_per_req=None,
        is_dummy=False,
    )
    owner._prepare_replicated_prefill_attn = lambda metadata, slots, *a, **kw: (metadata, slots)
    owner._draft_window_graph_binding_provider = c.provider
    owner._draft_window_graph_support = object()
    owner._draft_window_request_state = NS(
        prompt_len=NS(np=np.array([1300, 2400])),
        max_seq_len=np.array([2600, 3700]),
    )
    owner._draft_window_eagle = EagleDraftWindow(owner, c.provider)
    c.tables.gather_block_tables = lambda *a, **kw: None
    return owner


@pytest.mark.parametrize("step", [0, 1, 2])
def test_eagle_exact_uses_current_target_query_not_stopping_budget(eagle, capture_inputs, step):
    from vllm_ascend.worker.v2.spec_decode.draft_window.config import validate_eagle_graph_exact

    owner, batch = eagle, eagle.input_batch
    owner.num_speculative_steps = 3
    owner._draft_window_request_state.prompt_len.np[:] = [138, 140]
    owner._draft_window_request_state.max_seq_len[:] = [202, 204]
    batch.seq_lens_np[:] = [208, 206]
    batch.query_start_loc_np[:] = [0, 4, 8]
    owner._draft_window_graph_binding_provider.proposal = (False, False)
    # Persistent slots [1, 0]: these completed Q4 tails exceed the business
    # budgets but are the canonical Target ends for this proposal.
    for rejected in [0] if step == 0 else [0, 3]:
        exact = torch.tensor([208 - rejected + step, 206 - rejected + step], dtype=torch.int32)
        validate_eagle_graph_exact(owner, batch, exact, num_reqs=2, padded=2, step=step)
    low = torch.tensor([207 if step == 0 else 208 - 4 + step, 206], dtype=torch.int32)
    high = torch.tensor([209 + step, 206], dtype=torch.int32)
    for exact in (low, high):
        with pytest.raises(NotImplementedError, match="outside its request and model capacity"):
            validate_eagle_graph_exact(owner, batch, exact, num_reqs=2, padded=2, step=step)


def test_eagle_exact_checks_current_slot_reuse_prefill_and_model_capacity(eagle):
    from vllm_ascend.worker.v2.spec_decode.draft_window.config import validate_eagle_graph_exact

    owner, batch = eagle, eagle.input_batch
    owner.num_speculative_steps = 3
    provider = owner._draft_window_graph_binding_provider
    provider.proposal = (False, False)
    batch.idx_mapping_np[:] = [0, 1]
    batch.seq_lens_np[:] = [138, 2500]
    batch.query_start_loc_np[:] = [0, 4, 8]
    # Reused persistent slot 0 now belongs to a shorter request. A prior
    # generation's prompt or max_tokens must not supply this row's bounds.
    owner._draft_window_request_state.prompt_len.np[:] = [138, 2400]
    owner._draft_window_request_state.max_seq_len[:] = [202, 2496]
    validate_eagle_graph_exact(owner, batch, torch.tensor([138, 2500]), num_reqs=2, padded=2, step=0)
    with pytest.raises(NotImplementedError):
        validate_eagle_graph_exact(owner, batch, torch.tensor([137, 2500]), num_reqs=2, padded=2, step=0)
    batch.seq_lens_np[0] = owner.max_model_len + 1
    with pytest.raises(RuntimeError, match="current Target query/exact extent"):
        validate_eagle_graph_exact(owner, batch, torch.tensor([4097, 2500]), num_reqs=2, padded=2, step=0)
    batch.seq_lens_np[0] = 100
    batch.num_computed_prefill_tokens_np[0], batch.num_scheduled_tokens[0] = 96, 4
    provider.proposal = (True, False)
    validate_eagle_graph_exact(owner, batch, torch.tensor([101, 2501]), num_reqs=2, padded=2, step=1)
    with pytest.raises(NotImplementedError):
        validate_eagle_graph_exact(owner, batch, torch.tensor([102, 2501]), num_reqs=2, padded=2, step=1)


@pytest.mark.parametrize("fail_forward", [False, True])
def test_eagle_native_first_wraps_real_prefill_and_preserves_target(eagle, capture_inputs, monkeypatch, fail_forward):
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

    c, owner = capture_inputs, eagle
    target = prepare(c, causal=True).attn_metadata
    target_layer = target["draft.a"]
    full_table, full_lengths = target_layer.block_tables, target_layer.seq_lens_gpu
    target["target.only"] = object()
    slots = {name: c.tables.slot_mappings[0] for name in owner.draft_attn_layer_names}
    calls = []

    def run_prefill(self, n, tokens, metadata, mappings, *args, **kwargs):
        calls.append("forward")
        assert n == 2 and tokens == 16
        assert set(metadata) == owner.draft_attn_layer_names
        assert c.provider._active_bank is c.bank
        assert metadata["draft.a"] is metadata["draft.b"] and metadata["draft.a"] is not target_layer
        assert metadata["draft.a"].block_tables is c.bank.window_block_table
        assert metadata["draft.a"].seq_lens_gpu is full_lengths and mappings is slots
        assert metadata["draft.a"].seq_lens_list == [1120, 1044]
        assert metadata["draft.a"].attn_state.name == "ChunkedPrefill"
        if fail_forward:
            raise RuntimeError("native first failed")

    monkeypatch.setattr(TargetDependentARSpeculator, "_prefill", run_prefill)
    monkeypatch.setattr(TargetDependentARSpeculator, "on_prefill_end", lambda *a: calls.append("end"))
    context = pytest.raises(RuntimeError, match="native first failed") if fail_forward else nullcontext()
    with (
        context,
        owner._draft_window_eagle.proposal_scope(
            owner.input_batch, dummy_run=False, is_profile=False, num_speculative_tokens=0
        ),
    ):
        owner._prefill(2, 16, target, slots, None)
        assert c.provider._active_bank is c.bank
        owner.on_prefill_end(2)
        assert c.provider._active_bank is None and owner._draft_window_eagle.next_step == 1
    assert calls == (["forward"] if fail_forward else ["forward", "end"])
    assert target_layer.block_tables is full_table and target_layer.seq_lens_gpu is full_lengths
    assert c.provider.quarantined is fail_forward
    assert (c.provider._active_bank is c.bank) is fail_forward


def test_eagle_window_off_delegates_prefill_and_completion_once(eagle, capture_inputs, monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import TargetDependentARSpeculator

    owner = eagle
    del owner._draft_window_eagle
    del owner._draft_window_graph_binding_provider
    metadata = prepare(capture_inputs, causal=True).attn_metadata
    slots = object()
    calls = []
    monkeypatch.setattr(TargetDependentARSpeculator, "_prefill", lambda self, *a: calls.append(a))
    monkeypatch.setattr(TargetDependentARSpeculator, "on_prefill_end", lambda self, n: calls.append(n))
    owner._prefill(2, 16, metadata, slots, None)
    owner.on_prefill_end(2)
    assert len(calls) == 2 and calls[1] == 2
    assert calls[0][:2] == (2, 16) and calls[0][3] is slots
    assert all(calls[0][2][name] is value for name, value in metadata.items())


def test_eagle_native_query_rejection_precedes_bank_updates(eagle, capture_inputs):
    c = capture_inputs
    target = prepare(c, causal=True).attn_metadata
    target["draft.a"].actual_seq_lengths_q = [8, 15]
    before = c.bank.window_block_table.clone(), c.bank.cpu_window_exact.clone()
    with (
        pytest.raises(NotImplementedError, match="canonical effective query"),
        c.provider.proposal_scope((False, False)),
    ):
        eagle._prefill(2, 16, target, {}, None)
    assert c.provider._active_bank is None
    for actual, expected in zip((c.bank.window_block_table, c.bank.cpu_window_exact), before):
        torch.testing.assert_close(actual, expected)


def test_eagle_merged_ar_uses_corrected_exact_and_independent_consumers(eagle, capture_inputs):
    c, state = capture_inputs, eagle._draft_window_eagle
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 2, 2, uniform_token_count=1)
    banks = [c.provider.register_capture_identity(descriptor=desc, consumer_ordinal=i) for i in range(6)]
    replays = []
    eagle.decode_cudagraph_manager = NS(run_fullgraph=lambda actual: replays.append(actual))
    c.tables.input_block_tables[0].copy_(torch.arange(64, dtype=torch.int32).reshape(2, 32))
    eagle.input_buffers.seq_lens.copy_(torch.tensor([2398, 1298], dtype=torch.int32))
    eagle._draft_window_request_state.prompt_len.np[:] = [1200, 2300]
    eagle.input_batch.has_prefill = False
    eagle.input_batch.prefill_len_np[:] = [2300, 1200]
    eagle.input_batch.num_computed_prefill_tokens_np[:] = [2300, 1200]
    # This decode has rejected queries; its corrected device ends legitimately
    # precede the uncorrected Target ends, but never the request's prompt.
    with state.proposal_scope(eagle.input_batch, dummy_run=False, is_profile=False):
        state.prepare_first_graph(c.bank.descriptor)
        c.provider.finish_update()
        eagle._multi_step_decode(2, False, desc, None, num_speculative_steps=7)
        assert replays == [desc] and state.next_step == 7
        assert len({bank.window_block_table.data_ptr() for bank in banks}) == 6
        for step, bank in enumerate(banks):
            assert bank.cpu_window_exact.tolist() == [1118 + step, 1042 + step]
            assert bank.window_block_table[0, 0] == 10 and bank.window_block_table[1, 0] == 34
        assert [params["actual_seq_lengths_kv"] for params in c.provider.get(None)] == [
            [1118 + step, 1042 + step] for step in range(6)
        ]
        c.provider.finish_update()
    assert state.intervals is None and state.next_step == 0 and c.provider.proposal is None
    addresses = [bank.window_block_table.data_ptr() for bank in banks]
    with state.proposal_scope(eagle.input_batch, dummy_run=False, is_profile=False, num_speculative_tokens=0):
        state.prepare_first_graph(c.bank.descriptor)
        c.provider.finish_update()
    assert [bank.window_block_table.data_ptr() for bank in banks] == addresses


@pytest.mark.parametrize("case", ["fewer", "more", "nested"])
def test_eagle_capture_consumer_scope_restores_after_errors(eagle, capture_inputs, case):
    c, state = capture_inputs, eagle._draft_window_eagle
    target = prepare(c, causal=True).attn_metadata
    message = {"fewer": "every actual", "more": "more consumers", "nested": "nested"}[case]
    with pytest.raises(RuntimeError, match=message), state.capture_scope(n=2, tokens=16):
        if case == "nested":
            with state.capture_scope(n=2, tokens=16):
                pass
        elif case == "more":
            state.capture_metadata(target, CUDAGraphMode.FULL)
            state.capture_metadata(target, CUDAGraphMode.FULL)
    assert state.capture_call is None
    with state.capture_scope(n=2, tokens=16):
        actual = state.capture_metadata(target, CUDAGraphMode.FULL)
        assert actual["draft.a"].block_tables is c.bank.window_block_table


def test_dflash_capture_rejects_nested_patch_and_restores(capture_inputs):
    original = cudagraph.build_attn_metadata
    with capture.draft_window_dflash_capture_scope(capture_inputs.owner):
        scoped = cudagraph.build_attn_metadata
        with (
            pytest.raises(RuntimeError, match="already scoped"),
            capture.draft_window_dflash_capture_scope(capture_inputs.owner),
        ):
            pass
        assert cudagraph.build_attn_metadata is scoped
    assert cudagraph.build_attn_metadata is original


@pytest.mark.parametrize("fail", [False, True])
def test_eagle_dp_idle_uses_baseline_scope_without_publishing_window(eagle, capture_inputs, fail):
    state = eagle._draft_window_eagle
    eagle.dp_size = 2
    eagle.input_batch.is_dummy = True
    context = pytest.raises(RuntimeError, match="idle failed") if fail else nullcontext()
    with context, state.proposal_scope(eagle.input_batch, dummy_run=True, is_profile=False):
        assert state.dp_dummy and not state.active() and capture_inputs.provider.get(None) == ()
        if fail:
            raise RuntimeError("idle failed")
    assert not state.dp_dummy and capture_inputs.provider.proposal is None
    assert capture_inputs.provider.quarantined is fail


@pytest.mark.parametrize("kind", ["dflash", "ar"])
@pytest.mark.parametrize("window_on", [False, True])
def test_real_graph_manager_replay_uses_owned_source_and_original_order(capture_inputs, monkeypatch, kind, window_on):
    from unittest.mock import MagicMock

    from vllm_ascend.worker.v2.spec_decode.dflash import aclgraph as parallel
    from vllm_ascend.worker.v2.spec_decode.target_dependent_ar import aclgraph as ar

    module = parallel if kind == "dflash" else ar
    manager_type = module.DFlashAclGraphManager if kind == "dflash" else module.AutoRegressiveAclGraphManager
    parent = module.DFlashCudaGraphManager if kind == "dflash" else module.SpeculatorCudaGraphManager
    manager = object.__new__(manager_type)
    provider, desc = capture_inputs.provider, capture_inputs.bank.descriptor
    metadata = [{"draft.a": object()}]
    window = NS(provider=provider, active=lambda: window_on, dp_dummy=False)
    manager.speculator = NS(
        _draft_window_graph_binding_provider=provider,
        _draft_window_eagle=window if window_on else None,
        build_fia_params=lambda *a: [{"layer_name": "draft.a", "actual_seq_lengths_kv": [2400, 1300]}],
    )
    manager.is_draft_model_prefill = False
    graph = MagicMock(spec=module.UpdatableGraph)
    events, sources = [], []
    graph.resolve_tasks.side_effect = lambda source: sources.append(source) or "resolved"
    graph.update.side_effect = lambda *a: events.append("update")
    manager.graphs = {desc: graph}
    manager.update_stream = NS(wait_stream=lambda stream: events.append("wait"))
    monkeypatch.setattr(torch.npu, "current_stream", lambda: object())
    monkeypatch.setattr(parent, "run_fullgraph", lambda self, actual: events.append("replay") or "result")
    assert manager._updatable_graph_replay(desc, None if kind == "dflash" and window_on else metadata) == "result"
    assert events == ["wait", "replay", "update"]
    assert len(sources) == 1
    if window_on:
        assert sources[0] is provider
    else:
        expected = module.ContextSource if kind == "dflash" else module.SharedSource
        assert isinstance(sources[0], expected)
    graph.update.assert_called_once_with(manager.update_stream, "resolved")
