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
from vllm_ascend.worker.v2.spec_decode import draft_window as source
from vllm_ascend.worker.v2.spec_decode.dflash import capture
from vllm_ascend.worker.v2.spec_decode.dflash import speculator as dflash
from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowGroupRoute, DraftWindowProviderStatus
from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider


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
        assert owner._dflash2_full_kv_reference is True


@pytest.mark.parametrize("fail_forward", [False, True])
def test_eagle_native_first_uses_the_real_prefill_preparation_hook(capture_inputs, monkeypatch, fail_forward):
    from vllm_ascend.worker.v2.spec_decode import draft_graph_review_limits
    from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_adapter import Eagle3GraphIntegrationMixin
    from vllm_ascend.worker.v2.spec_decode.target_dependent_ar.speculator import AscendTargetDependentARSpeculator

    c = capture_inputs
    target = prepare(c, causal=True).attn_metadata
    target_layer = target["draft.a"]
    full_table, full_lengths = target_layer.block_tables, target_layer.seq_lens_gpu
    slots = {"draft.a": c.tables.slot_mappings[0], "draft.b": c.tables.slot_mappings[0]}
    batch = NS(
        num_reqs=2,
        num_tokens_after_padding=16,
        query_start_loc_np=np.array([0, 8, 16]),
        seq_lens_np=np.array([2400, 1300]),
        idx_mapping_np=np.array([0, 1]),
        idx_mapping=torch.tensor([0, 1]),
        has_prefill=True,
    )
    owner = object.__new__(Eagle3GraphIntegrationMixin)
    owner.can_reuse_target_attn_metadata = True
    owner._draft_window_graph_binding_provider = c.provider
    owner.input_batch, owner.block_tables = batch, c.tables
    owner.num_speculative_steps = 7
    owner.vllm_config = NS(model_config=NS(max_model_len=4096))
    owner._initialize_eagle3_draft_window_state()
    calls = []
    c.tables.gather_block_tables = lambda *args, **kwargs: calls.append("gather")
    # Static checkpoint qualification is covered at the support boundary;
    # this exercises the actual first-step geometry and bank ownership.
    monkeypatch.setattr(draft_graph_review_limits, "validate_eagle_single_request_review_exact", lambda *a, **kw: None)

    def run_prefill(self, n, tokens, metadata, mappings, *args, **kwargs):
        calls.append("forward")
        assert self._draft_window_graph_binding_provider._active_bank is c.bank
        assert metadata["draft.a"].block_tables is c.bank.window_block_table
        assert metadata["draft.a"].seq_lens_gpu is full_lengths and mappings is slots
        if fail_forward:
            raise RuntimeError("native first failed")

    monkeypatch.setattr(AscendTargetDependentARSpeculator, "_prefill", run_prefill)
    context = pytest.raises(RuntimeError, match="native first failed") if fail_forward else nullcontext()
    with context, c.provider.proposal_scope((False, False)):
        metadata, actual_slots = owner._build_drafter_attn_metadata(batch, target, slots)
        assert calls == ["gather"]
        assert metadata["draft.a"] is metadata["draft.b"] and metadata["draft.a"] is not target_layer
        assert target_layer.block_tables is full_table and target_layer.seq_lens_gpu is full_lengths
        assert metadata["draft.a"].seq_lens_list == c.bank.cpu_window_exact.tolist()
        assert metadata["draft.a"].attn_state.name == "ChunkedPrefill"
        owner._prefill(2, 16, metadata, actual_slots, None)
        assert c.provider._active_bank is c.bank
        owner.on_prefill_end(2)
        assert c.provider._active_bank is None and owner._eagle3_window_next_step == 1
    assert calls == ["gather", "forward"]
    assert c.provider.quarantined is fail_forward
    assert (c.provider._active_bank is c.bank) is fail_forward


def test_eagle_native_query_rejection_precedes_bank_updates(capture_inputs):
    from vllm_ascend.worker.v2.spec_decode.draft_window_eagle3_adapter import Eagle3GraphIntegrationMixin

    c = capture_inputs
    target = prepare(c, causal=True).attn_metadata
    target["draft.a"].actual_seq_lengths_q = [8, 15]
    owner = object.__new__(Eagle3GraphIntegrationMixin)
    owner.can_reuse_target_attn_metadata = True
    owner._draft_window_graph_binding_provider = c.provider
    owner._eagle3_window_next_step = 0
    batch = NS(num_reqs=2, query_start_loc_np=np.array([0, 8, 16]), num_tokens_after_padding=16)
    before = c.bank.window_block_table.clone(), c.bank.cpu_window_exact.clone()
    with (
        pytest.raises(NotImplementedError, match="canonical effective query"),
        c.provider.proposal_scope((False, False)),
    ):
        owner._build_drafter_attn_metadata(batch, target, {})
    assert c.provider._active_bank is None
    for actual, expected in zip((c.bank.window_block_table, c.bank.cpu_window_exact), before):
        torch.testing.assert_close(actual, expected)
