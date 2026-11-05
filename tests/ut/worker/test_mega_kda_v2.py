# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 MegaKDA contracts; real metadata/dispatch code, CPU copy/kernel doubles.

No vLLM/NPU installation is required. These checks do not replace the real
NPU cache, graph replay and checkpoint gates documented in the runbook.
"""

import ast
import math
import sys
from copy import copy
from dataclasses import dataclass
from enum import IntEnum
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from tests.ut.ops import test_kimi_mega_kda as contracts


@pytest.fixture
def runtime():
    return contracts.runtime.__wrapped__()


class _CopyPlan:
    """CPU stand-in for the existing sealed page-copy kernels."""

    def gather_into(self, state, packed, indices, flags=None):
        valid = (indices >= 0) & (indices < state.shape[0])
        packed.copy_(state.index_select(0, indices.clamp_min(0).long()))
        packed[~valid] = 0
        if flags is not None:
            packed[~flags] = 0

    def scatter(self, state, packed, indices):
        valid = (indices >= 0) & (indices < state.shape[0])
        state.index_copy_(0, indices[valid].long(), packed[valid])


def _paged_layer(runtime, batch=4, spec_len=8):
    layer = contracts._attention(runtime, batch=batch, spec_len=spec_len)
    conv, recurrent = layer.kv_cache
    # One raw backing with physical pages. This mirrors the V2 hybrid overlay,
    # including guards in the unused bytes of every page and at both ends.
    conv_bytes = conv[0].numel() * conv.element_size()
    state_bytes = recurrent[0].numel() * recurrent.element_size()
    page_bytes = conv_bytes + state_bytes + 64
    backing = torch.full((32 + conv.shape[0] * page_bytes + 32,), 37, dtype=torch.int8)
    raw = backing[32:-32]
    layer.kv_cache = (
        torch.as_strided(raw.view(torch.bfloat16), conv.shape, (page_bytes // 2, *conv.stride()[1:])),
        torch.as_strided(
            raw[conv_bytes:].view(torch.float32), recurrent.shape, (page_bytes // 4, *recurrent.stride()[1:])
        ),
    )
    layer._mega_use_paged_cache = True
    layer._mega_max_batch = batch
    assert layer._prepare_mega_weights({})
    layer._mega_cache_adapter = runtime["MegaKDAPagedCache"](
        layer.kv_cache, layer._mega_cache_workspace, _CopyPlan(), _CopyPlan()
    )
    return layer, backing, page_bytes


def test_v2_workspace_is_shared_by_layers_and_prepared_before_cache_binding(runtime):
    layers = [contracts._attention(runtime), contracts._attention(runtime)]
    workspaces = {}
    for layer in layers:
        layer._mega_use_paged_cache = True
        layer._mega_max_batch = 4
        assert layer._prepare_mega_weights(workspaces)
        assert layer._mega_cache_adapter is None
    assert len(workspaces) == 1
    assert layers[0]._mega_cache_workspace is layers[1]._mega_cache_workspace
    workspace = layers[0]._mega_cache_workspace
    assert workspace.conv.shape == (5, 10, 24)
    assert workspace.recurrent.shape == (33, 2, 4, 4)


@pytest.mark.parametrize("padding", [False, True])
def test_v2_paged_update_preserves_unscheduled_pages_and_null_slot(runtime, padding):
    layer, backing, page_bytes = _paged_layer(runtime)
    metadata = contracts._metadata()
    if padding:
        metadata.num_spec_decodes = 3
        metadata.num_actual_tokens = 24
        metadata.mega_query_lens = (8, 8, 8)
        metadata.spec_state_indices_tensor[-1].zero_()
        metadata.mega_first_verify[-1] = 0
    plan, hidden = contracts._plan(runtime, layer, metadata, rows=32)
    assert plan is not None
    before = backing.clone()
    old_conv, old_state = (state.clone() for state in layer.kv_cache)

    def operator(*args):
        # Distinct dense staging rows, including graph padding. Real and null
        # scheduler IDs must never be passed straight to these local buffers.
        assert args[9].is_contiguous() and args[10].is_contiguous()
        assert args[11].tolist() == [1, 2, 3, 4]
        assert args[12].tolist() == list(range(1, 33))
        args[9][1:] += 1
        args[10][1:] += 2
        return torch.zeros_like(hidden)

    layer._mega_operator.side_effect = operator
    layer.forward(hidden, torch.arange(32))
    expected = before.clone()
    for slot in metadata.spec_state_indices_tensor[:, 0].unique().tolist():
        if slot:
            start = 32 + slot * page_bytes
            payload = old_conv[slot] + 1
            expected[start : start + payload.numel() * 2] = payload.reshape(-1).view(torch.int8)
    conv_bytes = old_conv[0].numel() * 2
    for slot in metadata.spec_state_indices_tensor.unique().tolist():
        if slot:
            start = 32 + slot * page_bytes + conv_bytes
            payload = old_state[slot] + 2
            expected[start : start + payload.numel() * 4] = payload.reshape(-1).view(torch.int8)
    # Compare raw bytes, including null block 0, padding, guards and all other
    # groups' unowned pages; matching logical output alone would miss overlap.
    torch.testing.assert_close(backing, expected, rtol=0, atol=0)


def test_v2_operator_failure_does_not_scatter_or_retry(runtime):
    layer, backing, _ = _paged_layer(runtime)
    _, hidden = contracts._plan(runtime, layer, contracts._metadata())
    before = backing.clone()
    layer._mega_operator.side_effect = RuntimeError("partial staging write")
    with pytest.raises(RuntimeError, match="partial staging"):
        layer.forward(hidden, torch.arange(32))
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    layer._forward_default.assert_not_called()


def test_v2_staging_capacity_is_checked_before_any_copy(runtime):
    layer, backing, _ = _paged_layer(runtime)
    layer._mega_cache_workspace.max_batch = 2
    before = backing.clone()
    plan, hidden = contracts._plan(runtime, layer, contracts._metadata())
    assert plan is None
    layer.forward(hidden, torch.arange(32))
    layer._mega_operator.assert_not_called()
    layer._forward_default.assert_called_once()
    torch.testing.assert_close(backing, before, rtol=0, atol=0)


@pytest.mark.parametrize("fail_seal", [False, True])
def test_v2_adapter_initialization_prepares_once_and_publishes_only_after_seal(runtime, monkeypatch, fail_seal):
    layers = [_paged_layer(runtime)[0] for _ in range(2)]
    before = [[state.clone() for state in layer.kv_cache] for layer in layers]
    prepared = []

    class Plan(_CopyPlan):
        @classmethod
        def prepare(cls, state, limit):
            plan = cls()
            plan.limit = limit
            plan.sealed = False
            prepared.append(plan)
            return plan

        def seal(self):
            assert all(layer._mega_cache_adapter is None for layer in layers)
            if fail_seal:
                raise RuntimeError("seal failed")
            self.sealed = True

    module = ModuleType("vllm_ascend.ops.kda_state_copy_plan")
    module.KDAStateCopyPlan = module.StridedKDAFallbackPlan = Plan
    module.supports_kda_state_copy = lambda state: True
    module.cache_signature = runtime["_mega_cache_layout"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    contracts._definitions("vllm_ascend/ops/kimi_mega_kda.py", {"initialize_mega_kda_cache"}, runtime)
    initialize = runtime["initialize_mega_kda_cache"]
    context = {str(index): layer for index, layer in enumerate(layers)}
    if fail_seal:
        with pytest.raises(RuntimeError, match="seal failed"):
            initialize(context)
        assert all(layer._mega_cache_adapter is None for layer in layers)
    else:
        initialize(context)
        assert len(prepared) == 2  # Conv and SSM, deduplicated across layers.
        assert [plan.limit for plan in prepared] == [4, 32]
        assert all(plan.sealed for plan in prepared)
        assert layers[0]._mega_cache_adapter.conv_plan is layers[1]._mega_cache_adapter.conv_plan
        # A cache rebind may change every address without changing its layout.
        initialize(context)
        assert all(layer._mega_cache_adapter.supports(layer.kv_cache, 8, 4) for layer in layers)
    for layer, states in zip(layers, before):
        for actual, expected in zip(layer.kv_cache, states):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("plan_name", ["KDAStateCopyPlan", "StridedKDAFallbackPlan"])
def test_gather_into_uses_caller_storage_and_preserves_original_gather(runtime, plan_name):
    contracts._definitions("vllm_ascend/ops/kda_state_copy_plan.py", {"_canonical"}, runtime)
    contracts._definitions(
        "vllm_ascend/ops/kda_state_copy_plan.py",
        {plan_name},
        runtime,
        methods={"gather", "gather_into"},
        bases="object",
    )
    plan = runtime[plan_name]()
    plan._indices = lambda state, indices: indices
    copies = []

    def launch(state, packed, indices, flags=None, *, to_cache):
        assert not to_cache
        copies.append(packed)
        _CopyPlan().gather_into(state, packed, indices, flags)

    plan._launch = plan._copy = launch
    state = torch.arange(5 * 16, dtype=torch.float32).reshape(5, 1, 4, 4)
    indices = torch.tensor([4, 1, -1], dtype=torch.int32)
    flags = torch.tensor([True, False, True])
    packed = torch.full((3, 1, 4, 4), float("nan"))
    pointer = packed.data_ptr()
    plan.gather_into(state, packed, indices, flags)
    assert copies[-1] is packed and packed.data_ptr() == pointer
    expected = torch.stack([state[4], torch.zeros_like(state[0]), torch.zeros_like(state[0])])
    torch.testing.assert_close(packed, expected)
    torch.testing.assert_close(plan.gather(state, indices, flags), expected)
    with pytest.raises((ValueError, RuntimeError), match="flags"):
        plan.gather_into(state, packed, indices, flags[:1])
    with pytest.raises((ValueError, RuntimeError), match="dtype"):
        plan.gather_into(state, packed.to(torch.bfloat16), indices)
    with pytest.raises((ValueError, RuntimeError), match="buffer"):
        plan.gather_into(state, packed.transpose(-1, -2), indices)


class _GraphMode(IntEnum):
    NONE = 0
    FULL = 1


class _Common(SimpleNamespace):
    def replace(self, **kwargs):
        result = copy(self)
        result.__dict__.update(kwargs)
        return result

    def unpadded(self, tokens, reqs):
        return self.replace(
            num_reqs=reqs,
            num_actual_tokens=tokens,
            query_start_loc=self.query_start_loc[: reqs + 1],
            query_start_loc_cpu=self.query_start_loc_cpu[: reqs + 1],
            seq_lens=self.seq_lens[:reqs],
            seq_lens_cpu_upper_bound=self.seq_lens_cpu_upper_bound[:reqs],
            block_table_tensor=self.block_table_tensor[:reqs],
            is_prefilling=self.is_prefilling[:reqs],
        )

    def compute_num_computed_tokens(self):
        return self.seq_lens - torch.diff(self.query_start_loc)


class _BuilderBase:
    """Only upstream allocation and capture-entry plumbing is substituted."""

    def __init__(self, spec, names, config, device):
        self.vllm_config = config
        self.kv_cache_spec = spec
        self.num_spec = 7
        self.use_spec_decode = True
        self.use_full_cuda_graph = config.graph_mode == _GraphMode.FULL
        self.decode_cudagraph_max_bs = 32
        self.spec_state_indices_tensor = torch.zeros(32, 8, dtype=torch.int32)
        self.non_spec_state_indices_tensor = torch.zeros(32, dtype=torch.int32)
        self.spec_query_start_loc = torch.zeros(33, dtype=torch.int32)
        self.non_spec_query_start_loc = torch.zeros(33, dtype=torch.int32)
        self.num_accepted_tokens = torch.zeros(32, dtype=torch.int32)
        self.spec_token_indx = torch.zeros(256, dtype=torch.int32)
        self.non_spec_token_indx = torch.zeros(256, dtype=torch.int32)

    def build_for_cudagraph_capture(self, common):
        return self.build(0, common, torch.diff(common.query_start_loc), torch.diff(common.query_start_loc_cpu) - 1)


@dataclass
class _HybridMetadata:
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(self, group_id, num_reqs):
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(self, builder, num_reqs):
        return {
            "num_accepted_tokens": self.num_accepted_tokens,
            "num_decode_draft_tokens_cpu": self.num_decode_draft_tokens_cpu,
        }


def _pipeline(runtime, mode):
    ns = runtime
    ns.update(
        np=np,
        math=math,
        copy=copy,
        CUDAGraphMode=_GraphMode,
        GDNAttentionMetadataBuilder=_BuilderBase,
        AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH=1),
        AscendCommonAttentionMetadata=_Common,
        mamba_get_block_table_tensor=lambda table, *args: table,
        ring_state_update_skipped=lambda: False,
        uses_turboquant_groups=lambda _: False,
        _GDN_GRAPH_DUMMY_ACCEPTED_TOKEN_COUNT=1,
        NULL_BLOCK_ID=0,
        PAD_SLOT_ID=-1,
        get_pcp_group=lambda: SimpleNamespace(world_size=1),
        MambaHybridAttnMetadata=_HybridMetadata,
        # Spec-only integration below never enters prefill chunk kernels.
        split_decodes_and_prefills=lambda m, **kwargs: (m.num_reqs, 0, m.num_actual_tokens, 0),
    )
    for name in (
        "AscendDSAMetadataBuilder",
        "AscendDSACPMetadataBuilder",
        "AscendDSAV41MetadataBuilder",
        "AscendSFAMetadataBuilder",
    ):
        ns[name] = type(name, (), {})
    builder_file = "vllm_ascend/ops/gdn_attn_builder.py"
    tree = ast.parse((contracts.ROOT / builder_file).read_text(encoding="utf8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    names.update(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("GDN") and node.decorator_list
    )
    names.update(
        {"AscendGDNAttentionMetadataBuilder", "AscendGDNHostMetadataBuilder", "AscendMegaKDAHostMetadataBuilder"}
    )
    contracts._definitions(builder_file, names, ns)
    contracts._definitions("vllm_ascend/worker/v2/attn_utils.py", {"build_attn_metadata"}, ns)
    contracts._definitions(
        "vllm_ascend/worker/v2/model_states/mamba_hybrid.py", {"AscendMegaKDAMambaHybridAttnMetadata"}, ns
    )
    ns["UpstreamMambaState"] = type("UpstreamMambaState", (), {"add_request": lambda *args: None})
    contracts._definitions(
        "vllm_ascend/worker/v2/model_states/mamba_hybrid.py",
        {"AscendMambaHybridModelState"},
        ns,
        methods={"prepare_attn", "_with_mega_request_history", "add_request"},
        bases="UpstreamMambaState",
    )
    layer, backing, page_bytes = _paged_layer(ns)
    config = SimpleNamespace(
        num_speculative_tokens=7,
        graph_mode=mode,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        parallel_config=None,
        compilation_config=SimpleNamespace(static_forward_context={layer.prefix: layer}),
    )
    builder = ns["AscendMegaKDAHostMetadataBuilder"](None, [layer.prefix], config, torch.device("cpu"))
    state = ns["AscendMambaHybridModelState"]()
    state.vllm_config = config
    state._mega_request_tracker = ns["MegaKDAFirstVerifyTracker"](5, torch.device("cpu"))
    state.max_model_len = 1024
    state.num_accepted_tokens_gpu = torch.tensor([1, 4, 2, 3], dtype=torch.int32)
    group = SimpleNamespace(layer_names=[layer.prefix], get_metadata_builder=lambda _: builder)
    kv_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=None)])
    return ns, layer, state, builder, [[group]], kv_config, backing, page_bytes


@pytest.mark.parametrize("mode", [_GraphMode.NONE, _GraphMode.FULL])
@pytest.mark.parametrize("preceding_step", ["prefill", "no_draft", "capture", "align"])
def test_v2_hybrid_state_to_builder_to_planner_to_operator(runtime, mode, preceding_step):
    ns, layer, state, builder, groups, kv_config, _, _ = _pipeline(runtime, mode)
    live = 3 if mode == _GraphMode.FULL else 4
    table = torch.arange(1, 33, dtype=torch.int32).reshape(4, 8)
    table[live:] = 0
    batch = SimpleNamespace(
        num_reqs=live,
        num_reqs_after_padding=4,
        num_tokens=live * 8,
        num_tokens_after_padding=32,
        is_prefilling_np=np.zeros(live, dtype=np.bool_),
        idx_mapping=torch.arange(live - 1, -1, -1),
        num_draft_tokens_per_req=np.full(live, 7, dtype=np.int32),
        num_scheduled_tokens=np.full(live, 8, dtype=np.int32),
        query_start_loc=torch.arange(5, dtype=torch.int32) * 8,
        query_start_loc_np=np.arange(5, dtype=np.int32) * 8,
        seq_lens=torch.tensor([18, 28, 38, 48 if live == 4 else 0]),
        seq_lens_np=np.array([18, 28, 38, 48 if live == 4 else 0], dtype=np.int32),
        dcp_local_seq_lens=None,
        positions=torch.arange(32),
        attn_state=None,
    )
    if preceding_step == "capture":
        dummy = copy(batch)
        dummy.num_reqs = 4
        dummy.num_tokens = 32
        dummy.is_prefilling_np = np.zeros(4, dtype=np.bool_)
        capture_metadata = state.prepare_attn(
            dummy, mode, (torch.zeros_like(table),), torch.zeros(1, 32), groups, kv_config, for_capture=True
        )[layer.prefix]
        assert capture_metadata.mega_first_verify.tolist() == [0] * 4
        assert state._mega_request_tracker.context.eq(-1).all()
    elif preceding_step == "no_draft":
        decode = copy(batch)
        decode.num_tokens = live
        decode.num_tokens_after_padding = 4
        decode.query_start_loc = torch.arange(5, dtype=torch.int32)
        decode.query_start_loc_np = np.arange(5, dtype=np.int32)
        decode.num_draft_tokens_per_req = np.zeros(live, dtype=np.int32)
        decode.num_scheduled_tokens = np.ones(live, dtype=np.int32)
        decode.seq_lens = batch.seq_lens - 8
        decode.seq_lens_np = batch.seq_lens_np - 8
        metadata = state.prepare_attn(decode, mode, (table,), torch.zeros(1, 4), groups, kv_config)[layer.prefix]
        assert metadata.num_spec_decodes == 0 and metadata.num_decodes == live
        plan, hidden = contracts._plan(ns, layer, metadata, rows=4 if mode == _GraphMode.FULL else live)
        assert plan is None  # S=1 is not this binary's configured S=8.
        layer.forward(hidden, torch.arange(hidden.shape[0]))
        layer._mega_operator.assert_not_called()
        layer._forward_default.assert_called_once()
        layer._forward_default.reset_mock()
    first_pointer = None
    layer._mega_operator.return_value = torch.zeros(32, 32, dtype=torch.bfloat16)
    for round_index in range(2):
        if round_index and preceding_step == "align":
            # Physical running blocks move at a prefix-cache boundary, while
            # scheduler request indices (including index 0) retain identity.
            table.copy_(table.roll(1, 0))
        metadata = state.prepare_attn(batch, mode, (table,), torch.zeros(1, 32), groups, kv_config)[layer.prefix]
        assert metadata.num_spec_decodes == live
        assert metadata.mega_query_lens == (8,) * live
        first = round_index == 0 and preceding_step != "no_draft"
        assert metadata.mega_first_verify.tolist() == [int(first)] * live + [0] * (4 - live)
        if first_pointer is not None:
            assert metadata.mega_first_verify.data_ptr() == first_pointer
        first_pointer = metadata.mega_first_verify.data_ptr()
        plan, hidden = contracts._plan(ns, layer, metadata, rows=32)
        assert plan is not None and (plan.batch, plan.spec_len) == (4, 8)
        expected = state.num_accepted_tokens_gpu[batch.idx_mapping].tolist() + [1] * (4 - live)
        assert plan.num_accepted_tokens.tolist() == expected
        assert plan.conv_num_accepted_tokens.tolist() == ([8] * live + [1] * (4 - live) if first else expected)
        layer.forward(hidden, torch.arange(32))
        batch.seq_lens[:live] += 1
        batch.seq_lens_np[:live] += 1
    assert layer._mega_operator.call_count == 2
    layer._forward_default.assert_not_called()
    if preceding_step == "capture":
        assert builder._mega_tracker.context.eq(-1).all()  # Dummy NULL rows only.
    else:
        assert builder._mega_tracker is None  # Real V2 steps use request history.


def test_v2_request_index_zero_and_reuse_reset_independently(runtime):
    ns, _, state, _, _, _, _, _ = _pipeline(runtime, _GraphMode.NONE)
    tracker = state._mega_request_tracker
    slots = torch.tensor([1, 2])  # Scheduler request 0 is tracker row 1.
    decode = torch.zeros(2, dtype=torch.bool)
    spec = ~decode
    assert tracker.update(slots, torch.tensor([500, 600]), decode, spec).tolist() == [1, 1]
    assert tracker.update(slots, torch.tensor([501, 601]), decode, spec).tolist() == [0, 0]
    state.add_request(0, SimpleNamespace())
    # New request starts from a longer cached prefix than the previous owner.
    assert tracker.update(slots, torch.tensor([700, 602]), decode, spec).tolist() == [1, 0]
    assert tracker.context[0] == -1
    # Other backends must not receive this Kimi-only kwarg.
    metadata = ns["AscendMegaKDAMambaHybridAttnMetadata"](decode, mega_first_verify=torch.ones(2))
    assert "mega_first_verify" not in metadata.get_extra_attn_kwargs(
        _BuilderBase(None, None, SimpleNamespace(graph_mode=0), None), 2
    )


def test_v2_dummy_step_does_not_consume_live_request_history(runtime):
    _, _, state, _, _, _, _, _ = _pipeline(runtime, _GraphMode.NONE)
    tracker = state._mega_request_tracker
    before = tracker.context.clone(), tracker.first.clone()
    metadata = _HybridMetadata(torch.zeros(2, dtype=torch.bool))
    dummy = SimpleNamespace(is_dummy=True, num_reqs=2, seq_lens=torch.zeros(2, dtype=torch.int32))
    result = state._with_mega_request_history(dummy, metadata, False)
    assert result.mega_first_verify.tolist() == [0, 0]
    torch.testing.assert_close(tracker.context, before[0])
    torch.testing.assert_close(tracker.first, before[1])
    state._mega_request_tracker = None
    assert state._with_mega_request_history(dummy, metadata, False) is metadata


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_request_tracker_is_allocated_only_when_enabled(runtime, enabled):
    class UpstreamState:
        def __init__(self, config):
            self.vllm_config = config
            self.max_num_reqs = 4
            self.device = torch.device("cpu")

    runtime["UpstreamState"] = UpstreamState
    contracts._definitions(
        "vllm_ascend/worker/v2/model_states/mamba_hybrid.py",
        {"AscendMambaHybridModelState"},
        runtime,
        methods={"__init__"},
        bases="UpstreamState",
    )
    state = runtime["AscendMambaHybridModelState"](SimpleNamespace(additional_config={"enable_mega_kda": enabled}))
    if enabled:
        assert state._mega_request_tracker.context.shape == (5,)
    else:
        assert state._mega_request_tracker is None
