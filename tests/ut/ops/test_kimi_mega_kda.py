# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for MegaKDA; external kernels/collectives are mocked.

Extract production definitions as the PP/SP contract tests do, so these tests
also run without installing the NPU-only vLLM runtime. No kernel arithmetic is
reimplemented here; numerical/cache equivalence requires the NPU test.
"""

import __future__

import ast
import math
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]


def _definitions(path, names, namespace, *, methods=None, bases=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(nodes) == len(names)
    for node in nodes:
        if isinstance(node, ast.ClassDef) and methods is not None:
            node.body = [item for item in node.body if getattr(item, "name", None) in methods]
            node.bases = [ast.Name(id=bases, ctx=ast.Load())]
            node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, path, "exec", flags=__future__.annotations.compiler_flag), namespace)


class _Unquantized:
    pass


class _OutputOp:
    def __init__(self):
        self.reduce_projected_output = MagicMock(side_effect=lambda x: x[: x.shape[0] // 2])


@pytest.fixture
def runtime():
    ns = {
        "__name__": __name__,
        "torch": torch,
        "nn": nn,
        "Module": nn.Module,
        "dataclass": dataclass,
        "Parameter": Parameter,
        "signature": signature,
        "import_module": MagicMock(),
        "torch_npu": SimpleNamespace(npu_format_cast=lambda x, fmt: x),
        "ACL_FORMAT_FRACTAL_ND": 2,
        "MEGA_KDA_ABI_POSITIONAL_ARGS": 20,
        "MEGA_KDA_CACHE_ALIGNMENT_BYTES": 16,
        "NULL_BLOCK_ID": 0,
    }
    _definitions(
        "vllm_ascend/ops/kimi_mega_kda.py",
        {
            "MegaKDAWeights",
            "MegaKDAFirstVerifyTracker",
            "MegaKDACacheWorkspace",
            "MegaKDAPagedBatch",
            "MegaKDAPagedCache",
            "_mega_cache_layout",
            "load_mega_kda_operator",
            "_as_int32_contiguous",
            "ensure_mega_kda_bf16",
            "ensure_mega_kda_nd",
            "mega_kda_nz_operand",
            "run_mega_recurrent_kda",
        },
        ns,
    )
    ns.update(
        KimiK3DeltaAttention=nn.Module,
        GDNAttentionMetadata=SimpleNamespace,
        KimiOProjMMReduceScatterOp=_OutputOp,
        UnquantizedLinearMethod=_Unquantized,
        CUDAGraphMode=SimpleNamespace(NONE="none"),
        get_forward_context=MagicMock(),
        eager_break_during_capture=lambda f: f,
        is_breakable_cudagraph_enabled=lambda: False,
        logger=MagicMock(),
        _PACKED_CONV_WEIGHT_NAME="ascend_conv1d_weight",
        _MEGA_KDA_MAX_BATCH_MULTIPLIER=32,
        _MEGA_KDA_CACHE_ALIGNMENT_BYTES=16,
        MEGA_KDA_NZ_FORMAT=29,
        tensor_model_parallel_all_reduce=MagicMock(side_effect=lambda x: x + 1),
        wait_for_kv_layer_from_connector=MagicMock(),
        record_attention_compute_start=MagicMock(),
        maybe_save_kv_layer_to_connector=MagicMock(),
    )
    methods = {
        "forward",
        "_plan_mega_kda",
        "_mega_cache_supported",
        "_build_mega_weights",
        "_prepare_mega_weights",
        "_current_mega_weight_signature",
        "_forward_mega_kda",
        "_execute_mega_kda",
        "_allocate_mega_output",
    }
    _definitions("vllm_ascend/ops/kimi_kda.py", {"_restore_token_rows", "_MegaKDAPlan"}, ns)
    _definitions("vllm_ascend/ops/kimi_kda.py", {"AscendKimiK3DeltaAttention"}, ns, methods=methods, bases="Module")
    return ns


def _linear(weight):
    return SimpleNamespace(weight=nn.Parameter(weight, requires_grad=False), quant_method=_Unquantized(), bias=None)


def _attention(ns, *, batch=4, spec_len=8):
    layer = ns["AscendKimiK3DeltaAttention"]()
    layer.prefix = "model.layers.0.self_attn"
    layer.hidden_size = 32
    layer.head_dim = 4
    layer.local_num_heads = 2
    layer.local_projection_size = 8
    layer.conv_size = 4
    layer.num_spec = spec_len - 1
    layer.tp_size = 2
    layer.uses_mixed_projection = False
    layer.use_mega_kda = True
    layer._mega_operator = MagicMock()
    layer._mega_weights = None
    layer._mega_weight_signature = None
    layer.in_proj_qkvgfab = _linear(torch.randn(38, 32, dtype=torch.bfloat16))
    layer.f_b_proj = _linear(torch.randn(8, 4, dtype=torch.bfloat16))
    layer.o_proj = _linear(torch.randn(32, 8, dtype=torch.bfloat16))
    layer.o_proj.reduce_results = True
    layer.o_proj.custom_op = None
    layer.o_norm = SimpleNamespace(weight=nn.Parameter(torch.ones(4)), eps=1e-6)
    layer.conv1d = SimpleNamespace(bias=None)
    layer.A_log = torch.zeros(2)
    layer.dt_bias = torch.zeros(8)
    layer.gate_lower_bound = -5.0
    layer.register_parameter(
        "ascend_conv1d_weight", nn.Parameter(torch.randn(4, 24, dtype=torch.bfloat16), requires_grad=False)
    )
    layer.kv_cache = (
        torch.zeros(batch * spec_len + 1, 3 + spec_len - 1, 24, dtype=torch.bfloat16),
        torch.zeros(batch * spec_len + 1, 2, 4, 4),
    )
    layer._forward_default = MagicMock(return_value=torch.full((batch * spec_len, 32), -1.0))
    assert layer._prepare_mega_weights()
    return layer


def _metadata(batch=4, spec_len=8):
    return SimpleNamespace(
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=batch,
        num_actual_tokens=batch * spec_len,
        mega_has_folded_prefills=False,
        mega_query_lens=(spec_len,) * batch,
        mega_first_verify=torch.ones(batch, dtype=torch.int32),
        spec_decode_metadata=SimpleNamespace(
            spec_causal_conv1d=SimpleNamespace(query_start_loc=torch.arange(batch + 1) * spec_len)
        ),
        spec_state_indices_tensor=torch.arange(1, batch * spec_len + 1).reshape(batch, spec_len),
        num_accepted_tokens=torch.full((batch,), 2, dtype=torch.int32),
    )


def _plan(ns, layer, metadata, *, rows=None):
    hidden = torch.zeros(rows or metadata.num_actual_tokens, 32, dtype=torch.bfloat16)
    ns["get_forward_context"].return_value = SimpleNamespace(
        attn_metadata={layer.prefix: metadata}, cudagraph_runtime_mode="none"
    )
    return layer._plan_mega_kda(hidden), hidden


def test_weights_slice_packed_projection_and_preserve_source(runtime):
    layer = _attention(runtime)
    packed = layer.in_proj_qkvgfab.weight
    before = packed.clone()
    weights = layer._mega_weights
    for result, source in (
        (weights.qkv, packed[:24]),
        (weights.output_gate, packed[24:32]),
        (weights.f_a, packed[32:36]),
        (weights.beta, packed[36:38]),
        (weights.f_b, layer.f_b_proj.weight),
        (weights.o_proj, layer.o_proj.weight),
    ):
        torch.testing.assert_close(result, source)
        assert result.storage_offset() == 0
        assert result.data_ptr() != source.data_ptr()
    assert weights.conv.shape == (3, 2, 4, 4)
    assert weights.o_norm.dtype == torch.float32
    torch.testing.assert_close(packed, before)


def test_mixed_projection_uses_original_factors(runtime):
    layer = _attention(runtime)
    layer.uses_mixed_projection = True
    layer.in_proj_qkvgfab = _linear(torch.randn(24, 32, dtype=torch.bfloat16))
    layer.fused_bfg_proj = _linear(torch.randn(18, 32, dtype=torch.bfloat16))
    layer.fused_bfg_proj.f_a_weight = nn.Parameter(torch.randn(4, 32, dtype=torch.bfloat16))
    layer.fused_bfg_proj.f_b_weight = nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    assert layer._prepare_mega_weights()
    weights = layer._mega_weights
    torch.testing.assert_close(weights.beta, layer.fused_bfg_proj.weight[:2])
    torch.testing.assert_close(weights.output_gate, layer.fused_bfg_proj.weight[10:])
    torch.testing.assert_close(weights.f_a, layer.fused_bfg_proj.f_a_weight)
    torch.testing.assert_close(weights.f_b, layer.fused_bfg_proj.f_b_weight)


@pytest.mark.parametrize("operand", ["in_proj_qkvgfab", "o_proj", "f_b_proj"])
@pytest.mark.parametrize("mismatch", ["dtype", "quantization", "bias"])
def test_unsupported_weights_keep_default_path(runtime, operand, mismatch):
    layer = _attention(runtime)
    projection = getattr(layer, operand)
    if mismatch == "dtype":
        projection.weight = nn.Parameter(projection.weight.float())
    elif mismatch == "quantization":
        projection.quant_method = object()
    else:
        projection.bias = torch.ones(1)
    assert not layer._prepare_mega_weights()
    plan, hidden = _plan(runtime, layer, _metadata())
    assert plan is None
    result = layer.forward(hidden, torch.zeros(hidden.shape[0]))
    assert result is layer._forward_default.return_value


def test_first_verify_has_separate_conv_and_recurrent_counts(runtime):
    layer = _attention(runtime)
    metadata = _metadata()
    metadata.mega_first_verify[2:] = 0
    plan, _ = _plan(runtime, layer, metadata)
    assert plan.conv_num_accepted_tokens.tolist() == [8, 8, 2, 2]
    assert plan.num_accepted_tokens.tolist() == [2, 2, 2, 2]
    assert plan.ssm_state_indices.tolist() == metadata.spec_state_indices_tensor.flatten().tolist()
    assert plan.conv_state_block_ids.tolist() == [1, 9, 17, 25]


@pytest.mark.parametrize("batch", [2, 4, 8, 16, 64])
@pytest.mark.parametrize("spec_len", [1, 4, 8])
def test_planner_accepts_rectangular_live_batches(runtime, batch, spec_len):
    layer = _attention(runtime, batch=batch, spec_len=spec_len)
    plan, _ = _plan(runtime, layer, _metadata(batch, spec_len))
    assert plan is not None
    assert (plan.batch, plan.spec_len) == (batch, spec_len)


@pytest.mark.parametrize("batch", [0, 1, 3, 65, 66])
def test_planner_rejects_unaligned_or_oversized_batch(runtime, batch):
    layer = _attention(runtime)
    plan, _ = _plan(runtime, layer, _metadata(batch))
    assert plan is None


@pytest.mark.parametrize(
    "changes",
    [
        {"num_prefills": 1},
        {"num_decodes": 1},
        {"mega_has_folded_prefills": True},
        {"mega_first_verify": None},
        {"mega_first_verify": torch.ones(3, dtype=torch.int32)},
        {"mega_query_lens": (8, 8, 0, 0)},
        {"mega_query_lens": (7, 9, 8, 8)},
        {"spec_state_indices_tensor": torch.ones(4, 7, dtype=torch.int32)},
        {"spec_state_indices_tensor": torch.ones(4, dtype=torch.int32)},
    ],
)
def test_invalid_or_unsupported_metadata_falls_back(runtime, changes):
    layer = _attention(runtime)
    metadata = _metadata()
    vars(metadata).update(changes)
    plan, _ = _plan(runtime, layer, metadata)
    assert plan is None


@pytest.mark.parametrize("graph_mode", ["none", "full", "piecewise"])
def test_graph_modes_are_not_an_enablement_gate(runtime, graph_mode):
    layer = _attention(runtime)
    _, hidden = _plan(runtime, layer, _metadata())
    context = runtime["get_forward_context"].return_value
    context.cudagraph_runtime_mode = graph_mode
    assert layer._plan_mega_kda(hidden) is not None
    context.attn_metadata = None
    assert layer._plan_mega_kda(hidden) is None


@pytest.mark.parametrize("spec_len", [1, 4])
def test_runtime_query_length_must_match_source_pr_contract(runtime, spec_len):
    layer = _attention(runtime)
    plan, _ = _plan(runtime, layer, _metadata(spec_len=spec_len))
    assert plan is None


def test_first_verify_buffer_may_include_unused_capacity(runtime):
    layer = _attention(runtime)
    metadata = _metadata()
    metadata.mega_first_verify = torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.int32)
    plan, _ = _plan(runtime, layer, metadata)
    assert plan is not None
    assert plan.conv_num_accepted_tokens.tolist() == [8, 2, 8, 2]


def test_dispatch_uses_runtime_shape_instead_of_reported_token_count(runtime):
    layer = _attention(runtime)
    metadata = _metadata()
    metadata.num_actual_tokens = 31
    plan, _ = _plan(runtime, layer, metadata, rows=32)
    assert plan is not None


@pytest.mark.parametrize("rows", [31, 33, 34, 64])
def test_hidden_rows_must_match_the_state_table(runtime, rows):
    layer = _attention(runtime)
    plan, _ = _plan(runtime, layer, _metadata(), rows=rows)
    assert plan is None


def test_graph_padding_uses_source_pr_runtime_batch(runtime):
    layer = _attention(runtime)
    metadata = _metadata()
    metadata.num_spec_decodes = 3
    metadata.num_actual_tokens = 24
    metadata.mega_query_lens = (8, 8, 8)
    metadata.spec_state_indices_tensor[-1].zero_()
    metadata.num_accepted_tokens[-1] = 1
    metadata.mega_first_verify[-1] = 0
    plan, _ = _plan(runtime, layer, metadata, rows=32)
    assert plan is not None and plan.batch == 4
    assert plan.conv_state_block_ids.tolist() == [1, 9, 17, 0]
    assert plan.conv_num_accepted_tokens.tolist() == [8, 8, 8, 1]
    assert plan.num_accepted_tokens.tolist() == [2, 2, 2, 1]


def test_breakable_segment_writes_to_the_same_output_buffer(runtime):
    layer = _attention(runtime)
    plan, hidden = _plan(runtime, layer, _metadata())
    output = layer._allocate_mega_output(hidden)
    pointer = output.data_ptr()
    for value in (1, 2):
        layer._mega_operator.return_value = torch.full_like(hidden, value)
        assert layer._forward_mega_kda(hidden, torch.arange(32), plan, output) is None
        assert output.data_ptr() == pointer
        torch.testing.assert_close(output, torch.full_like(hidden, value + 1))


def test_non_breakable_forward_returns_operator_storage_without_output_copy(runtime):
    layer = _attention(runtime)
    _, hidden = _plan(runtime, layer, _metadata())
    layer.tp_size = 1
    result = torch.randn_like(hidden)
    layer._mega_operator.return_value = result
    layer._allocate_mega_output = MagicMock(side_effect=AssertionError("redundant output allocation"))
    layer._forward_mega_kda = MagicMock(side_effect=AssertionError("redundant output copy"))
    assert layer.forward(hidden, torch.arange(32)).data_ptr() == result.data_ptr()


def test_planner_reuses_group_conv_acceptance_without_per_layer_device_ops(runtime, monkeypatch):
    layer = _attention(runtime)
    metadata = _metadata()
    metadata.mega_conv_accepted_tokens = torch.tensor([8, 2, 8, 2], dtype=torch.int32)
    hidden = torch.zeros(32, 32, dtype=torch.bfloat16)
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata={layer.prefix: metadata})
    monkeypatch.setattr(torch, "where", MagicMock(side_effect=AssertionError("per-layer Select")))
    monkeypatch.setattr(torch, "full_like", MagicMock(side_effect=AssertionError("per-layer Fill")))
    for _ in range(2):
        plan = layer._plan_mega_kda(hidden)
        assert plan.conv_num_accepted_tokens.data_ptr() == metadata.mega_conv_accepted_tokens.data_ptr()
    metadata.mega_conv_accepted_tokens = metadata.mega_conv_accepted_tokens[:3]
    assert layer._plan_mega_kda(hidden) is None


def test_compiler_boundary_resolves_current_layer_and_metadata(runtime):
    _definitions("vllm_ascend/ops/kimi_kda.py", {"_kimi_kda_with_output", "_kimi_kda_with_output_fake"}, runtime)
    layer = _attention(runtime)
    metadata = _metadata()
    _, hidden = _plan(runtime, layer, metadata)
    context = runtime["get_forward_context"].return_value
    context.no_compile_layers = {layer.prefix: layer}
    output = layer._allocate_mega_output(hidden)
    layer._mega_operator.return_value = torch.zeros_like(hidden)
    runtime["_kimi_kda_with_output"](hidden, torch.arange(32), output, layer.prefix)
    layer._mega_operator.assert_called_once()
    layer._forward_default.assert_not_called()
    metadata.num_prefills = 1
    runtime["_kimi_kda_with_output"](hidden, torch.arange(32), output, layer.prefix)
    layer._forward_default.assert_called_once()
    torch.testing.assert_close(output, layer._forward_default.return_value.to(output.dtype))
    runtime["get_forward_context"].reset_mock()
    runtime["_kimi_kda_with_output_fake"](hidden, torch.arange(32), output, layer.prefix)
    runtime["get_forward_context"].assert_not_called()


def test_functional_fake_requires_only_shape_and_no_forward_context(runtime):
    _definitions("vllm_ascend/ops/kimi_kda.py", {"_kimi_kda_fake"}, runtime)
    hidden = torch.empty(32, 32, dtype=torch.bfloat16)
    result = runtime["_kimi_kda_fake"](hidden, torch.arange(32), "target", 16)
    assert result.shape == (16, 32) and result.dtype == hidden.dtype
    runtime["get_forward_context"].assert_not_called()


@pytest.mark.parametrize("breakable", [False, True])
def test_torch_compile_does_not_freeze_profile_time_routing(runtime, breakable):
    _definitions(
        "vllm_ascend/ops/kimi_kda.py",
        {"_kimi_kda", "_kimi_kda_fake", "_kimi_kda_with_output", "_kimi_kda_with_output_fake"},
        runtime,
    )
    runtime["is_breakable_cudagraph_enabled"] = lambda: breakable
    # Register the production schema with CPU dispatch for this isolated test.
    # This exercises real Dynamo/FakeTensor without pretending to execute NPU.
    library = torch.library.Library("vllm", "FRAGMENT")
    library.define(
        "ascend_kimi_kda_with_output(Tensor hidden_states, Tensor positions, Tensor(a!) output, str layer_name) -> ()"
    )
    library.impl("ascend_kimi_kda_with_output", runtime["_kimi_kda_with_output"], "CPU")
    torch.library.register_fake("vllm::ascend_kimi_kda_with_output", runtime["_kimi_kda_with_output_fake"], lib=library)
    library.define(
        "ascend_kimi_kda(Tensor hidden_states, Tensor positions, str layer_name, SymInt output_rows) -> Tensor"
    )
    library.impl("ascend_kimi_kda", runtime["_kimi_kda"], "CPU")
    torch.library.register_fake("vllm::ascend_kimi_kda", runtime["_kimi_kda_fake"], lib=library)
    layer = _attention(runtime)
    _, hidden = _plan(runtime, layer, _metadata())
    context = runtime["get_forward_context"].return_value
    context.no_compile_layers = {layer.prefix: layer}
    layer._mega_operator.return_value = torch.zeros_like(hidden)
    layer._forward_default.return_value = layer._forward_default.return_value.to(hidden.dtype)
    try:
        compiled = torch.compile(layer, backend="eager", fullgraph=True)
        context.attn_metadata = None
        torch.testing.assert_close(compiled(hidden, torch.arange(32)), layer._forward_default.return_value)
        context.attn_metadata = {layer.prefix: _metadata()}
        torch.testing.assert_close(compiled(hidden, torch.arange(32)), torch.ones_like(hidden))
        assert layer._mega_operator.call_count == 1
    finally:
        torch._dynamo.reset()
        library._destroy()


@pytest.mark.parametrize("conv_size", [3, 4, 5])
@pytest.mark.parametrize("spec_len", [1, 4, 8])
def test_cache_capacity_and_page_gaps_are_checked_before_launch(runtime, conv_size, spec_len):
    layer = _attention(runtime, spec_len=spec_len)
    layer.conv_size = conv_size
    required = conv_size + spec_len - 2
    recurrent = layer.kv_cache[1]
    layer.kv_cache = (torch.zeros(33, required - 1, 24, dtype=torch.bfloat16), recurrent)
    assert not layer._mega_cache_supported(spec_len)
    layer.kv_cache = (torch.zeros(33, required, 24, dtype=torch.bfloat16), recurrent)
    assert layer._mega_cache_supported(spec_len)
    # Same logical payload, different physical page stride: external ABI is
    # not allowed to reinterpret the backing allocation as dense pages.
    layer.kv_cache = (torch.zeros(66, required, 24, dtype=torch.bfloat16)[::2], recurrent)
    assert not layer._mega_cache_supported(spec_len)


@pytest.mark.parametrize("reduction", ["all_reduce", "outer_reduce_scatter", "custom_reduce_scatter", "tp1"])
def test_projection_and_collective_contract(runtime, reduction):
    layer = _attention(runtime)
    metadata = _metadata()
    plan, hidden = _plan(runtime, layer, metadata, rows=32)
    assert plan is not None
    layer._mega_operator.return_value = torch.zeros(4, 8, 32, dtype=torch.bfloat16)
    if reduction == "tp1":
        layer.tp_size = 1
    if reduction in ("outer_reduce_scatter", "custom_reduce_scatter"):
        layer.o_proj.reduce_results = False
    if reduction == "custom_reduce_scatter":
        # Installed AFTER KDA construction, as in the real decoder.
        layer.o_proj.custom_op = _OutputOp()
    output = layer._execute_mega_kda(hidden, torch.arange(32), plan)
    assert layer._mega_operator.call_count == 1
    args = layer._mega_operator.call_args.args
    assert len(args) == 20
    assert args[0].shape == (4, 8, 32)
    assert args[-1] == 1e-6
    assert args[9] is layer.kv_cache[0] and args[10] is layer.kv_cache[1]
    assert all(x.dtype == torch.int32 and x.is_contiguous() for x in args[11:15])
    runtime["wait_for_kv_layer_from_connector"].assert_called_once_with(layer.prefix)
    runtime["record_attention_compute_start"].assert_called_once()
    runtime["maybe_save_kv_layer_to_connector"].assert_called_once()
    if reduction == "all_reduce":
        runtime["tensor_model_parallel_all_reduce"].assert_called_once()
        torch.testing.assert_close(output, torch.ones(32, 32, dtype=torch.bfloat16))
    else:
        runtime["tensor_model_parallel_all_reduce"].assert_not_called()
        assert output.shape[0] == (16 if reduction == "custom_reduce_scatter" else 32)
    layer._forward_default.assert_not_called()


def test_unknown_custom_output_contract_uses_default(runtime):
    layer = _attention(runtime)
    layer.o_proj.custom_op = object()
    plan, _ = _plan(runtime, layer, _metadata())
    assert plan is None


def test_operator_failure_is_never_retried_with_default_cache(runtime):
    layer = _attention(runtime)
    plan, hidden = _plan(runtime, layer, _metadata())
    layer._mega_operator.side_effect = RuntimeError("kernel failed after state mutation")
    with pytest.raises(RuntimeError, match="state mutation"):
        layer._execute_mega_kda(hidden, torch.arange(32), plan)
    layer._forward_default.assert_not_called()


def test_reload_requires_rebuilding_derived_weights(runtime):
    layer = _attention(runtime)
    plan, hidden = _plan(runtime, layer, _metadata())
    with torch.no_grad():
        layer.f_b_proj.weight.add_(1)
    with pytest.raises(RuntimeError, match="weights changed"):
        layer._execute_mega_kda(hidden, torch.arange(32), plan)
    layer._mega_operator.assert_not_called()
    assert layer._prepare_mega_weights()
    torch.testing.assert_close(layer._mega_weights.f_b, layer.f_b_proj.weight)


def test_tracker_prefill_verify_reorder_single_token_reuse_and_repeat(runtime):
    tracker = runtime["MegaKDAFirstVerifyTracker"](16, torch.device("cpu"))
    slots = torch.tensor([2, 5])
    spec = torch.ones(2, dtype=torch.bool)
    prefill = torch.zeros(2, dtype=torch.bool)
    context = torch.tensor([10, 20])
    assert tracker.update(slots, context, prefill, spec).tolist() == [1, 1]
    # Rebuilding metadata must not consume a Verify round.
    assert tracker.update(slots, context, prefill, spec).tolist() == [1, 1]
    assert tracker.update(slots.flip(0), torch.tensor([22, 12]), prefill, spec).tolist() == [0, 0]
    # A one-token prompt in reused slot 2 resets its epoch, independently of
    # the query length and without disturbing the unscheduled slot 5.
    tracker.update(torch.tensor([2]), torch.tensor([0]), torch.tensor([True]), torch.tensor([False]))
    assert tracker.update(slots, torch.tensor([1, 24]), prefill, spec).tolist() == [1, 0]
    assert tracker.update(
        torch.tensor([0, -1, 100]), torch.zeros(3, dtype=torch.int64), torch.zeros(3).bool(), torch.ones(3).bool()
    ).tolist() == [0, 0, 0]


def test_tracker_no_draft_decode_then_verify_uses_previous_accepted_count(runtime):
    tracker = runtime["MegaKDAFirstVerifyTracker"](16, torch.device("cpu"))
    slots = torch.tensor([2, 5])
    prefill = torch.ones(2, dtype=torch.bool)
    no_spec = torch.zeros(2, dtype=torch.bool)
    tracker.update(slots, torch.zeros(2, dtype=torch.int64), prefill, no_spec)
    assert tracker.update(slots, torch.tensor([10, 20]), ~prefill, no_spec).tolist() == [0, 0]
    # A dynamic no-draft step must consume first-verify even at the same context.
    assert tracker.update(slots, torch.tensor([10, 20]), ~prefill, ~no_spec).tolist() == [0, 0]
    assert tracker.update(slots.flip(0), torch.tensor([21, 11]), ~prefill, ~no_spec).tolist() == [0, 0]
    # Reusing a slot through prefill still resets it for a new request.
    tracker.update(slots[:1], torch.tensor([0]), prefill[:1], no_spec[:1])
    assert tracker.update(slots, torch.tensor([1, 22]), ~prefill, ~no_spec).tolist() == [1, 0]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("cache_mode", ["none", "align", "all"])
@pytest.mark.parametrize("eager", [False, True])
def test_constructor_preserves_shared_configuration(runtime, enabled, cache_mode, eager):
    class Upstream(nn.Module):
        def __init__(self, config, vllm_config, prefix):
            super().__init__()
            self.o_norm = SimpleNamespace(eps=1e-5)
            self.conv_size = 4
            self.local_projection_size = 8
            self.model_config = SimpleNamespace(dtype=torch.bfloat16)
            self.conv1d = SimpleNamespace(
                weight=torch.zeros(24, 1, 4),
                quant_method=SimpleNamespace(process_weights_after_loading=lambda: None),
            )

    runtime.update(
        KimiK3DeltaAttention=Upstream,
        validate_additional_config_bool=lambda value, path: value,
        wraps=lambda f: lambda g: g,
    )
    runtime["import_module"].return_value = SimpleNamespace(mega_recurrent_kda=MagicMock())
    _definitions(
        "vllm_ascend/ops/kimi_kda.py",
        {"AscendKimiK3DeltaAttention"},
        runtime,
        methods={"__init__"},
        bases="KimiK3DeltaAttention",
    )
    config = SimpleNamespace(
        additional_config={"enable_mega_kda": enabled},
        model_config=SimpleNamespace(enforce_eager=eager, enable_sleep_mode=True),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2, decode_context_parallel_size=2),
        cache_config=SimpleNamespace(mamba_cache_mode=cache_mode, enable_prefix_caching=True, use_kda_recoverssm=True),
        lora_config=object(),
        kv_transfer_config=object(),
    )
    before = repr(config)
    layer = runtime["AscendKimiK3DeltaAttention"](SimpleNamespace(rms_norm_eps=1e-6), config, "")
    assert layer.use_mega_kda is enabled
    assert repr(config) == before
    assert not hasattr(layer, "_mega_runtime_supported")
    assert runtime["import_module"].call_count == int(enabled)


def test_operator_loader_rejects_19_argument_abi(runtime):
    namespace = {}
    exec("def old(" + ",".join(f"x{i}" for i in range(19)) + "): pass", namespace)
    runtime["import_module"].return_value = SimpleNamespace(mega_recurrent_kda=namespace["old"])
    with pytest.raises(RuntimeError, match="19-argument ND"):
        runtime["load_mega_kda_operator"]()
    original = ImportError("missing external package")
    runtime["import_module"].side_effect = original
    with pytest.raises(ImportError, match="20-argument") as error:
        runtime["load_mega_kda_operator"]()
    assert error.value.__cause__ is original


def test_config_validation_is_kimi_only(runtime):
    _definitions(
        "vllm_ascend/ascend_config.py", {"AscendConfig"}, runtime, methods={"_validate_mega_kda"}, bases="object"
    )
    config = runtime["AscendConfig"]()
    config.enable_mega_kda = False
    config._validate_mega_kda(None)
    config.enable_mega_kda = True
    hf = SimpleNamespace(model_type="kimi_linear", linear_attn_config={"num_heads": 96, "use_full_rank_gate": True})
    vc = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=hf))
    config._validate_mega_kda(vc)
    hf.model_type = "glm5next"
    with pytest.raises(ValueError, match="Kimi K3"):
        config._validate_mega_kda(vc)
    hf.model_type = "kimi_linear"
    hf.linear_attn_config["use_full_rank_gate"] = False
    with pytest.raises(ValueError, match="use_full_rank_gate"):
        config._validate_mega_kda(vc)


def test_contiguous_cache_slices_keep_their_original_storage(runtime):
    layer = _attention(runtime)
    conv, recurrent = layer.kv_cache
    conv_backing = torch.zeros(conv.numel() + 32, dtype=conv.dtype)
    state_backing = torch.zeros(recurrent.numel() + 32, dtype=recurrent.dtype)
    layer.kv_cache = (conv_backing[32:].view_as(conv), state_backing[32:].view_as(recurrent))
    assert layer._mega_cache_supported(8)
    assert all(state.storage_offset() == 32 for state in layer.kv_cache)
    plan, hidden = _plan(runtime, layer, _metadata())
    layer._mega_operator.return_value = torch.zeros_like(hidden)
    layer._execute_mega_kda(hidden, torch.arange(32), plan)
    args = layer._mega_operator.call_args.args
    assert args[9] is layer.kv_cache[0] and args[10] is layer.kv_cache[1]


@pytest.mark.parametrize("operand", [0, 1])
def test_unaligned_contiguous_cache_falls_back_before_mutation(runtime, operand):
    layer = _attention(runtime)
    cache = list(layer.kv_cache)
    state = cache[operand]
    backing = torch.zeros(state.numel() + 1, dtype=state.dtype)
    cache[operand] = backing[1:].view_as(state)
    assert cache[operand].is_contiguous() and cache[operand].data_ptr() % 16 != 0
    layer.kv_cache = tuple(cache)
    assert not layer._mega_cache_supported(8)
    plan, _ = _plan(runtime, layer, _metadata())
    assert plan is None
    layer._mega_operator.assert_not_called()


@pytest.mark.parametrize("graph_batch", [2, 4])
def test_builder_uses_bound_capacity_and_resets_folded_prefill(runtime, graph_batch):
    class BaseBuilder:
        def __init__(self, spec, names, config, device):
            self.vllm_config = config
            self.kv_cache_spec = spec
            self.spec_sequence_indices = torch.arange(graph_batch)
            self.use_full_cuda_graph = graph_batch > 2
            self.use_spec_decode = True
            self.num_spec = 7

        def build(self, *args):
            return SimpleNamespace(
                num_spec_decodes=2,
                spec_state_indices_tensor=torch.zeros(graph_batch, 8),
                num_accepted_tokens=torch.full((graph_batch,), 2, dtype=torch.int32),
            )

    runtime.update(
        AscendGDNHostMetadataBuilder=BaseBuilder,
        mamba_get_block_table_tensor=lambda table, *args: table,
    )
    _definitions("vllm_ascend/ops/gdn_attn_builder.py", {"_remove_spec_graph_padding_queries"}, runtime)
    _definitions(
        "vllm_ascend/ops/gdn_attn_builder.py",
        {"AscendMegaKDAHostMetadataBuilder"},
        runtime,
        methods={"__init__", "_ensure_mega_tracker", "build"},
        bases="AscendGDNHostMetadataBuilder",
    )
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context={}),
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    builder = runtime["AscendMegaKDAHostMetadataBuilder"](None, ["target"], config, torch.device("cpu"))
    assert not builder._ensure_mega_tracker()
    config.compilation_config.static_forward_context["target"] = SimpleNamespace(
        kv_cache=(torch.zeros(16, 10, 24), torch.zeros(16, 2, 4, 4))
    )
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=16,
        query_start_loc_cpu=torch.tensor([0, 8, 16]),
        query_start_loc=torch.tensor([0, 8, 16]),
        block_table_tensor=torch.tensor([[2], [5]]),
        seq_lens=torch.tensor([18, 28]),
        seq_lens_cpu_upper_bound=None,
        is_prefilling=torch.tensor([False, False]),
    )
    common.unpadded = lambda *args: common
    common.compute_num_computed_tokens = lambda: common.seq_lens - torch.diff(common.query_start_loc)
    drafts = torch.tensor([7, 7])
    metadata = builder.build(0, common, None, drafts)
    assert metadata.mega_query_lens == (8, 8)
    assert not metadata.mega_has_folded_prefills
    assert metadata.mega_first_verify.tolist() == [1, 1] + [0] * (graph_batch - 2)
    first_pointer = metadata.mega_first_verify.data_ptr()
    assert builder._mega_tracker.context.numel() == 16
    assert builder.build(0, common, None, drafts).mega_first_verify.tolist() == [1, 1] + [0] * (graph_batch - 2)
    common.seq_lens += 2
    updated = builder.build(0, common, None, drafts)
    assert updated.mega_first_verify.tolist() == [0] * graph_batch
    assert updated.mega_first_verify.data_ptr() == first_pointer
    assert metadata.mega_first_verify.tolist() == [0] * graph_batch
    common.is_prefilling[0] = True
    common.query_start_loc_cpu = common.query_start_loc = torch.tensor([0, 1, 9])
    assert builder.build(0, common, None, drafts).mega_has_folded_prefills
    common.is_prefilling[0] = False
    common.seq_lens += 1
    assert builder.build(0, common, None, drafts).mega_first_verify.tolist() == [1, 0] + [0] * (graph_batch - 2)


def test_model_postload_visits_only_target_attention_and_mm_forwards(runtime):
    target_type = runtime["AscendKimiK3DeltaAttention"]
    target = _attention(runtime)
    target._prepare_mega_weights = MagicMock()
    other = nn.Linear(2, 2)
    other._prepare_mega_weights = MagicMock()
    runtime.update(
        UpstreamKimiLinearForCausalLM=nn.Module,
        AscendKimiK3DeltaAttentionBase=target_type,
        get_ascend_config=lambda: SimpleNamespace(enable_mega_kda=True),
    )
    _definitions(
        "vllm_ascend/models/kimi_k3.py",
        {"AscendKimiLinearForCausalLM"},
        runtime,
        methods={"process_weights_after_loading"},
        bases="Module",
    )
    model = runtime["AscendKimiLinearForCausalLM"]()
    model.model = nn.ModuleList([target, other])
    model.process_weights_after_loading()
    target._prepare_mega_weights.assert_called_once()
    other._prepare_mega_weights.assert_not_called()
    _definitions(
        "vllm_ascend/models/kimi_k3.py",
        {"AscendKimiK3ForConditionalGeneration"},
        runtime,
        methods={"process_weights_after_loading"},
        bases="Module",
    )
    multimodal = runtime["AscendKimiK3ForConditionalGeneration"]()
    multimodal.language_model = model
    multimodal.process_weights_after_loading()
    assert target._prepare_mega_weights.call_count == 2


def test_custom_op_reduces_already_projected_output_once(runtime):
    runtime["sp_reduce_scatter"] = MagicMock(side_effect=lambda x: x[::2])
    _definitions(
        "vllm_ascend/ops/linear_op.py",
        {"KimiOProjMMReduceScatterOp"},
        runtime,
        methods={"reduce_projected_output"},
        bases="object",
    )
    output = torch.randn(8, 32)
    result = runtime["KimiOProjMMReduceScatterOp"]().reduce_projected_output(output)
    runtime["sp_reduce_scatter"].assert_called_once_with(output)
    torch.testing.assert_close(result, output[::2])


def test_normal_decode_uses_one_accepted_token(runtime):
    layer = _attention(runtime, spec_len=1)
    metadata = _metadata(spec_len=1)
    metadata.num_spec_decodes = 0
    metadata.num_decodes = 4
    metadata.non_spec_query_start_loc = torch.arange(5)
    metadata.non_spec_state_indices_tensor = torch.arange(1, 5)
    metadata.mega_first_verify = None
    plan, _ = _plan(runtime, layer, metadata)
    assert plan is not None and plan.spec_len == 1
    assert plan.conv_num_accepted_tokens.tolist() == plan.num_accepted_tokens.tolist() == [1] * 4


@pytest.mark.parametrize("mega", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_runner_binds_dense_cache_only_for_eligible_layers(mega, padded):
    # Execute the production Mamba allocation branch without importing a worker.
    # The dense view must remain inside the existing allocation and preserve
    # the generic page layout whenever the layer did not opt in.
    path = "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "isinstance"
        and len(node.test.args) == 2
        and isinstance(node.test.args[1], ast.Name)
        and node.test.args[1].id == "MambaSpec"
        and any(isinstance(item, ast.Name) and item.id == "state_tensors" for item in ast.walk(node))
    )
    branch.orelse = []
    loop = ast.For(
        target=ast.Name(id="layer_name", ctx=ast.Store()),
        iter=ast.Name(id="names", ctx=ast.Load()),
        body=[branch],
        orelse=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))

    class MambaSpec:
        shapes = ((3, 24), (2, 4, 4))
        dtypes = (torch.bfloat16, torch.float32)
        page_size_bytes = 512

    raw = torch.zeros(4 * 512, dtype=torch.int8)
    strided_result = (torch.zeros(4, 3, 24), torch.zeros(4, 2, 4, 4))
    runner = SimpleNamespace(
        hybrid_with_attn_and_mamba=True,
        _adjust_kv_layout=MagicMock(return_value=strided_result),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "target": SimpleNamespace(_requires_dense_kda_cache=mega),
            }
        ),
    )
    ns = dict(
        torch=torch,
        math=math,
        logger=MagicMock(),
        MambaSpec=MambaSpec,
        current_kv_cache_spec=MambaSpec(),
        names=["target"],
        self=runner,
        kv_cache_config=SimpleNamespace(num_blocks=4),
        kv_cache_raw_tensors={"target": raw},
        kv_caches={},
        uses_padded_page_layout=padded,
        get_dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size(),
    )
    exec(compile(module, path, "exec"), ns)
    result = ns["kv_caches"]["target"]
    if padded or not mega:
        assert result is strided_result
        runner._adjust_kv_layout.assert_called_once()
    else:
        runner._adjust_kv_layout.assert_not_called()
        assert [tuple(t.shape) for t in result] == [(4, 3, 24), (4, 2, 4, 4)]
        assert all(t.is_contiguous() for t in result)
        assert result[0].untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
        assert result[1].storage_offset() * result[1].element_size() == result[0].numel() * 2
        result[1].fill_(1)
        assert result[0].count_nonzero() == 0


@pytest.mark.parametrize("dense", [False, True])
def test_v2_dense_cache_views_are_opt_in_and_stay_in_original_budget(runtime, dense):
    fallback = MagicMock(return_value="page-strided")
    runtime.update(
        math=math,
        get_dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size(),
        _adjust_dsv4_kv_layout=fallback,
    )
    _definitions("vllm_ascend/worker/v2/attn_utils.py", {"_reshape_mamba_kv_cache"}, runtime)
    raw = torch.zeros(3 * 32, dtype=torch.int8)
    spec = SimpleNamespace(
        page_size_padded=32, page_size_bytes=16, shapes=((4,), (2,)), dtypes=(torch.bfloat16, torch.float32)
    )
    result = runtime["_reshape_mamba_kv_cache"](raw, spec, dense=dense)
    if dense:
        fallback.assert_not_called()
        assert [t.shape for t in result] == [torch.Size([3, 4]), torch.Size([3, 2])]
        assert all(t.is_contiguous() for t in result)
        assert result[1].data_ptr() == raw.data_ptr() + 24
        result[1].fill_(1)
        assert result[0].count_nonzero() == 0
        assert raw[48:].count_nonzero() == 0
    else:
        fallback.assert_called_once()
        assert result == "page-strided"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("separate_view", [False, True])
def test_v2_shared_backing_never_changes_page_layout(shared, separate_view):
    path = "vllm_ascend/worker/v2/attn_utils.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Name)
        and node.test.func.id == "isinstance"
        and len(node.test.args) == 2
        and isinstance(node.test.args[1], ast.Name)
        and node.test.args[1].id == "MambaSpec"
        and any(isinstance(item, ast.Name) and item.id == "mamba_cache" for item in ast.walk(node))
    )
    branch.orelse = []
    loop = ast.For(
        target=ast.Name(id="layer_name", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant("target")], ctx=ast.Load()),
        body=[branch],
        orelse=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    raw = torch.zeros(64, dtype=torch.int8)
    caches = {"target": raw}
    if shared:
        caches["other"] = raw[:] if separate_view else raw
    reshape = MagicMock(return_value=[torch.zeros(4, 2)])
    spec = SimpleNamespace(page_size_bytes=16, page_size_padded=None)
    ns = dict(
        torch=torch,
        MambaSpec=SimpleNamespace,
        kv_cache_spec=spec,
        raw_cache=raw,
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "target": SimpleNamespace(_requires_dense_kda_cache=True),
                }
            )
        ),
        kv_cache_raw_tensors=caches,
        _reshape_mamba_kv_cache=reshape,
        kv_cache_config=SimpleNamespace(num_blocks=4),
        kv_caches={},
        logger=MagicMock(),
    )
    _definitions(path, {"_is_private_mamba_backing"}, ns)
    exec(compile(module, path, "exec"), ns)
    reshape.assert_called_once_with(raw, spec, dense=not shared)
