# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for MegaKDA; external kernels/collectives are mocked.

Extract production definitions as the PP/SP contract tests do, so these tests
also run without installing the NPU-only vLLM runtime. No kernel arithmetic is
reimplemented here; numerical/cache equivalence requires the NPU test.
"""

import __future__

import ast
from contextlib import nullcontext
from dataclasses import dataclass, replace
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
        "MEGA_KDA_REPLAY_ABI_REQUIRED_ARGS": (
            "hidden_states",
            "qkv_projection_weight",
            "decay_projection_a_weight",
            "decay_projection_b_weight",
            "beta_projection_weight",
            "output_gate_projection_weight",
            "output_norm_weight",
            "output_projection_weight",
            "conv1d_weight",
            "conv_state",
            "checkpoint_state",
            "replay_u",
            "replay_k",
            "replay_decay",
            "state_indices",
            "conv_num_accepted_tokens",
            "A_log",
            "dt_bias",
            "scale",
            "lower_bound",
            "rms_norm_eps",
            "cu_seqlens",
        ),
        "NULL_BLOCK_ID": 0,
    }
    _definitions(
        "vllm_ascend/ops/kimi_mega_kda.py",
        {
            "MegaKDAWeights",
            "MegaKDAFirstVerifyTracker",
            "load_mega_kda_operator",
            "_as_int32_contiguous",
            "ensure_mega_kda_bf16",
            "ensure_mega_kda_nd",
            "mega_kda_nz_operand",
            "run_mega_recurrent_kda",
            "run_commit_recurrent_kda",
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
        npu_stream_switch=lambda stream, *, enabled: nullcontext(),
    )
    methods = {
        "forward",
        "_plan_mega_kda",
        "_get_kda_forward_metadata",
        "_mega_cache_supported",
        "_build_mega_weights",
        "_prepare_mega_weights",
        "_release_default_kda_weights",
        "_released_mega_nd_params",
        "_forward_default",
        "_current_mega_weight_signature",
        "_forward_mega_kda",
        "_execute_mega_kda",
        "_prepare_mega_replayssm",
        "_allocate_mega_output",
    }
    _definitions("vllm_ascend/ops/kimi_kda.py", {"_restore_token_rows", "_MegaKDAPlan", "_MegaKDAReplayPlan"}, ns)
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
    layer.use_mega_kda_replayssm = True
    layer._mega_operator = MagicMock()
    layer._commit_operator = MagicMock()
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
    record = (layer.local_num_heads, spec_len, layer.head_dim)
    layer.kv_cache = (
        torch.zeros(batch * spec_len + 1, 3 + spec_len - 1, 24, dtype=torch.bfloat16),
        torch.zeros(batch * spec_len + 1, 2, 4, 4),
        torch.zeros((batch * spec_len + 1, *record), dtype=torch.float32),
        torch.zeros((batch * spec_len + 1, *record), dtype=torch.float16),
        torch.zeros((batch * spec_len + 1, *record), dtype=torch.float32),
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


def test_replay_abi_uses_page_tensors_without_max_batch(runtime):
    namespace = {}
    exec("def replay(**kwargs): return kwargs", namespace)
    operator = namespace["replay"]
    replay_u = torch.zeros(16, 2, 8, 4)
    replay_k = torch.zeros(16, 2, 8, 4, dtype=torch.float16)
    replay_decay = torch.zeros_like(replay_u)
    assert "max_batch" not in runtime["MEGA_KDA_REPLAY_ABI_REQUIRED_ARGS"]
    result = operator(
        replay_u=replay_u,
        replay_k=replay_k,
        replay_decay=replay_decay,
        state_indices=torch.tensor([3, 7], dtype=torch.int32),
    )
    assert result["replay_u"].shape[0] == 16


def _replay_layer(runtime, *, num_blocks=26):
    """Page-shaped five-state cache for the ReplaySSM commit/verify tests."""
    layer = _attention(runtime)
    layer.use_mega_kda_replayssm = True
    layer._commit_operator = MagicMock()
    record = (layer.local_num_heads, layer.num_spec + 1, layer.head_dim)
    layer.kv_cache = (
        torch.zeros(
            num_blocks,
            layer.conv_size + layer.num_spec - 1,
            3 * layer.local_projection_size,
            dtype=torch.bfloat16,
        ),
        torch.zeros(num_blocks, layer.local_num_heads, layer.head_dim, layer.head_dim),
        torch.zeros((num_blocks, *record), dtype=torch.float32),
        torch.zeros((num_blocks, *record), dtype=torch.float16),
        torch.zeros((num_blocks, *record), dtype=torch.float32),
    )
    return layer


def test_replay_commit_skips_the_first_verify_after_prefill(runtime):
    layer = _replay_layer(runtime)
    metadata = _metadata()
    # Request 0 enters its first Verify (prefill just finished), so its page may
    # still hold the previous occupant's records; the others have a pending one.
    metadata.mega_first_verify = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata={layer.prefix: metadata})

    plan = layer._prepare_mega_replayssm()
    assert plan is not None
    accepted = layer._commit_operator.call_args.kwargs["num_accepted_tokens"]
    assert accepted.tolist() == [0, 2, 2, 2]


def test_replay_commit_passes_only_the_live_requests(runtime):
    layer = _replay_layer(runtime)
    metadata = _metadata()
    # A FULL graph stages four request rows but only two are live; the padding
    # rows all carry NULL_BLOCK_ID and would violate the operator's
    # distinct-slots contract.
    metadata.num_spec_decodes = 2
    metadata.mega_first_verify = torch.zeros(4, dtype=torch.int32)
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata={layer.prefix: metadata})

    layer._prepare_mega_replayssm()
    call = layer._commit_operator.call_args.kwargs
    assert call["state_indices"].tolist() == [1, 9]
    assert call["num_accepted_tokens"].tolist() == [2, 2]


def test_replay_inputs_are_prepared_once_per_metadata(runtime, monkeypatch):
    layer = _replay_layer(runtime)
    metadata = _metadata()
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata={layer.prefix: metadata})

    first = layer._prepare_mega_replayssm()
    assert first is not None

    # The second KDA layer in the same forward must reuse the first layer's
    # request-level preparation.  Commit itself is still invoked for the layer.
    monkeypatch.setattr(torch, "where", MagicMock(side_effect=AssertionError("replay mask rebuilt per layer")))
    second = layer._prepare_mega_replayssm()
    assert second is not None
    assert first.state_indices.data_ptr() == second.state_indices.data_ptr()
    assert first.mega_state_indices.data_ptr() == second.mega_state_indices.data_ptr()
    assert first.accepted_tokens.data_ptr() == second.accepted_tokens.data_ptr()


@pytest.mark.parametrize(
    "entry,gather_input,fallback",
    [
        ("op", False, False),
        ("op", True, False),
        ("op", True, True),
        ("eager", False, False),
        ("output_op", False, False),
    ],
)
def test_commit_and_all_gather_event_order(runtime, monkeypatch, entry, gather_input, fallback):
    """Commit must run once after KV readiness and join before MegaKDA."""
    _definitions("vllm_ascend/ops/kimi_kda.py", {"_kimi_kda", "_kimi_kda_with_output"}, runtime)
    timeline = MagicMock()
    main, auxiliary = timeline.main, timeline.auxiliary
    main.record_event.return_value = "ready"
    auxiliary.record_event.return_value = "done"
    monkeypatch.setattr(torch, "npu", SimpleNamespace(current_stream=lambda: main), raising=False)
    runtime["_mega_kda_commit_stream"] = lambda: auxiliary
    runtime["npu_stream_switch"] = MagicMock(return_value=nullcontext())
    layer = _replay_layer(runtime)
    metadata = _metadata()
    metadata.mega_first_verify = torch.tensor([1, 0, 0, 0], dtype=torch.int32)
    _, hidden = _plan(runtime, layer, metadata)
    metadata.num_prefills = int(fallback)
    runtime["get_forward_context"].return_value.no_compile_layers = {layer.prefix: layer}
    layer._kimi_c8_metadata_prelaunch = timeline.prelaunch
    timeline.attach_mock(runtime["wait_for_kv_layer_from_connector"], "kv_ready")
    timeline.attach_mock(layer._commit_operator, "commit")
    timeline.attach_mock(layer._mega_operator, "mega")
    timeline.attach_mock(layer._forward_default, "fallback")
    runtime["sp_all_gather"] = timeline.gather
    timeline.gather.return_value = hidden
    layer._mega_operator.return_value = torch.zeros_like(hidden)
    if entry == "eager":
        layer.forward(hidden, torch.arange(32))
    elif entry == "output_op":
        runtime["_kimi_kda_with_output"](hidden, torch.arange(32), torch.empty_like(hidden), layer.prefix)
    else:
        runtime["_kimi_kda"](hidden[:16] if gather_input else hidden, torch.arange(32), layer.prefix, 32, gather_input)
    names = [entry[0] for entry in timeline.mock_calls]
    if fallback:
        assert names == ["prelaunch", "gather", "fallback"]
        runtime["npu_stream_switch"].assert_not_called()
        return
    if gather_input:
        assert names == [
            "prelaunch",
            "kv_ready",
            "main.record_event",
            "auxiliary.wait_event",
            "commit",
            "auxiliary.record_event",
            "gather",
            "main.wait_event",
            "mega",
        ]
        auxiliary.wait_event.assert_called_once_with("ready")
        main.wait_event.assert_called_once_with("done")
        runtime["npu_stream_switch"].assert_called_once_with(auxiliary, enabled=True)
    else:
        assert names == ["prelaunch", "kv_ready", "commit", "mega"]
        runtime["npu_stream_switch"].assert_called_once_with(None, enabled=False)
    timeline.prelaunch.assert_called_once_with({layer.prefix: metadata})
    assert layer._commit_operator.call_args.kwargs["num_accepted_tokens"].tolist() == [0, 2, 2, 2]
    assert layer._mega_operator.call_args.kwargs["conv_num_accepted_tokens"].tolist() == [1, 2, 2, 2]


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
        get_ascend_config=lambda: SimpleNamespace(enable_mega_kda_replayssm=True),
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


def test_pd_decode_worker_releases_nd_projections(runtime):
    layer = _attention(runtime)
    assert layer._prepare_mega_weights()
    assert not layer._mega_nd_released
    nz_pointer = layer._mega_weights.qkv.data_ptr()

    layer._mega_pd_decode_worker = True
    expected = (
        layer.in_proj_qkvgfab.weight.numel() * 2 + layer.o_proj.weight.numel() * 2 + layer.f_b_proj.weight.numel() * 2
    )
    released = layer._release_default_kda_weights()

    assert released == expected
    assert released > 0
    assert layer._mega_nd_released
    assert layer.in_proj_qkvgfab.weight.numel() == 0
    assert layer.o_proj.weight.numel() == 0
    assert layer.f_b_proj.weight.numel() == 0
    # The NZ operand set stays resident and untouched.
    assert layer._mega_weights.qkv.data_ptr() == nz_pointer
    assert layer._mega_weights.qkv.numel() > 0
    assert layer.get_parameter("ascend_conv1d_weight").numel() > 0
    assert layer.o_norm.weight.numel() > 0


def test_released_default_path_fails_loudly(runtime):
    layer = _attention(runtime)
    layer._mega_nd_released = True
    hidden = torch.randn(8, layer.hidden_size)
    # _attention() stubs the instance attribute; call the extracted method.
    forward_default = type(layer)._forward_default

    # After the post-profile release there is no dummy run left that may take
    # this path: every caller (real prefill/recompute, plan-None fallback)
    # must fail loudly instead of producing garbage from empty weights.
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata=None)
    with pytest.raises(RuntimeError, match="released"):
        forward_default(layer, hidden, torch.arange(8))
    runtime["get_forward_context"].return_value = SimpleNamespace(attn_metadata={layer.prefix: object()})
    with pytest.raises(RuntimeError, match="released"):
        forward_default(layer, hidden, torch.arange(8))


def test_release_mega_kda_nd_weights_targets_pd_decode_layers_only(runtime):
    _definitions("vllm_ascend/ops/kimi_kda.py", {"release_mega_kda_nd_weights"}, runtime)
    eligible = _attention(runtime)
    eligible._mega_pd_decode_worker = True
    plain = _attention(runtime)
    model = SimpleNamespace(modules=lambda: [eligible, plain])
    nz_pointer = eligible._mega_weights.qkv.data_ptr()

    released = runtime["release_mega_kda_nd_weights"](model)

    assert released > 0
    assert eligible._mega_nd_released
    assert eligible.in_proj_qkvgfab.weight.numel() == 0
    assert eligible._mega_weights.qkv.data_ptr() == nz_pointer
    # Non-PD-decode layers keep their ND weights and stay unreleased.
    assert not plain._mega_nd_released
    assert plain.in_proj_qkvgfab.weight.numel() > 0
    # A second call is a no-op (idempotent across repeated profile runs).
    assert runtime["release_mega_kda_nd_weights"](model) == 0


def test_pd_prefill_worker_skips_mega_operand_build(runtime):
    layer = _attention(runtime)
    assert layer._mega_weights is not None

    layer._mega_pd_prefill_worker = True
    assert not layer._prepare_mega_weights()
    assert layer._mega_weights is None
    # The ND projections stay resident: the prefill worker still needs them.
    assert layer.in_proj_qkvgfab.weight.numel() > 0
    assert layer.o_proj.weight.numel() > 0
    assert not layer._mega_nd_released


@pytest.mark.parametrize(
    "mode,enabled,producer,consumer,prepared,expected",
    [
        ("C8", True, True, False, False, 0),
        ("C16", True, True, False, False, 0),
        ("C8", True, False, True, True, 0),
        ("C16", True, False, True, True, 0),
        ("C8", True, False, True, False, 3),
        ("C16", True, False, True, False, 3),
        ("C8", False, False, True, True, 3),
        ("C16", False, False, True, True, 3),
        ("C8", True, False, False, True, 3),
        ("C16", True, True, True, True, 3),
    ],
)
def test_mega_spec_reservation_matches_existing_execution(mode, enabled, producer, consumer, prepared, expected):
    @dataclass(frozen=True)
    class Spec:
        num_speculative_blocks: int = 3
        layout: tuple = ("conv", "checkpoint", "replay_u", "replay_k", "replay_decay")

    original = Spec()

    class Base:
        def get_kv_cache_spec(self, config):
            return original

    scope = {"Base": Base, "replace": replace}
    _definitions(
        "vllm_ascend/ops/kimi_kda.py",
        {"AscendKimiK3DeltaAttention"},
        scope,
        methods={"get_kv_cache_spec", "get_state_shape"},
        bases="Base",
    )
    kind = scope["AscendKimiK3DeltaAttention"]
    layer, peer = kind(), kind()
    layer.use_mega_kda_replayssm = enabled
    layer._mega_pd_prefill_worker, layer._mega_pd_decode_worker = producer, consumer
    layer._mega_weights = object()
    peer._mega_weights = object() if prepared else None
    config = SimpleNamespace(
        additional_config={"mla_cache_mode": mode},
        num_speculative_tokens=3,
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer, "peer": peer}),
    )
    actual = layer.get_kv_cache_spec(config)
    assert actual.num_speculative_blocks == expected
    assert actual.layout is original.layout and original.num_speculative_blocks == config.num_speculative_tokens == 3
    if expected == 3:
        assert actual is original


def test_mega_plan_accepts_one_external_state_column(runtime):
    layer = _attention(runtime, batch=8, spec_len=4)
    layer.kv_cache = tuple(state[:9] for state in layer.kv_cache)
    metadata = _metadata(batch=8, spec_len=4)
    metadata.spec_state_indices_tensor = torch.arange(1, 9, dtype=torch.int32).reshape(8, 1)
    plan, _ = _plan(runtime, layer, metadata)
    assert plan is not None and plan.batch == 8 and plan.spec_len == 4
    assert plan.conv_state_block_ids.tolist() == list(range(1, 9))
