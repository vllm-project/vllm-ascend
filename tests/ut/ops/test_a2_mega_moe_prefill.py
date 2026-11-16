# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only policy and production-function integration tests; no vLLM/NPU imports."""

import __future__

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("a2_prefill_policy", ROOT / "vllm_ascend/a2_mega_moe.py")
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), path, "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace


def config():
    return NS(
        model_config=NS(
            architectures=["DeepseekV41ForCausalLM"],
            runner_type="generate",
            enforce_eager=True,
            dtype=torch.bfloat16,
            hf_text_config=NS(hidden_size=5120, moe_intermediate_size=2304, num_experts_per_tok=6),
            get_num_experts=lambda: 384,
        ),
        parallel_config=NS(
            enable_expert_parallel=True,
            data_parallel_size=2,
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
            use_ubatching=False,
            world_size_across_dp=16,
        ),
        kv_transfer_config=NS(kv_role="kv_producer"),
        use_v2_model_runner=True,
        lora_config=None,
        speculative_config=NS(method="dspark"),
    )


def validate(cfg, **kw):
    policy.validate_a2_mega_moe_prefill(
        cfg, **({"dynamic_eplb": False, "redundant_experts": 0, "mix_placement": False} | kw)
    )


def test_defaults_and_strict_schema():
    assert not policy.A2MegaMoEPrefillConfig().enabled
    assert policy.A2MegaMoEPrefillConfig(enabled=True).scope == "prefill"
    for kw in ({"enabled": 1}, {"scope": "all"}, {"scope": "prefill_mixed"}, {"min_tokens": 1}):
        with pytest.raises(ValidationError):
            policy.A2MegaMoEPrefillConfig(**kw)


@pytest.mark.parametrize("fused,nz,installed", [(1, 1, True), (0, 1, True), (2, 0, True), (2, 1, False)])
def test_startup_rejects_wrong_operator_or_missing_package(fused, nz, installed):
    tree = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
    validate = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_user_input_ranges")
    validate.decorator_list = []
    ns = dict(importlib=NS(util=NS(find_spec=lambda name: object() if installed else None)))
    exec(compile(ast.Module(body=[validate], type_ignores=[]), "config", "exec"), ns)
    cfg = NS(
        weight_nz_mode=nz,
        enable_mc2_hierarchy_comm=False,
        a2_mega_moe_prefill=policy.A2MegaMoEPrefillConfig(enabled=True),
        enable_fused_mc2=fused,
    )
    with pytest.raises(ValueError):
        ns[validate.name](cfg)


def test_target_allows_existing_dspark_and_draft_is_excluded():
    cfg = config()
    validate(cfg)
    cfg.model_config.runner_type = "draft"
    cfg.model_config.architectures = ["DeepseekV41DSparkModel"]
    cfg.parallel_config.prefill_context_parallel_size = 1
    assert policy.is_draft_model(cfg)
    validate(cfg)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel_config", "data_parallel_size", 1),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 4),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", False),
        ("parallel_config", "enable_eplb", True),
        ("parallel_config", "use_ubatching", True),
        ("parallel_config", "expert_placement_strategy", "round_robin"),
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", torch.float16),
        ("model_config", "architectures", ["Qwen4ExpForCausalLM"]),
        ("kv_transfer_config", "kv_role", "kv_consumer"),
        ("kv_transfer_config", "kv_role", "kv_both"),
    ],
)
def test_invalid_worker(section, field, value):
    cfg = config()
    setattr(getattr(cfg, section), field, value)
    with pytest.raises(ValueError):
        validate(cfg)


@pytest.mark.parametrize("kw", [dict(dynamic_eplb=True), dict(redundant_experts=1), dict(mix_placement=True)])
def test_invalid_expert_placement(kw):
    with pytest.raises(ValueError):
        validate(config(), **kw)


@pytest.mark.parametrize(
    "field,value", [("hidden_size", 2560), ("moe_intermediate_size", 640), ("num_experts_per_tok", 10)]
)
def test_invalid_model_shape(field, value):
    cfg = config()
    setattr(cfg.model_config.hf_text_config, field, value)
    with pytest.raises(ValueError):
        validate(cfg)


def phase(prefills, decodes=0):
    return NS(num_reqs=prefills + decodes, num_prefills=prefills, num_decodes=decodes, max_query_len=1)


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"encoder": phase(2), "decoder": phase(2)}, (1, 0)),
        ({"one_token_prefix_tail": phase(1)}, (1, 0)),
        ({"bounded_replay": phase(2)}, (1, 0)),
        ({"empty_peer": phase(0)}, (0, 0)),
        ({"mixed": phase(1, 1)}, (0, 1)),
        ({"decode": phase(0, 1)}, (0, 1)),
        ({"unknown": NS(max_query_len=4096)}, (0, 1)),
        ({}, (0, 1)),
        (None, (0, 1)),
    ],
)
def test_cpu_phase_evidence(metadata, expected):
    assert policy.prefill_vote(metadata) == expected


def test_group_vote_includes_idle_and_veto_peers():
    for peer in (phase(0), phase(2), phase(0, 1), NS()):
        votes = [policy.prefill_vote(phase(1))] * 8 + [policy.prefill_vote(peer)] * 8
        prefill, veto = map(max, zip(*votes))
        assert policy.select_prefill_from_votes(prefill, veto) == (
            peer.num_decodes == 0 if hasattr(peer, "num_decodes") else False
        )


def test_empty_pcp_placeholder_does_not_veto_peer_prefill():
    placeholder = phase(0, 1)
    placeholder.num_actual_tokens = 0
    assert policy.prefill_vote(placeholder) == (0, 0)


def selector_namespace(enabled=True):
    ns = dict(
        torch=torch,
        prefill_vote=policy.prefill_vote,
        select_prefill_from_votes=policy.select_prefill_from_votes,
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_a2_draft_model=policy.is_draft_model,
        get_ascend_config=lambda: NS(
            a2_mega_moe_prefill=policy.A2MegaMoEPrefillConfig(enabled=enabled), enable_fused_mc2=1
        ),
        get_mc2_tokens_capacity=lambda: 4096,
        _MC2_TOKENS_PER_RANK_LIMIT=512,
        is_moe_model=lambda cfg: True,
        use_cann_megamoe=lambda cfg: not policy.is_draft_model(cfg),
        get_ep_group=lambda: NS(world_size=16),
        MoECommType=NS(FUSED_MC2="fused", MC2="mc2", ALLGATHER="gather"),
        QuantType=NS(NONE="none"),
        MoECommPolicy=NS(ALLGATHER="gather"),
        get_current_hardware_profile=lambda: NS(moe_comm_policy="a2"),
        logger=NS(debug=Mock()),
    )
    functions(
        "vllm_ascend/ascend_forward_context.py",
        ["select_moe_comm_method", "_select_capacity_and_expert_density_moe_comm_method"],
        ns,
    )
    ns["_MOE_COMM_SELECTORS"] = {"a2": ns["_select_capacity_and_expert_density_moe_comm_method"]}
    return ns


def test_selector_and_fallback_threshold():
    ns = selector_namespace()
    ns["select_a2_mega_moe_prefill"] = Mock(return_value=True)
    select = ns["select_moe_comm_method"]
    assert select(4096, config(), in_profile_run=True) == "fused"
    ns["select_a2_mega_moe_prefill"].return_value = False
    assert select(512, config()) == "mc2"
    assert select(513, config()) == "gather"
    ns["select_a2_mega_moe_prefill"].reset_mock()
    assert select(32, config(), is_draft_model=True) == "mc2"
    ns["select_a2_mega_moe_prefill"].assert_not_called()


def test_disabled_selector_does_not_vote():
    ns = selector_namespace(enabled=False)
    ns["select_a2_mega_moe_prefill"] = Mock(side_effect=AssertionError("disabled worker voted"))
    assert ns["select_moe_comm_method"](32, config()) == "mc2"


def test_actual_collective_gate(monkeypatch):
    group = object()
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.distributed.parallel_state", NS(get_mc2_group=lambda: NS(cpu_group=group))
    )
    calls = []

    def reduce(votes, op, group):
        calls.append(votes.clone())
        # Simulate a peer with a decode request. It must veto local prefill.
        votes[1] = 1

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    ns = dict(torch=torch, prefill_vote=policy.prefill_vote, select_prefill_from_votes=policy.select_prefill_from_votes)
    functions("vllm_ascend/ascend_forward_context.py", ["select_a2_mega_moe_prefill"], ns)
    select = ns["select_a2_mega_moe_prefill"]
    assert not select(phase(1), in_profile_run=False, graph_enabled=False)
    assert len(calls) == 1 and calls[0].device.type == "cpu"
    assert select(None, in_profile_run=True, graph_enabled=False)
    assert not select(phase(1), in_profile_run=True, graph_enabled=True)
    assert len(calls) == 1


def test_policy_keeps_a2_disabled_without_opt_in():
    assert not policy.a2_prefill_enabled(NS())
    assert not policy.a2_prefill_enabled(Mock())


def w4a8_class():
    tree = ast.parse((ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendW4A8DynamicFusedMoEMethod")
    cls.decorator_list = []
    cfg = config()
    cfg.quant_config = NS(quant_description={}, get_name=lambda: "ascend")
    ns = dict(
        torch=torch,
        AscendMoEScheme=object,
        QuantType=NS(W4A8="w4a8"),
        MoEWeights=NS,
        ASCEND_QUANTIZATION_METHOD="ascend",
        COMPRESSED_TENSORS_METHOD="compressed",
        _MODELSLIM_SCALE_BIAS_PARTITIONS=16,
        _INT4_VALUES_PER_INT32=8,
        _INT4_VALUES_PER_BYTE=2,
        get_current_vllm_config=lambda: cfg,
        get_mc2_group=lambda: NS(),
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True)),
        a2_prefill_enabled=policy.a2_prefill_enabled,
        use_cann_megamoe=lambda cfg: not policy.is_draft_model(cfg),
        maybe_trans_nz=lambda tensor: tensor,
        _EXTRA_CTX=NS(use_mega_moe=True),
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "w4a8", "exec", flags=__future__.annotations.compiler_flag), ns
    )
    return ns[cls.name], ns, cfg


def test_w4a8_prefill_lists_keep_ordinary_fallback_and_one_owner():
    cls, ns, cfg = w4a8_class()
    method = cls()
    assert method.use_expert_weight_list
    # Isolate backend ownership from device-only scale encoding/NZ conversion.
    method.process_weights_after_loading_modelslim = lambda layer: None
    layer = torch.nn.Module()
    for name, shape, dtype in (
        ("w13_weight", (2, 8, 8), torch.int8),
        ("w2_weight", (2, 8, 8), torch.int8),
        ("w13_weight_scale", (2, 16), torch.int64),
        ("w2_weight_scale", (2, 16), torch.int64),
        ("w13_scale_bias", (2, 16), torch.float32),
        ("w2_scale_bias", (2, 16), torch.float32),
    ):
        layer.register_parameter(name, torch.nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False))
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight")
    payload = method.get_fused_mc2_weights(layer)
    ordinary_w1, _, ordinary_w2, _ = method._get_mlp_weights(layer)
    assert payload.w1[0].dtype == torch.int8 and ordinary_w1[0].dtype == torch.int32
    assert payload.w1[0].untyped_storage().data_ptr() == ordinary_w1[0].untyped_storage().data_ptr()
    assert payload.w2[0].untyped_storage().data_ptr() == ordinary_w2[0].untyped_storage().data_ptr()
    assert payload.w1_scale_bias[0].dtype == torch.float32
    assert len(method._get_mlp_scale_bias(layer)[0]) == 2
    cfg.model_config.runner_type = "draft"
    assert not cls().use_expert_weight_list


@pytest.mark.parametrize("enabled,budget,expected", [(True, 32768, 32768), (False, 32768, 16384)])
def test_sender_capacity_not_policy_threshold(enabled, budget, expected):
    ns = dict(
        _mc2_tokens_capacity=None,
        use_cann_megamoe=lambda cfg: True,
        get_ascend_config=lambda: NS(
            enable_fused_mc2=1, enable_prefill_mc2=False, a2_mega_moe_prefill=NS(enabled=enabled)
        ),
        _is_decode_only_node=lambda cfg: False,
        a2_prefill_enabled=policy.a2_prefill_enabled,
        _MEGA_MOE_TOKENS_PER_RANK_LIMIT=16384,
    )
    functions("vllm_ascend/ascend_forward_context.py", ["set_mc2_tokens_capacity"], ns)
    cfg = config()
    cfg.scheduler_config = NS(max_num_batched_tokens=budget)
    ns["set_mc2_tokens_capacity"](cfg, 16, 1)
    assert ns["_mc2_tokens_capacity"] == expected


def test_pcp_and_bounded_replay_mask_uses_real_padding():
    tree = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/prepare_finalize.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PrepareAndFinalizeWithMC2")
    prepare = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    # Execute production preparation without importing the NPU module.
    context = NS(mc2_mask=torch.ones(4, dtype=torch.bool), padded_num_tokens=4, a2_mega_moe_prefill=True)
    ns = dict(
        torch=torch,
        nn=torch.nn,
        _EXTRA_CTX=context,
        QuantType=NS(NONE="none"),
        MoEPrepareOutput=NS,
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True)),
        a2_prefill_enabled=policy.a2_prefill_enabled,
        get_forward_context=lambda: NS(is_padding=torch.tensor([False, True, False, True])),
    )
    exec(
        compile(
            ast.Module(body=[prepare], type_ignores=[]), "prepare", "exec", flags=__future__.annotations.compiler_flag
        ),
        ns,
    )
    out = ns["prepare"](NS(tp_size=1), torch.ones(4, 8), torch.ones(4, 384))
    assert out.mc2_mask.tolist() == [True, False, True, False]


def test_receive_capacity_covers_skewed_ep16_load():
    tree = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/moe_comm_method.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMC2CommImpl")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_init_mega_moe_symm_buffer")
    dispatcher_type = type("Dispatcher", (), {})
    dispatcher = dispatcher_type()
    dispatcher.global_bs = 0
    dispatcher.ep_world_size = 16
    dispatcher.ep_rank_id = 0
    dispatcher.max_num_tokens_per_rank = 4096
    allocate = Mock(return_value="buffer")
    comm = NS(
        token_dispatcher=dispatcher,
        get_symm_buffer_for_mega_moe=allocate,
        moe_config=NS(experts_per_token=6, num_experts=384, hidden_dim=5120, intermediate_size_per_partition=2304),
    )
    ns = dict(
        TokenDispatcherWithMC2=dispatcher_type,
        get_mc2_group=lambda: NS(device_group="hccl"),
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True), mega_moe_max_tokens=65536),
        a2_prefill_enabled=policy.a2_prefill_enabled,
        logger=NS(info=Mock()),
    )
    exec(
        compile(ast.Module(body=[init], type_ignores=[]), "buffer", "exec", flags=__future__.annotations.compiler_flag),
        ns,
    )
    assert ns[init.name](comm, 2, 258, is_decode_only_node=False) == "buffer"
    assert allocate.call_args.kwargs["max_recv_token_num"] == 393216
    assert allocate.call_args.args[:4] == ("hccl", 384, 4096, 6)


def test_comm_cache_key_distinguishes_shape_and_topk():
    ns = {}
    functions("vllm_ascend/ops/fused_moe/moe_comm_method.py", ["_moe_config_key"], ns)
    key = ns["_moe_config_key"]
    model = NS(
        num_experts=384,
        num_local_experts=24,
        hidden_dim=5120,
        intermediate_size_per_partition=2304,
        experts_per_token=6,
    )
    original = key("fused", model)
    model.experts_per_token = 3
    assert key("fused", model) != original
    model.experts_per_token = 6
    model.intermediate_size_per_partition = 3072
    assert key("fused", model) != original


def test_standalone_precision_check_rejects_large_errors_and_nan():
    ns = dict(torch=torch, BF16_TOLERANCE=2**-6, REQUIRED_MATCHED_RATIO=0.99, MAX_ABS_ERROR=1.0)
    functions("tests/e2e/nightly/single_node/ops/test_a2_deepseek_prefill.py", ["assert_bf16_close"], ns)
    check = ns["assert_bf16_close"]
    expected = torch.zeros(1024)
    check(expected + 0.01, expected)
    for value in (float("nan"), 2.0):
        actual = expected.clone()
        actual[0] = value
        with pytest.raises(AssertionError):
            check(actual, expected)


def test_mrv2_draft_overrides_target_prefill_route_before_prepare():
    tree = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/routed_experts.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendRoutedExperts")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_impl")
    context = NS(moe_comm_type="fused", use_mega_moe=True, in_profile_run=False, max_tokens_across_dp=32)
    select = Mock(return_value="mc2")
    prepare = Mock(side_effect=RuntimeError("prepared draft"))
    get_comm = Mock(return_value=NS(prepare=prepare))
    ns = dict(
        torch=torch,
        get_forward_context=lambda: NS(),
        _EXTRA_CTX=context,
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True)),
        a2_prefill_enabled=policy.a2_prefill_enabled,
        select_moe_comm_method=select,
        get_moe_comm_method=get_comm,
        QuantType=NS(NONE="bf16"),
    )
    exec(
        compile(
            ast.Module(body=[forward], type_ignores=[]), "draft", "exec", flags=__future__.annotations.compiler_flag
        ),
        ns,
    )
    cfg = config()
    cfg.model_config.runner_type = "draft"
    layer = NS(
        enable_npugraph_ex_static_kernel=False,
        _vllm_config=cfg,
        quant_type="w4a8",
        moe_config=NS(is_sequence_parallel=False),
    )
    with pytest.raises(RuntimeError, match="prepared draft"):
        ns[forward.name](layer, hidden_states=torch.ones(32, 8), router_logits=torch.ones(32, 128))
    select.assert_called_once_with(32, cfg, is_draft_model=True, draft_moe_quant_type="w4a8")
    get_comm.assert_called_once_with("mc2", layer.moe_config)
