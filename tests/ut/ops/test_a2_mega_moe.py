# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for phase selection and weight equivalence, without a vLLM install.

The leaf module deliberately has no vLLM/NPU dependency. Load it by path so
these numerical tests can also run with --noconftest on a development laptop.
"""

import ast
import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("a2_mega_moe_cpu_test", ROOT / "vllm_ascend/a2_mega_moe.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
Settings = MODULE.A2MegaMoEPrefillConfig


def make_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            tensor_parallel_size=8,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            world_size_across_dp=8,
        ),
        model_config=SimpleNamespace(
            architectures=["Qwen4ExpForCausalLM"],
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(hidden_size=2560, moe_intermediate_size=640, num_experts_per_tok=10),
            get_num_experts=lambda: 512,
        ),
        speculative_config=None,
        lora_config=None,
        quant_config=None,
    )


def qsa_metadata(*phases):
    return SimpleNamespace(
        qsa_request_plan=SimpleNamespace(requests=tuple(SimpleNamespace(is_prefill=p) for p in phases))
    )


def test_explicit_prefill_enable():
    cfg = TypeAdapter(Settings).validate_python({"enabled": True})
    metadata = {"gdn": SimpleNamespace(), "qsa": qsa_metadata(True, True)}
    assert cfg.selects(metadata)
    assert not Settings().selects(metadata)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        SimpleNamespace(),
        qsa_metadata(False),
        qsa_metadata(True, False),
        SimpleNamespace(qsa_request_plan=None),
        qsa_metadata(),
        SimpleNamespace(num_prefills=1, num_decode_tokens=1),
        {"qsa": qsa_metadata(True), "other": SimpleNamespace(num_prefills=1, num_decode_tokens=1)},
    ],
)
def test_decode_mixed_and_unknown_never_selected(metadata):
    assert not Settings(enabled=True).selects(metadata)


def test_standard_metadata_and_profile():
    cfg = Settings(enabled=True)
    assert cfg.selects(SimpleNamespace(num_prefills=1, num_decode_tokens=0))
    assert cfg.selects(None, in_profile_run=True)
    assert not Settings().selects(qsa_metadata(True), in_profile_run=True)


@pytest.mark.parametrize("kwargs", [{"min_tokens": 1024}, {"max_tokens": 4096}, {"enabled": "true"}, {"unknown": 1}])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises((ValueError, ValidationError)):
        Settings(**kwargs)


def test_model_eligibility():
    MODULE.validate_a2_prefill_model(make_config(), dynamic_eplb=False, redundant_experts=0, mix_placement=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_parallel_size", 2),
        ("tensor_parallel_size", 4),
        ("pipeline_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("enable_expert_parallel", False),
    ],
)
def test_unsupported_topology(field, value):
    cfg = make_config()
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError, match="TP8/EP8"):
        MODULE.validate_a2_prefill_model(cfg, dynamic_eplb=False, redundant_experts=0, mix_placement=False)


@pytest.mark.parametrize(
    "field,value", [("speculative_config", object()), ("lora_config", object()), ("quant_config", object())]
)
def test_unsupported_model_features(field, value):
    cfg = make_config()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        MODULE.validate_a2_prefill_model(cfg, dynamic_eplb=False, redundant_experts=0, mix_placement=False)


@pytest.mark.parametrize("kwargs", [{"dynamic_eplb": True}, {"redundant_experts": 1}, {"mix_placement": True}])
def test_expert_mapping_rejected(kwargs):
    options = dict(dynamic_eplb=False, redundant_experts=0, mix_placement=False)
    options.update(kwargs)
    with pytest.raises(ValueError, match="static contiguous"):
        MODULE.validate_a2_prefill_model(make_config(), **options)


@pytest.mark.parametrize("activation", ["swiglu", "situ"])
def test_640_to_768_preserves_expert_output_and_gate_up_order(activation):
    torch.manual_seed(17)
    w1 = (torch.randn(2, 16, 1280) * 0.1).bfloat16()
    w2 = (torch.randn(2, 640, 16) * 0.1).bfloat16()
    before1, before2 = w1.clone(), w2.clone()
    first, second = MODULE.pad_bf16_expert_weights(w1, w2)
    x = torch.randn(5, 16, dtype=torch.float64)

    def expert(up_gate, down):
        gate, up = (x @ up_gate.double()).chunk(2, dim=-1)
        if activation == "swiglu":
            activated = torch.nn.functional.silu(gate) * up
        else:
            activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(up / 25))
        return activated @ down.double()

    for index in range(2):
        assert first[index].shape == (16, 1536)
        assert second[index].shape == (768, 16)
        torch.testing.assert_close(first[index][:, :640], w1[index, :, :640], rtol=0, atol=0)
        torch.testing.assert_close(first[index][:, 768:1408], w1[index, :, 640:], rtol=0, atol=0)
        assert torch.count_nonzero(first[index][:, 640:768]) == 0
        assert torch.count_nonzero(first[index][:, 1408:]) == 0
        assert torch.count_nonzero(second[index][640:]) == 0
        torch.testing.assert_close(
            expert(first[index], second[index]), expert(w1[index], w2[index]), rtol=1e-10, atol=1e-10
        )
    torch.testing.assert_close(w1, before1, rtol=0, atol=0)
    torch.testing.assert_close(w2, before2, rtol=0, atol=0)


def load_forward_policy(settings):
    """Execute real selector functions with only their distributed dependencies replaced."""
    source = ast.parse((ROOT / "vllm_ascend/ascend_forward_context.py").read_text())
    names = {
        "use_cann_megamoe",
        "_select_capacity_and_expert_density_moe_comm_method",
        "_select_fused_or_capacity_moe_comm_method",
        "_select_capacity_and_world_size_moe_comm_method",
        "select_moe_comm_method",
        "set_mc2_tokens_capacity",
    }
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    code = ast.Module(body=nodes, type_ignores=[])
    comm = Enum("MoECommType", "ALLGATHER MC2 ALLTOALL FUSED_MC2")
    policy = Enum("MoECommPolicy", "ALLGATHER CAPACITY_AND_EXPERT_DENSITY FUSED_OR_CAPACITY CAPACITY_AND_WORLD_SIZE")
    graph = Enum("CUDAGraphMode", "NONE FULL PIECEWISE")
    cfg = SimpleNamespace(a2_mega_moe_prefill=settings, enable_fused_mc2=1, enable_prefill_mc2=False)
    namespace = dict(
        VllmConfig=Any,
        Any=Any,
        QuantType=SimpleNamespace(NONE=0),
        CUDAGraphMode=graph,
        MoECommType=comm,
        MoECommPolicy=policy,
        HardwareCapability=SimpleNamespace(CANN_MEGAMOE=0),
        get_current_hardware_profile=lambda: SimpleNamespace(
            moe_comm_policy=policy.CAPACITY_AND_EXPERT_DENSITY, supports=lambda _: True
        ),
        get_ascend_config=lambda: cfg,
        get_ep_group=lambda: SimpleNamespace(world_size=8),
        is_mega_moe_supported=lambda: True,
        is_moe_model=lambda _: True,
        get_mc2_tokens_capacity=lambda: 4096,
        logger=Mock(),
        _is_decode_only_node=lambda _: False,
        _mc2_tokens_capacity=None,
        _MEGA_MOE_TOKENS_PER_RANK_LIMIT=16384,
        _MC2_TOKENS_PER_RANK_LIMIT=512,
        _DISPATCH_FFN_COMBINE_TOKENS_PER_RANK_LIMIT=512,
    )
    exec(compile(code, "ascend_forward_context.py", "exec"), namespace)
    namespace["_MOE_COMM_SELECTORS"] = {
        policy.CAPACITY_AND_EXPERT_DENSITY: namespace["_select_capacity_and_expert_density_moe_comm_method"],
        policy.FUSED_OR_CAPACITY: namespace["_select_fused_or_capacity_moe_comm_method"],
        policy.CAPACITY_AND_WORLD_SIZE: namespace["_select_capacity_and_world_size_moe_comm_method"],
    }
    return SimpleNamespace(**namespace), namespace


@pytest.mark.parametrize(
    "tokens,phase,graph,expected",
    [
        (1024, (True,), "NONE", "FUSED_MC2"),
        (4096, (True,), "NONE", "FUSED_MC2"),
        (1, (True,), "NONE", "FUSED_MC2"),
        (32768, (True,), "NONE", "FUSED_MC2"),
        (2048, (True, False), "NONE", "ALLGATHER"),
        (2048, (False,), "NONE", "ALLGATHER"),
        (2048, (True,), "FULL", "ALLGATHER"),
        (2048, (True,), "PIECEWISE", "ALLGATHER"),
    ],
)
def test_real_selector_dispatch(tokens, phase, graph, expected):
    policy, _ = load_forward_policy(Settings(enabled=True))
    result = policy.select_moe_comm_method(
        tokens,
        make_config(),
        attn_metadata={"qsa": qsa_metadata(*phase)},
        cudagraph_runtime_mode=getattr(policy.CUDAGraphMode, graph),
    )
    assert result.name == expected


def test_disabled_ep_lora_and_draft_paths():
    policy, _ = load_forward_policy(Settings())
    assert not policy.use_cann_megamoe(make_config())
    assert policy.select_moe_comm_method(2048, make_config(), attn_metadata=qsa_metadata(True)).name == "ALLGATHER"
    policy, _ = load_forward_policy(Settings(enabled=True))
    cfg = make_config()
    cfg.parallel_config.enable_expert_parallel = False
    assert policy.select_moe_comm_method(2048, cfg, attn_metadata=qsa_metadata(True)).name == "ALLGATHER"
    cfg = make_config()
    cfg.lora_config = object()
    assert policy.select_moe_comm_method(2048, cfg, attn_metadata=qsa_metadata(True)).name == "ALLTOALL"
    assert (
        policy.select_moe_comm_method(2048, make_config(), is_draft_model=True, attn_metadata=qsa_metadata(True)).name
        == "ALLGATHER"
    )


def test_real_capacity_uses_scheduler_without_a_token_policy_limit():
    policy, namespace = load_forward_policy(Settings(enabled=True))
    cfg = make_config()
    cfg.scheduler_config = SimpleNamespace(max_num_batched_tokens=32768)
    policy.set_mc2_tokens_capacity(cfg, max_num_reqs=256, uniform_decode_query_len=1)
    assert namespace["_mc2_tokens_capacity"] == 32768
    assert policy.select_moe_comm_method(4096, cfg, in_profile_run=True).name == "FUSED_MC2"
    assert policy.select_moe_comm_method(4096, cfg).name == "ALLGATHER"


def test_real_weight_producer_uses_one_padded_list_for_both_backends(monkeypatch):
    source = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/routed_experts.py").read_text())
    owner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AscendUnquantizedFusedMoEMethod")
    methods = [
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"process_weights_after_loading", "get_mlp_weights", "get_fused_mc2_weights"}
    ]

    class Parent:
        def process_weights_after_loading(self, layer):
            pass

    class Upstream(Parent):
        pass

    namespace = dict(
        torch=torch,
        UnquantizedFusedMoEMethod=Upstream,
        get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=1, a2_mega_moe_prefill=Settings(enabled=True)),
        use_cann_megamoe=lambda _: True,
        get_current_vllm_config=lambda: make_config(),
        get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: False),
        HardwareCapability=SimpleNamespace(CANN_MEGAMOE_MXFP=0),
        replace_parameter=lambda layer, name, data: setattr(layer, name, torch.nn.Parameter(data)),
        pad_bf16_expert_weights=MODULE.pad_bf16_expert_weights,
        _EXTRA_CTX=SimpleNamespace(use_mega_moe=True),
        MoEWeights=SimpleNamespace,
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), "routed_experts.py", "exec"), namespace)
    producer = type("Producer", (Upstream,), {n.name: namespace[n.name] for n in methods})()
    producer._maybe_pad_weight = lambda w: w
    producer.dynamic_eplb = False
    producer.moe = SimpleNamespace(has_bias=False)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(empty_cache=lambda: None), raising=False)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.randn(2, 1280, 16).bfloat16())
    layer.w2_weight = torch.nn.Parameter(torch.randn(2, 16, 640).bfloat16())
    producer.process_weights_after_loading(layer)
    assert layer.a2_mega_moe_intermediate_size == 768
    assert not hasattr(layer, "w13_weight") and not hasattr(layer, "w2_weight")
    w1, w2 = producer.get_mlp_weights(layer)
    payload = producer.get_fused_mc2_weights(layer)
    assert w1 is payload.w1 and w2 is payload.w2
    assert w1[0].shape == (16, 1536) and w2[0].shape == (768, 16)


def test_real_a2_buffer_uses_physical_width_and_safe_route_capacity():
    source = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/moe_comm_method.py").read_text())
    owner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "FusedMC2CommImpl")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_init_mega_moe_symm_buffer")

    class Dispatcher:
        global_bs = 0
        max_num_tokens_per_rank = 4096
        ep_world_size = 8
        ep_rank_id = 0

    namespace = dict(
        torch=torch,
        TokenDispatcherWithMC2=Dispatcher,
        get_mc2_group=lambda: SimpleNamespace(device_group="ep8"),
        get_ascend_config=lambda: SimpleNamespace(a2_mega_moe_prefill=Settings(enabled=True), mega_moe_max_tokens=1),
        logger=Mock(),
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), "moe_comm_method.py", "exec"), namespace)
    instance = SimpleNamespace(
        token_dispatcher=Dispatcher(),
        moe_config=SimpleNamespace(
            experts_per_token=10, num_experts=512, hidden_dim=2560, intermediate_size_per_partition=640
        ),
        get_symm_buffer_for_mega_moe=Mock(),
    )
    namespace[method.name](instance, is_decode_only_node=False, physical_intermediate_size=768)
    args, kwargs = instance.get_symm_buffer_for_mega_moe.call_args
    assert args == ("ep8", 512, 4096, 10)
    assert kwargs["max_recv_token_num"] == 327680
    assert kwargs["intermediate_hidden"] == 768


def test_real_config_requires_explicit_cann_backend_and_package():
    source = ast.parse((ROOT / "vllm_ascend/ascend_config.py").read_text())
    owner = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "AscendConfig")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_user_input_ranges")
    method.decorator_list = []
    package = SimpleNamespace(util=SimpleNamespace(find_spec=Mock(return_value=object())))
    namespace = dict(importlib=package)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "ascend_config.py", "exec"), namespace)
    validate = namespace[method.name]
    cfg = SimpleNamespace(
        weight_nz_mode=1,
        enable_mc2_hierarchy_comm=False,
        enable_fused_mc2=2,
        a2_mega_moe_prefill=Settings(enabled=True),
    )
    validate(cfg)
    assert cfg.enable_fused_mc2 == 1 and namespace["_MEGA_MOE_SUPPORTED"] is True
    for value in (0, 1):
        cfg.enable_fused_mc2 = value
        with pytest.raises(ValueError, match="enable_fused_mc2=2"):
            validate(cfg)
    cfg.enable_fused_mc2 = 2
    package.util.find_spec.return_value = None
    with pytest.raises(ValueError, match="installation"):
        validate(cfg)
