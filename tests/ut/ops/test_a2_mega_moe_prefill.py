# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only policy and production-function integration tests; no vLLM/NPU imports."""

import __future__

import ast
import copy
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from pydantic import TypeAdapter, ValidationError, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass

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
    for kw in (
        {"enabled": 1},
        {"scope": "all"},
        {"scope": "prefill_mixed"},
        {"min_tokens": 1},
        {"_replicated_pcp_draft": True},
    ):
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


def replicated_pcp_harness():
    """Re-enter the real feature validator when the actual PCP helper replaces VllmConfig."""

    @validated_dataclass
    class ValidatedConfig:
        model_config: Any
        parallel_config: Any
        kv_transfer_config: Any
        engram_config: Any
        additional_config: dict
        use_v2_model_runner: bool = True
        lora_config: Any = None

        @model_validator(mode="after")
        def validate_config(self):
            validate(self)
            return self

    original = config()
    parallel_values = vars(original.parallel_config) | {"eplb_config": NS(num_redundant_experts=0)}
    parallel_cls = dataclasses.make_dataclass("Parallel", [(key, Any) for key in parallel_values])
    target = ValidatedConfig(
        model_config=original.model_config,
        parallel_config=parallel_cls(**parallel_values),
        kv_transfer_config=original.kv_transfer_config,
        engram_config=NS(dp_shared_memory=True),
        additional_config={"enable_fused_mc2": 2, "a2_mega_moe_prefill": {"enabled": True}},
    )
    ns = dict(
        replace=dataclasses.replace,
        dataclass_replace=dataclasses.replace,
        logger=NS(warning_once=Mock()),
    )
    ns["rebuild_replicated_pcp_draft"] = policy.rebuild_replicated_pcp_draft
    functions("vllm_ascend/worker/v2/spec_decode/pcp_utils.py", ["prepare_replicated_pcp_config"], ns)
    return target, ns["prepare_replicated_pcp_config"]


def test_dspark_pcp_reconstruction_keeps_target_enabled_and_draft_ordinary():
    target, prepare = replicated_pcp_harness()
    draft, replicated = prepare(target)
    assert replicated and draft.parallel_config.prefill_context_parallel_size == 1
    # This intermediate config still has target identity, as in the reported stack.
    assert draft.model_config is target.model_config
    assert draft.model_config.runner_type == "generate"
    assert policy.is_draft_model(draft)
    assert not policy.is_draft_model(target)
    assert target.parallel_config.prefill_context_parallel_size == 8
    assert target.additional_config["a2_mega_moe_prefill"]["enabled"] is True
    assert draft.additional_config["a2_mega_moe_prefill"]["enabled"] is True
    validate(target)
    validate(draft)


def test_dspark_load_model_reconstruction_retains_draft_identity():
    target, prepare = replicated_pcp_harness()
    draft, _ = prepare(target)
    # DFlash first copies the config; load_dspark_model then replaces it while
    # retaining the target model_config for target-layer metadata.
    copied = copy.copy(draft)
    loaded = dataclasses.replace(copied, parallel_config=dataclasses.replace(copied.parallel_config))
    reloaded = dataclasses.replace(loaded)
    deepcopied = copy.deepcopy(reloaded)
    for stage in (copied, loaded, reloaded, deepcopied):
        validate(stage)
        assert policy.is_draft_model(stage)
        assert stage.model_config.runner_type == "generate"
    assert not policy.is_draft_model(target)
    assert target.parallel_config.prefill_context_parallel_size == 8


def test_draft_marker_uses_declared_fields_and_preserves_config_hash_input():
    target, prepare = replicated_pcp_harness()
    draft, _ = prepare(target)
    # vLLM replace rejects __dict__ attributes absent from dataclass fields.
    assert set(vars(draft)) <= {field.name for field in dataclasses.fields(draft)}
    # VllmConfig.compute_hash uses json.dumps(additional_config).
    assert json.dumps(draft.additional_config, sort_keys=True) == json.dumps(target.additional_config, sort_keys=True)
    parsed = TypeAdapter(policy.A2MegaMoEPrefillConfig).validate_python(draft.additional_config["a2_mega_moe_prefill"])
    assert parsed.enabled and parsed.scope == "prefill"


def test_real_pcp1_target_still_fails_after_dspark_reconstruction():
    target, prepare = replicated_pcp_harness()
    prepare(target)
    wrong_parallel = dataclasses.replace(target.parallel_config, prefill_context_parallel_size=1)
    with pytest.raises(ValidationError, match="DP2/TP1/PCP8"):
        dataclasses.replace(target, parallel_config=wrong_parallel)


def test_replicated_pcp_failure_does_not_change_target():
    target, _ = replicated_pcp_harness()
    draft_parallel = dataclasses.replace(target.parallel_config, prefill_context_parallel_size=1)

    def fail(source, **changes):
        candidate = NS(
            model_config=source.model_config,
            parallel_config=draft_parallel,
            additional_config=changes["additional_config"],
        )
        assert policy.is_draft_model(candidate)
        assert not policy.is_draft_model(source)
        raise RuntimeError("failed reconstruction")

    with pytest.raises(RuntimeError, match="failed reconstruction"):
        policy.rebuild_replicated_pcp_draft(target, fail)
    with pytest.raises(ValidationError, match="DP2/TP1/PCP8"):
        dataclasses.replace(target, parallel_config=draft_parallel)


def test_nested_replicated_pcp_rebuilds_keep_target_identity():
    target, _ = replicated_pcp_harness()
    other, prepare_other = replicated_pcp_harness()
    draft_parallel = dataclasses.replace(target.parallel_config, prefill_context_parallel_size=1)

    def rebuild(source, **changes):
        candidate = NS(
            model_config=source.model_config,
            parallel_config=draft_parallel,
            additional_config=changes["additional_config"],
        )
        assert policy.is_draft_model(candidate)
        other_draft, _ = prepare_other(other)
        assert policy.is_draft_model(other_draft)
        assert policy.is_draft_model(candidate)
        assert not policy.is_draft_model(other)
        return dataclasses.replace(source, parallel_config=draft_parallel, **changes)

    draft = policy.rebuild_replicated_pcp_draft(target, rebuild)
    assert policy.is_draft_model(draft)
    assert not policy.is_draft_model(target)
    assert not policy.is_draft_model(other)


@pytest.mark.parametrize("additional", [None, {}, {"a2_mega_moe_prefill": {"enabled": False}}])
def test_replicated_pcp_without_opt_in_preserves_existing_behavior(additional):
    target = config()
    target.additional_config = additional
    draft = NS(model_config=target.model_config, parallel_config=NS(prefill_context_parallel_size=1))

    def rebuild(source, **changes):
        assert source is target
        assert changes == {"parallel_config": draft.parallel_config}
        assert not policy.is_draft_model(draft)
        return draft

    result = policy.rebuild_replicated_pcp_draft(target, rebuild, parallel_config=draft.parallel_config)
    assert result is draft
    assert not policy.is_draft_model(result)


def test_internal_dspark_config_uses_ordinary_backend():
    target, prepare = replicated_pcp_harness()
    draft, _ = prepare(target)
    ns = dict(
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_a2_draft_model=policy.is_draft_model,
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True), enable_fused_mc2=1),
        get_current_hardware_profile=lambda: NS(moe_comm_policy="a2", supports=lambda capability: True),
        MoECommPolicy=NS(CAPACITY_AND_EXPERT_DENSITY="a2"),
        HardwareCapability=NS(CANN_MEGAMOE="megamoe"),
        is_mega_moe_supported=lambda: True,
        is_moe_model=lambda cfg: True,
        get_ep_group=lambda: NS(world_size=16),
    )
    functions("vllm_ascend/ascend_forward_context.py", ["use_cann_megamoe"], ns)
    assert ns["use_cann_megamoe"](target)
    assert not ns["use_cann_megamoe"](draft)


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


def selector_namespace(enabled=True):
    ns = dict(
        torch=torch,
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_a2_draft_model=policy.is_draft_model,
        get_ascend_config=lambda: NS(
            a2_mega_moe_prefill=policy.A2MegaMoEPrefillConfig(enabled=enabled), enable_fused_mc2=1
        ),
        get_mc2_tokens_capacity=Mock(return_value=4096 if enabled else 512),
        _MC2_TOKENS_PER_RANK_LIMIT=512,
        is_moe_model=lambda cfg: True,
        use_cann_megamoe=Mock(side_effect=AssertionError("target eligibility must be validated at startup")),
        get_ep_group=lambda: NS(world_size=16),
        MoECommType=NS(FUSED_MC2="fused", MC2="mc2", ALLGATHER="gather"),
        QuantType=NS(NONE="none"),
        MoECommPolicy=NS(ALLGATHER="gather"),
        get_current_hardware_profile=lambda: NS(moe_comm_policy="a2"),
        logger=Mock(),
    )
    functions(
        "vllm_ascend/ascend_forward_context.py",
        ["select_moe_comm_method", "_select_capacity_and_expert_density_moe_comm_method"],
        ns,
    )
    ns["_MOE_COMM_SELECTORS"] = {"a2": ns["_select_capacity_and_expert_density_moe_comm_method"]}
    return ns


@pytest.mark.parametrize("tokens", [None, 0, 1, 6, 256, 257, 512, 513, 4096, 4097])
def test_validated_p_target_always_selects_megamoe(tokens, monkeypatch):
    ns = selector_namespace()
    reduce = Mock(side_effect=AssertionError("selection must not initiate a collective"))
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert ns["select_moe_comm_method"](tokens, config()) == "fused"
    ns["get_mc2_tokens_capacity"].assert_not_called()
    ns["use_cann_megamoe"].assert_not_called()
    reduce.assert_not_called()
    ns["logger"].info.assert_not_called()


def test_explicit_draft_keeps_ordinary_selection_capacity():
    ns = selector_namespace()
    select = ns["select_moe_comm_method"]
    assert select(512, config(), is_draft_model=True) == "mc2"
    assert select(513, config(), is_draft_model=True) == "gather"


@pytest.mark.parametrize("identity", ["draft", "architecture", "replicated"])
def test_fixed_target_routing_preserves_all_draft_identities(identity):
    ns = selector_namespace()
    cfg = model_config_for_identity(identity)
    assert ns["select_moe_comm_method"](32, cfg) == "mc2"
    assert ns["select_moe_comm_method"](513, cfg) == "gather"


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_target_and_draft_routes(enabled):
    ns = selector_namespace(enabled=enabled)
    cfg = config()
    if enabled:
        cfg.model_config.runner_type = "draft"
    assert ns["select_moe_comm_method"](512, cfg) == "mc2"
    assert ns["select_moe_comm_method"](513, cfg) == "gather"


def test_mrv2_real_dummy_and_kv_contexts_select_without_collectives(monkeypatch):
    ns = selector_namespace()
    reduce = Mock(side_effect=AssertionError("KV-only contexts must not initiate EP votes"))
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        NS(
            _is_decode_only_node=lambda cfg: False,
            get_mc2_mask=lambda: None,
            get_mrv2_in_profile_run=lambda: False,
            sync_v2_extra_kwargs=lambda cfg: None,
            use_cann_megamoe=lambda cfg: True,
            select_moe_comm_method=ns["select_moe_comm_method"],
        ),
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.ops.fused_moe.moe_comm_method", NS(get_moe_comm_method=lambda method: None)
    )
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.utils", NS(get_dynamic_mx_quant_scale_alg=lambda cfg: None)
    )
    monkeypatch.setitem(sys.modules, "vllm.config", NS(CUDAGraphMode=NS(NONE="none")))
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed",
        NS(
            get_dp_group=lambda: NS(world_size=2),
            get_tensor_model_parallel_world_size=lambda: 1,
        ),
    )
    tree = ast.parse((ROOT / "vllm_ascend/platform.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NPUPlatform")
    hook = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_additional_forward_context")
    hook.decorator_list = []
    ns["math"] = math
    exec(
        compile(
            ast.Module(body=[hook], type_ignores=[]), "platform", "exec", flags=__future__.annotations.compiler_flag
        ),
        ns,
    )
    cfg = config()
    dp = NS(num_tokens_across_dp_cpu=torch.tensor([6, 6]))
    # The log's DP0 real prefill and DP1 dummy both enter the same fixed backend.
    for ep_rank in range(16):
        real = ep_rank < 8
        metadata = {
            "attention": NS(
                num_actual_tokens=3 if real else 6, num_reqs=1, num_prefills=int(real), num_decodes=int(not real)
            )
        }
        result = ns[hook.name](None, metadata, cfg, dp, num_tokens=6)
        assert result["moe_comm_type"] == "fused" and result["use_mega_moe"]
    # Only DP0 later enters this KV-only context. It must not wait for EP8-15.
    result = ns[hook.name](None, None, cfg, None, num_tokens=None)
    assert result["num_tokens"] is None and result["mc2_mask"] is None
    reduce.assert_not_called()


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


@pytest.mark.parametrize("padding", [[False, True, False, True], [True, True, True, True]])
def test_pcp_replay_and_idle_dummy_masks_use_real_padding(padding):
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
        get_forward_context=lambda: NS(is_padding=torch.tensor(padding)),
    )
    exec(
        compile(
            ast.Module(body=[prepare], type_ignores=[]), "prepare", "exec", flags=__future__.annotations.compiler_flag
        ),
        ns,
    )
    out = ns["prepare"](NS(tp_size=1), torch.ones(4, 8), torch.ones(4, 384))
    assert out.mc2_mask.tolist() == [not value for value in padding]


def test_fixed_route_rejects_input_above_sender_capacity_before_launch():
    tree = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/moe_comm_method.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedMC2CommImpl")
    apply = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_cann_mega_moe")
    ns = dict(TokenDispatcherWithMC2=NS)
    exec(
        compile(
            ast.Module(body=[apply], type_ignores=[]), "capacity", "exec", flags=__future__.annotations.compiler_flag
        ),
        ns,
    )
    operator = Mock(side_effect=AssertionError("oversize input must not reach the operator"))
    comm = NS(token_dispatcher=NS(max_num_tokens_per_rank=4096), mega_moe=operator)
    with pytest.raises(ValueError, match="received 4097 tokens per rank"):
        ns[apply.name](comm, NS(hidden_states=torch.empty(4097, 1)), None, False)
    operator.assert_not_called()


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


def model_config_for_identity(identity):
    if identity == "replicated":
        target, prepare = replicated_pcp_harness()
        return prepare(target)[0]
    cfg = config()
    if identity == "draft":
        cfg.model_config.runner_type = "draft"
    elif identity == "architecture":
        cfg.model_config.architectures = ["DeepseekV41DSparkModel"]
    else:
        assert identity == "target"
    return cfg


@pytest.mark.parametrize("identity", ["draft", "architecture", "replicated"])
@pytest.mark.parametrize("quant_type", ["w4a8", "w8a8", "bf16"])
def test_mrv2_draft_overrides_target_prefill_route_before_prepare(identity, quant_type):
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
        is_draft_model=policy.is_draft_model,
        select_moe_comm_method=select,
        get_moe_comm_method=get_comm,
        QuantType=NS(NONE="bf16"),
        get_current_hardware_profile=lambda: NS(supports=lambda capability: False),
        HardwareCapability=NS(CANN_MEGAMOE_MXFP="mxfp"),
    )
    exec(
        compile(
            ast.Module(body=[forward], type_ignores=[]), "draft", "exec", flags=__future__.annotations.compiler_flag
        ),
        ns,
    )
    cfg = model_config_for_identity(identity)
    layer = NS(
        enable_npugraph_ex_static_kernel=False,
        _vllm_config=cfg,
        quant_type=quant_type,
        moe_config=NS(is_sequence_parallel=False),
    )
    with pytest.raises(RuntimeError, match="prepared draft"):
        ns[forward.name](layer, hidden_states=torch.ones(32, 8), router_logits=torch.ones(32, 128))
    select.assert_called_once_with(32, cfg, is_draft_model=True, draft_moe_quant_type=quant_type)
    get_comm.assert_called_once_with("mc2", layer.moe_config)


def runner_constructor():
    """Execute the production constructor with device setup replaced by stubs."""
    tree = ast.parse((ROOT / "vllm_ascend/ops/fused_moe/fused_moe.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendMoERunner")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    class Base:
        def __init__(self, layer_name, moe_config, router, routed_experts, *args):
            self.moe_config = moe_config
            self.routed_experts = routed_experts

    ns = dict(
        Base=Base,
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_draft_model=policy.is_draft_model,
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True)),
        DEEPSEEK_V41_ROUTED_EXPERTS=384,
        QuantType=NS(W4A8="w4a8"),
        get_tp_group=Mock(),
        get_dp_group=Mock(),
        get_ep_group=Mock(),
        get_mc2_group=Mock(),
        setup_moe_comm_method=Mock(),
        get_moe_comm_method=Mock(return_value=None),
        MoECommType=NS(FUSED_MC2="fused", ALLTOALL="alltoall"),
    )
    cls.bases = [ast.Name(id="Base", ctx=ast.Load())]
    cls.body = [init]
    cls.decorator_list = []
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            "runner",
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    return ns[cls.name], ns


@pytest.mark.parametrize("identity", ["target", "draft", "architecture", "replicated"])
@pytest.mark.parametrize("quant_type", ["w4a8", "w8a8", "bf16"])
def test_runner_quantization_restriction_applies_only_to_target(identity, quant_type):
    cls, ns = runner_constructor()
    routed = NS(quant_type=quant_type, _vllm_config=model_config_for_identity(identity))
    shape = NS(hidden_dim=5120, num_experts=384, ep_size=16)
    router = object()
    if identity == "target" and quant_type != "w4a8":
        with pytest.raises(ValueError, match="requires dynamic INT8/INT4 W4A8"):
            cls("test", shape, router, routed)
        ns["setup_moe_comm_method"].assert_not_called()
    else:
        cls("test", shape, router, routed)
        assert routed.router is router
        ns["setup_moe_comm_method"].assert_called_once_with(shape)


@pytest.mark.parametrize("quant_type", ["w8a8", "bf16"])
def test_disabled_feature_keeps_existing_target_quantization(quant_type):
    cls, ns = runner_constructor()
    ns["get_ascend_config"] = lambda: NS(a2_mega_moe_prefill=NS(enabled=False))
    routed = NS(quant_type=quant_type, _vllm_config=config())
    cls("test", NS(hidden_dim=5120, num_experts=384, ep_size=16), object(), routed)
    ns["setup_moe_comm_method"].assert_called_once()
