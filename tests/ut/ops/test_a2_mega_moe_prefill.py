# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only policy and regression tests for the A2 prefill MegaMoE backport.

No vLLM/NPU imports: the policy module is loaded standalone and production
functions/classes are extracted from source via AST and executed against
stub namespaces, mirroring the upstream PR test design.
"""

import __future__

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import ANY, Mock

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
            data_parallel_size=1,
            tensor_parallel_size=1,
            prefill_context_parallel_size=8,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            enable_eplb=False,
            use_ubatching=False,
            world_size_across_dp=8,
        ),
        kv_transfer_config=NS(kv_role="kv_producer"),
        additional_config={"a2_mega_moe_prefill": {"enabled": True, "scope": "prefill"}},
        use_v2_model_runner=True,
        lora_config=None,
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


def test_target_topology_validates_and_draft_identities_are_excluded():
    cfg = config()
    validate(cfg)
    assert not policy.is_draft_model(cfg)

    draft_by_runner = config()
    draft_by_runner.model_config.runner_type = "draft"
    draft_by_runner.parallel_config.prefill_context_parallel_size = 1
    assert policy.is_draft_model(draft_by_runner)
    validate(draft_by_runner)

    draft_by_arch = config()
    draft_by_arch.model_config.architectures = ["DeepseekV41DSparkModel"]
    draft_by_arch.parallel_config.prefill_context_parallel_size = 1
    assert policy.is_draft_model(draft_by_arch)
    validate(draft_by_arch)

    draft_by_pcp_signature = config()
    draft_by_pcp_signature.parallel_config.prefill_context_parallel_size = 1
    assert policy.is_draft_model(draft_by_pcp_signature)
    validate(draft_by_pcp_signature)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "prefill_context_parallel_size", 4),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "decode_context_parallel_size", 2),
        ("parallel_config", "enable_expert_parallel", False),
        ("parallel_config", "enable_eplb", True),
        ("parallel_config", "use_ubatching", True),
        ("model_config", "enforce_eager", False),
        ("model_config", "dtype", torch.float16),
        ("model_config", "architectures", ["Qwen4ExpForCausalLM"]),
        ("kv_transfer_config", "kv_role", "kv_consumer"),
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


def test_policy_disabled_without_opt_in():
    assert not policy.a2_prefill_enabled(NS())
    assert not policy.a2_prefill_enabled(Mock())


def test_rebuild_without_opt_in_preserves_existing_behavior():
    target = config()
    target.additional_config = None
    draft = NS(model_config=target.model_config, parallel_config=NS(prefill_context_parallel_size=1))

    def rebuild(source, **changes):
        assert source is target
        assert changes == {"parallel_config": draft.parallel_config}
        assert not policy.is_draft_model(draft)
        return draft

    result = policy.rebuild_replicated_pcp_draft(target, rebuild, parallel_config=draft.parallel_config)
    assert result is draft


def megamoe_gating_namespace(a2_enabled=True):
    ns = dict(
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_a2_draft_model=policy.is_draft_model,
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=a2_enabled), enable_fused_mc2=2),
        get_current_hardware_profile=lambda: NS(
            moe_comm_policy="capacity_density",
            supports=lambda capability: True,
        ),
        MoECommPolicy=NS(CAPACITY_AND_EXPERT_DENSITY="capacity_density"),
        HardwareCapability=NS(CANN_MEGAMOE="megamoe"),
        is_mega_moe_supported=lambda: True,
        is_moe_model=lambda cfg: True,
        get_ep_group=lambda: NS(world_size=8),
    )
    functions("vllm_ascend/ascend_forward_context.py", ["use_cann_megamoe"], ns)
    return ns


def test_use_cann_megamoe_target_true_draft_false():
    ns = megamoe_gating_namespace()
    assert ns["use_cann_megamoe"](config())
    draft = config()
    draft.parallel_config.prefill_context_parallel_size = 1
    assert not ns["use_cann_megamoe"](draft)


def test_use_cann_megamoe_disabled_when_opt_in_off():
    ns = megamoe_gating_namespace(a2_enabled=False)
    assert not ns["use_cann_megamoe"](config())


def selector_namespace(enabled=True):
    ns = dict(
        torch=torch,
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_a2_draft_model=policy.is_draft_model,
        get_ascend_config=lambda: NS(
            a2_mega_moe_prefill=policy.A2MegaMoEPrefillConfig(enabled=enabled), enable_fused_mc2=2
        ),
        get_mc2_tokens_capacity=Mock(return_value=512),
        _MC2_TOKENS_PER_RANK_LIMIT=512,
        is_moe_model=lambda cfg: True,
        use_cann_megamoe=Mock(side_effect=AssertionError("target eligibility is validated at startup")),
        get_ep_group=lambda: NS(world_size=8),
        MoECommType=NS(FUSED_MC2="fused", MC2="mc2", ALLGATHER="gather", ALLTOALL="alltoall"),
        QuantType=NS(NONE="none"),
        MoECommPolicy=NS(
            ALLGATHER="allgather",
            CAPACITY_AND_EXPERT_DENSITY="capacity_density",
            FUSED_OR_CAPACITY="fused_or_capacity",
            CAPACITY_AND_WORLD_SIZE="capacity_world",
        ),
        get_current_hardware_profile=lambda: NS(moe_comm_policy="capacity_density"),
        logger=Mock(),
    )
    functions(
        "vllm_ascend/ascend_forward_context.py",
        ["select_moe_comm_method", "_select_capacity_and_expert_density_moe_comm_method"],
        ns,
    )
    ns["_MOE_COMM_SELECTORS"] = {"capacity_density": ns["_select_capacity_and_expert_density_moe_comm_method"]}
    return ns


@pytest.mark.parametrize("tokens", [None, 0, 1, 6, 256, 257, 512, 513, 4096, 4097])
def test_validated_p_target_always_selects_fused_mc2(tokens):
    ns = selector_namespace()
    assert ns["select_moe_comm_method"](tokens, config()) == "fused"
    ns["get_mc2_tokens_capacity"].assert_not_called()
    ns["use_cann_megamoe"].assert_not_called()


@pytest.mark.parametrize("tokens", [0, 512, 513, 4096])
def test_draft_routes_stay_ordinary_on_ep8(tokens):
    ns = selector_namespace()
    select = ns["select_moe_comm_method"]
    # 48 experts per device on EP8 never satisfies the <=24 MC2 density rule.
    assert select(tokens, config(), is_draft_model=True) == "gather"
    draft = config()
    draft.model_config.runner_type = "draft"
    draft.parallel_config.prefill_context_parallel_size = 1
    assert select(tokens, draft) == "gather"


def w4a8_class(megamoe_fn):
    tree = ast.parse((ROOT / "vllm_ascend/quantization/methods/w4a8/w4a8.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendW4A8DynamicFusedMoEMethod")
    cls.decorator_list = []
    cfg = config()
    cfg.quant_config = NS(quant_description={"group_size": 0}, get_name=lambda: "ascend")
    ns = dict(
        torch=torch,
        torch_npu=NS(npu_format_cast=lambda weight, fmt: weight),
        ACL_FORMAT_FRACTAL_NZ=2,
        AscendMoEScheme=object,
        QuantType=NS(W4A8="w4a8"),
        MoEWeights=NS,
        ASCEND_QUANTIZATION_METHOD="ascend",
        COMPRESSED_TENSORS_METHOD="compressed",
        _MODELSLIM_SCALE_BIAS_PARTITIONS=16,
        _INT4_VALUES_PER_INT32=8,
        _INT4_VALUES_PER_BYTE=2,
        get_current_vllm_config=lambda: cfg,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_mc2_group=lambda: NS(),
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True), eplb_config=NS(dynamic_eplb=False)),
        a2_prefill_enabled=policy.a2_prefill_enabled,
        use_cann_megamoe=megamoe_fn,
        _EXTRA_CTX=NS(use_mega_moe=True),
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), "w4a8", "exec", flags=__future__.annotations.compiler_flag),
        ns,
    )
    return ns[cls.name], ns, cfg


def make_layer():
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
    return layer


def test_w4a8_target_uses_single_list_payload_for_both_backends():
    cls, _, _ = w4a8_class(lambda cfg: True)
    method = cls()
    assert method.use_expert_weight_list
    method.process_weights_after_loading_modelslim = lambda layer: None
    layer = make_layer()
    method.process_weights_after_loading(layer)
    assert not hasattr(layer, "w13_weight")
    payload = method.get_fused_mc2_weights(layer)
    ordinary_w1, _, ordinary_w2, _ = method._get_mlp_weights(layer)
    assert payload.w1[0].dtype == torch.int8 and ordinary_w1[0].dtype == torch.int32
    assert payload.w1[0].untyped_storage().data_ptr() == ordinary_w1[0].untyped_storage().data_ptr()
    assert payload.w2[0].untyped_storage().data_ptr() == ordinary_w2[0].untyped_storage().data_ptr()
    assert payload.w1_scale_bias[0].dtype == torch.float32
    assert len(method._get_mlp_scale_bias(layer)[0]) == 2


def test_w4a8_lazy_revaluation_saves_draft_layout_mismatch():
    # Scheme construction happens before the EP group is ready for the internal
    # DSpark draft, so use_cann_megamoe() misses there.
    megamoe = Mock(return_value=False)
    cls, _, cfg = w4a8_class(megamoe)
    method = cls()
    assert not method.use_expert_weight_list
    method.process_weights_after_loading_modelslim = lambda layer: None
    layer = make_layer()

    # By weight finalization the EP group exists and MegaMoe is selected.
    megamoe.return_value = True
    method.process_weights_after_loading(layer)
    assert method.use_expert_weight_list
    assert hasattr(layer, "w13_weight_list") and not hasattr(layer, "w13_weight")
    # Ordinary GMM views used by draft forwards must still resolve.
    ordinary_w1, _, _, _ = method._get_mlp_weights(layer)
    assert ordinary_w1[0].dtype == torch.int32


def test_w4a8_packed_layout_kept_when_megamoe_absent():
    cls, _, _ = w4a8_class(lambda cfg: False)
    method = cls()
    assert not method.use_expert_weight_list
    method.process_weights_after_loading_modelslim = lambda layer: None
    layer = make_layer()
    method.process_weights_after_loading(layer)
    assert hasattr(layer, "w13_weight") and not hasattr(layer, "w13_weight_list")
    assert layer.w13_weight.dtype == torch.int32


def method_from_class(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), path, "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace


def routed_forward_namespace(current_config):
    stop = RuntimeError("stop after route resolution")
    select = Mock(return_value="gather")
    context = NS(moe_comm_type="fused", use_mega_moe=True, in_profile_run=False, max_tokens_across_dp=32)
    get_comm = Mock(return_value=NS(prepare=Mock(side_effect=stop)))
    ns = dict(
        torch=torch,
        get_forward_context=lambda: NS(),
        _EXTRA_CTX=context,
        get_ascend_config=lambda: NS(a2_mega_moe_prefill=NS(enabled=True)),
        a2_prefill_enabled=policy.a2_prefill_enabled,
        is_draft_model=policy.is_draft_model,
        get_current_vllm_config_or_none=lambda: current_config,
        select_moe_comm_method=select,
        get_moe_comm_method=get_comm,
        QuantType=NS(NONE="bf16"),
        get_current_hardware_profile=lambda: NS(supports=lambda capability: False),
        HardwareCapability=NS(CANN_MEGAMOE_MXFP="mxfp"),
    )
    method_from_class("vllm_ascend/ops/fused_moe/routed_experts.py", "AscendRoutedExperts", "forward_impl", ns)
    return ns, select, get_comm, stop


def _routed_layer():
    return NS(
        enable_npugraph_ex_static_kernel=False,
        quant_type="w4a8",
        moe_config=NS(is_sequence_parallel=False),
        router=NS(),
        _use_v2_model_runner=False,
        dynamic_eplb=False,
        return_with_event=False,
    )


def test_routed_forward_without_config_context_keeps_platform_route():
    # Real v2 worker forwards have no current-vllm-config contextvar; the
    # platform forward-context hook already selected the route.
    ns, select, get_comm, stop = routed_forward_namespace(None)
    with pytest.raises(RuntimeError, match="stop after route"):
        ns["forward_impl"](_routed_layer(), hidden_states=torch.ones(32, 8), router_logits=torch.ones(32, 128))
    select.assert_not_called()
    get_comm.assert_called_once_with("fused", ANY)


def test_routed_forward_draft_config_reselects_ordinary_route():
    draft = config()
    draft.parallel_config.prefill_context_parallel_size = 1
    ns, select, get_comm, stop = routed_forward_namespace(draft)
    with pytest.raises(RuntimeError, match="stop after route"):
        ns["forward_impl"](_routed_layer(), hidden_states=torch.ones(32, 8), router_logits=torch.ones(32, 128))
    select.assert_called_once_with(32, draft, is_draft_model=True, draft_moe_quant_type="w4a8")
    get_comm.assert_called_once_with("gather", ANY)


def test_routed_forward_target_config_keeps_fused_route():
    ns, select, get_comm, stop = routed_forward_namespace(config())
    with pytest.raises(RuntimeError, match="stop after route"):
        ns["forward_impl"](_routed_layer(), hidden_states=torch.ones(32, 8), router_logits=torch.ones(32, 128))
    select.assert_not_called()
    get_comm.assert_called_once_with("fused", ANY)
