# SPDX-License-Identifier: Apache-2.0
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter, ValidationError

from vllm_ascend.expert_offload.config import ExpertOffloadConfig


def make_configs():
    vc = SimpleNamespace(
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            model="/models/target",
            enforce_eager=True,
            enable_sleep_mode=False,
            hf_text_config=SimpleNamespace(model_type="deepseek_v4"),
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=1),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            enable_dbo=False,
        ),
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0), prefetch=SimpleNamespace(offload_group_size=0)
        ),
        load_config=SimpleNamespace(load_format="auto"),
    )
    ac = SimpleNamespace(
        mix_placement=False,
        enable_force_eplb=False,
        enable_fused_mc2=0,
        multistream_overlap_shared_expert=False,
        eplb_config=SimpleNamespace(dynamic_eplb=False),
        rl_config=SimpleNamespace(enabled=False),
    )
    return vc, ac


def make_dspark():
    return SimpleNamespace(
        method="dspark",
        enforce_eager=True,
        enable_adaptive_verification=False,
        draft_model_config=SimpleNamespace(
            model="/models/target",
            hf_config=SimpleNamespace(model_type="deepseek_v4", architectures=["DSparkDraftModel"]),
        ),
    )


def test_checkpoint_dspark_with_eager_drafting_is_supported():
    vc, ac = make_configs()
    vc.speculative_config = make_dspark()
    ExpertOffloadConfig(enabled=True, num_device_experts=96).validate(vc, ac)


@pytest.mark.parametrize(
    "field,value", [("method", "mtp"), ("enforce_eager", False), ("enable_adaptive_verification", True)]
)
def test_unsupported_speculative_modes_are_rejected(field, value):
    vc, ac = make_configs()
    vc.speculative_config = make_dspark()
    setattr(vc.speculative_config, field, value)
    with pytest.raises(ValueError, match="speculative_config"):
        ExpertOffloadConfig(enabled=True, num_device_experts=96).validate(vc, ac)


def test_external_dspark_checkpoint_and_multi_request_drafting_are_rejected():
    vc, ac = make_configs()
    vc.speculative_config = make_dspark()
    vc.speculative_config.draft_model_config.model = "/models/other"
    with pytest.raises(ValueError, match="speculative_config"):
        ExpertOffloadConfig(enabled=True, num_device_experts=96).validate(vc, ac)
    vc.speculative_config = make_dspark()
    vc.scheduler_config.max_num_seqs = 2
    with pytest.raises(ValueError, match="max_num_seqs"):
        ExpertOffloadConfig(enabled=True, num_device_experts=96).validate(vc, ac)


def test_defaults_and_disabled_validation():
    cfg = ExpertOffloadConfig()
    assert not cfg.enabled and cfg.num_device_experts is None
    cfg.validate(None, None)
    cfg.validate_layer(0, 0)


@pytest.mark.parametrize("mode", ["NONE", "FULL_DECODE_ONLY"])
def test_explicit_eager_or_decode_graph_configuration(mode):
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    vc, ac = make_configs()
    vc.model_config.enforce_eager = False
    vc.compilation_config = SimpleNamespace(mode=CompilationMode.NONE, cudagraph_mode=getattr(CUDAGraphMode, mode))
    ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize("mode", ["PIECEWISE", "FULL", "FULL_AND_PIECEWISE"])
def test_unsupported_graph_modes_fail_early(mode):
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    vc, ac = make_configs()
    vc.model_config.enforce_eager = False
    vc.compilation_config = SimpleNamespace(mode=CompilationMode.NONE, cudagraph_mode=getattr(CUDAGraphMode, mode))
    with pytest.raises(ValueError, match="FULL_DECODE_ONLY"):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": True},
        {"enabled": "false"},
        {"enabled": 1},
        {"num_device_experts": True},
        {"num_device_experts": 0},
        {"num_device_experts": -1},
        {"num_device_experts": 2.5},
        {"num_device_experts": "4"},
        {"num_device_expert": 4},
    ],
)
def test_schema_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        TypeAdapter(ExpertOffloadConfig).validate_python(values)


def test_nested_dict_parsing_and_immutability():
    cfg = TypeAdapter(ExpertOffloadConfig).validate_python({"enabled": True, "num_device_experts": 4})
    cfg.validate(*make_configs())
    cfg.validate_layer(8, 4)
    with pytest.raises(FrozenInstanceError):
        cfg.enabled = False


@pytest.mark.parametrize("experts,top_k", [(4, 1), (3, 1), (8, 5), (8, 0)])
def test_layer_capacity_bounds(experts, top_k):
    with pytest.raises(ValueError, match="top_k"):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate_layer(experts, top_k)


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("use_v2_model_runner", False, "Runner V2"),
        ("model_config.enable_sleep_mode", True, "sleep mode"),
        ("model_config.hf_text_config.model_type", "qwen3_moe", "DeepSeek V4"),
        ("scheduler_config.async_scheduling", True, "async_scheduling"),
        ("parallel_config.tensor_parallel_size", 2, "tensor_parallel_size"),
        ("parallel_config.data_parallel_size", 2, "data_parallel_size"),
        ("parallel_config.pipeline_parallel_size", 2, "pipeline_parallel_size"),
        ("parallel_config.prefill_context_parallel_size", 2, "prefill_context_parallel_size"),
        ("parallel_config.decode_context_parallel_size", 2, "decode_context_parallel_size"),
        ("parallel_config.enable_expert_parallel", True, "enable_expert_parallel"),
        ("parallel_config.enable_eplb", True, "enable_eplb"),
        ("parallel_config.enable_dbo", True, "enable_dbo"),
        ("lora_config", object(), "lora_config"),
        ("speculative_config", object(), "speculative_config"),
        ("kv_transfer_config", object(), "kv_transfer_config"),
        ("offload_config.uva.cpu_offload_gb", 1, "generic CPU"),
        ("offload_config.prefetch.offload_group_size", 1, "generic CPU"),
        ("load_config.load_format", "dummy", "checkpoint loading"),
    ],
)
def test_unsupported_runner_combinations(path, value, match):
    vc, ac = make_configs()
    obj = vc
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    with pytest.raises(ValueError, match=match):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize(
    "name", ["mix_placement", "enable_force_eplb", "enable_fused_mc2", "multistream_overlap_shared_expert"]
)
def test_unsupported_ascend_combinations(name):
    vc, ac = make_configs()
    setattr(ac, name, True)
    with pytest.raises(ValueError, match=name):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)


@pytest.mark.parametrize("name", ["dynamic_eplb", "rl"])
def test_unsupported_weight_mutation(name):
    vc, ac = make_configs()
    if name == "rl":
        ac.rl_config.enabled = True
    else:
        ac.eplb_config.dynamic_eplb = True
    with pytest.raises(ValueError):
        ExpertOffloadConfig(enabled=True, num_device_experts=4).validate(vc, ac)
