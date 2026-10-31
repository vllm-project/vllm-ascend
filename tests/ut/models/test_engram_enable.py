# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise production loading and execution with the Engram master switch off."""

from __future__ import annotations

import ast
import sys
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = "vllm_ascend/models/deepseek_v41/model.py"


def production(relative, name, namespace, *, methods=None):
    # Execute complete production methods while isolating unavailable NPU deps.
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    if methods is not None:
        node.bases = [ast.parse("torch.nn.Module", mode="eval").body]
        node.decorator_list = []
        node.body = [child for child in node.body if getattr(child, "name", None) in methods]
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec")
    exec(code, namespace)
    return namespace[name]


def method(relative, cls, name, namespace):
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    owner = next(node for node in tree.body if getattr(node, "name", None) == cls)
    node = next(node for node in owner.body if getattr(node, "name", None) == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "additional", [None, {}, {"engram_config": {}}, {"engram_config": {"enabled": False, "backend": "elastic"}}]
)
@pytest.mark.parametrize("native", [None, SimpleNamespace(cpu_offload=True, dp_shared_memory=True)])
def test_platform_disables_engram_before_auto_resolution_and_validation(additional, native):
    validate = production("vllm_ascend/platform.py", "_validate_engram_config", {})
    # No model/parallel attributes: none may be inspected when disabled.
    config = SimpleNamespace(additional_config=additional, engram_config=native)
    validate(config)
    assert config.engram_config is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("layers", [[], [0]])
@pytest.mark.parametrize("resolved", [False, True])
def test_runtime_requires_master_switch_layers_and_resolved_config(enabled, layers, resolved):
    helper = production(
        MODEL,
        "_engram_enabled_for_runtime",
        {
            "get_ascend_config": lambda: SimpleNamespace(engram_config=SimpleNamespace(enabled=enabled)),
            "engram_enabled": lambda config: bool(config.engram_layer_ids),
        },
    )
    assert helper(
        SimpleNamespace(engram_layer_ids=layers), SimpleNamespace(engram_config=object() if resolved else None)
    ) is bool(enabled and layers and resolved)


@pytest.mark.parametrize("backend", ["uva", "elastic"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_disabled_model_builds_and_runs_without_engram_resources(backend, compiled, sequence_parallel):
    forbidden = Mock(side_effect=AssertionError("Disabled Engram path was entered"))
    config = SimpleNamespace(
        vocab_size=16,
        hidden_size=4,
        index_topk=1,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        norm_eps=1e-6,
        hc_mult=2,
        hc_sinkhorn_iters=1,
        hc_eps=1e-6,
        rope_parameters={"original_max_position_embeddings": 64},
        engram_layer_ids=[0, 1],
    )
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, model="/missing/checkpoint"),
        quant_config=None,
        cache_config=None,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sequence_parallel),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=4),
        # Even a stale explicit config must not re-enable modules or tables.
        engram_config=SimpleNamespace(cpu_offload=True, dp_shared_memory=True),
    )
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "_DS41_WHEEL_OPS": False,
        "normalize_deepseek_v41_config": lambda config: config,
        "get_ascend_config": lambda: SimpleNamespace(
            engram_config=SimpleNamespace(
                enabled=False, backend=backend, dp_shared_memory=False, replicate_across_dp=False
            )
        ),
        "engram_enabled": lambda config: bool(config.engram_layer_ids),
        "current_platform": SimpleNamespace(device_type="cpu"),
        "get_pp_group": lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        "VocabParallelEmbedding": lambda vocab, dim, **kwargs: torch.nn.Embedding(vocab, dim),
        "RMSNorm": lambda *args, **kwargs: torch.nn.Identity(),
        "DeepseekV41MoE": lambda **kwargs: torch.nn.Identity(),
        "enable_dsa_cp": lambda: False,
        "build_layer_plan": lambda config: SimpleNamespace(candidate_topk_blocks=1),
        "DeepseekV41SharedAttentionState": lambda *args: SimpleNamespace(reset=lambda: None),
        "engram_cpu_offload": forbidden,
        "resolve_dp_shared_memory": forbidden,
        "get_rotation_path": forbidden,
        "preflight_engram_checkpoint": forbidden,
        "load_engram_rotation_block": forbidden,
        "create_engram_hash_state": forbidden,
        "EngramLayout": SimpleNamespace(from_config=forbidden),
        "AscendEngram": forbidden,
        "AscendParallelEngramEmbedding": forbidden,
        "is_forward_context_available": lambda: False,
        "envs": SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),
        "sp_shard": lambda tensor: tensor,
        "sp_all_gather": lambda tensor: tensor,
    }
    production(MODEL, "_engram_enabled_for_runtime", namespace)
    decoder = production(MODEL, "DeepseekV41DecoderLayer", namespace, methods={"__init__"})
    decoder.attention_cls = staticmethod(lambda **kwargs: SimpleNamespace(wo_b=SimpleNamespace(reduce_results=True)))

    def make_layers(count, factory, prefix):
        return 0, count, torch.nn.ModuleList(factory(f"{prefix}.{i}") for i in range(count))

    namespace["make_layers"] = make_layers
    cls = production(
        MODEL,
        "DeepseekV41Model",
        namespace,
        methods={
            "__init__",
            "_make_empty_intermediate_tensors",
            "embed_input_ids",
            "forward",
            "prepare_engram_inputs",
            "prepare_engram_graph_inputs",
            "prime_engram_v2_graph_inputs",
        },
    )
    cls.decoder_layer_cls = decoder
    model = cls(vllm_config=vc)
    assert not model.has_engram and model.engram_layout is None
    assert model.engram_rotation is None and model.engram_hash is None
    assert model._engram_elastic_runtime is None and model._engram_prepare_stream is None
    assert all(layer.engram is None for layer in model.layers)
    assert not any("engram" in name for name, _ in model.named_parameters())
    assert model.prepare_engram_inputs(None, None) == {}
    assert model.prepare_engram_graph_inputs(4) == {}
    assert model.prime_engram_v2_graph_inputs(4) == {}
    model.prepare_engram = forbidden
    model.aux_hidden_state_layers = ()
    for layer in model.layers:
        layer.forward = lambda positions, hidden, pre_mix, residual, **kwargs: (hidden + 1, pre_mix)
        layer.hc_collapse = lambda hidden, pre_mix: hidden.mean(1)
    forward = torch.compile(model, backend="eager", fullgraph=True) if compiled else model
    for ids in (torch.tensor([1, 2]), torch.tensor([3, 4])):
        actual = forward(ids, torch.arange(2), None)
        torch.testing.assert_close(actual, model.embed_input_ids(ids) + 2)
    forbidden.assert_not_called()


def test_disabled_checkpoint_loading_filters_all_engram_parameters():
    load = method(MODEL, "AscendDeepseekV41LLMForCausalLM", "load_weights", {})
    loader = Mock(side_effect=lambda weights: {name for name, _ in weights})
    model = SimpleNamespace(model=SimpleNamespace(has_engram=False), _load_model_weights=loader)
    names = ["model.layers.0.engram.embed.weight", "model.layers.0.engram.wk.weight", "model.layers.0.mlp.weight"]
    assert load(model, ((name, torch.zeros(1)) for name in names)) == {names[-1]}


@pytest.mark.parametrize("enabled", [False, True])
def test_model_state_selection_follows_actual_engram_modules(monkeypatch, enabled):
    default, engram = type("DefaultState", (), {}), type("EngramState", (), {})
    for name, attribute, value in (
        ("vllm_ascend.worker.v2.model_states.default", "AscendModelState", default),
        ("vllm_ascend.models.deepseek_v41.engram.model_state", "EngramModelState", engram),
    ):
        module = ModuleType(name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    select = method(MODEL, "AscendDeepseekV41LLMForCausalLM", "get_model_state_cls", {})
    assert select(SimpleNamespace(has_engram=enabled)) is (engram if enabled else default)


@pytest.mark.parametrize("mode", ["NONE", "FULL"])
def test_v1_skips_disabled_engram_preparation_and_retirement(mode):
    from vllm.config.compilation import CUDAGraphMode

    forbidden = Mock(side_effect=AssertionError("Engram hook was called"))
    model = Mock(return_value=torch.ones(1))
    model.has_engram = False
    model.prepare_engram_inputs = model.retire_engram_lookups = forbidden
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode[mode], device_metadata_executor=None)
    run = method(
        "vllm_ascend/worker/model_runner_v1.py",
        "NPUModelRunner",
        "_model_forward",
        {"torch": torch, "get_forward_context": lambda: context, "CUDAGraphMode": CUDAGraphMode, "partial": partial},
    )
    runner = SimpleNamespace(
        model=model, enable_enpu=False, attn_backend=object(), _update_full_graph_params_if_needed=lambda *args: None
    )
    torch.testing.assert_close(run(runner, 1), torch.ones(1))
    forbidden.assert_not_called()


def test_v1_disabled_elastic_does_not_allocate_placeholder_mask():
    sanitize = method(
        "vllm_ascend/worker/model_runner_v1.py",
        "NPUModelRunner",
        "_sanitize_placeholder_input_ids_for_forward",
        {"torch": torch, "PLACEHOLDER_TOKEN_ID": -1},
    )
    runner = SimpleNamespace(
        ascend_config=SimpleNamespace(engram_config=SimpleNamespace(enabled=False, backend="elastic")),
        input_ids=SimpleNamespace(gpu=torch.tensor([7, 8])),
    )
    sanitize(runner, SimpleNamespace(scheduled_spec_decode_tokens={}), 2)
    assert not hasattr(runner, "_engram_dead_input_mask")


def test_v2_default_state_skips_disabled_engram_hooks():
    forbidden = Mock(side_effect=AssertionError("Engram hook was called"))
    state = SimpleNamespace(
        model=SimpleNamespace(
            has_engram=False,
            prepare_engram_inputs=forbidden,
            prepare_engram_graph_inputs=forbidden,
        )
    )
    path = "vllm_ascend/worker/v2/model_states/default.py"
    assert method(path, "AscendModelState", "prepare_engram_inputs", {})(state, None, None) == {}
    assert method(path, "AscendModelState", "prepare_engram_dummy_inputs", {})(state, 0, 4) == {}
    forbidden.assert_not_called()
