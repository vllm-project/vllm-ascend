# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed-tensors rules must use the text-only MTP module names."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from torch import nn
from vllm.config import CompilationMode, get_current_vllm_config, set_current_vllm_config

from vllm_ascend.models.glm5next import mtp as mtp_module
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.quantization.configs.compressed_tensors_config import AscendCompressedTensorsConfig


def make_quant_config(scope):
    def args(bits, dynamic):
        return {
            "num_bits": bits,
            "type": "float",
            "symmetric": True,
            "group_size": 32,
            "strategy": "group",
            "dynamic": dynamic,
            "scale_dtype": "torch.uint8",
        }

    quant = AscendCompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "mixed-precision",
            "config_groups": {
                "experts": {
                    "targets": [f"{scope}layers.45.mlp.experts.0.{proj}_proj" for proj in ("gate", "up", "down")],
                    "weights": args(4, False),
                    "input_activations": args(4, True),
                    "format": "mxfp4-pack-quantized",
                },
                "linears": {
                    "targets": ["Linear"],
                    "weights": args(8, False),
                    "input_activations": args(8, True),
                    "format": "mxfp8-quantized",
                },
            },
            "ignore": [f"{scope}layers.45.self_attn.q_a_proj", "language_model.lm_head"],
        }
    )
    quant._add_fused_moe_to_target_scheme_map()
    return quant


def expert_quant_type(quant, name):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(layer)
    scheme = quant.get_scheme_dict(layer, name)
    return quant._detect_quant_type(scheme["weights"], scheme["input_activations"], scheme["format"])


@pytest.mark.parametrize("scope", ["model.language_model.", "language_model.model.", "model."])
@pytest.mark.parametrize("prefix", ["", "draft"])
def test_maps_expert_targets_and_ignores_without_mutating_target(scope, prefix):
    quant = make_quant_config(scope)
    config = SimpleNamespace(quant_config=quant, model_config=object())
    original = copy.deepcopy(quant)

    draft = mtp_module._prepare_mtp_vllm_config(config, prefix)

    root = f"{prefix}.model." if prefix else "model."
    for proj in ("gate", "up", "down"):
        assert expert_quant_type(draft.quant_config, f"{root}layers.45.mlp.experts.0.{proj}_proj") == "W4A4_MXFP4"
    # A missed ignore would silently select the generic MXFP8 Linear scheme.
    assert draft.quant_config.get_scheme_dict(nn.Linear(4, 4), f"{root}layers.45.self_attn.q_a_proj") is None
    assert draft.quant_config.get_scheme_dict(nn.Linear(4, 4), f"{prefix}.lm_head" if prefix else "lm_head") is None
    assert draft is not config
    assert draft.model_config is config.model_config
    assert draft.quant_config is not quant
    assert quant.target_scheme_map == original.target_scheme_map
    assert quant.ignore == original.ignore
    assert quant.quant_description == original.quant_description
    assert draft.quant_config.quant_description is not quant.quant_description
    assert draft.quant_config.target_scheme_map["Linear"] == quant.target_scheme_map["Linear"]
    assert draft.quant_config.target_scheme_map["Linear"] is not quant.target_scheme_map["Linear"]


def test_inherited_target_names_select_wrong_expert_precision():
    config = SimpleNamespace(quant_config=make_quant_config("language_model.model."))
    name = "model.layers.45.mlp.experts.0.gate_proj"
    # Before the fix, the explicit expert target misses and RoutedExperts
    # falls back to the generic Linear-derived MXFP8 scheme.
    assert expert_quant_type(config.quant_config, name) == "W8A8_MXFP8"

    draft = mtp_module._prepare_mtp_vllm_config(config)

    assert expert_quant_type(draft.quant_config, name) == "W4A4_MXFP4"


def test_explicit_only_expert_targets_match_after_remapping():
    quant = make_quant_config("language_model.model.")
    del quant.target_scheme_map["Linear"]
    del quant.target_scheme_map["RoutedExperts"]
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    nn.Module.__init__(layer)
    name = "model.layers.45.mlp.experts.0.gate_proj"
    assert quant.get_scheme_dict(layer, name) is None

    draft = mtp_module._prepare_mtp_vllm_config(SimpleNamespace(quant_config=quant))

    assert expert_quant_type(draft.quant_config, name) == "W4A4_MXFP4"


@pytest.mark.parametrize("lm_head", ["lm_head", "language_model.lm_head"])
@pytest.mark.parametrize("prefix", ["", "draft"])
def test_maps_lm_head_with_module_boundary(lm_head, prefix):
    quant = make_quant_config("model.")
    quant.ignore = [lm_head, f"{lm_head}.weight", "lm_head_extra", "vision_model.proj"]

    draft = mtp_module._prepare_mtp_vllm_config(SimpleNamespace(quant_config=quant), prefix)

    head = f"{prefix}.lm_head" if prefix else "lm_head"
    assert draft.quant_config.ignore == [head, f"{head}.weight", "lm_head_extra", "vision_model.proj"]


@pytest.mark.parametrize("quant", [None, SimpleNamespace(), SimpleNamespace(get_name=lambda: "ascend")])
def test_non_compressed_tensors_configs_are_unchanged(quant):
    config = SimpleNamespace(quant_config=quant)
    assert mtp_module._prepare_mtp_vllm_config(config) is config


def make_vllm_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace()),
        quant_config=make_quant_config("language_model.model."),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.mark.parametrize("prefix", ["", "draft"])
def test_constructor_uses_draft_config_and_restores_target_context(prefix):
    config = make_vllm_config()
    seen = {}
    expected_model_prefix = f"{prefix}.model" if prefix else "model"

    def predictor(*, vllm_config, prefix):
        seen["config"] = vllm_config
        assert get_current_vllm_config() is vllm_config
        assert prefix == expected_model_prefix
        expert_name = f"{prefix}.layers.45.mlp.experts.0.gate_proj"
        assert expert_quant_type(vllm_config.quant_config, expert_name) == "W4A4_MXFP4"
        return nn.Module()

    with (
        set_current_vllm_config(config),
        patch.object(mtp_module, "Glm5NextMultiTokenPredictor", side_effect=predictor),
        patch.object(mtp_module, "is_rot_weight_used", return_value=False),
        patch.object(mtp_module.Glm5NextMTP, "set_moe_parameters"),
    ):
        model = mtp_module.Glm5NextMTP(vllm_config=config, prefix=prefix)
        assert get_current_vllm_config() is config
        assert model.quant_config is seen["config"].quant_config
        assert model.quant_config is not config.quant_config
        assert "group_size" not in config.quant_config.quant_description


def test_constructor_restores_target_context_after_failure():
    config = make_vllm_config()

    def fail_predictor(**kwargs):
        assert get_current_vllm_config() is kwargs["vllm_config"]
        assert get_current_vllm_config() is not config
        raise RuntimeError("draft construction failed")

    with (
        set_current_vllm_config(config),
        patch.object(mtp_module, "Glm5NextMultiTokenPredictor", side_effect=fail_predictor),
    ):
        with pytest.raises(RuntimeError, match="draft construction failed"):
            mtp_module.Glm5NextMTP(vllm_config=config)
        assert get_current_vllm_config() is config
