# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi checkpoint geometry and the payload shared by startup and live reload."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file
from transformers import PretrainedConfig

from tests.e2e.common.rlhf.weight_transfer.case import KIMI_K3_CASE, QWEN35_CASE, pytest_model_cases
from tests.e2e.common.rlhf.weight_transfer.kimi_k3 import kimi_k3_meta_model
from tests.e2e.common.rlhf.weight_transfer.weight_source import FixedRandomWeightSource


def _text_config(*, tiny=False):
    # Published Kimi-K3 widths; only counts/schedules follow the reduced case.
    values = dict(
        hidden_size=7168,
        vocab_size=163840,
        intermediate_size=33792,
        num_attention_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        moe_intermediate_size=3072,
        num_shared_experts=2,
        routed_expert_hidden_size=3584,
        latent_moe_use_norm=True,
        **KIMI_K3_CASE.hf_overrides["text_config"],
    )
    if tiny:
        values.update(
            hidden_size=8,
            vocab_size=16,
            intermediate_size=12,
            num_attention_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            moe_intermediate_size=4,
            routed_expert_hidden_size=4,
            linear_attn_config={
                **values["linear_attn_config"],
                "num_heads": 2,
                "head_dim": 2,
            },
        )
    return PretrainedConfig(**values)


def test_checkpoint_geometry_covers_kda_mla_moe_and_residuals():
    model = kimi_k3_meta_model(_text_config())
    shapes = {name: tuple(parameter.shape) for name, parameter in model.named_parameters()}
    assert len(shapes) == 178
    assert all(parameter.is_meta for parameter in model.parameters())
    assert shapes["model.embed_tokens.weight"] == shapes["lm_head.weight"] == (163840, 7168)
    for index in range(3):
        prefix = f"model.layers.{index}.self_attn."
        for projection in ("q", "k", "v"):
            assert shapes[prefix + f"{projection}_proj.weight"] == (12288, 7168)
            assert shapes[prefix + f"{projection}_conv1d.weight"] == (12288, 1, 4)
        assert shapes[prefix + "A_log"] == (96,)
        assert shapes[prefix + "dt_bias"] == (12288,)
        assert shapes[prefix + "f_a_proj.weight"] == (128, 7168)
        assert shapes[prefix + "f_b_proj.weight"] == (12288, 128)
        assert shapes[prefix + "b_proj.weight"] == (96, 7168)
        assert shapes[prefix + "o_norm.weight"] == (128,)
    assert shapes["model.layers.3.self_attn.q_b_proj.weight"] == (18432, 1536)
    assert shapes["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"] == (576, 7168)
    assert shapes["model.layers.3.self_attn.kv_b_proj.weight"] == (24576, 512)
    assert shapes["model.layers.0.mlp.down_proj.weight"] == (7168, 33792)
    for index in (1, 2, 3):
        prefix = f"model.layers.{index}.block_sparse_moe."
        assert shapes[prefix + "gate.e_score_correction_bias"] == (8,)
        for expert in range(8):
            assert shapes[prefix + f"experts.{expert}.w1.weight"] == (3072, 3584)
            assert shapes[prefix + f"experts.{expert}.w2.weight"] == (3584, 3072)
            assert shapes[prefix + f"experts.{expert}.w3.weight"] == (3072, 3584)
        assert shapes[prefix + "shared_experts.gate_proj.weight"] == (6144, 7168)
        assert shapes[prefix + "routed_expert_down_proj.weight"] == (3584, 7168)
        assert shapes[prefix + "routed_expert_up_proj.weight"] == (7168, 3584)
        assert shapes[prefix + "routed_expert_norm.weight"] == (3584,)
    for index in range(4):
        prefix = f"model.layers.{index}."
        assert shapes[prefix + "self_attention_res_norm.weight"] == (7168,)
        assert shapes[prefix + "mlp_res_proj.weight"] == (1, 7168)
    assert shapes["model.output_attn_res_proj.weight"] == (1, 7168)
    assert not any("ascend_conv1d_weight" in name or name.startswith("language_model.") for name in shapes)
    assert not any("vision" in name or "mm_projector" in name for name in shapes)


@pytest.fixture
def source():
    config = SimpleNamespace(text_config=_text_config(tiny=True))
    case = replace(KIMI_K3_CASE, hf_overrides={})
    with (
        patch("tests.e2e.common.rlhf.weight_transfer.weight_source.AutoConfig.from_pretrained", return_value=config),
        patch("tests.e2e.common.rlhf.weight_transfer.weight_source.AutoModelForCausalLM.from_config") as auto_model,
    ):
        result = FixedRandomWeightSource(case, torch.device("cpu"))
    auto_model.assert_not_called()
    return result


def test_source_metadata_and_reiteration_are_identical(source):
    first = dict(source)
    second = dict(source.iter_full())
    assert len(first) == len(source.metadata) == 178
    assert list(first) == [meta.name for meta in source.metadata]
    for meta in source.metadata:
        tensor = first[meta.name]
        assert tensor.shape == meta.shape
        assert tensor.dtype == meta.dtype == torch.bfloat16
        torch.testing.assert_close(tensor, second[meta.name], rtol=0, atol=0)
        assert torch.isfinite(tensor).all()
    # Seeds depend on the full name, including each convolution shard and expert.
    assert not torch.equal(
        first["model.layers.0.self_attn.q_conv1d.weight"], first["model.layers.0.self_attn.k_conv1d.weight"]
    )
    assert not torch.equal(
        first["model.layers.1.block_sparse_moe.experts.0.w1.weight"],
        first["model.layers.1.block_sparse_moe.experts.1.w1.weight"],
    )


def test_startup_checkpoint_contains_exactly_the_transfer_payload(source, tmp_path):
    config_file = tmp_path / "published-config.json"
    config_file.write_text('{"model_type": "kimi_k3"}\n', encoding="utf-8")
    target = tmp_path / "checkpoint"
    with patch("huggingface_hub.hf_hub_download", return_value=str(config_file)):
        source.write_checkpoint(target)
    assert (target / "config.json").read_bytes() == config_file.read_bytes()
    checkpoint = load_file(str(target / "model.safetensors"))
    payload = dict(source)
    assert checkpoint.keys() == payload.keys()
    for name, tensor in payload.items():
        torch.testing.assert_close(checkpoint[name], tensor, rtol=0, atol=0)


def test_existing_cases_keep_the_transformers_meta_model_path():
    config = SimpleNamespace(text_config=PretrainedConfig())
    model = torch.nn.Linear(4, 2, bias=False, device="meta")
    with (
        patch("tests.e2e.common.rlhf.weight_transfer.weight_source.AutoConfig.from_pretrained", return_value=config),
        patch(
            "tests.e2e.common.rlhf.weight_transfer.weight_source.AutoModelForCausalLM.from_config", return_value=model
        ) as auto_model,
    ):
        source = FixedRandomWeightSource(QWEN35_CASE, torch.device("cpu"))
    auto_model.assert_called_once_with(config.text_config, trust_remote_code=True)
    assert [(meta.name, meta.shape) for meta in source.metadata] == [("weight", (2, 4))]


def test_kimi_case_runs_only_unpacked_ipc_without_skips():
    parameters = pytest_model_cases("npu_ipc", model_cases=(KIMI_K3_CASE,))
    assert len(parameters) == 1
    assert parameters[0].values == (KIMI_K3_CASE, False)
    assert not any(mark.name == "skip" for mark in parameters[0].marks)
    assert pytest_model_cases("hccl", model_cases=(KIMI_K3_CASE,)) == []
    text = KIMI_K3_CASE.hf_overrides["text_config"]
    assert text["num_experts_per_token"] <= text["num_experts"]
    assert text["quantization_config"] is None
    assert KIMI_K3_CASE.hf_overrides["architectures"] == ["KimiLinearForCausalLM"]
