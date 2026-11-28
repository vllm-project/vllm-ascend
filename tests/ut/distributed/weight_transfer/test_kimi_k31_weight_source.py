# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file

from tests.e2e.common.rlhf.weight_transfer.case import KIMI_K31_CASE, pytest_model_cases
from tests.e2e.common.rlhf.weight_transfer.weight_source import make_weight_source


def test_kimi_k31_unpacked_checkpoint_matches_transfer(tmp_path):
    # Small widths keep this CPU test cheap while retaining every checkpoint name.
    overrides = KIMI_K31_CASE.hf_overrides["text_config"]
    config = SimpleNamespace(
        **(
            overrides
            | {
                "linear_attn_config": overrides["linear_attn_config"] | {"num_heads": 2, "head_dim": 2},
                "dsa_config": overrides["dsa_config"] | {"index_num_heads": 2, "index_head_dim": 2},
            }
        ),
        hidden_size=8,
        vocab_size=16,
        intermediate_size=12,
        num_attention_heads=2,
        first_k_dense_replace=1,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        moe_intermediate_size=4,
        num_shared_experts=2,
        routed_expert_hidden_size=4,
    )
    case = replace(KIMI_K31_CASE, hf_overrides={})
    with patch(
        "tests.e2e.common.rlhf.weight_transfer.weight_source.AutoConfig.from_pretrained",
        return_value=SimpleNamespace(text_config=config),
    ):
        source = make_weight_source(case, torch.device("cpu"))

    asset = tmp_path / "asset"
    asset.write_text("{}", encoding="utf-8")
    target = tmp_path / "reference"
    with patch("huggingface_hub.hf_hub_download", return_value=str(asset)):
        source.write_checkpoint(target)
    reference = load_file(str(target / "model.safetensors"))
    assert len(reference) == len(source.metadata) == 183
    assert reference.keys() == {meta.name for meta in source.metadata}
    for name, value in source.iter_full():
        torch.testing.assert_close(reference[name], value, rtol=0, atol=0)

    kda = "language_model.model.layers.0.self_attn."
    assert reference[kda + "A_log"].shape == (128,)
    assert reference[kda + "q_conv1d.weight"].shape == (4, 1, 4)
    assert not torch.equal(reference[kda + "q_conv1d.weight"], reference[kda + "k_conv1d.weight"])
    indexer = "language_model.model.layers.3.self_attn.indexer."
    assert reference[indexer + "wq_b.weight"].shape == (4, 4)
    assert reference[indexer + "wk.weight"].shape == (2, 8)
    assert reference[indexer + "weights_proj.weight"].shape == (2, 8)
    assert reference[indexer + "k_norm.bias"].shape == (2,)
    assert not any("ascend_conv1d_weight" in name for name in reference)


def test_kimi_k31_only_selects_unpacked_ipc():
    cases = (KIMI_K31_CASE,)
    (param,) = pytest_model_cases("npu_ipc", model_cases=cases)
    assert param.values == (KIMI_K31_CASE, False)
    assert all(mark.name != "skip" for mark in param.marks)
    assert pytest_model_cases("hccl", model_cases=cases) == []
