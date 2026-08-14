# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.patch.worker import patch_deepseek_v2

_should_skip_indexer_init = patch_deepseek_v2._should_skip_indexer_init


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_mla_init_forwards_non_causal_multi_token_decode(monkeypatch):
    monkeypatch.setattr(
        patch_deepseek_v2,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    for name in (
        "ReplicatedLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "RMSNorm",
    ):
        monkeypatch.setattr(
            patch_deepseek_v2,
            name,
            MagicMock(return_value=object()),
        )
    monkeypatch.setattr(
        patch_deepseek_v2,
        "get_rope",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        patch_deepseek_v2,
        "MLAModules",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    wrapper = MagicMock(return_value=object())
    monkeypatch.setattr(
        patch_deepseek_v2,
        "MultiHeadLatentAttentionWrapper",
        wrapper,
    )

    patch_deepseek_v2._deepseek_v2_mla_attention_init(
        torch.nn.Module(),
        vllm_config=SimpleNamespace(),
        config=SimpleNamespace(
            rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default"},
        ),
        hidden_size=16,
        num_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=2,
        v_head_dim=4,
        q_lora_rank=None,
        kv_lora_rank=8,
        prefix="model.layers.0.self_attn",
        non_causal_multi_token_decode=True,
    )

    assert wrapper.call_args.kwargs["non_causal_multi_token_decode"] is True
