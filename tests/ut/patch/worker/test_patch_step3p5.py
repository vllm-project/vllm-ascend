# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn
from vllm.model_executor.models import step3p5

from vllm_ascend.patch.worker import patch_step3p5


@pytest.fixture
def layer_factory(monkeypatch):
    # Keep the real upstream attention constructor and config helpers. Only
    # replace device/distributed boundaries, so obsolete kwargs still fail.
    for name in ("QKVParallelLinear", "RowParallelLinear", "ColumnParallelLinear", "GemmaRMSNorm"):
        monkeypatch.setattr(step3p5, name, lambda *args, **kwargs: nn.Identity())
    rope = Mock(side_effect=lambda **kwargs: nn.Identity())
    attention = Mock(side_effect=lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(step3p5, "get_rope", rope)
    monkeypatch.setattr(step3p5, "Attention", attention)
    monkeypatch.setattr(step3p5, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(patch_step3p5, "get_tp_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(patch_step3p5, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(patch_step3p5, "GemmaRMSNorm", lambda *args: nn.Identity())
    moe = Mock(side_effect=lambda *args, **kwargs: nn.Identity())
    dense = Mock(side_effect=lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(patch_step3p5, "FusedMoEBlock", moe)
    monkeypatch.setattr(patch_step3p5, "Step3p5MLP", dense)

    def make(config, layer_idx, *, tp=2, pp=1, ep=True):
        monkeypatch.setattr(step3p5, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(patch_step3p5, "get_tensor_model_parallel_world_size", lambda: tp)
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=config),
            cache_config=SimpleNamespace(sliding_window=256),
            quant_config=None,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=tp, pipeline_parallel_size=pp, enable_expert_parallel=ep
            ),
        )
        layer = step3p5.Step3p5DecoderLayer(vllm_config, prefix=f"model.layers.{layer_idx}")
        assert vllm_config.cache_config.sliding_window is None
        return layer

    return make, rope, attention, moe, dense


def legacy_config():
    return SimpleNamespace(
        att_impl_type="GQA",
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_attention_groups=2,
        head_dim=4,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        num_hidden_layers=2,
        layer_types=["full_attention", "sliding_attention"],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 4,
        },
        sliding_window=128,
        rope_theta=[10000, 20000],
        rope_scaling={"rope_type": "yarn", "factor": 2.0},
        yarn_only_types=["full_attention"],
        partial_rotary_factors=[1.0, 0.5],
        use_rope_layers=[True, False],
        use_head_wise_attn_gate=True,
        moe_layers_enum="1",
    )


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_remote_config_uses_current_attention_signature(layer_factory, layer_idx):
    make, rope, attention, moe, dense = layer_factory
    config = legacy_config()
    layer = make(config, layer_idx)

    expected_rope = {"rope_type": "yarn", "factor": 2.0} if layer_idx == 0 else {"rope_type": "default"}
    expected_rope.update(rope_theta=config.rope_theta[layer_idx], partial_rotary_factor=[1.0, 0.5][layer_idx])
    assert rope.call_args.kwargs["rope_parameters"] == expected_rope
    assert layer.self_attn.total_num_heads == [8, 4][layer_idx]
    assert layer.self_attn.use_rope is (layer_idx == 0)
    assert layer.self_attn.use_head_wise_attn_gate
    assert attention.call_args.kwargs["per_layer_sliding_window"] == (128 if layer_idx else None)
    assert layer.use_moe is (layer_idx == 1)
    assert layer.use_sequence_parallel_moe is (layer_idx == 1)
    assert layer.self_attn.o_proj.reduce_results is (layer_idx == 0)
    if layer_idx:
        assert moe.call_args.kwargs["is_sequence_parallel"] is True
        dense.assert_not_called()
    else:
        dense.assert_called_once()
        moe.assert_not_called()


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_native_config_and_trailing_mtp_layer(layer_factory, layer_idx):
    make, rope, attention, moe, dense = layer_factory
    config = legacy_config()
    del config.att_impl_type
    del config.moe_layers_enum
    config.mtp_layer_types = ["sliding_attention"]
    config.mlp_layer_types = ["sparse", "dense"]
    config.mtp_mlp_layer_types = ["sparse"]
    config.use_rope_layers = None
    config.num_sliding_attention_heads = 6
    config.rope_parameters = {
        "full_attention": {"rope_type": "default", "rope_theta": 30000},
        "sliding_attention": {"rope_type": "default", "rope_theta": 40000, "partial_rotary_factor": 0.5},
    }
    layer = make(config, layer_idx)

    assert layer.self_attn.total_num_heads == (8 if layer_idx == 0 else 6)
    assert layer.self_attn.rope_theta == (30000 if layer_idx == 0 else 40000)
    assert layer.self_attn.use_rope
    assert layer.use_moe is (layer_idx != 1)
    assert layer.self_attn.o_proj.reduce_results is (layer_idx == 1)
    assert "partial_rotary_factor" not in config.rope_parameters["full_attention"]


@pytest.mark.parametrize("tp,pp,ep", [(1, 1, True), (2, 2, True), (2, 1, False)])
def test_sequence_parallel_guard_is_preserved(layer_factory, tp, pp, ep):
    make, _, _, moe, _ = layer_factory
    layer = make(legacy_config(), 1, tp=tp, pp=pp, ep=ep)
    assert layer.use_moe
    assert not layer.use_sequence_parallel_moe
    assert layer.self_attn.o_proj.reduce_results
    assert moe.call_args.kwargs["is_sequence_parallel"] is False


def test_unsupported_attention_fails_before_construction(layer_factory):
    make, rope, attention, _, _ = layer_factory
    config = legacy_config()
    config.att_impl_type = "unsupported"
    with pytest.raises(ValueError, match="Unsupported attention implementation: unsupported"):
        make(config, 0)
    rope.assert_not_called()
    attention.assert_not_called()
