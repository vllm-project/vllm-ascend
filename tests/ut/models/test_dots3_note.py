# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.models import deepseek_v2
from vllm.transformers_utils.configs.dots3_note import Dots3NoteConfig

from vllm_ascend.models import register_model
from vllm_ascend.models.dots3_note.audio import prepare_audio_features
from vllm_ascend.models.dots3_note.model import Dots3NoteModel, get_dots3_note_layer_config
from vllm_ascend.models.dots3_note.multimodal import (
    Dots3NoteForCausalLM,
    Dots3NoteProcessingInfo,
    DotsMoEVisionTransformer,
)


def test_dots3_note_uses_upstream_config_and_processor():
    config = Dots3NoteConfig()

    assert config.model_type == "dots3_note"
    assert Dots3NoteProcessingInfo.__module__.endswith("_common.processor")


def test_dots3_note_models_are_registered_to_ascend_implementations():
    register_model()

    assert ModelRegistry.models["Dots3NoteForCausalLM"].module_name == ("vllm_ascend.models.dots3_note")


def test_dots3_note_audio_features_follow_checkpoint_contract():
    config = SimpleNamespace(
        chunk_seconds=60,
        merge_factor=1,
        sampling_rate=16000,
    )

    outputs = prepare_audio_features([torch.zeros(16000)], config)

    assert outputs["audio_features"].shape == (1, 128, 6000)
    assert outputs["audio_sample_lens"].tolist() == [16000]
    assert outputs["audio_segment_counts"].tolist() == [1]
    assert outputs["audio_token_lengths"].tolist() == [13]


def test_dots3_note_vision_weight_loader_requires_all_packed_shards():
    module = nn.Module()
    module.fc13 = nn.Linear(2, 4, bias=False)
    loaded_shards = []
    module.fc13.weight.weight_loader = lambda param, weight, shard_id: loaded_shards.append(shard_id)

    loaded = DotsMoEVisionTransformer.load_weights(
        module,
        [
            ("fc1.weight", torch.ones(2, 2)),
            ("fc3.weight", torch.ones(2, 2)),
        ],
    )

    assert loaded == {"fc13.weight"}
    assert loaded_shards == [0, 1]


def test_dots3_note_weight_mapper_routes_all_submodels():
    weights = [
        ("model.layers.0.weight", torch.tensor(0)),
        ("lm_head.weight", torch.tensor(1)),
        ("vision_encoder.patch_embed.weight", torch.tensor(2)),
        ("audio_encoder.conv.weight", torch.tensor(3)),
    ]

    mapped = list(Dots3NoteForCausalLM.hf_to_vllm_mapper.apply(weights))

    assert [name for name, _ in mapped] == [
        "language_model.model.layers.0.weight",
        "language_model.lm_head.weight",
        "visual.patch_embed.weight",
        "audio_tower.conv.weight",
    ]


def test_dots3_note_projects_sliding_layer_config():
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=16,
        q_lora_rank=1024,
        index_topk=2048,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        attention_gate_type="headwise",
        swa_num_attention_heads=8,
        swa_q_lora_rank=768,
        swa_qk_nope_head_dim=64,
        swa_qk_rope_head_dim=32,
        swa_v_head_dim=96,
        swa_attention_gate_type="elementwise",
        swa_kv_lora_rank=256,
        sliding_window_size=512,
        rope_theta=10_000,
        swa_rope_theta=1_000,
        layer_types=["full_attention", "sliding_attention"],
        moe_layer_freq=1,
        n_routed_experts=8,
    )

    projected = get_dots3_note_layer_config(config, 1)

    assert projected is not config
    assert projected.num_attention_heads == 8
    assert not hasattr(projected, "index_topk")
    assert projected.q_lora_rank == 768
    assert projected.qk_nope_head_dim == 64
    assert projected.qk_rope_head_dim == 32
    assert projected.v_head_dim == 96
    assert projected.kv_lora_rank == 256
    assert projected.attention_gate_type == "elementwise"
    assert projected.rope_parameters == {"rope_type": "default", "rope_theta": 1_000}
    assert projected.n_routed_experts == 8


def test_dots3_note_main_model_filters_mtp_weights(monkeypatch):
    model = Dots3NoteModel.__new__(Dots3NoteModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    captured: list[str] = []

    def load_weights(_self, weights):
        captured.extend(name for name, _ in weights)
        return set(captured)

    monkeypatch.setattr(deepseek_v2.DeepseekV2Model, "load_weights", load_weights)
    result = model.load_weights(
        [
            ("model.layers.0.input_layernorm.weight", torch.ones(1)),
            ("model.layers.2.self_attn.q_proj.weight", torch.ones(1)),
            ("model.mtp.embed_tokens.weight", torch.ones(1)),
            ("model.norm.weight", torch.ones(1)),
        ]
    )

    assert result == {
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
    }
