# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.models import deepseek_mtp
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo
from vllm.transformers_utils import config as config_module

from vllm_ascend.models import register_model
from vllm_ascend.patch.dots3_note_audio import prepare_audio_features
from vllm_ascend.patch.dots3_note_config import (
    Dots3NoteAudioConfig,
    Dots3NoteConfig,
    Dots3NoteVisionConfig,
)
from vllm_ascend.patch.dots3_note_model import (
    Dots3NoteForCausalLM,
    Dots3NoteProcessingInfo,
    DotsMoEVisionTransformer,
    _get_dots3_note_audio_config_path,
    _get_dots3_note_vision_config_path,
    _get_dots_image_processor,
)
from vllm_ascend.patch.dots3_note_mtp import (
    Dots3NoteMTPModel,
    Dots3NoteMultiTokenPredictor,
)
from vllm_ascend.patch.platform import patch_dots3_note


def test_dots3_note_config_and_models_are_registered():
    register_model()

    assert config_module._CONFIG_REGISTRY["dots3_note"] is Dots3NoteConfig
    assert MODELS_CONFIG_MAP["Dots3NoteForCausalLM"] is patch_dots3_note.Dots3NoteForCausalLMConfig
    assert ModelRegistry.models["Dots3NoteForCausalLM"].module_name == ("vllm_ascend.patch.dots3_note_model")
    assert ModelRegistry.models["Dots3NoteMTPModel"].module_name == ("vllm_ascend.patch.dots3_note_mtp")

    config = Dots3NoteConfig(
        vision_config={"embed_dim": 32},
        audio_config={"sampling_rate": 16000},
    )
    assert isinstance(config.vision_config, Dots3NoteVisionConfig)
    assert isinstance(config.audio_config, Dots3NoteAudioConfig)


def test_dots3_note_config_maps_release_fields():
    config = Dots3NoteConfig(
        architectures=["Dots3NoteForCausalLM"],
        layer_types=["full_attention", "sliding_attention"],
        attention_gate_type="headwise",
        swa_attention_gate_type="elementwise",
        sliding_window_size=512,
        swa_q_lora_rank=1024,
        swa_qk_rope_head_dim=64,
        swa_v_head_dim=128,
    )

    assert config.model_type == "dots3_note"
    assert config.architectures == ["Dots3NoteForCausalLM"]
    assert config.sdpa_gate_type == "headwise"
    assert config.swa_attention_gate_type == "elementwise"
    assert config.sliding_window == 512
    assert config.use_sliding_window is True
    assert config.num_nextn_predict_layers == 1


def test_dots3_note_detects_fused_tower_configs(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"vision_config": {}, "audio_config": {}}')

    assert _get_dots3_note_vision_config_path(str(tmp_path)) == str(config_path)
    assert _get_dots3_note_audio_config_path(str(tmp_path)) == str(config_path)


def test_dots3_note_loads_nested_vision_processor_config(tmp_path):
    (tmp_path / "preprocessor_config.json").write_text(
        '{"vision_config":{"patch_size":14,"temporal_patch_size":1,'
        '"merge_size":2,"min_pixels":3136,"max_pixels":1016064}}'
    )

    processor = _get_dots_image_processor(str(tmp_path))

    assert processor.patch_size == 14
    assert processor.temporal_patch_size == 1
    assert processor.merge_size == 2
    assert processor.size["longest_edge"] == 1016064


def test_dots3_note_model_config_uses_raw_video_backend():
    model_config = MagicMock()
    model_config.multimodal_config.media_io_kwargs = {}

    patch_dots3_note.Dots3NoteForCausalLMConfig.verify_and_update_model_config(model_config)

    assert model_config.multimodal_config.media_io_kwargs["video"] == {
        "num_frames": 1,
        "video_backend": "nemotron_vl",
    }


def test_dots3_note_processing_info_reuses_qwen2vl_image_accounting():
    assert Dots3NoteProcessingInfo._get_vision_info is Qwen2VLProcessingInfo._get_vision_info

    ctx = MagicMock()
    ctx.get_merged_mm_kwargs.side_effect = lambda kwargs: kwargs
    info = Dots3NoteProcessingInfo(ctx)
    info.get_hf_config = MagicMock(
        return_value=SimpleNamespace(
            vision_config=SimpleNamespace(
                patch_size=14,
                spatial_merge_size=2,
                temporal_patch_size=2,
            )
        )
    )

    image_size, num_tokens = info._get_vision_info(
        image_width=56,
        image_height=56,
        image_processor=SimpleNamespace(size={"shortest_edge": 56 * 56, "longest_edge": 56 * 56}),
        mm_kwargs={},
    )

    assert (image_size.width, image_size.height) == (56, 56)
    assert num_tokens == 4


def test_dots3_note_audio_features_follow_checkpoint_contract():
    config = Dots3NoteAudioConfig(
        whisper_config={"num_mel_bins": 128},
        chunk_seconds=60,
        sampling_rate=16000,
        merge_factor=1,
    )

    outputs = prepare_audio_features([torch.zeros(16000)], config)

    assert outputs["audio_features"].shape == (1, 128, 6000)
    assert outputs["audio_sample_lens"].tolist() == [16000]
    assert outputs["audio_segment_counts"].tolist() == [1]
    assert outputs["audio_token_lengths"].tolist() == [13]


def test_dots3_note_mtp_reuses_deepseek_loader_with_dots3_note_weight_layout():
    assert issubclass(Dots3NoteMTPModel, deepseek_mtp.DeepSeekMTP)
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=2,
            num_nextn_predict_layers=3,
            use_dedicated_mtp_embeddings=True,
        ),
        model=SimpleNamespace(mtp_start_layer_idx=2, mtp_head_sharing="full"),
    )
    weights = [
        ("model.embed_tokens.weight", torch.tensor(0)),
        ("model.mtp.embed_tokens.weight", torch.tensor(1)),
        ("lm_head.weight", torch.tensor(2)),
        ("model.layers.3.enorm.weight", torch.tensor(3)),
    ]

    prepared = list(Dots3NoteMTPModel._prepare_weights(model, weights))

    assert [(name, weight.item()) for name, weight in prepared] == [
        ("model.layers.2.embed_tokens.weight", 1),
        ("model.layers.2.shared_head.head.weight", 2),
    ]


def test_dots3_note_mtp3_reuses_one_physical_layer():
    predictor = SimpleNamespace(
        mtp_head_sharing="full",
        mtp_start_layer_idx=46,
        num_mtp_layers=1,
    )

    physical_layers = [
        Dots3NoteMultiTokenPredictor._get_physical_layer_idx(predictor, spec_step_idx) for spec_step_idx in range(3)
    ]

    assert physical_layers == [46, 46, 46]


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


def test_dots3_note_load_weights_filters_towers_from_language_model(monkeypatch, tmp_path):
    vision_path = tmp_path / "model-vision.safetensors"
    audio_path = tmp_path / "model-audio.safetensors"
    vision_path.touch()
    audio_path.touch()
    checkpoint_weights = {
        str(vision_path): {"vision_encoder.patch_embed.weight": torch.tensor(4)},
        str(audio_path): {"audio_encoder.conv.weight": torch.tensor(5)},
    }

    class FakeSafeOpen:
        def __init__(self, path, **_kwargs):
            self.weights = checkpoint_weights[path]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return self.weights.keys()

        def get_tensor(self, key):
            return self.weights[key]

    captured = {}

    def load_language(weights):
        captured["language"] = [(name, weight.item()) for name, weight in weights]
        return {name for name, _weight in captured["language"]}

    def load_tower(name):
        def load(weights):
            captured[name] = [(weight_name, weight.item()) for weight_name, weight in weights]
            return {weight_name for weight_name, _weight in captured[name]}

        return load

    monkeypatch.setattr("vllm_ascend.patch.dots3_note_model.safe_open", FakeSafeOpen)
    model = Dots3NoteForCausalLM.__new__(Dots3NoteForCausalLM)
    nn.Module.__init__(model)
    model.model_path = str(tmp_path)
    model.language_model = SimpleNamespace(load_weights=load_language)
    model.vision_tower = SimpleNamespace(load_weights=load_tower("vision"))
    model.audio_tower = SimpleNamespace(load_weights=load_tower("audio"))

    loaded = model.load_weights(
        iter(
            [
                ("audio_encoder.conv.weight", torch.tensor(0)),
                ("vision_encoder.patch_embed.weight", torch.tensor(1)),
                ("model.layers.0.weight", torch.tensor(2)),
                ("lm_head.weight", torch.tensor(3)),
            ]
        )
    )

    assert captured == {
        "language": [("model.layers.0.weight", 2), ("lm_head.weight", 3)],
        "vision": [("patch_embed.weight", 4)],
        "audio": [("conv.weight", 5)],
    }
    assert loaded == {
        "language_model.model.layers.0.weight",
        "language_model.lm_head.weight",
        "vision_tower.patch_embed.weight",
        "audio_tower.conv.weight",
    }
