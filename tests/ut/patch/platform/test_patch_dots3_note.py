import asyncio
from importlib import reload
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from transformers import PretrainedConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.renderers.online_renderer import OnlineRenderer

from vllm_ascend.core.kv_cache_interface import (
    AscendSFAIndexerCacheSpec,
    Dots3NoteMLAAttentionSpec,
    Dots3NoteSlidingWindowMLASpec,
)
from vllm_ascend.patch.platform import patch_dots3_note
from vllm_ascend.patch.platform.patch_kv_cache_utils import (
    _get_kv_cache_config_deepseek_v4,
    _get_kv_cache_groups_uniform_groups,
    group_and_unify_kv_cache_specs,
)


def test_mtp_config_override_uses_checkpoint_layout():
    hf_config = PretrainedConfig(
        architectures=["Dots3NoteForCausalLM"],
        model_type="dots3_note",
        num_hidden_layers=46,
        num_nextn_predict_layers=3,
        mtp_head_sharing=None,
    )
    indexed_weights = {
        "model.layers.46.self_attn.k_rope_only_layernorm.weight",
        "model.mtp.embed_tokens.weight",
    }
    with patch.object(
        patch_dots3_note,
        "_mtp_weight_is_indexed",
        side_effect=lambda _, weight_name: weight_name in indexed_weights,
    ):
        result = SpeculativeConfig.hf_config_override(hf_config)

    assert result is hf_config
    assert result.original_model_type == "dots3_note"
    assert result.model_type == "deepseek_mtp"
    assert result.architectures == ["Dots3NoteMTPModel"]
    assert result.n_predict == 3
    assert result.mtp_head_sharing == "full"
    assert result.k_rope_only_layernorm is True
    assert result.use_dedicated_mtp_embeddings is True


def test_dots3_note_kv_plan_allocates_sliding_layers():
    specs = {
        "full": Dots3NoteMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="auto",
        ),
        "indexer": AscendSFAIndexerCacheSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        ),
        **{
            f"sliding_{index}": Dots3NoteSlidingWindowMLASpec(
                block_size=128,
                num_kv_heads=1,
                head_size=1088,
                dtype=torch.bfloat16,
                cache_dtype_str="auto",
                sliding_window=513,
            )
            for index in range(3)
        },
    }
    grouped_specs = group_and_unify_kv_cache_specs(specs)
    assert grouped_specs is not None
    groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(num_gpu_blocks_override=None))

    _, tensors = _get_kv_cache_config_deepseek_v4(
        vllm_config,
        groups,
        available_memory=1 << 30,
    )

    allocated_layers = {layer_name for tensor in tensors for layer_name in tensor.shared_by}
    assert allocated_layers == set(specs)


def test_video_question_uses_last_video_user_message():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "first"}},
                {"type": "text", "text": "first question"},
            ],
        },
        {"role": "assistant", "content": "answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "last "},
                {"type": "video", "video": "bytes"},
                {"type": "text", "text": "question"},
            ],
        },
    ]

    assert patch_dots3_note._video_question(messages) == "last question"


def test_preprocess_chat_delegates_non_dots3_note_request(monkeypatch):
    original = AsyncMock(return_value="result")
    monkeypatch.setattr(patch_dots3_note, "_original_preprocess_chat", original)
    serving = SimpleNamespace(model_config=SimpleNamespace(architectures=["DeepseekV2ForCausalLM"]))
    request = object()
    messages = [{"role": "user", "content": "hello"}]

    assert asyncio.run(patch_dots3_note._preprocess_chat(serving, request, messages)) == "result"
    original.assert_awaited_once_with(serving, request, messages)


def test_preprocess_chat_copies_dots3_note_request_controls(monkeypatch):
    captured: ChatCompletionRequest | None = None

    async def preprocess(_, request, __):
        nonlocal captured
        captured = request
        return "result"

    monkeypatch.setattr(patch_dots3_note, "_original_preprocess_chat", preprocess)
    serving = SimpleNamespace(model_config=SimpleNamespace(architectures=["Dots3NoteForCausalLM"]))
    request = ChatCompletionRequest(
        model="dots3_note",
        messages=[],
        max_tokens=64,
        media_io_kwargs={"video": {"video_backend": "opencv", "num_frames": 8}},
        mm_processor_kwargs={"existing": True},
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "video"}},
                {"type": "text", "text": "describe"},
            ],
        }
    ]

    assert asyncio.run(patch_dots3_note._preprocess_chat(serving, request, messages)) == "result"
    assert captured is not None
    assert captured is not request
    assert request.mm_processor_kwargs == {"existing": True}
    assert captured.mm_processor_kwargs == {
        "existing": True,
        "max_new_tokens": 64,
        "video_question": "describe",
    }
    assert captured.media_io_kwargs == {"video": {"video_backend": "nemotron_vl", "num_frames": 8}}

    chat_params = captured.build_chat_params(None, "auto").with_defaults(
        default_mm_processor_kwargs=captured.mm_processor_kwargs,
    )
    assert chat_params.mm_processor_kwargs == captured.mm_processor_kwargs


def test_platform_patch_reload_is_idempotent():
    original_preprocess_chat = patch_dots3_note._original_preprocess_chat
    original_hf_config_override = patch_dots3_note._original_hf_config_override
    reload(patch_dots3_note)

    assert OnlineRenderer.preprocess_chat is patch_dots3_note._preprocess_chat
    assert patch_dots3_note._original_preprocess_chat is original_preprocess_chat
    assert patch_dots3_note._original_hf_config_override is original_hf_config_override
