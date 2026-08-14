# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, cast

from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    VerifyAndUpdateConfig,
)
from vllm.reasoning import ReasoningParserManager
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.tool_parsers import ToolParserManager
from vllm.transformers_utils import config as config_module
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from vllm_ascend.patch.dots3_note_config import Dots3NoteConfig


def _is_dots3_note_architecture(model_config: Any) -> bool:
    return "Dots3NoteForCausalLM" in (getattr(model_config, "architectures", None) or ())


config_module._CONFIG_REGISTRY["dots3_note"] = Dots3NoteConfig

_original_is_deepseek_mla = getattr(
    ModelArchConfigConvertorBase.is_deepseek_mla,
    "_vllm_ascend_dots3_note_original",
    ModelArchConfigConvertorBase.is_deepseek_mla,
)


def _is_deepseek_mla(self: ModelArchConfigConvertorBase) -> bool:
    if getattr(self.hf_text_config, "model_type", None) == "dots3_note":
        return True
    return _original_is_deepseek_mla(self)


cast(Any, _is_deepseek_mla)._vllm_ascend_dots3_note_original = _original_is_deepseek_mla
ModelArchConfigConvertorBase.is_deepseek_mla = _is_deepseek_mla


class Dots3NoteForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config) -> None:
        mm_config = model_config.multimodal_config
        if mm_config is None:
            return
        video_kwargs = mm_config.media_io_kwargs.setdefault("video", {})
        video_kwargs.setdefault("num_frames", 1)
        video_kwargs.setdefault("video_backend", "nemotron_vl")


MODELS_CONFIG_MAP["Dots3NoteForCausalLM"] = Dots3NoteForCausalLMConfig


_original_hf_config_override = getattr(
    SpeculativeConfig.hf_config_override,
    "_vllm_ascend_dots3_note_original",
    SpeculativeConfig.hf_config_override,
)


def _mtp_weight_is_indexed(hf_config: Any, weight_name: str) -> bool | None:
    model = getattr(hf_config, "_name_or_path", None)
    if not model:
        return None
    index = get_hf_file_to_dict(
        SAFE_WEIGHTS_INDEX_NAME,
        model,
        getattr(hf_config, "_commit_hash", None),
    )
    if index is None:
        return None
    return weight_name in index.get("weight_map", {})


def _hf_config_override(hf_config: Any) -> Any:
    if getattr(hf_config, "model_type", None) != "dots3_note":
        return _original_hf_config_override(hf_config)

    n_predict = getattr(hf_config, "num_nextn_predict_layers", None) or 1
    first_mtp_layer = hf_config.num_hidden_layers
    has_k_rope_only_layernorm = _mtp_weight_is_indexed(
        hf_config,
        f"model.layers.{first_mtp_layer}.self_attn.k_rope_only_layernorm.weight",
    )
    has_dedicated_mtp_embeddings = _mtp_weight_is_indexed(
        hf_config,
        "model.mtp.embed_tokens.weight",
    )
    updates = {
        "original_model_type": "dots3_note",
        "model_type": "deepseek_mtp",
        "num_nextn_predict_layers": n_predict,
        "mtp_head_sharing": getattr(hf_config, "mtp_head_sharing", "full") or "full",
        "n_predict": n_predict,
        "architectures": ["Dots3NoteMTPModel"],
    }
    if has_k_rope_only_layernorm is not None:
        updates["k_rope_only_layernorm"] = has_k_rope_only_layernorm
    if has_dedicated_mtp_embeddings is not None:
        updates["use_dedicated_mtp_embeddings"] = has_dedicated_mtp_embeddings
    hf_config.update(updates)
    return hf_config


cast(Any, _hf_config_override)._vllm_ascend_dots3_note_original = _original_hf_config_override
SpeculativeConfig.hf_config_override = _hf_config_override

ReasoningParserManager.register_lazy_module(
    name="dots",
    module_path="vllm.reasoning.qwen3_engine_reasoning_parser",
    class_name="Qwen3ParserReasoningAdapter",
)
ToolParserManager.register_lazy_module(
    name="dots",
    module_path="vllm_ascend.patch.dots_tool_parser",
    class_name="DotsToolParser",
)
_original_preprocess_chat = getattr(
    OnlineRenderer.preprocess_chat,
    "_vllm_ascend_dots3_note_original",
    OnlineRenderer.preprocess_chat,
)


def _message_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _video_question(messages: list[Any]) -> str | None:
    for message in reversed(messages):
        if _message_field(message, "role") != "user":
            continue
        content = _message_field(message, "content")
        if not isinstance(content, list):
            continue
        if not any(_message_field(part, "type") in {"video", "video_url"} for part in content):
            continue
        return "".join(_message_field(part, "text") or "" for part in content if _message_field(part, "type") == "text")
    return None


async def _preprocess_chat(
    self: OnlineRenderer,
    request: Any,
    messages: list[Any],
    *args,
    **kwargs,
):
    if not _is_dots3_note_architecture(self.model_config):
        return await _original_preprocess_chat(self, request, messages, *args, **kwargs)

    mm_processor_kwargs = dict(getattr(request, "mm_processor_kwargs", None) or {})
    max_new_tokens = (
        request.max_completion_tokens
        if getattr(request, "max_completion_tokens", None) is not None
        else getattr(request, "max_tokens", None)
    )
    if max_new_tokens is not None:
        mm_processor_kwargs["max_new_tokens"] = max_new_tokens
    question = _video_question(messages)
    if question is not None:
        mm_processor_kwargs["video_question"] = question

    media_io_kwargs = dict(getattr(request, "media_io_kwargs", None) or {})
    video_kwargs = dict(media_io_kwargs.get("video") or {})
    video_kwargs["video_backend"] = "nemotron_vl"
    media_io_kwargs["video"] = video_kwargs
    request = request.model_copy(
        update={
            "media_io_kwargs": media_io_kwargs,
            "mm_processor_kwargs": mm_processor_kwargs or None,
        }
    )
    return await _original_preprocess_chat(self, request, messages, *args, **kwargs)


cast(Any, _preprocess_chat)._vllm_ascend_dots3_note_original = _original_preprocess_chat
OnlineRenderer.preprocess_chat = _preprocess_chat
