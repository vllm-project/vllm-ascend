# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from functools import lru_cache
from typing import Annotated, Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.nn import LayerNorm
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessor
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import utils as dist_utils
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM
from vllm.model_executor.models.dots_ocr import (
    DotsPatchEmbed,
    DotsSwiGLUFFN,
    VisionRotaryEmbedding,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    run_dp_sharded_mrope_vision_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    DictEmbeddingItems,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.multimodal.processing.processor import PlaceholderFeaturesInfo
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_ascend.patch.dots3_note_audio import (
    Dots3NoteAudioTower,
    prepare_audio_features,
)
from vllm_ascend.patch.dots3_note_config import (
    Dots3NoteAudioConfig,
    Dots3NoteConfig,
    Dots3NoteVisionConfig,
)
from vllm_ascend.patch.dots3_note_video import (
    ALIGN as VIDEO_ALIGN,
)
from vllm_ascend.patch.dots3_note_video import (
    FPS_MIN_FRAMES,
    V2_PF_FLOOR,
    format_timestamp,
    preprocess_video,
)

IMAGE_START_TOKEN = "<|img|>"
IMAGE_PAD_TOKEN = "<|imgpad|>"
IMAGE_END_TOKEN = "<|endofimg|>"
IMAGE_PLACEHOLDER = IMAGE_START_TOKEN + IMAGE_PAD_TOKEN + IMAGE_END_TOKEN
AUDIO_START_TOKEN = "<|audio_comp_start|>"
AUDIO_PAD_TOKEN = "<|audio_comp_pad|>"
AUDIO_END_TOKEN = "<|audio_comp_end|>"
AUDIO_PLACEHOLDER = AUDIO_START_TOKEN + AUDIO_PAD_TOKEN + AUDIO_END_TOKEN
VIDEO_PLACEHOLDER = "<|video_pad|>"


def _get_tp_world_size() -> int:
    try:
        return get_tensor_model_parallel_world_size()
    except AssertionError:
        return 1


class Dots3NoteImagePixelInputs(TensorSchema):
    type: Literal["pixel_values"]
    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cps")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class Dots3NoteImageEmbeddingInputs(TensorSchema):
    type: Literal["image_embeds"]
    image_embeds: Annotated[torch.Tensor, TensorShape("nf", "hs")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


Dots3NoteImageInputs: TypeAlias = Dots3NoteImagePixelInputs | Dots3NoteImageEmbeddingInputs


class Dots3NoteAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    audio_features: Annotated[torch.Tensor, TensorShape("ns", 128, "nf")]
    audio_sample_lens: Annotated[torch.Tensor, TensorShape("ns")]
    audio_segment_counts: Annotated[torch.Tensor, TensorShape("na")]
    audio_token_lengths: Annotated[torch.Tensor, TensorShape("na")]


class Dots3NoteAudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    audio_embeds: Annotated[list[torch.Tensor], TensorShape("na", "nt", "hs", dynamic_dims={"nt"})]


Dots3NoteAudioInputs: TypeAlias = Dots3NoteAudioFeatureInputs | Dots3NoteAudioEmbeddingInputs


@lru_cache
def _get_dots_image_processor(model_path: str) -> Qwen2VLImageProcessor:
    config = _load_json_config(os.path.join(model_path, "preprocessor_config.json"))
    if isinstance(config.get("vision_config"), dict):
        return Qwen2VLImageProcessor.from_dict(config["vision_config"])
    return Qwen2VLImageProcessor.from_pretrained(model_path)


@lru_cache
def _load_json_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _get_dots3_note_vision_config_path(model_path: str) -> str | None:
    config_path = os.path.join(model_path, "config.json")
    config = _load_json_config(config_path)
    return config_path if "vision_config" in config else None


def _get_dots3_note_audio_config_path(model_path: str) -> str | None:
    config_path = os.path.join(model_path, "config.json")
    config = _load_json_config(config_path)
    return config_path if "audio_config" in config else None


def _resolve_dots3_note_vision_config(
    model_path: str,
    vision_config: dict | Dots3NoteVisionConfig | None,
) -> Dots3NoteVisionConfig:
    top_config = _load_json_config(os.path.join(model_path, "config.json"))
    if isinstance(vision_config, Dots3NoteVisionConfig):
        return vision_config
    return Dots3NoteVisionConfig(**(vision_config or top_config.get("vision_config") or {}))


def _resolve_dots3_note_audio_config(
    model_path: str,
    audio_config: dict | Dots3NoteAudioConfig | None,
) -> Dots3NoteAudioConfig:
    top_config = _load_json_config(os.path.join(model_path, "config.json"))
    if isinstance(audio_config, Dots3NoteAudioConfig):
        return audio_config
    return Dots3NoteAudioConfig(**(audio_config or top_config.get("audio_config") or {}))


def _build_embedding_image_inputs(
    data: object,
    spatial_merge_size: int,
) -> dict[str, torch.Tensor] | None:
    if isinstance(data, torch.Tensor):
        if data.ndim == 2:
            tensors = [data]
        elif data.ndim == 3:
            tensors = [item for item in data]
        else:
            return None
    elif isinstance(data, list) and all(isinstance(item, torch.Tensor) and item.ndim == 2 for item in data):
        tensors = data
    else:
        return None

    if not tensors:
        return None

    image_grid_thw = torch.tensor(
        [[1, spatial_merge_size, int(item.shape[0]) * spatial_merge_size] for item in tensors],
        dtype=torch.long,
    )
    return {
        "image_embeds": torch.cat(tensors, dim=0),
        "image_grid_thw": image_grid_thw,
    }


class Dots3NoteMultiModalDataParser(Qwen2VLMultiModalDataParser):
    def _parse_image_data(self, data):
        if isinstance(data, dict):
            return super()._parse_image_data(data)

        embedding_inputs = _build_embedding_image_inputs(
            data,
            self._spatial_merge_size,
        )
        if embedding_inputs is not None:
            return DictEmbeddingItems(
                embedding_inputs,
                modality="image",
                required_fields={"image_embeds", "image_grid_thw"},
                fields_factory=_create_qwen2vl_field_factory(self._spatial_merge_size),
            )

        return super()._parse_image_data(data)

    def _parse_audio_data(self, data):
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=lambda _: {"audio_embeds": MultiModalFieldConfig.batched("audio")},
            )
        return super()._parse_audio_data(data)


class Dots3NoteProcessor:
    image_token = IMAGE_PLACEHOLDER
    audio_token = AUDIO_PLACEHOLDER
    video_token = VIDEO_PLACEHOLDER

    def __init__(
        self,
        model_path: str,
        tokenizer,
        config: Dots3NoteConfig,
    ) -> None:
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.config = config
        self.image_processor = (
            _get_dots_image_processor(model_path)
            if _get_dots3_note_vision_config_path(model_path) is not None
            else None
        )
        self.audio_config = (
            _resolve_dots3_note_audio_config(model_path, config.audio_config)
            if _get_dots3_note_audio_config_path(model_path) is not None
            else None
        )

    @staticmethod
    def _get_video_question(text: str, explicit_question: object) -> str:
        if isinstance(explicit_question, str):
            return explicit_question
        user_start = text.rfind("<|user|>")
        user_end = text.find("<|endofuser|>", max(user_start, 0))
        if user_start >= 0 and user_end > user_start:
            text = text[user_start + len("<|user|>") : user_end]
        return text.replace(VIDEO_PLACEHOLDER, "", 1).strip()

    @staticmethod
    def _rewrite_video_prompt(text: str, question: str) -> tuple[str, str]:
        if not question:
            if VIDEO_PLACEHOLDER not in text:
                raise ValueError("Dots3 Note video prompt is missing the video placeholder")
            return text, VIDEO_PLACEHOLDER

        question_start = text.find(question) if question else -1
        marker_start = text.find(
            VIDEO_PLACEHOLDER,
            question_start + len(question),
        )
        if question_start < 0 or marker_start < 0:
            raise ValueError("Dots3 Note video question must precede the video placeholder in the prompt")

        question_end = question_start + len(question)
        separator = text[question_end:marker_start]
        rewritten = text[:question_start] + VIDEO_PLACEHOLDER + text[marker_start + len(VIDEO_PLACEHOLDER) :]
        reference_tail = question + separator + VIDEO_PLACEHOLDER
        return rewritten, reference_tail

    def _process_video(
        self,
        text: str,
        video: object,
        return_tensors: str | None,
        kwargs: Mapping[str, Any],
        *,
        question: str | None = None,
        reference_tail: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.image_processor is None:
            raise ValueError("Raw video inputs require Dots3 Note vision weights")

        output_reserve = kwargs.get("output_reserve")
        if output_reserve is not None:
            output_reserve = int(output_reserve)

        if question is None:
            question = self._get_video_question(text, kwargs.get("video_question"))
        if reference_tail is None:
            _, reference_tail = self._rewrite_video_prompt(text, question)

        result = preprocess_video(
            video,
            prompt=text,
            question=question,
            tokenizer=self.tokenizer,
            seq=int(kwargs.get("seq", 131072)),
            output_reserve=output_reserve,
            max_new_tokens=int(kwargs.get("max_new_tokens", 0)),
            audio_cap=float(kwargs.get("audio_cap", 1.0)),
            audio_sr=int(kwargs.get("audio_sr", 16000)),
            k_mode=str(kwargs.get("k_mode", "eval_ek")),
        )

        image_inputs = self.image_processor(
            images=result.frames,
            return_tensors=return_tensors,
        )
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = self.config.vision_config.spatial_merge_size**2
        image_token_lengths = (image_grid_thw.to(dtype=torch.long).prod(dim=-1) // merge_length).tolist()

        audio_inputs: dict[str, torch.Tensor] = {}
        audio_token_lengths: list[int] = []
        if result.audio_segments:
            if self.audio_config is None:
                raise ValueError("Video audio requires Dots3 Note audio weights")
            audio_inputs = prepare_audio_features(
                result.audio_segments,
                self.audio_config,
            )
            audio_token_lengths = audio_inputs["audio_token_lengths"].tolist()

        token_ids: list[int] = []
        is_embed: list[bool] = []
        layout: list[int] = []
        for modality, item_idx in result.layout:
            if modality == "image":
                token_length = int(image_token_lengths[item_idx])
                timestamp_ids = self.tokenizer.encode(
                    format_timestamp(result.timestamps[item_idx]),
                    add_special_tokens=False,
                )
                token_ids.extend(timestamp_ids)
                is_embed.extend([False] * len(timestamp_ids))
                token_ids.append(self.config.image_start_token_id)
                is_embed.append(False)
                token_ids.extend([self.config.image_token_id] * token_length)
                is_embed.extend([True] * token_length)
                token_ids.append(self.config.image_end_token_id)
                is_embed.append(False)
                layout.append(token_length)
            else:
                token_length = int(audio_token_lengths[item_idx])
                token_ids.append(self.config.audio_start_token_id)
                is_embed.append(False)
                token_ids.extend([self.config.audio_token_id] * token_length)
                is_embed.extend([True] * token_length)
                token_ids.append(self.config.audio_end_token_id)
                is_embed.append(False)
                layout.append(-token_length)
        reference_tail_ids = self.tokenizer.encode(reference_tail, add_special_tokens=False)
        token_ids.extend(reference_tail_ids)
        is_embed.extend([False] * len(reference_tail_ids))

        outputs = {
            "video_pixel_values": pixel_values,
            "video_image_grid_thw": image_grid_thw,
            "video_frame_counts": torch.tensor([len(result.frames)], dtype=torch.long),
            "video_layout": torch.tensor(layout, dtype=torch.long),
            "video_layout_counts": torch.tensor([len(layout)], dtype=torch.long),
            "video_token_ids": torch.tensor(token_ids, dtype=torch.long),
            "video_is_embed": torch.tensor(is_embed, dtype=torch.bool),
            "video_prompt_lengths": torch.tensor([len(token_ids)], dtype=torch.long),
            "video_audio_counts": torch.tensor([len(result.audio_segments)], dtype=torch.long),
        }
        outputs.update({f"video_{key}": value for key, value in audio_inputs.items()})
        return outputs

    def _build_input_ids_with_mm_placeholders(
        self,
        text: str,
        image_grid_thw: torch.Tensor,
    ) -> list[int] | None:
        prompt_segments = text.split(IMAGE_PLACEHOLDER)
        num_images = int(image_grid_thw.shape[0])
        if len(prompt_segments) != num_images + 1:
            return None

        merge_length = self.config.vision_config.spatial_merge_size**2
        image_token_counts = (image_grid_thw.to(dtype=torch.long).prod(dim=-1) // merge_length).tolist()

        prompt_token_ids: list[int] = []
        for image_idx, prompt_segment in enumerate(prompt_segments[:-1]):
            if prompt_segment:
                prompt_token_ids.extend(self.tokenizer.encode(prompt_segment, add_special_tokens=False))
            prompt_token_ids.append(self.config.image_start_token_id)
            prompt_token_ids.extend([self.config.image_token_id] * int(image_token_counts[image_idx]))
            prompt_token_ids.append(self.config.image_end_token_id)

        last_prompt_segment = prompt_segments[-1]
        if last_prompt_segment:
            prompt_token_ids.extend(self.tokenizer.encode(last_prompt_segment, add_special_tokens=False))

        return prompt_token_ids

    def __call__(
        self,
        text: str | Sequence[str] | None = None,
        images=None,
        audio=None,
        videos=None,
        return_tensors: str | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Tokenize text and assemble any Dots3 Note multimodal inputs."""
        tokenizer_kwargs = {}
        for key in ("padding", "truncation", "add_special_tokens"):
            if key in kwargs:
                tokenizer_kwargs[key] = kwargs[key]
        tokenizer_kwargs.setdefault("add_special_tokens", False)

        if text is None:
            text = ""
        original_text = text
        multimodal_inputs: dict[str, object] = {}
        prompt_token_ids = None
        video_items = None
        video_question = None
        video_reference_tail = None

        if videos is not None:
            if images is not None or audio is not None:
                raise ValueError("Dots3 Note video requests cannot mix separate image or audio inputs")
            if not isinstance(text, str):
                raise ValueError("Dots3 Note video requests require a text prompt")
            video_items = videos if isinstance(videos, list) else [videos]
            if len(video_items) != 1:
                raise ValueError("Dots3 Note currently supports one video per request")
            video_question = self._get_video_question(
                text,
                kwargs.get("video_question"),
            )
            text, video_reference_tail = self._rewrite_video_prompt(
                text,
                video_question,
            )

        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            **tokenizer_kwargs,
        )

        if video_items is not None:
            assert isinstance(original_text, str)
            multimodal_inputs.update(
                self._process_video(
                    original_text,
                    video_items[0],
                    return_tensors,
                    kwargs,
                    question=video_question,
                    reference_tail=video_reference_tail,
                )
            )

        if images is not None:
            if self.image_processor is None:
                raise ValueError(
                    "Raw image inputs require Dots3 Note vision weights under the model path. "
                    "Use image_embeds for offline embedding input."
                )

            image_kwargs = {}
            for key in (
                "do_resize",
                "resample",
                "do_rescale",
                "rescale_factor",
                "do_normalize",
                "image_mean",
                "image_std",
                "do_convert_rgb",
                "min_pixels",
                "max_pixels",
                "size",
            ):
                if key in kwargs:
                    image_kwargs[key] = kwargs[key]
            image_inputs = self.image_processor(
                images=images,
                return_tensors=return_tensors,
                **image_kwargs,
            )
            multimodal_inputs.update(dict(image_inputs))

            if isinstance(text, str):
                image_grid_thw = image_inputs.get("image_grid_thw")
                if isinstance(image_grid_thw, torch.Tensor):
                    prompt_token_ids = self._build_input_ids_with_mm_placeholders(
                        text,
                        image_grid_thw,
                    )

        if audio is not None:
            if self.audio_config is None:
                raise ValueError(
                    "Raw audio inputs require Dots3 Note audio weights under the model path. "
                    "Use audio_embeds for offline embedding input."
                )
            audio_items = audio if isinstance(audio, list) else [audio]
            multimodal_inputs.update(prepare_audio_features(audio_items, self.audio_config))

        if prompt_token_ids is not None:
            text_inputs = {
                "input_ids": [prompt_token_ids],
                "attention_mask": [[1] * len(prompt_token_ids)],
            }

        return BatchFeature(
            data={**dict(text_inputs), **multimodal_inputs},
            tensor_type=return_tensors,
        )


class Dots3NoteProcessingInfo(Qwen2VLProcessingInfo):
    def get_hf_config(self) -> Dots3NoteConfig:
        config = self.ctx.get_hf_config(Dots3NoteConfig)
        config.vision_config = _resolve_dots3_note_vision_config(
            self.ctx.model_config.model,
            config.vision_config,
        )
        config.audio_config = _resolve_dots3_note_audio_config(
            self.ctx.model_config.model,
            config.audio_config,
        )
        return config

    def get_hf_processor(self, **kwargs: object) -> Dots3NoteProcessor:
        return Dots3NoteProcessor(
            self.ctx.model_config.model,
            self.get_tokenizer(),
            self.get_hf_config(),
        )

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessor:
        image_processor = self.get_hf_processor(**kwargs).image_processor
        if image_processor is None:
            raise ValueError(
                "Raw image processing is unavailable because this Dots3 Note model "
                "path does not contain online vision weights."
            )
        return image_processor

    def _has_vision_tower(self) -> bool:
        return _get_dots3_note_vision_config_path(self.ctx.model_config.model) is not None

    def _has_audio_tower(self) -> bool:
        return _get_dots3_note_audio_config_path(self.ctx.model_config.model) is not None

    def get_data_parser(self):
        return Dots3NoteMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            target_sr=self.get_hf_config().audio_config.sampling_rate,
            target_channels=1,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        limits: dict[str, int | None] = {}
        enable_mm_embeds = self.ctx.get_mm_config().enable_mm_embeds
        if self._has_vision_tower():
            limits["image"] = None
        elif enable_mm_embeds:
            limits["image"] = 0
        if self._has_audio_tower():
            limits["audio"] = None
        elif enable_mm_embeds:
            limits["audio"] = 0
        if self._has_vision_tower():
            limits["video"] = 1
        return limits

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        limits: dict[str, int] = {}
        if mm_counts.get("image", 0) > 0:
            limits["image"] = self.get_max_image_tokens() if self._has_vision_tower() else 0
        if mm_counts.get("audio", 0) > 0:
            limits["audio"] = seq_len if self._has_audio_tower() else 0
        if mm_counts.get("video", 0) > 0:
            limits["video"] = seq_len if self._has_vision_tower() else 0
        return limits


class Dots3NoteDummyInputsBuilder(BaseDummyInputsBuilder[Dots3NoteProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return (
            IMAGE_PLACEHOLDER * mm_counts.get("image", 0)
            + AUDIO_PLACEHOLDER * mm_counts.get("audio", 0)
            + VIDEO_PLACEHOLDER * mm_counts.get("video", 0)
        )

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        dummy_data: MultiModalDataDict = {}
        num_images = mm_counts.get("image", 0)
        if num_images > 0:
            target_width, target_height = self.info.get_image_size_with_most_features()
            dummy_data["image"] = self._get_dummy_images(
                width=target_width,
                height=target_height,
                num_images=num_images,
                overrides=mm_options.get("image"),
            )
        num_audios = mm_counts.get("audio", 0)
        if num_audios > 0:
            audio_config = self.info.get_hf_config().audio_config
            dummy_data["audio"] = self._get_dummy_audios(
                length=audio_config.chunk_seconds * audio_config.sampling_rate,
                num_audios=num_audios,
                overrides=mm_options.get("audio"),
            )
        num_videos = mm_counts.get("video", 0)
        if num_videos > 0:
            target_width, target_height = self.info.get_image_size_with_most_features(
                V2_PF_FLOOR * VIDEO_ALIGN * VIDEO_ALIGN
            )
            videos = self._get_dummy_videos(
                width=target_width,
                height=target_height,
                num_frames=FPS_MIN_FRAMES,
                num_videos=num_videos,
                overrides=mm_options.get("video"),
            )
            dummy_data["video"] = [
                (
                    video,
                    {
                        "fps": 1.0,
                        "duration": float(len(video)),
                        "total_num_frames": len(video),
                    },
                )
                for video in videos
            ]
        return dummy_data


class Dots3NoteMultiModalProcessor(BaseMultiModalProcessor[Dots3NoteProcessingInfo]):
    def _get_existing_image_token_placeholders(
        self,
        prompt_ids: list[int],
        expected_count: int,
    ) -> list[PlaceholderFeaturesInfo]:
        return self._find_image_token_placeholders(
            prompt_ids,
            {"image": [[] for _ in range(expected_count)]},
        )

    def _find_image_token_placeholders(
        self,
        prompt_ids: list[int],
        mm_prompt_updates: Mapping[str, Sequence[Sequence[object]]],
    ) -> list[PlaceholderFeaturesInfo]:
        image_updates = mm_prompt_updates.get("image", [])
        if not image_updates:
            return []

        config = self.info.get_hf_config()
        start_token_id = config.image_start_token_id
        image_token_id = config.image_token_id
        end_token_id = config.image_end_token_id

        placeholders: list[PlaceholderFeaturesInfo] = []
        prompt_len = len(prompt_ids)
        item_idx = 0
        start_idx = 0

        while start_idx < prompt_len and item_idx < len(image_updates):
            if prompt_ids[start_idx] != start_token_id:
                start_idx += 1
                continue

            end_idx = start_idx + 1
            while end_idx < prompt_len and prompt_ids[end_idx] == image_token_id:
                end_idx += 1

            if end_idx >= prompt_len or prompt_ids[end_idx] != end_token_id:
                start_idx += 1
                continue

            tokens = prompt_ids[start_idx : end_idx + 1]
            is_embed = torch.zeros(len(tokens), dtype=torch.bool)
            if len(tokens) > 2:
                is_embed[1:-1] = True

            placeholders.append(
                PlaceholderFeaturesInfo(
                    modality="image",
                    item_idx=item_idx,
                    start_idx=start_idx,
                    tokens=tokens,
                    is_embed=is_embed,
                )
            )
            item_idx += 1
            start_idx = end_idx + 1

        return placeholders

    def _can_use_direct_mm_prompt(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
    ) -> bool:
        return prompt_text.count(IMAGE_PLACEHOLDER) == mm_items.get_all_counts().get("image", 0)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        video_count = mm_items.get_all_counts().get("video", 0)
        if video_count > 0:
            prompt_text = (
                prompt
                if isinstance(prompt, str)
                else self.info.get_tokenizer().decode(
                    prompt,
                    skip_special_tokens=False,
                )
            )
            prompt_ids, processed_data, _ = self._apply_hf_processor_text_mm(
                prompt_text=prompt_text,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )
            return prompt_ids, processed_data, False

        image_count = mm_items.get_all_counts().get("image", 0)
        if image_count > 0:
            if isinstance(prompt, list):
                existing_placeholders = self._get_existing_image_token_placeholders(
                    prompt,
                    image_count,
                )
                if len(existing_placeholders) == image_count:
                    mm_processed_data = self._apply_hf_processor_mm_only(
                        mm_items=mm_items,
                        hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                        tokenization_kwargs=tokenization_kwargs,
                    )
                    image_grid_thw = mm_processed_data.get("image_grid_thw")
                    if isinstance(image_grid_thw, torch.Tensor):
                        merge_length = self.info.get_hf_config().vision_config.spatial_merge_size ** 2
                        expected_token_counts = (image_grid_thw.reshape(-1, 3).prod(dim=-1) // merge_length).tolist()
                        actual_token_counts = [int(placeholder.is_embed.sum()) for placeholder in existing_placeholders]
                        if actual_token_counts == expected_token_counts:
                            return prompt, mm_processed_data, True
                    else:
                        return prompt, mm_processed_data, True

            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                prompt_text = self.info.get_tokenizer().decode(
                    prompt,
                    skip_special_tokens=False,
                )

            if self._can_use_direct_mm_prompt(prompt_text, mm_items):
                return self._apply_hf_processor_text_mm(
                    prompt_text=prompt_text,
                    mm_items=mm_items,
                    hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                )

        return super()._apply_hf_processor_main(
            prompt=prompt,
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
            enable_hf_prompt_update=enable_hf_prompt_update,
        )

    def _find_mm_placeholders(
        self,
        new_token_ids: list[int],
        mm_prompt_updates,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        placeholders = super()._find_mm_placeholders(new_token_ids, mm_prompt_updates)
        image_placeholders = placeholders.get("image", [])
        expected_image_count = len(mm_prompt_updates.get("image", []))
        if image_placeholders or expected_image_count == 0:
            return placeholders

        fallback_placeholders = self._find_image_token_placeholders(
            new_token_ids,
            mm_prompt_updates,
        )
        if len(fallback_placeholders) != expected_image_count:
            return placeholders

        return {**placeholders, "image": fallback_placeholders}

    def _apply_prompt_updates(
        self,
        token_ids: list[int],
        mm_prompt_updates,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        new_token_ids, placeholders = super()._apply_prompt_updates(
            token_ids,
            mm_prompt_updates,
        )
        image_updates = mm_prompt_updates.get("image", [])
        if len(placeholders.get("image", [])) == len(image_updates):
            return new_token_ids, placeholders

        tokenizer = self.info.get_tokenizer()
        prompt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        if prompt_text.count(IMAGE_PLACEHOLDER) != len(image_updates):
            return new_token_ids, placeholders

        prompt_segments = prompt_text.split(IMAGE_PLACEHOLDER)
        rebuilt_token_ids: list[int] = []
        for item_idx, prompt_segment in enumerate(prompt_segments[:-1]):
            if prompt_segment:
                rebuilt_token_ids.extend(tokenizer.encode(prompt_segment, add_special_tokens=False))

            item_updates = image_updates[item_idx]
            if not item_updates:
                return new_token_ids, placeholders
            rebuilt_token_ids.extend(item_updates[0].content.full)

        last_prompt_segment = prompt_segments[-1]
        if last_prompt_segment:
            rebuilt_token_ids.extend(tokenizer.encode(last_prompt_segment, add_special_tokens=False))

        image_placeholders = self._find_image_token_placeholders(
            rebuilt_token_ids,
            mm_prompt_updates,
        )
        if len(image_placeholders) != len(image_updates):
            return new_token_ids, placeholders

        return rebuilt_token_ids, {**placeholders, "image": image_placeholders}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        image_token_id = config.image_token_id
        start_token_id = config.image_start_token_id
        end_token_id = config.image_end_token_id

        def get_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)
            merge_length = config.vision_config.spatial_merge_size**2
            num_tokens = int(grid_thw.prod()) // merge_length
            token_ids = [start_token_id] + [image_token_id] * num_tokens + [end_token_id]
            return PromptUpdateDetails.select_token_id(token_ids, image_token_id)

        replacements: list[PromptUpdate] = []
        if mm_items.get_all_counts().get("image", 0) > 0:
            replacements.append(
                PromptReplacement(
                    modality="image",
                    target=IMAGE_PLACEHOLDER,
                    replacement=get_replacement,
                )
            )

        audio_start_token_id = config.audio_start_token_id
        audio_token_id = config.audio_token_id
        audio_end_token_id = config.audio_end_token_id

        def get_audio_replacement(item_idx: int):
            out_item = out_mm_kwargs["audio"][item_idx]
            if "audio_token_lengths" in out_item:
                token_length = int(out_item["audio_token_lengths"].data.item())
            else:
                audio_embeds = out_item["audio_embeds"].data
                token_length = int(audio_embeds.shape[0])
            token_ids = [audio_start_token_id] + [audio_token_id] * token_length + [audio_end_token_id]
            return PromptUpdateDetails.select_token_id(token_ids, audio_token_id)

        if mm_items.get_all_counts().get("audio", 0) > 0:
            replacements.append(
                PromptReplacement(
                    modality="audio",
                    target=AUDIO_PLACEHOLDER,
                    replacement=get_audio_replacement,
                )
            )

        def get_video_replacement(item_idx: int):
            out_item = out_mm_kwargs["video"][item_idx]
            token_ids = out_item["video_token_ids"].data
            is_embed = out_item["video_is_embed"].data
            assert isinstance(token_ids, torch.Tensor)
            assert isinstance(is_embed, torch.Tensor)
            return PromptUpdateDetails(
                full=token_ids.tolist(),
                is_embed=lambda _tokenizer, _full: is_embed,
            )

        if mm_items.get_all_counts().get("video", 0) > 0:
            replacements.append(
                PromptReplacement(
                    modality="video",
                    target=VIDEO_PLACEHOLDER,
                    replacement=get_video_replacement,
                )
            )
        return replacements

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = _create_qwen2vl_field_factory(self.info.get_hf_config().vision_config.spatial_merge_size)(hf_inputs)
        segment_counts = hf_inputs.get("audio_segment_counts")
        if isinstance(segment_counts, torch.Tensor):
            fields.update(
                audio_features=MultiModalFieldConfig.flat_from_sizes("audio", segment_counts),
                audio_sample_lens=MultiModalFieldConfig.flat_from_sizes("audio", segment_counts, keep_on_cpu=True),
                audio_segment_counts=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
                audio_token_lengths=MultiModalFieldConfig.batched("audio", keep_on_cpu=True),
            )
        if "audio_embeds" in hf_inputs:
            fields["audio_embeds"] = MultiModalFieldConfig.batched("audio")
        video_frame_counts = hf_inputs.get("video_frame_counts")
        if isinstance(video_frame_counts, torch.Tensor):
            video_grid_thw = hf_inputs["video_image_grid_thw"]
            assert isinstance(video_grid_thw, torch.Tensor)
            video_pixel_sizes = torch.tensor(
                [int(video_grid_thw.prod(dim=-1).sum())],
                dtype=torch.long,
            )
            fields.update(
                video_pixel_values=MultiModalFieldConfig.flat_from_sizes("video", video_pixel_sizes),
                video_image_grid_thw=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_frame_counts, keep_on_cpu=True
                ),
                video_frame_counts=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                video_layout=MultiModalFieldConfig.flat_from_sizes(
                    "video", hf_inputs["video_layout_counts"], keep_on_cpu=True
                ),
                video_layout_counts=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                video_token_ids=MultiModalFieldConfig.flat_from_sizes(
                    "video", hf_inputs["video_prompt_lengths"], keep_on_cpu=True
                ),
                video_is_embed=MultiModalFieldConfig.flat_from_sizes(
                    "video", hf_inputs["video_prompt_lengths"], keep_on_cpu=True
                ),
                video_prompt_lengths=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                video_audio_counts=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
            )
            audio_counts = hf_inputs["video_audio_counts"]
            assert isinstance(audio_counts, torch.Tensor)
            if int(audio_counts.sum()) > 0:
                audio_segment_counts = hf_inputs["video_audio_segment_counts"]
                assert isinstance(audio_segment_counts, torch.Tensor)
                audio_chunk_counts = torch.tensor([int(audio_segment_counts.sum())], dtype=torch.long)
                fields.update(
                    video_audio_features=MultiModalFieldConfig.flat_from_sizes("video", audio_chunk_counts),
                    video_audio_sample_lens=MultiModalFieldConfig.flat_from_sizes(
                        "video", audio_chunk_counts, keep_on_cpu=True
                    ),
                    video_audio_segment_counts=MultiModalFieldConfig.flat_from_sizes(
                        "video", audio_counts, keep_on_cpu=True
                    ),
                    video_audio_token_lengths=MultiModalFieldConfig.flat_from_sizes(
                        "video", audio_counts, keep_on_cpu=True
                    ),
                )
        return fields


class PatchMergerAdapter(nn.Module):
    def __init__(
        self,
        config: Dots3NoteVisionConfig,
        *,
        use_data_parallel: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        in_dim = config.adapter_in_dim
        out_dim = config.adapter_out_dim
        merge_size = config.adapter_merge_size
        merged_dim = in_dim * merge_size**2
        self.merged_dim = merged_dim
        self.ln_q = LayerNorm(in_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                merged_dim,
                merged_dim,
                bias=True,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.mlp.0",
                disable_tp=use_data_parallel,
            ),
            nn.GELU(),
            RowParallelLinear(
                merged_dim,
                out_dim,
                bias=True,
                quant_config=quant_config,
                return_bias=False,
                prefix=f"{prefix}.mlp.2",
                disable_tp=use_data_parallel,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.reshape(-1, self.merged_dim)
        return self.mlp(x)


class DotsVisionAttention(nn.Module):
    def __init__(
        self,
        config: Dots3NoteVisionConfig,
        *,
        use_data_parallel: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = config.embed_dim
        num_heads = config.num_attention_heads
        tp_world = _get_tp_world_size()
        use_replicated_attention = use_data_parallel or num_heads % tp_world != 0
        self.tp_size = 1 if use_replicated_attention else tp_world
        self.hidden_size_per_attention_head = dist_utils.divide(dim, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_replicated_attention,
        )
        self.proj = RowParallelLinear(
            input_size=dim,
            output_size=dim,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_replicated_attention,
        )
        self.q_norm = (
            RMSNorm(self.hidden_size_per_attention_head, eps=config.rms_norm_eps) if config.use_qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.hidden_size_per_attention_head, eps=config.rms_norm_eps) if config.use_qk_norm else None
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            enable_fp32_compute=True,
        )

    def split_qkv(
        self,
        qkv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len, batch_size, _ = qkv.shape
        qkv = qkv.view(
            seq_len,
            batch_size,
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        return qkv.unbind(dim=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        *,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = hidden_states.unsqueeze(1)
        qkv, _ = self.qkv(x)
        q, k, v = self.split_qkv(qkv)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        batch_size = q.shape[1]
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

        qk_concat = torch.cat([q, k], dim=0)
        qk_rotated = self.apply_rotary_emb(
            qk_concat,
            rotary_pos_emb.cos(),
            rotary_pos_emb.sin(),
        )
        q, k = torch.chunk(qk_rotated, 2, dim=0)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        context_layer = context_layer.permute(1, 0, 2, 3).contiguous()
        context_layer = context_layer.view(context_layer.shape[0], batch_size, -1)
        out, _ = self.proj(context_layer)
        return out.squeeze(1)


def _dots_swiglu(
    config: Dots3NoteVisionConfig,
    intermediate_size: int,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> DotsSwiGLUFFN:
    config = copy(config)
    config.intermediate_size = intermediate_size
    return DotsSwiGLUFFN(config, quant_config=quant_config, prefix=prefix)


class MoESwiGLUFFN(nn.Module):
    def __init__(
        self,
        config: Dots3NoteVisionConfig,
        layer_number: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_routed = config.pyramid_num_routed[layer_number]
        self.capacity_factor = config.capacity_factor
        self.router_scoring_func = config.router_scoring_func
        self.router_scale = config.router_scale
        self.register_buffer(
            "router_bias",
            torch.zeros(self.num_routed, dtype=torch.float32),
        )
        self.experts = nn.ModuleList(
            [
                _dots_swiglu(
                    config,
                    config.moe_intermediate_size,
                    quant_config,
                    f"{prefix}.experts.{expert_idx}",
                )
                for expert_idx in range(self.num_routed)
            ]
        )
        self.gate_weight = nn.Parameter(torch.empty((self.num_routed, config.embed_dim), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-9
        x_flat = x.contiguous().view(-1, x.shape[-1])
        num_tokens = x_flat.shape[0]
        gate_logits = F.linear(x_flat.float(), self.gate_weight.float())
        if self.router_scoring_func == "sigmoid":
            gating_prob = torch.sigmoid(gate_logits)
        else:
            gating_prob = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)

        topk = min(int(self.capacity_factor), self.num_routed)
        gating_with_bias = gating_prob + self.router_bias.float().unsqueeze(0)
        _, topk_indices = torch.topk(gating_with_bias, k=topk, dim=-1, sorted=False)
        routed_weights = gating_prob.gather(1, topk_indices)
        if self.router_scoring_func == "sigmoid" and topk > 1:
            routed_weights = routed_weights / (routed_weights.sum(dim=-1, keepdim=True) + epsilon)
        routed_weights = (routed_weights * self.router_scale).to(x_flat.dtype)

        aggregated_output = torch.zeros_like(x_flat)
        aggregated_gate = torch.zeros(num_tokens, dtype=x_flat.dtype, device=x.device)
        for expert_idx, expert in enumerate(self.experts):
            selected_mask = topk_indices == expert_idx
            if not selected_mask.any():
                continue
            n_idx, top = torch.where(selected_mask)
            expert_output = expert(x_flat[n_idx].contiguous())
            contrib = expert_output * routed_weights[n_idx, top].unsqueeze(-1)
            aggregated_output[n_idx] = aggregated_output[n_idx] + contrib
            aggregated_gate[n_idx] = aggregated_gate[n_idx] + routed_weights[n_idx, top]

        return aggregated_output / (aggregated_gate.unsqueeze(-1) + epsilon)


class MoEVisionBlock(nn.Module):
    def __init__(
        self,
        config: Dots3NoteVisionConfig,
        layer_number: int,
        *,
        use_data_parallel: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn = DotsVisionAttention(
            config,
            use_data_parallel=use_data_parallel,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm_1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.norm_2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        is_moe = (
            config.pyramid_num_routed
            and layer_number < len(config.pyramid_num_routed)
            and config.pyramid_num_routed[layer_number] > 0
        )
        if is_moe:
            self.mlp = MoESwiGLUFFN(
                config,
                layer_number,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = _dots_swiglu(
                config,
                config.intermediate_size,
                quant_config,
                f"{prefix}.mlp",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm_1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states


class DotsMoEVisionTransformer(nn.Module):
    def __init__(
        self,
        config: Dots3NoteVisionConfig,
        *,
        use_data_parallel: bool,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.hidden_size
        self.patch_embed = DotsPatchEmbed(config)
        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )
        self.blocks = nn.ModuleList(
            [
                MoEVisionBlock(
                    config,
                    layer_idx,
                    use_data_parallel=use_data_parallel,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps) if config.post_norm else None
        self.adapter = PatchMergerAdapter(
            config,
            use_data_parallel=use_data_parallel,
            quant_config=quant_config,
            prefix=f"{prefix}.adapter",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def get_pos_ids_by_grid(self, grid_thw: list[list[int]]) -> list[torch.Tensor]:
        rope_merge_size = self.spatial_merge_size if self.config.pre_pixel_shuffle else 1
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        return pos_ids

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> torch.Tensor:
        pos_ids = torch.cat(self.get_pos_ids_by_grid(grid_thw), dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        pos_ids = pos_ids.to(rotary_pos_emb_full.device)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int | None:
        max_seqlen = None
        if self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
            AttentionBackendEnum.TRITON_ATTN,
        }:
            max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        return max_seqlen

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        grid_tensor = torch.tensor(
            grid_thw,
            device=hidden_states.device,
            dtype=torch.long,
        )
        cu_seqlens = torch.repeat_interleave(
            grid_tensor[:, 1] * grid_tensor[:, 2],
            grid_tensor[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_tensor.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm(hidden_states)
        return self.adapter(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = self.state_dict(keep_vars=True)
        expected_checkpoint_names: set[str] = set()
        for name in params_dict:
            if "fc13." in name:
                expected_checkpoint_names.add(name.replace("fc13.", "fc1."))
                expected_checkpoint_names.add(name.replace("fc13.", "fc3."))
            else:
                expected_checkpoint_names.add(name)

        loaded_params: set[str] = set()
        loaded_checkpoint_names: set[str] = set()
        unexpected: set[str] = set()
        for checkpoint_name, loaded_weight in weights:
            name = checkpoint_name
            shard_id: int | None = None
            if "fc1." in name:
                name = name.replace("fc1.", "fc13.")
                shard_id = 0
            elif "fc3." in name:
                name = name.replace("fc3.", "fc13.")
                shard_id = 1

            if (
                name.endswith(".bias")
                and name not in params_dict
                and name.removesuffix("bias") + "weight" in params_dict
            ):
                continue
            if name not in params_dict:
                unexpected.add(checkpoint_name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            loaded_checkpoint_names.add(checkpoint_name)

        missing = expected_checkpoint_names - loaded_checkpoint_names
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)}")
            raise ValueError("Invalid Dots3 Note vision checkpoint: " + "; ".join(details))
        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    Dots3NoteMultiModalProcessor,
    info=Dots3NoteProcessingInfo,
    dummy_inputs=Dots3NoteDummyInputsBuilder,
)
class Dots3NoteForCausalLM(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        },
    )
    packed_modules_mapping = DeepseekV2ForCausalLM.packed_modules_mapping
    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        if modality.startswith("audio"):
            return AUDIO_PLACEHOLDER
        if modality.startswith("video"):
            return VIDEO_PLACEHOLDER
        raise ValueError("Only image, audio, and video modalities are supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: Dots3NoteConfig = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model_path = vllm_config.model_config.model
        multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.has_vision_tower = _get_dots3_note_vision_config_path(self.model_path) is not None
        self.has_audio_tower = _get_dots3_note_audio_config_path(self.model_path) is not None
        self.config.vision_config = _resolve_dots3_note_vision_config(
            self.model_path,
            self.config.vision_config,
        )
        self.config.audio_config = _resolve_dots3_note_audio_config(
            self.model_path,
            self.config.audio_config,
        )

        self.configure_mm_token_handling(
            vocab_size=self.config.vocab_size,
            mm_token_ids=[
                self.config.image_start_token_id,
                self.config.image_token_id,
                self.config.image_end_token_id,
                self.config.audio_start_token_id,
                self.config.audio_token_id,
                self.config.audio_end_token_id,
            ],
        )

        self.vision_tower: DotsMoEVisionTransformer | None = None
        if self.has_vision_tower:
            with self._mark_tower_model(vllm_config, "image"):
                self.vision_tower = DotsMoEVisionTransformer(
                    self.config.vision_config,
                    use_data_parallel=self.use_data_parallel,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )

        self.audio_tower: Dots3NoteAudioTower | None = None
        if self.has_audio_tower:
            with self._mark_tower_model(vllm_config, "audio"):
                self.audio_tower = Dots3NoteAudioTower(
                    self.config.audio_config,
                    prefix=maybe_prefix(prefix, "audio_tower"),
                )

        with self._mark_language_model(vllm_config):
            self.language_model: DeepseekV2ForCausalLM = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=self.config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["DeepseekV2ForCausalLM"],
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _parse_and_validate_image_input(
        self,
        **kwargs: object,
    ) -> Dots3NoteImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        if pixel_values is None and image_embeds is None:
            return None
        if pixel_values is not None:
            return Dots3NoteImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        return Dots3NoteImageEmbeddingInputs(
            type="image_embeds",
            image_embeds=image_embeds,
            image_grid_thw=image_grid_thw,
        )

    def _process_image_input(
        self,
        image_input: Dots3NoteImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()
        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.language_model.model.embed_tokens.weight.dtype)
        else:
            if self.vision_tower is None:
                raise ValueError(
                    "Raw image inputs require a new_ve vision encoder under "
                    "the model path. Use image_embeds for offline embedding input."
                )
            pixel_values = image_input["pixel_values"].type(self.vision_tower.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_thw_list,
                    rope_type="rope_3d",
                )
            else:
                image_embeds = self.vision_tower(pixel_values, grid_thw_list)

        merge_size = self.config.vision_config.spatial_merge_size
        sizes = (torch.tensor(grid_thw_list, dtype=torch.long).prod(-1) // (merge_size * merge_size)).tolist()
        return image_embeds.split(sizes)

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> Dots3NoteAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        if audio_features is None and audio_embeds is None:
            return None
        if audio_embeds is not None:
            return Dots3NoteAudioEmbeddingInputs(
                type="audio_embeds",
                audio_embeds=audio_embeds,
            )
        return Dots3NoteAudioFeatureInputs(
            type="audio_features",
            audio_features=audio_features,
            audio_sample_lens=kwargs.pop("audio_sample_lens"),
            audio_segment_counts=kwargs.pop("audio_segment_counts"),
            audio_token_lengths=kwargs.pop("audio_token_lengths"),
        )

    def _process_audio_input(
        self,
        audio_input: Dots3NoteAudioInputs,
    ) -> tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return tuple(
                embedding.type(self.language_model.model.embed_tokens.weight.dtype)
                for embedding in audio_input["audio_embeds"]
            )
        if self.audio_tower is None:
            raise ValueError(
                "Raw audio inputs require a new_ae audio encoder under the model path. "
                "Use audio_embeds for offline embedding input."
            )
        return self.audio_tower(
            audio_input["audio_features"],
            audio_input["audio_sample_lens"],
            audio_input["audio_segment_counts"],
            audio_input["audio_token_lengths"],
        )

    def _process_video_input(self, **kwargs: object) -> tuple[torch.Tensor, ...]:
        pixel_values = kwargs.get("video_pixel_values")
        image_grid_thw = kwargs.get("video_image_grid_thw")
        frame_counts = kwargs.get("video_frame_counts")
        layout = kwargs.get("video_layout")
        layout_counts = kwargs.get("video_layout_counts")
        if pixel_values is None:
            return ()
        if not all(isinstance(value, torch.Tensor) for value in (image_grid_thw, frame_counts, layout, layout_counts)):
            raise ValueError("Incomplete Dots3 Note video encoder inputs")
        if self.vision_tower is None:
            raise ValueError("Raw video inputs require Dots3 Note vision weights")

        assert isinstance(pixel_values, torch.Tensor)
        assert isinstance(image_grid_thw, torch.Tensor)
        assert isinstance(frame_counts, torch.Tensor)
        assert isinstance(layout, torch.Tensor)
        assert isinstance(layout_counts, torch.Tensor)
        grid_list = image_grid_thw.reshape(-1, 3).tolist()
        if int(frame_counts.sum()) != len(grid_list):
            raise ValueError("Dots3 Note video frame count does not match image grids")
        pixel_values = pixel_values.type(self.vision_tower.dtype)
        if self.use_data_parallel:
            vision_embeddings = list(
                run_dp_sharded_mrope_vision_model(
                    self.vision_tower,
                    pixel_values,
                    grid_list,
                    rope_type="rope_3d",
                )
            )
        else:
            merged_embeddings = self.vision_tower(pixel_values, grid_list)
            merge_size = self.config.vision_config.spatial_merge_size
            frame_sizes = (torch.tensor(grid_list, dtype=torch.long).prod(-1) // (merge_size * merge_size)).tolist()
            vision_embeddings = list(merged_embeddings.split(frame_sizes))

        audio_embeddings: tuple[torch.Tensor, ...] = ()
        audio_features = kwargs.get("video_audio_features")
        if audio_features is not None:
            if self.audio_tower is None:
                raise ValueError("Video audio requires Dots3 Note audio weights")
            audio_sample_lens = kwargs.get("video_audio_sample_lens")
            audio_segment_counts = kwargs.get("video_audio_segment_counts")
            audio_token_lengths = kwargs.get("video_audio_token_lengths")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    audio_features,
                    audio_sample_lens,
                    audio_segment_counts,
                    audio_token_lengths,
                )
            ):
                raise ValueError("Incomplete Dots3 Note video audio inputs")
            assert isinstance(audio_features, torch.Tensor)
            assert isinstance(audio_sample_lens, torch.Tensor)
            assert isinstance(audio_segment_counts, torch.Tensor)
            assert isinstance(audio_token_lengths, torch.Tensor)
            audio_embeddings = self.audio_tower(
                audio_features,
                audio_sample_lens,
                audio_segment_counts,
                audio_token_lengths,
            )

        per_video_layouts = layout.split(layout_counts.tolist())
        outputs: list[torch.Tensor] = []
        vision_cursor = 0
        audio_cursor = 0
        for video_layout in per_video_layouts:
            chunks: list[torch.Tensor] = []
            for token_length in video_layout.tolist():
                if token_length > 0:
                    embedding = vision_embeddings[vision_cursor]
                    vision_cursor += 1
                    expected_length = token_length
                else:
                    embedding = audio_embeddings[audio_cursor]
                    audio_cursor += 1
                    expected_length = -token_length
                if embedding.shape[0] != expected_length:
                    raise ValueError(
                        "Dots3 Note video embedding length mismatch: "
                        f"expected {expected_length}, got {embedding.shape[0]}"
                    )
                chunks.append(embedding)
            outputs.append(torch.cat(chunks, dim=0))

        if vision_cursor != len(vision_embeddings) or audio_cursor != len(audio_embeddings):
            raise ValueError("Dots3 Note video layout did not consume all encoder outputs")
        return tuple(outputs)

    def get_num_mm_encoder_tokens(self, num_image_tokens: int) -> int:
        return num_image_tokens * (self.config.vision_config.spatial_merge_size**2)

    def get_num_mm_connector_tokens(self, num_vision_tokens: int) -> int:
        return num_vision_tokens // (self.config.vision_config.spatial_merge_size**2)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            multimodal_embeddings += self._process_image_input(image_input)
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is not None:
            multimodal_embeddings += self._process_audio_input(audio_input)
        multimodal_embeddings += self._process_video_input(**kwargs)
        return multimodal_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def iter_language_weights():
            for name, weight in weights:
                if name.startswith(("vision_encoder.", "audio_encoder.")):
                    continue
                yield name, weight

        loaded_params = {f"language_model.{name}" for name in self.language_model.load_weights(iter_language_weights())}

        if self.vision_tower is not None:
            vision_path = os.path.join(self.model_path, "model-vision.safetensors")
            if not os.path.exists(vision_path):
                raise FileNotFoundError(f"Dots3 Note vision tower is enabled but weights are missing: {vision_path}")

            def iter_vision_weights():
                with safe_open(vision_path, framework="pt", device="cpu") as handle:
                    for key in handle.keys():  # noqa: SIM118
                        yield key.removeprefix("vision_encoder."), handle.get_tensor(key)

            vision_loaded = self.vision_tower.load_weights(iter_vision_weights())
            loaded_params.update(f"vision_tower.{name}" for name in vision_loaded)

        if self.audio_tower is None:
            return loaded_params

        audio_path = os.path.join(self.model_path, "model-audio.safetensors")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Dots3 Note audio tower is enabled but weights are missing: {audio_path}")

        def iter_audio_weights():
            with safe_open(audio_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():  # noqa: SIM118
                    yield key.removeprefix("audio_encoder."), handle.get_tensor(key)

        audio_loaded = self.audio_tower.load_weights(iter_audio_weights())
        loaded_params.update(f"audio_tower.{name}" for name in audio_loaded)
        return loaded_params

    def get_mm_mapping(self) -> MultiModelKeys:
        if self.vision_tower is None and self.audio_tower is None:
            return MultiModelKeys.from_string_field(language_model="language_model")
        connectors = []
        towers = []
        if self.vision_tower is not None:
            connectors.append("vision_tower.adapter")
            towers.append("vision_tower.")
        if self.audio_tower is not None:
            connectors.append("audio_tower.audio_adapter")
            towers.append("audio_tower.")
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=connectors,
            tower_model=towers,
        )
