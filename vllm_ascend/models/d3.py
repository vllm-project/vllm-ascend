# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import TYPE_CHECKING

from transformers.models.qwen3_vl import Qwen3VLProcessor
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.adapters import as_seq_cls_model
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_ascend.models.qwen3_5_vl import (
    AscendQwen3_5ForConditionalGeneration,
    AscendQwen3_5ProcessingInfo,
    AscendQwen3_5VLProcessor,
)

READOUT_WEIGHTS = "readout.safetensors"
IMAGE_SIZE = {"shortest_edge": 65_536, "longest_edge": 1_638_400}
VIDEO_FRAME_MAX_PIXELS = 200_704
VIDEO_FRAME_MIN_PIXELS = 4_096
VIDEO_MAX_FRAMES = 32


class D3ProcessingInfo(AscendQwen3_5ProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        processor = super().get_hf_processor(**kwargs)
        processor.image_processor.size = dict(IMAGE_SIZE)
        video = processor.video_processor
        factor = video.patch_size * video.merge_size
        video.size = {
            "shortest_edge": VIDEO_FRAME_MIN_PIXELS,
            "longest_edge": VIDEO_FRAME_MAX_PIXELS * VIDEO_MAX_FRAMES,
        }
        video.cap_pixels_per_frame = True
        video.max_video_tokens = VIDEO_FRAME_MAX_PIXELS // factor**2
        video.do_sample_frames = False
        return processor


if TYPE_CHECKING:
    _D3ForSequenceClassificationBase = AscendQwen3_5ForConditionalGeneration
else:
    _D3ForSequenceClassificationBase = as_seq_cls_model(AscendQwen3_5ForConditionalGeneration)


@default_pooling_type(seq_pooling_type="LAST")
@MULTIMODAL_REGISTRY.register_processor(
    AscendQwen3_5VLProcessor,
    info=D3ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class AscendD3ForSequenceClassification(_D3ForSequenceClassificationBase):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "language_model.": "language_model.model.",
            "mtp.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        model_config = vllm_config.model_config
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=model_config.model,
                revision=model_config.revision,
                prefix="score.",
                allow_patterns_overrides=[READOUT_WEIGHTS],
            )
        ]

    @staticmethod
    def is_unused_checkpoint_weight(name: str) -> bool:
        return name == "weight"
