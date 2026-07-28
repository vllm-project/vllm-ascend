#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Thin vLLM-facing processor adapter for Kimi K3 images."""

from transformers import BaseImageProcessor, BatchFeature, TensorType
from transformers.processing_utils import ProcessorMixin
from vllm.multimodal.inputs import VisionChunk
from vllm.tokenizers.hf import HfTokenizer


class KimiK3Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
        media_token_id: int,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.media_token_id = media_token_id

    @staticmethod
    def _inject_image_sizes(text: str, vision_chunks: list[VisionChunk]) -> str:
        marker = "<|media_begin|>image<|media_content|>"
        for chunk in vision_chunks:
            if chunk["type"] != "image":
                raise ValueError("Kimi K3 currently supports image inputs only")
            image = chunk["image"]
            if not hasattr(image, "size"):
                raise ValueError("Kimi K3 image processor did not resolve the input to a PIL image")
            width, height = image.size
            replacement = f"<|media_begin|>image {width}x{height}<|media_content|>"
            text = text.replace(marker, replacement, 1)
        return text

    def __call__(
        self,
        text: str | list[str] | None = None,
        vision_chunks: list[VisionChunk] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        del kwargs
        if vision_chunks is not None:
            # K3's remote image processor accepts vLLM VisionChunkImage
            # directly (it is a TypedDict) and normalizes image payloads in
            # place.  Do this before injecting the resolved WxH prompt.
            mm_inputs = self.image_processor.preprocess(vision_chunks, return_tensors=return_tensors)
        else:
            mm_inputs = {}

        if text is None:
            text_inputs = {}
        else:
            texts = [text] if isinstance(text, str) else list(text)
            if vision_chunks:
                if len(texts) != 1:
                    raise ValueError("Kimi K3 multimodal preprocessing expects one prompt per processor call")
                texts[0] = self._inject_image_sizes(texts[0], vision_chunks)
            text_inputs = self.tokenizer(texts)
            input_ids: list[list[int]] = text_inputs["input_ids"]  # type: ignore[assignment]
            if vision_chunks:
                token_counts = [self.image_processor.media_tokens_calculator(item) for item in vision_chunks]
                for row in input_ids:
                    expanded: list[int] = []
                    for token in row:
                        if token == self.media_token_id:
                            expanded.extend([token] * token_counts.pop(0))
                        else:
                            expanded.append(token)
                    row[:] = expanded
                if token_counts:
                    raise ValueError("Not all Kimi K3 images had a matching media placeholder")

        return BatchFeature(data={**text_inputs, **mm_inputs}, tensor_type=return_tensors)


__all__ = ["KimiK3Processor"]
