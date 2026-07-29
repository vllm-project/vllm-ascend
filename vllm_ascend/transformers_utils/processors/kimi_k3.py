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
from vllm.tokenizers.hf import HfTokenizer


class KimiK3Processor(ProcessorMixin):
    """HF-style adapter for K3's standard ``image`` modality.

    The checkpoint image processor consumes media dictionaries, while vLLM
    supplies bare image objects. Prompt expansion is intentionally left to the
    model's prompt-update hook so cached and uncached processor paths share one
    implementation.
    """

    attributes = ["image_processor", "tokenizer"]

    def __init__(
        self,
        image_processor: BaseImageProcessor,
        tokenizer: HfTokenizer,
    ) -> None:
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: object | list[object] | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        del kwargs
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            media_inputs = [{"type": "image", "image": image} for image in images]
            mm_inputs = self.image_processor.preprocess(
                media_inputs,
                return_tensors=return_tensors,
            )
        else:
            mm_inputs = {}

        if text is None:
            text_inputs = {}
        else:
            texts = [text] if isinstance(text, str) else list(text)
            text_inputs = self.tokenizer(texts)

        return BatchFeature(data={**text_inputs, **mm_inputs}, tensor_type=return_tensors)


__all__ = ["KimiK3Processor"]
