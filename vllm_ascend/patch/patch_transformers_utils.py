#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

from typing import TYPE_CHECKING, Any, Optional, Union, cast

from transformers.processing_utils import ProcessorMixin
from typing_extensions import TypeVar
from vllm.transformers_utils import processor
from vllm.transformers_utils.processor import (_merge_mm_kwargs,
                                               cached_get_image_processor,
                                               cached_get_processor,
                                               cached_get_video_processor)

if TYPE_CHECKING:
    from vllm.config import ModelConfig

_P = TypeVar("_P", bound=ProcessorMixin, default=ProcessorMixin)


def get_processor(
    processor_name: str,
    *args: Any,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    processor_cls: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    """Load a processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoProcessor

    processor_factory = (AutoProcessor if processor_cls == ProcessorMixin or
                         isinstance(processor_cls, tuple) else processor_cls)

    try:
        processor = processor_factory.from_pretrained(
            processor_name,
            *args,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the processor. If the processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    if not isinstance(processor, processor_cls):
        raise TypeError("Invalid type of HuggingFace processor. "
                        f"Expected type: {processor_cls}, but "
                        f"found type: {type(processor)}")

    return processor


def cached_processor_from_config(
    model_config: "ModelConfig",
    processor_cls: Union[type[_P], tuple[type[_P], ...]] = ProcessorMixin,
    **kwargs: Any,
) -> _P:
    return cached_get_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        processor_cls=processor_cls,  # type: ignore[arg-type]
        **_merge_mm_kwargs(model_config, **kwargs),
    )


def get_image_processor(
    processor_name: str,
    *args: Any,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load an image processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers import AutoImageProcessor
    from transformers.image_processing_utils import BaseImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(
            processor_name,
            *args,
            revision=revision,
            trust_remote_code=trust_remote_code,
            **kwargs)
    except ValueError as e:
        # If the error pertains to the processor class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        # Unlike AutoTokenizer, AutoImageProcessor does not separate such errors
        if not trust_remote_code:
            err_msg = (
                "Failed to load the image processor. If the image processor is "
                "a custom processor not yet available in the HuggingFace "
                "transformers library, consider setting "
                "`trust_remote_code=True` in LLM or using the "
                "`--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e

    return cast(BaseImageProcessor, processor)


def cached_image_processor_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    return cached_get_image_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        **_merge_mm_kwargs(model_config, **kwargs),
    )


def get_video_processor(
    processor_name: str,
    *args: Any,
    revision: Optional[str] = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a video processor for the given model name via HuggingFace."""
    # don't put this import at the top level
    # it will call torch.cuda.device_count()
    from transformers.image_processing_utils import BaseImageProcessor

    processor = get_processor(
        processor_name,
        *args,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    return cast(BaseImageProcessor, processor.video_processor)


def cached_video_processor_from_config(
    model_config: "ModelConfig",
    **kwargs: Any,
):
    return cached_get_video_processor(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
        **_merge_mm_kwargs(model_config, **kwargs),
    )


# Adapted from vllm: https://github.com/vllm-project/vllm/pull/17948
# Pass `revision` param to transformer processor to avoid using `main` as
# default branch when using modelscope.
# Find more details at:
# https://github.com/vllm-project/vllm-ascend/issues/829
processor.get_processor = get_processor
processor.cached_processor_from_config = cached_processor_from_config
processor.get_image_processor = get_image_processor
processor.cached_image_processor_from_config = cached_image_processor_from_config
processor.get_video_processor = get_video_processor
processor.cached_video_processor_from_config = cached_video_processor_from_config
