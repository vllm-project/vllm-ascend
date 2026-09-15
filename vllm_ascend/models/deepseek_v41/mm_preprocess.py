# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 preprocessing from the supported vLLM common implementation.

Only public naming aliases live here; tokenization, image transforms and
compressor-alignment placeholder updates are owned by upstream vLLM.
"""

from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    COMPRESS_PAD_TO as COMPRESS_PAD_TO,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE as IMAGE,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_END as IMAGE_END,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_NEW_LINE as IMAGE_NEW_LINE,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_PAD_ID as IMAGE_PAD_ID,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_PAD_TOKEN_NAME as IMAGE_PAD_TOKEN_NAME,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_PLACEHOLDER as IMAGE_PLACEHOLDER,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_SENTINEL_BASE_ID as IMAGE_TOKEN_ID,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    IMAGE_START as IMAGE_START,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLDummyInputsBuilder as DeepseekV41VLDummyInputsBuilder,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLImageProcessor as DeepseekV41VLImageProcessor,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLMultiModalProcessor as DeepseekV41VLMultiModalProcessor,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLProcessingInfo as DeepseekV41VLProcessingInfo,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    DeepseekV4VLProcessor as DeepseekV41VLProcessor,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    image_sentinel_mask as image_sentinel_mask,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    image_token_types as image_token_types,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    llm_grid as llm_grid,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    load_image as load_image,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    num_image_tokens as num_image_tokens,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    safe_resize as safe_resize,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    solve_resize_ratio as solve_resize_ratio,
)
from vllm.models.deepseek_v4_1.common.mm_preprocess import (
    validate_image_sentinel_ids as validate_image_sentinel_ids,
)


def leading_compressor_pad(start_pos: int) -> int:
    return COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO


__all__ = [
    "IMAGE_TOKEN_ID",
    "DeepseekV41VLImageProcessor",
    "DeepseekV41VLProcessor",
    "DeepseekV41VLProcessingInfo",
    "DeepseekV41VLDummyInputsBuilder",
    "DeepseekV41VLMultiModalProcessor",
]
