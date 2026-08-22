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
#

import torch
import vllm.model_executor.models.utils as models_utils
from vllm.model_executor.models.utils import (
    _embedding_count_expression,
    _flatten_embeddings,
)
from vllm.multimodal import NestedTensors


def merge_multimodal_embeddings_310(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # Boolean IndexPut fails on 310P; use integer indices instead.
        positions = is_multimodal.reshape(-1).cpu().nonzero(as_tuple=False).flatten()
        positions = positions.to(device=inputs_embeds.device)
        inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).index_copy_(
            0, positions, mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during index put operation") from e

    return inputs_embeds


models_utils._merge_multimodal_embeddings = merge_multimodal_embeddings_310
