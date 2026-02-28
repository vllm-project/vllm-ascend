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
"""
Patch for vLLM's _merge_multimodal_embeddings to avoid D2H sync issues on Ascend.
Replaces the masked_scatter_ operation with a custom index-based assignment.
"""

import sys
import torch
from vllm.model_executor.models import utils


def masked_scatter_with_index_put(inputs_embeds, is_multimodal, mm_embeds_flat):
    is_multimodal = is_multimodal.bool()
    row_indices = is_multimodal.squeeze(-1).nonzero().squeeze(-1)
    num_rows_to_replace = row_indices.size(0)
    inputs_embeds[row_indices] = mm_embeds_flat[:num_rows_to_replace]
    
    return inputs_embeds


def NPU_merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    def masked_scatter_with_index_put(inputs_embeds, is_multimodal, mm_embeds_flat):
        is_multimodal = is_multimodal.bool()
        row_indices = is_multimodal.squeeze(-1).nonzero().squeeze(-1)
        num_rows_to_replace = row_indices.size(0)
        inputs_embeds[row_indices] = mm_embeds_flat[:num_rows_to_replace]
        
        return inputs_embeds
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        inputs_embeds=masked_scatter_with_index_put(inputs_embeds, is_multimodal.unsqueeze(-1),mm_embeds_flat.to(dtype=input_dtype))
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


def apply_patch():
    utils._merge_multimodal_embeddings = NPU_merge_multimodal_embeddings



# Automatically apply the patch when this module is imported
apply_patch()
