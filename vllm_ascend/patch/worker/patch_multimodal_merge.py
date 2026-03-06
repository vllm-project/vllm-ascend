#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
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

import torch
import vllm
from vllm.logger import logger
from vllm.model_executor.models.utils import _embedding_count_expression, _flatten_embeddings
from vllm.multimodal import NestedTensors


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    is_multimodal: torch.Tensor,
    multimodal_embeddings: NestedTensors,
) -> torch.Tensor:
    """
    Merge ``multimodal_embeddings`` into ``inputs_embeds`` by overwriting the
    positions in ``inputs_embeds`` corresponding to placeholder tokens in
    ``input_ids``.

    Note:
        This updates ``inputs_embeds`` in place.
    """
    flattened = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype
    num_multimodal_tokens = flattened.shape[0]
    num_placeholders = is_multimodal.sum().item()
    assert isinstance(num_placeholders, int)

    if num_multimodal_tokens == num_placeholders:
        try:
            inputs_embeds[is_multimodal] = flattened.to(dtype=input_dtype)
        except RuntimeError as e:
            raise ValueError("Error during masked scatter operation") from e
    elif num_multimodal_tokens < num_placeholders:
        # When the visual encoder compresses tokens (e.g., via pooling/merge
        # layers), fewer multimodal embeddings are produced than placeholder
        # positions. Only overwrite the first num_multimodal_tokens positions
        # and leave the remaining placeholder positions unchanged.
        logger.warning_once(
            "Multimodal token count (%d) is less than placeholder count (%d). "
            "Only the first %d placeholder positions will be overwritten.",
            num_multimodal_tokens,
            num_placeholders,
            num_multimodal_tokens,
        )
        placeholder_indices = is_multimodal.nonzero(as_tuple=True)[0]
        selected_indices = placeholder_indices[:num_multimodal_tokens]
        inputs_embeds[selected_indices] = flattened.to(dtype=input_dtype)
    else:
        expr = _embedding_count_expression(multimodal_embeddings)
        raise ValueError(
            f"Attempted to assign {expr} = {num_multimodal_tokens} "
            f"multimodal tokens to {num_placeholders} placeholders"
        )

    return inputs_embeds


vllm.model_executor.models.utils._merge_multimodal_embeddings = _merge_multimodal_embeddings
