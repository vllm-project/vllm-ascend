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
from vllm.model_executor.models.utils import (_embedding_count_expression,
                                              _flatten_embeddings)
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
    try:
        inputs_embeds[is_multimodal] = flattened
    except RuntimeError as e:
        num_expected_tokens = is_multimodal.sum().item()
        assert isinstance(num_expected_tokens, int)

        if flattened.shape[0] != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)
            raise ValueError(
                f"Attempted to assign {expr} = {flattened.shape[0]} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e
        else:
            raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


vllm.model_executor.models.utils._merge_multimodal_embeddings = _merge_multimodal_embeddings
