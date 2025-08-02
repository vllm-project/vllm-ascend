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

from typing import Tuple

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding


def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index)
    # Adapt: avoid create added_vocab_mask when added_vocab_start_index == added_vocab_end_index.
    if added_vocab_start_index == added_vocab_end_index:
        valid_offset = (org_vocab_start_index * org_vocab_mask)
        vocab_mask = org_vocab_mask
    else:
        added_vocab_mask = (input_ >= added_vocab_start_index) & (
            input_ < added_vocab_end_index)
        added_offset = added_vocab_start_index - (
            org_vocab_end_index -
            org_vocab_start_index) - num_org_vocab_padding
        valid_offset = (org_vocab_start_index *
                        org_vocab_mask) + (added_offset * added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
    # Adapt end.
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


def vocab_parallel_embedding_forward(self, input_):
    if self.tp_size > 1:
        # Build the mask.
        masked_input, input_mask = get_masked_input_and_mask(
            input_, self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index)
    else:
        masked_input = input_
    # Get the embeddings.
    output_parallel = self.quant_method.embedding(self, masked_input.long())
    # Mask the output embedding.
    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    # Reduce across all the model parallel GPUs.
    output = tensor_model_parallel_all_reduce(output_parallel)
    return output


VocabParallelEmbedding.forward = vocab_parallel_embedding_forward
