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
from vllm.distributed import tensor_model_parallel_all_reduce


def vocab_embedding_forward(self, input_):
    if self.tp_size > 1:
        # Build the mask.
        masked_input, input_mask = self._get_masked_input_and_mask(
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
