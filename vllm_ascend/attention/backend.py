#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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



from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.attention.backend import AttentionImpl

@dataclass
class BackendRuntimeInputBatch:
    num_reqs_padded: int
    query_start_loc_np: np.ndarray
    query_start_lo: torch.Tensor

    
class AscendFiaAttentionImpl(AttentionImpl):
    runtime_input = None

    @classmethod
    def preprocess(cls,
                   num_reqs: int,
                   query_start_loc_np: np.ndarray,
                   decode_query_len: int,
                   batch_desc: BatchExecutionDescriptor):
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            return None
        
        if cls.runtime_input == None:
            cls.runtime_input = BackendRuntimeInputBatch()
            cls.runtime_input.query_start_loc_np = np

        num_reqs_padded = batch_desc.num_reqs or num_reqs
        num_tokens_padded = batch_desc.num_tokens
        if num_tokens_padded == num_reqs_padded * decode_query_len:
            # Uniform-batch case: num_reqs must be no greater than num_reqs_padded
            assert num_reqs <= num_reqs_padded

            last_loc = query_start_loc_np[num_reqs]
            query_start_loc_np[num_reqs + 1 : num_reqs_padded + 1] = (
                np.arange(1, num_reqs_padded + 1 - num_reqs) * decode_query_len + last_loc
            )
        else:
            # Mixed-batch case: num_reqs must equal num_reqs_padded
            assert num_reqs == num_reqs_padded

            # Insert a dummy request instead of setting query_start_loc[num_reqs] = num_tokens_padded directly
            query_start_loc_np[num_reqs_padded + 1] = num_tokens_padded
            num_reqs_padded = num_reqs_padded + 1
        return num_reqs_padded, query_start_loc_np