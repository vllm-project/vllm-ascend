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

from vllm.config import get_current_vllm_config
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.utils import singleton

@dataclass
class BackendExtraInput:
    """Lazily filled by ``prepare_extra_input``; constructed empty then populated."""

    num_reqs_padded: int = 0
    query_start_loc_np: np.ndarray | None = None
    query_start_loc: torch.Tensor | None = None

class BackendExtraInputPreparer(ABC):
    @abstractmethod
    def prepare(self,
                num_reqs: int,
                query_start_loc_np: np.ndarray,
                decode_query_len: int,
                batch_desc: BatchExecutionDescriptor):
        raise NotImplementedError

@singleton
class FiaExtraInputPreparer(BackendExtraInputPreparer):
    def __init__(self):
        vllm_config = get_current_vllm_config()
        max_num_reqs = vllm_config.scheduler_config.max_num_reqs

        self.extra_input = BackendExtraInput(
            query_start_loc_np=np.empty(max_num_reqs + 2, dtype=np.int32),
            query_start_loc=torch.zeros(
                max_num_reqs + 2, dtype=torch.int32, device="npu"
            ),
        )

    def prepare(self,
                num_reqs: int,
                query_start_loc_np: np.ndarray,
                decode_query_len: int,
                batch_desc: BatchExecutionDescriptor):
        """
        This function is only designed to satisfied the constraint that when the layout is TND,
        the first dimension of `hidden_states` must equal the last element of `actual_seq_lengths_q`.
        """
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

        self.extra_input.num_reqs_padded = num_reqs_padded
        self.extra_input.query_start_loc_np[:num_reqs_padded + 1] = query_start_loc_np[:num_reqs_padded + 1]
        async_copy_to_gpu(self.extra_input.query_start_loc_np, out=self.extra_input.query_start_loc)

        return self.extra_input
    
class AscendAttentionBackend(AttentionBackend):
    @staticmethod
    @abstractmethod
    def get_extra_input_preparer() -> BackendExtraInputPreparer:
        raise NotImplementedError
