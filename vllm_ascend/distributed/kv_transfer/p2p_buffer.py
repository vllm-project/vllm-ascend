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

from typing import Optional, List
import zlib

import torch
import llm_datadist  # type: ignore

from vllm.distributed.kv_transfer.kv_lookup_buffer.base import (
    KVLookupBufferBase,
)
from vllm.logger import init_logger

from vllm_ascend.distributed.kv_transfer.llmdatadist_pipe import LLMDataDistPipe

logger = init_logger(__name__)


TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32,
}


# Hash a string into a int32 value.
def int32_hash(data):
    assert isinstance(data, str)
    data = data.encode("utf-8")
    return zlib.adler32(data)


class LLMDataDistBuffer(KVLookupBufferBase):
    def __init__(self, data_pipe: LLMDataDistPipe):
        self.data_pipe = data_pipe
        # Consumer buffer need these information to construct receiving buffer.
        self.num_layers: int = None
        self.num_heads: int = None
        self.head_size: int = None
        self.dtype: torch.dtype = None
        self.hidden_size: int = None
        self.key_buffer = None
        self.value_buffer = None
        self.hidden_buffer = None

    def insert(
        self,
        input_tokens: torch.Tensor,
        roi: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        hidden: torch.Tensor,
        req_id: str,
    ) -> None:
        """
        seq_len: num_tokens of current request.
        input_tokens: [seq_len]
        roi: [seq_len]
        key: [num_layers, seq_len, num_kv_heads, head_size]
        value: [num_layers, seq_len, num_kv_heads, head_size]
        hidden: [seq_len, hidden_size]
        """
        orig_k_shape = key.shape
        num_layers = orig_k_shape[0]

        # unsequeeze all tensors to make first dim to 1.
        # This is because D node can only pull one batch data from P.
        # So we make first dim to 1 here in order to pull full data.
        key = key.view(num_layers, -1).unsqueeze(0)
        value = value.view(num_layers, -1).unsqueeze(0)
        hidden = hidden.unsqueeze(0)

        hidden_dtype = key.dtype
        # initialize LLMDatadist data structure
        key_desc = llm_datadist.CacheDesc(
            1,
            key.shape,
            TORCH_DTYPE_TO_NPU_DTYPE[hidden_dtype],
            seq_len_dim_index=1,
        )
        value_desc = llm_datadist.CacheDesc(
            1,
            value.shape,
            TORCH_DTYPE_TO_NPU_DTYPE[hidden_dtype],
            seq_len_dim_index=1,
        )
        hidden_desc = llm_datadist.CacheDesc(
            1,
            hidden.shape,
            TORCH_DTYPE_TO_NPU_DTYPE[hidden_dtype],
            seq_len_dim_index=-1,
        )

        req_id = int32_hash(req_id)
        key_cache_key = llm_datadist.CacheKey(
            self.data_pipe.cluster_id, req_id, 1
        )
        value_cache_key = llm_datadist.CacheKey(
            self.data_pipe.cluster_id, req_id, 2
        )
        hidden_cache_key = llm_datadist.CacheKey(
            self.data_pipe.cluster_id, req_id, 3
        )

        # Currently we use hash value of request id as key, so no need to send input_tokens
        self.key_buffer = self.data_pipe.send_tensor(
            key, key_desc, key_cache_key
        )
        self.value_buffer = self.data_pipe.send_tensor(
            value, value_desc, value_cache_key
        )
        self.hidden_buffer = self.data_pipe.send_tensor(
            hidden, hidden_desc, hidden_cache_key
        )

    def drop_select(
        self,
        input_tokens: torch.Tensor,
        roi: Optional[torch.Tensor],
        req_id: str,
    ) -> List[Optional[torch.Tensor]]:
        """Select and *drop* KV cache entries from the lookup buffer.

        The functionality is similar to the following python statements
        ```
        ret = buffer.pop(input_tokens, roi)
        return ret
        ```

        Args:
            input_tokens (torch.Tensor): token IDs.
            roi (torch.Tensor): A binary mask on top of the input tokens

        Returns:
            A list of tensors including:
                key: [num_layers, num_tokens, num_heads, head_size]
                value: [num_layers, num_tokens, num_heads, head_size]
                hidden_or_intermediate_states: [num_tokens, hidden_size]
                roi: None (Currently we don't supported roi)
        """
        orig_req_id = req_id
        req_id = int32_hash(req_id)
        num_tokens = input_tokens.shape[0]
        kv_shape = (
            1,
            self.num_layers,
            num_tokens * self.num_heads * self.head_size,
        )
        hidden_shape = (1, num_tokens, self.hidden_size)
        key_desc = llm_datadist.CacheDesc(
            1,
            kv_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[self.dtype],
            seq_len_dim_index=-1,
        )
        value_desc = llm_datadist.CacheDesc(
            1,
            kv_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[self.dtype],
            seq_len_dim_index=-1,
        )
        hidden_desc = llm_datadist.CacheDesc(
            1,
            hidden_shape,
            TORCH_DTYPE_TO_NPU_DTYPE[self.dtype],
            seq_len_dim_index=-1,
        )

        key_cache_key = llm_datadist.CacheKey(
            self.data_pipe.cluster_id, req_id, 1
        )
        value_cache_key = llm_datadist.CacheKey(
            self.data_pipe.cluster_id, req_id, 2
        )
        hidden_cache_key = llm_datadist.CacheKey(
            self.data_pipe.cluster_id, req_id, 3
        )

        # Deallocate buffer allocated in last round.
        if self.key_buffer:
            self.data_pipe.deallocate_buffer(self.key_buffer)
            self.data_pipe.deallocate_buffer(self.value_buffer)
            self.data_pipe.deallocate_buffer(self.hidden_buffer)

        try:
            self.key_buffer, key = self.data_pipe.recv_tensor(
                key_desc, key_cache_key
            )
            self.value_buffer, value = self.data_pipe.recv_tensor(
                value_desc, value_cache_key
            )
            self.hidden_buffer, hidden = self.data_pipe.recv_tensor(
                hidden_desc, hidden_cache_key
            )
            key = key.view(
                self.num_layers, num_tokens, self.num_heads, self.head_size
            )
            value = value.view(
                self.num_layers, num_tokens, self.num_heads, self.head_size
            )
            hidden = hidden.view(num_tokens, self.hidden_size)
        except Exception as e:
            logger.warning(
                f"Faile to receive kv cache and hidden states of request: {orig_req_id}"
                f"Error is {str(e)}"
            )
            return [None, None, None, None]

        return [key, value, hidden, roi]

    def close(self):
        pass
