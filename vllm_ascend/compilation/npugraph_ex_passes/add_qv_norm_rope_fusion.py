#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
import functools
import sys

import torch
from torch._inductor.pattern_matcher import Match
from vllm.logger import logger


@functools.lru_cache(None)
# The replacement registered here will be actually executed after AOT.
class QKNormRopeFusionPattern:

    def __init__(self,
                 vllm_config,
                 head_dim,
                 num_heads,
                 num_kv_heads,
                 eps=1e-6):
        self.vllm_config = vllm_config
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = eps
        self.device = vllm_config.device_config.device if vllm_config.device_config else None

    def _extra_stream_scope_check(match: Match) -> bool:
        """
        Checks if all nodes in the same stream.
        """
        non_default_streams = set()
        has_default = False

        for node in match.nodes:
            if node.op == "call_function":
                current_stream = node.meta.get("stream_label")
                if current_stream is None:
                    has_default = True
                else:
                    non_default_streams.add(current_stream)
                    if len(non_default_streams) > 1:
                        logger.debug(
                            f"Cross-stream operation detected in pattern match for AddRMSNormQuant. "
                            f"Multiple streams found: {non_default_streams}. "
                            f"Fusion is not supported for cross-stream operations."
                        )
                        return False

        if has_default and len(non_default_streams) > 0:
            logger.debug(
                f"Cross-stream operation detected in pattern match for AddRMSNormQuant. "
                f"Multiple streams found: {non_default_streams}. "
                f"Fusion is not supported for cross-stream operations.")
            return False

        return True

    def get_inputs():
        T = 5
        qkv = torch.empty(T,
                          self.q_size + 2 * self.kv_size,
                          dtype=torch.bfloat16,
                          device="npu")
        q_weight = torch.empty(self.head_dim,
                               dtype=torch.bfloat16,
                               device="npu")
        k_weight = torch.empty(self.head_dim,
                               dtype=torch.bfloat16,
                               device="npu")
        cos = torch.empty(1,
                          T,
                          1,
                          self.head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        sin = torch.empty(1,
                          T,
                          1,
                          self.head_dim,
                          dtype=torch.bfloat16,
                          device="npu")
        return [qkv, q_weight, k_weight, cos, sin]
    
    