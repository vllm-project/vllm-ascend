#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch
from vllm.model_executor.layers.rotary_embedding.dual_chunk_rope import (
    DualChunkRotaryEmbedding,
)
from vllm.platforms import current_platform


def _dual_chunk_rope_init(
    self,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    chunk_size: int,
    local_size: int,
) -> None:
    # Call nn.Module.__init__ directly to avoid the CUDA device lookup in
    # the original DualChunkRotaryEmbedding.__init__.
    torch.nn.Module.__init__(self)
    self.head_size = head_size
    self.rotary_dim = rotary_dim
    self.max_position_embeddings = max_position_embeddings
    self.base = base
    self.is_neox_style = is_neox_style
    self.chunk_size = chunk_size
    self.local_size = local_size
    self.dtype = dtype
    # Use current_platform.device_name instead of the hardcoded
    # "cuda:<N>" string so that this works on Ascend NPU and any other
    # non-CUDA backend.  (Fixes vllm-ascend issue #4309.)
    self.device = torch.device(current_platform.device_name)
    (q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache) = (
        self._compute_cos_sin_cache()
    )

    self.register_buffer("cos_sin_q_cache", q_cache, persistent=False)
    self.register_buffer("cos_sin_qc_cache", qc_cache, persistent=False)
    self.register_buffer("cos_sin_k_cache", k_cache, persistent=False)
    self.register_buffer(
        "cos_sin_qc_no_clamp_cache", qc_no_clamp_cache, persistent=False
    )
    self.register_buffer("cos_sin_q_inter_cache", q_inter_cache, persistent=False)


# Patch DualChunkRotaryEmbedding.__init__ to replace the hardcoded
# `torch.device(f"cuda:{torch.cuda.current_device()}")` with a
# platform-aware device string so that Qwen2.5-7B-Instruct-1M (and any
# other model that uses DualChunkAttention) can run on Ascend NPU.
# TODO: Remove this patch when upstream vLLM fixes the hardcoded CUDA
# device in dual_chunk_rope.py.
DualChunkRotaryEmbedding.__init__ = _dual_chunk_rope_init
