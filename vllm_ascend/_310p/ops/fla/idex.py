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
# from collections.abc import Iterable
# mypy: ignore-errors
import torch
from vllm.model_executor.layers.fla.ops.index import prepare_lens
from vllm.model_executor.layers.fla.ops.utils import tensor_cache


@tensor_cache
def prepare_chunk_indices_310(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:

    seq_lens = prepare_lens(cu_seqlens)
    num_chunks = (seq_lens + chunk_size - 1) // chunk_size

    indices_list = []
    for n in num_chunks.tolist():
        indices_list.append(torch.arange(n, device=cu_seqlens.device))
    indices = torch.cat(indices_list)

    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], dim=1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets_310(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:

    seq_lens = prepare_lens(cu_seqlens)

    num_chunks = (seq_lens + chunk_size - 1) // chunk_size

    return torch.cat([torch.tensor([0], device=cu_seqlens.device, dtype=cu_seqlens.dtype), num_chunks]).cumsum(dim=-1)
