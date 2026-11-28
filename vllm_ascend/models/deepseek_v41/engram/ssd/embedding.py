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
"""Consumer view over the SSD-backed engram tables.

The embedding owns no weights: rows live in the SSD row cache's payload
planes and are produced per step by :class:`EngramSSDRuntime`. This class
only carries the shard geometry the existing preparation code queries
(``dim``/``n_hash_cols``/``head_start``/``part_n_hash_cols``/``tp_size``/
``dp_size``) so the model, loader and capture checks keep a single shape.
"""

from __future__ import annotations

import torch
from torch import nn


class AscendSSDEngramEmbedding(nn.Module):
    """Weightless per-layer view onto the shared SSD runtime."""

    def __init__(self, num_embeddings: int, dim: int, head_sizes: tuple[int, ...],
                 layer_hash_index: int, tp_size: int, head_start: int,
                 part_n_hash_cols: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.layer_hash_index = layer_hash_index
        self.head_sizes = tuple(head_sizes)
        self.n_hash_cols = len(head_sizes)
        self.tp_size = tp_size
        # SSD 共享载荷平面: 每同 TP 位 DP 组一份, 各 rank 独立查找自己的行
        self.dp_size = 1
        self.head_start = head_start
        self.part_n_hash_cols = part_n_hash_cols
        self.vocab_start_idx = sum(self.head_sizes[:head_start])
        self.vocab_end_idx = sum(self.head_sizes[:head_start + part_n_hash_cols])
        # 消费接口: prepare 代码通过 lookup 形状推导输出宽度
        self.is_ssd = True

    def extra_repr(self) -> str:
        return (f"ssd rows={self.num_embeddings}, dim={self.dim}, "
                f"cols={self.head_start}..{self.head_start + self.part_n_hash_cols}")

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "AscendSSDEngramEmbedding produces rows via EngramSSDRuntime.prepare; "
            "the eager forward path must route through the model's prepare_engram")
