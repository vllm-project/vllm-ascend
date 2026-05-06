#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    UnquantizedEmbeddingMethod,
)

from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, AscendVocabParallelEmbedding
from vllm_ascend.utils import maybe_trans_nz


class AscendUnquantizedEmbeddingMethod310(UnquantizedEmbeddingMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_nz = maybe_trans_nz(layer.weight)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight_nz, bias)


class AscendVocabParallelEmbedding310(AscendVocabParallelEmbedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings, embedding_dim, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )
        if quant_config is None:
            self.quant_method = AscendUnquantizedEmbeddingMethod310()


class AscendParallelLMHead310(AscendParallelLMHead):
    """
    Register ParallelLMHead as a custom op for Atlas 310p.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        org_num_embeddings: int | None = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_embeddings, embedding_dim, bias, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )

        if quant_config is None:
            self.quant_method = AscendUnquantizedEmbeddingMethod310()
