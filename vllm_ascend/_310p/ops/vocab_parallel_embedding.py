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
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE

from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead
from vllm_ascend.utils import maybe_trans_nz


class _AscendParallelLMHead310QuantMethod(QuantizeMethodBase):
    def __init__(self, base_method: QuantizeMethodBase):
        self._base_method = base_method

    def __getattr__(self, name: str):
        return getattr(self._base_method, name)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self._base_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._base_method.apply(layer, x, bias=bias)

    def embedding(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._base_method.embedding(layer, x)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._base_method.process_weights_after_loading(layer)
        layer.weight.data = maybe_trans_nz(layer.weight.data)


class AscendParallelLMHead310(AscendParallelLMHead):
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
            num_embeddings,
            embedding_dim,
            bias,
            params_dtype,
            org_num_embeddings,
            padding_size,
            quant_config,
            prefix,
        )
        self.quant_method = _AscendParallelLMHead310QuantMethod(self.quant_method)
