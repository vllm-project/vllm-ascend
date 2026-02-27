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
import types

import torch
import torch_npu
from torch import nn
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, AscendVocabParallelEmbedding
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


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
        AscendVocabParallelEmbedding.__init__(
            self, num_embeddings, embedding_dim, params_dtype, org_num_embeddings, padding_size, quant_config, prefix
        )

        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(torch.empty(self.num_embeddings_per_partition, dtype=params_dtype))
            set_weight_attrs(
                self.bias,
                {
                    "output_dim": 0,
                    "weight_loader": self.weight_loader,
                },
            )
        else:
            self.register_parameter("bias", None)

        def patched_process_weights_after_loading(self, layer: nn.Module) -> None:
            layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, ACL_FORMAT_FRACTAL_NZ)

        self.quant_method.process_weights_after_loading = types.MethodType(
            patched_process_weights_after_loading, self.quant_method
        )
