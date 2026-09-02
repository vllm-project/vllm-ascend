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
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    UnquantizedEmbeddingMethod,
)

from vllm_ascend._310p.dflash_full_and_piecewise import (
    is_310p_dflash_full_and_piecewise,
)
from vllm_ascend.ops.vocab_parallel_embedding import AscendParallelLMHead, AscendVocabParallelEmbedding
from vllm_ascend.utils import maybe_trans_nz


def _uses_private_draft_embedding_310(forward_context) -> bool:
    """Keep Hybrid Draft compile/capture/replay on one lookup branch.

    The Draft backbone is first AOT-compiled while its temporary runtime mode
    is NONE and that compiled graph is later captured/replayed as PIECEWISE.
    Routing by the effective runtime mode would therefore bake GatherV2 into
    the graph before PIECEWISE capture. Component identity plus the configured
    Hybrid scope is the stable capture-time contract.
    """
    return (
        bool(getattr(forward_context, "is_draft_model", False))
        and is_310p_dflash_full_and_piecewise(
            getattr(forward_context, "vllm_config", None),
        )
    )


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

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if is_forward_context_available():
            forward_context = get_forward_context()
            if _uses_private_draft_embedding_310(forward_context):
                return self.embedding_gather_elements_310(input_)
        return super().forward(input_)

    def embedding_gather_elements_310(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Exact embedding lookup via GatherElements for a private 310P route.

        The normal ``forward`` remains unchanged. This sibling entry preserves
        the existing vocabulary masking and TP reduction, replacing only the
        failing GatherV2 lookup used by 310P DFlash Hybrid Draft PIECEWISE.
        """
        if getattr(self, "forward_type", None) == "embed_tp":
            raise NotImplementedError(
                "310P DFlash Hybrid private embedding does not support "
                "embedding TP"
            )
        if not isinstance(
            self.quant_method,
            AscendUnquantizedEmbeddingMethod310,
        ):
            raise NotImplementedError(
                "310P DFlash Hybrid private embedding requires an "
                "unquantized embedding"
            )

        if self.tp_size > 1:
            masked_input, input_mask = self._mask_input_for_vocab_range(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
        else:
            masked_input = input_

        flat_indices = masked_input.long().reshape(-1)
        gather_indices = flat_indices.unsqueeze(-1).expand(
            -1,
            self.embedding_dim,
        )
        output_parallel = torch.gather(
            self.weight,
            dim=0,
            index=gather_indices,
        ).view(*masked_input.shape, self.embedding_dim)

        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        return torch.ops.vllm.maybe_pad_and_reduce(output_parallel)


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
