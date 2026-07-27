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


import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parameter import Parameter
from vllm.distributed import divide
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    pad_vocab_size,
)
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_embed_tp_group, get_lmhead_tp_group
from vllm_ascend.utils import embedding_tp_enable, get_potential_max_tokens, lmhead_tp_enable, vllm_version_is


class AscendVocabParallelEmbedding(VocabParallelEmbedding):
    """
    Register VocabParallelEmbedding as a custom op for Ascend.
    AscendVocabParallelEmbedding support different communication parallel groups
    Added the feature of lmheadTP in pure dp scenario
    """

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
        nn.Module.__init__(self)
        self.forward_type = None
        if lmhead_tp_enable() and "head" in prefix:
            self.comm_group = get_lmhead_tp_group()
        elif embedding_tp_enable() and "embed_tokens" in prefix:
            self.comm_group = get_embed_tp_group()
            self.forward_type = "embed_tp"
        else:
            self.comm_group = get_tp_group()

        self.tp_size = self.comm_group.world_size
        self.tp_rank = self.comm_group.rank_in_group

        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size, self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings, self.padding_size
        )
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

        self.shard_indices = self._get_indices(
            self.num_embeddings_padded,
            self.org_vocab_size_padded,
            self.num_embeddings,
            self.org_vocab_size,
            self.tp_rank,
            self.tp_size,
        )
        self.embedding_dim = embedding_dim
        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()

        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(type(quant_method))
        if is_embedding_layer and not quant_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod."
            )

        self.quant_method: QuantizeMethodBase = quant_method

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded, self.tp_size)
        assert self.shard_indices.num_elements_padded == self.num_embeddings_per_partition
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index - self.shard_indices.org_vocab_start_index
        )
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index - self.shard_indices.added_vocab_start_index
        )

        self.quant_method.create_weights(
            self,
            self.embedding_dim,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            self.num_embeddings_padded,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

    def _mask_input_for_vocab_range(
        self,
        input_: torch.Tensor,
        org_vocab_start_index: int,
        org_vocab_end_index: int,
        num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # torch.compile will fuse all of the pointwise ops below
        # into a single kernel, making it very fast
        org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
        # Adapt: avoid create added_vocab_mask when added_vocab_start_index == added_vocab_end_index.
        if added_vocab_start_index == added_vocab_end_index:
            valid_offset = org_vocab_start_index * org_vocab_mask
            vocab_mask = org_vocab_mask
        else:
            added_vocab_mask = (input_ >= added_vocab_start_index) & (input_ < added_vocab_end_index)
            added_offset = (
                added_vocab_start_index - (org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
            )
            valid_offset = (org_vocab_start_index * org_vocab_mask) + (added_offset * added_vocab_mask)
            vocab_mask = org_vocab_mask | added_vocab_mask
        # Adapt end.
        input_ = vocab_mask * (input_ - valid_offset)
        return input_, ~vocab_mask

    def forward(self, input_):
        if self.forward_type == "embed_tp":
            return self._forward_embed_tp(input_)
        else:
            return self._forward_origin(input_)

    def _forward_embed_tp(self, input_):
        num_tokens = input_.shape[0]

        # potential_max_tokens is computed once in the model runner __init__, so
        # reading it here is a cheap global lookup. Validate before allocating so
        # an oversized batch fails fast.
        capacity = get_potential_max_tokens()
        if num_tokens > capacity:
            raise ValueError(
                f"embedding_tp static capacity {capacity} < num_tokens "
                f"{num_tokens}; increase max_cudagraph_capture_size or "
                f"max_num_batched_tokens."
            )

        # Lazy init on first call (profiling run, which precedes ACL graph
        # capture). Static buffers keep a stable device address across all
        # later capture/replay cycles — graph replay requires the same
        # address that was recorded at capture (comm_group.all_gather and
        # reduce_scatter internally torch.empty() per call, which would
        # desync the HCCL operator recorded at capture).
        # Mirrors the OTP v13 fix in dsa_v1.py:_forward_o_proj.
        if not hasattr(self, "_embed_ag_in_buf"):
            device = input_.device
            # all_gather buffers carry token IDs (int64).
            self._embed_ag_in_buf = torch.zeros((capacity,), dtype=input_.dtype, device=device)
            self._embed_ag_out_buf = torch.empty((self.tp_size * capacity,), dtype=input_.dtype, device=device)
            # reduce_scatter buffers carry bf16 embeddings.
            self._embed_rs_in_buf = torch.empty(
                (self.tp_size * capacity, self.embedding_dim), dtype=self.params_dtype, device=device
            )
            self._embed_rs_out_buf = torch.empty((capacity, self.embedding_dim), dtype=self.params_dtype, device=device)

        # Pad input into the address-stable all_gather input buffer.
        self._embed_ag_in_buf.zero_()
        self._embed_ag_in_buf[:num_tokens].copy_(input_)
        dist.all_gather_into_tensor(self._embed_ag_out_buf, self._embed_ag_in_buf, group=self.comm_group.device_group)
        complete_input = self._embed_ag_out_buf

        # Masking unchanged; padding rows map to OOB and get masked to 0
        # via masked_fill_ below (token_id=0 stays in-range after shift).
        masked_input, input_mask = self._mask_input_for_vocab_range(
            complete_input,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            self.shard_indices.num_org_vocab_padding,
            self.shard_indices.added_vocab_start_index,
            self.shard_indices.added_vocab_end_index,
        )
        # Embedding lookup is a local op (F.embedding); its fresh allocation
        # does not affect ACL graph replay. Copy into the static rs_in
        # buffer so reduce_scatter reads from a stable address.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        self._embed_rs_in_buf.copy_(output_parallel)
        self._embed_rs_in_buf.masked_fill_(input_mask.unsqueeze(-1), 0)
        dist.reduce_scatter_tensor(self._embed_rs_out_buf, self._embed_rs_in_buf, group=self.comm_group.device_group)

        # Strip padding rows; preserve the original return shape.
        return self._embed_rs_out_buf[:num_tokens].view(num_tokens, -1)

    def _forward_origin(self, input_):
        if self.tp_size > 1:
            # Build the mask.
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
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = torch.ops.vllm.maybe_pad_and_reduce(output_parallel)
        return output


class AscendParallelLMHead(ParallelLMHead):
    """
    Register ParallelLMHead as a custom op for Ascend."""

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


class AscendLogitsProcessor(LogitsProcessor):
    """
    Register LogitsProcessor as a custom op for Ascend.
    Added the feature of lmheadTP in pure dp scenario
    """

    def _apply_head(
        self,
        lm_head: AscendParallelLMHead,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if vllm_version_is("0.25.1"):
            return lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        return super()._apply_head(lm_head, hidden_states, embedding_bias)

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if lmhead_tp_enable():
            return self._get_logits_lmheadtp(hidden_states, lm_head, embedding_bias)
        else:
            return self._get_logits_normal(hidden_states, lm_head, embedding_bias)

    def _get_logits_lmheadtp(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Gather hidden states from all devices in tensor parallel group
        gathered_hidden_states = get_lmhead_tp_group().all_gather(hidden_states, dim=0)
        logits = self._apply_head(lm_head, gathered_hidden_states, embedding_bias)
        # Gather logits for tensor parallel
        if not get_ascend_config().enable_reduce_sample:
            logits = get_lmhead_tp_group().all_to_all(logits)

        # Remove paddings in vocab (if any)
        if logits is not None:
            if not get_ascend_config().enable_reduce_sample:
                logits = logits[..., : self.org_vocab_size]
            else:
                logits = logits[..., : lm_head.num_org_embeddings_per_partition]
        return logits

    def _get_logits_normal(
        self,
        hidden_states: torch.Tensor,
        lm_head: AscendParallelLMHead,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        # Gather logits for tensor parallel
        if not get_ascend_config().enable_reduce_sample:
            logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any)
        if logits is not None:
            if not get_ascend_config().enable_reduce_sample:
                logits = logits[..., : self.org_vocab_size]
            else:
                logits = logits[..., : lm_head.num_org_embeddings_per_partition]

        return logits
