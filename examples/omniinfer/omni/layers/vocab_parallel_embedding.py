# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from torch.nn.parameter import Parameter
from typing import Optional, Tuple
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, 
    pad_vocab_size, 
    VocabParallelEmbedding as VocabParallelEmbeddingGPU
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase, method_has_implemented_embedding)

from vllm.model_executor.utils import set_weight_attrs

from vllm.distributed import (divide,
                              tensor_model_parallel_all_gather,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from omni.adaptors.vllm.distributed.parallel_state import get_local_world_group, get_world_group
from omni.adaptors.vllm.distributed.communication_op import all_gather_local, reduce_scatter_local
from omni.models.common.config.model_config import model_extra_config

DEFAULT_VOCAB_PADDING_SIZE = 64
 

def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.jit.script will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ <
                                                          org_vocab_end_index)
    # Adapt: avoid create added_vocab_mask when added_vocab_start_index == added_vocab_end_index.
    if added_vocab_start_index == added_vocab_end_index:
        valid_offset = (org_vocab_start_index *
                        org_vocab_mask)
        vocab_mask = org_vocab_mask
    else:
        added_vocab_mask = (input_ >= added_vocab_start_index) & (
            input_ < added_vocab_end_index)
        added_offset = added_vocab_start_index - (
            org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
        valid_offset = (org_vocab_start_index *
                        org_vocab_mask) + (added_offset * added_vocab_mask)
        vocab_mask = org_vocab_mask | added_vocab_mask
    # Adapt end.
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask
 
 
class VocabParallelEmbedding(VocabParallelEmbeddingGPU):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 parallel_lmhead: bool = False):
 
        torch.nn.Module.__init__(self)
 
        # Keep the input dimensions.
        # adapt: lm_head use local tp
        self.parallel_lmhead = parallel_lmhead
        if parallel_lmhead:
            tp_rank = get_world_group().local_rank
            self.tp_size = get_local_world_group().world_size
        else:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
 
        # adapt end.
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size,
                                                    self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings,
            self.padding_size)
        if self.org_vocab_size_padded > self.num_embeddings_padded:
            raise RuntimeError("self.org_vocab_size_padded > self.num_embeddings_padded")
 
        self.shard_indices = self._get_indices(self.num_embeddings_padded,
                                               self.org_vocab_size_padded,
                                               self.num_embeddings,
                                               self.org_vocab_size, tp_rank,
                                               self.tp_size)
        self.embedding_dim = embedding_dim
 
        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self, prefix=prefix)
        if quant_method is None:
            quant_method = UnquantizedEmbeddingMethod()
 
        # If we are making an embedding layer, then our quantization linear
        # method must implement the embedding operation. If we are another
        # layer type like ParallelLMHead, this is not important.
        is_embedding_layer = type(self.__class__) is VocabParallelEmbedding
        linear_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method))
        if is_embedding_layer and not linear_method_implements_embedding:
            raise NotImplementedError(
                f"The class {type(quant_method).__name__} must implement "
                "the 'embedding' method, see UnquantizedEmbeddingMethod.")
 
        self.quant_method: QuantizeMethodBase = quant_method
 
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded,
                                                   self.tp_size)
        if self.shard_indices.num_elements_padded != self.num_embeddings_per_partition:
            raise RuntimeError("self.shard_indices.num_elements_padded != self.num_embeddings_per_partition")
        self.num_org_embeddings_per_partition = (
            self.shard_indices.org_vocab_end_index -
            self.shard_indices.org_vocab_start_index)
        self.num_added_embeddings_per_partition = (
            self.shard_indices.added_vocab_end_index -
            self.shard_indices.added_vocab_start_index)
 
        self.quant_method.create_weights(self,
                                         self.embedding_dim,
                                         [self.num_embeddings_per_partition],
                                         self.embedding_dim,
                                         self.num_embeddings_padded,
                                         params_dtype=params_dtype,
                                         weight_loader=self.weight_loader)
 
    def forward_vocab(self, input_, reduce = 0):
        if self.tp_size > 1:
            # Build the mask.
            if self.parallel_lmhead:
                input_ = all_gather_local(input_, idx=0, dim=0)

            masked_input, input_mask = get_masked_input_and_mask(
                input_, self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index)
        else:
            masked_input = input_
 
        if masked_input.dtype != torch.long:
            masked_input = masked_input.long()
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self, masked_input)
        # Mask the output embedding.
        if self.tp_size > 1:
            # adapter for faster
            output_parallel *= ~input_mask.unsqueeze(-1)
        if reduce == 0:
            output = tensor_model_parallel_all_reduce(output_parallel)
        elif self.parallel_lmhead:
            output = reduce_scatter_local(output_parallel, idx=0)
        else:
            # Reduce across all the model parallel GPUs.
            output = tensor_model_parallel_reduce_scatter(output_parallel)
        return output
 
class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.
 
    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.
 
    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """
 
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 bias: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 parallel_lmhead: bool = True):
        super().__init__(num_embeddings, embedding_dim, params_dtype,
                         org_num_embeddings, padding_size, quant_config,
                         prefix, parallel_lmhead)
        self.quant_config = quant_config
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            })
        else:
            self.register_parameter("bias", None)

    def tie_weights(self, embed_tokens: VocabParallelEmbedding):
        """Tie the weights with word embeddings."""
        # GGUF quantized embed_tokens.
        if self.quant_config and self.quant_config.get_name() == "gguf":
            return embed_tokens
        else:
            self.weight = embed_tokens.weight
            return self

    def forward(self, hidden_states, embedding_bias):
        if model_extra_config.parall_config.dp_size > 1:
            hidden_states = get_local_world_group().all_gather(hidden_states, dim=0)

        logits = self.quant_method.apply(self,
                                         hidden_states,
                                         bias=embedding_bias)

        if model_extra_config.parall_config.dp_size > 1:
            logits = get_local_world_group().all_to_all(logits)
        else:
            logits = tensor_model_parallel_all_gather(logits)

        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits