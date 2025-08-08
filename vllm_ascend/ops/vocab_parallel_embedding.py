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

from typing import Optional, Tuple

import torch
from torch.nn import Module
import torch.distributed as dist
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    DEFAULT_VOCAB_PADDING_SIZE,
    pad_vocab_size,
    UnquantizedEmbeddingMethod,
    ParallelLMHead
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor,
    _apply_logits_processors,
    _prune_hidden_states
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding
)

from vllm_ascend.distributed.parallel_state import get_lmheadtp_group
from vllm_ascend.ascend_config import get_ascend_config


class AscendVocabParallelEmbedding(VocabParallelEmbedding):

    def _get_masked_input_and_mask(
            self, input_: torch.Tensor, org_vocab_start_index: int,
            org_vocab_end_index: int, num_org_vocab_padding: int,
            added_vocab_start_index: int,
            added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # torch.compile will fuse all of the pointwise ops below
        # into a single kernel, making it very fast
        org_vocab_mask = (input_ >= org_vocab_start_index) & (
            input_ < org_vocab_end_index)
        # Adapt: avoid create added_vocab_mask when added_vocab_start_index == added_vocab_end_index.
        if added_vocab_start_index == added_vocab_end_index:
            valid_offset = (org_vocab_start_index * org_vocab_mask)
            vocab_mask = org_vocab_mask
        else:
            added_vocab_mask = (input_ >= added_vocab_start_index) & (
                input_ < added_vocab_end_index)
            added_offset = added_vocab_start_index - (
                org_vocab_end_index -
                org_vocab_start_index) - num_org_vocab_padding
            valid_offset = (org_vocab_start_index *
                            org_vocab_mask) + (added_offset * added_vocab_mask)
            vocab_mask = org_vocab_mask | added_vocab_mask
        # Adapt end.
        input_ = vocab_mask * (input_ - valid_offset)
        return input_, ~vocab_mask

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = self._get_masked_input_and_mask(
                input_, self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index)
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = self.quant_method.embedding(self,
                                                      masked_input.long())
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


class CustomParallelLMHead(ParallelLMHead):
    
    """Costom Parallelized LM head, added the feature of lmheadTP in pure dp scenario
    
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
                 prefix: str = ""): 
        Module.__init__(self)
        self._enable_lmhead_tp = False
        if get_ascend_config().lmhead_tensor_parallel_size is not None:
            tp_rank = get_lmheadtp_group().rank_in_group
            self.tp_size = get_lmheadtp_group().world_size
            self._enable_lmhead_tp = True
        else:
            tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
        
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        num_added_embeddings = num_embeddings - self.org_vocab_size
        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size,
                                                    self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(
            self.org_vocab_size_padded + num_added_embeddings,
            self.padding_size)
        assert self.org_vocab_size_padded <= self.num_embeddings_padded

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
        is_embedding_layer = type(self) is VocabParallelEmbedding
        quant_method_implements_embedding = method_has_implemented_embedding(
            type(quant_method))
        if is_embedding_layer and not quant_method_implements_embedding:
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
        assert (self.shard_indices.num_elements_padded ==
                self.num_embeddings_per_partition)
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
        
class CustomLogitsProcessor(LogitsProcessor):
    """Custom logits processor extending base LogitsProcessor functionality.
    Added the feature of lmheadTP in pure dp scenario
    """
    
    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:
        super().__init__(
            vocab_size=vocab_size,
            org_vocab_size=org_vocab_size,
            scale=scale,
            logits_as_input=logits_as_input,
            soft_cap=soft_cap
        )

    def forward(
        self,
        lm_head: CustomParallelLMHead,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.logits_as_input:
            logits = hidden_states
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(hidden_states,
                                                     sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            if sampling_metadata is not None and \
                sampling_metadata.seq_groups is not None:
                logits = _apply_logits_processors(logits, sampling_metadata)
        
        return logits

    def _get_logits(
            self,
            hidden_states: torch.Tensor,
            lm_head: CustomParallelLMHead,
            embedding_bias: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
        """
        Compute logits for next token prediction using parallel processing.
        
        Args:
            hidden_states: Current hidden states from the model with shape [batch_size, hidden_size]
            lm_head: Parallel embedding layer for vocabulary predictions
            embedding_bias: Optional bias tensor to add to logits with shape [vocab_size]
            
        Returns:
            Logits tensor for next token prediction with shape [batch_size, vocab_size] or None
        """

        if lm_head._enable_lmhead_tp:
            # _enable_lmhead_tp used in the graph mode，and batch_size is the same across different dp.
            lmhead_tp_size = lm_head.tp_size
            local_batch_size = hidden_states.size(0)
            vocab_size_per_partition = lm_head.num_embeddings_per_partition

            # Gather hidden states from all devices in tensor parallel group
            gathered_hidden_states = get_lmheadtp_group().all_gather(hidden_states, dim=0)
        else:
            gathered_hidden_states = hidden_states

        # Compute logits using quantized matrix multiplication
        local_logits = lm_head.quant_method.apply(
            lm_head,
            gathered_hidden_states,
            bias=embedding_bias
        )

        if lm_head._enable_lmhead_tp:
            # Prepare for all-to-all communication to redistribute logits
            input_split_sizes = [local_batch_size] * get_lmheadtp_group().world_size
            output_split_sizes = [local_batch_size * vocab_size_per_partition] * lmhead_tp_size

            all_to_all_output = torch.empty(
                local_batch_size * (vocab_size_per_partition * lmhead_tp_size),
                dtype=local_logits.dtype, 
                device='npu'
            )
                
            # Perform all-to-all communication to get correct logit partitions
            dist.all_to_all_single(
                all_to_all_output,
                local_logits,
                output_split_sizes,
                input_split_sizes,
                group=get_lmheadtp_group().device_group
            )
            
            # Reshape and combine logits from all partitions
            reshaped_logits = all_to_all_output.view(
                lmhead_tp_size, local_batch_size, vocab_size_per_partition
            )
            combined_logits = reshaped_logits.permute(1, 0, 2).reshape(local_batch_size, -1)
        else:
            # Gather logits for tensor parallel
            combined_logits = self._gather_logits(local_logits)

        # Remove paddings in vocab (if any)
        if combined_logits is not None:
            combined_logits = combined_logits[..., :self.org_vocab_size]
            
        return combined_logits
