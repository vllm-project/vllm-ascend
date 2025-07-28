from vllm.model_executor.layers.logits_processor import \
    LogitsProcessor
import torch
from typing import Optional
from vllm_ascend.distributed.parallel_state import get_lm_tp_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.forward_context import ForwardContext, get_forward_context

class CustomLogitsProcessor(LogitsProcessor):
    
    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__(vocab_size, org_vocab_size, scale, logits_as_input, soft_cap)

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        num_reqs_across_dp: torch.Tensor,
        num_tokens_across_dp: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # num_reqs_across_dp2 = get_forward_context().dp_metadata.num_reqs_across_dp
        # print("#################num_reqs_across_dp2",num_reqs_across_dp2)
        if self.logits_as_input:
            logits = hidden_states
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(hidden_states,
                                                        sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, num_reqs_across_dp, num_tokens_across_dp, embedding_bias)
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
        lm_head: VocabParallelEmbedding,
        num_reqs_across_dp: torch.Tensor,
        num_tokens_across_dp: torch.Tensor,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        import numpy
        import numpy as np
        # # Gather logits for TP
        lm_tp_size = lm_head.lm_tp_size
        local_batch_size = hidden_states.size(0)
        local_output_dim = lm_head.num_embeddings_per_partition
        flag=0
        if num_reqs_across_dp[lm_head.lm_tp_rank] == 0:
            flag=1
        lm_total_batchsize = sum(num_reqs_across_dp)
        if lm_total_batchsize == 0:
            return None
        gathered_input = [torch.empty(batch_size, hidden_states.size(
            1), dtype=hidden_states.dtype, device=hidden_states.device) for batch_size in num_reqs_across_dp]
        torch.distributed.all_gather(
                gathered_input, hidden_states, group=get_lm_tp_group().device_group)
        complete_input = torch.cat(gathered_input, dim=0)
        
        # Compute logits using quantized matrix multiplication
        logits = lm_head.quant_method.apply(lm_head,
                                            complete_input,
                                            bias=embedding_bias)
           
        # # Prepare for all-to-all communication to redistribute logits
        output_splits = []
        if flag==0:
            for x in num_tokens_across_dp:
                output_splits.append(local_batch_size*local_output_dim)
            all_to_all_result = torch.empty(local_batch_size * lm_tp_size * local_output_dim,
                                            dtype=logits.dtype, device=logits.device)
        else:
            for x in num_tokens_across_dp:
                output_splits.append(0)
            all_to_all_result = torch.tensor([], dtype=logits.dtype, device=logits.device)
        input_splits = torch.Tensor(num_reqs_across_dp).int().tolist()
        # # # Perform all-to-all communication to get correct logit partitions
        torch.distributed.all_to_all_single(
                            all_to_all_result,
                            logits,
                            output_splits,
                            input_splits,
                            group=get_lm_tp_group().device_group)
        logits = all_to_all_result.view(local_batch_size, -1)
     

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits
