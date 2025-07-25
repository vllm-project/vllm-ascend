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
        cu_tokens_across_dp_cpu: torch.Tensor,
        num_tokens_across_dp: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        # cu_tokens_across_dp_cpu2 = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
        # print("#################cu_tokens_across_dp_cpu2",cu_tokens_across_dp_cpu2)
        if self.logits_as_input:
            logits = hidden_states
        else:
            if sampling_metadata is not None:
                hidden_states = _prune_hidden_states(hidden_states,
                                                        sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, cu_tokens_across_dp_cpu, num_tokens_across_dp, embedding_bias)
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
        cu_tokens_across_dp_cpu: torch.Tensor,
        num_tokens_across_dp: torch.Tensor,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        import numpy
        import numpy as np
        # # Gather logits for TP
        # logits = self._gather_logits(logits)
        # cu_tokens_across_dp_cpu = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
        local_batch_size = hidden_states.size(0)
        local_output_dim = lm_head.num_embeddings_per_partition
        lm_total_batchsize = sum(cu_tokens_across_dp_cpu.numpy())
        
        # gathered_input = []
        # for x in range(len(num_tokens_across_dp)):
        #     if num_tokens_across_dp[x] != 1:
        #         gathered_input.append(torch.empty(cu_tokens_across_dp_cpu[x], hidden_states.size(1), dtype=hidden_states.dtype, device=hidden_states.device))
        #     else:
        #         gathered_input.append(torch.tensor(1, dtype=hidden_states.dtype, device=hidden_states.device))
        gathered_input = [torch.empty(batch_size, hidden_states.size(
            1), dtype=hidden_states.dtype, device=hidden_states.device) for batch_size in cu_tokens_across_dp_cpu]
        torch.distributed.all_gather(
            gathered_input, hidden_states, group=get_lm_tp_group().device_group)
        complete_input = torch.cat(gathered_input, dim=0)
        
        # Compute logits using quantized matrix multiplication
        logits = lm_head.quant_method.apply(lm_head,
                                            complete_input,
                                            bias=embedding_bias)
        # # Prepare for all-to-all communication to redistribute logits
        input_splits = cu_tokens_across_dp_cpu.int().tolist()
        output_splits = []
        for x in num_tokens_across_dp:
            if x != 1:
                # 如果不等于 1，复制两次
                output_splits.append(local_batch_size * local_output_dim)
            else:
                # 如果等于 1，保留一次
                output_splits.append(0)
        output_splits = torch.Tensor(output_splits).int().tolist()
        all_to_all_result = torch.empty(lm_total_batchsize * local_output_dim,
                                        dtype=logits.dtype, device=logits.device)
        # # # Perform all-to-all communication to get correct logit partitions
        torch.distributed.all_to_all_single(
                            all_to_all_result,
                            logits,
                            output_splits,
                            input_splits,
                            group=get_lm_tp_group().device_group)
        logits = all_to_all_result.view(lm_total_batchsize, local_output_dim)
        

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits
