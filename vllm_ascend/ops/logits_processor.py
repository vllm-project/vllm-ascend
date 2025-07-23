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
        cu_tokens_across_dp_cpu: int,
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
            logits = self._get_logits(hidden_states, lm_head, cu_tokens_across_dp_cpu, embedding_bias)
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
        cu_tokens_across_dp_cpu: int,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # # # Get the logits for the next tokens.
        # logits = lm_head.quant_method.apply(lm_head,
        #                                     hidden_states,
        #                                     bias=embedding_bias)

        # # Gather logits for TP
        # logits = self._gather_logits(logits)
        # cu_tokens_across_dp_cpu = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
        
        
        #adapt: 8DP to 8TP
        hidden_states = get_lm_tp_group().all_gather(hidden_states, dim=0)
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head,
                                                hidden_states,
                                                bias=embedding_bias)
        
        world_size = get_lm_tp_group().world_size
        vocab_size = logits.shape[-1] * world_size
        logits = logits.view(-1)

        output_ = torch.zeros_like(logits).view(-1)
        torch.distributed.all_to_all_single(output_,
                                            logits,
                                            group=get_lm_tp_group().device_group)
        output_ = output_.view(world_size, -1,  vocab_size // world_size).transpose(0, 1)
        logits = output_.reshape(-1, vocab_size)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits
