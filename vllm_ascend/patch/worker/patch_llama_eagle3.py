import numpy as np
import torch
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.distributed.parallel_state import get_tp_group

def compute_logits(
    self,
    hidden_states: torch.Tensor,
    enable_reduce_sample: bool = True,
) -> torch.Tensor | None:
    if enable_reduce_sample:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits
        logits = logits.contiguous()
        next_token = greedy_sample(logits)
        bias = torch.index_select(self.draft_id_to_target_id, dim=0, index=next_token.view(-1)).view(next_token.shape)
        return next_token + bias
    else:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    tp_group = get_tp_group()
    B, V_local = logits.shape
    rank = tp_group.rank_in_group
    world_size = tp_group.world_size

    local_max_logits, local_max_indices = logits.max(dim=-1)

    local_global_idx = local_max_indices + rank * V_local  # [B]

    # [B, world_size]
    gathered_logits = tp_group.all_gather(local_max_logits.unsqueeze(-1), dim=-1)
    gathered_global_idx = tp_group.all_gather(local_global_idx.unsqueeze(-1), dim=-1) # [B, world_size]
    global_max_rank = gathered_logits.argmax(dim=-1)  # [B]
    target_argmax = gathered_global_idx.gather(
        dim=-1,
        index=global_max_rank.unsqueeze(-1)
    ).squeeze(-1) # [B]
    return target_argmax

Eagle3LlamaForCausalLM.compute_logits = compute_logits