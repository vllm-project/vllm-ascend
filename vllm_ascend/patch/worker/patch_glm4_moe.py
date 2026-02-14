import torch
from vllm.model_executor.models.glm4_moe import Glm4MoE

from vllm_ascend.ops.triton.muls_add import muls_add_triton


def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    router_logits = self.gate(hidden_states.to(dtype=torch.float32))

    fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if self.shared_experts is not None:
        shared_output, final_hidden_states = fused_moe_out
        assert shared_output is not None
        final_hidden_states = muls_add_triton(
            x=final_hidden_states,
            y=shared_output,
            scale=float(self.routed_scaling_factor),
        )
    else:
        final_hidden_states = fused_moe_out * self.routed_scaling_factor

    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)
    return final_hidden_states.view(num_tokens, hidden_dim)


Glm4MoE.forward = forward
