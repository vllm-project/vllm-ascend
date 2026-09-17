# SPDX-License-Identifier: Apache-2.0
"""Temporary v0.27.1 Qwen gate control for acceptance-rate experiments.

Restore only gate placement/policy, not the full v0.27.1 runtime. Remove this
patch after the diagnostic; it is not a production precision policy.
"""

from functools import wraps

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import sequence_parallel_chunk

logger = init_logger(__name__)


def _with_v0271_gate_policy(original_init):
    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Construction precedes weight loading: undo the newer blanket FP32
        # pre-cast before process_weights_after_loading can create weight_fp32.
        self.gate.precast_fp32_weight = False
        self.gate.ascend_v0271_gate_control = True
        logger.info_once(
            "QWEN_V0271_GATE_CONTROL: restored external native gate; blanket FP32 weight pre-cast disabled."
        )

    return init


def qwen3_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    assert hidden_states.dim() <= 2, "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
    is_input_1d = hidden_states.dim() == 1
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    if self.experts.is_internal_router:
        # In this case, the gate/router runs inside the MoERunner class
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        # v0.27.1 Ascend takes this path when weight_fp32 is absent.
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]

    # return to 1d if input is 1d
    return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


def qwen3_next_forward(
    self,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
) -> torch.Tensor:
    # NOTE: hidden_states can have either 1D or 2D shape.
    orig_shape = hidden_states.shape
    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)

    if self.is_sequence_parallel and not already_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)

    # Pinned main adds replicated shared experts; retain that non-gate behavior.
    # v0.28.0 does not have replicate_shared_expert.
    replicated_shared_output = (
        self.shared_expert(hidden_states)
        if getattr(self, "replicate_shared_expert", False) and self.shared_expert is not None
        else None
    )
    if self.experts.is_internal_router:
        # In this case, the gate/router runs inside the MoERunner class
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

    if replicated_shared_output is not None:
        final_hidden_states += replicated_shared_output

    if self.is_sequence_parallel and not already_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
        final_hidden_states = final_hidden_states[:num_tokens]

    return final_hidden_states.view(orig_shape)


# Qwen3.5/3.6 reuse Qwen3NextSparseMoeBlock, including the DSpark target.
Qwen3MoeSparseMoeBlock.__init__ = _with_v0271_gate_policy(Qwen3MoeSparseMoeBlock.__init__)
Qwen3MoeSparseMoeBlock.forward = qwen3_moe_forward
Qwen3NextSparseMoeBlock.__init__ = _with_v0271_gate_policy(Qwen3NextSparseMoeBlock.__init__)
Qwen3NextSparseMoeBlock.forward = qwen3_next_forward
