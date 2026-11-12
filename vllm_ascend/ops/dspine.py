# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import math

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_reduce


def score_dspine_transitions(
    injection,
    norm,
    lm_head,
    codebook: torch.Tensor,
    candidate_ids: torch.Tensor,
    final_pre_injection: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    lm_head_nd: torch.Tensor,
    predecessor_weight: torch.Tensor,
    message_weight: torch.Tensor,
    write_weight: torch.Tensor,
) -> torch.Tensor:
    """Exact predecessor-conditioned refinement using NPU-dispatched ATen ops.

    ND vocabulary rows are gathered before casting to FP32; replicated bridge
    weights are cached in FP32 before graph capture. Flattened linear inputs
    and a 3D bmm make the matrix operations explicit without a generic einsum
    decomposition or a packed/NZ weight lookup in the scoring path.
    """
    num_reqs, num_steps, top_k = candidate_ids.shape
    hidden_size = final_pre_injection.shape[-1]
    final_layer = injection.num_layers - 1
    predecessors = torch.cat((anchor_token_ids[:, None, None].expand(-1, 1, top_k), candidate_ids[:, :-1]), dim=1)
    # The candidate TopK only returns original-vocabulary IDs; added/LoRA rows
    # are outside that search. Each standard TP shard contributes its own rows.
    shard = lm_head.shard_indices
    local = (candidate_ids >= shard.org_vocab_start_index) & (candidate_ids < shard.org_vocab_end_index)
    local_ids = torch.where(local, candidate_ids - shard.org_vocab_start_index, 0)
    successors = lm_head_nd.index_select(0, local_ids.reshape(-1)).float()
    successors = successors.reshape(num_reqs * num_steps, top_k, hidden_size)
    exact_features = codebook.index_select(0, predecessors.reshape(-1).long()).float()
    exact_features = exact_features * math.sqrt(injection.transfer_dim)

    # Preserve upstream's mixed precision: receiver projection/gate in the
    # model dtype, predecessor gate/message/write and residual refinement FP32.
    receiver = injection.project(final_layer, final_pre_injection.reshape(-1, hidden_size))
    receiver_gate = injection.receiver_gate(receiver).float().reshape(num_reqs, num_steps, 1, -1)
    predecessor_gate = F.linear(exact_features, predecessor_weight).reshape(num_reqs, num_steps, top_k, -1)
    gate = torch.sigmoid(receiver_gate + predecessor_gate + injection.gate_bias.float())
    message = F.linear(exact_features, message_weight).reshape_as(gate)
    update = F.linear((gate * message).reshape(-1, gate.shape[-1]), write_weight)
    update = update.reshape(num_reqs, num_steps, top_k, hidden_size)
    receiver_states = final_pre_injection[:, :, None, :]
    refined = receiver_states.float() + injection.rms(receiver_states) * update
    normalized = refined / (refined.square().mean(-1, keepdim=True) + norm.variance_epsilon).sqrt()
    normalized = normalized * norm.weight.float()
    scores = torch.bmm(
        normalized.reshape(num_reqs * num_steps, top_k, hidden_size),
        successors.transpose(1, 2).contiguous(),
    ).reshape(num_reqs, num_steps, top_k, top_k)
    scores.masked_fill_(~local.unsqueeze(-2), 0)
    if lm_head.tp_size > 1:
        scores = tensor_model_parallel_all_reduce(scores)
    return scores
