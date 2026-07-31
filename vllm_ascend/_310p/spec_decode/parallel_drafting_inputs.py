#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import torch

# The only kernel/allocation block size this scope supports. Owned by the
# attention adapter because it is the kernel's constraint; imported rather than
# redeclared so the two cannot drift. Everything that claims to be "the block
# size" is checked against it rather than against each other: cross-checking
# three sources only proves they agree, not that they equal the value the custom
# FIA kernel was validated at.
from vllm_ascend._310p.attention.parallel_draft_attention import FIA_BLOCK_SIZE


def resolve_310p_block_size(proposer) -> int:
    """Read the selected kernel block size and pin it to FIA_BLOCK_SIZE.

    310P only. Never call this on the Triton path: that path must keep using its
    own source (proposer.kernel_block_size for DFlash, the group's
    kv_cache_spec.block_size for DSpark) so A2/A3 behaviour is unchanged.

    Runs once per draft step rather than per layer, so the handful of attribute
    reads here are not worth caching.
    """
    gid = proposer.kv_cache_gid
    selected = proposer.runner.input_batch.block_table[gid].block_size
    if selected != FIA_BLOCK_SIZE:
        raise RuntimeError(
            f"this scope only covers kernel block size {FIA_BLOCK_SIZE}, but KV cache "
            f"group {gid} selected {selected}; re-scope before changing anything"
        )
    if proposer.kernel_block_size != FIA_BLOCK_SIZE:
        raise RuntimeError(
            f"proposer.kernel_block_size is {proposer.kernel_block_size}, expected "
            f"{FIA_BLOCK_SIZE}; the drafter and the block table disagree about the "
            f"kernel block size, which would corrupt slot mapping"
        )
    # runner.kernel_block_sizes[gid] is a *candidate list* (typically [128, 64]),
    # never a scalar, so check membership rather than equality.
    candidates = proposer.runner.kernel_block_sizes[gid]
    if FIA_BLOCK_SIZE not in candidates:
        raise RuntimeError(
            f"{FIA_BLOCK_SIZE} is not in the runner's candidate list {candidates} for "
            f"KV cache group {gid}"
        )
    return FIA_BLOCK_SIZE


def expand_parallel_drafting_inputs(
    *,
    next_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    out_input_ids: torch.Tensor,
    out_context_positions: torch.Tensor,
    out_query_positions: torch.Tensor,
    out_context_slot_mapping: torch.Tensor,
    out_query_slot_mapping: torch.Tensor,
    out_token_indices: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_rejected_tokens: torch.Tensor | None,
    parallel_drafting_token_id: int,
    block_size: int,
    num_query_per_req: int,
    num_speculative_tokens: int,
    total_input_tokens: int,
    batch_size: int,
    sample_from_anchor: bool = False,
) -> None:
    """Triton-free DFlash/DSpark input expansion for 310P.

    Field-for-field equivalent of ``copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid``
    in ``vllm_ascend/ops/triton/spec_decode/utils.py``. Writes in place into the
    caller's persistent buffers and returns nothing.

    ``sample_from_anchor`` selects DSpark's sampling layout (take ``num_speculative_tokens``
    positions starting at the anchor) instead of DFlash's (skip the anchor, take the
    next ``num_speculative_tokens``).

    No ``.cpu()`` / ``.item()`` / ``.tolist()`` anywhere: every value stays on device
    so this adds no synchronization to the draft step.
    """
    device = target_positions.device
    b, q, k = batch_size, num_query_per_req, num_speculative_tokens

    # The scheduled context is copied verbatim over the whole ragged range. The
    # rejected tail deliberately stays in place: it is excluded downstream by the
    # shortened seq_lens, and the new query slots overwrite those cache positions.
    # Assumes query_start_loc[batch_size] == total_input_tokens, which every caller
    # guarantees; checking it here would cost a D2H on the draft hot path.
    out_context_positions[:total_input_tokens] = target_positions[:total_input_tokens]
    out_context_slot_mapping[:total_input_tokens] = context_slot_mapping[:total_input_tokens]

    ctx_end = query_start_loc[1 : b + 1].to(torch.int64)
    if num_rejected_tokens is None:
        num_rejected = torch.zeros(b, dtype=torch.int64, device=device)
    else:
        num_rejected = num_rejected_tokens[:b].to(torch.int64)

    # Absolute position of the last still-valid context token per request.
    last_pos = target_positions[ctx_end - num_rejected - 1].to(torch.int64)
    # Total KV length after dropping the rejected tail: where this round's query
    # block starts in the cache. Derived from seq_lens (total KV), not from the
    # scheduled segment length.
    effective_seq_len = seq_lens[:b].to(torch.int64) - num_rejected

    q_arange = torch.arange(q, dtype=torch.int64, device=device)
    out_query_positions[: b * q] = (last_pos[:, None] + 1 + q_arange).flatten()

    # int64 throughout so `block_id * block_size` cannot overflow before it is
    # narrowed back into the int32 destination buffer.
    cache_pos = effective_seq_len[:, None] + q_arange
    block_id = block_table[:b].gather(1, cache_pos // block_size).to(torch.int64)
    out_query_slot_mapping[: b * q] = (block_id * block_size + cache_pos % block_size).flatten()

    ids = out_input_ids[: b * q].view(b, q)
    ids.fill_(parallel_drafting_token_id)
    ids[:, 0] = next_token_ids[:b]

    anchor_offset = 0 if sample_from_anchor else 1
    req_base = torch.arange(b, dtype=torch.int64, device=device)[:, None] * q
    k_arange = torch.arange(k, dtype=torch.int64, device=device)
    out_token_indices[: b * k] = (req_base + k_arange + anchor_offset).flatten()
