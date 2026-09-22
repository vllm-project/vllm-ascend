# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Uniform Engram DP exchange, backported from vLLM f84b0c4bce.

For single-node PP=PCP=DCP=1, the existing DP group has exactly the
membership of upstream's node-local Engram DP group. Reuse it without
creating another communicator or modifying vLLM parallel state.
"""

import torch
from vllm.distributed import get_dp_group, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4_1.common.engram import DEAD_ID
from vllm.triton_utils import tl, triton


def get_engram_dp_group():
    group = get_dp_group()
    return group if group.world_size > 1 else None


def get_engram_dp_size():
    group = get_engram_dp_group()
    return group.world_size if group is not None else 1


def engram_head_shard_rank() -> int:
    """This rank's slot among the hash-head shards of one engram table.

    TP-major, so the shards a DP gather brings in are contiguous heads and
    the following TP gather completes the head order.
    """
    dp_group = get_engram_dp_group()
    dp_size = dp_group.world_size if dp_group is not None else 1
    dp_rank = dp_group.rank_in_group if dp_group is not None else 0
    return get_tensor_model_parallel_rank() * dp_size + dp_rank


def engram_gathered_num_tokens() -> int:
    """Per-replica token slot for the node-local Engram DP group."""
    dp_metadata = get_forward_context().dp_metadata
    if dp_metadata is None:
        raise RuntimeError("a DP-shared engram table needs DP token metadata")
    group = get_engram_dp_group()
    assert group is not None
    # Engram groups are contiguous slices of the full DP group.
    start = get_dp_group().rank_in_group - group.rank_in_group
    return int(dp_metadata.num_tokens_across_dp_cpu[start : start + group.world_size].max())


def gather_engram_hashes(hash_ids: torch.Tensor, *, dp_shared_memory: bool = False) -> torch.Tensor:
    """Collect the n-gram ids of every DP replica sharing one table.

    Replicas are padded to a common token slot, so the gathered shape is
    static under CUDA graph capture (where DP already pads alike).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_shared_memory:
        return hash_ids
    slot = engram_gathered_num_tokens()
    if hash_ids.shape[0] > slot:
        raise ValueError("Engram token count exceeds the DP token slot")
    if hash_ids.shape[0] < slot:
        pad = hash_ids.new_full((slot - hash_ids.shape[0], *hash_ids.shape[1:]), DEAD_ID)
        hash_ids = torch.cat((hash_ids, pad))
    return dp_group.all_gather(hash_ids, dim=0)


@triton.jit(do_not_specialize=["num_tokens", "token_start", "num_elements"])
def _engram_select_rows_kernel(
    gathered,
    output,
    num_tokens,
    token_start,
    num_elements,
    LOCAL_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tokens = token_start + offsets // WIDTH
    cols = offsets % WIDTH
    source = (cols // LOCAL_WIDTH * num_tokens + tokens) * LOCAL_WIDTH
    source += cols % LOCAL_WIDTH
    values = tl.load(gathered + source, (offsets < num_elements) & (tokens < num_tokens), other=0)
    tl.store(output + offsets, values, offsets < num_elements)


def _engram_select_rows(
    gathered: torch.Tensor,
    output: torch.Tensor,
    source_tokens: int,
    token_start: int,
    local_width: int,
) -> None:
    """Copy one token window out of a rank-major gathered buffer.

    Both gathers land rank-major ([rank][token][local width]); this walks the
    window the rank keeps and lays its ranks out side by side as width.
    """
    if output.numel() == 0:
        return
    _engram_select_rows_kernel[(triton.cdiv(output.numel(), 1024),)](
        gathered,
        output,
        source_tokens,
        token_start,
        output.numel(),
        local_width,
        output.shape[1] * output.shape[2],
        BLOCK_SIZE=1024,
    )


def _gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Exchange DP tokens for heads, retaining only this replica's tokens."""
    dp_group = get_engram_dp_group()
    assert dp_group is not None
    slot, remainder = divmod(staged.shape[0], dp_group.world_size)
    assert remainder == 0 and 0 <= num_tokens <= slot
    gathered = dp_group.all_gather(staged, dim=0)
    local_heads, dim = staged.shape[1:]
    rows = staged.new_empty((num_tokens, dp_group.world_size * local_heads, dim))
    _engram_select_rows(
        gathered,
        rows,
        staged.shape[0],
        dp_group.rank_in_group * slot,
        local_heads * dim,
    )
    return rows
