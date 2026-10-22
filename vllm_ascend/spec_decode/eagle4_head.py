# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device-only candidate rescoring using portable PyTorch NPU operators."""

import torch
from eagle_harness.models.low_rank_head import low_rank_scores

# Bound the gathered full-head rows, including their FP32 accumulation copy.
_MAX_GATHER_ELEMENTS = 1 << 20
_CANDIDATE_TILE = 64


def local_candidates(
    hidden: torch.Tensor,
    down_proj: torch.Tensor,
    score_head: torch.Tensor,
    weight: torch.Tensor,
    num_candidates: int,
    vocab_start: int,
    vocab_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official low-rank scoring followed by NPU candidate selection/rescoring."""
    scores = low_rank_scores(hidden, down_proj, score_head, vocab_rows)
    local_ids = torch.topk(scores[:, :vocab_rows], num_candidates, dim=-1, sorted=False).indices
    return rescore_candidates(weight, hidden, local_ids), local_ids + vocab_start


def rescore_candidates(weight: torch.Tensor, hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Rescore selected vocabulary rows, accumulating in FP32 like EAGLE-4."""
    if hidden.device.type == "npu":
        from vllm_ascend.ops.triton.eagle4_head import gather_dot

        return gather_dot(weight, hidden, indices)
    rows, candidates = indices.shape
    logits = hidden.new_empty((rows, candidates), dtype=torch.float32)
    if rows == 0:
        return logits
    # Tile both dimensions: no [batch, candidates, hidden] allocation whose
    # size grows without bound at high concurrency. Shapes are host metadata.
    col_tile = min(candidates, _CANDIDATE_TILE)
    row_tile = min(rows, max(1, _MAX_GATHER_ELEMENTS // (col_tile * weight.shape[1])))
    for row in range(0, rows, row_tile):
        x = hidden[row : row + row_tile].float().unsqueeze(-1)
        for col in range(0, candidates, col_tile):
            ids = indices[row : row + row_tile, col : col + col_tile]
            selected = weight[ids].float()
            values = torch.bmm(selected, x).squeeze(-1)
            # The official gather-dot rounds back to the draft dtype before
            # the candidate-tail promotes to FP32.
            logits[row : row + row_tile, col : col + col_tile] = values.to(hidden.dtype).float()
    return logits


def candidates_to_logits(logits: torch.Tensor, ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Merge rank-major candidates to [batch, vocab] without changing row order."""
    full = logits.new_full((logits.shape[0], vocab_size), float("-inf"))
    return full.scatter_(1, ids, logits)


def pack_candidates(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Send logit bits and token IDs in one HCCL all-gather."""
    return torch.cat((logits.contiguous().view(torch.int32), ids.to(torch.int32)), dim=1)


def unpack_candidates(packed: torch.Tensor, tp_size: int, candidates: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = packed.shape[0]
    shards = packed.view(rows, tp_size, 2 * candidates)
    logits = shards[:, :, :candidates].contiguous().view(torch.float32).view(rows, tp_size * candidates)
    ids = shards[:, :, candidates:].to(torch.int64).reshape(rows, tp_size * candidates)
    return logits, ids
