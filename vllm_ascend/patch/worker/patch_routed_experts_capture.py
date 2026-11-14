# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

from vllm_ascend.ascend_forward_context import MoECommType

logger = logging.getLogger(__name__)

# Strategies whose ``prepare()`` splits the token dimension across the TP group.
# ALLGATHER is absent on purpose: it gathers the batch before routing.
_TP_SHARDED_COMM_TYPES = frozenset(
    {
        MoECommType.ALLTOALL,
        MoECommType.MC2,
        MoECommType.FUSED_MC2,
    }
)

_ORIGINAL_RESTORE_LOCAL_TOPK_IDS = RoutedExpertsCapturer._restore_local_topk_ids


def _extra_attr(ctx: Any, name: str) -> Any:
    """Read a vllm-ascend forward-context extra from the context being restored.

    ``_EXTRA_CTX`` resolves the process-global forward context, which forces a
    second ``get_forward_context()`` lookup inside a hook that was already handed
    the context. Reading the same object keeps the two in sync (the V1 runner
    sets the extra as an attribute, the V2 runner publishes it through
    ``additional_kwargs``), and keeps the hook usable when only the capture call
    itself is wired to a context.
    """
    additional_kwargs = getattr(ctx, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        value = additional_kwargs.get(name)
        if value is not None:
            return value
    return getattr(ctx, name, None)


def _padded_batch_rows(ctx: Any, tp_size: int) -> int:
    """Row count the MoE dispatch padded the token dimension to before splitting it.

    Taken from the target ``prepare()`` padded to rather than from this rank's
    shard, which only adds up to the total when the split is even: All2All splits
    a batch with ``num_tokens >= tp_size`` unevenly whenever it is not a multiple
    of the TP size. There is deliberately no ``shard_rows * tp_size`` fallback --
    it would silently bring that rank-dependent length back.
    """
    comm_type = _extra_attr(ctx, "moe_comm_type")
    prepare_finalize = getattr(_extra_attr(ctx, "moe_comm_method"), "prepare_finalize", None)
    padded_num_tokens = _extra_attr(ctx, "padded_num_tokens")
    num_tokens = getattr(prepare_finalize, "num_tokens", None)

    if getattr(prepare_finalize, "replace_allreduce", False):
        # Sequence parallel: the model wrapper already chunked the sequence to a
        # multiple of the TP size, so this batch-wide target is the padded length.
        total_rows = padded_num_tokens if isinstance(padded_num_tokens, int) else None
    elif comm_type == MoECommType.ALLTOALL:
        # All2All pads to the TP size, and only when the batch is smaller than it.
        total_rows = max(num_tokens, tp_size) if isinstance(num_tokens, int) else None
    elif isinstance(num_tokens, int) and isinstance(padded_num_tokens, int):
        # MC2/FUSED_MC2 pad to ``padded_num_tokens``, and only when the batch is
        # smaller than that target.
        total_rows = max(num_tokens, padded_num_tokens)
    else:
        total_rows = None

    if total_rows is None:
        raise AssertionError(
            "RoutedExpertsCapturer: cannot rebuild the TP-sharded routing batch, the MoE dispatch "
            f"metadata is missing (comm_type={comm_type}, num_tokens={num_tokens!r}, "
            f"padded_num_tokens={padded_num_tokens!r}). Refusing to infer the length from this rank's "
            "shard, which is only valid when the split is even."
        )
    return total_rows


def _gather_tp_sharded_topk_ids(topk_ids: torch.Tensor, tp_size: int, total_rows: int) -> torch.Tensor:
    """Rebuild the pre-split batch from every TP rank's routing shard.

    ``tensor_split`` reproduces the chunks ``prepare()``/``finalize()`` use, so
    the all-gather restores the original row order.
    """
    gathered = torch.empty(
        (total_rows, topk_ids.shape[1]),
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    dist.all_gather(
        list(torch.tensor_split(gathered, tp_size, dim=0)),
        topk_ids.contiguous(),
        get_tp_group().device_group,
    )
    return gathered


def _restore_ascend_tp_sharded_rows(self, topk_ids: torch.Tensor, ctx: Any) -> torch.Tensor | None:
    """Rebuild this DP rank's rows from uneven TP shards.

    Returns ``None`` for every layout upstream already restores correctly, so
    ``_restore_local_topk_ids`` can defer to it: a cross-DP batch, a local batch
    and a gathered DP/EP shard group were never split across the TP group, and
    upstream covers them (including the fail-fast checks for their metadata).
    """
    tp_size = self.tp_size
    if tp_size <= 1 or _extra_attr(ctx, "moe_comm_type") not in _TP_SHARDED_COMM_TYPES:
        return None

    n = topk_ids.shape[0]
    dp_metadata = ctx.dp_metadata

    if dp_metadata is None:
        # All2All/MC2 split the token dimension across the TP group even with
        # DP=1, so ``topk_ids`` only holds this rank's shard. Capturing it as the
        # whole batch leaves the other ``device_buffer`` rows stale, and routing
        # replay then serves wrong expert ids for them. The condition is
        # rank-invariant, so every TP rank gathers in lockstep.
        total_rows = _padded_batch_rows(ctx, tp_size)
        restored = _gather_tp_sharded_topk_ids(topk_ids, tp_size, total_rows)
        batch_descriptor = getattr(ctx, "batch_descriptor", None)
        if batch_descriptor is None:
            # No pre-SP batch size to trim to. Rows past the batch are the padding
            # ``prepare()`` appended, and the consumer only reads
            # ``device_buffer[:num_tokens]``.
            return restored
        return restored[: batch_descriptor.num_tokens]

    num_tokens_dp = dp_metadata.num_tokens_across_dp_cpu
    num_local_tokens = int(num_tokens_dp[self.dp_rank].item())

    if n == int(num_tokens_dp.sum().item()):
        # Naive dispatch: all DP ranks' tokens are concatenated before routing,
        # so the rows were never split across the TP group.
        return None

    shard_sizes = dp_metadata.local_sizes
    if shard_sizes is not None and n == sum(shard_sizes):
        # Gathered DP/EP shards: already DP-major/TP-minor, so upstream only has
        # to select this rank's shard group.
        return None

    if n == num_local_tokens:
        # Modular-kernel path: DP combine happens inside ``quant_method.apply``
        # and ``select_experts`` only sees this rank's tokens.
        return None

    max_tokens = int(num_tokens_dp.max().item())
    if n == max_tokens * len(num_tokens_dp):
        # Padded all-gather across the DP group: each rank occupies a contiguous
        # block of ``max_tokens`` rows, and only this rank's real tokens are kept.
        start = self.dp_rank * max_tokens
        return topk_ids[start : start + num_local_tokens]

    # SP + modular-kernel path: ``prepare()`` split the batch across the TP group
    # with ``tensor_split``, so this rank only sees its shard. Gather every shard
    # back into the batch-wide row order and trim the trailing padding. The TP
    # group is always initialized on real rollout workers, and every rank in the
    # group reaches this branch in lockstep (bind is per-FusedMoE layer, SP is a
    # global condition), so a bare all_gather here will not deadlock -- let it
    # raise if the precondition is violated rather than skip silently.
    total_rows = _padded_batch_rows(ctx, tp_size)
    restored = _gather_tp_sharded_topk_ids(topk_ids, tp_size, total_rows)
    return restored[:num_local_tokens]


def _restore_local_topk_ids(self, topk_ids: torch.Tensor, ctx: Any) -> torch.Tensor:
    """Restore this DP rank's token rows, trimming trailing SP padding.

    Only the TP-shard layout needs Ascend-specific handling: ``prepare()`` uses
    ``tensor_split``, so the local shard length is rank-dependent and the
    upstream equal-shape ``all_gather`` cannot rebuild the batch.
    """
    restored = _restore_ascend_tp_sharded_rows(self, topk_ids, ctx)
    if restored is None:
        return _ORIGINAL_RESTORE_LOCAL_TOPK_IDS(self, topk_ids, ctx)
    return restored


RoutedExpertsCapturer._restore_local_topk_ids = _restore_local_topk_ids
