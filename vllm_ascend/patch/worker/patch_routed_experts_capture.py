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

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import RoutedExpertsCapturer

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

logger = logging.getLogger(__name__)


def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
    """Capture expert routing decisions for a specific layer.

    Under data parallelism, ``topk_ids`` may have four different batch
    layouts depending on where the DP combine happens and whether
    Sequence Parallelism (SP) is active for the MoE layer:
      - ``n == total`` (naive dispatch): all DP ranks' tokens are
        concatenated before routing; we slice out this rank's span
        using the cumulative per-rank counts.
      - ``n == token_num_per_dp`` (modular-kernel path): DP combine
        happens inside ``quant_method.apply``; ``select_experts`` only
        ever sees this rank's tokens, so we take the whole tensor.
      - ``n == total_with_padding`` (padded all-gather path): tokens are
        padded to max_tokens before all-gather across DP group; each
        DP rank occupies a contiguous block of size max_tokens, and we
        extract only the actual tokens for this rank (skip padding).
        When all DP ranks have equal token counts, ``total == total_with_padding``,
        so the naive dispatch branch fires instead (equivalent result).
      - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
        path): tokens were split along dim=0 across the TP group by
        ``_sequence_parallel_context``
        (``moe_runner_base.py:_sequence_parallel_context``), so each
        TP rank only sees its shard. We all-gather along dim=0 to
        reconstruct this DP rank's full routing tensor. SP pads with
        ceil-div (see ``_compute_sp_num_tokens`` in
        ``forward_context.py``), so the gathered tensor may contain a
        few trailing padding rows which are trimmed by the downstream
        ``[:token_num_per_dp]`` slice.

    Args:
        layer_id: The layer index.
        topk_ids: Tensor of shape (batch_size, num_routed_experts).
    """

    ctx = get_forward_context()
    if ctx.dp_metadata is None:  # single dp
        start_loc = 0
        end_loc = topk_ids.shape[0]
        token_num_per_dp = topk_ids.shape[0]
    else:  # multi dp
        num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
        token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
        total = int(num_tokens_dp.sum().item())
        n = topk_ids.shape[0]

        # Calculate total with padding for all-gather scenario.
        # When tokens are padded to max_tokens before all-gather across DP group,
        # the total size becomes max_tokens * dp_size.
        # Example: DP0 has 5 tokens, DP1 has 7 tokens, max_tokens=7.
        # After padding: DP0 has 7 tokens, DP1 has 7 tokens.
        # After all-gather: total_with_padding = 7 * 2 = 14.
        max_tokens = int(num_tokens_dp.max().item())
        total_with_padding = max_tokens * len(num_tokens_dp)

        if n == total:
            # Naive dispatch: all DP ranks' tokens concatenated
            # before routing. This rank owns tokens
            # [end_loc - token_num_per_dp, end_loc).
            cumsum = torch.cumsum(num_tokens_dp, dim=0)
            end_loc = int(cumsum[self.dp_rank].item())
            start_loc = end_loc - token_num_per_dp
        elif n == token_num_per_dp:
            # Modular-kernel path: DP combine happens inside
            # quant_method.apply; select_experts only sees this
            # rank's tokens, take the whole tensor.
            start_loc = 0
            end_loc = token_num_per_dp
        elif n == total_with_padding:
            # NOTE(Ronald1995): When all DP ranks have equal token counts,
            # total == total_with_padding, so the first branch (n == total)
            # fires instead. This overlap is intentional since both branches
            # produce equivalent results in that case.

            # Padded all-gather path: tokens are padded to max_tokens before
            # all-gather across DP group. Each DP rank occupies a contiguous
            # block of size max_tokens. Extract only the actual tokens for
            # this rank (skip padding).
            # Example: dp_rank=0, max_tokens=7, token_num_per_dp=5.
            # start_loc = 0 * 7 = 0
            # end_loc = 0 + 5 = 5 (only first 5 tokens are valid)

            start_loc = self.dp_rank * max_tokens
            end_loc = start_loc + token_num_per_dp
        elif (
            self.tp_size > 1
            and n != token_num_per_dp
            and (
                # all2all scenario use tensor split, different tp rank have different
                # size of tokens.
                n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
                or n == token_num_per_dp // self.tp_size
                # mc2 scenario will pad dp tokens to max_tokens and then ceil-div.
                or n == (max_tokens + self.tp_size - 1) // self.tp_size
            )
        ):
            # SP + modular-kernel path. All-gather across the TP
            # group along dim=0 to reconstruct the full per-DP-rank
            # tensor; keep only the first ``token_num_per_dp`` rows
            # (trailing rows are SP ceil-div padding). The TP group
            # is always initialized on real rollout workers, and
            # every rank in the group reaches this branch in
            # lockstep (bind is per-FusedMoE layer, SP is a global
            # condition), so a bare all_gather here will not
            # deadlock -- let it raise if the precondition is
            # violated rather than skip silently.
            #
            # ``topk_ids`` is already whatever the router produced
            # (typically int32/int64, both supported by NCCL); the
            # downstream ``device_buffer[...] = topk_ids[...]``
            # setitem narrows into int32 automatically.

            # NOTE(Ronald1995): if total_num_per_dp == max_tokens,
            # it will be both all2all and mc2 scenario.
            # but we fires all2all scenario first.
            # the result will be the same.
            # all2all scenario in vllm-ascend.
            if _EXTRA_CTX.moe_comm_type == MoECommType.ALLTOALL:
                gather_topk_ids_shape = (
                    (token_num_per_dp, topk_ids.shape[1])
                    if token_num_per_dp >= self.tp_size
                    else (self.tp_size, topk_ids.shape[1])
                )
            # mc2 scenario in vllm-ascend
            else:
                gather_topk_ids_shape = (n * self.tp_size, topk_ids.shape[1])

            gather_topk_ids = torch.empty(
                gather_topk_ids_shape,
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )
            split_topk_ids = torch.tensor_split(gather_topk_ids, self.tp_size, dim=0)
            dist.all_gather(list(split_topk_ids), topk_ids, get_tp_group().device_group)
            topk_ids = gather_topk_ids
            start_loc = 0
            end_loc = token_num_per_dp
        else:
            sp_expected = (token_num_per_dp + self.tp_size - 1) // self.tp_size if self.tp_size > 0 else -1
            raise AssertionError(
                "RoutedExpertsCapturer: unexpected topk_ids batch "
                f"dim {n} (expected {total}, {token_num_per_dp}, "
                f"{total_with_padding}, or {sp_expected} for "
                f"dp_rank={self.dp_rank}, tp_size={self.tp_size})"
            )

    # Defensive: model may expose more layers than the capture buffer
    # was sized for (unusual, but guards against miss-config).
    if layer_id >= self.device_buffer.shape[1]:
        return

    self.device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[start_loc:end_loc, :]


RoutedExpertsCapturer.capture = capture
