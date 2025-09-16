# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_dp_group, get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig


class FusedMoEPrepareAndFinalize(ABC):
    """
    Abstract base class for MoE (Mixture-of-Experts) tensor preparation and finalization
    in distributed environments. Subclasses implement specific communication strategies
    (e.g., AllGather, All2All, MC2, Naive Multicast) to handle tensor padding, slicing,
    broadcasting, and reduction across TP/DP/EP groups.

    Attributes:
        moe_config (FusedMoEConfig): Configuration object containing TP/DP/EP group info,
                                     sizes, ranks, and communication settings.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

    @abstractmethod
    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare tensors before MoE computation. May involve:
          - Padding to align communication boundaries
          - Slicing across tensor-parallel ranks
          - Broadcasting across data-parallel ranks
          - Recomputing router logits if needed

        Args:
            hidden_states (torch.Tensor): Input features, shape [num_tokens, hidden_size]
            router_logits (torch.Tensor): Router outputs, shape [num_tokens, num_experts]
            enable_shared_expert_dp (bool): Skip DP communication for shared experts
            rm_router_logits (bool): Discard input router_logits and recompute via gate
            replace_allreduce (bool): Bypass default all-reduce behavior
            gate (nn.Module, optional): Gate network to recompute router_logits if needed

        Returns:
            Tuple of:
                - processed hidden_states (may be padded/sliced/broadcasted)
                - processed router_logits (may be recomputed or broadcasted)
                - optional communication mask (e.g., mc2_mask for sparse ops)
        """
        raise NotImplementedError("Prepare not implemented.")

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """
        Finalize MoE output. May involve:
          - Gathering sliced tensors across TP ranks
          - Reducing or scattering across DP ranks
          - Unpadding to original token count
          - Applying all-reduce across TP/EP if requested

        Args:
            hidden_states (torch.Tensor): MoE layer output, possibly padded or sliced
            reduce_results (bool): Whether to apply all-reduce across TP/EP groups

        Returns:
            torch.Tensor: Final output with shape [original_num_tokens, hidden_size]
        """
        raise NotImplementedError("Finalize function not implemented.")


class FusedMoEPrepareAndFinalizeWithMC2(FusedMoEPrepareAndFinalize):
    """
    MoE communication strategy using MC2 (Memory-Centric Communication).
    Designed for Ascend or environments requiring explicit padding and slicing control.
    Relies on `mc2_mask` and `padded_num_tokens` from forward_context for alignment.
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """
        Restore original TP configuration.
        vLLM flattens TP and DP into a single dimension; this method recovers
        the true TP world size and rank for correct tensor slicing.
        """
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preparation steps:
          1. Fetch `mc2_mask` and target padding length from forward context.
          2. Pad `hidden_states` and `router_logits` to target length if needed.
          3. If TP > 1, split tensors along token dimension and select current TP rank's slice.
          4. Split and return corresponding `mc2_mask`.

        Skips padding/slicing if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            Tuple of (hidden_states, router_logits, mc2_mask), possibly sliced/padded.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            forward_context = get_forward_context()
            mc2_mask = forward_context.mc2_mask
            target_pad_length = forward_context.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            # Pad if necessary (unless shared expert DP is enabled)
            if pad_size > 0 and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            # Slice across TP ranks
            if self.tp_size > 1:
                if not self.enable_shared_expert_dp:
                    split_hidden_states = torch.tensor_split(hidden_states,
                                                             self.tp_size,
                                                             dim=0)
                    split_router_logits = torch.tensor_split(router_logits,
                                                             self.tp_size,
                                                             dim=0)
                    hidden_states = split_hidden_states[self.tp_rank]
                    router_logits = split_router_logits[self.tp_rank]
                    self.split_hidden_states = split_hidden_states  # Save for finalize

                # Also slice mc2_mask
                split_mc2_mask = torch.tensor_split(mc2_mask,
                                                    self.tp_size,
                                                    dim=0)
                mc2_mask = split_mc2_mask[self.tp_rank]

        return hidden_states, router_logits, mc2_mask

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices from all TP ranks to reconstruct full tensor.
          2. Unpad to original token count if padding was applied.
          3. Return tensor with shape [original_num_tokens, hidden_size].

        Skips communication and unpadding if `enable_shared_expert_dp` or `replace_allreduce` is True.
        """
        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                # All-gather across TP group
                dist.all_gather(list(self.split_hidden_states), hidden_states,
                                self.moe_config.tp_group.device_group)
                hidden_states = torch.cat(self.split_hidden_states, dim=0)

            # Unpad if necessary
            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[:self.num_tokens]

        return hidden_states


class FusedMoEPrepareAndFinalizeWithAll2All(FusedMoEPrepareAndFinalize):
    """
    MoE communication strategy using All-to-All style slicing.
    Similar to MC2 but does not use mc2_mask; instead pads to TP size for uniform slicing.
    Will be used when num_tokens exceed mc2's limitation (512 tokens/rank).
    """

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        """Restore original TP configuration (same as MC2)."""
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preparation steps:
          1. Pad hidden_states and router_logits to next multiple of TP size.
          2. If TP > 1, split along token dim and select current TP rank's slice.
          3. Save splits for later all-gather in finalize.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.

        Returns:
            Tuple of (hidden_states, router_logits, None) — no mask used in All2All.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens  # Pad to TP size (cyclic)

            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            if self.tp_size > 1:
                split_hidden_states = torch.tensor_split(hidden_states,
                                                         self.tp_size,
                                                         dim=0)
                split_router_logits = torch.tensor_split(router_logits,
                                                         self.tp_size,
                                                         dim=0)
                self.split_hidden_states = split_hidden_states

                hidden_states = split_hidden_states[self.tp_rank]
                router_logits = split_router_logits[self.tp_rank]

        return hidden_states, router_logits, None

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If TP > 1, all-gather slices to reconstruct full tensor.
          2. Unpad to original token count.
          3. Return [original_num_tokens, hidden_size] tensor.

        Skips if `enable_shared_expert_dp` or `replace_allreduce` is True.
        """
        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                dist.all_gather(list(self.split_hidden_states), hidden_states,
                                self.moe_config.tp_group.device_group)
                hidden_states = torch.cat(self.split_hidden_states, dim=0)

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[:self.num_tokens]

        return hidden_states


class FusedMoEPrepareAndFinalizeWithAllGather(FusedMoEPrepareAndFinalize):
    """
    MoE communication strategy using All-Gather + Reduce-Scatter.
    Designed for DP > 1: gather inputs across DP ranks before MoE, scatter outputs after.
    Uses `max_tokens_across_dp` from forward_context for padding alignment.
    """

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preparation steps:
          1. Fetch max token count across DP group from forward context.
          2. Pad local tensors to that size.
          3. All-gather across DP group to form global input tensor.
          4. Optionally recompute router_logits using gate if `rm_router_logits=True`.

        Returns:
            Tuple of (global_hidden_states, global_router_logits, None)
        """
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if self.moe_config.dp_size > 1:
            forward_context = get_forward_context()
            max_tokens_across_dp = forward_context.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                if not rm_router_logits:
                    router_logits = nn.functional.pad(router_logits,
                                                      (0, 0, 0, pad_size))

            # All-gather across DP group
            hidden_states = self.moe_config.dp_group.all_gather(
                hidden_states, 0)
            if rm_router_logits:
                router_logits, _ = gate(hidden_states)  # Recompute globally
            else:
                router_logits = self.moe_config.dp_group.all_gather(
                    router_logits, 0)

        return hidden_states, router_logits, None

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert, reduce-scatter output across DP group.
          2. Slice to original local token count.
          3. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [original_local_num_tokens, hidden_size]
        """
        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[:self.num_tokens]

        if reduce_results and (self.moe_config.tp_size > 1
                               or self.moe_config.ep_size > 1):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states


class FusedMoEPrepareAndFinalizeWithNaiveMulticast(FusedMoEPrepareAndFinalize):
    """
    MoE communication strategy using Naive Multicast (point-to-point broadcast).
    Will be used in prefill when using allgather in decode. Each DP rank broadcasts its slice to all others.
    Uses `cu_tokens_across_dp_cpu` (cumulative tokens) to locate slice boundaries.
    """

    def _naive_multicast(self, x: torch.Tensor,
                         cu_tokens_across_dp_cpu: torch.Tensor):
        """
        Naive multicast implementation:
          1. Create global buffer sized by total tokens across DP.
          2. Current rank copies its slice into its designated buffer region.
          3. Each rank broadcasts its slice to all others via P2P.

        Args:
            x (torch.Tensor): Local tensor [local_tokens, hidden_size]
            cu_tokens_across_dp_cpu (torch.Tensor): Cumulative token counts per DP rank

        Returns:
            torch.Tensor: Global tensor [total_tokens, hidden_size]
        """
        assert len(x.shape) == 2, "Input must be 2D [tokens, features]"
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        # Copy local slice into buffer
        start = 0 if self.moe_config.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.moe_config.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.moe_config.dp_rank]
        buffer[start:end, :].copy_(x)

        # Broadcast each slice to all ranks
        for idx in range(self.moe_config.dp_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            get_dp_group().broadcast(buffer[start:end, :], idx)
        return buffer

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preparation steps:
          1. Fetch cumulative token boundaries from forward context.
          2. Multicast hidden_states and router_logits to form global tensors.
          3. Optionally recompute router_logits globally if `rm_router_logits=True`.

        Returns:
            Tuple of (global_hidden_states, global_router_logits, None)
        """
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if self.moe_config.dp_size > 1:
            self.cu_tokens_across_dp_cpu = get_forward_context(
            ).dp_metadata.cu_tokens_across_dp_cpu
            hidden_states = self._naive_multicast(hidden_states,
                                                  self.cu_tokens_across_dp_cpu)
            if rm_router_logits:
                router_logits, _ = gate(hidden_states)
            else:
                router_logits = self._naive_multicast(
                    router_logits, self.cu_tokens_across_dp_cpu)

        return hidden_states, router_logits, None

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """
        Finalization steps:
          1. If DP > 1 and not shared expert:
               - All-reduce across DP
               - Slice to current rank's token range using cu_tokens_across_dp_cpu
          2. If `reduce_results=True` and TP/EP > 1, apply tensor_model_parallel_all_reduce.

        Returns:
            Tensor with shape [local_num_tokens, hidden_size]
        """
        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            start = 0 if self.moe_config.dp_rank == 0 else self.cu_tokens_across_dp_cpu[
                self.moe_config.dp_rank - 1]
            end = self.cu_tokens_across_dp_cpu[self.moe_config.dp_rank]
            hidden_states = get_dp_group().all_reduce(
                hidden_states)  # Sum across DP
            hidden_states = hidden_states[start:end, :]

        if reduce_results and (self.moe_config.tp_size > 1
                               or self.moe_config.ep_size > 1):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states
