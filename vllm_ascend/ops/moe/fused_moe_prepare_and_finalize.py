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
        raise NotImplementedError("Prepare not implemented.")

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        raise NotImplementedError("Combine function not implemented.")


class FusedMoEPrepareAndFinalizeWithMC2(FusedMoEPrepareAndFinalize):

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        # NOTE: Since vLLM flatten tp across dp, we need to restore the original
        # tp_size and tp_rank.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The target_pad_length is calculated in forward_context, here we pad the
        hidden states and router logits. And if TP size > 1, we also need to split
        the tensors accordingly.
        """
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if not self.replace_allreduce:
            self.num_tokens, _ = hidden_states.shape
            forward_context = get_forward_context()
            mc2_mask = forward_context.mc2_mask
            target_pad_length = forward_context.padded_num_tokens
            pad_size = target_pad_length - self.num_tokens

            if pad_size > 0 and not self.enable_shared_expert_dp:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

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
                    self.split_hidden_states = split_hidden_states

                split_mc2_mask = torch.tensor_split(mc2_mask,
                                                    self.tp_size,
                                                    dim=0)
                mc2_mask = split_mc2_mask[self.tp_rank]

        return hidden_states, router_logits, mc2_mask

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """If TP size > 1, all-gather the hidden states to get the final output.
        
        Also, unpad the hidden states if needed.
        """
        if not (self.enable_shared_expert_dp or self.replace_allreduce):
            if self.tp_size > 1:
                dist.all_gather(list(self.split_hidden_states), hidden_states,
                                self.moe_config.tp_group.device_group)
                hidden_states = torch.cat(self.split_hidden_states, dim=0)

            if self.num_tokens < hidden_states.shape[0]:
                hidden_states = hidden_states[:self.num_tokens]

        return hidden_states


class FusedMoEPrepareAndFinalizeWithAll2All(FusedMoEPrepareAndFinalize):

    def __init__(self, moe_config: FusedMoEConfig):
        super().__init__(moe_config)
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        # NOTE: Since vLLM flatten tp across dp, we need to restore the original
        # tp_size and tp_rank.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.replace_allreduce = replace_allreduce
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if not (self.replace_allreduce or self.enable_shared_expert_dp):
            self.num_tokens, _ = hidden_states.shape
            pad_size = self.tp_size - self.num_tokens

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
        """If TP size > 1, all-gather the hidden states to get the final output.

        Also, unpad the hidden states if needed.
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

    def prepare(self,
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                enable_shared_expert_dp: bool = False,
                rm_router_logits: bool = False,
                replace_allreduce: bool = False,
                gate=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """When DP size > 1, pad the hidden states and router logits for communication."""
        self.rm_router_logits = rm_router_logits
        self.enable_shared_expert_dp = enable_shared_expert_dp

        if self.moe_config.dp_size > 1:
            forward_context = get_forward_context()
            max_tokens_across_dp = forward_context.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                if not self.rm_router_logits:
                    router_logits = nn.functional.pad(router_logits,
                                                      (0, 0, 0, pad_size))

            hidden_states = self.moe_config.dp_group.all_gather(
                hidden_states, 0)
            if self.rm_router_logits:
                router_logits, _ = gate(hidden_states)
            else:
                router_logits = self.moe_config.dp_group.all_gather(
                    router_logits, 0)

        return hidden_states, router_logits, None

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """When DP size > 1, reduce-scatter the hidden states to get the final output.

        When TP size > 1, all-reduce the hidden states to get the final output.
        """
        if self.moe_config.dp_size > 1 and not self.enable_shared_expert_dp:
            hidden_states = get_dp_group().reduce_scatter(hidden_states, 0)
            hidden_states = hidden_states[:self.num_tokens]

        if reduce_results and (self.moe_config.tp_size > 1
                               or self.moe_config.ep_size > 1):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states
