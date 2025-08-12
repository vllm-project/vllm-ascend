# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.distributed.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region, all_to_all_hp2sp,
    all_to_all_sp2hp, gather_from_sequence_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region)
from vllm_ascend.ops.comm_utils import async_all_to_all


class MoEDispatcherConfig:

    def __init__(self):
        self.num_local_experts: int = 0
        self.num_moe_experts: int = 0
        self.moe_pad_expert_input_to_capacity: bool = False
        self.moe_expert_capacity_factor: Optional[float] = None
        self.moe_router_topk: int = 2
        self.moe_grouped_gemm: bool = False
        self.group_topk: int = 0
        self.num_groups: int = 1
        self.expert_bias: torch.Tensor = None
        self.scaling_factor: Optional[float] = None
        self.is_fused: bool = True

    def set_num_local_experts(self, num_local_experts):
        self.num_local_experts = num_local_experts
        return self

    def set_num_moe_experts(self, num_moe_experts):
        self.num_moe_experts = num_moe_experts
        return self

    def set_moe_pad_expert_input_to_capacity(self,
                                             moe_pad_expert_input_to_capacity):
        self.moe_pad_expert_input_to_capacity = moe_pad_expert_input_to_capacity
        return self

    def set_moe_expert_capacity_factor(self, moe_expert_capacity_factor):
        self.moe_expert_capacity_factor = moe_expert_capacity_factor
        return self

    def set_moe_router_topk(self, moe_router_topk):
        self.moe_router_topk = moe_router_topk
        return self

    def set_moe_grouped_gemm(self, moe_grouped_gemm):
        self.moe_grouped_gemm = moe_grouped_gemm
        return self

    def set_group_topk(self, group_topk):
        self.group_topk = group_topk
        return self

    def set_num_groups(self, num_groups):
        self.num_groups = num_groups
        return self

    def set_expert_bias(self, expert_bias):
        self.expert_bias = expert_bias
        return self

    def set_scaling_factor(self, scaling_factor):
        self.scaling_factor = scaling_factor
        return self

    def set_is_fused(self, is_fused):
        self.is_fused = is_fused
        return self

    def build(self):
        return self


class MoEDispatcher:

    def __init__(self, config: MoEDispatcherConfig) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.config = config
        self.shared_experts = None

    def set_shared_experts(self, shared_experts):
        self.shared_experts = shared_experts

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @property
    def tp_ep_group(self):
        """Get expert tensor and model parallel group."""
        return None

    @property
    def tp_ep_size(self):
        return 1


class MoEAlltoAllSeqOverLapDispatcher(MoEDispatcher):
    overlap_stream = None
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.

    """

    def __init__(self, config: MoEDispatcherConfig):
        """
        Initialize the AlltoAllSeq token dispatcher.

        Args:
            config (MoEDispatcherConfig): Configuration for the transformer model.
        """
        super().__init__(config)
        self.num_local_experts = config.num_local_experts
        self.config = config
        # use MOEAlltoAllSEQTokenDispatcher to init

        self.hidden_shape = None
        self.num_input_tokens = None
        self.num_experts = config.num_moe_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = (self.ep_rank * self.num_local_experts)

        self.local_expert_indices = [
            local_expert_indices_offset + i
            for i in range(self.num_local_experts)
        ]
        assert (len(self.local_expert_indices) == self.num_local_experts
                ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (self.local_expert_indices[i] ==
                    self.local_expert_indices[i + 1] -
                    1), "local_expert_indices must be continuous"
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.routing_map = None
        self.hidden_shape_before_permute = None

        # [tp_ep_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert_cpu = None
        self.num_global_tokens_per_local_expert = None

        # A cuda stream synchronization is needed in self.token_permutation()
        # in some cases, because there are several non-blocking DtoH data
        # transfers called in self.preprocess(). The synchronization happens
        # at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall",
        # "before_finish", and "no_sync".
        self.device_sync_point = "no_sync"

        # cached intermediate tensors.
        self.cached_permutated_local_input_tokens = None
        self.cached_global_input_tokens = None
        self.cached_shared_expert_output = None
        self.tokens_per_expert = None
        self.perm1_finish_event = None
        self.global_input_tokens_local_experts_indices = None

        if MoEAlltoAllSeqOverLapDispatcher.overlap_stream is None:
            MoEAlltoAllSeqOverLapDispatcher.overlap_stream = torch.npu.Stream()

        self.overlap_stream = MoEAlltoAllSeqOverLapDispatcher.overlap_stream

    def preprocess(self,
                   indices: torch.Tensor,
                   with_sync=True) -> torch.Tensor:
        """
        Preprocess routing map for AlltoAll communication and token permutation.
        This method computes the number of tokens assigned to each expert based on
        the routing map. It also initializes the necessary data structures for
        AlltoAll communication, such as input and output splits, and the mapping
        between global tokens and local experts.

        Args:
            routing_map (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].

        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        num_local_tokens_per_expert = torch.histc(indices,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)

        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.ep_size

        # Dropless
        self.num_out_tokens = indices.numel()
        if self.ep_size > 1 or self.num_local_experts > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.device_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed to get the
            # `tokens_per_expert` CPU value.
            self.device_sync_point = "before_finish"

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (num_local_tokens_per_expert.reshape(
                ep_size, self.num_local_experts).sum(axis=1).to(
                    torch.device("cpu"), non_blocking=True).numpy())
            num_global_tokens_per_expert = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=self.ep_group).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
                0]:self.local_expert_indices[-1] + 1]
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before sum."
                )
            self.output_splits = (self.num_global_tokens_per_local_expert.sum(
                axis=-1).to(torch.device("cpu"), non_blocking=True).numpy())
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(
                axis=0)
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts)
            num_tokens_per_local_expert = num_local_tokens_per_expert

        if self.num_local_experts > 1 and with_sync:
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            self.device_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel())

        return num_tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """
        Dispatch tokens to local experts using AlltoAllSeq communication.

        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            probs (torch.Tensor): Probs of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            routing_map (torch.Tensor): Mapping of tokens assigned to experts.
                Shape: [num_tokens, num_experts].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.top_indices = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for routing map"

        # Permutation 1: input to AlltoAll input
        def alltoall_token_permutation1(hidden_states, routing_map):
            assert self.hidden_shape is not None
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
            tokens_per_expert = self.preprocess(routing_map)
            if self.tp_ep_size > 1:
                hidden_states = all_to_all_sp2hp(hidden_states,
                                                 group=self.tp_ep_group)
            self.hidden_shape_before_permute = hidden_states.shape

            if self.device_sync_point == "before_permutation_1":
                torch.npu.current_stream().synchronize()

            permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=self.top_indices,
                num_out_tokens=self.num_out_tokens,
            )
            return permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert

        permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert = alltoall_token_permutation1(
            hidden_states, routing_map)
        self.reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping
        # permute 1

        ep_group = self.ep_group

        # Perform expert parallel AlltoAll communication
        if self.device_sync_point == "before_ep_alltoall":
            torch.npu.current_stream().synchronize()
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            ep_group,
        )

        # shared experts compute
        if self.shared_experts is not None:
            (share_experts_output), *_ = self.shared_experts(hidden_states)
        else:
            share_experts_output = None

        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        def alltoall_token_permutation2(global_input_tokens):
            # Permutation 2: Sort tokens by local expert.
            if self.num_local_experts > 1:
                global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                    global_input_tokens,
                    self.global_input_tokens_local_experts_indices)

            # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
            # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
            if self.tp_ep_size > 1 and self.config.moe_grouped_gemm:
                global_input_tokens = all_gather_last_dim_from_tensor_parallel_region(
                    global_input_tokens, self.tp_ep_group)
            if self.device_sync_point == "before_finish":
                torch.npu.current_stream().synchronize()

            return global_input_tokens

        # token premute2 input
        global_input_tokens = alltoall_token_permutation2(global_input_tokens)

        return share_experts_output, global_input_tokens, tokens_per_expert

    def token_unpermutation(self,
                            hidden_states: torch.Tensor,
                            bias: torch.Tensor = None):
        """
        Reverse the token permutation to restore the original order.

        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """

        def alltoall_token_unpermutation1(hidden_states):
            assert bias is None, "Bias is not supported in MoEAlltoAllSeqTokenDispatcher"
            # Perform tensor parallel Reduce-Scatter
            # hidden_states: [SEQL, H] -> [SEQL, H/TP]
            if self.tp_ep_size > 1:
                hidden_states = reduce_scatter_last_dim_to_tensor_parallel_region(
                    hidden_states, group=self.tp_ep_group)

            # Unpermutation 2: expert output to AlltoAll input
            if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
                hidden_states = torch_npu.npu_moe_token_unpermute(
                    hidden_states,
                    self.reversed_global_input_permutation_mapping)

            return hidden_states

        hidden_states = alltoall_token_unpermutation1(hidden_states)

        ep_group = self.ep_group
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits, ep_group)
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        def alltoall_token_unpermutation2(permutated_local_input_tokens):
            # Unpermutation 1: AlltoAll output to output

            output = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=permutated_local_input_tokens,
                sorted_indices=self.reversed_local_input_permutation_mapping.
                to(torch.int32),
                probs=self.probs,
                restore_shape=self.hidden_shape_before_permute)

            # Perform tensor parallel AlltoAll communication
            # output: [S*B, H/TP] -> [S*B/TP, H]
            if self.tp_ep_size > 1:
                output = all_to_all_hp2sp(output, self.tp_ep_group)

            # Reshape the output tensor
            output = output.view(self.hidden_shape)
            return output

        output = alltoall_token_unpermutation2(permutated_local_input_tokens)

        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None
        self.num_global_tokens_per_local_expert_cpu = None

        return output, None


class MoETokenDispatcher(ABC):

    def __init__(self) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.shared_experts = None

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group().device_group

    @property
    def ep_rank(self):
        return get_ep_group().rank_in_group

    @property
    def ep_size(self):
        return get_ep_group().world_size

    @property
    def tp_ep_group(self):
        """Get expert tensor and model parallel group."""
        return None

    @property
    def tp_ep_size(self):
        return 1

    @abstractmethod
    def token_permutation(self,
                          top_k: int,
                          num_experts: int,
                          hidden_states: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          expert_map: torch.Tensor = None,
                          log2phy: torch.Tensor = None,
                          global_redundant_expert_num: int = 0):
        """Dispatch tokens to experts.
        Args:
            hidden_states (torch.Tensor): Input hidden_states.
            topk_weights (torch.Tensor): The routing probability tensor [num_tokens, num_experts].
            topk_ids (torch.Tensor): Token to expert mapping tensor.
        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self,
                            expert_output: torch.Tensor,
                            bias: torch.Tensor = None):
        """Restores the expert output to its original ordering.
        Args:
            expert_output (torch.Tensor): The output tensor from the expert models.
            bias (torch.Tensor): The bias tensor.
        Returns:
            (torch.Tensor, torch.Tensor): Unpermuted activation and optional bias.
        """
        raise NotImplementedError("Restore function not implemented.")

    def set_shared_experts(self, shared_experts):
        """Set shared expert to the dispatcher."""
        self.shared_experts = shared_experts


class UnquantizedTokenDispatcherWithAll2AllV(MoETokenDispatcher):
    overlap_stream = None
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self):
        """
        Initialize the AlltoAllSeq token dispatcher.
        Args:
            config (MoEDispatcherConfig): Configuration for the transformer model.
        """
        super().__init__()
        self.moe_grouped_gemm = False
        self.expert_map = None

        self.top_k = 0
        self.num_local_experts = 0
        self.num_experts = 0
        self.expert_ids_per_ep_rank = None
        self.local_expert_indices = None

        self.hidden_shape = None
        self.num_input_tokens = None

        self.topk_weights = None
        self.input_splits = None
        self.output_splits = None
        self.topk_ids = None
        self.hidden_shape_before_permute = None

        # [tp_ep_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert_cpu = None
        self.num_global_tokens_per_local_expert = None

        # A cuda stream synchronization is needed in self.token_permutation()
        # in some cases, because there are several non-blocking DtoH data
        # transfers called in self.preprocess(). The synchronization happens
        # at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall",
        # "before_finish", and "no_sync".
        self.device_sync_point = "no_sync"

        # cached intermediate tensors.
        self.cached_permutated_local_input_tokens = None
        self.cached_global_input_tokens = None
        self.cached_shared_expert_output = None
        self.tokens_per_expert = None
        self.perm1_finish_event = None
        self.global_input_tokens_local_experts_indices = None

        if MoEAlltoAllSeqOverLapDispatcher.overlap_stream is None:
            MoEAlltoAllSeqOverLapDispatcher.overlap_stream = torch.npu.Stream()

        self.overlap_stream = MoEAlltoAllSeqOverLapDispatcher.overlap_stream

    def preprocess(self,
                   indices: torch.Tensor,
                   with_sync=True) -> torch.Tensor:
        """
        Preprocess routing map for AlltoAll communication and token permutation.
        This method computes the number of tokens assigned to each expert based on
        the routing map. It also initializes the necessary data structures for
        AlltoAll communication, such as input and output splits, and the mapping
        between global tokens and local experts.
        Args:
            topk_ids (torch.Tensor): The mapping of tokens to experts, with shape
                [num_tokens, num_experts].
        Returns:
            torch.Tensor: Tensor containing the number of tokens assigned to local expert.
        """
        self.num_local_experts = self.num_experts // self.ep_size
        assert self.num_local_experts > 0, "Expected at least one expert"
        if self.num_local_experts > 1:
            self.expert_ids_per_ep_rank = torch.tensor(
                [i % self.num_local_experts for i in range(self.num_experts)],
                dtype=torch.int32,
                device=torch.npu.current_device(),
            )

        local_expert_indices_offset = (self.ep_rank * self.num_local_experts)

        self.local_expert_indices = [
            local_expert_indices_offset + i
            for i in range(self.num_local_experts)
        ]
        assert (len(self.local_expert_indices) == self.num_local_experts
                ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (self.local_expert_indices[i] ==
                    self.local_expert_indices[i + 1] -
                    1), "local_expert_indices must be continuous"

        num_local_tokens_per_expert = torch.histc(indices,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)

        ep_size = self.ep_size

        # Dropless
        self.num_out_tokens = indices.numel()
        if self.ep_size > 1 or self.num_local_experts > 1:
            # Token dropless and enable ep. A synchronization is needed before expert parallel
            # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
            self.device_sync_point = "before_ep_alltoall"
        else:
            # Token dropless and no ep. A synchronization is needed to get the
            # `tokens_per_expert` CPU value.
            self.device_sync_point = "before_finish"

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (num_local_tokens_per_expert.reshape(
                ep_size, self.num_local_experts).sum(axis=1).to(
                    torch.device("cpu"), non_blocking=True).numpy())
            num_global_tokens_per_expert = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert,
                group=self.ep_group).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
                0]:self.local_expert_indices[-1] + 1]
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before sum."
                )
            self.output_splits = (self.num_global_tokens_per_local_expert.sum(
                axis=-1).to(torch.device("cpu"), non_blocking=True).numpy())
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(
                axis=0)
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts)
            num_tokens_per_local_expert = num_local_tokens_per_expert

        if self.num_local_experts > 1 and with_sync:
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            self.device_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel())

        return num_tokens_per_local_expert

    def token_permutation(self,
                          top_k: int,
                          num_experts: int,
                          hidden_states: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          expert_map: torch.Tensor = None,
                          log2phy: torch.Tensor = None,
                          global_redundant_expert_num: int = 0):
        """
        Dispatch tokens to local experts using AlltoAllSeq communication.
        Args:
            hidden_states (torch.Tensor): Input token embeddings.
            topk_weights (torch.Tensor): Probs of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
            topk_ids (torch.Tensor): Mapping of tokens assigned to experts.
                Shape: [num_tokens, num_experts].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Permuted token embeddings for local experts.
                - Number of tokens per expert.
        """
        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        self.top_indices = topk_ids
        self.expert_map = expert_map
        self.top_k = top_k
        self.num_experts = num_experts
        assert topk_weights.dim() == 2, "Expected 2D tensor for topk_weights"
        assert topk_ids.dim() == 2, "Expected 2D tensor for routing map"

        # Permutation 1: input to AlltoAll input
        def alltoall_token_permutation1(hidden_states, topk_ids):
            assert self.hidden_shape is not None
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
            tokens_per_expert = self.preprocess(topk_ids)
            if self.tp_ep_size > 1:
                hidden_states = all_to_all_sp2hp(hidden_states,
                                                 group=self.tp_ep_group)
            self.hidden_shape_before_permute = hidden_states.shape

            if self.device_sync_point == "before_permutation_1":
                torch.npu.current_stream().synchronize()

            permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=self.top_indices,
                num_out_tokens=self.num_out_tokens,
            )
            return permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert

        permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert = alltoall_token_permutation1(
            hidden_states, topk_ids)
        self.reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping
        # permute 1

        # Perform expert parallel AlltoAll communication
        if self.device_sync_point == "before_ep_alltoall":
            torch.npu.current_stream().synchronize()
        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )

        # shared experts compute
        if self.shared_experts is not None:
            (share_experts_output), *_ = self.shared_experts(hidden_states)
        else:
            share_experts_output = None

        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        def alltoall_token_permutation2(global_input_tokens):
            # Permutation 2: Sort tokens by local expert.
            if self.num_local_experts > 1:
                global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                    global_input_tokens,
                    self.global_input_tokens_local_experts_indices)

            # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
            # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
            if self.tp_ep_size > 1 and self.moe_grouped_gemm:
                global_input_tokens = all_gather_last_dim_from_tensor_parallel_region(
                    global_input_tokens, self.tp_ep_group)
            if self.device_sync_point == "before_finish":
                torch.npu.current_stream().synchronize()

            return global_input_tokens

        # token premute2 input
        global_input_tokens = alltoall_token_permutation2(global_input_tokens)

        return share_experts_output, global_input_tokens, tokens_per_expert

    def preprocess_and_permtute1(self,
                                 hidden_states: torch.Tensor,
                                 topk_weights: torch.Tensor,
                                 topk_ids: torch.Tensor,
                                 shared_experts=None,
                                 shared_experts_input: torch.Tensor = None):
        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        self.top_indices = topk_ids
        assert topk_weights.dim() == 2, "Expected 2D tensor for topk_weights"
        assert topk_ids.dim() == 2, "Expected 2D tensor for routing map"
        assert self.hidden_shape is not None

        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(topk_ids, with_sync=False)
        self.hidden_shape_before_permute = hidden_states.shape

        if self.device_sync_point == "before_permutation_1":
            torch.npu.current_stream().synchronize()

        event = torch.npu.current_stream().record_event()
        self.perm1_finish_event = torch.npu.Event()
        with torch.npu.stream(self.overlap_stream):
            assert self.overlap_stream is not None
            self.overlap_stream.wait_event(event)

            if shared_experts is not None:
                shared_output = shared_experts(shared_experts_input)
                self.cached_shared_expert_output = shared_output

            hidden_states, self.reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                tokens=hidden_states,
                indices=self.top_indices,
                num_out_tokens=self.num_out_tokens,
            )

            self.perm1_finish_event.record()

        # repeat interleve will launch a sync on current_stream.
        if self.num_local_experts > 1:
            self.device_sync_point = "no_sync"
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel())

        self.cached_permutated_local_input_tokens = hidden_states
        self.tokens_per_expert = tokens_per_expert

    def dispatch_alltoall(self):
        # Perform expert parallel AlltoAll communication
        if self.device_sync_point == "before_ep_alltoall":
            torch.npu.current_stream().synchronize()

        torch.npu.current_stream().wait_event(self.perm1_finish_event)
        self.perm1_finish_event = None
        _, self.cached_global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            self.cached_permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )
        permute1_ep_all_to_all_handle.wait()
        if self.cached_permutated_local_input_tokens is None:
            raise ValueError(
                "cached_permutated_local_input_tokens must be set before operations."
            )
        self.cached_permutated_local_input_tokens.untyped_storage().resize_(0)
        self.cached_permutated_local_input_tokens = None

    def permute2(self):
        global_input_tokens = self.cached_global_input_tokens
        if self.num_local_experts > 1:
            global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                self.cached_global_input_tokens,
                self.global_input_tokens_local_experts_indices)
            assert self.cached_global_input_tokens is not None
            self.cached_global_input_tokens.untyped_storage().resize_(0)
            self.cached_global_input_tokens = None

        return global_input_tokens, self.tokens_per_expert

    def unpermute1(self, hidden_states: torch.Tensor):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping)
        self.cached_global_output_tokens = hidden_states
        self.reversed_global_input_permutation_mapping = None

    def combine_alltoall(self):
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, self.cached_local_output_tokens, handle = async_all_to_all(
            self.cached_global_output_tokens, self.input_splits,
            self.output_splits, self.ep_group)
        handle.wait()
        self.cached_global_output_tokens.untyped_storage().resize_(0)
        self.cached_global_output_tokens = None
        self.input_splits = None
        self.output_splits = None

    def unpermute2(self):
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=self.cached_local_output_tokens,
            sorted_indices=self.reversed_local_input_permutation_mapping.to(
                torch.int32),
            probs=self.topk_weights,
            restore_shape=self.hidden_shape_before_permute)

        output = output.view(self.hidden_shape)

        self.topk_weights = None
        self.reversed_local_input_permutation_mapping = None
        self.cached_local_output_tokens.untyped_storage().resize_(0)
        self.cached_local_output_tokens = None

        return output

    def token_unpermutation(self,
                            hidden_states: torch.Tensor,
                            bias: torch.Tensor = None):
        """
        Reverse the token permutation to restore the original order.
        Args:
            hidden_states (torch.Tensor): Output from local experts.
            bias (torch.Tensor, optional): Bias tensor (not supported).
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Unpermuted token embeddings in the original order.
                - None (bias is not supported).
        """

        def alltoall_token_unpermutation1(hidden_states):
            assert bias is None, "Bias is not supported in MoEAlltoAllSeqTokenDispatcher"
            # Perform tensor parallel Reduce-Scatter
            # hidden_states: [SEQL, H] -> [SEQL, H/TP]
            if self.tp_ep_size > 1:
                hidden_states = reduce_scatter_last_dim_to_tensor_parallel_region(
                    hidden_states, group=self.tp_ep_group)

            # Unpermutation 2: expert output to AlltoAll input
            if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
                hidden_states = torch_npu.npu_moe_token_unpermute(
                    hidden_states,
                    self.reversed_global_input_permutation_mapping)

            return hidden_states

        hidden_states = alltoall_token_unpermutation1(hidden_states)

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits,
            self.ep_group)
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        def alltoall_token_unpermutation2(permutated_local_input_tokens):
            # Unpermutation 1: AlltoAll output to output

            output = torch_npu.npu_moe_token_unpermute(
                permuted_tokens=permutated_local_input_tokens,
                sorted_indices=self.reversed_local_input_permutation_mapping.
                to(torch.int32),
                probs=self.topk_weights,
                restore_shape=self.hidden_shape_before_permute)

            # Perform tensor parallel AlltoAll communication
            # output: [S*B, H/TP] -> [S*B/TP, H]
            if self.tp_ep_size > 1:
                output = all_to_all_hp2sp(output, self.tp_ep_group)

            # Reshape the output tensor
            output = output.view(self.hidden_shape)
            return output

        output = alltoall_token_unpermutation2(permutated_local_input_tokens)

        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None
        self.num_global_tokens_per_local_expert_cpu = None

        return output


class QuantizedTokenDispatcherWithAll2All(MoETokenDispatcher):

    def __init__(self):
        super().__init__()
        self._meta = {}

    def _save_meta(self, **kwargs):
        self._meta.update(kwargs)

    def init_routing(self, hidden_states, topk_ids, global_num_experts):
        num_tokens, _ = hidden_states.shape
        row_idx_len = num_tokens * self._meta["top_k"]
        row_idx = (torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=hidden_states.device).view(
                                    self._meta["top_k"],
                                    -1).permute(1, 0).contiguous())
        hidden_states, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=num_tokens)

        return hidden_states, expanded_row_idx, expanded_expert_idx

    def token_permutation(self,
                          top_k: int,
                          num_experts: int,
                          hidden_states: torch.Tensor,
                          topk_weights: torch.Tensor,
                          topk_ids: torch.Tensor,
                          expert_map: torch.Tensor = None,
                          log2phy: torch.Tensor = None,
                          global_redundant_expert_num: int = 0):
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]
        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        num_tokens, _ = hidden_states.shape

        self._save_meta(top_k=top_k)

        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            if hasattr(torch_npu, "npu_moe_init_routing_quant"):
                quantized_tokens, expanded_row_idx, global_expert_tokens, _, token_scales = torch_npu.npu_moe_init_routing_quant(
                    hidden_states,
                    expert_idx=topk_ids.to(torch.int32),
                    active_num=0,
                    expert_capacity=0,
                    expert_num=global_num_experts,
                    drop_pad_mode=0,
                    expert_tokens_num_mode=2,
                    expert_tokens_before_capacity_flag=False,
                    quant_mode=1,
                )
            else:
                hidden_states, expanded_row_idx, expanded_expert_idx = self.init_routing(
                    hidden_states, topk_ids, global_num_experts)
                global_expert_tokens = torch.bincount(
                    expanded_expert_idx, minlength=global_num_experts)
                global_expert_tokens = global_expert_tokens.to(torch.int32)
                quantized_tokens, token_scales = torch_npu.npu_dynamic_quant(
                    hidden_states)

            gather_sizes = global_expert_tokens.new_empty(
                global_expert_tokens.shape[0])
            torch.distributed.all_to_all_single(gather_sizes,
                                                global_expert_tokens)

            token_counts_combined = torch.stack(
                [gather_sizes, global_expert_tokens], dim=0)
            token_counts_combined = token_counts_combined.view(
                2, self.ep_size, -1).sum(dim=2)
            token_counts_combined_cpu = token_counts_combined.to(
                torch.device("cpu"), non_blocking=True).numpy()
            all_tokens = gather_sizes.sum()

            gathered_tokens = quantized_tokens.new_empty(
                all_tokens.item(), quantized_tokens.shape[1])
            dynamic_scale = token_scales.new_empty(gathered_tokens.shape[0])
            gather_size_list = token_counts_combined_cpu[1]
            scatter_size_list = token_counts_combined_cpu[0]

            torch.distributed.all_to_all_single(gathered_tokens,
                                                quantized_tokens,
                                                scatter_size_list,
                                                gather_size_list)
            torch.distributed.all_to_all_single(dynamic_scale, token_scales,
                                                scatter_size_list,
                                                gather_size_list)

            hidden_states, dynamic_scale, inverse_indices, expert_tokens = torch_npu.npu_moe_re_routing(
                gathered_tokens,
                gather_sizes.view(self.ep_size, -1),
                per_token_scales=dynamic_scale)
            expert_tokens = expert_tokens.to(torch.int64)
            group_list_type = 1
        else:
            hidden_states, expanded_row_idx, expanded_expert_idx = self.init_routing(
                hidden_states, topk_ids, num_experts)

            expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
                expanded_expert_idx, num_experts)
            expert_tokens = expert_tokens.to(torch.int64)
            group_list_type = 0
            dynamic_scale = None

        self._save_meta(topk_ids=topk_ids,
                        topk_weights=topk_weights,
                        expert_map=expert_map,
                        original_shape=original_shape,
                        quantized_tokens_shape=quantized_tokens.shape,
                        inverse_indices=inverse_indices,
                        scatter_size_list=scatter_size_list,
                        gather_size_list=gather_size_list,
                        expanded_row_idx=expanded_row_idx)
        return hidden_states, expert_tokens, group_list_type, dynamic_scale

    def token_unpermutation(self, expert_output, bias=None):
        if self._meta["expert_map"] is not None:
            reordered_outputs = torch.index_select(
                expert_output,
                dim=0,
                # Workaround: Convert to float so that argsort runs on AI Core instead of slower AICPU
                index=self._meta["inverse_indices"].to(
                    torch.float32).argsort().to(torch.int32))

            hidden_states = reordered_outputs.new_empty(
                self._meta["quantized_tokens_shape"])
            torch.distributed.all_to_all_single(
                hidden_states, reordered_outputs,
                self._meta["gather_size_list"],
                self._meta["scatter_size_list"])

            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                hidden_states,
                skip1=None,
                skip2=None,
                bias=None,
                scales=self._meta["topk_weights"],
                expanded_src_to_dst_row=self._meta["expanded_row_idx"],
                export_for_source_row=None,
                drop_pad_mode=2)
        else:
            # TODO: Reorder device memory 2 times here, replace the current
            # implementation here when suitable operators become available.
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                expert_output,
                skip1=None,
                skip2=None,
                bias=None,
                scales=self._meta["topk_weights"],
                expanded_src_to_dst_row=self._meta["expanded_row_idx"],
                export_for_source_row=self._meta["topk_ids"],
            )
        if len(self._meta["original_shape"]) == 3:
            final_hidden_states = final_hidden_states.view(
                self._meta["original_shape"])

        return final_hidden_states
