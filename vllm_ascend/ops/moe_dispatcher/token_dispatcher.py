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
from typing import Optional

import torch
import torch_npu

from vllm.distributed.parallel_state import get_ep_group
from vllm_ascend.distributed.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region, all_to_all_hp2sp,
    all_to_all_sp2hp, gather_from_sequence_parallel_region,
    reduce_scatter_last_dim_to_tensor_parallel_region)
from vllm_ascend.ops.comm_utils import async_all_to_all
from vllm_ascend.ops.moe_dispatcher.moe_utils import (
    get_capacity, permute, sort_chunks_by_idxs, topk_softmax_with_capacity,
    unpermute)

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


class MoeDispatcherConfig:

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

    def __init__(self, config: MoeDispatcherConfig) -> None:
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

    def __init__(self, config: MoeDispatcherConfig):
        """
        Initialize the AlltoAllSeq token dispatcher.

        Args:
            config (MoeDispatcherConfig): Configuration for the transformer model.
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

        local_expert_indices_offset = (
                self.ep_rank * self.num_local_experts
        )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert (
                len(self.local_expert_indices) == self.num_local_experts
        ), "Invalid local expert indices"
        for i in range(len(self.local_expert_indices) - 1):
            assert (
                    self.local_expert_indices[i] == self.local_expert_indices[i + 1] - 1
            ), "local_expert_indices must be continuous"
        self.probs = None
        self.input_splits = None
        self.output_splits = None
        self.routing_map = None
        self.hidden_shape_before_permute = None

        # [tp_ep_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert_cpu = None
        self.num_global_tokens_per_local_expert = None
        input_chunk_idxs = torch.arange(self.num_experts)
        # [num_local_experts, ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        # [ep_size, num_local_experts]. Restore the output chunks by local experts.
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel().to(torch.device("cpu"), non_blocking=True)

        # Token drop and padding.
        # We need to keep track of the token num if we drop tokens without padding them.
        self.num_out_tokens = None
        # Drop and pad the input to capacity.
        self.drop_and_pad = self.config.moe_pad_expert_input_to_capacity
        if self.drop_and_pad:
            assert self.config.moe_expert_capacity_factor is not None
        self.capacity = None

        # A cuda stream synchronization is needed in self.token_permutation()
        # in some cases, because there are several non-blocking DtoH data
        # transfers called in self.preprocess(). The synchronization happens
        # at different points based on MoE settings as late as possible.
        # Valid sync points are "before_permutation_1", "before_ep_alltoall",
        # "before_finish", and "no_sync".
        self.cuda_sync_point = "no_sync"

        # cached intermediate tensors.
        self.cached_permutated_local_input_tokens = None
        self.cached_global_input_tokens = None
        self.cached_shared_expert_output = None
        self.tokens_per_expert = None

        if MoEAlltoAllSeqOverLapDispatcher.overlap_stream is None:
            MoEAlltoAllSeqOverLapDispatcher.overlap_stream = torch.npu.Stream()

        self.overlap_stream = MoEAlltoAllSeqOverLapDispatcher.overlap_stream

    def preprocess(self, indices: torch.Tensor, with_sync=True) -> torch.Tensor:
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
        num_local_tokens_per_expert = torch.histc(
            indices, bins=self.num_experts, min=0, max=self.num_experts
        )

        # num_local_tokens_per_expert: [num_experts]

        ep_size = self.ep_size
        if self.drop_and_pad:
            # Drop and pad the input to capacity.
            num_tokens = indices.numel()
            self.capacity = get_capacity(
                num_tokens=num_tokens,
                num_experts=self.num_experts,
                capacity_factor=self.config.moe_expert_capacity_factor,
            )
            self.num_out_tokens = self.capacity * self.num_experts
            num_tokens_per_local_expert = torch.full(
                (self.num_local_experts,), self.capacity * self.ep_size, dtype=torch.long
            )
            self.num_global_tokens_per_local_expert_cpu = torch.full(
                (self.num_experts * self.tp_ep_size,), self.capacity, dtype=torch.long
            )
            return num_tokens_per_local_expert
        elif self.config.moe_expert_capacity_factor is not None:
            # Token drop but no pad. A synchronization is needed before the first
            # permutation to get the `num_out_tokens` CPU value.
            self.num_out_tokens = num_local_tokens_per_expert.sum().to(
                torch.device("cpu"), non_blocking=True
            )
            self.cuda_sync_point = "before_permutation_1"
        else:
            # Dropless
            self.num_out_tokens = indices.numel()
            if self.ep_size > 1 or self.num_local_experts > 1:
                # Token dropless and enable ep. A synchronization is needed before expert parallel
                # AlltoAll communication to get the `input_splits` and `output_splits` CPU values.
                self.cuda_sync_point = "before_ep_alltoall"
            else:
                # Token dropless and no ep. A synchronization is needed to get the
                # `tokens_per_expert` CPU value.
                self.cuda_sync_point = "before_finish"

        if ep_size > 1:
            # ===================================================
            # Calculate input_splits, output_splits for alltoall-v.
            # ===================================================
            self.input_splits = (
                num_local_tokens_per_expert.reshape(ep_size, self.num_local_experts)
                .sum(axis=1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_global_tokens_per_expert = gather_from_sequence_parallel_region(
                num_local_tokens_per_expert, group=self.ep_group
            ).reshape(ep_size, self.num_experts)
            self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[0]:
                                                                                      self.local_expert_indices[-1] + 1]
            self.output_splits = (
                self.num_global_tokens_per_local_expert.sum(axis=-1)
                .to(torch.device("cpu"), non_blocking=True)
                .numpy()
            )
            num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(axis=0)
            # ===================================================
            # num_global_tokens_per_expert: [ep_size, num_experts]
            # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
            # num_tokens_per_local_expert: [num_local_experts]
            # ===================================================
        else:
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                -1, self.num_experts
            )
            num_tokens_per_local_expert = num_local_tokens_per_expert

        if self.num_local_experts > 1 and with_sync:
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

            # self.num_global_tokens_per_local_expert_cpu = (
            #     self.num_global_tokens_per_local_expert.view(-1, self.num_local_experts).to(
            #         torch.device("cpu"), non_blocking=True
            #     )
            # )
            # if not hasattr(self, "comm_stream"):
            #     self.comm_stream = torch.npu.Stream()
            # self.comm_stream.wait_stream(torch.npu.current_stream())

        return num_tokens_per_local_expert

    def routing(self, probs):
        seq_length, bsz = probs.shape[:2]
        probs = probs.view(-1, self.config.num_moe_experts)
        if self.config.is_fused:
            score_function = "sigmoid"
        else:
            score_function = "softmax"

        scores, routing_map, _, top_indices = topk_softmax_with_capacity(
            probs,
            self.config.moe_router_topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            group_topk=self.config.group_topk,
            num_groups=self.config.num_groups,
            expert_bias=self.config.expert_bias,
            scaling_factor=self.config.scaling_factor,
            score_function=score_function
        )
        self.top_indices = top_indices
        return scores, routing_map

    def preprocess_overlap(self, routing_map):
        num_tokens_per_local_expert = self.preprocess(routing_map)
        self.num_global_tokens_per_local_expert = self.num_global_tokens_per_local_expert
        self.input_splits = self.input_splits
        self.output_splits = self.output_splits
        self.num_out_tokens = self.num_out_tokens
        self.num_global_tokens_per_local_expert_cpu = self.num_global_tokens_per_local_expert_cpu
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
        self.routing_map = routing_map
        self.top_indices = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for routing map"

        # Permutation 1: input to AlltoAll input
        def alltoall_token_permutation1(hidden_states, routing_map):
            hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
            tokens_per_expert = self.preprocess(routing_map)
            if self.tp_ep_size > 1:
                hidden_states = all_to_all_sp2hp(hidden_states, group=self.tp_ep_group)
            self.hidden_shape_before_permute = hidden_states.shape

            if self.cuda_sync_point == "before_permutation_1":
                torch.npu.current_stream().synchronize()
            if not self.config.is_fused:
                permutated_local_input_tokens, reversed_local_input_permutation_mapping = permute(
                    hidden_states,
                    routing_map,
                    num_out_tokens=self.num_out_tokens,
                )
            else:
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
        if self.cuda_sync_point == "before_ep_alltoall":
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
                    self.global_input_tokens_local_experts_indices
                )

            # Perform tensor parallel AllGather on the hidden dimension to obtain the input tokens.
            # global_input_tokens: [SEQL, H/TP] -> [SEQL, H]
            if self.tp_ep_size > 1 and self.config.moe_grouped_gemm:
                global_input_tokens = all_gather_last_dim_from_tensor_parallel_region(
                    global_input_tokens, self.tp_ep_group
                )
            if self.cuda_sync_point == "before_finish":
                torch.npu.current_stream().synchronize()

            return global_input_tokens

        # token premute2 input
        global_input_tokens = alltoall_token_permutation2(global_input_tokens)

        return share_experts_output, global_input_tokens, tokens_per_expert

    def preprocess_and_permtute1(
            self,
            hidden_states: torch.Tensor,
            probs: torch.Tensor,
            routing_map: torch.Tensor,
            shared_experts=None,
            shared_experts_input: torch.Tensor = None
    ):
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.top_indices = routing_map
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert routing_map.dim() == 2, "Expected 2D tensor for routing map"

        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(routing_map, with_sync=False)
        self.hidden_shape_before_permute = hidden_states.shape

        if self.cuda_sync_point == "before_permutation_1":
            torch.npu.current_stream().synchronize()

        event = torch.npu.current_stream().record_event()
        self.perm1_finish_event = torch.npu.Event()
        with torch.npu.stream(self.overlap_stream):
            self.overlap_stream.wait_event(event)

            if shared_experts is not None:
                shared_output = shared_experts(shared_experts_input)
                self.cached_shared_expert_output = shared_output

            if not self.config.is_fused:
                hidden_states, self.reversed_local_input_permutation_mapping = permute(
                    hidden_states,
                    routing_map,
                    num_out_tokens=self.num_out_tokens,
                )
            else:
                hidden_states, self.reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                    tokens=hidden_states,
                    indices=self.top_indices,
                    num_out_tokens=self.num_out_tokens,
                )

            self.perm1_finish_event.record()

        # repeat interleve will launch a sync on current_stream.
        if self.num_local_experts > 1:
            self.cuda_sync_point = "no_sync"
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank, self.num_global_tokens_per_local_expert.ravel()
            )

        self.cached_permutated_local_input_tokens = hidden_states
        self.tokens_per_expert = tokens_per_expert

    def dispatch_alltoall(self):
        ep_group = self.ep_group

        # Perform expert parallel AlltoAll communication
        if self.cuda_sync_point == "before_ep_alltoall":
            torch.npu.current_stream().synchronize()

        torch.npu.current_stream().wait_event(self.perm1_finish_event)
        self.perm1_finish_event = None
        _, self.cached_global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            self.cached_permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            ep_group,
        )
        permute1_ep_all_to_all_handle.wait()
        self.cached_permutated_local_input_tokens.untyped_storage().resize_(0)
        self.cached_permutated_local_input_tokens = None

    def permute2(self):
        global_input_tokens = self.cached_global_input_tokens
        if self.num_local_experts > 1:
            global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
                self.cached_global_input_tokens,
                self.global_input_tokens_local_experts_indices
            )
            self.cached_global_input_tokens.untyped_storage().resize_(0)
            self.cached_global_input_tokens = None

        return global_input_tokens, self.tokens_per_expert

    def unpermute1(
            self,
            hidden_states: torch.Tensor
    ):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states,
                self.reversed_global_input_permutation_mapping
            )
        self.cached_global_output_tokens = hidden_states
        self.reversed_global_input_permutation_mapping = None

    def combine_alltoall(self):
        ep_group = self.ep_group
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, self.cached_local_output_tokens, handle = async_all_to_all(
            self.cached_global_output_tokens,
            self.input_splits,
            self.output_splits,
            ep_group
        )
        handle.wait()
        self.cached_global_output_tokens.untyped_storage().resize_(0)
        self.cached_global_output_tokens = None
        self.input_splits = None
        self.output_splits = None

    def unpermute2(self):
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=self.cached_local_output_tokens,
            sorted_indices=self.reversed_local_input_permutation_mapping.to(torch.int32),
            probs=self.probs,
            restore_shape=self.hidden_shape_before_permute
        )

        output = output.view(self.hidden_shape)

        self.probs = None
        self.reversed_local_input_permutation_mapping = None
        self.cached_local_output_tokens.untyped_storage().resize_(0)
        self.cached_local_output_tokens = None

        return output

    def token_unpermutation(
            self,
            hidden_states: torch.Tensor,
            bias: torch.Tensor = None
    ):
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
                hidden_states = reduce_scatter_last_dim_to_tensor_parallel_region(hidden_states, group=self.tp_ep_group)

            # Unpermutation 2: expert output to AlltoAll input
            if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
                hidden_states = torch_npu.npu_moe_token_unpermute(
                    hidden_states,
                    self.reversed_global_input_permutation_mapping
                )
                # hidden_states = sort_chunks_by_idxs(
                #     hidden_states,
                #     self.num_global_tokens_per_local_expert_cpu.T.ravel(),
                #     self.restore_output_by_local_experts,
                # )
            return hidden_states

        hidden_states = alltoall_token_unpermutation1(hidden_states)

        ep_group = self.ep_group
        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            self.input_splits,
            self.output_splits,
            ep_group
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        def alltoall_token_unpermutation2(permutated_local_input_tokens):
            # Unpermutation 1: AlltoAll output to output
            if self.config.is_fused:
                # permuted_probs = (self.probs.T.contiguous().masked_select(self.routing_map.T.contiguous())
                #                   .view(-1, self.config.moe_router_topk))
                output = torch_npu.npu_moe_token_unpermute(
                    permuted_tokens=permutated_local_input_tokens,
                    sorted_indices=self.reversed_local_input_permutation_mapping.to(torch.int32),
                    probs=self.probs,
                    restore_shape=self.hidden_shape_before_permute
                )
            else:
                output = unpermute(
                    permutated_local_input_tokens,
                    self.reversed_local_input_permutation_mapping,
                    probs=self.probs,
                    restore_shape=self.hidden_shape_before_permute,
                    routing_map=self.routing_map,
                )

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
