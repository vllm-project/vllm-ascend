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
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_ep_group

from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.comm_utils import (
    async_all_to_all, gather_from_sequence_parallel_region)
from vllm_ascend.utils import (AscendDeviceType, get_ascend_device_type,
                               is_hierarchical_communication_enabled)


@dataclass
class TokenDispatchResult:
    hidden_states: torch.Tensor
    group_list: torch.Tensor
    group_list_type: int
    dynamic_scale: torch.Tensor | None = field(default=None)
    topk_scales: torch.Tensor | None = field(default=None)
    context_metadata: dict = field(default_factory=dict)


@dataclass
class TokenCombineResult:
    routed_out: torch.Tensor


class MoETokenDispatcher(ABC):

    def __init__(self, **kwargs) -> None:
        """
        Initialize the MoE Token Dispatcher.
        """
        self.top_k = kwargs.get("top_k", 0)
        self.num_experts = kwargs.get("num_experts", 0)

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

    @abstractmethod
    def token_dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: Optional[torch.Tensor] = None,
        log2phy: Optional[torch.Tensor] = None,
        global_redundant_expert_num: int = 0,
        shared_experts: Optional[Any] = None,
        quantized_x_for_share: Optional[Any] = None,
        dynamic_scale_for_share: Optional[Any] = None,
        mc2_mask: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        with_quant: bool = False,
        dynamic_eplb: bool = False,
        pertoken_scale: Optional[torch.Tensor] = None,
    ) -> TokenDispatchResult:
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self,
                      hidden_states: torch.Tensor,
                      context_metadata: dict,
                      bias: torch.Tensor | None = None) -> TokenCombineResult:
        raise NotImplementedError("Combine function not implemented.")


class TokenDispatcherWithMC2(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_group = get_mc2_group().device_group
        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        self.ep_rank_id = get_mc2_group().rank_in_group
        self.ep_world_size = get_mc2_group().world_size
        self.enable_dispatch_v2 = hasattr(torch_npu,
                                          "npu_moe_distribute_dispatch_v2")
        self.need_extra_args = (
            get_ascend_device_type() == AscendDeviceType.A3)

        # NOTE: When in A2, setting the environment variables HCCL_INTRA_PCIE_ENABLE=1 and
        # HCCL_INTRA_ROCE_ENABLE=0 can reduce cross-machine communication traffic and significantly
        # improve communication performance.
        self.need_expert_scale = is_hierarchical_communication_enabled()
        self.with_quant = False

        # Here we need to calculate the global_bs = max_bs_per_rank * ep_world_size to execute
        # dispatch & combine operators with different input num_tokens per rank.
        vllm_config = get_current_vllm_config()
        scheduler_config = vllm_config.scheduler_config
        compilation_config = vllm_config.compilation_config
        speculative_config = vllm_config.speculative_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        uniform_decode_query_len = 1 if not speculative_config else \
            1 + speculative_config.num_speculative_tokens
        decode_max_num_seqs = getattr(scheduler_config, 'decode_max_num_seqs',
                                      0)
        max_num_reqs = max(scheduler_config.max_num_seqs, decode_max_num_seqs)
        if compilation_config.cudagraph_capture_sizes:
            max_num_tokens = compilation_config.max_cudagraph_capture_size
        else:
            max_num_tokens = min(max_num_reqs * uniform_decode_query_len, 512)
        num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
        self.global_bs = num_tokens_per_tp_rank * self.ep_world_size
        self.fused_global_bs = max_num_tokens * self.ep_world_size

    def get_dispatch_mc2_kwargs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor,
        mc2_mask: torch.Tensor,
        global_redundant_expert_num: int = 0,
    ):
        quant_mode = 2 if self.with_quant else 0
        self.moe_expert_num = len(expert_map) + global_redundant_expert_num
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
            "expert_token_nums_type": 0,
        }

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.need_extra_args:
            stage1_kwargs.update({
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.need_expert_scale:
            stage1_kwargs.update({
                "expert_scales":
                topk_weights.to(torch.float32),
            })

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None):
        self.with_quant = with_quant

        # Apply log2phy if needed
        if log2phy is not None:
            topk_ids = log2phy[topk_ids]

        kwargs_mc2 = self.get_dispatch_mc2_kwargs(hidden_states, topk_weights,
                                                  topk_ids, expert_map,
                                                  mc2_mask,
                                                  global_redundant_expert_num)
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            **kwargs_mc2
        ) if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
            **kwargs_mc2)
        # comm_stream.wait_stream(torch.npu.current_stream())
        expand_x, dynamic_scale, assist_info_for_combine, expert_token_nums, \
            ep_recv_counts, tp_recv_counts, expand_scales = output[0:7]

        context_metadata = {
            "topk_ids": topk_ids,
            "topk_weights": topk_weights,
            "expert_map": expert_map,
            "ep_recv_counts": ep_recv_counts,
            "tp_recv_counts": tp_recv_counts,
            "assist_info_for_combine": assist_info_for_combine,
            "shared_experts": shared_experts,
            "expand_scales": expand_scales
        }

        group_list_type = 0

        return TokenDispatchResult(hidden_states=expand_x,
                                   dynamic_scale=dynamic_scale,
                                   group_list=expert_token_nums,
                                   group_list_type=group_list_type,
                                   context_metadata=context_metadata)

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor,
                              context_metadata: dict):
        expert_map = context_metadata["expert_map"]
        topk_ids = context_metadata["topk_ids"]
        topk_weights = context_metadata["topk_weights"]
        ep_recv_counts = context_metadata["ep_recv_counts"]
        tp_recv_counts = context_metadata["tp_recv_counts"]
        assist_info_for_combine = context_metadata["assist_info_for_combine"]
        expand_scales = context_metadata["expand_scales"]

        assert expert_map is not None

        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": self.global_bs,
        }

        if self.with_quant:
            tp_recv_counts = torch.empty(1,
                                         dtype=torch.int32,
                                         device=hidden_states.device)

        stage3_kwargs = {
            "ep_send_counts": ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
            "expand_scales": expand_scales,
        }

        if self.enable_dispatch_v2:
            stage3_kwargs["assist_info_for_combine"] = assist_info_for_combine
        else:
            stage3_kwargs["expand_idx"] = assist_info_for_combine

        if self.need_extra_args:
            stage3_kwargs.update({
                "tp_send_counts": tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })

        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states,
                                                context_metadata)
        combined_output = torch_npu.npu_moe_distribute_combine_v2(**kwargs_mc2) \
            if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine(**kwargs_mc2)

        return TokenCombineResult(routed_out=combined_output)


class TokenDispatcherWithAllGather(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_router_weight_on_input = False
        self.max_num_tokens = kwargs.get("max_num_tokens")
        num_experts_local = kwargs.get("num_local_experts", 0)
        self.num_experts_local = num_experts_local.item() if torch.is_tensor(
            num_experts_local) else int(num_experts_local)
        self.original_shape = None
        self.with_quant = False

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None):
        self.with_quant = with_quant
        self.original_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        self.apply_router_weight_on_input = apply_router_weight_on_input
        if self.apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * \
                topk_weights.to(hidden_states.dtype)
        if expert_map is not None:
            global_num_experts = len(expert_map) + global_redundant_expert_num
            mask = (expert_map[topk_ids] != -1)
            topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group(
            ).rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local

        sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
            torch.ops._C_ascend.npu_moe_init_routing_custom(
                hidden_states,
                topk_ids,
                scale=pertoken_scale,
                active_num=num_tokens * self.top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=1
                if self.with_quant and pertoken_scale is None else -1,
            ))
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode
        context_metadata = {
            "topk_weights": topk_weights,
            "expanded_row_idx": expanded_row_idx
        }

        return TokenDispatchResult(
            hidden_states=sorted_hidden_states,
            dynamic_scale=pertoken_scale if self.with_quant else None,
            group_list=expert_tokens,
            group_list_type=group_list_type,
            context_metadata=context_metadata,
        )

    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert self.original_shape is not None
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=torch.abs(context_metadata["expanded_row_idx"]),
            probs=context_metadata["topk_weights"])
        if len(self.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(self.original_shape)

        # these values are no longer used, so they need to be set to None for memory release.
        return TokenCombineResult(routed_out=final_hidden_states)


class TokenDispatcherWithAll2AllV(MoETokenDispatcher):
    """
    The implementation of the AlltoAll-based token dispatcher, which handles token
    dispatching on the sequence level instead of token level. The core of this implementation
    lies in each device dispatching on the entire sequence, with the hidden state being partitioned.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.with_quant = False
        self.num_local_experts = kwargs.get("num_local_experts", 0)

        self.hidden_shape = None
        self.hidden_shape_before_permute = None

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

        # TODO: Try local_rank = ep_group.rank_in_group
        local_rank = torch.distributed.get_rank(group=self.ep_group)
        backend = self.ep_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[Any] = None,
                       quantized_x_for_share: Optional[Any] = None,
                       dynamic_scale_for_share: Optional[Any] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False,
                       dynamic_eplb: bool = False,
                       pertoken_scale: Optional[torch.Tensor] = None):
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape

        if log2phy is not None:
            topk_ids = log2phy[topk_ids]

        (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        ) = self._dispatch_preprocess(hidden_states, topk_ids)

        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens)
            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale, output_splits, input_splits, self.ep_group)
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens, output_splits, input_splits,
            self.ep_group)
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        # Postprocess
        global_input_tokens, dynamic_scale_final, reversed_global_input_permutation_mapping = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all,
            global_input_tokens_local_experts_indices)

        context_metadata = {
            "input_splits":
            input_splits,
            "output_splits":
            output_splits,
            "topk_weights":
            topk_weights,
            "reversed_local_input_permutation_mapping":
            reversed_local_input_permutation_mapping,
            "reversed_global_input_permutation_mapping":
            reversed_global_input_permutation_mapping
        }

        return TokenDispatchResult(
            hidden_states=global_input_tokens,
            dynamic_scale=dynamic_scale_final,
            group_list=tokens_per_expert,
            group_list_type=1,
            context_metadata=context_metadata,
        )

    def token_combine(self, hidden_states, context_metadata, bias=None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        # 1. Preprocess using metadata
        hidden_states = self._combine_preprocess(hidden_states,
                                                 context_metadata)

        # 2. AllToAll
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states,
            context_metadata["input_splits"],
            context_metadata["output_splits"],
            self.ep_group,
        )
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        # 3. Postprocess using metadata
        output = self._combine_postprocess(permutated_local_input_tokens,
                                           context_metadata)

        return TokenCombineResult(routed_out=output)

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        (
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        ) = self._preprocess(topk_ids)

        self.hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=self.num_out_tokens,
        )

        return (
            permutated_local_input_tokens,
            reversed_local_input_permutation_mapping,
            tokens_per_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        )

    def _preprocess(self, topk_ids: torch.Tensor):
        num_local_tokens_per_expert = torch.histc(topk_ids,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)

        ep_size = self.ep_size
        self.num_out_tokens = topk_ids.numel()

        input_splits = (num_local_tokens_per_expert.reshape(
            ep_size,
            self.num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                   non_blocking=True).numpy())

        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert,
            group=self.ep_group).reshape(ep_size, self.num_experts)
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        if num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum.")

        output_splits = (num_global_tokens_per_local_expert.sum(axis=-1).to(
            torch.device("cpu"), non_blocking=True).numpy())
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(
            axis=0)

        global_input_tokens_local_experts_indices = None
        if self.num_local_experts > 1:
            if num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                num_global_tokens_per_local_expert.ravel())
        else:
            torch.npu.synchronize()

        return (
            num_tokens_per_local_expert,
            input_splits,
            output_splits,
            num_global_tokens_per_local_expert,
            global_input_tokens_local_experts_indices,
        )

    def _dispatch_postprocess(self, global_input_tokens,
                              dynamic_scale_after_all2all,
                              global_input_tokens_local_experts_indices):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, dynamic_scale_after_all2all, None

        # Handle quantized case
        if self.with_quant:
            assert global_input_tokens_local_experts_indices is not None, \
                "global_input_tokens_local_experts_indices must be provided"
            dynamic_scale_after_all2all, _ = torch_npu.npu_moe_token_permute(
                dynamic_scale_after_all2all.unsqueeze(-1),
                global_input_tokens_local_experts_indices)
            dynamic_scale_after_all2all = dynamic_scale_after_all2all.squeeze(
                -1)

        # Non-quantized case
        global_input_tokens, reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens, global_input_tokens_local_experts_indices)
        return global_input_tokens, dynamic_scale_after_all2all, reversed_global_input_permutation_mapping

    def _combine_preprocess(self, hidden_states: torch.Tensor,
                            context_metadata: dict) -> torch.Tensor:
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            rev_global = context_metadata[
                "reversed_global_input_permutation_mapping"]
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, rev_global)
        return hidden_states

    def _combine_postprocess(self, permutated_local_input_tokens: torch.Tensor,
                             context_metadata: dict) -> torch.Tensor:
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=context_metadata[
                "reversed_local_input_permutation_mapping"].to(torch.int32),
            probs=context_metadata["topk_weights"],
            restore_shape=self.hidden_shape_before_permute,
        )
        output = output.view(self.hidden_shape)
        return output
