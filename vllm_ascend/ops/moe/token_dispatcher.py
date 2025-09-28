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
from typing import Any, Dict, Optional

import torch
import torch_npu
from vllm.distributed.parallel_state import get_ep_group

import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.moe.comm_utils import (
    async_all_to_all, gather_from_sequence_parallel_region)
from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version

_Dispatchers: Dict[str, Any] = {}


def _register_token_dispatcher(dispatcher: Any):
    _Dispatchers[dispatcher.__class__.__name__] = dispatcher


def get_token_dispatcher(name: str):
    return _Dispatchers.get(name)


def setup_token_dispatchers(ep_size: int, **kwargs):
    existing_dispatchers = set(_Dispatchers.keys())

    if ep_size == 1 and "TokenDispatcherWithAllGather" not in existing_dispatchers:
        _register_token_dispatcher(TokenDispatcherWithAllGather(**kwargs))
    elif envs_ascend.VLLM_ENABLE_FUSED_EXPERTS_ALLGATHER_EP and ep_size > 1 \
        and "TokenDispatcherWithAllGather" not in existing_dispatchers:
        _register_token_dispatcher(TokenDispatcherWithAllGather(**kwargs))
    elif ep_size < 16 and "TokenDispatcherWithAll2AllV" not in existing_dispatchers:
        _register_token_dispatcher(TokenDispatcherWithAll2AllV(**kwargs))
    elif ep_size >= 16:
        if "TokenDispatcherWithAll2AllV" not in existing_dispatchers:
            _register_token_dispatcher(TokenDispatcherWithAll2AllV(**kwargs))
        if "TokenDispatcherWithMC2" not in existing_dispatchers:
            _register_token_dispatcher(TokenDispatcherWithMC2(**kwargs))


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
    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[torch.Tensor] = None,
                       shared_gate_up: Optional[torch.Tensor] = None,
                       shared_dequant_scale: Optional[torch.Tensor] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
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
            get_ascend_soc_version() == AscendSocVersion.A3)

        # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
        self.a3_need_extra_args = \
            get_ascend_soc_version() == AscendSocVersion.A3
        self.output = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.shared_act = None
        self.topk_ids = None
        self.topk_weights = None
        self.shared_experts = None
        self.mc2_mask = None
        self.with_quant = False

    def get_dispatch_mc2_kwargs(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor,
        global_redundant_expert_num: int = 0,
    ):
        if self.with_quant:
            quant_mode = 2
            if (expert_map is not None):
                moe_expert_num = len(expert_map) + global_redundant_expert_num
            else:
                moe_expert_num = global_redundant_expert_num
        else:
            quant_mode = 0
            moe_expert_num = len(expert_map)
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
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
        if self.a3_need_extra_args and self.enable_dispatch_v2:
            stage1_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })

        kwargs_mc2.update(stage1_kwargs)
        return kwargs_mc2

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[torch.Tensor] = None,
                       shared_gate_up: Optional[torch.Tensor] = None,
                       shared_dequant_scale: Optional[torch.Tensor] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.expert_map = expert_map
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights
        self.shared_experts = shared_experts
        self.mc2_mask = mc2_mask

        kwargs_mc2 = self.get_dispatch_mc2_kwargs(hidden_states, topk_weights,
                                                  topk_ids, expert_map,
                                                  global_redundant_expert_num)
        self.output = torch_npu.npu_moe_distribute_dispatch_v2(
            **kwargs_mc2
        ) if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
            **kwargs_mc2)
        # comm_stream.wait_stream(torch.npu.current_stream())
        expand_x, dynamic_scale, self.assist_info_for_combine, \
            expert_token_nums, self.ep_recv_counts = self.output[0:5]

        if self.with_quant:
            if shared_experts is not None:
                shared_act_out = shared_experts.act_fn(
                    (shared_gate_up, shared_dequant_scale))
                self.shared_act, self.swiglu_out_scale = \
                    shared_act_out[0], shared_act_out[1]

        else:
            if shared_experts is not None:
                shared_gate_up, _ = shared_experts.gate_up_proj(hidden_states)
                self.shared_act = shared_experts.act_fn(shared_gate_up)
        group_list_type = 1
        return {
            "group_list_type": group_list_type,
            "hidden_states": expand_x,
            "group_list": expert_token_nums,
            "dynamic_scale": dynamic_scale,
        }

    def get_combine_mc_kwargs(self, hidden_states: torch.Tensor):
        assert self.expert_map is not None
        assert self.topk_weights is not None
        assert self.topk_ids is not None
        assert self.output is not None
        moe_expert_num = len(self.expert_map)
        # moeCombine
        kwargs_mc2 = {
            "expand_x": hidden_states,
            "expert_ids": self.topk_ids,
            "expert_scales": self.topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
        }
        if self.with_quant:
            tp_recv_counts = torch.empty(1,
                                         dtype=torch.int32,
                                         device=hidden_states.device)
        else:
            tp_recv_counts = self.output[5]
        stage3_kwargs = {
            "ep_send_counts": self.ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }
        if self.enable_dispatch_v2:
            stage3_kwargs.update({
                "assist_info_for_combine":
                self.assist_info_for_combine,
            })
        else:
            stage3_kwargs.update({
                "expand_idx": self.assist_info_for_combine,
            })
        if self.need_extra_args:
            stage3_kwargs.update({
                "tp_send_counts": tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.a3_need_extra_args and self.enable_dispatch_v2:
            stage3_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })
        kwargs_mc2.update(stage3_kwargs)
        return kwargs_mc2

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        kwargs_mc2 = self.get_combine_mc_kwargs(hidden_states)
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(
            **kwargs_mc2
        ) if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine(
            **kwargs_mc2)
        if self.shared_experts is None:
            return hidden_states
        else:
            if self.with_quant:
                shared_hidden_states, _ = self.shared_experts.down_proj(
                    (self.shared_act, self.swiglu_out_scale))
            else:
                shared_hidden_states, _ = self.shared_experts.down_proj(
                    self.shared_act)
            return hidden_states, shared_hidden_states


class TokenDispatcherWithAllGather(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_router_weight_on_input = False
        self.max_num_tokens = kwargs.get("max_num_tokens")
        self.num_experts_local = kwargs.get("num_local_experts", 0)
        self.sorted_weights = None
        self.expanded_row_idx = None
        self.sorted_token_indices = None
        self.original_shape = None
        self.mask = None
        self.expert_map = None
        self.topk_weights = None
        self.topk_ids = None
        self.with_quant = False

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[torch.Tensor] = None,
                       shared_gate_up: Optional[torch.Tensor] = None,
                       shared_dequant_scale: Optional[torch.Tensor] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.original_shape = hidden_states.shape

        num_tokens = hidden_states.shape[:-1].numel()
        self.expert_map = expert_map
        self.topk_weights = topk_weights
        self.topk_ids = topk_ids
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
            global_num_experts = len(expert_map)
            mask = (expert_map[topk_ids] != -1)
            self.topk_weights = topk_weights * mask
            first_expert_idx = get_ep_group(
            ).rank_in_group * self.num_experts_local
            last_expert_idx = first_expert_idx + self.num_experts_local
        else:
            first_expert_idx = 0
            last_expert_idx = self.num_experts_local
            global_num_experts = self.num_experts_local

        sorted_hidden_states, self.expanded_row_idx, expert_tokens, pertoken_scale = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * self.top_k,
                expert_num=global_num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=1 if self.with_quant else -1,
            ))
        expert_tokens = expert_tokens.to(torch.int64)
        group_list_type = 1  # `count` mode
        return {
            "group_list_type": group_list_type,
            "hidden_states": sorted_hidden_states,
            "group_list": expert_tokens,
            "dynamic_scale": pertoken_scale if self.with_quant else None,
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        assert self.original_shape is not None
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=self.expanded_row_idx,
            probs=self.topk_weights)
        if len(self.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(self.original_shape)
        return final_hidden_states


# mypy: disable-error-code="override"
class UnquantizedTokenDispatcherWithFusedExpertsMoge(MoETokenDispatcher):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_router_weight_on_input = False
        self.local_ep = 1
        self.local_num_experts = self.num_experts // self.local_ep
        self.local_num_group = self.top_k // self.local_ep
        self.bsz = None

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[torch.Tensor] = None,
                       shared_gate_up: Optional[torch.Tensor] = None,
                       shared_dequant_scale: Optional[torch.Tensor] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
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

        self.bsz, _ = hidden_states.shape
        flatten_topk_ids = topk_ids.view(-1)
        self.sorted_topk_ids = torch.argsort(flatten_topk_ids.float())
        self.sorted_topk_ids = self.sorted_topk_ids.to(torch.int32)
        sorted_hidden_states = hidden_states.index_select(
            0, self.sorted_topk_ids // self.local_num_group)

        experts_id = torch.arange(0,
                                  self.local_num_experts,
                                  dtype=topk_ids.dtype,
                                  device=topk_ids.device)
        num_tokens_per_expert = (
            flatten_topk_ids.unsqueeze(-1) == experts_id).to(
                torch.float32).sum(0)
        topk_scales = topk_weights.view(-1).index_select(
            0, self.sorted_topk_ids).unsqueeze(-1)
        group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)
        group_list_type = 0
        return {
            "group_list_type": group_list_type,
            "hidden_states": sorted_hidden_states,
            "group_list": group_list,
            "topk_scales": topk_scales,
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        unsorted_topk_ids = torch.argsort(self.sorted_topk_ids.float()).to(
            torch.int32)
        unsorted_hidden_states = hidden_states.index_select(
            0, unsorted_topk_ids)
        final_hidden_states = unsorted_hidden_states.reshape(
            self.bsz, self.top_k // self.local_ep, -1).sum(1)
        return final_hidden_states


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
        self.num_global_redundant_experts = kwargs.get(
            "num_global_redundant_experts", 0)
        self.num_experts = self.num_experts + self.num_global_redundant_experts

        self.hidden_shape = None
        self.topk_weights = None
        self.input_splits = None
        self.output_splits = None
        self.hidden_shape_before_permute = None

        # [tp_ep_size * ep_size, num_local_experts]. Represents the number of tokens sent
        # to each local expert by all ranks.
        self.num_global_tokens_per_local_expert = None

        # cached intermediate tensors.
        self.tokens_per_expert = None
        self.global_input_tokens_local_experts_indices = None

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

    def token_dispatch(self,
                       hidden_states: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       expert_map: Optional[torch.Tensor] = None,
                       log2phy: Optional[torch.Tensor] = None,
                       global_redundant_expert_num: int = 0,
                       shared_experts: Optional[torch.Tensor] = None,
                       shared_gate_up: Optional[torch.Tensor] = None,
                       shared_dequant_scale: Optional[torch.Tensor] = None,
                       mc2_mask: Optional[torch.Tensor] = None,
                       apply_router_weight_on_input: bool = False,
                       with_quant: bool = False):
        self.with_quant = with_quant
        self.hidden_shape = hidden_states.shape
        self.topk_weights = topk_weights
        assert topk_weights.dim() == 2, "Expected 2D tensor for topk_weights"
        assert topk_ids.dim() == 2, "Expected 2D tensor for routing map"

        if log2phy is not None:
            topk_ids = log2phy[topk_ids]

        permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert = self._dispatch_preprocess(
            hidden_states, topk_ids)
        self.reversed_local_input_permutation_mapping = reversed_local_input_permutation_mapping

        dynamic_scale_after_all2all = None
        if self.with_quant:
            permutated_local_input_tokens, dynamic_scale = torch_npu.npu_dynamic_quant(
                permutated_local_input_tokens)

            _, dynamic_scale_after_all2all, permute2_ep_all_to_all_handle = async_all_to_all(
                dynamic_scale,
                self.output_splits,
                self.input_splits,
                self.ep_group,
            )
            permute2_ep_all_to_all_handle.wait()
            dynamic_scale.untyped_storage().resize_(0)

        _, global_input_tokens, permute1_ep_all_to_all_handle = async_all_to_all(
            permutated_local_input_tokens,
            self.output_splits,
            self.input_splits,
            self.ep_group,
        )
        permute1_ep_all_to_all_handle.wait()
        permutated_local_input_tokens.untyped_storage().resize_(0)

        global_input_tokens, dynamic_scale = self._dispatch_postprocess(
            global_input_tokens, dynamic_scale_after_all2all)
        return {
            "hidden_states": global_input_tokens,
            "group_list": tokens_per_expert,
            "dynamic_scale": dynamic_scale,
            "group_list_type": 1
        }

    def token_combine(self,
                      hidden_states: torch.Tensor,
                      bias: torch.Tensor = None):
        assert bias is None, "Bias is not supported in MoEAlltoAllvTokenDispatcher."

        hidden_states = self._combine_preprocess(hidden_states)

        # Perform expert parallel AlltoAll communication
        # hidden_states: [SEQL, H] -> [SEQL, H/TP]
        _, permutated_local_input_tokens, handle = async_all_to_all(
            hidden_states, self.input_splits, self.output_splits,
            self.ep_group)
        handle.wait()
        hidden_states.untyped_storage().resize_(0)

        output = self._combine_postprocess(permutated_local_input_tokens)

        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

        return output

    def _dispatch_preprocess(self, hidden_states, topk_ids):
        assert self.hidden_shape is not None
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self._preprocess(topk_ids)

        self.hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, reversed_local_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            tokens=hidden_states,
            indices=topk_ids,
            num_out_tokens=self.num_out_tokens,
        )
        return permutated_local_input_tokens, reversed_local_input_permutation_mapping, tokens_per_expert

    def _preprocess(self, topk_ids: torch.Tensor) -> torch.Tensor:
        num_local_tokens_per_expert = torch.histc(topk_ids,
                                                  bins=self.num_experts,
                                                  min=0,
                                                  max=self.num_experts)

        ep_size = self.ep_size

        # Dropless
        self.num_out_tokens = topk_ids.numel()

        # ===================================================
        # Calculate input_splits, output_splits for alltoall-v.
        # ===================================================
        self.input_splits = (num_local_tokens_per_expert.reshape(
            ep_size,
            self.num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                   non_blocking=True).numpy())
        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert,
            group=self.ep_group).reshape(ep_size, self.num_experts)
        self.num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, self.local_expert_indices[
            0]:self.local_expert_indices[-1] + 1]
        if self.num_global_tokens_per_local_expert is None:
            raise ValueError(
                "num_global_tokens_per_local_expert must be set before sum.")
        self.output_splits = (self.num_global_tokens_per_local_expert.sum(
            axis=-1).to(torch.device("cpu"), non_blocking=True).numpy())
        num_tokens_per_local_expert = self.num_global_tokens_per_local_expert.sum(
            axis=0)
        # ===================================================
        # num_global_tokens_per_expert: [ep_size, num_experts]
        # num_global_tokens_per_local_expert: [ep_size, num_local_experts]
        # num_tokens_per_local_expert: [num_local_experts]
        # ===================================================

        if self.num_local_experts > 1:
            if self.num_global_tokens_per_local_expert is None:
                raise ValueError(
                    "num_global_tokens_per_local_expert must be set before operations."
                )
            self.global_input_tokens_local_experts_indices = torch.repeat_interleave(
                self.expert_ids_per_ep_rank,
                self.num_global_tokens_per_local_expert.ravel())

        return num_tokens_per_local_expert

    def _dispatch_postprocess(self, global_input_tokens, dynamic_scale=None):
        # Early return if no local experts or no tokens
        if self.num_local_experts <= 1:
            return global_input_tokens, None

        # Handle quantized case
        if self.with_quant:
            assert self.global_input_tokens_local_experts_indices is not None, \
            "global_input_tokens_local_experts_indices must be initialized before calling _dispatch_postprocess"
            expert_idx_2d = self.global_input_tokens_local_experts_indices.unsqueeze(
                -1)
            active_num = self.global_input_tokens_local_experts_indices.numel()

            # Handle case with no active tokens
            if active_num <= 0:
                self.reversed_global_input_permutation_mapping = self.global_input_tokens_local_experts_indices
                return global_input_tokens, dynamic_scale

            # Process with active tokens
            global_input_tokens, self.reversed_global_input_permutation_mapping, _, expanded_scale = torch_npu.npu_moe_init_routing_v2(
                global_input_tokens,
                expert_idx_2d,
                scale=dynamic_scale,
                active_num=active_num,
                expert_capacity=0,
                expert_num=self.num_local_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.num_local_experts],
                quant_mode=-1,
                row_idx_type=0)
            return global_input_tokens, expanded_scale

        # Handle non-quantized case
        global_input_tokens, self.reversed_global_input_permutation_mapping = torch_npu.npu_moe_token_permute(
            global_input_tokens,
            self.global_input_tokens_local_experts_indices)
        return global_input_tokens, None

    def _combine_preprocess(self, hidden_states):
        # Unpermutation 2: expert output to AlltoAll input
        if hidden_states.shape[0] > 0 and self.num_local_experts > 1:
            hidden_states = torch_npu.npu_moe_token_unpermute(
                hidden_states, self.reversed_global_input_permutation_mapping)

        return hidden_states

    def _combine_postprocess(self, permutated_local_input_tokens):
        # Unpermutation 1: AlltoAll output to output
        output = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=permutated_local_input_tokens,
            sorted_indices=self.reversed_local_input_permutation_mapping.to(
                torch.int32),
            probs=self.topk_weights,
            restore_shape=self.hidden_shape_before_permute)

        # Reshape the output tensor
        output = output.view(self.hidden_shape)
        return output
