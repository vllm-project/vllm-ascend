from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.distributed.communication_op import \
    data_parallel_reduce_scatter
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

    @abstractmethod
    def prepare(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the MoE communication method.

        This method is called before quant_method.apply to prepare the
        communication method. It can be used to initialize any necessary
        resources or configurations.
        """
        pass

    @abstractmethod
    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """Finalize the MoE communication method.

        This method is called after quant_method.apply to finalize the
        communication method. It can be used to clean up any resources or
        configurations.
        """
        pass

    @abstractmethod
    def permute(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pre-process before MLP.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (num_tokens, hidden_size)
            topk_ids (torch.Tensor): Tensor of shape (num_tokens, top_k_num)
            topk_weights (torch.Tensor): Tensor of shape (num_tokens, top_k_num)
            expert_map (torch.Tensor): Tensor of shape (global_num_experts, )
                Mapping from global expert IDs to local expert IDs.
            num_experts (int): Number of local experts (experts on this device).

        Returns:
            tuple[torch.Tensor, torch.Tensor, int]: Return a tuple containing:
                - permuted_hidden_states (torch.Tensor): Tensor of shape
                    (num_tokens * top_k_num, hidden_size) after permuting
                    hidden_states based on topk_ids.
                - expert_tokens (torch.Tensor): Tensor of shape (num_experts, )
                    Number of tokens assigned to each expert.
                - group_list_type (int): Type of group list, 0 for `cumsum`
                    and 1 for `count`. This is mainly for `npu_grouped_matmul`
                    to determine how to handle the output.
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def unpermute(self, mlp_output: torch.Tensor,
                  hidden_states: torch.Tensor) -> None:
        """Post-process after MLP.

        Args:
            mlp_output (torch.Tensor): Tensor of shape
                (num_tokens * top_k_num, hidden_size) after MLP.
            hidden_states (torch.Tensor): Tensor of shape
                (num_tokens, hidden_size) to be updated with the final output.
        """
        pass


class DummyCommImpl(MoECommMethod):

    def prepare(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dummy prepare method that does nothing."""
        return hidden_states, router_logits

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """Dummy finalize method that does nothing."""
        return hidden_states

    def permute(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Dummy implementation, make sure the output shapes are correct."""
        top_k_num = topk_ids.shape[1]
        permuted_hidden_states = hidden_states.repeat_interleave(top_k_num,
                                                                 dim=0)
        expert_tokens = torch.zeros((num_experts, ),
                                    dtype=torch.int64,
                                    device=hidden_states.device)
        group_list_type = 0
        return permuted_hidden_states, expert_tokens, group_list_type

    def unpermute(self, mlp_output: torch.Tensor,
                  hidden_states: torch.Tensor) -> None:
        """Dummy implementation that does nothing."""
        pass


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def prepare(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """When DP size > 1, pad the hidden states and router logits for communication."""
        if self.moe_config.dp_size > 1:
            forward_context = get_forward_context()
            max_tokens_across_dp = forward_context.max_tokens_across_dp

            self.num_tokens = hidden_states.shape[0]
            pad_size = max_tokens_across_dp - self.num_tokens
            if pad_size > 0:
                hidden_states = nn.functional.pad(hidden_states,
                                                  (0, 0, 0, pad_size))
                router_logits = nn.functional.pad(router_logits,
                                                  (0, 0, 0, pad_size))

            hidden_states = self.moe_config.dp_group.all_gather(
                hidden_states, 0)
            router_logits = self.moe_config.dp_group.all_gather(
                router_logits, 0)

        return hidden_states, router_logits

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """When DP size > 1, reduce-scatter the hidden states to get the final output.

        When TP size > 1, all-reduce the hidden states to get the final output.
        """
        if self.moe_config.dp_size > 1:
            hidden_states = data_parallel_reduce_scatter(hidden_states, dim=0)
            hidden_states = hidden_states[:self.num_tokens]

        if reduce_results and (self.moe_config.tp_size > 1
                               or self.moe_config.ep_size > 1):
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)

        return hidden_states

    def permute(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,  # noqa: F841
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_tokens = hidden_states.shape[0]

        self.topk_weights = topk_weights
        self.topk_ids = topk_ids

        first_expert_idx = 0
        if expert_map is not None:
            # FIXME: npu_grouped_matmul output random values at [num_valid_tokens:, ...]
            # So we need to filter out invalid tokens by zeroing their weights.
            # This is a workaround and should be removed after the issue is fixed
            mask = expert_map[topk_ids] != -1
            # NOTE: This is equivalent to self.topk_weights[~mask] = 0.0,
            # but ~mask will dispatch to aclnnNonzeroV2, which is not supported in ACL Graph
            self.topk_weights = torch.where(mask, topk_weights, 0.0)

            first_expert_idx = self.moe_config.ep_rank * num_experts
        last_expert_idx = first_expert_idx + num_experts

        permuted_hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * self.moe_config.experts_per_token,
                expert_num=self.moe_config.num_experts,
                expert_tokens_num_type=1,  # Only support `count` mode now
                expert_tokens_num_flag=True,  # Output `expert_tokens`
                active_expert_range=[first_expert_idx, last_expert_idx],
                quant_mode=-1,
            ))
        self.expanded_row_idx = expanded_row_idx
        permuted_hidden_states = permuted_hidden_states

        group_list_type = 1  # `count` mode

        return permuted_hidden_states, expert_tokens, group_list_type

    def unpermute(self, mlp_output: torch.Tensor,
                  hidden_states: torch.Tensor) -> None:
        hidden_states[:] = torch_npu.npu_moe_token_unpermute(
            permuted_tokens=mlp_output,
            sorted_indices=self.expanded_row_idx,
            probs=self.topk_weights)


class NativeAllGatherCommImpl(AllGatherCommImpl):
    """This implementation should be compatible with all scenarios.

    Note that this implementation purely consists of native PyTorch ops
    and does not use any NPU-specific ops. So the performance may not be optimal.
    But it is a good fallback for scenarios where NPU-specific ops are not available.
    """

    def permute(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_tokens = hidden_states.shape[0]

        # Generate token indices and flatten
        token_indices = torch.arange(num_tokens,
                                     device=hidden_states.device,
                                     dtype=torch.int64)
        token_indices = (token_indices.unsqueeze(1).expand(
            -1, self.moe_config.experts_per_token).reshape(-1))

        # Flatten token-to-expert mappings and map to local experts
        weights_flat = topk_weights.view(-1)
        experts_flat = topk_ids.view(-1)
        local_experts_flat = (expert_map[experts_flat]
                              if expert_map is not None else experts_flat)

        # Filter valid token-expert pairs
        mask = local_experts_flat != -1
        # FIXME: npu_grouped_matmul output random values at [num_valid_tokens:, ...]
        # So we need to filter out invalid tokens by zeroing their weights.
        # This is a workaround and should be removed after the issue is fixed
        filtered_weights = torch.where(mask, weights_flat,
                                       torch.zeros_like(weights_flat)).to(
                                           topk_weights.dtype)
        filtered_experts = torch.where(
            mask,
            local_experts_flat,
            torch.full_like(local_experts_flat, num_experts),
        ).to(topk_ids.dtype)

        # Sort by local expert IDs
        sort_indices = torch.argsort(filtered_experts.view(torch.float32))
        self.sorted_token_indices = token_indices[sort_indices]
        self.sorted_weights = filtered_weights[sort_indices]

        # Compute token counts with minlength of num_experts
        # This is equivalent to but faster than:
        # >>> token_counts = torch.bincount(filtered_experts, minlength=num_experts)[:-1]
        token_counts = torch.zeros(num_experts + 1,
                                   device=hidden_states.device,
                                   dtype=torch.int64)
        ones = torch.ones_like(filtered_experts, dtype=torch.int64)
        token_counts.scatter_add_(0, filtered_experts.to(torch.int64), ones)
        expert_tokens = token_counts[:num_experts]

        # Rearrange hidden_states
        permuted_hidden_states = hidden_states[self.sorted_token_indices]

        group_list_type = 1  # `count` mode

        return permuted_hidden_states, expert_tokens, group_list_type

    def unpermute(self, mlp_output: torch.Tensor,
                  hidden_states: torch.Tensor) -> None:
        mlp_output = mlp_output * self.sorted_weights.unsqueeze(1)

        final_hidden_states = torch.zeros_like(hidden_states)
        final_hidden_states.index_add_(0, self.sorted_token_indices,
                                       mlp_output)

        hidden_states[:] = final_hidden_states


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.
    
    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config: Optional[FusedMoEConfig]):
        super().__init__(moe_config)

        # NOTE: We do not need to use mc2_group's rank and world size
        # because ep_group and mc2_group basically have the same init params.
        # We only init another group because of the restriction of MC2:
        # "No other groups can be used in the same process as the MC2 group."
        self.mc2_comm_name = get_mc2_group().device_group._get_backend(
            torch.device("npu")).get_hccl_comm_name(self.moe_config.ep_rank)

        # Feature flags
        self.enable_dispatch_v2 = hasattr(torch_npu,
                                          "npu_moe_distribute_dispatch_v2")
        self.is_ascend_a3 = get_ascend_soc_version() == AscendSocVersion.A3
        self.need_extra_args = self.is_ascend_a3
        self._restore_tp_across_dp()

    def _restore_tp_across_dp(self):
        # NOTE: Since vLLM flatten tp across dp, we need to restore the original
        # tp_size and tp_rank.
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

    def prepare(
            self, hidden_states: torch.Tensor,
            router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The target_pad_length is calculated in forward_context, here we pad the
        hidden states and router logits. And if TP size > 1, we also need to split
        the tensors accordingly.
        """
        self.num_tokens, _ = hidden_states.shape
        forward_context = get_forward_context()
        self.mc2_mask = forward_context.mc2_mask
        target_pad_length = forward_context.padded_num_tokens
        pad_size = target_pad_length - self.num_tokens

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
            split_mc2_mask = torch.tensor_split(self.mc2_mask,
                                                self.tp_size,
                                                dim=0)
            self.split_hidden_states = split_hidden_states

            hidden_states = split_hidden_states[self.tp_rank]
            router_logits = split_router_logits[self.tp_rank]
            self.mc2_mask = split_mc2_mask[self.tp_rank]

        return hidden_states, router_logits

    def finalize(self, hidden_states: torch.Tensor,
                 reduce_results: bool) -> torch.Tensor:
        """If TP size > 1, all-gather the hidden states to get the final output.
        
        Also, unpad the hidden states if needed.
        """
        if self.tp_size > 1:
            dist.all_gather(list(self.split_hidden_states), hidden_states,
                            self.moe_config.tp_group.device_group)
            hidden_states = torch.cat(self.split_hidden_states, dim=0)

        if self.num_tokens < hidden_states.shape[0]:
            hidden_states = hidden_states[:self.num_tokens]

        return hidden_states

    def permute(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Store tensors needed for post_process
        self.topk_ids = topk_ids
        self.topk_weights = topk_weights.to(torch.float32)

        dispatch_kwargs = {
            "x": hidden_states,
            "expert_ids": self.topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_config.num_experts,
            "global_bs": 0,
            "scales": None,
            "quant_mode": 0,
            "group_ep": self.mc2_comm_name,
            "ep_world_size": self.moe_config.ep_size,
            "ep_rank_id": self.moe_config.ep_rank,
        }

        if self.need_extra_args:
            dispatch_kwargs.update({
                "group_tp": self.mc2_comm_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.is_ascend_a3 and self.enable_dispatch_v2:
            dispatch_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })

        dispatch = torch_npu.npu_moe_distribute_dispatch_v2 if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch

        (
            permuted_hidden_states,
            _,  # dynamic_scale is not used
            self.assist_info_for_combine,
            expert_tokens,
            self.ep_recv_counts,
            self.tp_recv_counts,
        ) = dispatch(**dispatch_kwargs)[:6]

        group_list_type = 1

        return permuted_hidden_states, expert_tokens, group_list_type

    def unpermute(self, mlp_output: torch.Tensor,
                  hidden_states: torch.Tensor) -> None:
        combine_kwargs = {
            "expand_x": mlp_output,
            "expert_ids": self.topk_ids,
            "expert_scales": self.topk_weights,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_config.num_experts,
            "global_bs": 0,
            "ep_send_counts": self.ep_recv_counts,
            "group_ep": self.mc2_comm_name,
            "ep_world_size": self.moe_config.ep_size,
            "ep_rank_id": self.moe_config.ep_rank,
        }

        if self.enable_dispatch_v2:
            combine_kwargs[
                "assist_info_for_combine"] = self.assist_info_for_combine
        else:
            combine_kwargs["expand_idx"] = self.assist_info_for_combine

        if self.need_extra_args:
            combine_kwargs.update({
                "tp_send_counts": self.tp_recv_counts,
                "group_tp": self.mc2_comm_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.is_ascend_a3 and self.enable_dispatch_v2:
            combine_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })

        combine = torch_npu.npu_moe_distribute_combine_v2 if self.enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine

        hidden_states[:] = combine(**combine_kwargs)
