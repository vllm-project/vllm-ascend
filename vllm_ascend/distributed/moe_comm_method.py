from abc import ABC, abstractmethod

import torch
import torch_npu
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import direct_register_custom_op

from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        top_k_num: int,
        global_num_experts: int,
    ):
        self.device = device
        self.dtype = dtype
        self.top_k_num = top_k_num
        self.global_num_experts = global_num_experts

    @abstractmethod
    def _pre_process(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Pre-process before MLP."""
        pass

    @abstractmethod
    def _post_process(self, mlp_output: torch.Tensor,
                      hidden_states: torch.Tensor) -> None:
        """Post-process after MLP."""
        pass


class DummyCommImpl(MoECommMethod):

    def _pre_process(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        return moe_comm_pre_process_fake(hidden_states, topk_ids, topk_weights,
                                         expert_map, num_experts)

    def _post_process(self, mlp_output: torch.Tensor,
                      hidden_states: torch.Tensor) -> None:
        """Dummy implementation that does nothing."""
        pass


class AllGatherCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.

    Note that this implementation purely consists of native PyTorch ops
    and does not use any NPU-specific ops. So the performance may not be optimal.
    But it is a good fallback for scenarios where NPU-specific ops are not available.
    """

    def _pre_process(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        print("Using AllGatherCommImpl for MoE communication.")
        num_tokens = hidden_states.shape[0]

        # Generate token indices and flatten
        token_indices = torch.arange(num_tokens,
                                     device=self.device,
                                     dtype=torch.int64)
        token_indices = (token_indices.unsqueeze(1).expand(
            -1, self.top_k_num).reshape(-1))

        # Flatten token-to-expert mappings and map to local experts
        weights_flat = topk_weights.view(-1)
        experts_flat = topk_ids.view(-1)
        local_experts_flat = (expert_map[experts_flat]
                              if expert_map is not None else experts_flat)

        # Filter valid token-expert pairs
        mask = local_experts_flat != -1
        filtered_weights = torch.where(mask, weights_flat,
                                       torch.zeros_like(weights_flat)).to(
                                           self.dtype)
        filtered_experts = torch.where(
            mask,
            local_experts_flat,
            torch.full_like(local_experts_flat, num_experts),
        ).to(topk_ids.dtype)
        self.mask = mask

        # Sort by local expert IDs
        sort_indices = torch.argsort(filtered_experts.view(torch.float32))
        self.sorted_token_indices = token_indices[sort_indices]
        self.sorted_weights = filtered_weights[sort_indices]

        # Compute token counts with minlength of num_experts
        # This is equivalent to but faster than:
        # >>> token_counts = torch.bincount(filtered_experts, minlength=num_experts)[:-1]
        token_counts = torch.zeros(num_experts + 1,
                                   device=self.device,
                                   dtype=torch.int64)
        ones = torch.ones_like(filtered_experts, dtype=torch.int64)
        token_counts.scatter_add_(0, filtered_experts.to(torch.int64), ones)
        token_counts = token_counts[:num_experts]
        expert_tokens = torch.cumsum(token_counts, dim=0, dtype=torch.int64)

        # Rearrange hidden_states
        permuted_hidden_states = hidden_states[self.sorted_token_indices]

        group_list_type = 0

        return permuted_hidden_states, expert_tokens, group_list_type

    def _post_process(self, mlp_output: torch.Tensor,
                      hidden_states: torch.Tensor) -> None:
        weighted_down_out = mlp_output * self.sorted_weights.unsqueeze(1)

        final_hidden_states = torch.zeros_like(hidden_states)

        # TODO: npu_grouped_matmul output random values at [num_valid_tokens:, ...]
        # This created multiple NaN and index_add_ will mix them up which harms accuracy
        # remove this mask and filter after it being fixed
        num_valid_tokens = self.mask.sum()
        valid_token_mask = (torch.arange(
            0, self.sorted_token_indices.shape[0],
            device=self.device).unsqueeze(1) < num_valid_tokens)
        valid_output = torch.where(valid_token_mask, weighted_down_out,
                                   torch.zeros_like(weighted_down_out)).to(
                                       self.dtype)
        final_hidden_states.index_add_(0, self.sorted_token_indices,
                                       valid_output)

        hidden_states[:] = final_hidden_states


class AllReduceCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=False`.
    2. If `npu_moe_init_routing_v2` is available, we will support `enable_expert_parallel=True`,
       and this implementation will become the default one, changing the name to `AllGather` at
       the same time.
    """

    def _pre_process(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,  # noqa: F841
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        print("Using AllReduceCommImpl for MoE communication.")
        num_tokens = hidden_states.shape[0]

        self.topk_weights = topk_weights
        self.topk_ids = topk_ids

        # 1. Prepare row indices for routing
        row_idx_len = num_tokens * self.top_k_num
        row_idx = torch.arange(row_idx_len,
                               dtype=torch.int32,
                               device=self.device)
        row_idx = row_idx.view(self.top_k_num, -1).permute(1, 0).contiguous()

        # 2. Initial routing to expand tokens and experts
        permuted_hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            ))
        # NOTE: Currently, V2 produces incorrect accuracy and weaker performance than V1
        # first_expert_idx = 0
        # if expert_map is not None:
        #     first_expert_idx = torch.nonzero(expert_map != -1, as_tuple=False)[0].item()
        # last_expert_idx = first_expert_idx + num_experts
        # permuted_hidden_states, expanded_row_idx, expert_tokens, _ = (
        #     torch_npu.npu_moe_init_routing_v2(
        #         hidden_states,
        #         topk_ids,
        #         active_num=num_tokens * self.top_k_num,
        #         expert_num=self.global_num_experts,
        #         expert_tokens_num_type=1,  # Only support `count` mode now
        #         expert_tokens_num_flag=True,  # Output `expert_tokens`
        #         active_expert_range=[first_expert_idx, last_expert_idx],
        #         quant_mode=-1,
        #     )
        # )
        self.expanded_row_idx = expanded_row_idx
        permuted_hidden_states = permuted_hidden_states

        # 3. Compute expert tokens
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts).to(torch.int64)
        # NOTE: This is also for npu_moe_init_routing_v2
        # expert_tokens = torch.cumsum(expert_tokens, 0)

        group_list_type = 0

        return permuted_hidden_states, expert_tokens, group_list_type

    def _post_process(self, mlp_output: torch.Tensor,
                      hidden_states: torch.Tensor) -> None:
        hidden_states[:] = torch_npu.npu_moe_finalize_routing(
            mlp_output,
            skip1=None,
            skip2=None,
            bias=None,
            scales=self.topk_weights,
            expanded_src_to_dst_row=self.expanded_row_idx,
            export_for_source_row=self.topk_ids,
            # NOTE: For npu_moe_init_routing_v2
            # drop_pad_mode=2,
        )


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.
    
    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        top_k_num: int,
        global_num_experts: int,
    ):
        super().__init__(device, dtype, top_k_num, global_num_experts)

        # Shared communication configurations
        ep_group = get_mc2_group()
        self.ep_rank_id = ep_group.rank_in_group
        self.ep_world_size = ep_group.world_size
        self.tp_world_size = get_tp_group().world_size

        device_group = ep_group.device_group
        local_rank = torch.distributed.get_rank(group=device_group)
        backend = device_group._get_backend(torch.device("npu"))
        self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)

        # Feature flags
        self.enable_dispatch_v2 = hasattr(torch_npu,
                                          "npu_moe_distribute_dispatch_v2")
        self.is_ascend_a3 = get_ascend_soc_version() == AscendSocVersion.A3
        self.need_extra_args = self.is_ascend_a3  # or is_torchair

        # Intermediate tensors to be passed from pre_process to post_process
        self.topk_ids = None
        self.topk_weights = None
        self.mc2_mask = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.tp_recv_counts = None

    def _pre_process(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_map: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Store tensors needed for post_process
        self.topk_ids = topk_ids.clone()
        self.topk_weights = topk_weights
        self.mc2_mask = get_forward_context().mc2_mask

        dispatch_kwargs = {
            "x": hidden_states,
            "expert_ids": self.topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.global_num_experts,
            "global_bs": 0,
            "scales": None,
            "quant_mode": 0,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }

        if self.need_extra_args:
            dispatch_kwargs.update({
                "group_tp": self.moe_all_to_all_group_name,
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
        ) = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_kwargs)[:6]

        group_list_type = 1

        return permuted_hidden_states, expert_tokens, group_list_type

    def _post_process(self, mlp_output: torch.Tensor,
                      hidden_states: torch.Tensor) -> None:
        combine_kwargs = {
            "expand_x": mlp_output,
            "expert_ids": self.topk_ids,
            "expert_scales": self.topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.global_num_experts,
            "global_bs": 0,
            "ep_send_counts": self.ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": self.ep_world_size,
            "ep_rank_id": self.ep_rank_id,
        }

        if self.enable_dispatch_v2:
            combine_kwargs[
                "assist_info_for_combine"] = self.assist_info_for_combine
        else:
            combine_kwargs["expand_idx"] = self.assist_info_for_combine

        if self.need_extra_args:
            combine_kwargs.update({
                "tp_send_counts": self.tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if self.is_ascend_a3 and self.enable_dispatch_v2:
            combine_kwargs.update({
                "x_active_mask": self.mc2_mask,
            })

        if self.enable_dispatch_v2:
            hidden_states[:] = torch_npu.npu_moe_distribute_combine_v2(
                **combine_kwargs)
        else:
            hidden_states[:] = torch_npu.npu_moe_distribute_combine(
                **combine_kwargs)


def moe_comm_pre_process(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.moe_comm_method
    return self._pre_process(hidden_states, topk_ids, topk_weights, expert_map,
                             num_experts)


def moe_comm_pre_process_fake(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    top_k_num = topk_ids.shape[1]
    permuted_hidden_states = hidden_states.repeat_interleave(top_k_num, dim=0)
    expert_tokens = torch.zeros((num_experts, ),
                                dtype=torch.int64,
                                device=hidden_states.device)
    group_list_type = 0
    return permuted_hidden_states, expert_tokens, group_list_type


def moe_comm_post_process(mlp_output: torch.Tensor,
                          hidden_states: torch.Tensor) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.moe_comm_method
    self._post_process(mlp_output, hidden_states)
    return


direct_register_custom_op(
    op_name="moe_comm_pre_process",
    op_func=moe_comm_pre_process,
    mutates_args=[],
    fake_impl=moe_comm_pre_process_fake,
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="moe_comm_post_process",
    op_func=moe_comm_post_process,
    mutates_args=["hidden_states"],
    fake_impl=lambda x, y: None,  # No-op for fake implementation
    dispatch_key="PrivateUse1",
)
