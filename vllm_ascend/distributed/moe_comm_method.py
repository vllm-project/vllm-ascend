from abc import ABC, abstractmethod

import torch
import torch_npu
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.utils import direct_register_custom_op


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
