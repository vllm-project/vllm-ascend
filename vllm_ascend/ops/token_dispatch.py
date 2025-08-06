from abc import abstractmethod
import torch
import torch_npu
from vllm.distributed.parallel_state import (get_dp_group, get_ep_group,
                                             get_tp_group)
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm.forward_context import get_forward_context

from vllm_ascend.utils import (AscendSocVersion, dispose_tensor,
                               get_all_reduce_merge_state,
                               get_ascend_soc_version,
                               get_rm_router_logits_state, is_310p)
from vllm_ascend.torchair.utils import npu_stream_switch, npu_wait_tensor

class MoETokenDispatcher:
    """
    MoE Token Dispatcher
    """

    def __init__(self, need_param) -> None:
        self.top_k = need_param['top_k']
        self.expert_map = need_param['expert_map']
        self.log2phy = need_param['log2phy']
        self.global_redundant_expert_num = need_param['global_redundant_expert_num']
        

    @property
    def ep_group(self):
        """Get expert model parallel group."""
        return get_ep_group()


    @abstractmethod
    def token_permutation(
        self, tokens: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor,
    ):
        """Dispatch tokens to experts.

        Args:
            tokens (torch.Tensor): Input tokens.
            probs (torch.Tensor): The routing probability tensor [num_tokens, num_experts].
            routing_map (torch.Tensor): Token to expert mapping tensor.

        Returns:
            torch.Tensor: Tokens tensor.
        """
        raise NotImplementedError("Dispatch function not implemented.")

    @abstractmethod
    def token_unpermutation(self, expert_output: torch.Tensor, bias: torch.Tensor = None):
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


class TokenDispatcherWithMC2(MoETokenDispatcher):
    def __init__(self, need_param):
        super(TokenDispatcherWithMC2, self).__init__(need_param=need_param)

    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights:torch.Tensor
    ):
        pass

    def token_unpermutation(down_out_list: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        pass


class UnquantizedTokenDispatcherWithMC2(TokenDispatcherWithMC2):
    def __init__(self, need_param):
        super(TokenDispatcherWithMC2, self).__init__(need_param=need_param)
        self.moe_all_to_all_group_name = need_param['moe_all_to_all_group_name']
        self.torchair_graph_enabled = need_param['torchair_graph_enabled']
        self.moe_parallel_config = need_param['moe_parallel_config']
        self.output = None
        self.dynamic_scale = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.shared_act = None


    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights:torch.Tensor
    ):
        quant_mode = 0
        ep_rank_id = self.moe_parallel_config.ep_rank
        ep_world_size = self.moe_parallel_config.ep_size

        # NOTE: Currently, when in A3 or in torchair graph, we need to pass in some extra param into dispatch & combine
        need_extra_args = (get_ascend_soc_version() == AscendSocVersion.A3
                        or self.torchair_graph_enabled)

        # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
        a3_need_extra_args = get_ascend_soc_version() == AscendSocVersion.A3

        enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask

        moe_expert_num = len(self.expert_map)
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
            "ep_world_size": ep_world_size,
            "ep_rank_id": ep_rank_id,
        }
        if need_extra_args:
            stage1_kwargs.update({
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if a3_need_extra_args and enable_dispatch_v2:
            stage1_kwargs.update({
                "x_active_mask": mc2_mask,
            })

        kwargs_mc2.update(stage1_kwargs)

        self.output = torch_npu.npu_moe_distribute_dispatch_v2(
            **kwargs_mc2
        ) if enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
            **kwargs_mc2)
        # comm_stream.wait_stream(torch.npu.current_stream())
        expand_x, self.dynamic_scale, self.assist_info_for_combine, expert_token_nums, self.ep_recv_counts = self.output[0:5]

        if self.shared_experts is not None:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(hidden_states, topk_weights)
                shared_gate_up, _ = self.shared_experts.gate_up_proj(hidden_states)
                npu_wait_tensor(shared_gate_up, expand_x)
                self.shared_act = self.shared_experts.act_fn(shared_gate_up)
        return expand_x, expert_token_nums

    def token_unpermutation(self, down_out_list: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor,):
        moe_expert_num = len(self.expert_map)
        enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask
        need_extra_args = (get_ascend_soc_version() == AscendSocVersion.A3
                        or self.torchair_graph_enabled)

        # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
        a3_need_extra_args = get_ascend_soc_version() == AscendSocVersion.A3
        
        ep_rank_id = self.moe_parallel_config.ep_rank
        ep_world_size = self.moe_parallel_config.ep_size
        # moeCombine
        kwargs_mc2 = {
            "expand_x": down_out_list,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": moe_expert_num,
            "global_bs": 0,
        }
        tp_recv_counts = self.output[5]
        stage3_kwargs = {
            "ep_send_counts": self.ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": ep_world_size,
            "ep_rank_id": ep_rank_id,
        }
        if enable_dispatch_v2:
            stage3_kwargs.update({
                "assist_info_for_combine":
                self.assist_info_for_combine,
            })
        else:
            stage3_kwargs.update({
                "expand_idx": self.assist_info_for_combine,
            })
        if need_extra_args:
            stage3_kwargs.update({
                "tp_send_counts": tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if a3_need_extra_args and enable_dispatch_v2:
            stage3_kwargs.update({
                "x_active_mask": mc2_mask,
            })
        kwargs_mc2.update(stage3_kwargs)

        hidden_states = torch_npu.npu_moe_distribute_combine_v2(
            **kwargs_mc2
        ) if enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine(
            **kwargs_mc2)

        if self.shared_experts is None:
            return hidden_states
        else:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(self.shared_act, down_out_list)
                shared_hidden_states, _ = self.shared_experts.down_proj(self.shared_act)
            return hidden_states, shared_hidden_states


class QuantizedTokenDispatcherWithMC2(TokenDispatcherWithMC2):
    def __init__(self, need_param):
        super(TokenDispatcherWithMC2, self).__init__(need_param=need_param)
        self.moe_expert_num = None
        self.torchair_graph_enabled = need_param['torchair_graph_enabled']
        self.moe_parallel_config = need_param['moe_parallel_config']
        self.output = None
        self.dynamic_scale = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.shared_act = None
        self.swiglu_out_scale = None

    def token_permutation(
            self, hidden_states: torch.Tensor, shared_gate_up: torch.Tensor, shared_dequant_scale: torch.Tensor
    ):
        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask
        assert mc2_mask is not None
        if self.log2phy is not None:
            topk_ids = self.log2phy[topk_ids]

        quant_mode = 2
        ep_group = get_mc2_group()
        ep_rank_id = ep_group.rank_in_group
        ep_world_size = ep_group.world_size

        # NOTE: Currently, when in A3 or in torchair graph, we need to pass in some extra param into dispatch & combine
        need_extra_args = (get_ascend_soc_version() == AscendSocVersion.A3
                        or self.torchair_graph_enabled)

        # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
        a3_need_extra_args = get_ascend_soc_version() == AscendSocVersion.A3

        enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")

        if (self.expert_map is not None):
            self.moe_expert_num = len(self.expert_map) + self.global_redundant_expert_num
        else:
            self.moe_expert_num = self.global_redundant_expert_num
        # hidden_states = hidden_states.bfloat16()
        kwargs_mc2 = {
            "x": hidden_states,
            "expert_ids": topk_ids,
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": 0,
        }

        stage1_kwargs = {
            "scales": None,
            "quant_mode": quant_mode,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": ep_world_size,
            "ep_rank_id": ep_rank_id,
        }
        if need_extra_args:
            stage1_kwargs.update({
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if a3_need_extra_args and enable_dispatch_v2:
            stage1_kwargs.update({
                "x_active_mask": mc2_mask,
            })
        kwargs_mc2.update(stage1_kwargs)

        output = torch_npu.npu_moe_distribute_dispatch_v2(
            **kwargs_mc2
        ) if enable_dispatch_v2 else torch_npu.npu_moe_distribute_dispatch(
            **kwargs_mc2)
        # comm_stream.wait_stream(torch.npu.current_stream())
        expand_x, dynamic_scale, self.assist_info_for_combine, expert_token_nums, self.ep_recv_counts = output[
            0:5]

        if self.shared_experts is not None:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(shared_gate_up, expand_x)
                shared_act_out = self.shared_experts.act_fn(
                    (shared_gate_up, shared_dequant_scale))
                self.shared_act, self.swiglu_out_scale = shared_act_out[0], shared_act_out[1]
        return expand_x, expert_token_nums, dynamic_scale

    def token_unpermutation(self, down_out_list: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, ):
        
        ep_group = get_mc2_group()
        ep_rank_id = ep_group.rank_in_group
        ep_world_size = ep_group.world_size
        
        forward_context = get_forward_context()
        mc2_mask = forward_context.mc2_mask
        # moeCombine
        kwargs_mc2 = {
            "expand_x": down_out_list,
            "expert_ids": topk_ids,
            "expert_scales": topk_weights.to(torch.float32),
            "expert_shard_type": 0,
            "shared_expert_rank_num": 0,
            "moe_expert_num": self.moe_expert_num,
            "global_bs": 0,
        }
        tp_recv_counts = torch.empty(1,
                                    dtype=torch.int32,
                                    device=hidden_states.device)
        stage3_kwargs = {
            "ep_send_counts": self.ep_recv_counts,
            "group_ep": self.moe_all_to_all_group_name,
            "ep_world_size": ep_world_size,
            "ep_rank_id": ep_rank_id,
        }
        # NOTE: Currently, when in A3 or in torchair graph, we need to pass in some extra param into dispatch & combine
        need_extra_args = (get_ascend_soc_version() == AscendSocVersion.A3
                        or self.torchair_graph_enabled)

        # NOTE: Currently, when in A3, we need to pass in some extra param into dispatch & combine
        a3_need_extra_args = get_ascend_soc_version() == AscendSocVersion.A3

        enable_dispatch_v2 = hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")

        if enable_dispatch_v2:
            stage3_kwargs.update({
                "assist_info_for_combine":
                self.assist_info_for_combine,
            })
        else:
            stage3_kwargs.update({
                "expand_idx": self.assist_info_for_combine,
            })
        if need_extra_args:
            stage3_kwargs.update({
                "tp_send_counts": tp_recv_counts,
                "group_tp": self.moe_all_to_all_group_name,
                "tp_world_size": 1,
                "tp_rank_id": 0,
            })
        if a3_need_extra_args and enable_dispatch_v2:
            stage3_kwargs.update({
                "x_active_mask": mc2_mask,
            })
        kwargs_mc2.update(stage3_kwargs)

        hidden_states = torch_npu.npu_moe_distribute_combine_v2(
            **kwargs_mc2
        ) if enable_dispatch_v2 else torch_npu.npu_moe_distribute_combine(
            **kwargs_mc2)

        if self.shared_experts is None:
            return hidden_states
        else:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(self.shared_act, down_out_list)
                shared_output, _ = self.shared_experts.down_proj(
                    (self.shared_act, self.swiglu_out_scale))
            return hidden_states, shared_output

class TokenDispatcherWithFusedExperts(MoETokenDispatcher):
    def __init__(self, need_param):
        super(TokenDispatcherWithMC2, self).__init__(need_param=need_param)
        self.apply_router_weight_on_input = need_param["apply_router_weight_on_input"]
        self.top_k = need_param["top_k"]
        self.max_num_tokens = need_param["max_num_tokens"]
        self.sorted_weights = None
        self.expanded_row_idx = None
        self.sorted_token_indices = None
        self.original_shape = None
        self.mask = None

    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights:torch.Tensor
    ):
        pass

    def token_unpermutation(down_out_list: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor):
        pass


class UnquantizedTokenDispatcherWithFusedExperts(TokenDispatcherWithFusedExperts):
    def __init__(self, need_param):
        super(TokenDispatcherWithFusedExperts, self).__init__(need_param=need_param)

    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, w1:torch.Tensor, topk_weights:torch.Tensor
    ):
        self.original_shape = hidden_states.shape
        # assert len(original_shape) == 2

        num_tokens = hidden_states.shape[:-1].numel()
        num_experts = w1.shape[0]
        dtype = hidden_states.dtype
        device = hidden_states.device
        # assert dtype in [torch.float32, torch.float16, torch.bfloat16
        #                  ], "Only float32, float16, and bfloat16 are supported"

        if self.apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        if self.expert_map is not None:
            # Generate token indices and flatten
            token_indices = (torch.arange(num_tokens,
                                        device=device,
                                        dtype=torch.int64).unsqueeze(1).expand(
                                            -1, self.top_k).reshape(-1))

            # Flatten token-to-expert mappings and map to local experts
            weights_flat = topk_weights.view(-1)
            experts_flat = topk_ids.view(-1)
            local_experts_flat = self.expert_map[experts_flat]

            # Filter valid token-expert pairs
            self.mask = local_experts_flat != -1
            filtered_weights = torch.where(
                self.mask, weights_flat, torch.zeros_like(weights_flat)).to(dtype)
            filtered_experts = torch.where(
                self.mask, local_experts_flat,
                torch.full_like(local_experts_flat,
                                num_experts)).to(topk_ids.dtype)

            # Sort by local expert IDs
            sort_indices = torch.argsort(filtered_experts.view(torch.float32))
            self.sorted_token_indices = token_indices[sort_indices]
            self.sorted_weights = filtered_weights[sort_indices]

            # Compute token counts with minlength of num_experts
            # This is equivalent to but faster than:
            # >>> token_counts = torch.bincount(filtered_experts, minlength=num_experts)[:-1]
            token_counts = torch.zeros(num_experts + 1,
                                    device=device,
                                    dtype=torch.int64)
            ones = torch.ones_like(filtered_experts, dtype=torch.int64)
            token_counts.scatter_add_(0, filtered_experts.to(torch.int64), ones)
            token_counts = token_counts[:num_experts]
            expert_tokens = torch.cumsum(token_counts, dim=0, dtype=torch.int64)

            # Rearrange hidden_states
            sorted_hidden_states = hidden_states[self.sorted_token_indices]
        else:
            row_idx_len = num_tokens * self.top_k
            row_idx = (torch.arange(0,
                                    row_idx_len,
                                    dtype=torch.int32,
                                    device=device).view(self.top_k, -1).permute(
                                        1, 0).contiguous())
            active_num = self.max_num_tokens if self.max_num_tokens is not None else num_tokens
            sorted_hidden_states, self.expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=active_num)

            expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
                expanded_expert_idx, num_experts)
            expert_tokens = expert_tokens.to(torch.int64)
        return sorted_hidden_states, expert_tokens

    def token_unpermutation(self, down_out_list: torch.Tensor, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        dtype = hidden_states.dtype
        device = hidden_states.device
        if self.expert_map is not None:
            weighted_down_out = down_out_list * self.sorted_weights.unsqueeze(1)

            final_hidden_states = torch.zeros(*self.original_shape,
                                            device=hidden_states.device,
                                            dtype=hidden_states.dtype)

            # TODO: npu_grouped_matmul output random values at [num_valid_tokens:, ...]
            # This created multiple NaN and index_add_ will mix them up which harms accuracy
            # remove this mask and filter after it being fixed
            num_valid_tokens = self.mask.sum()
            valid_token_mask = torch.arange(
                0, self.sorted_token_indices.shape[0],
                device=device).unsqueeze(1) < num_valid_tokens
            valid_output = torch.where(
                valid_token_mask, weighted_down_out,
                torch.zeros_like(weighted_down_out)).to(dtype)
            final_hidden_states.index_add_(0, self.sorted_token_indices, valid_output)
        else:
            scales = torch.ones_like(
                topk_weights) if self.apply_router_weight_on_input else topk_weights
            # TODO: Reorder device memory 2 times here, replace the current
            # implementation here when suitable operators become available.
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                down_out_list,
                skip1=None,
                skip2=None,
                bias=None,
                scales=scales,
                expanded_src_to_dst_row=self.expanded_row_idx,
                export_for_source_row=topk_ids,
            )

        return final_hidden_states

class QuantizedTokenDispatcherWithFusedExperts(TokenDispatcherWithFusedExperts):
    def __init__(self, need_param):
        super(TokenDispatcherWithFusedExperts, self).__init__(need_param=need_param)

    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, w1:torch.Tensor, topk_weights:torch.Tensor
    ):
        self.original_shape = hidden_states.shape
        if len(self.original_shape) == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        num_tokens, _ = hidden_states.shape
        num_experts = w1.shape[0]
        dtype = hidden_states.dtype
        device = hidden_states.device

        if self.expert_map is not None:
            # Generate token indices and flatten
            token_indices = (torch.arange(num_tokens,
                                        device=device,
                                        dtype=torch.int64).unsqueeze(1).expand(
                                            -1, self.top_k).reshape(-1))

            # Flatten token-to-expert mappings and map to local experts
            weights_flat = topk_weights.view(-1)
            experts_flat = topk_ids.view(-1)
            local_experts_flat = self.expert_map[experts_flat]

            # Filter valid token-expert pairs
            self.mask = local_experts_flat != -1
            filtered_weights = torch.where(
                self.mask, weights_flat, torch.zeros_like(weights_flat)).to(dtype)
            filtered_experts = torch.where(
                self.mask, local_experts_flat,
                torch.full_like(local_experts_flat,
                                num_experts)).to(topk_ids.dtype)

            # Sort by local expert IDs
            sort_indices = torch.argsort(filtered_experts)
            self.sorted_token_indices = token_indices[sort_indices]
            self.sorted_weights = filtered_weights[sort_indices]

            # Compute token counts with minlength of num_experts
            # This is equivalent to but faster than:
            # >>> token_counts = torch.bincount(filtered_experts, minlength=num_experts)[:-1]
            token_counts = torch.zeros(num_experts + 1,
                                    device=device,
                                    dtype=torch.int64)
            ones = torch.ones_like(filtered_experts, dtype=torch.int64)
            token_counts.scatter_add_(0, filtered_experts.to(torch.int64), ones)
            expert_tokens = token_counts[:num_experts]
            # Rearrange hidden_states
            hidden_states = hidden_states[self.sorted_token_indices]
            self.group_list_type = 1
        else:
            row_idx_len = num_tokens * self.top_k
            row_idx = torch.arange(0,
                                row_idx_len,
                                dtype=torch.int32,
                                device=topk_weights.device).view(
                                    self.top_k, -1).permute(1, 0).contiguous()
            hidden_states, self.expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens)

            expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
                expanded_expert_idx, num_experts)
            expert_tokens = expert_tokens.to(torch.int64)
            self.group_list_type = 0
        return hidden_states, expert_tokens

    def token_unpermutation(self, down_out_list: torch.Tensor, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        dtype = hidden_states.dtype
        device = hidden_states.device
        if self.expert_map is not None:
            hidden_states.mul_(self.sorted_weights.unsqueeze(1))
            final_hidden_states = torch.zeros(*self.original_shape,
                                            device=device,
                                            dtype=dtype)

            num_valid_tokens = self.mask.sum()
            valid_token_mask = torch.arange(
                0, self.sorted_token_indices.shape[0],
                device=device).unsqueeze(1) < num_valid_tokens
            hidden_states = hidden_states.masked_fill_(~valid_token_mask,
                                                    0).to(dtype)
            final_hidden_states.index_add_(0, self.sorted_token_indices, hidden_states)
        else:
            # TODO: Reorder device memory 2 times here, replace the current
            # implementation here when suitable operators become available.
            final_hidden_states = torch_npu.npu_moe_finalize_routing(
                hidden_states,
                skip1=None,
                skip2=None,
                bias=None,
                scales=topk_weights,
                expanded_src_to_dst_row=self.expanded_row_idx,
                export_for_source_row=topk_ids,
            )

        if len(self.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(self.original_shape)
        return final_hidden_states

class QuantizedTokenDispatcherWithFusedExpertsAllGather(TokenDispatcherWithFusedExperts):
    def __init__(self, need_param):
        super(TokenDispatcherWithFusedExperts, self).__init__(need_param=need_param)
        self.batch_size = None
        self.hidden_size = None

    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights:torch.Tensor
    ):
        self.original_shape = hidden_states.shape
        if len(self.original_shape) == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens = hidden_states.shape[0]
        self.batch_size, self.hidden_size = hidden_states.shape
        topk_weights = topk_weights.to(hidden_states.dtype)

        ep_group = get_ep_group().device_group
        ep_rank = torch.distributed.get_rank(group=ep_group)
        ep_size = torch.distributed.get_world_size(ep_group)

        global_num_experts = len(self.expert_map)
        local_num_experts = global_num_experts // ep_size

        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

        hidden_states, self.expanded_x_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            scale=pertoken_scale,
            offset=None,
            active_num=num_tokens * self.top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[
                ep_rank * local_num_experts, (ep_rank + 1) * local_num_experts
            ],
            quant_mode=-1,
            row_idx_type=1)
        return hidden_states, expert_tokens

    def token_unpermutation(self, hidden_states: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                            pertoken_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor, expert_tokens: torch.Tensor):
        sorted_topk_weight = torch.index_select(topk_weights.view(-1), 0,
                                            self.expanded_x_idx)
        row_index = self.expanded_x_idx // topk_ids.shape[-1]
        row_index = row_index.to(torch.int64)
        share_input = torch.zeros((self.batch_size, self.hidden_size),
                                dtype=torch.bfloat16,
                                device="npu")

        final_hidden_states = torch_npu.npu_grouped_matmul_finalize_routing(
            hidden_states,
            w2,
            scale=w2_scale.to(torch.float32),
            bias=None,
            pertoken_scale=pertoken_scale.view(-1),
            group_list=expert_tokens,
            shared_input=share_input,
            logit=sorted_topk_weight.to(torch.float32),
            row_index=row_index,
            output_bs=self.batch_size).to(torch.bfloat16)

        if len(self.original_shape) == 3:
            final_hidden_states = final_hidden_states.view(self.original_shape)
        return final_hidden_states

class UnquantizedTokenDispatcherWithFusedExpertsMoge(TokenDispatcherWithFusedExperts):
    def __init__(self, need_param):
        super(TokenDispatcherWithFusedExperts, self).__init__(need_param=need_param)
        self.moe_parallel_config = need_param["moe_parallel_config"]
        self.global_num_experts = need_param["moe_parallel_config"]
        self.bsz = None

    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights:torch.Tensor
    ):
        ep_size = self.moe_parallel_config.ep_size
        local_num_experts = self.global_num_experts // ep_size
        local_num_group = self.top_k // ep_size

        if self.apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        self.bsz, _ = hidden_states.shape
        flatten_topk_ids = topk_ids.view(-1)
        self.sorted_topk_ids = torch.argsort(flatten_topk_ids.float())
        self.sorted_topk_ids = self.sorted_topk_ids.to(torch.int32)
        self.sorted_hidden_states = hidden_states.index_select(
            0, self.sorted_topk_ids // local_num_group)
  
        experts_id = torch.arange(0,
                                local_num_experts,
                                dtype=topk_ids.dtype,
                                device=topk_ids.device)
        num_tokens_per_expert = (flatten_topk_ids.unsqueeze(-1) == experts_id).to(
            torch.float32).sum(0)
        self.topk_scales = topk_weights.view(-1).index_select(
            0, self.sorted_topk_ids).unsqueeze(-1)
        group_list = num_tokens_per_expert.cumsum(dim=0).to(torch.int64)
        return hidden_states, group_list

    def token_unpermutation(self, down_out_list: torch.Tensor):
        ep_size = self.moe_parallel_config.ep_size
        unsorted_topk_ids = torch.argsort(self.sorted_topk_ids.float()).to(torch.int32)
        unsorted_hidden_states = down_out_list.index_select(0, unsorted_topk_ids)
        final_hidden_states = unsorted_hidden_states.reshape(
            self.bsz, self.top_k // ep_size, -1).sum(1)
        return final_hidden_states