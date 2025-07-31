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
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, shared_experts: torch.Tensor, topk_weights:torch.Tensor
    ):
        pass

    def token_unpermutation(down_out_list: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, shared_experts: torch.Tensor):
        pass


class UnquantizedTokenDispatcherWithMC2(TokenDispatcherWithMC2):
    def __init__(self, need_param):
        super(UnquantizedTokenDispatcherWithMC2, self).__init__(need_param=need_param)
        self.moe_all_to_all_group_name = need_param['moe_all_to_all_group_name']
        self.torchair_graph_enabled = need_param['torchair_graph_enabled']
        self.moe_parallel_config = need_param['moe_parallel_config']
        self.output = None
        self.dynamic_scale = None
        self.assist_info_for_combine = None
        self.ep_recv_counts = None
        self.shared_act = None


    def token_permutation(
        self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, shared_experts: torch.Tensor, topk_weights:torch.Tensor
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

        if shared_experts is not None:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(hidden_states, topk_weights)
                shared_gate_up, _ = shared_experts.gate_up_proj(hidden_states)
                npu_wait_tensor(shared_gate_up, expand_x)
                self.shared_act = shared_experts.act_fn(shared_gate_up)
        return expand_x, expert_token_nums

    def token_unpermutation(self, down_out_list: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, shared_experts: torch.Tensor):
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

        if shared_experts is None:
            return hidden_states
        else:
            with npu_stream_switch("moe_secondary", 0):
                npu_wait_tensor(self.shared_act, down_out_list)
                shared_hidden_states, _ = shared_experts.down_proj(self.shared_act)
            return hidden_states, shared_hidden_states


class QuantizedTokenDispatcherWithMC2(TokenDispatcherWithMC2):
    def __init__(self, need_param):
        super(QuantizedTokenDispatcherWithMC2, self).__init__(need_param=need_param)
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
