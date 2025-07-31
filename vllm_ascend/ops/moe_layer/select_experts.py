from abc import ABC, abstractmethod
import torch_npu
from vllm_ascend.ops.fused_moe import select_experts
import vllm_ascend.envs as envs_ascend
SELECT_GATING_TOPK_SOTFMAX_EXPERTS: bool = envs_ascend.SELECT_GATING_TOPK_SOTFMAX_EXPERTS


class BaseSelectExperts(ABC):

    def __init__(self):
        need_param = SelectExpertConfig.get_config
        self.top_k = need_param["top_k"]
        self.e_score_correction_bias = need_param["e_score_correction_bias"]
        self.topk_group = need_param["topk_group"]
        self.num_expert_group = need_param["num_expert_group"]
        self.custom_routing_function = need_param["custom_routing_function"]
        self.scoring_func = need_param["scoring_func"]
        self.global_num_experts = need_param["global_num_experts"]
        self.use_grouped_topk = need_param['use_grouped_topk']
        self.renormalize = need_param['renormalize']

    def forward(self, router_logits: torch.Tensor, x: torch.Tensor):
        if self.global_num_experts == 256:
            topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=self.top_k,  # topk当前写8
                bias=self.e_score_correction_bias,
                k_group=self.topk_group,  # fix: 4
                group_count=self.num_expert_group,  # fix 8
                group_select_mode=1,  # 0: group中的最大; 1: topk2.sum(fix)
                renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                # out_flag=False, # todo new api; 第三个输出是否输出
                # y2_flag=False, # old api; 第三个输出是否输出
                routed_scaling_factor=1,
                eps=float(1e-20))
        else:
            topk_weights, topk_ids = select_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=self.use_grouped_topk,
                renormalize=self.renormalize,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias,
            )
        return topk_weights, topk_ids


class UnquantizedSelectExperts(BaseSelectExperts):
    def __init__(self):
        super().__init__()
        
    def forward(self, router_logits: torch.Tensor, x: torch.Tensor):
        if SELECT_GATING_TOPK_SOTFMAX_EXPERTS:
            topk_weights, topk_ids = select_gating_top_k_softmax_experts(
                hidden_states=x,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize)
        else:
            topk_weights, topk_ids = super().forward(router_logits, x)

        return topk_weights, topk_ids


class QuantizedSelectExperts(BaseSelectExperts):
    def __init__(self):
        super().__init__()

    def forward(self, router_logits: torch.Tensor, x: torch.Tensor):

        return  super().forward(router_logits, x)



