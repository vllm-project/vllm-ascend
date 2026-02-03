import os
from collections.abc import Callable

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe.layer import get_compressed_expert_map

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.eplb.core.eplb_utils import (
    expert_file_to_tensor,
    generate_log2phy_map,
    init_eplb_config,
)
from vllm_ascend.ops.fused_moe.experts_selector import select_experts, zero_experts_compute
from vllm_ascend.ops.fused_moe.fused_moe import AscendFusedMoE, AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.fused_moe.moe_comm_method import setup_moe_comm_method

"""
Patch init for scenario for redundant experts
"""


def init_eplb_config(eplb_config, layer_id, moe_config):
    import os

    import torch

    expert_map_path = eplb_config.expert_map_path
    n_experts = moe_config.num_experts
    ep_size = moe_config.ep_size
    ep_rank = moe_config.ep_rank
    eplb_enable = eplb_config.dynamic_eplb
    n_redundant = eplb_config.num_redundant_experts if eplb_enable else 0
    global_placement = None

    if expert_map_path:
        if not (os.path.exists(expert_map_path) and os.access(expert_map_path, os.R_OK)):
            raise ValueError("Invalid EPLB path")
        eplb_enable = True
        global_placement, physical_count = expert_file_to_tensor(expert_map_path, layer_id)
        if physical_count is not None:
            n_redundant = physical_count - n_experts
            if not moe_config.supports_eplb:
                raise ValueError("Eplb supports only w8a8_dynamic quantization.")
        else:
            eplb_enable = False

    if global_placement is None:
        # Use the env-var driven logic here for all ranks:
        EXPERT_PARTITION_SPLIT = os.getenv("EXPERT_PARTITION_SPLIT", "")
        if not EXPERT_PARTITION_SPLIT:
            # fallback: assign all experts to rank 0 if env var missing
            EXPERT_PARTITION_SPLIT = ep_size
        else:
            logger.info(
                f"Overwriting expert map to replicated experts because EXPERT_PARTITION_SPLIT is set.\nReplicated every {EXPERT_PARTITION_SPLIT} accelerators."
            )
            EXPERT_PARTITION_SPLIT = int(EXPERT_PARTITION_SPLIT)

        LOCAL_NUM_EXPERTS_ENV = os.getenv("LOCAL_NUM_EXPERTS", "0")
        if LOCAL_NUM_EXPERTS_ENV and int(LOCAL_NUM_EXPERTS_ENV) > 0:
            local_num_experts_env = int(LOCAL_NUM_EXPERTS_ENV)
        else:
            local_num_experts_env = n_experts // EXPERT_PARTITION_SPLIT

        global_placement = []

        for rank_id in range(ep_size):
            representative_ep_rank = rank_id % EXPERT_PARTITION_SPLIT

            start = representative_ep_rank * local_num_experts_env
            end = start + local_num_experts_env
            # Clip end to max experts
            if end > n_experts:
                end = n_experts

            expert_indices = torch.arange(start, end, dtype=torch.int32)
            global_placement.append(expert_indices)

    if ep_size == 1:
        assert not eplb_enable, "EPLB must be used in expert parallelism."
        return None, None, None, n_redundant

    global_expert_map = []
    local_expert_map = None

    for rankid in range(ep_size):
        expert_map = torch.full((n_experts,), -1, dtype=torch.int32)
        local_placement = global_placement[rankid]
        expert_map[local_placement] = torch.arange(local_placement.shape[0], dtype=torch.int32)
        global_expert_map.append(expert_map)
        if rankid == ep_rank:
            # Move to device if needed
            device = getattr(moe_config, "device", None)
            if device:
                local_expert_map = expert_map.to(device)
            else:
                local_expert_map = expert_map

    log2phy = generate_log2phy_map(global_expert_map, ep_rank).to(device) if eplb_enable else None

    return torch.stack(global_expert_map), local_expert_map, log2phy, n_redundant


def __init__(self, *args, **kwargs):
    super(AscendFusedMoE, self).__init__(*args, **kwargs)

    num_experts = kwargs["num_experts"]
    intermediate_size = kwargs["intermediate_size"]

    AscendFusedMoE.moe_counter += 1
    self.moe_instance_id = AscendFusedMoE.moe_counter

    self._expert_map = None
    self.log2phy = None

    if self.quant_config is None:
        self.quant_method = AscendUnquantizedFusedMoEMethod(self.moe_config)
    else:
        self.quant_method = self.quant_config.get_quant_method(self, self.layer_name)

    assert self.quant_method is not None

    self.moe_config.tp_group = get_tp_group()
    self.moe_config.dp_group = get_dp_group()
    self.moe_config.ep_group = get_ep_group()
    self.moe_config.mc2_group = get_mc2_group()
    self.moe_config.supports_eplb = self.quant_method.supports_eplb
    ascend_config = get_ascend_config()
    # flashcommon3 gate stream
    self.multistream_overlap_gate = ascend_config.multistream_overlap_gate
    if self.multistream_overlap_gate and AscendFusedMoE.gate_stream is None:
        AscendFusedMoE.gate_stream = torch.npu.Stream()
    if self.custom_routing_function is None and self.e_score_correction_bias is not None:
        vllm_config = get_current_vllm_config()
        self.e_score_correction_bias.data = self.e_score_correction_bias.data.to(dtype=vllm_config.model_config.dtype)

    # init moe
    eplb_config = ascend_config.eplb_config
    self.global_expert_map, self._expert_map, self.log2phy, self.global_redundant_expert_num = init_eplb_config(
        eplb_config, self.moe_instance_id, self.moe_config
    )
    self.global_num_experts = num_experts + self.global_redundant_expert_num
    self.dynamic_eplb = eplb_config.dynamic_eplb and (self.log2phy is not None)
    self.local_num_experts = (
        torch.sum(self._expert_map != -1).item() if self._expert_map is not None else self.global_num_experts
    )

    # Maybe overwrite
    if int(os.getenv("GLOBAL_NUM_EXPERTS", "0")) > 1:
        self.global_num_experts = int(os.getenv("GLOBAL_NUM_EXPERTS", "0"))
        self.num_experts = self.global_num_experts
        self.local_num_experts = int(os.getenv("LOCAL_NUM_EXPERTS", "0"))

    if self._expert_map is not None:
        logger.info_once(
            "[EP Rank %s/%s] Expert parallelism is enabled. Local/global"
            " number of experts: %s/%s. Experts local to global index map:"
            " %s.",
            self.ep_rank,
            self.ep_size,
            self.local_num_experts,
            self.global_num_experts,
            get_compressed_expert_map(self._expert_map),
        )
    if self.dynamic_eplb:
        self.moe_load = torch.zeros(self.local_num_experts, dtype=torch.int64).npu()

    self.moe_config.num_experts = self.global_num_experts
    self.moe_config.num_local_experts = self.local_num_experts
    self.moe_config.global_redundant_expert_num = self.global_redundant_expert_num

    moe_quant_params = {
        "num_experts": self.local_num_experts,
        "hidden_size": self.hidden_size,
        "intermediate_size_per_partition": self.intermediate_size_per_partition,
        "params_dtype": self.params_dtype,
        "weight_loader": self.weight_loader,
    }
    # need full intermediate size pre-sharding for WNA16 act order
    if self.quant_method.__class__.__name__ in ("GPTQMarlinMoEMethod", "CompressedTensorsWNA16MoEMethod"):
        moe_quant_params["intermediate_size_full"] = intermediate_size
    self.quant_method.create_weights(layer=self, **moe_quant_params)

    self.enable_shared_expert_dp = ascend_config.enable_shared_expert_dp

    setup_moe_comm_method(self.moe_config)
    self.quant_type = self._get_quant_type()


"""
Need to .clone() for w13 and w2 weights to make it contiguous
"""


def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    use_grouped_topk: bool,
    top_k: int,
    router_logits: torch.Tensor,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    enable_force_load_balance: bool = False,
    log2phy: torch.Tensor = None,
    **kwargs,
) -> torch.Tensor:
    zero_expert_num = getattr(layer, "zero_expert_num", 0)
    zero_expert_type = getattr(layer, "zero_expert_type", None)
    topk_weights, topk_ids = select_experts(
        hidden_states=x,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=use_grouped_topk,
        renormalize=renormalize,
        topk_group=topk_group,
        num_expert_group=num_expert_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        global_num_experts=global_num_experts,
    )

    if zero_expert_num > 0 and zero_expert_type is not None:
        topk_ids, topk_weights, zero_expert_result = zero_experts_compute(
            expert_indices=topk_ids,
            expert_scales=topk_weights,
            num_experts=global_num_experts,
            zero_expert_type=zero_expert_type,
            hidden_states=x,
        )

    topk_weights = topk_weights.to(x.dtype)
    # this is a naive implementation for experts load balance so as
    # to avoid accumulating too much tokens on a single rank.
    # currently it is only activated when doing profile runs.
    if enable_force_load_balance:
        random_matrix = torch.rand(topk_ids.size(0), global_num_experts, device=topk_ids.device)
        topk_ids = torch.argsort(random_matrix, dim=1)[:, : topk_ids.size(1)].to(topk_ids.dtype)

    moe_comm_method = get_forward_context().moe_comm_method
    final_hidden_states = moe_comm_method.fused_experts(
        hidden_states=x,
        w1=layer.w13_weight.clone(),  ## Clone to make contiguous
        w2=layer.w2_weight.clone(),  ## Clone to make contiguous
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        apply_router_weight_on_input=apply_router_weight_on_input,
        dynamic_eplb=self.dynamic_eplb,
        log2phy=log2phy,
        mc2_mask=kwargs.get("mc2_mask"),
    )
    if zero_expert_num > 0 and zero_expert_type is not None:
        final_hidden_states += zero_expert_result
    return final_hidden_states
