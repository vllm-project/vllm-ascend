import torch
import torch.distributed as dist
from vllm.logger import logger
from omni_placement.omni_planner import OmniPlanner

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.eplb_updator import EplbUpdator


class OmniEplbUpdator(EplbUpdator):
    """EplbUpdator subclass that delegates heat collection to OmniPlanner.

    Differences from EplbUpdator:
    - No EplbProcess subprocess, no shared_dict, no comm_group
    - forward_before() is a no-op (no D2D weight transfer)
    - forward_end() calls compute_and_set_moe_load every step
    - compute_and_set_moe_load() passes per-layer load to OmniPlanner.record_activation
    - warm_up_eplb() skips P2P warmup
    """

    def __init__(self, eplb_config):
        self.eplb_config = eplb_config
        self.multi_stage = eplb_config.eplb_policy_type == 3
        self.rank_id = dist.get_rank()
        self.expert_map_record_path = eplb_config.expert_map_record_path
        logger.info("[eplb/omni] OmniEplbUpdator initialized, rank=%s", self.rank_id)

    def set_adaptor(self, adaptor: VllmEplbAdaptor):
        self.adaptor = adaptor
        self.num_moe_layers = self.adaptor.num_moe_layers
        self.world_size = dist.get_world_size()

    def reset_log2phy(self):
        for local_idx, layer in enumerate(self.adaptor.moe_layers):
            layer.log2phy = OmniPlanner().get_log2phy(local_idx)
        logger.info("[eplb/omni] OmniEplbUpdator re-register log2phy completed, rank=%s", self.rank_id)

    def forward_before(self):
        OmniPlanner().place_experts()

    def forward_end(self, eplb_heat_collection_status: bool = True):
        if eplb_heat_collection_status:
            self.compute_and_set_moe_load()

    def compute_and_set_moe_load(self):
        local_load = self.adaptor.get_rank_expert_workload()
        for layer_idx in range(local_load.shape[0]):
            OmniPlanner().record_activation(layer_idx, local_load[layer_idx])
        self.adaptor.clear_all_moe_loads()

    def warm_up_eplb(self):
        logger.info("[eplb/omni] OmniEplbUpdator warm-up completed (no-op), rank=%s", self.rank_id)

    def shutdown(self):
        pass
