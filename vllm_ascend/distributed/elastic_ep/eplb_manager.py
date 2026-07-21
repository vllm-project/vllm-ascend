import torch
from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import init_logger

from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map

logger = init_logger(__name__)


class ElasticEplbManager:
    def __init__(self, worker):
        self.model = worker.get_model()
        model_config = self.model.config
        self.num_dense_layers = getattr(model_config, "first_k_dense_replace", 0)
        self.num_moe_layers = model_config.num_hidden_layers - self.num_dense_layers
        self.expert_maps = None

    def get_expert_maps(self):
        all_layer_global_expert_map = []
        for layer_id in range(self.num_moe_layers):
            map_cpu = self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
            all_layer_global_expert_map.append(map_cpu)
        self.expert_maps = torch.stack(all_layer_global_expert_map)
        return self.expert_maps.clone()

    def eplb(self, old_ep_size, new_ep_size):
        assert old_ep_size != new_ep_size
        # Fetch rank_id dynamically: the eplb group may have been switched
        # (e.g. scale-down runs eplb after switch_and_prepare switches to the
        # new group), so a rank cached at __init__ time would be stale.
        rank_id = get_eplb_group().rank_in_group

        expert_map = []
        for layer_id in range(self.num_moe_layers):
            expert_map.append(self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.expert_map)
        expert_map = torch.stack(expert_map).unsqueeze(1).npu()
        expert_maps = get_eplb_group().all_gather(expert_map, dim=1).cpu()
        for layer_id in range(self.num_moe_layers):
            # Scale Up
            if old_ep_size < new_ep_size:
                expert_maps_this_layer = expert_maps[layer_id]
            # Scale Down
            else:
                if rank_id >= old_ep_size:
                    return
                num_logical_experts = expert_maps.shape[-1]
                experts_per_npu = expert_maps.max() + 1
                num_npus = expert_maps.shape[1]
                assert experts_per_npu * num_npus >= num_logical_experts
                expert_maps_this_layer = expert_maps[layer_id][:new_ep_size]
            self.model.model.layers[
                self.num_dense_layers + layer_id
            ].mlp.experts.global_expert_map = expert_maps_this_layer.cpu()
            self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.log2phy.copy_(
                generate_log2phy_map(expert_maps_this_layer, rank_id)
            )


