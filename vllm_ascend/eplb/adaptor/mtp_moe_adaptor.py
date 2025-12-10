from typing import Any

import torch
import torch.distributed as dist

from vllm_ascend.eplb.adaptor.deepseek_moe_adaptor import DeepSeekMoeAdaptor


class MtpMoeAdaptor(DeepSeekMoeAdaptor):

    def __init__(self, model, **args):
        super().__init__(model, **args)
        self.init_eplb_params()
        self.init_eplb_param_dict()
        self.init_expert_maps()

    def set_mtp_layers(self, mtp_instance, num_mtp_layers):
        self.mtp_instance = mtp_instance
        self.num_mtp_layers = num_mtp_layers

    def init_eplb_params(self):
        self.num_dense_layers = 0
        self.global_expert_num = self.model.config.num_experts
        self.num_moe_layers = self.model.config.num_hidden_layers - self.num_dense_layers

    def init_eplb_param_dict(self):
        super().init_eplb_param_dict()
        if any("w13_weights_offset" in name for name,_ in self.mtp_instance.named_parameters()) is not None:
            self.mtp_expert_weight_names = [
                "w13_weight_list", "w2_weight_list",
                "w13_weight_scale_fp32_list", "w13_weight_offset",
                "w2_weight_scale_list", "w2_weight_offset"
            ]
        else:
            self.mtp_expert_weight_names = ["w13_weight", "w2_weight"]

    def init_expert_maps(self):
        self.expert_map_per_layer = dict(
        )  # reference to expert map on device for expert map update
        self.expert_map_per_layer_cpu = dict(
        )  # copy of expert map on CPU to avoid device synchronize frequently
        for layer_idx in range(self.num_moe_layers):
            self.expert_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_expert_map(self.num_dense_layers + layer_idx)
        for mpt_layer_idx in range(self.num_mtp_layers):
            self.expert_map_per_layer[self.num_dense_layers + self.num_moe_layers + mpt_layer_idx] = \
                self.model.get_expert_map(self.num_dense_layers + self.num_moe_layers + mpt_layer_idx)
        # TODO: here we set number of buffer tensor equal to number of expert in each laryer, which can be improved
        num_buffer_tensor = torch.where(
            self.expert_map_per_layer[self.num_dense_layers] != -1)[0].numel()
        self.buffer_tensor_list: list[list[Any]] = [
            [] for _ in range(num_buffer_tensor)
        ]
        self.init_buffer_tensor(num_buffer_tensor)
        self.expert_param_per_layer = dict()
        self.init_expert_param_per_layer()
        self.log2phy_map_per_layer = dict()
        for layer_idx in range(self.num_moe_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + layer_idx] = \
                self.model.get_log2phy_map(self.num_dense_layers + layer_idx)
        for mpt_layer_idx in range(self.num_mtp_layers):
            self.log2phy_map_per_layer[self.num_dense_layers + self.num_moe_layers + mpt_layer_idx] = \
                self.model.get_log2phy_map(self.num_dense_layers + self.num_moe_layers + mpt_layer_idx)
        self.all_topk_ids = []

    def init_buffer_tensor(self, num_buffer_tensor):
        super().init_buffer_tensor(num_buffer_tensor)
        mtp_param_dict = dict(self.mtp_instance.named_parameters())
        for mtp_layer_idx in range(self.num_mtp_layers):
            self.expert_param_per_layer[self.num_dense_layers +
                                        self.num_moe_layers +
                                        mtp_layer_idx] = list()
        for local_expert_id in range(num_local_expert):
            for mtp_layer_idx in range(self.num_mtp_layers):
                self.expert_param_per_layer[
                    self.num_dense_layers + self.num_moe_layers +
                    mtp_layer_idx].append([
                    mtp_param_dict["model.layers." +
                                   str(self.num_dense_layers +
                                       self.num_moe_layers +
                                       mtp_layer_idx) +
                                   ".mtp_block.mlp.experts." +
                                   name].data[local_expert_id]
                    for name in self.mtp_expert_weight_names
                ])

    def get_rank_expert_workload(self) -> torch.Tensor:
        self.moe_load = self.model.get_all_moe_loads()
        self.moe_load = torch.cat([
            self.moe_load,
            self.mtp_instance.model.get_all_moe_loads().to(
                device=self.moe_load.device)
        ],
            dim=0)
        return self.moe_load

    def get_init_expert_map(self, num_moe_layers):
        expert_map = self.model.get_all_expert_map(num_moe_layers)
        expert_map = torch.cat([
            expert_map,
            self.mtp_instance.model.get_all_expert_map().to(
                device=expert_map.device)
        ],
            dim=0)
        if dist.is_initialized():
            world_size = dist.get_world_size()
        gathered = torch.empty(
            (world_size, *expert_map.shape),  # [W, L, E]
            dtype=expert_map.dtype,
            device=expert_map.device)

        dist.all_gather_into_tensor(gathered, expert_map)
        all_maps = gathered.permute(1, 0, 2)
        all_expert_maps = all_maps.cpu()

        for layer_idx in range(num_moe_layers):
            self.expert_map_per_layer_cpu[self.num_dense_layers + layer_idx] = \
                all_expert_maps[layer_idx][self.rank_id]

    def determine_expert_map_all(self):
        if self.world_size == 1:
            local_ids = torch.arange(self.global_expert_num, dtype=torch.int32)
            return local_ids.view(1, 1, -1).expand(self.num_moe_layers, 1, -1)

        local_num_experts = self.global_expert_num // self.world_size

        expert_map_all = torch.full(
            (self.num_moe_layers + self.num_mtp_layers, self.world_size,
             self.global_expert_num),
            -1,
            dtype=torch.int32)

        for r in range(self.world_size):
            if r < self.world_size - 1:
                start = r * local_num_experts
                end = (r + 1) * local_num_experts
                local_count = local_num_experts
            else:
                start = r * local_num_experts
                end = self.global_expert_num
                local_count = self.global_expert_num - r * local_num_experts

            if r < self.init_redundancy_expert:
                local_count += 1
                if end < self.global_expert_num:
                    end += 1
                else:
                    start -= 1

            local_ids = torch.arange(local_count, dtype=torch.int32)
            expert_map_all[:, r, start:end] = local_ids.unsqueeze(0).expand(
                self.num_moe_layers + self.num_mtp_layers, -1)

        return expert_map_all


