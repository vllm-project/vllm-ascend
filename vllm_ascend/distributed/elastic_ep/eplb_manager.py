import json
import os
import uuid
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed import P2POp
from vllm.config import get_current_vllm_config
from vllm.distributed.parallel_state import get_eplb_group
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config

from ...eplb.core.eplb_utils import generate_log2phy_map
from ..parallel_state import get_dynamic_eplb_group
from .policy_default_elastic_eplb import DefaultElasticEplb


class ElasticEplbManager:
    def __init__(self, worker):
        self.rank_id = get_eplb_group().rank_in_group
        self.dynamic_eplb = get_ascend_config().eplb_config.dynamic_eplb
        self.model = worker.get_model()
        model_config = self.model.config
        self.num_dense_layers = getattr(model_config, "first_k_dense_replace", 0)
        self.num_moe_layers = model_config.num_hidden_layers - self.num_dense_layers
        self.tp_size = worker.parallel_config.tensor_parallel_size
        if self.dynamic_eplb:
            self.shared_dict = worker.model_runner.shared_dict
            self.eplb_updator = worker.model_runner.eplb_updator
            self.eplb_adaptor = worker.model_runner.eplb_adaptor
            self.eplb_loader = worker.model_runner.eplb_loader
            self.eplb_worker = worker.model_runner.eplb_process.worker
            self.expert_maps = None
            self.policy = DefaultElasticEplb()

    def get_expert_maps(self):
        if self.dynamic_eplb:
            eplb_updator = self.eplb_updator
            if eplb_updator.cur_iterations >= eplb_updator.expert_heat_collection_interval:
                while eplb_updator.cur_iterations != 0:
                    eplb_updator.forward_before()
                    eplb_updator.forward_end()
            self.expert_maps = self.shared_dict["expert_maps"]
        else:
            all_layer_global_expert_map = []
            for layer_id in range(self.num_moe_layers):
                map_cpu = self.model.model.layers[self.num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
                all_layer_global_expert_map.append(map_cpu)

            self.expert_maps = torch.stack(all_layer_global_expert_map)

        return self.expert_maps.clone()

    def reset_eplb_updator(self):
        if self.dynamic_eplb:
            self.eplb_updator.cur_iterations = 0
            self.eplb_updator.update_info_all = []

    def set_new_comm_group(self):
        if self.dynamic_eplb:
            self.eplb_updator.comm_group = get_dynamic_eplb_group()
            self.eplb_updator.world_size = get_dynamic_eplb_group().world_size
            self.eplb_loader.comm_group = get_dynamic_eplb_group()

    def eplb(self, old_ep_size, new_ep_size):
        assert old_ep_size != new_ep_size

        if not self.dynamic_eplb:
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
                    if self.rank_id >= old_ep_size:
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
                    generate_log2phy_map(expert_maps_this_layer, self.rank_id)
                )
            return

        # Fetch expert_map from eplb_adaptor to prevent unfinished expert updates.
        old_expert_maps = self.get_expert_maps()
        assert old_expert_maps.shape[1] == old_ep_size

        num_local_experts = old_expert_maps.max() + 1
        old_placement = global2local(old_expert_maps, num_local_experts)

        local_load = self.eplb_adaptor.get_rank_expert_workload().unsqueeze(1)
        moe_load = get_dynamic_eplb_group().all_gather(local_load, dim=1).cpu()
        if old_ep_size < new_ep_size:
            moe_load = moe_load[:, :old_ep_size]

        self.policy.set_new_ep_size(new_ep_size)
        new_placement = self.policy.rebalance_experts(old_placement, moe_load)
        new_placement = torch.tensor(new_placement)
        new_expert_maps = local2global(new_placement)

        self.shared_dict["expert_maps"] = new_expert_maps
        self.eplb_worker.old_expert_maps = new_expert_maps
        self.expert_maps = new_expert_maps

        # Scale Up
        if old_ep_size < new_ep_size:
            shape = list(new_expert_maps.shape)
            shape[1] = new_ep_size - old_ep_size
            old_expert_maps = torch.cat([old_expert_maps, torch.full(shape, -1)], dim=1)
        # Scale Down
        else:
            shape = list(new_expert_maps.shape)
            shape[1] = old_ep_size - new_ep_size
            new_expert_maps = torch.cat([new_expert_maps, torch.full(shape, -1)], dim=1)

        update_info = compose_expert_update_info_greedy_optimized(new_expert_maps, old_expert_maps)

        for send_info, recv_info, new_expert_map, layer_id in update_info:
            send_info_this_rank = send_info.get(self.rank_id, [])
            recv_info_this_rank = recv_info.get(self.rank_id, [])
            new_expert_map_this_rank = new_expert_map[self.rank_id]
            new_log2phy_map_this_rank = generate_log2phy_map(new_expert_map, self.rank_id)
            layer_id += self.eplb_adaptor.num_dense_layers

            recv_expert_list = self.d2d_transfer_experts(
                send_info_this_rank,
                recv_info_this_rank,
                new_expert_map_this_rank,
                layer_id,
            )

            self.update_expert_map_and_weight(
                new_expert_map_this_rank,
                new_log2phy_map_this_rank,
                recv_expert_list,
                layer_id,
            )

        self.eplb_adaptor.model.clear_all_moe_loads()
        self.eplb_updator.cur_iterations = 0
        torch_npu.npu.synchronize()

        if self.rank_id == 0:
            logger.info("[Elastic EP] EPLB finished, new expert parallel size: %s", new_ep_size)

    def d2d_transfer_experts(
        self,
        expert_send_info: list,
        expert_recv_info: list,
        updated_expert_map: torch.Tensor,
        layer_id: int,
    ) -> list[tuple[int, int]]:
        p2p_ops = []

        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            local_expert_id = self.eplb_adaptor.expert_map_per_layer_cpu[layer_id][global_expert_id_to_send].item()
            for index, src_tensor in enumerate(self.eplb_adaptor.expert_param_per_layer[layer_id][local_expert_id]):
                op = object.__new__(P2POp)
                op.op = dist.isend
                op.tensor = src_tensor
                op.group_peer = dst_rank
                p2p_ops.append((op, global_expert_id_to_send))

        recv_expert_list = []
        for buffer_tensor_id, recv_info in enumerate(expert_recv_info):
            recv_rank, global_expert_id_to_recv = recv_info
            for buffer_tensor in self.eplb_adaptor.buffer_tensor_list[buffer_tensor_id]:
                op = object.__new__(P2POp)
                op.op = dist.irecv
                op.tensor = buffer_tensor
                op.group_peer = recv_rank
                p2p_ops.append((op, global_expert_id_to_recv))
            local_expert_to_replace = updated_expert_map[global_expert_id_to_recv].item()
            recv_expert_list.append((local_expert_to_replace, buffer_tensor_id))

        p2p_ops = sorted(p2p_ops, key=lambda x: x[1])
        p2p_ops = [item[0] for item in p2p_ops]

        device_communicator = get_dynamic_eplb_group().device_communicator
        device_communicator.batch_isend_irecv(p2p_ops)

        return recv_expert_list

    def update_expert_map_and_weight(
        self,
        updated_expert_map: torch.Tensor,
        updated_log2phy_map: torch.Tensor,
        recv_expert_list: list[tuple[int, int]],
        layer_id: int,
    ):
        # update expert_map
        self.eplb_adaptor.do_update_expert_map(layer_id, updated_expert_map)

        # update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(layer_id, updated_log2phy_map)

        # update expert weight
        for recv_expert_info in recv_expert_list:
            local_expert_to_replace, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(layer_id, local_expert_to_replace, buffer_tensor_id)


def compose_expert_update_info_greedy_optimized(
    updated_expert_maps,
    current_expert_maps,
    rank_to_node=None,
):
    num_layers = current_expert_maps.shape[0]
    for layer_id in range(num_layers):
        updated = updated_expert_maps[layer_id]
        current = current_expert_maps[layer_id]

        expert_send_info: dict = {}
        expert_recv_info: dict = {}

        if torch.equal(updated, current):
            yield expert_send_info, expert_recv_info, updated, layer_id
            continue

        dst_ranks, recv_experts = torch.where((current == -1) & (updated != -1))
        src_ranks, send_experts = torch.where((current != -1) & (updated == -1))

        recv_map = {}
        for dst, exp in zip(dst_ranks.tolist(), recv_experts.tolist()):
            recv_map.setdefault(exp, []).append(dst)

        for expert_id, dst_list in recv_map.items():
            mask_send = send_experts == expert_id
            primary_src = src_ranks[mask_send].tolist()
            mask_keep = current[:, expert_id] != -1
            secondary_src = torch.where(mask_keep)[0].tolist()
            candidates = primary_src + [r for r in secondary_src if r not in primary_src]

            if rank_to_node is not None:
                for dst in dst_list:
                    best_src = None
                    best_score = (1, float("inf"))  # (node_penalty, load)
                    for src in candidates:
                        same_node = rank_to_node[src] == rank_to_node[dst]
                        load = len(expert_send_info.get(src, []))
                        score = (0 if same_node else 1, load)
                        if score < best_score:
                            best_score = score
                            best_src = src
                    expert_send_info.setdefault(best_src, []).append((dst, expert_id))
                    expert_recv_info.setdefault(dst, []).append((best_src, expert_id))
            else:
                for dst in dst_list:
                    loads = [len(expert_send_info.get(src, [])) for src in candidates]
                    best_src = candidates[loads.index(min(loads))]
                    expert_send_info.setdefault(best_src, []).append((dst, expert_id))
                    expert_recv_info.setdefault(dst, []).append((best_src, expert_id))

        yield expert_send_info, expert_recv_info, updated, layer_id


def global2local(placement: torch.Tensor, E_local: int) -> torch.Tensor:
    L, G, _ = placement.shape
    device = placement.device

    pt_local = torch.full((L, G, E_local), fill_value=-1, dtype=torch.long, device=device)

    valid = placement >= 0
    l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

    slot_idx = placement[l_idx, g_idx, k_idx]

    pt_local[l_idx, g_idx, slot_idx] = k_idx

    return pt_local


def local2global(placement_local: torch.Tensor) -> torch.Tensor:
    L, G, E_local = placement_local.shape
    device = placement_local.device

    max_id = torch.max(placement_local)
    E_global = (max_id + 1).item() if max_id >= 0 else 0

    if E_global == 0:
        return torch.empty((L, G, 0), dtype=torch.long, device=device)

    placement_global = torch.full((L, G, E_global), fill_value=-1, dtype=torch.long, device=device)

    valid = placement_local >= 0
    l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
    gid_idx = placement_local[l_idx, g_idx, slot_idx]

    placement_global[l_idx, g_idx, gid_idx] = slot_idx

    return placement_global


def generate_global_placement(n_expert, ep_size, n_redundant):
    if (n_expert + n_redundant) % ep_size != 0:
        raise ValueError("(n_expert + n_redundant) % ep_size must be 0")
    all_experts = np.arange(n_expert + n_redundant)
    groups = np.array_split(all_experts, ep_size)
    groups = [group % n_expert for group in groups]
    return torch.tensor(groups, dtype=torch.int32)


def export_tensor_to_file(expert_maps, expert_map_record_path: str):
    num_local_experts = expert_maps.max() + 1

    expert_maps_list = expert_maps.tolist()
    record: dict[str, Any] = {"moe_layer_count": len(expert_maps_list), "layer_list": []}

    for layer_idx, layer_data in enumerate(expert_maps_list):
        layer_record: dict[str, Any] = {
            "layer_id": layer_idx,
            "device_count": len(layer_data),
            "device_list": [],
        }

        for device_idx, experts in enumerate(layer_data):
            placement = [experts.index(i) for i in range(num_local_experts)]
            device_record = {"device_id": device_idx, "device_expert": placement}
            layer_record["device_list"].append(device_record)

        record["layer_list"].append(layer_record)

    with open(expert_map_record_path, "w") as f:
        json.dump(record, f, indent=4)


def generate_expert_maps_file():
    hf_text_config = get_current_vllm_config().model_config.hf_text_config
    num_dense_layers = getattr(hf_text_config, "first_k_dense_replace", 0)
    num_hidden_layers = getattr(hf_text_config, "num_hidden_layers", 0)
    num_moe_layers = num_hidden_layers - num_dense_layers
    num_logical_experts = getattr(hf_text_config, "n_routed_experts", None) or getattr(
        hf_text_config, "num_experts", None
    )
    assert num_logical_experts is not None and num_logical_experts > 0
    num_redundant_experts = get_ascend_config().eplb_config.num_redundant_experts
    parallel_config = get_current_vllm_config().parallel_config
    ep_size = parallel_config.data_parallel_size * parallel_config.tensor_parallel_size
    global_placement = generate_global_placement(num_logical_experts, ep_size, num_redundant_experts).unsqueeze(0)
    global_placement = global_placement.repeat(num_moe_layers, 1, 1)
    global_expert_maps = local2global(global_placement)
    file_path = os.path.join("/tmp", uuid.uuid4().hex + ".json")
    export_tensor_to_file(global_expert_maps, file_path)

    return file_path
