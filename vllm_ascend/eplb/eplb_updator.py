#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
# Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this updator.
import numpy
import torch
import torch.distributed as dist
import vllm.envs as envs
from vllm.logger import logger

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor
from vllm_ascend.eplb.core.eplb_device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.eplb_worker import EplbProcess


class EplbUpdator:
    def __init__(self, eplb_config, loader: D2DExpertWeightLoader, eplb_process: EplbProcess, process):
        self.eplb_config = eplb_config
        self.init_eplb(self.eplb_config.expert_map_path, process)
        self.eplb_loader = loader
        self.eplb_process = eplb_process
        self.shared_dict = self.eplb_process.shared_dict
        self.moe_imbalance_dict: dict[int, float] = {}

    def set_adaptor(self, adaptor: VllmEplbAdaptor):
        self.adaptor = adaptor
        self.num_moe_layers = self.adaptor.num_moe_layers
        local_load = self.adaptor.get_rank_expert_workload()
        self.world_size = dist.get_world_size()
        self.device = local_load.device
        shape = (self.world_size, *local_load.shape)
        self._gather_buffer = torch.empty(shape, dtype=local_load.dtype, device=self.device)
        self.eplb_loader.num_layers = self.adaptor.num_dense_layers + self.adaptor.num_moe_layers

    def init_eplb(self, expert_map_path, process):
        self.rank_id = dist.get_rank()
        self.num_expert_load_gather = 10
        self.periodic_load_gather = True
        self.expert_heat_collection_interval: torch.int64 = self.eplb_config.expert_heat_collection_interval
        self.expert_map_path = expert_map_path
        self.expert_map_record_path = self.eplb_config.expert_map_record_path

        try:
            if not envs.VLLM_ALLOW_EXPERT_LOAD_COLLECTING:
                self.num_expert_load_gather = self.expert_heat_collection_interval
                self.periodic_load_gather = False
        except Exception:
            self.num_expert_load_gather = self.expert_heat_collection_interval
            self.periodic_load_gather = False

        self.reqs = []
        self.update_info_all = []

        self.cur_iterations: torch.int64 = 0

        self.algorithm_execution_interval: torch.int64 = self.eplb_config.algorithm_execution_interval

        self.process = process

        logger.info(f"[ModelRunner] Launched EPLB process (pid={self.process.pid})")

    def update_iteration(self):
        self.cur_iterations += 1
        if self.cur_iterations == (
            self.expert_heat_collection_interval + self.algorithm_execution_interval + self.num_moe_layers
        ):
            if self.expert_map_record_path is not None:
                self.adaptor._export_tensor_to_file(self.shared_dict["expert_maps"], self.expert_map_record_path)

            self.adaptor.model.clear_all_moe_loads()
            self.cur_iterations = 0

    def get_update_info_flag(self):
        return self.cur_iterations == (self.expert_heat_collection_interval + self.algorithm_execution_interval - 1)

    def wakeup_eplb_worker_flag(self):
        return self.cur_iterations == (self.expert_heat_collection_interval - 1)

    def update_expert_weight_flag(self):
        weight_update_counter = self.cur_iterations - (
            self.expert_heat_collection_interval + self.algorithm_execution_interval
        )
        return weight_update_counter >= 0 and weight_update_counter < self.num_moe_layers

    def wakeup_eplb_worker(self):
        self.eplb_process.planner_q.put(1)

    def forward_before(self):
        if self.update_expert_weight_flag():
            (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = self.update_info_all.pop(
                0
            )
            log2phy_map_this_rank = torch.from_numpy(numpy.array(log2phy_map))
            self.eplb_loader.set_log2phy_map(log2phy_map_this_rank)
            updated_expert_map_this_rank = torch.from_numpy(numpy.array(updated_expert_map))
            self.eplb_loader.generate_expert_d2d_transfer_task(
                expert_send_info,
                expert_recv_info,
                updated_expert_map_this_rank,
                layer_id + self.adaptor.num_dense_layers,
            )

            # set asynchronous stream for d2d expert weight update
            self.reqs = []
            self.eplb_loader.asyn_expert_weight_transfer(self.reqs)

    def take_update_info_from_eplb_process(self):
        # Batch after eplb process being triggered, get update info provided by eplb process
        if self.get_update_info_flag():
            self.update_info_all = self.eplb_process.block_update_q.get()

    def forward_end(self):
        if self.wakeup_eplb_worker_flag():
            self.compute_and_set_moe_load()
            self.wakeup_eplb_worker()

        if self.update_expert_weight_flag() and self.expert_map_record_path is None:
            self.eplb_loader.update_expert_map_and_weight(self.reqs)

        self.update_iteration()

    def compute_and_set_moe_load(self):
        local_load = self.adaptor.get_rank_expert_workload()
        dist.all_gather_into_tensor(self._gather_buffer, local_load)

        moe_load = self._gather_buffer.permute(1, 0, 2)
        self.shared_dict["moe_load"] = moe_load.cpu()
        logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] shape={moe_load.shape}")

        if dist.get_rank() == 0:
            self.compute_moe_imbalance(moe_load)
            self.summarize_moe_imbalance()

        return moe_load

    def compute_moe_imbalance(self, moe_load: torch.Tensor):
        self.moe_imbalance_dict.clear()

        layer_card_load = moe_load.sum(dim=-1).cpu().float()

        for layer_idx in range(layer_card_load.size(0)):
            layer_load = layer_card_load[layer_idx]

            mean_load = layer_load.mean().item()
            max_load = layer_load.max().item()

            moe_load_imbalance = max_load / (mean_load + 1e-6)

            logger.debug(f"[ModelRunner][MOE_load_stats][Layer {layer_idx}] PAR={moe_load_imbalance:.4f}")

            self.moe_imbalance_dict[layer_idx] = moe_load_imbalance

    def summarize_moe_imbalance(self):
        values = list(self.moe_imbalance_dict.values())
        if not values:
            logger.info("[MOE_load_stats] No data available.")
            return

        avg_imbalance = sum(values) / len(values)
        max_imbalance = max(values)
        min_imbalance = min(values)

        logger.info(
            f"[ModelRunner][MOE_load_stats] Peak-to-Average-Ratio: "
            f"Mean={avg_imbalance:.4f}, Max={max_imbalance:.4f}, Min={min_imbalance:.4f}"
        )

    def warm_up_eplb(self):
        self.shared_dict["expert_maps"] = self.adaptor.get_global_expert_map()
        self.compute_and_set_moe_load()

        src_tensor = torch.empty((1,), device=self.device)
        self_rank = dist.get_rank()

        comm_op_list = []

        for dst_rank in range(self.world_size):
            if dst_rank == self_rank:
                continue
            comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank))

        for src_rank in range(self.world_size):
            if src_rank == self_rank:
                continue
            comm_op_list.append(dist.P2POp(dist.irecv, src_tensor, src_rank))
        if comm_op_list:
            reqs = dist.batch_isend_irecv(comm_op_list)

        for req in reqs:
            req.wait()

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            logger.info("[ModelRunner] EPLB process terminated")
