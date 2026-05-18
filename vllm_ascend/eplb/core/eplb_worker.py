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
import os
from multiprocessing import Process, Queue
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed import get_ep_group
from vllm.logger import logger

from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory


class EplbWorker:
    def __init__(self, shared_dict, policy_type, enable_d2d: bool = True):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type)
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.rank_id = get_ep_group().rank_in_group
        self.multi_stage = policy_type == 3

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D
        # Get initial expert_map
        torch.set_num_threads(1)
        logger.info("[EPLB-DEBUG] worker pid=%s do_update: start, rank_id=%s", os.getpid(), self.rank_id)
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            if self.old_expert_maps is not None:
                self.num_local_experts = self.old_expert_maps.max() + 1
            else:
                raise ValueError("Failed to get expert_maps from shared_dict.")
        logger.info(
            "[EPLB-DEBUG] worker pid=%s do_update: old_expert_maps shape=%s num_local_experts=%s",
            os.getpid(), self.old_expert_maps.shape, self.num_local_experts,
        )

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            logger.info("[EPLB-DEBUG] worker pid=%s do_update: load_info is None, returning", os.getpid())
            return
        logger.info("[EPLB-DEBUG] worker pid=%s do_update: load_info shape=%s", os.getpid(), load_info.shape)

        # Get the updated expert table based on the workload information
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        logger.info("[EPLB-DEBUG] worker pid=%s do_update: old_placement shape=%s", os.getpid(), old_placement.shape)
        _, _, new_placement = self.calculate_rebalance_experts(load_info, old_placement)
        logger.info("[EPLB-DEBUG] worker pid=%s do_update: new_placement computed", os.getpid())

        if self.rank_id == 0:
            if self.multi_stage:
                hotness = self._calculate_hotness(old_placement, load_info.sum(0))
            else:
                hotness = self._calculate_hotness(old_placement, load_info)
            current_mean, current_max = self._compute_imbalance(old_placement, hotness)
            update_mean, update_max = self._compute_imbalance(new_placement, hotness)
            logger.info(
                "[Expert Hotness] Current: mean=%.3f, max=%.3f, Updated: mean=%.3f, max=%.3f",
                current_mean,
                current_max,
                update_mean,
                update_max,
            )

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        self.check_expert_placement(old_placement, new_placement)
        new_expert_maps = self.local2global(new_placement)
        logger.info("[EPLB-DEBUG] worker pid=%s do_update: new_expert_maps shape=%s", os.getpid(), new_expert_maps.shape)
        self.update_expert_map(new_expert_maps)

        update_info = self.compose_expert_update_info_greedy(new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps
        logger.debug("EPLB Process compute complete")

        packed_update_info = self.pack_update_info(update_info)
        logger.info(
            "[EPLB-DEBUG] worker pid=%s do_update: packed %s layers",
            os.getpid(), len(packed_update_info),
        )

        return packed_update_info

    def check_expert_placement(self, old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            if torch.unique(new_placement[layer_id]).numel() < torch.unique(old_placement[layer_id]).numel():
                logger.error("There exists expert not placed on any rank in layer %s", layer_id)
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    logger.error(
                        "Replicated experts are placed on the same NPU; "
                        "expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move], old_placement_check[expert_not_move]):
                    logger.error(
                        "There exists expert movement inside NPU; expert placement on layer %s, rank %s is invalid",
                        layer_id,
                        rank_id,
                    )
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        logger.info(
            "[EPLB-DEBUG] worker pid=%s compose_greedy: num_layers=%s rank_id=%s",
            os.getpid(), num_layers, self.rank_id,
        )
        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                logger.info(
                    "[EPLB-DEBUG] worker pid=%s compose_greedy: layer %s no change, yielding empty",
                    os.getpid(), layer_id,
                )
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    updated_expert_maps_this_layer,
                    layer_id,
                )
                continue

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where(
                (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
            )

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where(
                (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
            )

            logger.info(
                "[EPLB-DEBUG] worker pid=%s compose_greedy: layer %s recv_count=%s send_count=%s",
                os.getpid(), layer_id, len(dst_rank_indices), len(src_rank_indices),
            )

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                # TODO: improve selection criterion of NPU sending expert_id,
                # considering intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            logger.info(
                "[EPLB-DEBUG] worker pid=%s compose_greedy: layer %s yielding send_keys=%s recv_keys=%s",
                os.getpid(), layer_id,
                list(expert_send_info_this_layer.keys()), list(expert_recv_info_this_layer.keys()),
            )
            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )

    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        Compute `new_map` by calling the `rebalance_experts` method of the policy instance.
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):
        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self, placement: torch.Tensor, E_local: int) -> tuple[torch.Tensor, torch.Tensor]:
        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local), fill_value=-1, dtype=torch.long, device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local

    def local2global(self, placement_local: torch.Tensor) -> torch.Tensor:
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

    def pack_update_info(self, update_info_generator):
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:
            send_info_this_rank = send_info.get(self.rank_id, [])
            recv_info_this_rank = recv_info.get(self.rank_id, [])
            logger.info(
                "[EPLB-DEBUG] worker pid=%s pack_update: layer=%s rank_id=%s send=%s recv=%s",
                os.getpid(), layer_id, self.rank_id, send_info_this_rank, recv_info_this_rank,
            )
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map[self.rank_id].numpy().tolist())

            log2phy_map = generate_log2phy_map(new_expert_map, self.rank_id)
            log2phy_all.append(log2phy_map.numpy().tolist())

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

    @staticmethod
    def _compute_imbalance(deployment_all_layer, hotness_all_layer: np.ndarray):
        imbalance_list = []
        deployment_all_layer = np.array(deployment_all_layer)
        for deployment, hotness in zip(deployment_all_layer, hotness_all_layer):
            counts = np.bincount(deployment.reshape(-1), minlength=hotness.shape[0])

            unit_hotness = np.divide(hotness, counts, out=np.zeros_like(hotness, dtype=float), where=counts != 0)

            stage_load = unit_hotness[deployment].sum(-1)
            stage_par = stage_load.max() / stage_load.mean()
            imbalance_list.append(stage_par)

        max_val = max(imbalance_list)
        mean_val = sum(imbalance_list) / len(imbalance_list)
        return mean_val, max_val

    @staticmethod
    def _calculate_hotness(deployment_all_layer, moe_load_all_layer):
        hotnesses = []
        num_of_expert = deployment_all_layer.shape[1] * deployment_all_layer.shape[2]
        for deployment, rank_load in zip(deployment_all_layer, moe_load_all_layer.numpy()):
            hotness = np.zeros(num_of_expert, dtype=rank_load.dtype)
            deployment_flat = deployment.ravel()
            rank_load_flat = rank_load.ravel()
            np.add.at(hotness, deployment_flat, rank_load_flat)
            hotnesses.append(hotness)

        return np.array(hotnesses)


class EplbProcess:
    def __init__(self, shared_dict, policy_type: int = 0, enable_d2d: bool = True):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q: Queue[Any] = Queue()
        self.block_update_q: Queue[Any] = Queue(maxsize=1)

        # Create EplbWorker instance
        self.worker = EplbWorker(self.shared_dict, self.policy_type, self.enable_d2d)

    def worker_process(self, planner_q, block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up,
        call do_update, then notify main process update is complete.
        """
        try:
            from ms_service_metric.adapters.vllm.adapter import get_vllm_adapter, initialize_vllm_metric  # type: ignore

            initialize_vllm_metric()
            adapter = get_vllm_adapter()
            logger.info("[EPLB metrics] The adapter initialized: %s", adapter.is_initialized())
        except Exception as e:
            logger.warning("[EPLB metrics] Failed to initialize metrics: %s", e)

        if self.policy_type == 3:
            from vllm_ascend.eplb.core.policy.policy_flashlb import warm_up

            warm_up()
        while True:
            try:
                logger.info("[EPLB-DEBUG] worker pid=%s: waiting for planner_q...", os.getpid())
                planner_q.get()
                logger.info("[EPLB-DEBUG] worker pid=%s: woken up, starting do_update()", os.getpid())

                packed_update_info = self.worker.do_update()

                logger.info(
                    "[EPLB-DEBUG] worker pid=%s: do_update() done, %s layers to update, putting to block_update_q",
                    os.getpid(), len(packed_update_info) if packed_update_info else 0,
                )
                while True:
                    if not block_update_q.empty():
                        logger.info("[EPLB-DEBUG] worker pid=%s: block_update_q still full, waiting...", os.getpid())
                        continue
                    block_update_q.put(packed_update_info)
                    logger.info("[EPLB-DEBUG] worker pid=%s: packed_update_info put in block_update_q", os.getpid())
                    break

            except Exception as e:
                logger.warning(
                    "[EPLB subprocess exiting due to error: %s]",
                    e,
                    exc_info=True,
                )
                break

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)

        proc.start()
        return proc
