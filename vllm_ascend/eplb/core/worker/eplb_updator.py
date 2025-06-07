# # TODO
# load ssd or d2d transformer for expert weight

# matrixaccLib-EPLB:

# Input 热度表

# output
# 加载到hbm的 tensor


# step1. collect

# step2. eplb algo
# step3. expert weight loading(ssd->host->hbm or d2d hbm) hbm buffer,  与后处理或者attention 计算掩盖

# step4. expert table apply & hbm buffer copy


import time
import numpy as np
import torch_npu
import logging
from multiprocessing import Process, Queue
import multiprocessing
from abc import ABC, abstractmethod
import torch.distributed as dist


# from vllm_ascend.eplb.core.loader.device_transfer_loader import D2DExpertWeightLoader, SSD2DExpertWeightLoader
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory,DynamicConfig

logger = logging.getLogger(__name__)

class EplbWorker:

    def __init__(self, device: int, shared_dict, policy_type, enable_d2d: bool = True, ):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type, DynamicConfig())
        self.device = device
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d


    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D

        #获取初始expert_map
        if self.old_expert_maps == None:
            self.old_expert_maps = self.get_init_expert_maps()

        #获取moe负载信息
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            return

        #根据负载信息，获取更新后的专家表
        changed, priority, new_expert_maps = self.calculate_rebalance_experts(load_info)

        #如果不需要更新，则跳过
        if changed == 0:
            return
        logger.debug(f"[EPLB Process on NPU:{self.device}] new_map differs, performing D2D")

        # 更新权重
        self.load_impl(new_expert_maps)

        #更新expert_map
        self.update_expert_map(new_expert_maps)
        self.old_expert_maps = new_expert_maps
        print("EPLB Process complete")
    #
    def compose_expert_update_info(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        num_ranks = current_expert_maps.shape[1]
        num_experts = current_expert_maps.shape[2]

        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id][:][:]
            current_expert_maps_this_layer = current_expert_maps[layer_id][:][:]

            expert_send_info_this_layer = dict()
            expert_pull_info_this_layer = dict()

            dst_rank_indices, experts_to_pull = torch.where((current_expert_maps_this_layer == -1) \
                & (updated_expert_maps_this_layer != -1))

            src_rank_indices, experts_to_send = torch.where((current_expert_maps_this_layer != -1) \
                & (updated_expert_maps_this_layer == -1))

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_pull[idx].item()
                if dst_rank_id not in expert_pull_info_this_layer:
                    expert_pull_info_this_layer[dst_rank_id] = []

                candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]
                #TODO: improve selection criterion of npu sending expert_id considering such as intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_pull_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            yield (expert_send_info_this_layer, expert_pull_info_this_layer, updated_expert_maps_this_layer, layer_id)


    def load_impl(self, new_expert_table):

        if self.enable_d2d:
            #通过D2D更新的专家权重，调用D2DExpertWeightLoader
            pass
            try:
                pass
                logger.debug(f"[EPLB Process on NPU:{self.device}] D2D update completed")
            except Exception as ex:
                logger.warning(f"[EPLB Process on NPU:{self.device}] D2D update failed: {ex}", exc_info=True)
        else:
            #通过SSD2D更新专家权重，调用SSD2DExpertWeightLoader
            try:
                pass
                logger.debug(f"[EPLB Process on NPU:{self.device}] SSD2D update completed")
            except Exception as ex:
                logger.warning(f"[EPLB Process on NPU:{self.device}] SSD2D update failed: {ex}", exc_info=True)


    def calculate_rebalance_experts(self,load_info):
        """
        通过 policy 实例的 rebalance_experts 方法计算 new_map。
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(self.old_expert_maps, load_info)
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
        """
        子进程计算出 new_map 后，把它写回 shared_dict["expert_map"]。
        """
        self.shared_dict["expert_maps"] = expert_maps



class EplbProcess:
    def __init__(self, device_id: int, shared_dict, policy_type: int = 1, enable_d2d: bool = True):
        """
        Args:
            device_id: NPU 设备号
            shared_dict: Manager().dict() 返回的跨进程共享字典
            policy_type: 整型，传给 PolicyFactory.generate_policy
            enable_d2d: 是否启用 D2D 加载
        """
        self.device_id = device_id
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d

        # 创建 EplbWorker 实例
        self.worker = EplbWorker(self.device_id, self.shared_dict, self.policy_type, self.enable_d2d)

        # 后面 _launch_process 会覆盖这两个属性
        self.planner_q = None
        self.block_update_q = None

    def worker_process(self):
        """
        子进程入口：绑定到指定 NPU，循环等待 planner_q 唤醒，调用 do_update，再通知主进程已完成更新。
        """
        try:
            torch_npu.npu.set_device(self.device_id)
        except Exception as e:
            logger.warning(
                f"[EPLB 子进程 {self.device_id}] 无法绑定 NPU: {e}", exc_info=True
            )
            return

        while True:
            try:
                # 阻塞等待主进程通知
                self.planner_q.get()

                # 执行一次更新
                self.worker.do_update()

                # 通知主进程已完成更新
                self.block_update_q.put(1)

            except Exception as e:
                logger.warning(
                    f"[EPLB 子进程 {self.device_id}] 异常退出: {e}", exc_info=True
                )
                break

    def _launch_process(self):
        """
        使用 spawn 模式，启动子进程并返回 (planner_q, block_update_q, proc)。
        """
        ctx = multiprocessing.get_context("spawn")
        planner_q = ctx.Queue()
        block_update_q = ctx.Queue()

        # 把队列赋给 worker_process 中使用
        self.planner_q = planner_q
        self.block_update_q = block_update_q

        proc = ctx.Process(
            target=self.worker_process,
            args=(),
            daemon=True
        )
        proc.start()
        return planner_q, block_update_q, proc

