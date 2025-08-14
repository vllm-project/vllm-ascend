import json
import threading
import time
from typing import Optional

import numpy as np
import prometheus_client
import torch
from vllm.distributed.parallel_state import get_ep_group
from vllm.logger import logger

from vllm_ascend.eplb.adaptor.vllm_adaptor import VllmEplbAdaptor

RECORDING_TIME = 10


class EplbStatLogger:
    _instance = None
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter

    def __init__(self, adaptor: VllmEplbAdaptor,
                 expert_map_path: Optional[str]):
        self.rank = get_ep_group().rank
        self.layers_num = adaptor.num_moe_layers
        self.global_expert_num = adaptor.global_expert_num
        self.ep_size = get_ep_group().world_size

        if expert_map_path is None:
            self.phy2log_map = [[i for i in range(self.global_expert_num)]
                                for _ in range(self.layers_num)]
        else:
            self.phy2log_map = self._expert_file_to_list(expert_map_path)
            self.global_expert_num = len(self.phy2log_map[0])

        self.local_expert_num = self.global_expert_num // self.ep_size

        labelnames_phy_load = ["rank", "layer", "phy_expert_id"]
        labelnames_phy2log = [
            "rank", "layer", "phy_expert_id", "log_expert_id"
        ]

        self.phy_expert = self._counter_cls(
            name="vllm:phy_expert_heat",
            documentation="Heat of each physical expert per rank",
            labelnames=labelnames_phy_load)

        self.phy2log = self._gauge_cls(
            name="vllm:phy2log",
            documentation="physical expert to logical expert per rank",
            labelnames=labelnames_phy2log)

        self.do_record_loop = threading.Thread(target=self.record_loop,
                                               daemon=True)
        self.moe_load = None

        self.update_load = False
        self.update_map = False

        # only init in rank0
        self.all_phy2log = []
        if self.rank == 0:
            for layer_id in range(self.layers_num):
                for phy_expert_id in range(self.global_expert_num):
                    self.phy_expert.labels(
                        rank=phy_expert_id // self.local_expert_num,
                        layer=layer_id,
                        phy_expert_id=phy_expert_id % self.local_expert_num)

            for layer_id in range(len(self.phy2log_map)):
                local_phy2log = []
                for phy_expert_id, log_expert_id in enumerate(
                        self.phy2log_map[layer_id]):
                    a = self.phy2log.labels(
                        rank=phy_expert_id // self.local_expert_num,
                        layer=layer_id,
                        phy_expert_id=phy_expert_id % self.local_expert_num,
                        log_expert_id=log_expert_id)
                    a.set(1)
                    local_phy2log.append(a)
                self.all_phy2log.append(local_phy2log)

            self.moe_load = torch.zeros(
                (self.layers_num, self.ep_size, self.local_expert_num))

            self.lock = threading.Lock()
            self.start_loop()

    @staticmethod
    def get_instance():
        if EplbStatLogger._instance is None:
            raise ValueError(
                "EplbStatLogger instance has not been initialized.")
        return EplbStatLogger._instance

    @staticmethod
    def init_instance(adaptor: VllmEplbAdaptor,
                      expert_map_path: Optional[str]):
        """Initialize the singleton instance of ExpertLoadBalancer."""
        EplbStatLogger._instance = EplbStatLogger(adaptor, expert_map_path)
        return EplbStatLogger._instance

    def record(self, moe_load, phy2log_map):
        if self.rank != 0:
            return
        try:
            with self.lock:
                if moe_load is not None:
                    torch.add(self.moe_load, moe_load, out=self.moe_load)
                    self.update_load = True

                if phy2log_map is not None:
                    self.phy2log_map = phy2log_map
                    self.update_map = True
        except Exception as e:
            logger.debug(f"Record moe_load or phy2log error, error result:{e}")

    def record_loop(self):
        while True:
            try:
                if self.update_load:
                    with self.lock:
                        self.update_load = False
                        moe_load = self.moe_load.tolist()
                        self.moe_load.zero_()
                    moe_load = np.array(moe_load)
                    res = np.zeros_like(moe_load)
                    res[..., 0] = moe_load[..., 0]
                    res[..., 1:] = moe_load[..., 1:] - moe_load[..., :-1]
                    res = res.reshape(self.layers_num, -1)
                    self.record_expert_load(res)

                if self.update_map:
                    with self.lock:
                        self.update_map = False
                        phy2log_map = self.phy2log_map
                    phy2log_map = np.array(phy2log_map).reshape(
                        self.layers_num, -1)
                    self.record_phy2log(phy2log_map)
            except Exception as e:
                logger.debug(
                    f"Record moe_load or phy2log prometheus error, error result:{e}"
                )
            time.sleep(RECORDING_TIME)

    def start_loop(self):
        self.do_record_loop.start()

    def record_phy2log(self, phy2log_map: list[list[int]]):
        for layer_id in range(len(phy2log_map)):
            for phy_expert_id, log_expert_id in enumerate(
                    phy2log_map[layer_id]):
                self.all_phy2log[layer_id][phy_expert_id].set(0)

                a = self.phy2log.labels(
                    rank=phy_expert_id // self.local_expert_num,
                    layer=layer_id,
                    phy_expert_id=phy_expert_id % self.local_expert_num,
                    log_expert_id=log_expert_id)
                a.set(1)
                self.all_phy2log[layer_id][phy_expert_id] = a

    def record_expert_load(self, moe_load: list[list[int]]):
        for layer_id in range(len(moe_load)):
            for phy_expert_id, load in enumerate(moe_load[layer_id]):
                self.phy_expert.labels(
                    rank=phy_expert_id // self.local_expert_num,
                    layer=layer_id,
                    phy_expert_id=phy_expert_id % self.local_expert_num,
                ).inc(load)

    def _expert_file_to_list(self, expert_map_path: str):
        with open(expert_map_path, "r") as f:
            data = json.load(f)

        phy2log_data = []
        for layer in data["layer_list"]:
            device_data = []
            for device in layer["device_list"]:
                device_data += device["device_expert"]
            phy2log_data.append(device_data)
        return phy2log_data

