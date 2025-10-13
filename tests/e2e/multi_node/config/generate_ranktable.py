#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ranktable.py - Ascend 多节点 RankTable 生成模块

可被其他模块导入调用：
    from ranktable import RankTableGenerator

    generator = RankTableGenerator(
        ips=["10.1.1.1", "10.1.1.2"],
        npus_per_node=8,
        network_card_name="eth0",
        prefill_device_cnt=8,
        decode_device_cnt=8,
        local_device_ids="0,1,2,3,4,5,6,7"
    )

    ranktable = generator.generate(output_path="ranktable.json")
"""

import json
import os
import socket
import subprocess
from typing import Dict, List, Optional

import torch.distributed as dist

try:
    from vllm_ascend.utils import (AscendSocVersion, get_ascend_soc_version,
                                   init_ascend_soc_version)
except ImportError:
    # 兼容环境：不影响其他模块导入
    AscendSocVersion = None

    def init_ascend_soc_version():
        pass

    def get_ascend_soc_version():
        return None


class RankTableGenerator:

    def __init__(
        self,
        ips: List[str],
        npus_per_node: int,
        network_card_name: str,
        prefill_device_cnt: int,
        decode_device_cnt: int,
        local_device_ids: Optional[str] = None,
        hccn_tool_path: str = "/usr/local/Ascend/driver/tools/hccn_tool",
    ):
        self.ips = ips
        self.npus_per_node = npus_per_node
        self.network_card_name = network_card_name
        self.prefill_device_cnt = prefill_device_cnt
        self.decode_device_cnt = decode_device_cnt
        self.local_device_ids = local_device_ids
        self.hccn_tool_path = hccn_tool_path

        # 推断当前节点信息
        self.local_host, self.node_rank = self._get_local_rank_info()
        self.nnodes = len(ips)
        self.master_addr = ips[0]
        self.master_port = "6657"
        self.world_size = npus_per_node * self.nnodes

        # 初始化分布式环境变量
        os.environ.setdefault("MASTER_ADDR", self.master_addr)
        os.environ.setdefault("MASTER_PORT", self.master_port)
        os.environ.setdefault("NODE_RANK", str(self.node_rank))
        os.environ.setdefault("WORLD_SIZE", str(self.world_size))

    # ----------------------------- 工具函数 -----------------------------

    def _get_local_rank_info(self):
        """根据本机 IP 匹配确定 node_rank"""
        local_ips = socket.gethostbyname_ex(socket.gethostname())[2]
        for i, ip in enumerate(self.ips):
            if ip in local_ips:
                return ip, i
        return "127.0.0.1", 0

    @staticmethod
    def _run_cmd(cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True,
                              shell=True).stdout.decode("utf-8").strip()

    # ----------------------------- 核心逻辑 -----------------------------

    def generate(self, output_path: Optional[str] = "ranktable.json") -> Dict:
        """生成 ranktable.json 或返回 ranktable dict"""
        init_ascend_soc_version()
        soc_info = get_ascend_soc_version()

        # 获取 NPU 信息
        num_cards = int(
            self._run_cmd("npu-smi info -l | grep 'Total Count'").split(":")
            [1].strip())
        chips_per_card = int(
            self._run_cmd("npu-smi info -l | grep 'Chip Count'").split("\n")
            [0].split(":")[1].strip())

        # 解析 device_id
        if self.local_device_ids:
            local_device_ids = [
                int(x) for x in self.local_device_ids.split(",")
            ]
        else:
            local_device_ids = [
                card_id * chips_per_card + chip_id
                for card_id in range(num_cards)
                for chip_id in range(chips_per_card)
            ]

        # local rank
        local_rank = os.environ.get("LOCAL_RANK", "0")

        # 收集设备信息
        local_device_list: List[Dict[str, str]] = []
        if local_rank == "0":
            for device_id in local_device_ids:
                chip_id = device_id % chips_per_card
                card_id = device_id // chips_per_card
                if soc_info == AscendSocVersion.A3:
                    device_ip = self._run_cmd(
                        f"{self.hccn_tool_path} -i {device_id} -vnic -g | grep ipaddr"
                    ).split(":")[1].strip()
                    super_device_id = self._run_cmd(
                        f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep SDID"
                    ).split(":")[1].strip()
                    super_pod_id = self._run_cmd(
                        f"npu-smi info -t spod-info -i {card_id} -c {chip_id} | grep 'Super Pod ID'"
                    ).split(":")[1].strip()
                    info = {
                        "server_id": self.local_host,
                        "device_id": str(device_id),
                        "device_ip": device_ip,
                        "super_pod_id": super_pod_id,
                        "super_device_id": super_device_id,
                    }
                else:
                    device_ip = self._run_cmd(
                        f"{self.hccn_tool_path} -i {device_id} -ip -g | grep ipaddr"
                    ).split(":")[1].strip()
                    info = {
                        "server_id": self.local_host,
                        "device_id": str(device_id),
                        "device_ip": device_ip,
                    }
                local_device_list.append(info)

        # 初始化分布式通信（GLOO）
        dist.init_process_group(backend=dist.Backend.GLOO)
        global_device_list = [None] * dist.get_world_size()
        dist.all_gather_object(global_device_list, local_device_list)
        global_device_list = [
            dev for sublist in global_device_list for dev in sublist
        ]

        for i, dev in enumerate(global_device_list):
            dev["cluster_id"] = str(i + 1)

        # 校验
        assert (
            self.prefill_device_cnt + self.decode_device_cnt
        ) <= len(global_device_list), (
            "prefill_device_cnt + decode_device_cnt must be <= total devices")

        ranktable = {
            "version":
            "1.2",
            "server_count":
            str(self.world_size),
            "prefill_device_list":
            global_device_list[:self.prefill_device_cnt],
            "decode_device_list":
            global_device_list[self.
                               prefill_device_cnt:self.prefill_device_cnt +
                               self.decode_device_cnt],
            "status":
            "completed",
        }

        if local_rank == "0" and output_path:
            with open(output_path, "w") as f:
                json.dump(ranktable, f, indent=4)
            print(f"✅ RankTable written to {output_path}")

        return ranktable
