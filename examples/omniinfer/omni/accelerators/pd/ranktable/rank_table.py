# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import dataclasses
import enum
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import get_type_hints

import pytz

from omni.accelerators.pd.ranktable.device import ServerGroup, Server, Device
from omni.accelerators.pd.utils import get_config_from_dict_or_env

GLOBAL_RANK_TABLE_ENV = "GLOBAL_RANK_TABLE_FILE_PATH"


class GroupType(enum.Enum):
    PREFILL = "1"
    DECODE = "2"


GROUP_INDEX_TO_TYPE = {
    "1": GroupType.PREFILL,
    "2": GroupType.DECODE
}

GROUP_TYPE_TO_ROLE = {
    GroupType.PREFILL: "prefill",
    GroupType.DECODE: "decode"
}

ARGS_TO_ENV = {
    "global_rank_table_path": "GLOBAL_RANK_TABLE_FILE_PATH",
    "local_rank_table_path": "RANK_TABLE_FILE_PATH",
    "prefill_pod_num": "PREFILL_POD_NUM",
    "decode_pod_num": "DECODE_POD_NUM",
}

ARGS_TO_DEFAULT = {
    "global_rank_table_path": None,
    "local_rank_table_path": None,
    "prefill_pod_num": 1,
    "decode_pod_num": 1,
}


@dataclass
class RankTableConfig:
    global_rank_table_path: str
    local_rank_table_path: str
    prefill_pod_num: int = 1
    decode_pod_num: int = 1

    @classmethod
    def from_dict_or_env(cls, data: dict):
        fields = {f.name for f in dataclasses.fields(cls)}
        fields_type = get_type_hints(cls)

        filtered_args_data = {k: v for k, v in data.items() if k in fields}

        args_data = {field: get_config_from_dict_or_env(filtered_args_data, field, ARGS_TO_ENV[field], 
                                                        ARGS_TO_DEFAULT[field], fields_type[field]) for field in fields}
        return cls(**args_data)

class GlobalRankTable:
    def __init__(self, config: RankTableConfig):
        self.config = config

        self._rank_table_info = self.get_ranktable_dict()
        self.group_dict = self.init_server_groups()

    def get_ranktable_dict(self):
        env_path = self.config.global_rank_table_path
        with open(env_path, 'r', encoding='utf-8') as f:
            rank_table = json.load(f)

        return rank_table

    def init_server_groups(self):
        # Obtain the original server_group_list information, merge device, and the sub-node not carries the IP.
        group_dict = {}
        cluster_id = 0
        for server_group in self._rank_table_info["server_group_list"]:
            group_id = int(server_group.get("group_id"))
            group_type = self.get_server_role(group_id)
            if group_type is None:
                raise ValueError("Unknown group id.")

            for server_info in server_group["server_list"]:
                for device_info in server_info["device"]:
                    device_info["cluster_id"] = cluster_id

                    cluster_id += 1
            group_dict.setdefault(
                group_id, ServerGroup(
                    server_group,
                    need_sort=self.get_server_role(group_id) == GROUP_TYPE_TO_ROLE[GroupType.PREFILL]
                )
            )

        return group_dict

    def get_group_type_from_server(self, server: Server):
        for group_type, group in self.group_dict.items():
            if group.contains(server):
                return group_type
        raise ValueError(f"Server ip {server.server_ip} is not in the groups.")

    def get_server_role(self, group_id):
        if group_id < self.config.prefill_pod_num:
            return GROUP_TYPE_TO_ROLE[GroupType.PREFILL]
        if self.config.prefill_pod_num <= group_id < self.config.prefill_pod_num + self.config.decode_pod_num:
            return GROUP_TYPE_TO_ROLE[GroupType.DECODE]
        return None

    def find_group_by_local_info(self, local_info_group):
        for group in self.group_dict.values():
            if group == local_info_group:
                return group
        raise ValueError(f"Local info group {local_info_group} not found.")

    @property
    def prefill_group(self):
        group_id_start = 0
        group_id_end = self.config.prefill_pod_num
        prefill_group_list = [self.group_dict.get(i, None) for i in range(group_id_start, group_id_end)]
        return prefill_group_list

    @property
    def decode_group(self):
        group_id_start = self.config.prefill_pod_num
        group_id_end = self.config.prefill_pod_num + self.config.decode_pod_num
        decode_group_list = [self.group_dict.get(i, None) for i in range(group_id_start, group_id_end)]
        return decode_group_list
