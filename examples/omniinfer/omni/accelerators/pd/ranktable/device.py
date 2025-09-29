# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

class Device:
    def __init__(self, device_info):
        self.device_id = int(device_info["device_id"])
        self.device_ip = device_info["device_ip"]
        self.rank_id = int(device_info["rank_id"])
        self.cluster_id = int(device_info.get("cluster_id", "0"))

    def __repr__(self) -> str:
        return ("Device("
                f"device_id={self.device_id}, "
                f"device_ip={self.device_ip}, "
                f"rank_id={self.rank_id}")

    def __eq__(self, other):
        return self.device_ip == other.device_ip


class Server:
    def __init__(self, server_info):
        self.server_id = server_info["server_id"]
        self.server_ip = server_info["server_ip"]
        self.device_list = self.init_device_list(server_info)

    @staticmethod
    def init_device_list(server_info):
        device_list = []
        for device_info in server_info.get("device", []):
            device_list.append(Device(device_info))
        return device_list

    def __len__(self):
        return len(self.device_list)

    def __repr__(self) -> str:
        return ("Server("
                f"server_id={self.server_id}, "
                f"server_ip={self.server_ip}, "
                f"device_list={self.device_list}")

    def __eq__(self, other):
        return self.server_ip == other.server_ip and self.device_list == other.device_list


class ServerGroup:
    def __init__(self, group_info, need_sort=False):
        self.group_id = int(group_info["group_id"])
        self.server_count = int(group_info.get("server_count", "0"))
        self.server_list = self.init_server_list(group_info)

        # Due to the scheduling initiated by Ray, the non-first nodes of prefill mast be sorted in lexicographical order.
        if need_sort:
            self.server_list = self.server_list[:1] + sorted(self.server_list[1:], key=lambda server: server.server_ip)

    def __eq__(self, other):
        return sorted(self.server_list, key=lambda x: x.server_ip) == sorted(other.server_list, key=lambda x: x.server_ip)

    @property
    def cluster_id_start(self) -> int:
        return self.server_list[0].device_list[0].cluster_id

    @property
    def host_ip(self) -> str:
        return self.server_list[0].server_ip

    @staticmethod
    def init_server_list(group_info):
        server_list = []
        for cluster_id, server_info in enumerate(group_info["server_list"]):
            server_list.append(
                Server(server_info))
        return server_list

    @property
    def device_list(self):
        device_list_all = []
        for server in self.server_list:
            device_list_all.extend(server.device_list)
        return device_list_all

    def __repr__(self) -> str:
        return ("Group("
                f"group_id={self.group_id}, "
                f"server_count={self.server_count}, "
                f"server_list={self.server_list}")

    def get_server_list_ip(self):
        return [server.server_ip for server in self.server_list]

    def contains(self, server: Server):
        for this_server in self.server_list:
            if this_server == server:
                return True
        return False

    def get_server_by_rank_id(self, rank_id):
        device = self.device_list[rank_id]
        for server in self.server_list:
            if device in server.device_list:
                return server
        return None
