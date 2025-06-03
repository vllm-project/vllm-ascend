# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import torch

class EplbForwarder:
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict

    # def get_load_info(self,workload_dir ):
    #     loader = EplbExpertDataCollect(workload_dir, map_location="cpu")
    #     load_info = loader.load_all()
    #     return load_info

    # def get_old_map(self,expertmap_dir):
    #     loader = EplbExpertMapCollect(workload_dir, map_location="cpu")
    #     old_map = loader.load_all()
    #     return old_map

    def get_init_expert_map(self):
        if "expert_map" not in self.shared_dict:
            return None
        return self.shared_dict["expert_map"]

    def fetch_and_sum_load_info(self):
        if "moe_load" not in self.shared_dict:
            return None
        return self.shared_dict["moe_load"]

    def load_experts_to_device(self):
        pass

    def update_expert_map(self,expert_map):
        self.shared_dict["expert_map"] = expert_map


