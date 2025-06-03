import torch
from vllm_ascend.eplb.eplb_planner.eplb_expert_data_collect import EplbExpertLoadCollect, EplbExpertMapCollect



class EplbPlanner:
    def __init__(self, shared_dict, policy_type=1):
        config = None
        self.shared_dict = shared_dict
        self.old_map = None
        self.policy = None

    def calculate_rebalance_experts(self,load_info):
        changed, priority, new_map = self.rebalance_experts(self.old_map, load_info)
        self.old_map = new_map
        return changed, priority, new_map

    def get_init_map(self):
        if "expert_map" not in self.shared_dict:
            return None
        else:
            self.old_map = self.shared_dict["expert_map"]
            return self.old_map

    def rebalance_experts(self, current_expert_table, expert_workload): 
        changed, sort_layers, new_map = self.policy.rebalance_experts(current_expert_table, expert_workload)
        if changed:
            pass
        return changed, sort_layers, new_map