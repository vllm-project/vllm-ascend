# TODO
# load ssd or d2d transformer for expert weight

# matrixaccLib-EPLB:

# Input 热度表
# output 加载到hbm的 tensor
# step1. collect
# step2. eplb algo
# step3. expert weight loading(ssd->host->hbm or d2d hbm) hbm buffer,  与后处理或者attention 计算掩盖
# step4. expert table apply & hbm buffer copy

from abc import ABC, abstractmethod
from vllm_ascend.eplb.core.loader.device_transfer_loader import D2DExpertWeightLoader
from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory

class EplbWorker:

    def __init__(self, old_expert_table):
        self.old_expert_table = old_expert_table
        self.policy = PolicyFactory.generate_policy(policy_type, DynamicConfig())
        # init process
    
    def do_update(self, expert_workload):
        # put data in to queue
        # in process self.policy.generate_policy() 
        # get epxert table && tensor

        # async stream 
        # D2D
        # H2D
        
    def load_impl(self, new_expert_table):
        raise NotImplementedError