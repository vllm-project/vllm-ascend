# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import torch
from vllm.distributed.afd_transfer.afd_connector.metadata import AFDConnectorData

@dataclass
class M2NAFDConnectorMetadata(AFDConnectorData):
    def __init__(self):
        self.topk_idx = None
        self.topk_weights = None
        self.moe_expert_num = 0
        self.scale = None
        self.handle = None
        self.quant_mode = 0
        self.aiv_num = 0
        self.batch_size = 0
        self.h = 0
        self.k = 0
        self.expert_token_nums_type = 0
        self.expand_x_type = torch.float16
        
@dataclass
class CAMM2NAFDConnectorMetadata(AFDConnectorData):
    def __init__(self, moe_expert_num=0,
        shared_expert_num = 0, scale=None, handle=None, quant_mode=0,
        aiv_num=0, batch_size=0, h=0, k=0):
        self.moe_expert_num = moe_expert_num
        self.shared_expert_num = shared_expert_num
        self.scale = scale
        self.handle = handle
        self.quant_mode = quant_mode
        self.aiv_num = aiv_num
        self.batch_size = batch_size
        self.h = h
        self.k = k

@dataclass
class CAMP2PAFDConnectorMetadata(AFDConnectorData):
    def __init__(self, moe_expert_num=0,
        shared_expert_num = 0, scale=None, handle=None, quant_mode=0,
        aiv_num=0, batch_size=0, h=0, k=0):
        self.moe_expert_num = moe_expert_num
        self.shared_expert_num = shared_expert_num
        self.scale = scale
        self.handle = handle
        self.quant_mode = quant_mode
        self.aiv_num = aiv_num
        self.batch_size = batch_size
        self.h = h
        self.k = k
