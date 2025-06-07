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

import torh.distinguish as dist

from vllm_ascend.eplb.adaptor.abstract_adaptor import EplbAdaptor
from vllm.logger import logger

class VllmEplbAdaptor(EplbAdaptor):
    
    def __init__(self, **args):
        self.model = model

    def get_rank_expert_workload(self, num_moe_layers):
        return self.model.get_all_moe_loads(num_moe_layers)
    
    def get_init_expert_map(self):
        expert_map = self.model.get_all_expert_map(self.num_moe_layers)
        if dist.is_initialized():
            world_size = dist.get_world_size()

        rank = dist.get_rank()

        tensor_list = [
            torch.zeros_like(expert_map) for _ in range(world_size)
        ]

        dist.all_gather(tensor_list, expert_map)
        gathered = torch.stack(tensor_list, dim=0)
        all_maps = gathered.permute(1, 0, 2).contiguous()

        all_expert_maps = all_maps.to(torch.device("cpu"))
        self.shared_dict["expert_maps"] = all_expert_maps
        logger.debug(f"[ModelRunner] Updated shared_dict['expert_map'] = {expert_map}")
        return all_expert_maps
    
    def do_update_expert_map(self):
        raise NotImplementedError
    
    def do_update_expert_weight(self):
        raise NotImplementedError