#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import torch


# 310P RC: non_blocking H2D copy in rot_pos_emb can race with subsequent indexing.
def rot_pos_emb_310(self, grid_thw: list[list[int]]):
    max_grid_size = max(max(h, w) for _, h, w in grid_thw)
    pos_ids = [
        self.rot_pos_ids(h, w, self.spatial_merge_size)
        if t == 1
        else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
        for t, h, w in grid_thw
    ]
    pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=False)

    cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)
    cos_combined = cos[pos_ids].flatten(1)
    sin_combined = sin[pos_ids].flatten(1)
    return cos_combined, sin_combined
