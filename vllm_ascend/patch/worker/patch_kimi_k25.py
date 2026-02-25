#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_k25_vit import Learnable2DInterpPosEmbDivided_fixed, get_rope_shape_decorate

logger = init_logger(__name__)


@get_rope_shape_decorate
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


class AscendLearnable2DInterpPosEmbDivided_fixed(nn.Module):
    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        pos_embs = []
        for t, h, w in grid_thws.tolist():
            x_device = x.device
            x_dtype = x.dtype
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                weight_fp32 = self.weight.to(dtype=torch.float32)
                weight_cpu = weight_fp32.to("cpu")
                pos_emb_2d = get_rope_shape(
                    weight_cpu,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )
                pos_emb_2d = pos_emb_2d.to(x_device, dtype=x_dtype)

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        out = x + torch.cat(pos_embs)
        return out


Learnable2DInterpPosEmbDivided_fixed.forward = AscendLearnable2DInterpPosEmbDivided_fixed.forward
