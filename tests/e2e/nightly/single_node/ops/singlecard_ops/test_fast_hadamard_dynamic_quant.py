#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import pytest
import torch
import torch.nn.functional as F
import torch_npu

from vllm_ascend.ops.fast_hadamard import (
    _fast_hadamard_dynamic_quant_int8_ref,
    fast_hadamard_dynamic_quant_last_dim,
)

torch.manual_seed(45)

BATCHES = [*range(1, 9), 16, 32, 64, 128, 256, 512, 4096]


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fast_hadamard_dynamic_quant_int8_matches_ref(batch: int, dtype: torch.dtype):
    x = torch.randn(batch, 128, device="npu", dtype=dtype)

    out, scale = fast_hadamard_dynamic_quant_last_dim(x)
    torch.npu.synchronize()

    ref_out, ref_scale = _fast_hadamard_dynamic_quant_int8_ref(x.cpu())
    scale = scale.cpu()
    out = out.cpu()

    scale_rel_err = ((scale - ref_scale).abs() / ref_scale.abs().clamp_min(1e-6)).max()
    cosine = F.cosine_similarity(out.float().reshape(1, -1), ref_out.float().reshape(1, -1))
    max_abs_diff = (out.to(torch.int16) - ref_out.to(torch.int16)).abs().max()
    assert scale_rel_err.item() < 0.02
    assert cosine.item() > 0.99
    assert max_abs_diff.item() <= 1

    torch_npu.npu.empty_cache()
