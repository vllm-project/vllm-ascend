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

"""Compare A4W4 expert GMM against per-expert quantized matmul on an idle NPU."""

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.quantization.methods.w4a4.w4a4_int_moe import AscendW4A4DynamicFusedMoEMethod


@pytest.mark.parametrize("hidden_size,intermediate_size", [(128, 128), (2560, 640)])
def test_w4a4_int_moe(hidden_size, intermediate_size):
    torch.manual_seed(17)
    method = AscendW4A4DynamicFusedMoEMethod.__new__(AscendW4A4DynamicFusedMoEMethod)
    method.in_dtype = torch.bfloat16
    layer = torch.nn.Module()
    originals = {}
    for prefix, out_size, in_size in (
        ("w13", 2 * intermediate_size, hidden_size),
        ("w2", hidden_size, intermediate_size),
    ):
        w = torch.randint(-8, 8, (3, out_size, in_size), device="npu", dtype=torch.int8)
        scale = torch.rand(3, out_size, 1, device="npu") * 0.05 + 0.001
        originals[prefix] = (w.clone(), scale.clone())
        for name, tensor in (("weight", w), ("weight_scale", scale), ("weight_offset", torch.zeros_like(scale))):
            layer.register_parameter(f"{prefix}_{name}", torch.nn.Parameter(tensor, requires_grad=False))
    method.process_weights_after_loading(layer)
    counts = [2, 0, 3]
    context = SimpleNamespace(
        layer=layer,
        dynamic_scale=None,
        group_list_type=0,
        group_list=torch.tensor([2, 2, 5], device="npu", dtype=torch.int64),
    )
    for prefix in ("w13", "w2"):
        x = torch.randn(5, originals[prefix][0].shape[-1], device="npu", dtype=torch.bfloat16)
        q, scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.quint4x2)
        got = method._gmm(context, q.clone(), scale, prefix)
        if prefix == "w13":
            context.hidden_states = x
            context.dynamic_scale = torch.ones_like(scale)
            torch.testing.assert_close(method.apply_gmm1(context), got, rtol=0, atol=0)
            context.dynamic_scale = None
        refs = []
        start = 0
        w, ws = originals[prefix]
        for expert, count in enumerate(counts):
            if count:
                packed = torch_npu.npu_convert_weight_to_int4pack(w[expert].to(torch.int32)).T
                refs.append(
                    torch_npu.npu_quant_matmul(
                        q[start : start + count].clone(),
                        packed,
                        scale=ws[expert].flatten(),
                        pertoken_scale=scale[start : start + count].clone(),
                        output_dtype=torch.bfloat16,
                    )
                )
            start += count
        ref = torch.cat(refs)
        print(prefix, "max BF16 difference", (got - ref).abs().max().item())
        torch.testing.assert_close(got, ref, rtol=0.02, atol=0.05)
        context.group_list_type = 1
        context.group_list = torch.tensor(counts, device="npu", dtype=torch.int64)
        torch.testing.assert_close(method._gmm(context, q.clone(), scale, prefix), ref, rtol=0.02, atol=0.05)
        print(prefix, "passed: non-unit scales, empty expert, cumulative and count group lists")
        context.group_list_type = 0
        context.group_list = torch.tensor([2, 2, 5], device="npu", dtype=torch.int64)

    layer.w13_weight_offset.data.fill_(1)
    try:
        method.process_weights_after_loading(layer)
    except ValueError as error:
        assert "symmetric weight offsets" in str(error)
    else:
        raise AssertionError("Nonzero offsets must be rejected")
    print("asymmetric offset rejection passed")
