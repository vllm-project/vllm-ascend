# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.fused_moe import routed_experts
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A3, reason="Checks the A3 fused MC2 weight layout")
@pytest.mark.parametrize("hidden_size,intermediate_size", [(128, 32), (4096, 1024)])
@torch.inference_mode()
def test_unquantized_expert_nz_gmm(monkeypatch, request, hidden_size, intermediate_size):
    # Exercise the post-load path without creating an engine or an EP group.
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    request.addfinalizer(lambda: setattr(torch.npu.config, "allow_internal_format", previous in (b"enable", "enable")))
    torch.npu.config.allow_internal_format = True
    monkeypatch.setattr(routed_experts, "get_ascend_config", lambda: SimpleNamespace(enable_fused_mc2=1))
    monkeypatch.setattr(routed_experts, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(routed_experts, "use_cann_megamoe", lambda _: False)
    method = object.__new__(routed_experts.AscendUnquantizedFusedMoEMethod)
    method.dynamic_eplb = False
    layer = torch.nn.Module()
    generator = torch.Generator().manual_seed(17)
    checkpoint = {}
    params = {}
    for name, rows, cols in [
        ("w13_weight", 2 * intermediate_size, hidden_size),
        ("w2_weight", hidden_size, intermediate_size),
    ]:
        w = torch.randint(-8, 9, (2, rows, cols), dtype=torch.int8, generator=generator).to(torch.bfloat16)
        checkpoint[name] = w
        params[name] = torch.nn.Parameter(w.npu(), requires_grad=False)
        layer.register_parameter(name, params[name])

    method.process_weights_after_loading(layer)
    group_ends = torch.tensor([4, 8], dtype=torch.int64, device="npu")
    for name, w in checkpoint.items():
        weight = getattr(layer, name)
        assert weight is params[name]
        assert torch_npu.get_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ
        # Each one-hot input selects a checkpoint column exactly, independently of NZ packing.
        columns = torch.tensor([0, 1, w.shape[2] // 2, w.shape[2] - 1])
        x = torch.zeros(4, w.shape[2], dtype=torch.bfloat16)
        x[torch.arange(4), columns] = 1
        expected = w.index_select(2, columns).transpose(1, 2).reshape(8, w.shape[1])
        result = torch_npu.npu_grouped_matmul(
            [x.repeat(2, 1).npu()], [weight], group_list=group_ends, group_type=0, group_list_type=0, split_item=2
        )[0]
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0, msg=lambda msg: f"{name}: {msg}")
