# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl


@pytest.mark.parametrize("pooling", [False, True])
@pytest.mark.parametrize("replacement", [False, True])
@torch.inference_mode()
def test_attention_outer_writeback_on_npu(pooling, replacement):
    query = torch.ones(7, 2, 128, device="npu", dtype=torch.bfloat16)
    output = torch.full_like(query, 2.0)
    inner_output = torch.full((16, 2, 128), 3.0, device="npu", dtype=query.dtype) if replacement else output
    impl = object.__new__(AscendAttentionBackendImpl)
    impl.key_cache = None
    impl.forward_impl = Mock(return_value=inner_output)
    impl._forward_encoder_attention = Mock(return_value=inner_output)
    metadata = SimpleNamespace(model_runner_type="pooling" if pooling else "generate", causal=False)
    layer = SimpleNamespace(layer_name="layer", _k_scale_float=1.0, _v_scale_float=1.0)
    torch.npu.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        returned = impl.forward(layer, query, None, None, (), metadata, output)
        torch.npu.synchronize()
    assert returned is output
    torch.testing.assert_close(output, inner_output[:7], rtol=0, atol=0)
    copies = sum(e.count for e in profile.key_averages() if e.key == "aten::copy_")
    assert copies == int(replacement)
