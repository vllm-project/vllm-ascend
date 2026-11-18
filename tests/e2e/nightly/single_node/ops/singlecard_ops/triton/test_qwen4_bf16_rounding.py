# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Rounding boundaries in the actual Ascend runtime entry points."""

import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ops.activation import AscendSiluAndMul


def test_shared_silu_scope():
    torch.manual_seed(42)
    x = torch.randn(17, 1280, device="npu", dtype=torch.bfloat16)
    with set_current_vllm_config(VllmConfig()):
        op = AscendSiluAndMul()
    assert torch.equal(op.forward_oot(x), torch_npu.npu_swiglu(x))
    op.materialize_bf16_silu = True
    gate, up = x.chunk(2, -1)
    gate = gate.float()
    expected = ((gate * gate.sigmoid()).bfloat16().float() * up.float()).bfloat16()
    assert torch.equal(op.forward_oot(x), expected)
    fp16 = x.half()
    assert torch.equal(op.forward_oot(fp16), torch_npu.npu_swiglu(fp16))


