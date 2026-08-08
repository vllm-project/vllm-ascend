# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend._310p.ops.swiglu_quant import swiglu_quant_310p


def test_swiglu_quant_matches_reference() -> None:
    gate_up = torch.tensor(
        [[10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 70.0, -80.0]],
        dtype=torch.float16,
    )
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = gate.clamp(max=10.0)
    up = up.clamp(min=-10.0, max=10.0)
    expected = (F.silu(gate) * up).to(torch.float16)
    expected_quantized = torch.full_like(expected, 7, dtype=torch.int8)
    expected_scale = torch.tensor([[0.25]])

    with patch(
        "torch_npu.npu_dynamic_quant",
        return_value=(expected_quantized, expected_scale),
        create=True,
    ) as dynamic_quant:
        quantized, scale = swiglu_quant_310p(gate_up, clamp_limit=10.0)

    torch.testing.assert_close(dynamic_quant.call_args.args[0], expected)
    torch.testing.assert_close(quantized, expected_quantized)
    torch.testing.assert_close(scale, expected_scale)


def test_swiglu_quant_rejects_unverified_variant() -> None:
    with pytest.raises(NotImplementedError):
        swiglu_quant_310p(torch.zeros(1, 4), glu_alpha=1.702)
