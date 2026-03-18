from unittest.mock import patch

import torch
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_ascend._310p.ops.layernorm import AscendRMSNormGated310


def test_rmsnorm_gated_310_forward_oot_uses_forward_native():
    layer = AscendRMSNormGated310(hidden_size=8, eps=1e-5)
    x = torch.randn(2, 8, dtype=torch.float32)
    z = torch.randn(2, 8, dtype=torch.float32)
    expected = torch.randn(2, 8, dtype=torch.float32)

    with patch.object(RMSNormGated, "forward_native", autospec=True, return_value=expected) as mock_forward_native:
        out = layer.forward_oot(x, z)

    mock_forward_native.assert_called_once_with(layer, x, z)
    assert out is expected


def test_rmsnorm_gated_310_forward_oot_uses_forward_native_without_gate():
    layer = AscendRMSNormGated310(hidden_size=8, eps=1e-5)
    x = torch.randn(2, 8, dtype=torch.float32)
    expected = torch.randn(2, 8, dtype=torch.float32)

    with patch.object(RMSNormGated, "forward_native", autospec=True, return_value=expected) as mock_forward_native:
        out = layer.forward_oot(x, None)

    mock_forward_native.assert_called_once_with(layer, x, None)
    assert out is expected
