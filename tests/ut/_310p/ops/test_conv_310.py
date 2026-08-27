from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.conv import Conv3dLayer

from vllm_ascend._310p.ops.conv import AscendConv3dLayer310


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]
    with set_current_vllm_config(mock_config):
        yield mock_config


def test_conv3d_310_forward_oot_uses_forward_native():
    layer = AscendConv3dLayer310(
        in_channels=2,
        out_channels=4,
        kernel_size=(2, 2, 2),
        stride=(2, 2, 2),
        bias=True,
        params_dtype=torch.float32,
    )
    x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float32)
    expected = torch.randn(1, 4, 2, 2, 2, dtype=torch.float32)

    with patch.object(Conv3dLayer, "forward_native", autospec=True, return_value=expected) as mock_forward_native:
        out = layer.forward_oot(x)

    mock_forward_native.assert_called_once_with(layer, x)
    assert out is expected
