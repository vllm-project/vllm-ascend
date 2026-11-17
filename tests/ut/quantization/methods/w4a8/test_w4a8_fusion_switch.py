from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.quantization.methods.w4a8.w4a8 import AscendW4A8DynamicFusedMoEMethod


@pytest.mark.parametrize("enabled", [False, True])
def test_w4a8_respects_fusion_switch(enabled):
    method = AscendW4A8DynamicFusedMoEMethod.__new__(AscendW4A8DynamicFusedMoEMethod)
    config = SimpleNamespace(ascend_fusion_config=SimpleNamespace(fusion_ops_gmmswigluquant=enabled))
    with patch("vllm_ascend.quantization.methods.w4a8.w4a8.get_ascend_config", return_value=config):
        assert method.supports_fused_activation(MoEActivation.SILU) is enabled
        assert method.supports_fused_activation(MoEActivation.GELU) is False
