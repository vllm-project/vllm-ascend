from vllm_ascend.ops.fused_moe.moe_utils import _get_cann_mega_moe_quant_settings
from vllm_ascend.quantization.quant_type import QuantType


def test_cann_mega_moe_quant_settings_none():
    assert _get_cann_mega_moe_quant_settings(QuantType.NONE) == (0, None, None)
