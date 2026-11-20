# SPDX-License-Identifier: Apache-2.0
"""Guards that keep the A3 INT8 shared-expert fusion on the W8A8 path."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.ops.fused_moe import shared_experts as shared_experts_module
from vllm_ascend.ops.fused_moe.shared_experts import AscendSharedExperts
from vllm_ascend.quantization.quant_type import QuantType

_MODULE = shared_experts_module.__name__


def _make_runner(quant_type: QuantType) -> AscendSharedExperts:
    """Build AscendSharedExperts without __init__; only the fusion state is used."""
    runner = object.__new__(AscendSharedExperts)
    runner.quant_type = quant_type
    runner.moe_config = SimpleNamespace(hidden_dim=4096, intermediate_size_per_partition=2048)
    runner.megamoe_shared_weights_ready = False
    runner._megamoe_shared_fusion_disabled = False
    runner._megamoe_shared_parts = {}
    runner._megamoe_shared_quant_type = QuantType.NONE
    runner._megamoe_shared_fusion = shared_experts_module.MegamoeSharedExpertFusion()
    # Mirror the remaining __init__ fields so this helper cannot drift again.
    runner._megamoe_shared_weight_dtype = None
    runner._megamoe_shared_fusion_platform = "A3/INT8"
    return runner


def _shared_linear(weight_dtype: torch.dtype, shared_quant_type: QuantType) -> SimpleNamespace:
    return SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(4096, 4096, dtype=weight_dtype)),
        quant_method=SimpleNamespace(quant_type=shared_quant_type),
    )


def test_setup_rejects_w4a8_routed_experts() -> None:
    """The routed W4A8 MegaMoe path must not arm the shared fusion."""
    runner = _make_runner(QuantType.W4A8)
    with (
        patch(f"{_MODULE}.use_cann_megamoe", return_value=True),
        patch(f"{_MODULE}.get_current_vllm_config", return_value=object()),
        patch(
            f"{_MODULE}.get_ascend_config",
            return_value=SimpleNamespace(megamoe_shared_expert_fusion=True),
        ),
        patch(
            f"{_MODULE}.get_current_hardware_profile",
            return_value=get_hardware_profile(AscendDeviceType.A3),
        ),
    ):
        runner._setup_megamoe_shared_fusion()

    assert runner._megamoe_shared_fusion_disabled is True


def test_collect_rejects_packed_int32_shared_weight() -> None:
    """A W4A8 shared linear materializes int32 packed weights: keep the separate forward."""
    runner = _make_runner(QuantType.W8A8)
    runner._collect_megamoe_shared_weight("w1", _shared_linear(torch.int32, QuantType.W4A8))

    assert runner._megamoe_shared_fusion_disabled is True
    assert runner._megamoe_shared_parts == {}


def test_collect_rejects_w4a8_shared_quant_type() -> None:
    """An int8 buffer that declares W4A8 does not match: reject it."""
    runner = _make_runner(QuantType.W8A8)
    runner._collect_megamoe_shared_weight("w1", _shared_linear(torch.int8, QuantType.W4A8))

    assert runner._megamoe_shared_fusion_disabled is True
    assert runner._megamoe_shared_parts == {}
