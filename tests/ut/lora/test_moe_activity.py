# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import routed_experts
from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("group_active", [False, True])
def test_expert_inputs_exclude_inactive_lora_but_keep_remote_adapters(quantized, group_active):
    context = SimpleNamespace(punica_wrapper=SimpleNamespace(no_lora=True, moe_lora_active=group_active))
    layer = SimpleNamespace(
        _ascend_moe_lora_context=context,
        ascend_expert_map=None,
        global_redundant_expert_num=0,
        ascend_mc2_mask=None,
        apply_router_weight_on_input=False,
        ascend_pertoken_scale=None,
    )
    if quantized:
        method = object.__new__(w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod)
        method.quant_type = QuantType.W8A8
        method.use_expert_weight_list = False
        module = w8a8_dynamic
    else:
        method = object.__new__(routed_experts.AscendUnquantizedFusedMoEMethod)
        method.dynamic_eplb = False
        module = routed_experts
    comm = SimpleNamespace(fused_experts=Mock())
    with patch.object(module, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=comm)):
        result = method.apply(
            layer,
            torch.ones(1, 4),
            torch.ones(1, 1),
            torch.zeros(1, 1, dtype=torch.int32),
            None,
            None,
        )
    assert result is comm.fused_experts.return_value
    assert comm.fused_experts.call_args.kwargs["fused_experts_input"].lora_context is (
        context if group_active else None
    )
