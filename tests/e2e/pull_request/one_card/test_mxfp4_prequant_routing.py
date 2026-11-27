# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch_npu
from vllm.config import VllmConfig, set_current_vllm_config

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_utils import maybe_normalize_mxfp_scale_layout
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,hidden", [(1, 128), (17, 128), (128, 4096)])
@torch.inference_mode()
def test_mxfp4_prequant_matches_quantizing_expanded_rows(dtype, rows, hidden):
    if not any(tag in torch.npu.get_device_name(0).lower() for tag in ("950", "910_95")):
        pytest.skip("MXFP4 routing requires Ascend A5")
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_ascend_config(config)
        torch.manual_seed(1024)
        x = torch.randn(rows, hidden, device="npu", dtype=dtype)
        x[0].zero_()
        logits = torch.randn(rows, 8, device="npu")
        weights, ids = logits.softmax(-1).topk(2)
        ids = ids.to(torch.int32)
        prepare = object.__new__(PrepareAndFinalizeWithAllGather)
        prepare.moe_config = SimpleNamespace(is_sequence_parallel=False, dp_size=1, pcp_size=1)
        baseline = prepare.prepare(x, logits)
        candidate = prepare.prepare(x, logits, quant_type=QuantType.W4A4MXFP)
        dispatcher = TokenDispatcherWithAllGather(top_k=2, num_local_experts=8)
        quant = MoEQuantParams(
            quant_type=QuantType.W4A4MXFP,
            mxfp=MoEMxfpParams(act_quant_type=torch_npu.float4_e2m1fn_x2, use_bf16=dtype == torch.bfloat16),
        )

        def dispatch(prepared):
            return dispatcher.token_dispatch(
                MoETokenDispatchInput(
                    hidden_states=prepared.hidden_states,
                    topk_weights=weights,
                    topk_ids=ids,
                    routing=MoeRouterInput(None, 0, None, False, prepared.pertoken_scale),
                    quant=quant,
                )
            )

        expected, actual = dispatch(baseline), dispatch(candidate)
        expected_x, expected_scale = torch_npu.npu_dynamic_mx_quant(
            expected.hidden_states, dst_type=torch_npu.float4_e2m1fn_x2
        )
        expected_scale = maybe_normalize_mxfp_scale_layout(expected_scale)
        assert actual.dynamic_scale is not None
        torch.testing.assert_close(actual.hidden_states.view(torch.uint8), expected_x.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(
            actual.dynamic_scale.view(torch.uint8), expected_scale.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual.group_list, expected.group_list, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.combine_metadata.expanded_row_idx, expected.combine_metadata.expanded_row_idx, rtol=0, atol=0
        )
