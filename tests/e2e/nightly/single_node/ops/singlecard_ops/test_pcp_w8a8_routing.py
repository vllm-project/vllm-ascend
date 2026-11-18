# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu

HIDDEN_SIZE = 4096
NUM_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 6


@pytest.mark.parametrize("num_tokens", [1, 17, 1024, 8192])
def test_prequantized_pcp_routing_matches_fused_quantization(num_tokens):
    """Quantizing before PCP gather preserves routing and INT8 precision."""
    torch.manual_seed(7721 + num_tokens)
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16, device="npu")
    # Padding rows must remain well-defined through either quantizer.
    if num_tokens > 1:
        hidden_states[0].zero_()
    topk_ids = torch.stack([torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(num_tokens)]).to(
        device="npu", dtype=torch.int32
    )
    quantized, scale = torch_npu.npu_dynamic_quant(hidden_states)
    routing_args = dict(
        active_num=num_tokens * TOP_K,
        expert_num=NUM_EXPERTS,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, LOCAL_EXPERTS],
    )
    reference = torch_npu.npu_moe_init_routing_v2(hidden_states, topk_ids, quant_mode=1, **routing_args)
    actual = torch_npu.npu_moe_init_routing_v2(quantized, topk_ids, scale=scale, quant_mode=-1, **routing_args)
    # Routing allocates capacity for all experts; only the local rows are valid.
    local_tokens = int(reference[2].sum().item())
    torch.testing.assert_close(actual[1], reference[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], reference[2], rtol=0, atol=0)
    # The standalone and fused CANN quantizers can round differently near
    # quantization boundaries, within one INT8 quantization step.
    torch.testing.assert_close(actual[0][:local_tokens], reference[0][:local_tokens], rtol=0, atol=1)
    torch.testing.assert_close(actual[3][:local_tokens], reference[3][:local_tokens], rtol=1e-6, atol=1e-8)
