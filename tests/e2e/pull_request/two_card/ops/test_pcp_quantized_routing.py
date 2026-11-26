# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import create_forward_context, get_forward_context, override_forward_context
from vllm.utils.network_utils import get_open_port

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.moe_quant import MoEMxfpParams, MoEQuantParams
from vllm_ascend.ops.fused_moe.dataclass.router_input import MoeRouterInput
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_utils import maybe_normalize_mxfp_scale_layout
from vllm_ascend.ops.fused_moe.prepare_finalize import PrepareAndFinalizeWithAllGather
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather
from vllm_ascend.quantization.quant_type import QuantType
from vllm_ascend.quantization.utils import DynamicMXQuantScaleAlg
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _compare_routing(rank, counts, quant, hidden_size=512):
    # Vary magnitude across ranks and blocks so swapping scales changes the output.
    torch.manual_seed(22230 + rank)
    x = torch.randn(counts[rank], hidden_size, dtype=torch.bfloat16)
    block_magnitudes = (2.0 ** (torch.arange(hidden_size // 32) % 7 - 3)).repeat_interleave(32)
    x = (x * block_magnitudes).bfloat16().npu()
    logits = torch.randn(counts[rank], 8).npu()
    _EXTRA_CTX.max_tokens_across_pcp = max(counts)
    layer = PrepareAndFinalizeWithAllGather(SimpleNamespace(is_sequence_parallel=False, pcp_size=2, dp_size=1))
    baseline = layer.prepare(x, logits)
    actual = layer.prepare(x, logits, quant_type=quant.quant_type)
    torch.testing.assert_close(actual.router_logits, baseline.router_logits, rtol=0, atol=0)
    assert actual.hidden_states.dtype == quant.get_dst_type
    assert actual.pertoken_scale is not None

    # All experts are local, so every gathered row (including padding) is checked after routing.
    dispatcher = TokenDispatcherWithAllGather(top_k=2, num_local_experts=8)
    weights, ids = baseline.router_logits.topk(2)

    def dispatch(prepared):
        return dispatcher.token_dispatch(
            MoETokenDispatchInput(
                hidden_states=prepared.hidden_states,
                topk_weights=weights,
                topk_ids=ids.int(),
                routing=MoeRouterInput(None, 0, None, False, pertoken_scale=prepared.pertoken_scale),
                quant=quant,
            )
        )

    routed, expected = dispatch(actual), dispatch(baseline)
    torch.testing.assert_close(routed.combine_metadata.expanded_row_idx, expected.combine_metadata.expanded_row_idx)
    torch.testing.assert_close(routed.group_list, expected.group_list)
    if quant.is_mxfp:
        torch.testing.assert_close(routed.hidden_states.view(torch.uint8), expected.hidden_states.view(torch.uint8))
        expected_scale = maybe_normalize_mxfp_scale_layout(expected.dynamic_scale)
        torch.testing.assert_close(routed.dynamic_scale.view(torch.uint8), expected_scale.view(torch.uint8))
    else:
        # Standalone and fused INT8 quantizers can differ by one rounding step.
        torch.testing.assert_close(routed.hidden_states, expected.hidden_states, rtol=0, atol=1)
        torch.testing.assert_close(routed.dynamic_scale, expected.dynamic_scale, rtol=1e-6, atol=1e-8)


def _worker(rank, port):
    torch.npu.set_device(rank)
    config = VllmConfig()
    with set_current_vllm_config(config):
        init_ascend_config(config)
        init_distributed_environment(2, rank, f"tcp://127.0.0.1:{port}", rank, backend="hccl")
        try:
            initialize_model_parallel(prefill_context_model_parallel_size=2, backend="hccl")
            with override_forward_context(create_forward_context(None, config)):
                int8 = MoEQuantParams(quant_type=QuantType.W8A8)
                _compare_routing(rank, (0, 5), int8)
                _compare_routing(rank, (4096, 4095), int8, hidden_size=4096)
                if get_ascend_device_type() == AscendDeviceType.A5:
                    mxfp8 = MoEQuantParams(
                        quant_type=QuantType.W8A8MXFP, mxfp=MoEMxfpParams(act_quant_type=torch.float8_e4m3fn)
                    )
                    for scale_alg in DynamicMXQuantScaleAlg:
                        get_forward_context().additional_kwargs["dynamic_mx_quant_scale_alg"] = scale_alg
                        for counts in ((17, 9), (0, 5)):
                            _compare_routing(rank, counts, mxfp8)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


@pytest.mark.skipif(get_ascend_device_type() == AscendDeviceType._310P, reason="310P uses a separate MoE path")
def test_pcp_quantization_matches_dispatch_after_gather():
    mp.spawn(_worker, args=(get_open_port(),), nprocs=2)
