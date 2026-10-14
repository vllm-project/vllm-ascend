# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device import device_op
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention, _get_fla_gdn_prefill_op


@pytest.mark.parametrize("chunk_size", [64, 128])
@pytest.mark.parametrize("lengths", [(1,), (65, 257), (0, 65, 0, 257, 0)])
@pytest.mark.parametrize("with_initial_state", [False, True])
@torch.inference_mode()
def test_fla_prefill_matches_torch_npu(monkeypatch, chunk_size, lengths, with_initial_state):
    """Compare outputs AND cache-layout states across fused backends on NPU."""
    config = AscendConfig(sparse_kv_offload_config=None, fla_chunk_size=chunk_size)
    monkeypatch.setattr(device_op, "get_ascend_config", lambda: config)
    fused_fwd = _get_fla_gdn_prefill_op()
    if fused_fwd is None:
        pytest.skip("FLA fused prefill is not installed")
    if not AscendGatedDeltaNetAttention._probe_fused_chunk():
        pytest.skip("torch_npu fused prefill unavailable on this CANN/device")
    torch.manual_seed(42)
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    keep = [i for i, length in enumerate(lengths) if length]
    compact = tuple([0] + [cu[i + 1] for i in keep])
    keep_device = torch.tensor(keep, dtype=torch.long, device="npu")
    meta = SimpleNamespace(
        cu_seqlens_host=tuple(cu),
        cu_seqlens_kern=compact if len(keep) != len(lengths) else None,
        keep_meta=keep_device if len(keep) != len(lengths) else None,
    )
    q = torch.randn(1, cu[-1], 2, 128, device="npu", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, cu[-1], 4, 128, device="npu", dtype=torch.bfloat16)
    g = -torch.rand(1, cu[-1], 4, device="npu", dtype=torch.float32) * 0.1
    beta = torch.rand_like(g).to(torch.bfloat16)
    state = torch.zeros(len(lengths), 4, 128, 128, device="npu", dtype=torch.bfloat16)
    if with_initial_state:
        state.normal_(std=0.1)
    original_state = state.clone()
    output, final = DeviceOperator.fla_gdn_prefill(q, k, v, g, beta, state, 128**-0.5, meta, fused_fwd)
    # OPS only receives nonempty rows; scatter them to the original cache layout.
    expected_output, compact_state = AscendGatedDeltaNetAttention._chunk_gated_delta_rule_fused(
        q,
        k,
        v,
        g,
        beta,
        original_state[keep_device],
        torch.tensor(compact, dtype=torch.int32, device="npu"),
        128**-0.5,
    )
    expected_state = original_state.clone()
    expected_state[keep_device] = compact_state.to(expected_state.dtype)
    torch.npu.synchronize()
    torch.testing.assert_close(state, original_state, rtol=0, atol=0)
    # BF16 cross-kernel tolerance; this is operator parity, not model accuracy.
    torch.testing.assert_close(output.float(), expected_output.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(final.float(), expected_state.float(), rtol=1e-2, atol=1e-2)
