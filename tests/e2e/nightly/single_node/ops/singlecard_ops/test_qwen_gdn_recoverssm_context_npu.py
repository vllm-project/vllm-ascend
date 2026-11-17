# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Q=4 GDN verify/commit across partial acceptance and stock-state transfer."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.e2e.nightly.single_node.ops.singlecard_ops.triton.gdn_recoverssm_test_utils import (  # noqa: E402
    check_accuracy,
    cpu_reference,
    model_inputs,
    npu_device,
)
from vllm_ascend.attention import recoverssm_gdn as runtime  # noqa: E402
from vllm_ascend.ops.triton.gdn.gdn_recoverssm_verify import gdn_recoverssm_verify  # noqa: E402

npu = pytest.fixture(npu_device, name="npu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp3_partial_acceptance_survives_stock_state_transfer(npu, dtype, monkeypatch, record_property):
    monkeypatch.setattr(runtime, "is_conv_state_dim_first", lambda: False)
    window, hk, hv, key, value, pages = 4, 4, 8, 128, 128, 5
    channels = 2 * hk * key + hv * value
    # Simulate receipt of only the stock KV state, without producer replay records.
    producer_conv = torch.randn(pages, 6, channels, device=npu, dtype=torch.bfloat16)
    producer_state = torch.randn(pages, hv, value, key, device=npu, dtype=dtype) * 0.02
    layer = SimpleNamespace(kv_cache=(producer_conv.clone(), producer_state.clone()))
    context = runtime.GDNRecoverSSMCommitContext.create([layer], spec_query_len=window, max_num_reqs=4)
    state_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=npu)
    starts = torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device=npu)
    for step, acceptance in enumerate(([1, 2, 3, 4], [4, 1, 2, 3])):
        checkpoint = layer.kv_cache[1].clone()
        conv_before = layer.kv_cache[0].clone()
        inputs = model_inputs(npu, 16, hk, hv, key, value)
        expected_out, histories, _ = cpu_reference(inputs, checkpoint, [0, 4, 8, 12, 16], [1, 2, 3, 4])
        actual = gdn_recoverssm_verify(
            *inputs,
            checkpoint_state=layer.kv_cache[1],
            replay_cache=layer._qwen_gdn_replay,
            query_start_loc=starts,
            state_indices=state_indices,
            spec_query_len=window,
        )
        torch.testing.assert_close(layer.kv_cache[1], checkpoint, rtol=0, atol=0)
        check_accuracy(actual, expected_out, record_property, f"step{step}/verify")
        context.commit(torch.tensor(acceptance, dtype=torch.int32, device=npu), state_indices, starts)
        for row, count in enumerate(acceptance):
            page = row + 1
            check_accuracy(
                layer.kv_cache[1][page], histories[row][count], record_property, f"step{step}/accepted{count}"
            )
            # Convolution is dim-last in the actual Ascend GDN runtime.
            torch.testing.assert_close(
                layer.kv_cache[0][page, :3], conv_before[page, count - 1 : count + 2], rtol=0, atol=0
            )
        torch.testing.assert_close(layer.kv_cache[1][0], checkpoint[0], rtol=0, atol=0)
    assert context.commit_steps == 2
