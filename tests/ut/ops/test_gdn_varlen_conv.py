# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.ops.triton.mamba import causal_conv1d as causal_conv1d_module


def test_varlen_conv_launch_matches_pinned_vllm_kernel_abi(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeKernel:
        def __getitem__(self, grid):
            captured["grid"] = grid

            def launch(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

            return launch

    monkeypatch.setattr(causal_conv1d_module, "_causal_conv1d_update_kernel", FakeKernel())

    x = torch.empty(4, 256)
    weight = torch.empty(4, 256)
    conv_state = torch.empty(2, 6, 256)
    state_indices = torch.tensor([0, 1], dtype=torch.int64)
    num_accepted_tokens = torch.tensor([1, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    out = torch.empty_like(x)

    result = causal_conv1d_module.causal_conv1d_update_varlen(
        x,
        conv_state,
        weight,
        conv_state_indices=state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=4,
        out=out,
    )

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert result is out
    assert args[7] is None  # block_idx_last_scheduled_token
    assert args[8] is None  # initial_state_idx
    assert args[9] is out
    assert kwargs["IS_APC_ENABLED"] is False
    assert kwargs["launch_pdl"] is False
