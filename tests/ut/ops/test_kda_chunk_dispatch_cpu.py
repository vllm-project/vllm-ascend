# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check KDA prefill dispatch without loading an NPU runtime."""

import ast
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch


def _load_dispatch(is_a5):
    path = Path(__file__).parents[3] / "vllm_ascend/ops/kda.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_chunk_kda")
    call = MagicMock(return_value=("output", "final_state"))
    proxy = MagicMock()
    proxy.bfloat16, proxy.float32, proxy.Tensor = torch.bfloat16, torch.float32, torch.Tensor
    proxy.ops._C_ascend.chunk_kda_fwd = call
    norm = MagicMock(side_effect=lambda x: x + 1)
    scope = {"torch": proxy, "Sequence": Sequence, "KDA_CHUNK_SIZE": 64, "l2norm_fwd": norm, "is_950": lambda: is_a5}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["run_chunk_kda"], call, norm


@pytest.mark.parametrize(
    "mode", ["a5", "other_device", "fp32_q", "bf16_beta", "softplus", "empty_sequence", "tensor_lengths"]
)
def test_chunk_prefill_preserves_raw_norm_and_fallback_contracts(mode):
    run, call, norm = _load_dispatch(mode != "other_device")
    dtype = torch.float32 if mode == "fp32_q" else torch.bfloat16
    q = torch.randn(1, 65, 12, 128, dtype=dtype)
    k, v, gate = torch.randn_like(q), torch.randn_like(q), q.to(torch.bfloat16)
    beta = torch.randn(1, 65, 12, dtype=torch.bfloat16 if mode == "bf16_beta" else torch.float32)
    lengths = (0, 65, 65) if mode == "empty_sequence" else (0, 32, 65)
    if mode == "tensor_lengths":
        lengths = torch.tensor(lengths)
    state = torch.empty(2, 12, 128, 128)
    actual = run(
        q,
        k,
        v,
        gate,
        beta,
        state,
        lengths,
        (0, 0, 1, 0),
        torch.empty(12),
        torch.empty(1536),
        lower_bound=None if mode == "softplus" else -5.0,
    )
    assert actual == ("output", "final_state")
    args, kwargs = call.call_args
    assert kwargs["use_qk_l2norm_in_kernel"] is (mode == "a5")
    assert kwargs["cu_seqlens"] is lengths
    assert kwargs["initial_state"] is state
    assert kwargs["state_v_first"] is True
    assert args[4] is beta
    if mode == "a5":
        norm.assert_not_called()
        assert args[0] is q and args[1] is k
    else:
        assert norm.call_count == 2
        torch.testing.assert_close(args[0], q + 1)
        torch.testing.assert_close(args[1], k + 1)
