import random

import numpy as np
import pytest
import torch
import torch_npu

import vllm_ascend  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
    chunk_gated_delta_rule_pytorch,
)


torch_npu.npu.set_compile_mode(jit_compile=False)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("use_gamma", [False, True])
@pytest.mark.parametrize("lengths", [(64,), (65,), (129,), (64, 17), (65, 64)])
def test_chunk_gated_delta_rule_fp32_state_on_ascend910b(
    state_dtype, use_gamma, lengths
):
    seed = 20260802 + sum(lengths) + len(lengths) * 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    nk, nv, dk, dv = 4, 8, 128, 128
    tokens = sum(lengths)
    query = torch.nn.functional.normalize(torch.randn(tokens, nk, dk), dim=-1).to(
        torch.bfloat16
    )
    key = torch.nn.functional.normalize(torch.randn(tokens, nk, dk), dim=-1).to(
        torch.bfloat16
    )
    value = (0.1 * torch.randn(tokens, nv, dv)).to(torch.bfloat16)
    beta = (0.1 + 0.4 * torch.rand(tokens, nv)).to(torch.bfloat16)
    gamma = -0.01 * torch.rand(tokens, nv, dtype=torch.float32)
    gamma_input = gamma if use_gamma else torch.zeros_like(gamma)
    initial_state = (0.1 * torch.randn(len(lengths), nv, dv, dk)).to(state_dtype)
    cu_seqlens = torch.tensor([0, *np.cumsum(lengths).tolist()], dtype=torch.int64)
    actual_seq_lengths = torch.tensor(lengths, dtype=torch.int32)
    scale = dk ** -0.5

    golden_out, golden_state = chunk_gated_delta_rule_pytorch(
        query,
        key,
        value,
        gamma_input,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    )
    assert golden_state is not None
    golden_state = golden_state.to(state_dtype)

    device = torch.device("npu:0")
    npu_out, npu_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        query.to(device),
        key.to(device),
        value.to(device),
        beta.to(device),
        initial_state.to(device),
        actual_seq_lengths.to(device),
        gamma.to(device) if use_gamma else None,
        scale,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(
        npu_out.float().cpu(),
        golden_out.float(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        npu_state.float().cpu(),
        golden_state.float(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
