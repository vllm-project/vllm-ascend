# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.gdn.gdn_recoverssm_verify import gdn_recoverssm_verify
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@pytest.fixture
def npu():
    torch.npu.set_device(0)
    init_device_properties_triton()
    return torch.device("npu")


def verify_one_token(npu, qk_dtype, gate_dtype, raw_b, normalize):
    heads, dim = 2, 128
    q = torch.zeros(1, 1, 1, dim, device=npu, dtype=qk_dtype)
    q[..., 0] = 0.125 if normalize else 1.0
    k = q.clone()
    v = torch.ones(1, 1, heads, dim, device=npu)
    state = torch.zeros(2, heads, dim, dim, device=npu)
    replay = torch.empty(2, heads, 1, 2 * dim + 1, device=npu)
    b = torch.full((1, heads), raw_b, device=npu, dtype=gate_dtype)
    out = gdn_recoverssm_verify(
        torch.zeros(heads, device=npu),
        torch.zeros_like(b),
        b,
        torch.zeros(heads, device=npu),
        q,
        k,
        v,
        checkpoint_state=state,
        replay_cache=replay,
        query_start_loc=torch.tensor([0, 1], device=npu, dtype=torch.int32),
        state_indices=torch.tensor([1], device=npu, dtype=torch.int32),
        spec_query_len=1,
        scale=1.0,
        use_qk_l2norm_in_kernel=normalize,
    )
    torch.testing.assert_close(state, torch.zeros_like(state), atol=0, rtol=0)
    return out, replay[1, :, 0, :dim], b


@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
def test_beta_preserves_native_gating_dtype(npu, gate_dtype):
    # Zero state and one-hot K make the correction exactly sigmoid(b).
    # A FP32 output exposes missing BF16 rounding without output cast masking it.
    out, correction, b = verify_one_token(npu, torch.float32, gate_dtype, 0.082, False)
    expected = torch.sigmoid(b.cpu().float()).to(gate_dtype).float()
    torch.testing.assert_close(out.cpu(), expected[None, :, :, None].expand_as(out.cpu()), atol=1e-7, rtol=0)
    torch.testing.assert_close(correction.cpu(), expected.T.expand_as(correction.cpu()), atol=1e-7, rtol=0)


def test_qk_normalization_preserves_activation_rounding(npu):
    # Native l2norm stores the normalized one-hot Q/K as BF16 ones. Keeping
    # normalization's FP32 intermediates instead produces an output below one.
    out, _, _ = verify_one_token(npu, torch.bfloat16, torch.float32, 100.0, True)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)
