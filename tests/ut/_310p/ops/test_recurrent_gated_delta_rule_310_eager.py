import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm_ascend._310p.ops.fla.fused_recurrent_gated_delta_rule import (
    fused_recurrent_gated_delta_rule_pytorch,
)
from vllm_ascend.patch.worker.patch_qwen3_5_310 import (
    npu_recurrent_gated_delta_rule_310,
)
from vllm_ascend.utils import enable_custom_op, is_310p

torch_npu.npu.set_compile_mode(jit_compile=False)


pytestmark = [
    pytest.mark.skipif(not hasattr(torch, "npu"), reason="torch.npu is required"),
    pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device is required"),
    pytest.mark.skipif(not is_310p(), reason="310P device is required"),
]


def _build_inputs(
    *,
    active_tokens: int = 2,
    padded_tokens: int = 4,
    num_states: int = 8,
    h: int = 1,
    hv: int = 1,
    dk: int = 128,
    dv: int = 128,
):
    q = torch.randn((1, padded_tokens, h, dk), device="npu", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn((1, padded_tokens, hv, dv), device="npu", dtype=torch.float16)
    beta = torch.randn((1, padded_tokens, hv), device="npu", dtype=torch.float16)
    state = torch.randn((num_states, hv, dv, dk), device="npu", dtype=torch.float32)
    cu_seqlens = torch.arange(active_tokens + 1, device="npu", dtype=torch.int32)
    ssm_state_indices = torch.tensor(
        [2, 5] + [-1] * (padded_tokens - active_tokens),
        device="npu",
        dtype=torch.int32,
    )
    return q, k, v, beta, state, cu_seqlens, ssm_state_indices


def test_recurrent_gated_delta_rule_310_eager_matches_reference_with_padded_tail():
    enable_custom_op()

    q, k, v, beta, state, cu_seqlens, ssm_state_indices = _build_inputs()

    out = npu_recurrent_gated_delta_rule_310(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=None,
        beta=beta.clone(),
        state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )

    ref_state = state.clone()
    ref_out, ref_state = fused_recurrent_gated_delta_rule_pytorch(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=None,
        beta=beta.clone(),
        initial_state=ref_state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    )

    active_token_count = int((cu_seqlens[1:] - cu_seqlens[:-1]).sum().item())
    torch.testing.assert_close(
        out[:, :active_token_count].cpu(),
        ref_out[:, :active_token_count].cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state.cpu(),
        ref_state.cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
