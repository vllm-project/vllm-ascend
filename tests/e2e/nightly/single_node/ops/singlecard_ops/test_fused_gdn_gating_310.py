import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

torch_npu.npu.set_compile_mode(jit_compile=False)


def npu_fused_gdn_gating_v310(a, b, A_log, dt_bias, beta=1.0, threshold=20.0):
    """Call FusedGdnGatingV310 custom operator on Ascend NPU."""
    out_g, out_beta = torch.ops._C_ascend.npu_fused_gdn_gating_310(a, b, A_log, dt_bias, float(beta), float(threshold))
    return out_g, out_beta


def golden_fused_gdn_gating_v310(a, b, A_log, dt_bias, beta=1.0, threshold=20.0):
    """Calculate golden reference using pure PyTorch in FP32."""
    compute_dtype = torch.float32

    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    A_log_f = A_log.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    batch, num_heads = a_f.shape

    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    x = a_f + dt_bias_expanded
    beta_x = beta * x
    
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    g = -torch.exp(A_log_expanded) * softplus_x
    g_out = g.unsqueeze(0)

    beta_output = torch.sigmoid(b_f).to(b.dtype)
    beta_output_out = beta_output.unsqueeze(0)

    return g_out.to(torch.float32), beta_output_out

@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
@pytest.mark.parametrize("num_heads", [12, 64, 128, 256])
def test_fused_gdn_gating_v310(batch_size, num_heads):
    """Test FusedGdnGatingV310 correctness against PyTorch golden implementation."""
    enable_custom_op()
    torch.manual_seed(2026)
    dtype = torch.float16
    beta = 1.0
    threshold = 20.0

    a = (torch.rand((batch_size, num_heads), dtype=dtype) * 20.0) - 10.0
    b = (torch.rand((batch_size, num_heads), dtype=dtype) * 20.0) - 10.0
    A_log = (torch.rand((num_heads,), dtype=dtype) * 20.0) - 10.0
    dt_bias = (torch.rand((num_heads,), dtype=dtype) * 20.0) - 10.0

    g_golden, beta_output_golden = golden_fused_gdn_gating_v310(a, b, A_log, dt_bias, beta, threshold)

    g_npu, beta_output_npu = npu_fused_gdn_gating_v310(a.npu(), b.npu(), A_log.npu(), dt_bias.npu(), beta, threshold)
    g_npu_cpu = g_npu.cpu().to(torch.float32)
    beta_output_npu_cpu = beta_output_npu.cpu().to(torch.float32)
    beta_output_golden = beta_output_golden.to(torch.float32)

    torch.testing.assert_close(
        g_npu_cpu, g_golden, rtol=3e-3, atol=1e-2, equal_nan=True, msg="Gating (g) output mismatch!"
    )
    torch.testing.assert_close(
        beta_output_npu_cpu,
        beta_output_golden,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
        msg="Beta output (sigmoid) mismatch!",
    )
