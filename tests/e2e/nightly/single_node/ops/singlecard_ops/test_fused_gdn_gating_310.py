import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw

torch_npu.npu.set_compile_mode(jit_compile=False)


def npu_fused_gdn_gating_310(
    A_log,
    a,
    b,
    dt_bias,
    beta=1.0,
    threshold=20.0,
):
    """Call FusedGdnGating."""
    # 确保传入 NPU 的张量在内存上是连续的
    if not A_log.is_contiguous():
        A_log = A_log.contiguous()
    if not dt_bias.is_contiguous():
        dt_bias = dt_bias.contiguous()
    g, beta_output = torch.ops._C_ascend.npu_fused_gdn_gating(
        A_log,
        a,
        b,
        dt_bias,
        float(beta),
        float(threshold),
    )
    return g, beta_output


def golden_fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, num_heads = a.shape
    compute_dtype = torch.float32

    # 强制在 CPU 上使用 float32 进行高精度基准计算，防止溢出
    A_log_f = A_log.to(compute_dtype)
    a_f = a.to(compute_dtype)
    b_f = b.to(compute_dtype)
    dt_bias_f = dt_bias.to(compute_dtype)

    A_log_expanded = A_log_f.unsqueeze(0).expand(batch, -1)
    dt_bias_expanded = dt_bias_f.unsqueeze(0).expand(batch, -1)

    x = a_f + dt_bias_expanded
    beta_x = beta * x
    softplus_o = torch.where(
        beta_x <= threshold,
        torch.log1p(torch.exp(beta_x)) / beta,
        x,
    )
    g = -torch.exp(A_log_expanded) * softplus_o
    g = g.unsqueeze(0)
    beta_output = torch.sigmoid(b_f).to(b.dtype)
    beta_output = beta_output.unsqueeze(0)

    return g.to(compute_dtype), beta_output


@pytest.mark.skipif(not is_310p_hw(), reason="Tested separately on a 310P machine.")
@pytest.mark.parametrize("batch_size", [1, 7, 37, 128, 512])
@pytest.mark.parametrize("num_heads", [4, 8, 16, 32, 64])
@pytest.mark.parametrize("beta", [0.5, 1.0])
@pytest.mark.parametrize("threshold", [1.0, 20.0])
def test_fused_gdn_gating_310(batch_size, num_heads, beta, threshold):
    enable_custom_op()
    dtype = torch.float16
    torch.manual_seed(42)
    A_log = torch.randn(num_heads, dtype=dtype)
    dt_bias = torch.randn(num_heads, dtype=dtype)
    a = torch.randn(batch_size, num_heads, dtype=dtype)
    b = torch.randn(batch_size, num_heads, dtype=dtype)
    # 注入边界测试值，强制触发 Softplus 和线性截断分支
    if batch_size >= 4 and num_heads >= 4:
        boundary = threshold / beta
        a[0, 0] = boundary + 2.0
        a[1, 1] = boundary 
        a[2, 2] = boundary - 0.5
        a[3, 3] = -boundary - 2.0
    # 获取 Golden 参考值
    g_golden, beta_out_golden = golden_fused_gdn_gating(A_log, a, b, dt_bias, beta, threshold)
    # 获取 NPU 执行结果
    g_npu, beta_out_npu = npu_fused_gdn_gating_310(A_log.npu(), a.npu(), b.npu(), dt_bias.npu(), beta, threshold)
    # 精度断言验证
    torch.testing.assert_close(
        g_npu.to(torch.float32).cpu(),
        g_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        beta_out_npu.to(torch.float32).cpu(),
        beta_out_golden.to(torch.float32).cpu(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
