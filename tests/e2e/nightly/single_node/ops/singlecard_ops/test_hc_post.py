import gc

import pytest
import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped]  # noqa: F401, E402

torch_npu.npu.config.allow_internal_format = True
enable_custom_op()


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mix_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("batch", "sequence", "heads", "hidden_size"),
    [
        (1, 1, 4, 2048),
        (1, 1, 4, 65),
        (2, 17, 4, 4096),
        (1, 3, 4, 4161),
        (1, 80, 4, 5120),
        (1, 96, 4, 7168),
        (1, 3, 2, 4161),
        (1, 1, 2, 2048),
    ],
)
def test_npu_hc_post_matches_ordered_fp32_reference(batch, sequence, heads, hidden_size, dtype, mix_dtype):
    """Fixed and runtime head counts must preserve FP32 accumulation across D tiles."""
    torch.manual_seed(1024)
    x = torch.randn(batch, sequence, hidden_size, dtype=dtype)
    residual = torch.randn(batch, sequence, heads, hidden_size, dtype=dtype)
    post = torch.randn(batch, sequence, heads, dtype=mix_dtype)
    comb = torch.randn(batch, sequence, heads, heads, dtype=mix_dtype)

    expected = x.float().unsqueeze(-2) * post.float().unsqueeze(-1)
    for head in range(heads):
        expected.add_(residual[..., head, :].float().unsqueeze(-2) * comb[..., head, :].float().unsqueeze(-1))
    expected = expected.to(dtype)

    actual = torch.ops._C_ascend.npu_hc_post(x.npu(), residual.npu(), post.npu(), comb.npu()).cpu()
    assert actual.shape == residual.shape
    assert actual.dtype == dtype
    tolerance = {torch.float16: 1e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    gc.collect()
    torch.npu.empty_cache()
