import importlib.util

import pytest
import torch


def test_attention_sigmoid_mul_operator_exists():
    assert importlib.util.find_spec("vllm_ascend.ops.triton.attention_sigmoid_mul") is not None


@pytest.mark.parametrize("shape", [(0, 4096), (1, 4096), (4, 4096), (16, 2051)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fake_has_fresh_contiguous_output(shape, dtype):
    from vllm_ascend.ops.triton.attention_sigmoid_mul import attention_sigmoid_mul_fake

    x = torch.empty(shape, dtype=dtype)
    g = torch.empty_like(x)
    y = attention_sigmoid_mul_fake(x, g)
    assert y.shape == x.shape and y.dtype == x.dtype and y.is_contiguous()
    if x.numel():
        assert y.data_ptr() != x.data_ptr() and y.data_ptr() != g.data_ptr()


@pytest.mark.parametrize(
    "x,g",
    [
        (torch.empty(4), torch.empty(4)),
        (torch.empty(2, 4), torch.empty(1, 4)),
        (torch.empty(2, 4), torch.empty(2, 4, dtype=torch.float16)),
        (torch.empty(2, 4, dtype=torch.int32), torch.empty(2, 4, dtype=torch.int32)),
    ],
)
def test_rejects_unsupported_inputs(x, g):
    from vllm_ascend.ops.triton.attention_sigmoid_mul import _validate_inputs

    with pytest.raises(ValueError):
        _validate_inputs(x, g)


def test_registered_fake_op():
    from torch._subclasses.fake_tensor import FakeTensorMode

    import vllm_ascend.ops.triton.attention_sigmoid_mul  # noqa: F401

    with FakeTensorMode():
        x = torch.empty(4, 4096, device="npu", dtype=torch.bfloat16)
        out = torch.ops.vllm.ascend_attention_sigmoid_mul(x, x)
        assert out.shape == x.shape and out.device == x.device and out.dtype == x.dtype
