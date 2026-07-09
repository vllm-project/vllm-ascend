import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _to_dtype(data, dtype, device=None):
    tensor = torch.tensor(data, dtype=torch.int32)
    tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_scatter_nd_update_contiguous(dtype, index_dtype):
    var = torch.arange(24, dtype=torch.int32).reshape(6, 4).to(dtype).npu()
    indices = torch.tensor([[0], [2], [5]], dtype=index_dtype, device="npu")
    updates = _to_dtype(
        [
            [101, 102, 103, 104],
            [111, 112, 113, 114],
            [121, 122, 123, 124],
        ],
        dtype,
        device="npu",
    )
    expected = var.cpu().clone()
    expected[[0, 2, 5], :] = updates.cpu()

    torch.ops._C_ascend.npu_scatter_nd_update(var, indices, updates)
    torch.npu.synchronize()

    torch.testing.assert_close(var.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_scatter_nd_update_stride0_view(dtype, index_dtype):
    base = torch.arange(6 * 2 * 4, dtype=torch.int32).reshape(6, 2, 4).to(dtype).npu()
    var = base[:, 0, :]
    assert not var.is_contiguous()
    assert var.stride(0) > var.shape[-1]

    indices = torch.tensor([[1], [4]], dtype=index_dtype, device="npu")
    updates = _to_dtype(
        [
            [31, 32, 33, 34],
            [41, 42, 43, 44],
        ],
        dtype,
        device="npu",
    )
    expected_base = base.cpu().clone()
    expected_base[[1, 4], 0, :] = updates.cpu()

    torch.ops._C_ascend.npu_scatter_nd_update(var, indices, updates)
    torch.npu.synchronize()

    torch.testing.assert_close(base.cpu(), expected_base, atol=0, rtol=0)
