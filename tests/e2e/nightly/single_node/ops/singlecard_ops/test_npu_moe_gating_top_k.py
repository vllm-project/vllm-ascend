import random

import numpy
import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# Fix random seed to ensure test reproducibility
seed = 45
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)


def softmax_func(x, axis=None):
    """Softmax implementation (adapted for numpy calculation)"""
    if "float16" in x.dtype.name:
        x = x.astype(numpy.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = numpy.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    res = y / x_sum
    return res, x_max, x_sum


def moe_gating_top_k_numpy_ref(x: torch.Tensor,
                               k: int,
                               bias: torch.Tensor | None,
                               k_group: int = 1,
                               group_count: int = 1,
                               group_select_mode: int = 0,
                               renorm: int = 0,
                               norm_type: int = 0,
                               y2_flag: bool = False,
                               routed_scaling_factor: float = 1.0,
                               eps: float = 1e-20) -> tuple:
    """NumPy reference implementation of MOE Gating TopK (for result comparison with NPU operator)"""
    dtype = x.dtype
    if dtype != torch.float32:
        x = x.to(dtype=torch.float32)
        if bias is not None:
            bias = bias.to(dtype=torch.float32)

    x = x.numpy()
    if bias is not None:
        bias = bias.numpy()

    if norm_type == 0:  # softmax normalization
        x, _, _ = softmax_func(x, -1)
    else:  # sigmoid normalization
        x = 1 / (1 + numpy.exp(-x))

    original_x = x
    if bias is not None:
        x = x + bias

    if group_count > 1:
        x = x.reshape(x.shape[0], group_count, -1)
        if group_select_mode == 0:
            group_x = numpy.amax(x, axis=-1)
        else:
            group_x = numpy.partition(x, -2, axis=-1)[..., -2:].sum(axis=-1)
        indices = numpy.argsort(-group_x, axis=-1, kind='stable')[:, :k_group]

        mask = numpy.ones((x.shape[0], group_count), dtype=bool)
        mask[numpy.arange(x.shape[0])[:, None], indices] = False
        x = numpy.where(mask[..., None], float('-inf'), x)
        x = x.reshape(x.shape[0], -1)

    _, indices = torch.sort(torch.from_numpy(x),
                            dim=-1,
                            stable=True,
                            descending=True)
    indices = numpy.asarray(indices[:, :k])

    y = numpy.take_along_axis(original_x, indices, axis=1)
    if norm_type == 1 or renorm == 1:
        y /= (numpy.sum(y, axis=-1, keepdims=True) + eps)
    y *= routed_scaling_factor

    y2 = original_x if y2_flag else None
    y = torch.tensor(y, dtype=dtype)
    return y, indices.astype(numpy.int32), y2


# pytest parameterized decorators (cover all test scenarios)
@pytest.mark.parametrize("group_select_mode", [0, 1])
@pytest.mark.parametrize("renorm", [1])
@pytest.mark.parametrize("norm_type", [0, 1])
@pytest.mark.parametrize("group_count", [1, 8])
@pytest.mark.parametrize("k_ranges", [4, 8, 12, 16, 6, 32])
@pytest.mark.parametrize("x_dim0_range", list(range(1, 17)))
@pytest.mark.parametrize("x_dim1_range", [256, 128, 64, 208, 192, 160])
def test_npu_moe_gating_topk_compare(group_select_mode: int,
                                     renorm: int,
                                     norm_type: int,
                                     group_count: int,
                                     k_ranges: int,
                                     x_dim0_range: int,
                                     x_dim1_range: int,
                                     device: str = "npu"):
    """Ascend NPU MOE Gating TopK operator test (compare with NumPy reference implementation)"""
    # Simplify parameter names for better readability
    k = k_ranges
    dim0 = x_dim0_range
    dim1 = x_dim1_range

    # Optimization 1: Remove redundant quotes in comments and fix comment format
    if k > dim1 // group_count:
        return

    # Construct test inputs
    x = numpy.random.uniform(-2, 2, (dim0, dim1)).astype(numpy.float32)
    bias = numpy.random.uniform(-2, 2, (dim1, )).astype(numpy.float32)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    bias_tensor = torch.tensor(bias, dtype=torch.float32)
    # Optimization 2: Fix k_group value to avoid irreproducibility caused by random.randint
    k_group = min(1, group_count)
    out_flag = False
    routed_scaling_factor = 1.0
    eps = 1e-20

    # Calculate NumPy reference results
    y, expert_idx, out = moe_gating_top_k_numpy_ref(
        x_tensor,
        k=k,
        bias=bias_tensor,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        y2_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
    )

    # Calculate NPU operator results
    y_npu, expert_idx_npu, out_npu = torch.ops._C_ascend.moe_gating_top_k(
        x_tensor.npu(),
        k=k,
        k_group=k_group,
        group_count=group_count,
        group_select_mode=group_select_mode,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=out_flag,
        routed_scaling_factor=routed_scaling_factor,
        eps=eps,
        bias_opt=bias_tensor.npu() if bias_tensor is not None else None,
    )

    assert numpy.allclose(y.cpu().numpy(), y_npu.cpu().numpy(), rtol=1e-3, atol=1e-5)
    assert numpy.allclose(expert_idx, expert_idx_npu.cpu().numpy(), rtol=1e-3, atol=1e-5)
    
if __name__ == "__main__":
    # Execute pytest tests
    pytest.main([__file__, "-sv"])
