import gc
import random

import numpy as np
import pytest
import torch
from torch import nn

from vllm_ascend.utils import enable_custom_op

enable_custom_op()
seed = 45
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



def npu_matmul_gelu_golden(x,
                            weight,
                            bias):
    res= torch.nn.functional.gelu(torch.mm(x, weight) + bias)
    return res


relative_tol = 1e-3
absolute_tol = 1e-3
error_tol = 1e-3


def verify_result(output, golden):
    output = output.cpu().detach().numpy().reshape(-1)
    golden = golden.cpu().detach().numpy().reshape(-1)
    different_element_results = np.isclose(output,
                                           golden,
                                           rtol=relative_tol,
                                           atol=absolute_tol,
                                           equal_nan=True)
    different_element_indexes = np.where(different_element_results == False)[0]
    error_ratio = float(different_element_indexes.size) / golden.size
    return error_ratio <= error_tol


@pytest.mark.parametrize(
    'm',
    [1, 3, 16, 64, 128, 255, 8192]
)
@pytest.mark.parametrize(
    'n',
    [
        512,
        1024,
        4096,
    ],
)
@pytest.mark.parametrize(
    'k',
    [
        128,
        256,
        512,
        1024,
    ],
)

def test_matmul_gelu(m: int, n: int, k: int):

    x = torch.randn(m, k).half().npu()
    weight = torch.randn(n, k).half().npu()
    weight_t = weight.transpose(1, 0).contiguous()
    bias = torch.randn(n).half().npu()
    torch.npu.synchronize()
    y = torch.ops._C_ascend.matmul_gelu(x,
                                      weight,
                                      bias)
    torch.npu.synchronize()

    y1 = npu_matmul_gelu_golden(x,
                                weight_t,
                                bias)
    torch.npu.synchronize()
    assert verify_result(y, y1)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()