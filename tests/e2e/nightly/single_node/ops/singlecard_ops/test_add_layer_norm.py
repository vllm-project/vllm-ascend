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


def npu_add_layer_norm_golden(input_x1,
                            input_x2,
                            input_gamma,
                            input_beta,
                            epsilon=1e-05):
    ori_x_shape = input_x1.shape
    layerNorm = nn.LayerNorm(ori_x_shape[-1], eps=epsilon)
    layerNorm.weight = nn.Parameter(input_gamma)
    layerNorm.bias = nn.Parameter(input_beta)

    return layerNorm(input_x1 + input_x2)


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

def test_add_layer_norm(m: int, n: int):
    x1 = torch.randn(m, n).half().npu()
    x2 = torch.randn(m, n).half().npu()
    gamma = torch.randn(n).half().npu()
    beta = torch.randn(n).half().npu()
    torch.npu.synchronize()
    y = torch.ops._C_ascend.add_layer_norm(x1,
                                              x2,
                                              gamma,
                                              beta,
                                              1e-5)[0]

    torch.npu.synchronize()

    y1 = npu_add_layer_norm_golden(x1,
                                x2,
                                gamma,
                                beta,
                                1e-05)
    torch.npu.synchronize()
    assert verify_result(y, y1)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
