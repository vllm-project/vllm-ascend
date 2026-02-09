import torch
from torch import Tensor

from vllm import ir

rms_param = lambda x, weight, epsilon: True

@ir.ops.rms_norm.register_impl(
    "ascend_c", supports_args=rms_param, supported=True
)
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float) -> Tensor:
    x, residual = ir.ops.rms_norm(x, weight, epsilon)
    return x, residual