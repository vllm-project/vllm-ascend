import torch
import vllm.distributed
import vllm.model_executor.layers.utils
from vllm.distributed.parallel_state import get_tp_group
from vllm.utils.torch_utils import direct_register_custom_op


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output_shape = (x.shape[0], weight.shape[0])
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(op_name="unquantized_gemm",
                          op_func=unquantized_gemm,
                          fake_impl=unquantized_gemm_fake,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.unquantized_gemm(x, weight, bias)


vllm.model_executor.layers.utils.default_unquantized_gemm = default_unquantized_gemm
