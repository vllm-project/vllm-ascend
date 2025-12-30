import torch
import vllm.distributed
import vllm.model_executor.layers.utils
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.distributed.parallel_state import get_tp_group
import vllm.distributed 

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

direct_register_custom_op(
    op_name="unquantized_gemm",
    op_func=unquantized_gemm,
    fake_impl=unquantized_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1"
)

def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.vllm.unquantized_gemm(x, weight, bias)

# def tensor_model_parallel_all_reduce_impl(input_: torch.Tensor) -> torch.Tensor:
#     """All-reduce the input tensor across model parallel group."""
#     return get_tp_group().all_reduce(input_)


# direct_register_custom_op(op_name="tensor_model_parallel_all_reduce",
#                           op_func=tensor_model_parallel_all_reduce_impl,
#                           fake_impl=lambda x: x,
#                           mutates_args=[],
#                           dispatch_key="PrivateUse1")

# vllm.distributed.tensor_model_parallel_all_reduce = tensor_model_parallel_all_reduce
vllm.model_executor.layers.utils.default_unquantized_gemm = default_unquantized_gemm
