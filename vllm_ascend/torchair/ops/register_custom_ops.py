import torch
import torch.nn.functional as F
from vllm.distributed import (tensor_model_parallel_all_reduce,
                              tensor_model_parallel_reduce_scatter)
from vllm.forward_context import get_forward_context
from vllm.utils import direct_register_custom_op


def _maybe_pad_and_reduce_impl(x: torch.Tensor) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return tensor_model_parallel_all_reduce(x)

    sp_enabled = forward_context.sp_enabled
    if sp_enabled:
        pad_size = forward_context.pad_size
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
        return tensor_model_parallel_reduce_scatter(x, 0)
    else:
        return tensor_model_parallel_all_reduce(x)


direct_register_custom_op(op_name="maybe_pad_and_reduce",
                          op_func=_maybe_pad_and_reduce_impl,
                          fake_impl=lambda x: x,
                          mutates_args=[],
                          dispatch_key="PrivateUse1")

