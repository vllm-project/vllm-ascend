import torch
import torch.distributed as dist
from vllm.distributed import get_dcp_group
from vllm.distributed.parallel_state import GroupCoordinator, get_dp_group
from vllm.forward_context import get_forward_context

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.distributed.parallel_state import get_fc3_quant_x_group


def get_decode_context_model_parallel_world_size() -> int:
    """Return DCP world size (v0.21.0 helper removed on vLLM main)."""
    return get_dcp_group().world_size


def get_decode_context_model_parallel_rank() -> int:
    """Return DCP rank within group (v0.21.0 helper removed on vLLM main)."""
    return get_dcp_group().rank_in_group


def fc3_all_gather_and_maybe_unpad_impl(
    x: torch.Tensor,
) -> torch.Tensor:
    try:
        forward_context = get_forward_context()
    except AssertionError:
        return x
    x = get_fc3_quant_x_group().all_gather(x, 0)
    dp_metadata = forward_context.dp_metadata
    if dp_metadata is None:
        pad_size = _EXTRA_CTX.pad_size
        if pad_size > 0:
            x = x[:-pad_size]
    else:
        # unpad
        num_tokens_across_dp_cpu = dp_metadata.num_tokens_across_dp_cpu
        result = torch.empty((num_tokens_across_dp_cpu.sum(), *x.shape[1:]), device=x.device, dtype=x.dtype)
        dp_size = get_dp_group().world_size
        x = x.view(dp_size, _EXTRA_CTX.padded_length, *x.shape[1:])
        offset = 0
        for idx in range(dp_size):
            num_tokens_dp = num_tokens_across_dp_cpu[idx]
            result[offset : offset + num_tokens_dp] = x[idx, :num_tokens_dp]
            offset += num_tokens_dp
        x = result

    return x


def all_gather_async(
    input: torch.Tensor, group: GroupCoordinator, output: torch.Tensor | None = None, async_op: bool = True
):
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)


def split_tensor_along_first_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
):
    """Split a tensor along its first dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                in memory.

    Returns:
        A list of Tensors
    """
    from vllm.distributed.utils import divide

    # Get the size and dimension.
    first_dim_size = divide(tensor.size()[0], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, first_dim_size, dim=0)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list
