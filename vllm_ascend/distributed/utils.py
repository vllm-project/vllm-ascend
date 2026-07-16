import torch
import torch.distributed as dist
from vllm.distributed import get_dcp_group
from vllm.distributed.parallel_state import GroupCoordinator


def get_decode_context_model_parallel_world_size() -> int:
    """Return DCP world size (v0.21.0 helper removed on vLLM main)."""
    return get_dcp_group().world_size


def get_decode_context_model_parallel_rank() -> int:
    """Return DCP rank within group (v0.21.0 helper removed on vLLM main)."""
    return get_dcp_group().rank_in_group


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
