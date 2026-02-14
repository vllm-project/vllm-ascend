import torch
import triton
import triton.language as tl

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def muls_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    scale,
    n_elements,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x * scale + y
        tl.store(output_ptr + offsets, output, mask=mask)


def muls_add_triton(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    assert x.shape == y.shape, "Input tensors must have the same shape."
    hidden_size = x.shape[-1]

    n_elements = x.numel()
    output = torch.empty_like(x)

    num_cores = get_vectorcore_num()

    BLOCK_SIZE = max(hidden_size // 2, 1024)

    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs = min(num_blocks, num_cores)

    muls_add_kernel[(num_programs,)](
        x,
        y,
        output,
        scale,
        n_elements,
        num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
