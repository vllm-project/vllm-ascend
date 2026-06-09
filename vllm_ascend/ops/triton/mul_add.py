import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def muls_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    scale,  # Scale factor.
    n_elements,  # Size of the vector.
    n_blocks,  # Total number of blocks.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
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

    # Determine the number of vector cores available
    num_cores = get_vectorcore_num()

    # Define block size
    BLOCK_SIZE = max(hidden_size // 2, 1024)

    # Calculate the number of programs to launch
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_programs = min(num_blocks, num_cores)

    # Launch the Triton kernel
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
