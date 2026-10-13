import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_ub_size_bytes


@triton.jit(
    do_not_specialize=[
        "total_batch",
    ]
)
def triton_rms_kernel(
    hidden_state_ptr,
    hidden_state_stride_bs,
    norm_output_ptr,
    variance_epsilon,
    total_batch,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    core_id = tl.program_id(0)
    core_num = tl.num_programs(0)
    batch_per_core = tl.cdiv(total_batch, core_num)
    start_batch = core_id * batch_per_core
    end_batch = tl.minimum(start_batch + batch_per_core, total_batch)
    offset_d = tl.arange(0, DIM)

    for row_start in tl.range(start_batch, end_batch, BLOCK_M):
        offset_row = row_start + tl.arange(0, BLOCK_M)
        mask_r = offset_row < total_batch
        mask_row = mask_r[:, None]
        offset_hidden = offset_row[:, None] * hidden_state_stride_bs + offset_d[None, :]

        x = tl.load(hidden_state_ptr + offset_hidden, mask=mask_row)

        variance = tl.sum(x * x, axis=-1) / DIM
        output = x * tl.rsqrt(variance[:, None] + variance_epsilon)

        tl.store(norm_output_ptr + offset_hidden, output, mask=mask_row)


def _rms_block_m(total_batch: int, num_vectorcore: int, dim: int, dtype) -> int:
    """Tile size used by ``triton_q_rms``.

    Derived adaptively from the runtime Unified Buffer (UB) size so small
    hidden dims get larger tiles (fewer loop iterations) while large hidden
    dims shrink the tile to avoid UB overflow. ``BLOCK_M`` is floored to a
    power of two (leftover rows are already masked), which keeps the JIT
    constexpr set small.
    """
    resv_buffer = 6144  # reserve 6 KB to prevent UB overflow
    available_ub_size = get_ub_size_bytes() - resv_buffer
    element_size = torch.empty(1, dtype=dtype).element_size()
    if element_size == 4:
        data_multiplier = 5  # input + offsets + mid_out + output + others
    elif element_size == 2:
        data_multiplier = 7  # input + offsets + mid_out + output + others
    else:
        raise NotImplementedError(f"triton_q_rms: element size ({element_size} B) not supported")
    row_block_size = int(available_ub_size / (dim * element_size * data_multiplier))
    batch_per_core = triton.cdiv(total_batch, num_vectorcore)
    raw = min(row_block_size, int(batch_per_core))
    return 1 << (max(raw, 1).bit_length() - 1)


def triton_q_rms(
    q,  # bs, 64, 512
    variance_epsilon,
):
    bs, head_num, dim = q.shape
    total_batch = bs * head_num
    q = q.view(total_batch, dim)

    if dim > 2048:
        raise NotImplementedError(f"triton_q_rms: dim > 2048 not supported, got {dim}")

    device_properties = triton.runtime.driver.active.utils.get_device_properties(q.device)
    num_vectorcore = device_properties.get("num_vectorcore", -1)

    BLOCK_M = _rms_block_m(total_batch, num_vectorcore, dim, q.dtype)

    grid = (num_vectorcore,)
    norm_output = torch.empty_like(q)

    triton_rms_kernel[grid](
        q,
        q.stride(0),
        norm_output,
        variance_epsilon,
        total_batch,
        dim,
        BLOCK_M,
    )
    return norm_output.view(bs, head_num, dim)
