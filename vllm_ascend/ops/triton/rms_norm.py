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

    # Adaptive calculate the block size
    resv_buffer = 6144 # reserve buffer 6KB to prevent UB overflow
    available_ub_size = get_ub_size_bytes() - resv_buffer
    element_size = torch.empty(1, dtype=q.dtype).element_size() * 8
    if element_size == 4:
        data_multiplier = 5  # input(4bytes) + offset(4bytes) + mid_out(4bytes) + output(4bytes) + others(4bytes) = 5x
    elif element_size == 2:
        data_multiplier = 7  # input(2bytes) + offset(4bytes) + mid_out(4bytes) + output(2bytes) + others(2bytes) = 7x
    else:
        raise NotImplementedError(f"triton_q_rms: Element_Size ({element_size}bytes) not supported")
    ROW_BLOCK_SIZE = (int)(available_ub_size / (dim * element_size * data_multiplier))

    batch_per_core = triton.cdiv(total_batch, num_vectorcore)
    raw = min(ROW_BLOCK_SIZE, batch_per_core)
    BLOCK_M = 1 << (raw.bit_length() - 1)

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
