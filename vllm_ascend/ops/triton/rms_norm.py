import torch
from vllm.triton_utils import tl, triton


@triton.jit
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


def _rms_block_m(total_batch: int, num_vectorcore: int) -> int:
    """Tile size used by ``triton_q_rms``.

    Flooring ``BLOCK_M`` to a power of two is math-equivalent (leftover rows
    are already masked) and cuts the JIT constexpr set from 16 values to
    ``{1, 2, 4, 8, 16}``.
    """
    row_block_size = 16
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

    BLOCK_M = _rms_block_m(total_batch, num_vectorcore)

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
