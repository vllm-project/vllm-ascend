import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _swiglu_quant_kernel(
    x_ptr,
    group_list_ptr,
    out_ptr,
    scale_ptr,
    TOTAL_COLS: tl.constexpr,
    HALF_COLS: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    NUM_EXPERTS_ALGIN: tl.constexpr,
    GROUP_LIST_TYPE: tl.constexpr,
    NUM_CORES: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
    SCALE: tl.constexpr,
):
    # calc real total_rows
    if GROUP_LIST_TYPE == 0:  # cusum
        total_rows = tl.load(group_list_ptr + NUM_EXPERTS).to(tl.int32)
    else:
        gl_offsets = tl.arange(0, NUM_EXPERTS_ALGIN)
        gl_mask = gl_offsets < NUM_EXPERTS
        group_list = tl.load(group_list_ptr + gl_offsets, gl_mask, other=0).to(tl.int32)
        total_rows = tl.sum(group_list)

    block_size = (total_rows - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= total_rows:
        return
    row_end = tl.minimum((pid + 1) * block_size, total_rows)

    for row_idx in range(row_begin, row_end):
        # swiglu
        x_offsets = row_idx * TOTAL_COLS + tl.arange(0, TOTAL_COLS)
        cur_x = tl.load(x_ptr + x_offsets)
        x1 = tl.extract_slice(cur_x, offsets=(0,), sizes=(HALF_COLS,), strides=(1,))
        x2 = tl.extract_slice(cur_x, offsets=(HALF_COLS,), sizes=(HALF_COLS,), strides=(1,))
        out = x1 * tl.sigmoid(x1) * x2

        # quant
        if SCALE:
            scale = tl.max(tl.abs(out)).to(tl.float32) / DTYPE_MAX
            # store scale
            tl.store(scale_ptr + row_idx, scale.to(scale_ptr.dtype.element_ty))
            for col_blk_idx in range(0, HALF_COLS, COL_BLOCK_SIZE):
                tmp_out = tl.extract_slice(out, offsets=(col_blk_idx,), sizes=(COL_BLOCK_SIZE,), strides=(1,))
                tmp_out = (tmp_out.to(tl.float32) / scale).to(x_ptr.dtype.element_ty)
                tmp_out = tmp_out.cast(tl.int8, overflow_mode="saturate")

                o_offsets = row_idx * HALF_COLS + col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)
                mask = (col_blk_idx + tl.arange(0, COL_BLOCK_SIZE)) < HALF_COLS
                tl.store(out_ptr + o_offsets, tmp_out.to(out_ptr.dtype.element_ty), mask=mask)
        else:
            # store out
            o_offsets = row_idx * HALF_COLS + tl.arange(0, HALF_COLS)
            tl.store(out_ptr + o_offsets, out.to(out_ptr.dtype.element_ty))


def swiglu_quant(x, group_list, group_list_type, need_quant=True):
    # group_list_type must be 0 cusum or 1 count
    if group_list_type not in [0, 1]:
        raise ValueError(f"group_list_type must be 0 or 1, but got {group_list_type}")
    s, h = x.shape
    out_dtype = torch.int8 if need_quant else x.dtype
    out = torch.empty((s, h // 2), dtype=out_dtype, device=x.device)
    scale = torch.empty((s,), dtype=torch.float32, device=x.device)
    num_experts = group_list.shape[0]
    # ub must be 32-byte aligned on npu
    if group_list.dtype == torch.int64:
        num_experts_algin = (num_experts + 7) // 8 * 8
    elif group_list.dtype == torch.int32:
        num_experts_algin = (num_experts + 15) // 16 * 16
    else:
        raise ValueError(f"group_list dtype must be torch.int32 or torch.int64, but got {group_list.dtype}")

    num_vectorcore = get_vectorcore_num()
    _swiglu_quant_kernel[(num_vectorcore,)](
        x,
        group_list,
        out,
        scale,
        TOTAL_COLS=h,
        HALF_COLS=h // 2,
        COL_BLOCK_SIZE=1536,
        NUM_EXPERTS=num_experts,
        NUM_EXPERTS_ALGIN=num_experts_algin,
        GROUP_LIST_TYPE=group_list_type,
        NUM_CORES=num_vectorcore,
        DTYPE_MAX=127,
        SCALE=need_quant,
        multibuffer=True,
    )
    return out, scale
