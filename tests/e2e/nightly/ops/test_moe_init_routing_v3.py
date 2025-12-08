import gc

import numpy as np
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

def moe_init_routing_golden(x, expert_idx, scale, offset, active_num, expert_capacity,
                expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, 
                active_expert_range, quant_mode, row_idx_type):
    if drop_pad_mode == 1:
        if expert_num <= 0:
            print("expert num can not be 0")
            return
    expert_start = active_expert_range[0] if drop_pad_mode == 0 else 0
    expert_end = active_expert_range[1] if drop_pad_mode == 0 else expert_num
    num_rows = x.shape[0]
    h = x.shape[1]
    k = expert_idx.shape[-1]
    expert_idx_in = expert_idx.copy().reshape(-1)
    actual_expert_total_num = np.sum((expert_idx_in >= expert_start) & (expert_idx_in < expert_end))

    expert_idx_in[(expert_idx_in < expert_start)] = np.int32(np.iinfo(np.int32).max)
    sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices[:actual_expert_total_num]
    else:
        expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
        tmp_indices = np.arange(actual_expert_total_num)
        expanded_row_idx[sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    if not expert_tokens_num_flag:
        expert_tokens_count = None
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 1:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate(
                    [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
            elif expert_tokens_num_type == 0:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate(
                    [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
                expert_tokens_count = np.cumsum(expert_tokens_count)
            elif expert_tokens_num_type == 2:
                expert_id, counts = np.unique(sorted_expert_idx[:actual_expert_total_num], return_counts=True)
                expert_tokens_count = np.column_stack((expert_id, counts))
                if expert_tokens_count.shape[0] < expert_num:
                    expert_tokens_count = np.concatenate((expert_tokens_count, [[0,0],]), axis=0)
        else:
            expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
            expert_tokens_count = np.concatenate(
                [expert_tokens_count, np.zeros((expert_end - expert_start) - len(expert_tokens_count)).astype(np.int64)])
        expert_tokens_count = expert_tokens_count.astype(np.int64)
    
    if drop_pad_mode == 0:
        if active_num == 0:
            active_num = actual_expert_total_num
        else:
            active_num = min(active_num, actual_expert_total_num)
        expanded_scale = None
        expanded_x = x[sorted_expert_indices[:active_num] // k, :]
        if scale is not None and quant_mode == -1:
            expanded_scale = scale[sorted_expert_indices[:active_num] // k]
    else:
        adapter_capacity(sorted_expert_indices, sorted_expert_idx, expert_capacity)

        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
        offset_tmp = 0
        lastExpertId = 0
        for i, val in enumerate(sorted_expert_indices):
            if val != -1:
                if lastExpertId != sorted_expert_idx[i]:
                    offset_tmp = 0
                    lastExpertId = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset_tmp] = sorted_expert_indices[i]
                offset_tmp = offset_tmp + 1
        
        expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        expanded_x_mask = np.full((expert_num * expert_capacity, h), 1, dtype=int)
        expanded_x = np.full((expert_num * expert_capacity, h), 0, dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = np.full((h,), 0, dtype=int)
    
    if quant_mode == -1:
        expanded_x = expanded_x
        expanded_row_idx = expanded_row_idx
        if scale is not None and drop_pad_mode == 1:
            expanded_scale = np.full((expert_num * expert_capacity,), 0, dtype=scale.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_scale[i] = scale[val // k]
        if scale is None:
            expanded_scale = None

    if quant_mode == 0:
        expanded_scale = None
        expanded_x_fp16 = expanded_x.astype(np.float16)
        scale_val = scale.astype(np.float16)
        offset_val = offset.astype(np.float16)
        scale_rst = expanded_x_fp16 * scale_val[0]
        add_offset = scale_rst + offset_val[0]
        round_data = np.rint(add_offset)
        round_data = np.clip(round_data, -128, 127)
        expanded_x = round_data.astype(np.int8)

    if quant_mode == 1:
        x_final = expanded_x.astype(np.float32)
        if scale is None:
            x_abs = np.abs(x_final)
            x_max = np.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = np.round(expanded_x).astype(np.int8)
        else:
            if scale.shape[0] == 1:
                x_final = x_final * scale
            else:
                if drop_pad_mode == 0:
                    x_final = x_final * scale[sorted_expert_idx[:active_num] - expert_start]

                else:
                    for i, val in enumerate(sort_row_tmp):
                        if val != -1:
                            x_final[i] = x_final[i] * scale[i // expert_capacity]
            x_abs = np.abs(x_final)
            x_max = np.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = np.round(expanded_x).astype(np.int8)
        if x.dtype == np.int8:
            expanded_scale == None
    if drop_pad_mode == 1:
        expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)
    
    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale


@torch.inference_mode()
def test_moe_init_routing_v3_kernel():
    N = 1024
    H = 4096
    K = 8
    torch.npu.config.allow_internal_format = True
    active_expert_range = [0, 256]
    expert_num = 256
    drop_pad_mode = 0
    expert_capacity = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    row_idx_type = 0
    active_num = 0
    scale = None
    offset = None
    x = torch.randint(-5, 5, (N, H), dtype=torch.int8).npu()
    expert_idx = torch.randint(0, expert_num, (N, K), dtype=torch.int32).npu()

    expanded_x_cpu, expanded_row_idx_cpu, expert_tokens_count_cpu, expanded_scale_cpu = moe_init_routing_golden(
        x.cpu(),
        expert_idx.cpu(),
        scale,
        offset,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        active_expert_range,
        quant_mode,
        row_idx_type)

    expanded_x_npu, expanded_row_idx_npu, expert_tokens_count_npu, expanded_scale_npu = torch.ops._C_ascend.npu_moe_init_routing_v2(
        x,
        expert_idx,
        scale,
        offset,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        active_expert_range,
        quant_mode,
        row_idx_type)

    torch.testing.assert_close(expanded_x_cpu, expanded_x_npu.cpu(), atol=1, rtol=0.0001)
    torch.testing.assert_close(expanded_row_idx_cpu, expanded_row_idx_npu.cpu(), atol=1, rtol=0.0001)
    torch.testing.assert_close(expert_tokens_count_cpu, expert_tokens_count_npu.cpu(), atol=1, rtol=0.0001)
    torch.testing.assert_close(expanded_scale_cpu, expanded_scale_npu.cpu(), atol=1, rtol=0.0001 )

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
