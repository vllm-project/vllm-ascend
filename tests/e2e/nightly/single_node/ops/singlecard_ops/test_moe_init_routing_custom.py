import itertools
import random

import numpy as np
import torch

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


def moe_init_routing_golden(x, expert_idx, scale, offset, active_num,
                            expert_capacity, expert_num, drop_pad_mode,
                            expert_tokens_num_type, expert_tokens_num_flag,
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
    actual_expert_total_num: int = np.sum((expert_idx_in >= expert_start)
                                          & (expert_idx_in < expert_end))

    expert_idx_in[(expert_idx_in
                   < expert_start)] = np.int32(np.iinfo(np.int32).max)
    sorted_expert_indices = np.argsort(expert_idx_in, axis=-1, kind="stable")
    sorted_expert_idx = expert_idx_in[sorted_expert_indices]
    if row_idx_type == 1:
        expanded_row_idx = sorted_expert_indices[:actual_expert_total_num]
    else:
        expanded_row_idx = np.ones(num_rows * k).astype(np.int32) * -1
        tmp_indices = np.arange(actual_expert_total_num)
        expanded_row_idx[
            sorted_expert_indices[:actual_expert_total_num]] = tmp_indices

    if not expert_tokens_num_flag:
        expert_tokens_count = torch.tensor([0])
    else:
        if drop_pad_mode == 0:
            if expert_tokens_num_type == 1:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate([
                    expert_tokens_count,
                    np.zeros((expert_end - expert_start) -
                             len(expert_tokens_count)).astype(np.int64)
                ])
            elif expert_tokens_num_type == 0:
                expert_tokens_count = np.bincount(
                    sorted_expert_idx[:actual_expert_total_num] - expert_start)
                expert_tokens_count = np.concatenate([
                    expert_tokens_count,
                    np.zeros((expert_end - expert_start) -
                             len(expert_tokens_count)).astype(np.int64)
                ])
                expert_tokens_count = np.cumsum(expert_tokens_count)
            elif expert_tokens_num_type == 2:
                expert_id, counts = np.unique(
                    sorted_expert_idx[:actual_expert_total_num],
                    return_counts=True)
                expert_tokens_count = np.column_stack((expert_id, counts))
                if expert_tokens_count.shape[0] < expert_num:
                    expert_tokens_count = np.concatenate(
                        (expert_tokens_count, [
                            [0, 0],
                        ]), axis=0)
        else:
            expert_tokens_count = np.bincount(
                sorted_expert_idx[:actual_expert_total_num] - expert_start)
            zeros_array = np.zeros(
                (expert_end - expert_start) - len(expert_tokens_count),
                dtype=np.int64)
            expert_tokens_count = np.concatenate(
                [expert_tokens_count, zeros_array])
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
        adapter_capacity(sorted_expert_indices, sorted_expert_idx,
                         expert_capacity)

        sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
        offset_tmp = 0
        lastExpertId = 0
        for i, val in enumerate(sorted_expert_indices):
            if val != -1:
                if lastExpertId != sorted_expert_idx[i]:
                    offset_tmp = 0
                    lastExpertId = sorted_expert_idx[i]
                sort_row_tmp[sorted_expert_idx[i] * expert_capacity +
                             offset_tmp] = sorted_expert_indices[i]
                offset_tmp = offset_tmp + 1

        expanded_row_idx = np.full(sorted_expert_indices.shape, -1)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_row_idx[val] = i

        expanded_x_mask = np.full((expert_num * expert_capacity, h),
                                  1,
                                  dtype=int)
        expanded_x = np.full((expert_num * expert_capacity, h),
                             0,
                             dtype=x.dtype)
        for i, val in enumerate(sort_row_tmp):
            if val != -1:
                expanded_x[i] = x[val // k]
                expanded_x_mask[i] = np.full((h, ), 0, dtype=int)

    if quant_mode == -1:
        expanded_x = expanded_x
        expanded_row_idx = expanded_row_idx
        if scale is not None and drop_pad_mode == 1:
            expanded_scale = np.full((expert_num * expert_capacity, ),
                                     0,
                                     dtype=scale.dtype)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_scale[i] = scale[val // k]
        if scale is None:
            expanded_scale = None

    if quant_mode == 0:
        expanded_scale = None
        expanded_x_fp16 = expanded_x.astype(np.float16)
        if scale is not None:
            scale_val = scale.astype(np.float16)
        else:
            raise ValueError("scale cannot be None when quant_mode is 0")
        if offset is not None:
            offset_val = offset.astype(np.float16)
        else:
            raise ValueError("offset cannot be None when quant_mode is 0")
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
                    x_final = x_final * scale[sorted_expert_idx[:active_num] -
                                              expert_start]

                else:
                    for i, val in enumerate(sort_row_tmp):
                        if val != -1:
                            x_final[i] = x_final[i] * scale[i //
                                                            expert_capacity]
            x_abs = np.abs(x_final)
            x_max = np.max(x_abs, axis=-1, keepdims=True)
            expanded_scale = x_max / 127
            expanded_x = x_final / expanded_scale
            expanded_x = np.round(expanded_x).astype(np.int8)
        if x.dtype == np.int8:
            expanded_scale = None
    if drop_pad_mode == 1:
        expanded_x = np.ma.array(expanded_x, mask=expanded_x_mask).filled(0)
        expanded_x = expanded_x.reshape(expert_num, expert_capacity, h)

    return expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale


def npu_pta(x, expert_idx, scale, offset, active_num, expert_capacity,
            expert_num, drop_pad_mode, expert_tokens_num_type,
            expert_tokens_num_flag, quant_mode, active_expert_range,
            row_idx_type):
    expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = torch.ops._C_ascend.npu_moe_init_routing_custom(
        x,
        expert_idx,
        scale=scale,
        offset=offset,
        active_num=active_num,
        expert_capacity=expert_capacity,
        expert_num=expert_num,
        drop_pad_mode=drop_pad_mode,
        expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=expert_tokens_num_flag,
        quant_mode=quant_mode,
        active_expert_range=active_expert_range,
        row_idx_type=row_idx_type)

    return expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale


def cmp_out_golden(x_golden, x_out, dtype):
    if dtype == 'int8':
        cmp = np.isclose(x_out.cpu().numpy()[:len(x_golden)], x_golden, atol=1)
    else:
        cmp = np.isclose(x_out.cpu().numpy()[:len(x_golden)],
                         x_golden,
                         rtol=1e-05,
                         atol=1e-05)
    return np.all(cmp)


def test_moe_npu(x, expert_idx, scale, offset, active_num, expert_capacity,
                 expert_num, drop_pad_mode, expert_tokens_num_type,
                 expert_tokens_num_flag, quant_mode, active_expert_range,
                 row_idx_type):
    x_npu = x.npu()
    expert_idx_npu = expert_idx.npu()
    scale_npu = scale.npu() if scale is not None else None
    offset_npu = offset.npu() if offset is not None else None

    x_numpy = x.numpy()
    expert_idx_numpy = expert_idx.numpy()
    scale_numpy = scale.numpy() if scale is not None else None
    offset_numpy = offset.numpy() if offset is not None else None

    expanded_x_golden, expanded_row_idx_golden, expert_token_cumsum_or_count_golden, expanded_scale_golden = moe_init_routing_golden(
        x_numpy, expert_idx_numpy, scale_numpy, offset_numpy, active_num,
        expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
        expert_tokens_num_flag, active_expert_range, quant_mode, row_idx_type)

    expanded_x, expanded_row_idx, expert_token_cumsum_or_count, expanded_scale = npu_pta(
        x_npu, expert_idx_npu, scale_npu, offset_npu, active_num,
        expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type,
        expert_tokens_num_flag, quant_mode, active_expert_range, row_idx_type)
    if quant_mode == -1:
        expanded_x_result = cmp_out_golden(expanded_x_golden, expanded_x,
                                           "float32")
    else:
        expanded_x_result = cmp_out_golden(expanded_x_golden, expanded_x,
                                           "int8")

    expanded_row_idx_result = cmp_out_golden(expanded_row_idx_golden,
                                             expanded_row_idx, "int32")

    if expert_tokens_num_flag:
        expert_tokens_result = cmp_out_golden(
            expert_token_cumsum_or_count_golden, expert_token_cumsum_or_count,
            "int64")
    else:
        expert_tokens_result = True

    if quant_mode == 1 or (quant_mode == -1 and scale is not None):
        expand_scale_result = cmp_out_golden(expanded_scale_golden.flatten(),
                                             expanded_scale, "float32")
    else:
        expand_scale_result = True

    compare_result = expanded_x_result and expanded_row_idx_result and expert_tokens_result and expand_scale_result
    # print('=======case result=======: ', compare_result)
    return compare_result


def test_moe_init_routing_custom():
    failed_test_cnt = 0
    drop_pad_mode = [0, 1]
    expert_tokens_num_type = [0, 1, 2]
    expert_tokens_num_flag = [True, False]
    quant_mode = [0, 1, -1]
    row_idx_type = [0, 1]
    scale_type = [0, 1, 2]
    product_result = itertools.product(drop_pad_mode, expert_tokens_num_type,
                                       expert_tokens_num_flag, quant_mode,
                                       row_idx_type, scale_type)

    for idx, (drop_pad_mode_, expert_tokens_num_type_, expert_tokens_num_flag_,
              quant_mode_, row_idx_type_,
              scale_type_) in enumerate(product_result, 5):
        expert_num_ = random.randint(2, 500)
        expert_start = random.randint(0, expert_num_ - 1)
        expert_end = random.randint(expert_start + 1, expert_num_)
        active_expert_range_ = [expert_start, expert_end]

        N = random.randint(1, 100)
        H = random.randint(12, 100)
        K = random.randint(1, 12)
        x_ = torch.randn(N, H, dtype=torch.float16) * 5
        expert_capacity_ = random.randint(1, N - 1) if N > 1 else 1
        expert_idx_ = torch.randint(0,
                                    expert_num_ - 1, (N, K),
                                    dtype=torch.int32)
        active_num_ = N * K

        if drop_pad_mode_ == 1:
            active_expert_range_ = [0, expert_num_]
            expert_tokens_num_type_ = 1
            row_idx_type_ = 0

        if quant_mode_ == 0:
            scale_ = torch.randn(1, dtype=torch.float)
            offset_ = torch.randn(1, dtype=torch.float)
        elif quant_mode_ == -1:
            scale_ = None
            offset_ = None
        else:
            if scale_type_ == 0:
                scale_ = None
                offset_ = None
            elif scale_type_ == 1:
                scale_ = torch.randn(1, H, dtype=torch.float)
                offset_ = None
            else:
                scale_ = torch.randn(active_expert_range_[1] -
                                     active_expert_range_[0],
                                     H,
                                     dtype=torch.float)
            offset_ = None

        result_pta = test_moe_npu(x_, expert_idx_, scale_, offset_,
                                  active_num_, expert_capacity_, expert_num_,
                                  drop_pad_mode_, expert_tokens_num_type_,
                                  expert_tokens_num_flag_, quant_mode_,
                                  active_expert_range_, row_idx_type_)
        if not result_pta:
            failed_test_cnt += 1

    assert (failed_test_cnt == 0)
