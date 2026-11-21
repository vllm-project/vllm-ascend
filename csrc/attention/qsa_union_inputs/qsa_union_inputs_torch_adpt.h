/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef VLLM_ASCEND_QSA_UNION_INPUTS_TORCH_ADPT_H
#define VLLM_ASCEND_QSA_UNION_INPUTS_TORCH_ADPT_H

#include "qsa_union_inputs_shape.h"

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_qsa_union_inputs(
    const at::Tensor& compact_indices, const at::Tensor& visible_lengths,
    const at::Tensor& group_row_map)
{
    TORCH_CHECK(compact_indices.scalar_type() == at::kInt,
                "compact_indices must be int32");
    TORCH_CHECK(visible_lengths.scalar_type() == at::kInt,
                "visible_lengths must be int32");
    TORCH_CHECK(group_row_map.scalar_type() == at::kInt,
                "group_row_map must be int32");
    const int64_t width = checked_qsa_union_width(compact_indices, visible_lengths, group_row_map);
    TORCH_CHECK(compact_indices.is_contiguous() &&
                    visible_lengths.is_contiguous() &&
                    group_row_map.is_contiguous(),
                "qsa_union_inputs requires contiguous inputs");

    const int64_t groups = group_row_map.size(0);
    at::Tensor union_idx = at::empty(
        {1, groups, width}, compact_indices.options());
    at::Tensor union_count = at::empty(
        {1, groups}, compact_indices.options());
    at::Tensor exact_mask = at::empty(
        {1, groups, width * 4}, compact_indices.options());
    if (groups != 0) {
        EXEC_NPU_CMD(aclnnQsaUnionInputs, compact_indices, visible_lengths,
                     group_row_map, union_idx, union_count, exact_mask);
    }
    return {union_idx, union_count, exact_mask};
}
}  // namespace vllm_ascend
#endif
