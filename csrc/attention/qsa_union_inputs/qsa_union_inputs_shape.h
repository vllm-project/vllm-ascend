/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 */
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

namespace vllm_ascend {
inline int64_t checked_qsa_union_width(
    const at::Tensor& compact_indices, const at::Tensor& visible_lengths,
    const at::Tensor& group_row_map)
{
    TORCH_CHECK(compact_indices.dim() == 2 && compact_indices.size(1) > 3,
                "compact_indices must have shape [rows, groups + 3]");
    TORCH_CHECK(visible_lengths.dim() == 1 &&
                    visible_lengths.sym_size(0) == compact_indices.sym_size(0),
                "visible_lengths must match compact_indices rows");
    TORCH_CHECK(group_row_map.dim() == 2 && group_row_map.size(1) == 8,
                "group_row_map must have shape [union_groups, 8]");
    const int64_t input_groups = compact_indices.size(1) - 3;
    const int64_t topk = input_groups + (input_groups < 512 ? 1 : 0);
    TORCH_CHECK(topk > 0 && (topk & (topk - 1)) == 0,
                "QSA Union packed width must be a positive power of two");
    return 8 * topk;
}
}  // namespace vllm_ascend
