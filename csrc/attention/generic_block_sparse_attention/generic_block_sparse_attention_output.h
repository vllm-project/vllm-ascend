/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_OUTPUT_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_OUTPUT_H
#include <torch/extension.h>

namespace vllm_ascend::gbsa_detail {
constexpr int64_t METADATA_SIZE = 1024;

inline std::tuple<at::Tensor, at::Tensor> MakeOutputs(
    const at::Tensor& q, c10::string_view layout_q, int64_t quant_mode,
    bool return_softmax_lse, c10::optional<at::ScalarType> attention_out_dtype)
{
    TORCH_CHECK(layout_q == "TND" || layout_q == "BNSD" || layout_q == "BSND",
                "layout_q must be TND, BNSD, or BSND");
    TORCH_CHECK(q.dim() == (layout_q == "TND" ? 3 : 4), "query rank does not match layout_q");
    TORCH_CHECK(attention_out_dtype.has_value() || quant_mode == 0,
                "attention_out_dtype must be specified when quant_mode != NO_QUANT");
    auto out = at::empty_symint(q.sym_sizes(), q.options().dtype(attention_out_dtype.value_or(q.scalar_type())));
    auto lse_options = q.options().dtype(at::kFloat);
    if (!return_softmax_lse) {
        return {out, at::empty({0}, lse_options)};
    }
    if (layout_q == "TND") {
        return {out, at::empty_symint({q.sym_size(0), q.sym_size(1), 1}, lse_options)};
    }
    if (layout_q == "BNSD") {
        return {out, at::empty_symint({q.sym_size(0), q.sym_size(1), q.sym_size(2), 1}, lse_options)};
    }
    return {out, at::empty_symint({q.sym_size(0), q.sym_size(2), q.sym_size(1), 1}, lse_options)};
}
} // namespace vllm_ascend::gbsa_detail
#endif
