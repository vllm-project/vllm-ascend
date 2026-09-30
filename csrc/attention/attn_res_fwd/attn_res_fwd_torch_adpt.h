// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attn_res_fwd_common.h"

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> attn_res_fwd(
    const at::Tensor& prefix, const c10::optional<at::Tensor>& addend,
    at::Tensor blocks, const at::Tensor& proj, const at::Tensor& norm,
    double eps, int64_t valid, const c10::optional<at::Tensor>& output_norm,
    double output_eps, int64_t write_idx, bool save_materialized, bool mix,
    bool optimize_prefill)
{
    return detail::attn_res_fwd_impl(
        prefix, addend, blocks, proj, norm, eps, valid, output_norm,
        output_eps, write_idx, save_materialized, mix, optimize_prefill);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> attn_res_fwd_meta(
    const at::Tensor& prefix, const c10::optional<at::Tensor>& addend,
    at::Tensor blocks, const at::Tensor& proj, const at::Tensor& norm,
    double eps, int64_t valid, const c10::optional<at::Tensor>& output_norm,
    double output_eps, int64_t write_idx, bool save_materialized, bool mix,
    bool optimize_prefill)
{
    auto output = at::empty_symint(prefix.sym_sizes(), prefix.options());
    auto raw_prefix = addend.has_value() ? at::empty_symint(prefix.sym_sizes(), prefix.options()) : prefix;
    auto materialized = save_materialized ? at::empty_symint(prefix.sym_sizes(), prefix.options()) : output;
    return {output, raw_prefix, materialized};
}
} // namespace vllm_ascend
