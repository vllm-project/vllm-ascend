// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../attn_res_fwd/attn_res_fwd_fused_common.h"

namespace vllm_ascend {
std::tuple<at::Tensor, at::Tensor, at::Tensor> attn_res_fwd_fused(
    const at::Tensor& prefix, const c10::optional<at::Tensor>& addend,
    at::Tensor blocks, const at::Tensor& proj, const at::Tensor& norm,
    double eps, int64_t valid, const c10::optional<at::Tensor>& output_norm,
    double output_eps, int64_t write_idx, bool save_materialized, bool mix)
{
    return detail::attn_res_fwd_fused_impl<false>(
        prefix, addend, blocks, proj, norm, eps, valid, output_norm,
        output_eps, write_idx, save_materialized, mix);
}
} // namespace vllm_ascend
