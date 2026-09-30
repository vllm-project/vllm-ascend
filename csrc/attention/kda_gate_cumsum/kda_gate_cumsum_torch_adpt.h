/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
#ifndef KDA_GATE_CUMSUM_TORCH_ADPT_H
#define KDA_GATE_CUMSUM_TORCH_ADPT_H

#include <string>

#include "../kda_torch_adpt_common.h"

namespace vllm_ascend {

at::Tensor kda_gate_cumsum(
    const at::Tensor &g,
    int64_t chunk_size,
    const c10::optional<at::Tensor> &A_log,
    const c10::optional<at::Tensor> &dt_bias,
    c10::optional<at::IntArrayRef> cu_seqlens,
    c10::optional<bool> use_gate_in_kernel,
    c10::optional<bool> safe_gate,
    c10::optional<double> lower_bound,
    c10::string_view layout)
{
    TORCH_CHECK(g.dim() == 3 || g.dim() == 4,
                "kda_gate_cumsum: g must be BSND/BNSD rank4 or TND/NTD rank3.");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64 || chunk_size == 128,
                "kda_gate_cumsum: chunk_size must be 32, 64 or 128.");
    auto gate_dtype = g.scalar_type();
    TORCH_CHECK(gate_dtype == at::kFloat || gate_dtype == at::kBFloat16 || gate_dtype == at::kHalf,
                "kda_gate_cumsum: g must be float32, bfloat16 or float16.");
    std::string layout_str(layout.data(), layout.size());
    TORCH_CHECK(layout_str == "BSND" || layout_str == "BNSD" || layout_str == "TND" || layout_str == "NTD",
                "kda_gate_cumsum: layout must be uppercase and one of BSND, BNSD, TND or NTD.");
    bool is_bnsd = layout_str == "BNSD";
    bool is_ntd = layout_str == "NTD";
    int64_t T = is_bnsd ? g.sizes()[2] : (is_ntd ? g.sizes()[1] : (g.dim() == 4 ? g.sizes()[1] : g.sizes()[0]));
    int64_t K = g.dim() == 4 ? g.sizes()[3] : g.sizes()[2];
    int64_t HV = is_bnsd ? g.sizes()[1] : (is_ntd ? g.sizes()[0] : (g.dim() == 4 ? g.sizes()[2] : g.sizes()[1]));
    TORCH_CHECK(K <= 256, "kda_gate_cumsum: K must be <= 256.");
    check_kda_cu_seqlens(cu_seqlens, T, "kda_gate_cumsum");
    TORCH_CHECK(!cu_seqlens.has_value() || g.dim() == 3 || g.sizes()[0] == 1,
                "kda_gate_cumsum: rank4 varlen input with cu_seqlens currently requires B=1.");

    bool use_gate = use_gate_in_kernel.value_or(false);
    bool safe = safe_gate.value_or(false);
    double lower = lower_bound.value_or(-5.0);
    at::Tensor A_log_tensor = A_log.value_or(at::Tensor());
    at::Tensor dt_bias_tensor = dt_bias.value_or(at::Tensor());
    if (use_gate) {
        TORCH_CHECK(A_log_tensor.defined(), "kda_gate_cumsum: A_log is required when use_gate_in_kernel=True.");
        TORCH_CHECK(A_log_tensor.scalar_type() == at::kFloat &&
                        A_log_tensor.dim() == 1 && A_log_tensor.sizes()[0] == HV,
                    "kda_gate_cumsum: A_log must be float32 with shape [HV].");
        TORCH_CHECK(safe, "kda_gate_cumsum: raw gate path currently requires safe_gate=True.");
        TORCH_CHECK(lower >= -5.0 && lower < 0.0, "kda_gate_cumsum: lower_bound must be in [-5, 0).");
    } else {
        TORCH_CHECK(!safe, "kda_gate_cumsum: safe_gate only applies when use_gate_in_kernel=True.");
    }

    at::Tensor gk = at::empty(g.sizes(), g.options().dtype(at::kFloat));
    char *layout_cstr = const_cast<char *>(layout_str.c_str());
    EXEC_NPU_CMD(
        aclnnKdaGateCumsum,
        g, A_log_tensor, dt_bias_tensor, cu_seqlens,
        chunk_size, use_gate, safe, lower, layout_cstr, gk
    );
    return gk;
}

} // namespace vllm_ascend

#endif
