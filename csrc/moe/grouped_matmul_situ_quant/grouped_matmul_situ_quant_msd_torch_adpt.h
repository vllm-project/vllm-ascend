// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cmath>

namespace ascend_kernel {
namespace {

// The public Torch schemas are shared with MX FP8/FP4. Only INT8 inputs use
// this versioned ACLNN entry: the original A5 binary ABI remains unchanged.
std::tuple<at::Tensor, at::Tensor> RunSituMsd(
    const at::Tensor &x, const std::vector<at::Tensor> &weights,
    const std::vector<at::Tensor> &weightScales,
    const std::optional<std::vector<at::Tensor>> &weightAssistMatrix,
    const std::optional<at::Tensor> &bias, const at::Tensor &xScale,
    const std::optional<at::Tensor> &smoothScale, const at::Tensor &groupList,
    int64_t dequantMode, int64_t dequantDtype, int64_t quantMode, int64_t groupListType,
    const std::optional<std::vector<int64_t>> &tuningConfig, double beta, double linearBeta,
    bool nz, bool stacked)
{
    TORCH_CHECK(x.is_privateuseone() && x.dim() == 2 && x.is_contiguous(),
                "SiTU MSD: x must be a contiguous NPU matrix");
    TORCH_CHECK(dequantMode == 0 && dequantDtype == 0 && quantMode == 0,
                "SiTU MSD requires dequant_mode=0, dequant_dtype=0 and quant_mode=0");
    TORCH_CHECK(!bias.has_value() || !bias->defined() || bias->numel() == 0,
                "SiTU MSD: bias is unsupported; use weight_assist_matrix for MSD compensation");
    TORCH_CHECK(!smoothScale.has_value() || !smoothScale->defined() || smoothScale->numel() == 0,
                "SiTU MSD: smooth_scale is unsupported");
    TORCH_CHECK(std::isfinite(beta) && beta > 0 && std::isfinite(linearBeta),
                "SiTU MSD: beta must be positive and both activation parameters must be finite");
    TORCH_CHECK(groupListType == 0 || groupListType == 1, "SiTU MSD: group_list_type must be 0 or 1");
    auto checkTensor = [&](const at::Tensor &t, at::ScalarType dtype, const char *name) {
        TORCH_CHECK(t.defined() && t.device() == x.device() && t.scalar_type() == dtype && t.is_contiguous(),
                    "SiTU MSD: ", name, " must be contiguous, have the required dtype and share x's device");
    };
    checkTensor(groupList, at::kLong, "group_list");
    TORCH_CHECK(groupList.dim() == 1 && groupList.numel() > 0, "SiTU MSD: group_list must have shape [E], E > 0");
    const int64_t e = groupList.numel();
    const int64_t m = x.size(0);
    const int64_t k = x.size(1);
    TORCH_CHECK(k > 0 && k % 64 == 0, "SiTU MSD: K must be a positive multiple of 64");
    const size_t count = stacked ? 1 : static_cast<size_t>(e);
    TORCH_CHECK(weights.size() == count && weightScales.size() == count,
                "SiTU MSD: weight and weight_scale must have matching stacked/list forms and expert counts");
    checkTensor(xScale, at::kFloat, "x_scale");
    TORCH_CHECK(xScale.numel() == m, "SiTU MSD: x_scale must contain one FP32 scale per row");
    TORCH_CHECK(weightScales[0].dim() > 0, "SiTU MSD: weight_scale must have at least one dimension");
    const int64_t n = weightScales[0].size(-1);
    TORCH_CHECK(n > 0 && n % 128 == 0, "SiTU MSD: N must be a positive multiple of 128");
    const int64_t expertsPerTensor = stacked ? e : 1;
    for (size_t i = 0; i < count; ++i) {
        const auto &w = weights[i];
        TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kInt,
                    "SiTU MSD: weight must be an NPU INT32 carrier of packed INT4 values");
        TORCH_CHECK(w.is_contiguous() && w.storage_offset() == 0,
                    "SiTU MSD: packed weight must own a contiguous view with zero storage offset");
        TORCH_CHECK(w.dim() == (stacked ? 3 : 2) && w.size(-2) == k && w.size(-1) == n / 8 &&
                        w.numel() == expertsPerTensor * k * (n / 8),
                    "SiTU MSD: weight must have logical shape [E,K,N/8] or per-expert [K,N/8]");
        const int64_t format = at_npu::native::custom_ops::get_npu_format(w);
        TORCH_CHECK(nz ? (format == 29 || format == 50) : (format == 2 && w.is_contiguous()),
                    "SiTU MSD: weight format must match the ND/NZ entry");
        checkTensor(weightScales[i], at::kLong, "weight_scale");
        TORCH_CHECK(weightScales[i].size(-1) == n && weightScales[i].numel() == expertsPerTensor * n,
                    "SiTU MSD: weight_scale must contain packed UINT64 bits for each expert/output channel");
    }
    std::vector<at::Tensor> assist;
    if (weightAssistMatrix.has_value() && !weightAssistMatrix->empty()) {
        assist = *weightAssistMatrix;
        TORCH_CHECK(assist.size() == count, "SiTU MSD: weight_assist_matrix must match the weight list");
        for (const auto &t : assist) {
            checkTensor(t, at::kFloat, "weight_assist_matrix");
            TORCH_CHECK(t.numel() == expertsPerTensor * n,
                        "SiTU MSD: weight_assist_matrix must contain E*N compensation values");
        }
    } else {
        // A one-element placeholder selects in-kernel MSD compensation.
        assist.push_back(at::empty({1}, x.options().dtype(at::kFloat)));
    }
    at::Tensor y = at::empty({m, n / 2}, x.options());
    at::Tensor yScale = at::empty({m}, x.options().dtype(at::kFloat));
    if (m == 0) {
        return {y, yScale};
    }
    const std::vector<int64_t> tuning = tuningConfig.has_value() && !tuningConfig->empty()
                                           ? *tuningConfig : std::vector<int64_t>{0};
    at::TensorList weightList(weights), scaleList(weightScales), assistList(assist);
    at::IntArrayRef tuningRef(tuning);
    const std::optional<at::Tensor> none = std::nullopt;
    const int64_t quantDtype = 0;
    const bool transposeWeight = false;
    const bool activateLeft = true;
    EXEC_NPU_CMD(aclnnGroupedMatmulSituQuantWeightNzV2, x, xScale, weightList, scaleList, groupList,
                 assistList, none, none, groupListType, beta, linearBeta, dequantMode, dequantDtype,
                 quantMode, quantDtype, transposeWeight, tuningRef, activateLeft, y, yScale);
    return {y, yScale};
}

} // namespace
} // namespace ascend_kernel
