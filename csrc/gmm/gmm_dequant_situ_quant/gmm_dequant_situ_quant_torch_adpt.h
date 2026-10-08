/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VLLM_ASCEND_GMM_DEQUANT_SITU_QUANT_TORCH_ADPT_H_
#define VLLM_ASCEND_GMM_DEQUANT_SITU_QUANT_TORCH_ADPT_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include "../../aclnn_torch_adapter/op_api_common.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"

namespace vllm_ascend {

// x_scale uses fixed-size UB chunks. Bound the AIC expert-of-block table.
constexpr int64_t GMSQ_MAX_M_BLOCKS = 256;

constexpr int32_t GMSQ_BM = 128;
constexpr int32_t GMSQ_BN = 128;
constexpr int32_t GMSQ_BK = 64;
constexpr int64_t GMSQ_MAX_EXPERTS = 128;
constexpr int64_t GMSQ_EXACT_MAX_K = (1LL << 24) / 1024;

// ACL tensor format ids (see acl_base.h / torch_npu Format enum).
constexpr int64_t ACL_FMT_ND = 2;
constexpr int64_t ACL_FMT_NCHW = 0; // ordinary contiguous base-format storage
constexpr int64_t ACL_FMT_FRACTAL_NZ = 29;

// ACLNN receives an ND descriptor of the packed carrier bytes. For native NZ
// weights, the physical layout is conveyed separately by weight_nz. Passing the
// INT32 view's original INT8 NZ descriptor would multiply the storage size by
// four and could trigger an unwanted format conversion. data_ptr() includes
// each expert slice's storage offset; no tensor storage is copied or changed.
struct GmmDequantSituQuantWeights {
    at::TensorList tensors;
};

inline aclTensorList *ConvertType(const GmmDequantSituQuantWeights &weights)
{
    static const auto createTensor = GET_OP_API_FUNC(aclCreateTensor);
    static const auto createTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (createTensor == nullptr || createTensorList == nullptr) {
        return nullptr;
    }
    std::vector<const aclTensor *> descriptors;
    descriptors.reserve(weights.tensors.size());
    for (const auto &weight : weights.tensors) {
        descriptors.push_back(createTensor(weight.sizes().data(), weight.dim(), ACL_INT32,
            weight.strides().data(), 0, ACL_FORMAT_ND, weight.sizes().data(), weight.dim(),
            weight.data_ptr()));
    }
    return createTensorList(descriptors.data(), descriptors.size());
}

std::tuple<at::Tensor, at::Tensor> gmm_dequant_situ_quant(
    const at::Tensor &x, at::TensorList weight, at::TensorList weight_scale,
    const at::Tensor &x_scale, const at::Tensor &group_list, at::TensorList weight_assist_matrix,
    double beta, std::optional<double> linear_beta, int64_t group_list_type)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == at::kChar, "x must be int8");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x_scale.scalar_type() == at::kFloat, "x_scale must be float32");
    TORCH_CHECK(x_scale.is_contiguous() && x_scale.numel() >= x.size(0),
                "x_scale must be contiguous with at least x.size(0) elements");
    TORCH_CHECK(!weight.empty(), "weight list must be non-empty");
    TORCH_CHECK(weight.size() == weight_scale.size(), "weight/weight_scale size mismatch");
    TORCH_CHECK(weight_assist_matrix.empty() || weight_assist_matrix.size() == weight.size(),
                "weight_assist_matrix size mismatch");
    TORCH_CHECK(group_list_type == 0 || group_list_type == 1, "group_list_type must be 0 or 1");
    TORCH_CHECK(group_list.dim() == 1, "group_list must be 1D");
    TORCH_CHECK(group_list.numel() >= static_cast<int64_t>(weight.size()),
                "group_list numel must be at least the number of experts");
    TORCH_CHECK(group_list.scalar_type() == at::kLong || group_list.scalar_type() == at::kInt ||
                    group_list.scalar_type() == at::kFloat,
                "group_list dtype must be int64, int32 or float32");

    TORCH_CHECK(x.device().type() == at::kPrivateUse1, "x must be on an NPU device");
    const int64_t device = x.get_device();
    TORCH_CHECK(c10_npu::current_device() == device,
                "gmm_dequant_situ_quant: current NPU device must match x.device");
    TORCH_CHECK(x_scale.device() == x.device() && group_list.device() == x.device(),
                "x_scale and group_list must be on x.device");
    for (size_t e = 0; e < weight.size(); ++e) {
        TORCH_CHECK(weight[e].device() == x.device() && weight_scale[e].device() == x.device(),
                    "weight and weight_scale must be on x.device (expert ", e, ")");
    }
    for (const auto &assist : weight_assist_matrix) {
        TORCH_CHECK(assist.device() == x.device(), "weight_assist_matrix must be on x.device");
    }

    int64_t C = x.sizes()[0];      // x rows (capacity, >= M_valid)
    int64_t K = x.sizes()[1];
    int64_t experts = static_cast<int64_t>(weight.size());
    TORCH_CHECK(experts <= GMSQ_MAX_EXPERTS, "gmm_dequant_situ_quant supports at most 128 experts");
    TORCH_CHECK(weight[0].dim() == 2, "weight[e] must be [K, N/8]");
    int64_t NP = weight[0].sizes()[1];  // packed N/8 per row
    TORCH_CHECK(NP > 0 && NP <= std::numeric_limits<int32_t>::max() / 8,
                "packed N/8 must be positive and fit the kernel int32 shape ABI");
    int64_t N = NP * 8;
    int64_t N2 = N / 2;
    TORCH_CHECK(K > 0 && K % GMSQ_BK == 0, "unsupported K: K must be a positive multiple of 64");
    TORCH_CHECK(N % GMSQ_BN == 0, "N must be a multiple of 128");
    TORCH_CHECK(N2 % GMSQ_BN == 0, "N/2 must be a multiple of 128");
    TORCH_CHECK(weight[0].scalar_type() == at::kInt, "weight must be int32 packed");
    TORCH_CHECK(weight_scale[0].scalar_type() == at::kLong, "weight_scale must be int64 carrier");
    for (int64_t e = 0; e < experts; ++e) {
        TORCH_CHECK(weight[e].dim() == 2 && weight[e].size(0) == K && weight[e].size(1) == NP &&
                        weight[e].scalar_type() == at::kInt,
                    "weight[e] must be int32 packed [K,N/8] with a common shape; expert ", e);
        TORCH_CHECK(weight_scale[e].scalar_type() == at::kLong && weight_scale[e].numel() >= N,
                    "weight_scale[e] must contain at least N int64 carriers; expert ", e);
    }

    // This is a computation limit, independent of the ND/NZ storage contract.
    // (low-aux)+16*high has integer intermediates of magnitude <=1024*K;
    // all are exactly representable in FP32 through K=16384.
    TORCH_CHECK(K <= GMSQ_EXACT_MAX_K,
                "unsupported K for exact FP32 MSD reconstruction: 1024*K must be <=2^24; got K=", K);
    const int64_t activeUpperBound = std::min(experts, C);
    const int64_t paddedBlockUpperBound = activeUpperBound + (C - activeUpperBound) / GMSQ_BM;
    TORCH_CHECK(paddedBlockUpperBound <= GMSQ_MAX_M_BLOCKS,
                "gmm_dequant_situ_quant: capacity/expert combination exceeds the ",
                GMSQ_MAX_M_BLOCKS, " padded M-block metadata limit");

    auto devOptI8 = at::TensorOptions().device(x.device()).dtype(at::kChar);
    auto devOptF32 = at::TensorOptions().device(x.device()).dtype(at::kFloat);

    // outputs: y [C, N2] int8, y_scale [C] fp32 (only first M_valid rows meaningful)
    at::Tensor y = at::empty({C, N2}, devOptI8);
    at::Tensor y_scale = at::empty({C}, devOptF32);

    // Preserve empty-rank output shapes without launching a kernel.
    if (C == 0) {
        return {y, y_scale};
    }

    // AllToAll histograms may arrive as float32. Normalize on the input device
    // and current stream on EVERY call: counts change between graph replays.
    // Contiguous int64 input is reused without a copy. Float inputs must carry
    // finite, nonnegative integer counts (or valid cumulative counts for type 0),
    // with total valid rows <= C; value validation is the caller's contract.
    // Do not read values on the CPU or cache this converted tensor.
    at::Tensor kernelGroupList = group_list;
    if (kernelGroupList.scalar_type() != at::kLong) {
        kernelGroupList = kernelGroupList.to(at::kLong);
    }
    kernelGroupList = kernelGroupList.contiguous();

    int64_t nzCount = 0;
    for (int64_t e = 0; e < experts; e++) {
        TORCH_CHECK(weight[e].data_ptr() != nullptr && weight_scale[e].data_ptr() != nullptr,
                    "weight and weight_scale must have non-null storage");
        int64_t wfmt = at_npu::native::get_npu_format(weight[e]);
        TORCH_CHECK(wfmt == ACL_FMT_ND || wfmt == ACL_FMT_NCHW || wfmt == ACL_FMT_FRACTAL_NZ,
                    "gmm_dequant_situ_quant: weight[e] must use contiguous ND (format 0/2) "
                    "or native packed INT8-NZ viewed as INT32 (format 29); weight conversion "
                    "is not performed, got format ", wfmt, " for expert ", e);
        if (wfmt != ACL_FMT_FRACTAL_NZ) {
            TORCH_CHECK(weight[e].is_contiguous(), "weight[e] must be contiguous (ND)");
        }
        if (wfmt == ACL_FMT_FRACTAL_NZ) {
            // The logical [K,N/8] INT32 carrier must retain row-major
            // strides from native INT8-NZ -> view(INT32). Expert unbind
            // slices may have nonzero storage_offset; data_ptr includes it.
            // Reject transposed/strided views without materializing them.
            TORCH_CHECK(weight[e].is_contiguous(),
                        "gmm_dequant_situ_quant: native packed INT8-NZ viewed as INT32 "
                        "must have a contiguous [K,N/8] carrier with strides [N/8,1]; "
                        "strided views are unsupported and no weight copy is performed; expert ", e);
            // dtype views share storage, so the storage descriptor retains
            // the packed INT8 dtype and NZ geometry through view(INT32).
            // The checks above require every weight to be on x's NPU;
            // torch_npu owns these storages as NPUStorageImpl. Read the
            // installed header's public descriptor directly: NPUBridge's
            // equivalent accessor is not exported by this torch_npu build.
            // This is host metadata only, with no materialization/cast.
            const auto *storage = static_cast<const torch_npu::NPUStorageImpl *>(
                weight[e].storage().unsafeGetStorageImpl());
            const auto &desc = storage->npu_desc_;
            TORCH_CHECK(desc.data_type_ == caffe2::TypeMeta::Make<int8_t>(),
                        "gmm_dequant_situ_quant: NZ weight must originate from native "
                        "packed INT8 storage before view(INT32), expert ", e);
            const auto &shape = desc.storage_sizes_;
            const size_t rank = shape.size();
            TORCH_CHECK(rank >= 4 && shape[rank - 4] == N / 64 &&
                            shape[rank - 3] == K / 16 && shape[rank - 2] == 16 &&
                            shape[rank - 1] == 32,
                        "gmm_dequant_situ_quant: native packed INT8-NZ storage must have "
                        "trailing geometry [N/64,K/16,16,32], expert ", e);
        }
        nzCount += (wfmt == ACL_FMT_FRACTAL_NZ) ? 1 : 0;
    }
    TORCH_CHECK(nzCount == 0 || nzCount == experts,
                "gmm_dequant_situ_quant: mixed ND/FRACTAL_NZ weight experts are not "
                "supported; all experts must use the same format");

    std::vector<at::Tensor> scales;
    scales.reserve(weight_scale.size());
    for (const auto &scale : weight_scale) {
        at::Tensor ndScale = at_npu::native::get_npu_format(scale) == ACL_FMT_ND
            ? scale : at_npu::native::npu_format_cast(scale, ACL_FMT_ND);
        TORCH_CHECK(ndScale.is_contiguous(), "weight_scale[e] must be contiguous (ND)");
        scales.push_back(ndScale.view({-1}));
    }
    at::Tensor xScale = x_scale.view({-1});
    GmmDequantSituQuantWeights weights{weight};
    at::TensorList scaleList(scales);
    const bool hasLinear = linear_beta.has_value();
    const double linearBeta = linear_beta.value_or(1.0);
    const bool weightNz = nzCount == experts;
    EXEC_NPU_CMD(aclnnGmmDequantSituQuant, x, weights, scaleList, xScale,
                 kernelGroupList, beta, linearBeta, hasLinear, group_list_type, weightNz, y, y_scale);
    return {y, y_scale};
}

} // namespace vllm_ascend

#endif // VLLM_ASCEND_GMM_DEQUANT_SITU_QUANT_TORCH_ADPT_H_
