/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_mhc_post.cpp
 * \brief MhcPost ACLNN API implementation
 */

#include "aclnn_mhc_post.h"
#include "mhc_post.h"

#include "aclnn_kernels/contiguous.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DIM_NUM_2 = 2;
static const int64_t DIM_NUM_3 = 3;
static const int64_t DIM_NUM_4 = 4;
static const int64_t DIM_IDX_0 = 0;
static const int64_t DIM_IDX_1 = 1;
static const int64_t DIM_IDX_2 = 2;
static const int64_t DIM_IDX_3 = 3;

struct MhcPostParams {
    const aclTensor *x = nullptr;
    const aclTensor *hRes = nullptr;
    const aclTensor *hOut = nullptr;
    const aclTensor *hPost = nullptr;
    aclTensor *out = nullptr;
};

static aclnnStatus CheckNotNull(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                                const aclTensor *hPost, aclTensor *out)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(hRes != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hRes must not be nullptr.");
    CHECK_COND(hOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hOut must not be nullptr.");
    CHECK_COND(hPost != nullptr, ACLNN_ERR_PARAM_NULLPTR, "hPost must not be nullptr.");
    CHECK_COND(out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const MhcPostParams &params)
{
    // x: FP16 or BF16
    const std::initializer_list<op::DataType> xSupportList = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(params.x, xSupportList, return ACLNN_ERR_PARAM_INVALID);

    // hRes: FP32 only
    OP_CHECK_DTYPE_NOT_MATCH(params.hRes, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    // hOut: must be same as x
    OP_CHECK_DTYPE_NOT_SAME(params.x, params.hOut, return ACLNN_ERR_PARAM_INVALID);

    // hPost: FP32 only
    OP_CHECK_DTYPE_NOT_MATCH(params.hPost, op::DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);

    // out: must be same as x
    OP_CHECK_DTYPE_NOT_SAME(params.x, params.out, return ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool IsTNDFormat(size_t dimNum)
{
    return dimNum == DIM_NUM_3;
}

static bool IsBSNDFormat(size_t dimNum)
{
    return dimNum == DIM_NUM_4;
}

static aclnnStatus CheckShape3D(const MhcPostParams &params)
{
    // x shape: [T, n, D]
    auto xDim0 = params.x->GetViewShape().GetDim(DIM_IDX_0);
    auto xDim1 = params.x->GetViewShape().GetDim(DIM_IDX_1);
    auto xDim2 = params.x->GetViewShape().GetDim(DIM_IDX_2);

    // hRes shape: [T, n, n]
    auto hResDim0 = params.hRes->GetViewShape().GetDim(DIM_IDX_0);
    auto hResDim1 = params.hRes->GetViewShape().GetDim(DIM_IDX_1);
    auto hResDim2 = params.hRes->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(hResDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[0] %ld is not equal to x dim[0] %ld", hResDim0, xDim0);
    CHECK_COND(hResDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[1] %ld is not equal to x dim[1] %ld", hResDim1, xDim1);
    CHECK_COND(hResDim2 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[2] %ld must equal hRes dim[1] %ld (n x n matrix)", hResDim2, hResDim1);

    // hOut shape: [T, D]
    auto hOutDim0 = params.hOut->GetViewShape().GetDim(DIM_IDX_0);
    auto hOutDim1 = params.hOut->GetViewShape().GetDim(DIM_IDX_1);
    CHECK_COND(hOutDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "hOut dim[0] %ld is not equal to x dim[0] %ld", hOutDim0, xDim0);
    CHECK_COND(hOutDim1 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "hOut dim[1] %ld is not equal to x dim[2] %ld", hOutDim1, xDim2);

    // hPost shape: [T, n]
    auto hPostDim0 = params.hPost->GetViewShape().GetDim(DIM_IDX_0);
    auto hPostDim1 = params.hPost->GetViewShape().GetDim(DIM_IDX_1);
    CHECK_COND(hPostDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "hPost dim[0] %ld is not equal to x dim[0] %ld", hPostDim0, xDim0);
    CHECK_COND(hPostDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "hPost dim[1] %ld is not equal to x dim[1] %ld", hPostDim1, xDim1);

    // out shape: [T, n, D]
    auto outDim0 = params.out->GetViewShape().GetDim(DIM_IDX_0);
    auto outDim1 = params.out->GetViewShape().GetDim(DIM_IDX_1);
    auto outDim2 = params.out->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(outDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "out dim[0] %ld is not equal to x dim[0] %ld", outDim0, xDim0);
    CHECK_COND(outDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "out dim[1] %ld is not equal to x dim[1] %ld", outDim1, xDim1);
    CHECK_COND(outDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "out dim[2] %ld is not equal to x dim[2] %ld", outDim2, xDim2);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape4D(const MhcPostParams &params)
{
    // x shape: [B, S, n, D]
    auto xDim0 = params.x->GetViewShape().GetDim(DIM_IDX_0);
    auto xDim1 = params.x->GetViewShape().GetDim(DIM_IDX_1);
    auto xDim2 = params.x->GetViewShape().GetDim(DIM_IDX_2);
    auto xDim3 = params.x->GetViewShape().GetDim(DIM_IDX_3);

    // hRes shape: [B, S, n, n]
    auto hResDim0 = params.hRes->GetViewShape().GetDim(DIM_IDX_0);
    auto hResDim1 = params.hRes->GetViewShape().GetDim(DIM_IDX_1);
    auto hResDim2 = params.hRes->GetViewShape().GetDim(DIM_IDX_2);
    auto hResDim3 = params.hRes->GetViewShape().GetDim(DIM_IDX_3);
    CHECK_COND(hResDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[0] %ld is not equal to x dim[0] %ld", hResDim0, xDim0);
    CHECK_COND(hResDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[1] %ld is not equal to x dim[1] %ld", hResDim1, xDim1);
    CHECK_COND(hResDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[2] %ld is not equal to x dim[2] %ld", hResDim2, xDim2);
    CHECK_COND(hResDim3 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "hRes dim[3] %ld must equal hRes dim[2] %ld (n x n matrix)", hResDim3, hResDim2);

    // hOut shape: [B, S, D]
    auto hOutDim0 = params.hOut->GetViewShape().GetDim(DIM_IDX_0);
    auto hOutDim1 = params.hOut->GetViewShape().GetDim(DIM_IDX_1);
    auto hOutDim2 = params.hOut->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(hOutDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "hOut dim[0] %ld is not equal to x dim[0] %ld", hOutDim0, xDim0);
    CHECK_COND(hOutDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "hOut dim[1] %ld is not equal to x dim[1] %ld", hOutDim1, xDim1);
    CHECK_COND(hOutDim2 == xDim3, ACLNN_ERR_PARAM_INVALID,
               "hOut dim[2] %ld is not equal to x dim[3] %ld", hOutDim2, xDim3);

    // hPost shape: [B, S, n]
    auto hPostDim0 = params.hPost->GetViewShape().GetDim(DIM_IDX_0);
    auto hPostDim1 = params.hPost->GetViewShape().GetDim(DIM_IDX_1);
    auto hPostDim2 = params.hPost->GetViewShape().GetDim(DIM_IDX_2);
    CHECK_COND(hPostDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "hPost dim[0] %ld is not equal to x dim[0] %ld", hPostDim0, xDim0);
    CHECK_COND(hPostDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "hPost dim[1] %ld is not equal to x dim[1] %ld", hPostDim1, xDim1);
    CHECK_COND(hPostDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "hPost dim[2] %ld is not equal to x dim[2] %ld", hPostDim2, xDim2);

    // out shape: [B, S, n, D]
    auto outDim0 = params.out->GetViewShape().GetDim(DIM_IDX_0);
    auto outDim1 = params.out->GetViewShape().GetDim(DIM_IDX_1);
    auto outDim2 = params.out->GetViewShape().GetDim(DIM_IDX_2);
    auto outDim3 = params.out->GetViewShape().GetDim(DIM_IDX_3);
    CHECK_COND(outDim0 == xDim0, ACLNN_ERR_PARAM_INVALID,
               "out dim[0] %ld is not equal to x dim[0] %ld", outDim0, xDim0);
    CHECK_COND(outDim1 == xDim1, ACLNN_ERR_PARAM_INVALID,
               "out dim[1] %ld is not equal to x dim[1] %ld", outDim1, xDim1);
    CHECK_COND(outDim2 == xDim2, ACLNN_ERR_PARAM_INVALID,
               "out dim[2] %ld is not equal to x dim[2] %ld", outDim2, xDim2);
    CHECK_COND(outDim3 == xDim3, ACLNN_ERR_PARAM_INVALID,
               "out dim[3] %ld is not equal to x dim[3] %ld", outDim3, xDim3);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const MhcPostParams &params)
{
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    auto hResDimNum = params.hRes->GetViewShape().GetDimNum();
    auto hOutDimNum = params.hOut->GetViewShape().GetDimNum();
    auto hPostDimNum = params.hPost->GetViewShape().GetDimNum();
    auto outDimNum = params.out->GetViewShape().GetDimNum();
    CHECK_COND(IsTNDFormat(xDimNum) || IsBSNDFormat(xDimNum), ACLNN_ERR_PARAM_INVALID,
               "x dim should be 3 (TND format) or 4 (BSND format), but got %zu", xDimNum);

    if (IsTNDFormat(xDimNum)) {
        CHECK_COND(hResDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "hRes dim num should be 3 for 3D format, but got %zu", hResDimNum);
        CHECK_COND(hOutDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "hOut dim num should be 2 for 3D format, but got %zu", hOutDimNum);
        CHECK_COND(hPostDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "hPost dim num should be 2 for 3D format, but got %zu", hPostDimNum);
        CHECK_COND(outDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "out dim num should be 3 for 3D format, but got %zu", outDimNum);
        CHECK_COND(CheckShape3D(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape for 3D format");
    } else {
        CHECK_COND(hResDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
                   "hRes dim num should be 4 for 4D format, but got %zu", hResDimNum);
        CHECK_COND(hOutDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "hOut dim num should be 3 for 4D format, but got %zu", hOutDimNum);
        CHECK_COND(hPostDimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID,
                   "hPost dim num should be 3 for 4D format, but got %zu", hPostDimNum);
        CHECK_COND(outDimNum == DIM_NUM_4, ACLNN_ERR_PARAM_INVALID,
                   "out dim num should be 4 for 4D format, but got %zu", outDimNum);
        CHECK_COND(CheckShape4D(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape for 4D format");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const MhcPostParams &params)
{
    op::Format xFormat = params.x->GetStorageFormat();
    op::Format outFormat = params.out->GetStorageFormat();

    CHECK_COND(!op::IsPrivateFormat(xFormat), ACLNN_ERR_PARAM_INVALID, "format of x %s is invalid.",
               op::ToString(xFormat).GetString());

    CHECK_COND(!op::IsPrivateFormat(outFormat), ACLNN_ERR_PARAM_INVALID, "format of out %s is invalid.",
               op::ToString(outFormat).GetString());

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParam(const MhcPostParams &params)
{
    CHECK_COND(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid format.");
    CHECK_COND(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid dtype.");
    CHECK_COND(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "invalid shape.");

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                                         const aclTensor *hPost, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor)
{
    CHECK_COND(CheckNotNull(x, hRes, hOut, hPost, out) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR,
               "one of required inputs for aclnnMhcPostGetWorkspaceSize is nullptr.");

    MhcPostParams params{x, hRes, hOut, hPost, out};

    aclnnStatus ret = CheckParam(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // Check if input tensors are empty
    if (x->IsEmpty() || hRes->IsEmpty() || hOut->IsEmpty() || hPost->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    L2_DFX_PHASE_1(aclnnMhcPost, DFX_IN(params.x, params.hRes, params.hOut, params.hPost), DFX_OUT(params.out));

    // Convert input tensors to contiguous
    auto reformatedX = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_COND(reformatedX != nullptr, ACLNN_ERR_INNER_NULLPTR, "x Contiguous failed.");
    auto reformatedHRes = l0op::Contiguous(hRes, uniqueExecutor.get());
    CHECK_COND(reformatedHRes != nullptr, ACLNN_ERR_INNER_NULLPTR, "hRes Contiguous failed.");
    auto reformatedHOut = l0op::Contiguous(hOut, uniqueExecutor.get());
    CHECK_COND(reformatedHOut != nullptr, ACLNN_ERR_INNER_NULLPTR, "hOut Contiguous failed.");
    auto reformatedHPost = l0op::Contiguous(hPost, uniqueExecutor.get());
    CHECK_COND(reformatedHPost != nullptr, ACLNN_ERR_INNER_NULLPTR, "hPost Contiguous failed.");

    // Call l0 interface: MhcPost kernel
    // Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
    const aclTensor *mhcPostResult =
        l0op::MhcPost(reformatedX, reformatedHRes, reformatedHOut, reformatedHPost, uniqueExecutor.get());
    CHECK_RET(mhcPostResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Convert output tensor to contiguous tensor and copy to output
    auto viewCopyResult = l0op::ViewCopy(mhcPostResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMhcPost);
    auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "This is an error in MhcPost launch aicore");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif