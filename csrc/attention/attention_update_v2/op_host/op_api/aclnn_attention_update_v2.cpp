/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_attention_update_v2.cpp
 * \brief
 */

#include "aclnn_attention_update_v2.h"
#include "attention_update_v2.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace attention_update_v2 {

static const int64_t LSE_TOTAL_LENGTH_DIM = 1;    // lse [sp, bsh]: bsh at dim 1
static const int64_t GO_TOTAL_LENGTH_DIM = 1;     // go  [sp, bsh, hd]: bsh at dim 1
static const int64_t GO_HD_DIM = 2;               // go  [sp, bsh, hd]: hd at dim 2

static const int64_t OUT_TOTAL_LENGTH_DIM = 0;    // out [bsh, hd]: bsh at dim 0
static const int64_t OUT_HD_DIM = 1;              // out [bsh, hd]: hd at dim 1
static const int64_t ATTENTION_UPDATE_V2_OUT_DIM_NUM = 2;
static const int64_t LSE_DIM_NUM = 2;             // lse [sp, bsh]: 2D
static const int64_t LSE_OUT_DIM_NUM = 1;         // lseOut [bsh]: 1D
static const int64_t GO_DIM_NUM = 3;              // go  [sp, bsh, hd]: 3D
static const int64_t HD_MULTIPLE = 8;
static const int64_t HD_MAX = 512;
static const int64_t SP_MAX = 16;

static const std::initializer_list<op::DataType> ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST = {
                                DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST_LOCALOUT = {
                                DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST_95 = {
                                DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static inline bool CheckNotNull(const aclTensor* lse, const aclTensor* localOut, const aclTensor* out) {
    OP_CHECK_NULL(lse, return false);
    OP_CHECK_NULL(localOut, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* lse, const aclTensor* localOut, const aclTensor* out) {
    if (lse->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(lse, ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST, return false);
    }
    if (localOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(localOut, ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST_LOCALOUT, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST_LOCALOUT, return false);
    return true;
}

static inline bool CheckDtypeValid_95(const aclTensor* lse, const aclTensor* localOut, const aclTensor* out, const aclTensor* lseOut, uint64_t updataType) {
    if (lse->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(lse, {DataType::DT_FLOAT}, return false);
    }
    if (localOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(localOut, ATTENTION_UPDATE_V2_DTYPE_SUPPORT_LIST_95, return false);
    }
    OP_CHECK_DTYPE_NOT_SAME(out, localOut, return false);
    if (updataType) {
        OP_CHECK_DTYPE_NOT_SUPPORT(lseOut, {DataType::DT_FLOAT}, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensor* lse, const aclTensor* localOut, const aclTensor* out,
                                const aclTensor* lseOut, const int64_t updateType) {
    auto lseShape = lse->GetViewShape();       // [sp, bsh]
    auto goShape  = localOut->GetViewShape();  // [sp, bsh, hd]

    OP_CHECK(lseShape.GetDimNum() == LSE_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of lse should be %ld, but which is %ld.",
                LSE_DIM_NUM, lseShape.GetDimNum()),
        return false);
    OP_CHECK(goShape.GetDimNum() == GO_DIM_NUM,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of go should be %ld, but which is %ld.",
                GO_DIM_NUM, goShape.GetDimNum()),
        return false);

    auto lseBshc      = lseShape.GetDim(LSE_TOTAL_LENGTH_DIM);  // bsh (dim 1)
    auto localOutBshc = goShape.GetDim(GO_TOTAL_LENGTH_DIM);     // bsh (dim 1)
    auto goHd         = goShape.GetDim(GO_HD_DIM);               // hd  (dim 2)

    OP_CHECK(goHd > 0 && goHd % HD_MULTIPLE == 0 && goHd <= HD_MAX,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of localOut[%ld] must be a multiple of 8 and <= %ld, but which is %ld.",
                GO_HD_DIM, HD_MAX, goHd),
        return false);
    OP_CHECK(lseBshc == localOutBshc,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of localOut[%ld]: %ld.",
                LSE_TOTAL_LENGTH_DIM, lseBshc, GO_TOTAL_LENGTH_DIM, localOutBshc),
        return false);

    if (updateType == 0 || updateType == 1) {
        auto outShape  = out->GetViewShape();
        auto outDimNum = outShape.GetDimNum();
        OP_CHECK(outDimNum == ATTENTION_UPDATE_V2_OUT_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of out should be %ld, but which is %ld.",
                    ATTENTION_UPDATE_V2_OUT_DIM_NUM, outDimNum),
            return false);

        auto outBshc = outShape.GetDim(OUT_TOTAL_LENGTH_DIM);
        OP_CHECK(lseBshc == outBshc,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of out[%ld]: %ld.",
                    LSE_TOTAL_LENGTH_DIM, lseBshc, OUT_TOTAL_LENGTH_DIM, outBshc),
            return false);

        auto outHd = outShape.GetDim(OUT_HD_DIM);
        OP_CHECK(goHd == outHd,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of localOut[%ld]: %ld must equal to size of out[%ld]: %ld.",
                    GO_HD_DIM, goHd, OUT_HD_DIM, outHd),
            return false);

        if (updateType == 1) {
            auto lseOutShape  = lseOut->GetViewShape();
            auto lseOutDimNum = lseOutShape.GetDimNum();
            OP_CHECK(lseOutDimNum == LSE_OUT_DIM_NUM,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim num of lseOut should be %ld, but which is %ld.",
                        LSE_OUT_DIM_NUM, lseOutDimNum),
                return false);

            auto lseOutBshc = lseOutShape.GetDim(OUT_TOTAL_LENGTH_DIM);
            OP_CHECK(lseBshc == lseOutBshc,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of lse[%ld]: %ld must equal to size of lseOut[%ld]: %ld.",
                        LSE_TOTAL_LENGTH_DIM, lseBshc, OUT_TOTAL_LENGTH_DIM, lseOutBshc),
                return false);
        }
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported updateType: %ld.", updateType);
        return false;
    }
    return true;
}

static inline bool CheckShape_95(const aclTensor* lse, const aclTensor* localOut, const aclTensor* out,
                                const aclTensor* lseOut, const int64_t updateType) {
    // sp count already checked in CheckSp_95; same shape rules as CheckShape
    return CheckShape(lse, localOut, out, lseOut, updateType);
}

static inline bool CheckSp_95(const aclTensor* lse, const aclTensor* localOut) {
    auto lseSp = lse->GetViewShape().GetDim(0);     // sp from lse [sp, bsh]
    auto goSp  = localOut->GetViewShape().GetDim(0); // sp from go  [sp, bsh, hd]

    OP_CHECK(lseSp > 0 && lseSp <= SP_MAX,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sp of lse must be in (0, %ld], but which is %ld.", SP_MAX, lseSp),
        return false);
    OP_CHECK(lseSp == goSp,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sp of lse: %ld must equal sp of localOut: %ld.", lseSp, goSp),
        return false);
    return true;
}

static inline bool CheckUpdateTypeAndLseOut(int64_t updateType, const aclTensor* lseOut) {
    if (updateType == 0 && lseOut != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lseOut should be nullptr when updateType is %ld.", updateType);
        return false;
    }
    if (updateType == 1 && lseOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "lseOut should not be nullptr when updateType is %ld.", updateType);
        return false;
    }

    if (updateType != 1 && updateType != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updateType should be 0 or 1 but got %ld.", updateType);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams_95(const aclTensor* lse, const aclTensor* localOut,
            const aclTensor* out, const aclTensor* lseOut, const int64_t updateType) {
    // lseOut作输出
    CHECK_RET(CheckUpdateTypeAndLseOut(updateType, lseOut), ACLNN_ERR_PARAM_INVALID);

    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(lse, localOut, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入sp是否支持
    CHECK_RET(CheckSp_95(lse, localOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid_95(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckShape_95(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const aclTensor* lse, const aclTensor* localOut,
            const aclTensor* out, const aclTensor* lseOut, const int64_t updateType) {
    // 启用lseout
    CHECK_RET(CheckUpdateTypeAndLseOut(updateType, lseOut), ACLNN_ERR_PARAM_INVALID);

    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(lse, localOut, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(lse, localOut, out), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入sp是否支持
    CHECK_RET(CheckSp_95(lse, localOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和out的shape是否一致
    CHECK_RET(CheckShape(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckShape_95(lse, localOut, out, lseOut, updateType), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

} // namespace attention_update_v2

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnAttentionUpdateV2GetWorkspaceSize(
    const aclTensor* lse, const aclTensor* localOut, int64_t updateType,
    aclTensor* out, aclTensor* lseOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    // printf("zhy test");
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnAttentionUpdateV2,
                   DFX_IN(lse, localOut, updateType),
                   DFX_OUT(out, lseOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    aclnnStatus ret;
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        ret = attention_update_v2::CheckParams_95(lse, localOut, out, lseOut, updateType);
    } else {
        ret = attention_update_v2::CheckParams(lse, localOut, out, lseOut, updateType);
    }
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (lse->IsEmpty() || localOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto lseContiguous      = l0op::Contiguous(lse, uniqueExecutor.get());
    auto localOutContiguous = l0op::Contiguous(localOut, uniqueExecutor.get());

    auto [AttentionUpdateV2Res, lseM] = l0op::AttentionUpdateV2(lseContiguous, localOutContiguous, updateType, uniqueExecutor.get());
    CHECK_RET(AttentionUpdateV2Res != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(AttentionUpdateV2Res, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (updateType == 1) {
        CHECK_RET(lseM != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyResultLseM = l0op::ViewCopy(lseM, lseOut, uniqueExecutor.get());
        CHECK_RET(viewCopyResultLseM != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAttentionUpdateV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAttentionUpdateV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif