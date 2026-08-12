/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mc2_aclnn_util.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "mc2_log_compat.h"

namespace MC2Aclnn {

static constexpr size_t MX_SCALE_MAX_DIM = 3;

bool IsNeedScaleTrans(const aclTensor *mxScaleTensor)
{
    if (mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        return Ops::Transformer::IsTransposeLastTwoDims(mxScaleTensor);
    }
    bool transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);          // 1:最后一个维度
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 2);    // 2:倒数第二个维度
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);     // 3:倒数第三个维度
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&                // 3:倒数第三个维度
        mxScaleTensor->GetViewStrides()[dimNum - 2] == lastDim * lastThirdDim) { // 2:倒数第二个维度
        transposeFlag = true;
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    return transposeFlag;
}

bool IsTensorContiguous(const aclTensor *tensor)
{
    OP_CHECK_NULL(tensor, return false);

    int dimNum = tensor->GetViewShape().GetDimNum();
    if (dimNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("IsTensorContiguous", "tensor", std::to_string(dimNum).c_str(),
                                              "The shape dim of tensor must be greater than 0");
        return false;
    }

    auto strides = tensor->GetViewStrides();
    auto shape = tensor->GetViewShape();
    int64_t expectedStride = 1;
    for (int i = dimNum - 1; i >= 0; i--) {
        int currentStride = strides[i];
        if (currentStride != expectedStride) {
            return false;
        }
        expectedStride *= shape.GetDim(i);
    }
    return true;
}

} // namespace MC2Aclnn
