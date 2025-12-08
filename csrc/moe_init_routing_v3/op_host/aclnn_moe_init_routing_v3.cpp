/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "aclnn_kernels/cast.h"
#include "opdev/common_types.h"
#include "moe_init_routing_v3.h"
#include "aclnn_moe_init_routing_v3.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
    static const int64_t MOE_DIM_2 = 2;
    static const int64_t MOE_DIM_1 = 1;
}

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_X= {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT, DataType::DT_INT8};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPERT_IDX = {DataType::DT_INT32};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SCALE = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_OFFSET= {DataType::DT_FLOAT};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_X_OUT = {DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT, DataType::DT_INT8};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_ROW_IDX_OUT = {DataType::DT_INT32};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPERT_TOKENS_COUNT_OR_CUMSUMOUT = {DataType::DT_INT64};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPANDED_SCALE_OUT = {DataType::DT_FLOAT};

static inline bool CheckNotNull(const aclTensor *x, 
                                const aclTensor *expertIdx,
                                const aclTensor *expandedXOut, 
                                const aclTensor *expandedRowIdxOut, 
                                const aclTensor *expertTokensCountOrCumsumOut, 
                                const aclTensor *expandedScaleOut) {
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(expertIdx, return false);
    OP_CHECK_NULL(expandedXOut,  return false);
    OP_CHECK_NULL(expandedRowIdxOut,  return false);
    OP_CHECK_NULL(expertTokensCountOrCumsumOut, return false);
    OP_CHECK_NULL(expandedScaleOut, return false);

    return true;
}

ACLNN_API aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(const aclTensor *x, 
                                                            const aclTensor *expertIdx,
                                                            const aclTensor *scaleOptional,
                                                            const aclTensor *offsetOptional, 
                                                            int64_t activeNum, 
                                                            int64_t expertCapacity, 
                                                            int64_t expertNum, 
                                                            int64_t dropPadMode, 
                                                            int64_t expertTokensNumType, 
                                                            bool expertTokensNumFlag, 
                                                            int64_t quantMode, 
                                                            const aclIntArray *activeExpertRangeOptional, 
                                                            int64_t rowIdxType, 
                                                            const aclTensor *expandedXOut, 
                                                            const aclTensor *expandedRowIdxOut, 
                                                            const aclTensor *expertTokensCountOrCumsumOut, 
                                                            const aclTensor *expandedScaleOut, 
                                                            uint64_t *workspaceSize, 
                                                            aclOpExecutor **executor)                                                                                 
{   
    L2_DFX_PHASE_1(aclnnMoeInitRoutingV3, 
                    DFX_IN(x, expertIdx, scaleOptional, offsetOptional, 
                            activeNum, expertCapacity, expertNum, dropPadMode, 
                            expertTokensNumType, expertTokensNumFlag, quantMode, activeExpertRangeOptional, rowIdxType), 
                    DFX_OUT(expandedXOut, expandedRowIdxOut, expertTokensCountOrCumsumOut, expandedScaleOut));
    // 参数检查
    auto ret = CheckNotNull(x, expertIdx, expandedXOut, expandedRowIdxOut, 
                            expertTokensCountOrCumsumOut, expandedScaleOut);

    CHECK_RET(ret, ACLNN_ERR_PARAM_NULLPTR);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，将输入self转换成连续的tensor
    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get()); 
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto expertIdxContiguous = l0op::Contiguous(expertIdx, uniqueExecutor.get()); 
    CHECK_RET(expertIdxContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor* scaleContiguous = nullptr;
    const aclTensor* offsetContiguous = nullptr;
    if (scaleOptional != nullptr) {
        scaleContiguous = l0op::Contiguous(scaleOptional, uniqueExecutor.get()); 
        CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    if (offsetOptional != nullptr) {
        offsetContiguous = l0op::Contiguous(offsetOptional, uniqueExecutor.get()); 
        CHECK_RET(offsetContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    auto routingResult = std::tuple<aclTensor*, aclTensor*, aclTensor*, aclTensor*>(nullptr, nullptr, nullptr, nullptr);
    // 调用v3版本的l0接口进行计算
    routingResult = l0op::MoeInitRoutingV3(xContiguous, expertIdxContiguous, scaleContiguous, offsetContiguous, 
                                        activeNum, expertCapacity, expertNum, dropPadMode, expertTokensNumType, expertTokensNumFlag,
                                        quantMode, activeExpertRangeOptional, rowIdxType, expandedXOut, expandedRowIdxOut, 
                                        expertTokensCountOrCumsumOut, expandedScaleOut, uniqueExecutor.get());
    auto [expandedXOut_, expandedRowIdxOut_, expertTokensCountOrCumsumOut_, expandedScaleOut_] = routingResult;
    bool hasNullptr = (expandedXOut_ == nullptr) || (expandedRowIdxOut_ == nullptr) || (expertTokensCountOrCumsumOut_ == nullptr) || (expandedScaleOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    // copyout结果，如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyExpandedXOutResult = l0op::ViewCopy(expandedXOut_, expandedXOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedXOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyExpandedRowIdxOutResult = l0op::ViewCopy(expandedRowIdxOut_, expandedRowIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedRowIdxOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyExpertTokensCountOrCumsumOutResult = l0op::ViewCopy(expertTokensCountOrCumsumOut_, expertTokensCountOrCumsumOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpertTokensCountOrCumsumOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyExpandedScaleOutResult = l0op::ViewCopy(expandedScaleOut_, expandedScaleOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpandedScaleOutResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
ACLNN_API aclnnStatus aclnnMoeInitRoutingV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            aclrtStream stream)
{
  L2_DFX_PHASE_2(aclnnMoeInitRoutingV3);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif