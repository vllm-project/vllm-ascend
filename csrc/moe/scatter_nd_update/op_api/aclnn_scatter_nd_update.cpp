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
 * \file aclnn_scatter_nd_update.cpp
 * \brief
 */

#include "aclnn_scatter_nd_update.h"
#include "scatter_nd_update.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_executor.h"
using namespace op;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BOOL, op::DataType::DT_INT16,
    op::DataType::DT_BF16,  op::DataType::DT_INT64,   op::DataType::DT_INT8, op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,    op::DataType::DT_FLOAT16, op::DataType::DT_BOOL,        op::DataType::DT_BF16,
    op::DataType::DT_INT64,    op::DataType::DT_INT8,    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                             op::DataType::DT_INT32};

static bool CheckNotNull(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    OP_CHECK_NULL(varRef, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(updates, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socVersion = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (socVersion == NpuArch::DAV_3510) {
        return ASCEND950_DTYPE_DTYPE_SUPPORT_LIST;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    // 检查self的数据类型是否在算子的支持列表内
    auto supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(varRef, supportList, return false);
    // 检查index的数据类型是否在算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(indices, INDEX_DTYPE_SUPPORT_LIST, return false);
    // varRef和updates的数据类型要一致
    if (varRef->GetDataType() != updates->GetDataType()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "updates dtype %s should be in same with varRef dtype %s.",
                op::ToString(updates->GetDataType()).GetString(), op::ToString(varRef->GetDataType()).GetString());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(varRef, indices, updates), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(varRef, indices, updates), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 判断指定轴范围是否连续
// startAxis: 起始轴（包含），endAxis: 结束轴（不包含）
// 当 startAxis == endAxis 时区间为空，视为平凡连续返回 true（不影响 arch35 的调用点，
// 那两个点外层都已 guard 住空区间；专门给 arch32 的 1D 场景放行非连续优化路径）。
static bool IsAxesContiguous(const aclTensor* tensor, int64_t startAxis, int64_t endAxis)
{
    if (tensor == nullptr || startAxis < 0 || endAxis < 0 || startAxis > endAxis) {
        return false;
    }
    if (startAxis == endAxis) {
        return true;
    }

    auto viewShape = tensor->GetViewShape();
    auto viewStrides = tensor->GetViewStrides();
    int64_t dimNum = viewShape.GetDimNum();
    if (endAxis > dimNum) {
        return false;
    }

    int64_t validStride = 1;
    for (int64_t i = endAxis - 1; i >= startAxis; i--) {
        if (viewShape.GetDim(i) == 1) {
            continue;
        }
        if (viewStrides[i] != validStride) {
            return false;
        }
        validStride *= viewShape.GetDim(i);
    }
    return true;
}

// arch32 (910b/910_93) 仅支持 var.stride[0] 非连续、其余 stride 全部连续的窄子集。
// 校验：dim>=1 全连续，dim 0 stride 大于 contiguous 期望值。
static bool IsSupportNonContiguousArch32(const aclTensor* varRef, int64_t indexAxisNum)
{
    if (varRef == nullptr || indexAxisNum < 1) {
        return false;
    }
    auto viewShape = varRef->GetViewShape();
    auto viewStrides = varRef->GetViewStrides();
    int64_t varRefDimNum = viewShape.GetDimNum();
    if (indexAxisNum > varRefDimNum) {
        return false;
    }
    // dim>=1 必须全部连续
    if (!IsAxesContiguous(varRef, 1, varRefDimNum)) {
        return false;
    }
    // dim 0 的 stride 必须 >= contiguous 期望值（>= 而非 ==，等于的退回连续路径）
    int64_t expectedStride0 = 1;
    for (int64_t i = 1; i < varRefDimNum; ++i) {
        expectedStride0 *= viewShape.GetDim(i);
    }
    return viewStrides[0] > expectedStride0;
}

// 判断 varRef 是否满足：总体非连续，非索引轴部分连续
// indexAxisNum: 索引轴的数量（varRef 的前 indexAxisNum 个维度是索引轴）
static bool IsSupportNonContiguous(const aclTensor* varRef, int64_t indexAxisNum)
{
    if (varRef == nullptr || indexAxisNum < 0) {
        return false;
    }

    auto viewShape = varRef->GetViewShape();
    int64_t varRefDimNum = viewShape.GetDimNum();
    if (indexAxisNum > varRefDimNum) {
        return false;
    }

    bool indicesContiguous = true;
    if (indexAxisNum > 0) {
        indicesContiguous = IsAxesContiguous(varRef, 0, varRefDimNum);
    }

    bool nonIndexAxesContiguous = true;
    if (indexAxisNum < varRefDimNum) {
        nonIndexAxesContiguous = IsAxesContiguous(varRef, indexAxisNum, varRefDimNum);
    }

    return nonIndexAxesContiguous && (!indicesContiguous);
}

// 执行 ScatterNdUpdate 算子计算（公共实现）
template <typename ExecutorT>
static aclnnStatus ExecuteScatterNdUpdate(const aclTensor* varRef, const aclTensor* indices, const aclTensor* updates,
                                          bool needViewCopy, uint64_t* workspaceSize, aclOpExecutor** executor,
                                          ExecutorT& uniqueExecutor)
{
    // 将 indices 转换成连续的 tensor
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 updates 转换成连续的 tensor
    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 执行 L0 算子
    auto scatterUpdateRes = l0op::ScatterNdUpdate(varRef, indicesContiguous, updatesContiguous, false,
                                                  uniqueExecutor.get());
    CHECK_RET(scatterUpdateRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果需要，将计算结果拷贝到输出 data 上
    if (needViewCopy) {
        auto viewCopyResult = l0op::ViewCopy(scatterUpdateRes, varRef, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取计算过程中需要使用的 workspace 大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 非连续优化路径：indices 与 updates 转连续，varRef 使用 CreateView 设置 stride
template <typename ExecutorT>
static aclnnStatus ProcessNonContiguousCase(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates,
                                            uint64_t* workspaceSize, aclOpExecutor** executor,
                                            ExecutorT& uniqueExecutor)
{
    // 将 indices 转换成连续的 tensor
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将 updates 转换成连续的 tensor
    auto updatesContiguous = l0op::Contiguous(updates, uniqueExecutor.get());
    CHECK_RET(updatesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 使用 CreateView 为 varRef 创建视图，保持原始 stride
    auto varRefView = uniqueExecutor->CreateView(varRef, varRef->GetViewShape(), varRef->GetStorageShape(),
                                                 varRef->GetViewStrides(), varRef->GetViewOffset());
    CHECK_RET(varRefView != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto scatterUpdateRes = l0op::ScatterNdUpdate(varRefView, indicesContiguous, updatesContiguous, false,
                                                  uniqueExecutor.get());
    CHECK_RET(scatterUpdateRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

// 常规路径：将所有输入转换成连续的 tensor
template <typename ExecutorT>
static aclnnStatus ProcessContiguousCase(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates,
                                         uint64_t* workspaceSize, aclOpExecutor** executor,
                                         ExecutorT& uniqueExecutor)
{
    // 将输入 varRef 转换成连续的 tensor
    auto varRefContiguous = l0op::Contiguous(varRef, uniqueExecutor.get());
    CHECK_RET(varRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool needViewCopy = !IsContiguous(varRef);
    return ExecuteScatterNdUpdate(varRefContiguous, indices, updates, needViewCopy, workspaceSize, executor,
                                  uniqueExecutor);
}

aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(aclTensor* varRef, const aclTensor* indices, const aclTensor* updates,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnScatterNdUpdate, DFX_IN(varRef, indices, updates), DFX_OUT(varRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(varRef, indices, updates);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    uint64_t varRefDimNum = varRef->GetViewShape().GetDimNum();
    uint64_t indicesDimNum = indices->GetViewShape().GetDimNum();
    uint64_t updatesDimNum = updates->GetViewShape().GetDimNum();

    // 检查维度数是否超过8维
    constexpr uint64_t MAX_DIM_NUM = 8;
    if (varRefDimNum > MAX_DIM_NUM || indicesDimNum > MAX_DIM_NUM || updatesDimNum > MAX_DIM_NUM) {
        OP_LOGW("varRef/indices/updates dim num(%lu, %lu, %lu) exceeds max limit %lu.", varRefDimNum, indicesDimNum,
                updatesDimNum, MAX_DIM_NUM);
    }

    if (varRef->IsEmpty() || indices->IsEmpty() || updates->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 判断 varRef 索引轴部分是否非连续，非索引轴部分是否连续
    auto indicesShape = indices->GetViewShape();
    constexpr int64_t MAX_INDICES_RANK = 4;
    // 索引轴的数量是 indices 的最后一个维度
    int64_t indexAxisNum = indicesShape.GetDim(indicesShape.GetDimNum() - 1);
    // 检查芯片架构，只在 arch35 时支持非连续场景
    auto& platformInfo = op::GetCurrentPlatformInfo();
    auto socVersion = platformInfo.GetCurNpuArch();

    // 检查是否满足准入条件
    bool dimCheck = (indicesDimNum > 0 && indexAxisNum <= MAX_INDICES_RANK);

    if (dimCheck) {
        if (socVersion == NpuArch::DAV_3510 && IsSupportNonContiguous(varRef, indexAxisNum)) {
            // arch35：非索引轴连续，索引轴非连续 → 通用 view 优化路径
            return ProcessNonContiguousCase(varRef, indices, updates, workspaceSize, executor, uniqueExecutor);
        }
        if (socVersion == NpuArch::DAV_2201 && IsSupportNonContiguousArch32(varRef, indexAxisNum)) {
            // arch32：仅支持 var.stride[0] 非连续、其余 stride 连续 → 透传给 arch32 tiling/kernel
            return ProcessNonContiguousCase(varRef, indices, updates, workspaceSize, executor, uniqueExecutor);
        }
    }

    // 常规路径：将所有输入转换成连续的 tensor
    return ProcessContiguousCase(varRef, indices, updates, workspaceSize, executor, uniqueExecutor);
}

aclnnStatus aclnnScatterNdUpdate(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnScatterNdUpdate);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
