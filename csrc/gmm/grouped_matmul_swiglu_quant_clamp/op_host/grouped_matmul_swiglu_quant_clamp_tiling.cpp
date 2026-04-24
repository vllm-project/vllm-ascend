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
 * \file grouped_matmul_swiglu_quant_tiling.cpp
 * \brief
 */
#include <climits>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "tiling_base/tiling_base.h"
#include "grouped_matmul_swiglu_quant_clamp_tiling.h"
using namespace ge;
using namespace AscendC;
using namespace GroupedMatmulSwigluQuantTiling;
using namespace Ops::Transformer::OpTiling;
namespace {
template <typename T>
static inline auto AlignUp(T a, T base) -> T
{
    if (base == 0) {
        return 0;
    }
    return (a + base - 1) / base * base;
}
} // namespace

namespace optiling {

struct GMMSwigluCompileInfo {
    uint64_t ubSize_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t baseM_ = 128;
    uint32_t baseN_ = 256;
};

static int64_t CalMaxRowInUb_A8W4(const gert::TilingContext *context, const uint64_t ubSize, const uint64_t n)
{
    const uint64_t ALIGNMENT = 8;
    const float WEIGHT_FACTOR = 8.5;
    const uint64_t ALIGNMENT_TERM_FACTOR = 4;
    const uint64_t LINEAR_TERM_FACTOR = 6;
    const uint64_t CONSTANT_TERM = 64;
    const uint64_t MIN_ROW_THRESHOLD = 1;

    // 表达式：8.5 * row * n + 4 * alignUp(row, 8) + 6n + 64 <= ubSize

    // 忽略对齐项的初始估计
    int64_t maxRowEstimate =
        (ubSize - CONSTANT_TERM - LINEAR_TERM_FACTOR * n) / static_cast<int64_t>(WEIGHT_FACTOR * n);

    // 考虑对齐影响
    uint64_t alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    uint64_t totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) +
                         ALIGNMENT_TERM_FACTOR * alignedRow + LINEAR_TERM_FACTOR * n + CONSTANT_TERM;

    // 如果超过UB大小，逐步减少row直到满足条件
    while (totalSize > ubSize && maxRowEstimate > 0) {
        maxRowEstimate--;
        alignedRow = (maxRowEstimate + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
        totalSize = static_cast<uint64_t>(WEIGHT_FACTOR * maxRowEstimate * n) + ALIGNMENT_TERM_FACTOR * alignedRow +
                    LINEAR_TERM_FACTOR * n + CONSTANT_TERM;
    }

    if (maxRowEstimate < MIN_ROW_THRESHOLD) {
        OP_LOGE(context->GetNodeName(), "GMM_SWIGLU_QUANT TILING: No valid row found for n = %lu, ubSize = %lu\n", n,
                ubSize);
        return 0;
    }
    return maxRowEstimate;
}

static int64_t CalMaxRowInUb(const gert::TilingContext *context, const uint64_t ubSize, const uint64_t n)
{
    uint64_t tmpBufSize = (n / SWIGLU_REDUCE_FACTOR) * FP32_DTYPE_SIZE;
    uint64_t perchannleBufSize = n * FP32_DTYPE_SIZE * DOUBLE_BUFFER;
    uint64_t reduceMaxResBufSize = BLOCK_BYTE;
    uint64_t reduceMaxTmpBufSize = BLOCK_BYTE;
    const uint64_t CONSTANT_TERM = 64;
    int64_t remainUbSize = ubSize - tmpBufSize - perchannleBufSize - reduceMaxResBufSize - reduceMaxTmpBufSize;
    int64_t maxRowInUb =
        remainUbSize / (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    int64_t curUb = DOUBLE_BUFFER * (maxRowInUb * (INT32_DTYPE_SIZE * n + n / SWIGLU_REDUCE_FACTOR) +
                                     AlignUp(maxRowInUb, FP32_BLOCK_SIZE) * FP32_DTYPE_SIZE);
    if (curUb > remainUbSize) {
        // 64 : make sure ub does not excceed maxUbSize after align up to 8
        maxRowInUb = (remainUbSize - CONSTANT_TERM) /
                     (n * INT32_DTYPE_SIZE + n / SWIGLU_REDUCE_FACTOR + FP32_DTYPE_SIZE) / DOUBLE_BUFFER;
    }
    if (maxRowInUb < 1) {
        // when n > (ubSize - 72) / 19 = 10330, maxRowInUb < 1
        OP_LOGE(context->GetNodeName(), "GMM_SWIGLU_QUANT TILING: n should not be greater than 10240, now is %lu\n", n);
    }
    return maxRowInUb;
}

static void SetTilingKey(gert::TilingContext *context, bool isSplitWorkSpace, bool isA8W4MSD)
{
    if (isA8W4MSD) { // A8W4 MSD tiling_key使用4
        context->SetTilingKey(A8W4_MSD_TILING_KEY_MODE);
        context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else if (isSplitWorkSpace) {
        context->SetTilingKey(SPLITWORKSPACE_TILING_KEY_MODE);
        context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    } else {
        context->SetTilingKey(COMMON_TILING_KEY_MODE);
        context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    }
}

ASCENDC_EXTERN_C graphStatus TilingGMMSwigluQuant(gert::TilingContext *context)
{
    // set info
    OP_LOGD(context->GetNodeName(), "Begin Run GMM Swiglu Tiling .");
    auto xDesc = context->GetInputDesc(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto weightDesc = context->GetInputDesc(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightDesc);
    ge::DataType xDType = xDesc->GetDataType();
    ge::DataType weightDType = weightDesc->GetDataType();

    bool isA8W4MSD = (xDType == ge::DataType::DT_INT8 && weightDType == ge::DataType::DT_INT4);
    auto compileInfoPtr = context->GetCompileInfo<GMMSwigluCompileInfo>();
    auto xTensor = context->GetInputTensor(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const int64_t m = xTensor->GetStorageShape().GetDim(0);
    const int64_t k = xTensor->GetStorageShape().GetDim(1);
    auto wTensor = context->GetInputTensor(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wTensor);
    // 如果属性不存在，使用无穷大作为默认值（表示无限制）
    auto attrs = context->GetAttrs();
    float limited = std::numeric_limits<float>::infinity();
    if (attrs != nullptr) {
        // std::cout<<"weinachuan1"<<limited<<std::endl;
        if (const double *limitedPtr = attrs->GetAttrPointer<double>(ATTR_INDEX_LIMITED)) {
            limited = static_cast<float>(*limitedPtr);  // 可能覆盖为有限值或 inf
            // std::cout<<"weinachuan2"<<limited<<std::endl;
        }
        // else: 保持默认 inf（属性不存在）
    }
    // std::cout<<"weinachuan3"<<limited<<std::endl;
    int64_t n = 0;
    if (wTensor->GetStorageShape().GetDimNum() == ND_WEIGHT_DIM_LIMIT) { // ND
        n = wTensor->GetStorageShape().GetDim(DIM_2);
    } else if (wTensor->GetStorageShape().GetDimNum() == NZ_WEIGHT_DIM_LIMIT) { // NZ
        n = wTensor->GetStorageShape().GetDim(DIM_1) * wTensor->GetStorageShape().GetDim(DIM_4);
    }
    auto wScaleTensor = context->GetInputTensor(WEIGHT_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, wScaleTensor);
    int64_t quantGroupNum = 0;
    if (wScaleTensor->GetStorageShape().GetDimNum() == PERCHANNEL_WSCALE_DIM_LIMIT) { // perChannel
        quantGroupNum = 1;
    } else if (wScaleTensor->GetStorageShape().GetDimNum() == PERGROUP_WSCALE_DIM_LIMIT) { // perGroup
        quantGroupNum = wScaleTensor->GetStorageShape().GetDim(1);
    }
    auto groupListTensor = context->GetDynamicInputTensor(GROUPLIST_INDEX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTensor);
    const int64_t groupNum = groupListTensor->GetStorageShape().GetDim(0);
    GMMSwigluQuantTilingData tilingData;
    int64_t row = 0;
    if (isA8W4MSD) {
        row = CalMaxRowInUb_A8W4(context, compileInfoPtr->ubSize_, n);
    } else {
        row = CalMaxRowInUb(context, compileInfoPtr->ubSize_, n);
    }

    tilingData.gmmSwigluBaseParams.set_groupNum(groupNum);
    tilingData.gmmSwigluBaseParams.set_coreNum(compileInfoPtr->aicNum_);
    tilingData.gmmSwigluBaseParams.set_K(k);
    tilingData.gmmSwigluBaseParams.set_N(n);
    tilingData.gmmSwigluBaseParams.set_M(m);
    tilingData.gmmSwigluBaseParams.set_baseM(A8W4_BASEM);
    tilingData.gmmSwigluBaseParams.set_baseN(A8W4_BASEN);
    tilingData.gmmSwigluBaseParams.set_limited(limited);
    tilingData.gmmSwiglu.set_maxProcessRowNum(row);
    tilingData.gmmSwiglu.set_groupListLen(groupNum);
    tilingData.gmmSwiglu.set_tokenLen(n);

    tilingData.gmmSwigluBaseParams.set_quantGroupNum(quantGroupNum);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    using namespace matmul_tiling;

    MatmulApiTiling tiling(ascendcPlatform);
    tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_INT4);
    tiling.SetBType(TPosition::GM, CubeFormat::NZ, matmul_tiling::DataType::DT_INT4);
    tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    tiling.SetBias(false);
    tiling.SetShape(A8W4_BASEM, A8W4_BASEN, k);
    tiling.SetFixSplit(A8W4_BASEM, A8W4_BASEN, A8W4_BASEK);
    tiling.SetOrgShape(m, n, k);
    tiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(
        tiling.GetTiling(tilingData.mmTilingData) == -1,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "grouped_matmul_swiglu_quant_tiling, get tiling failed"),
        return GRAPH_FAILED);
    if (isA8W4MSD) {
        tilingData.mmTilingData.set_baseM(A8W4_BASEM);
        tilingData.mmTilingData.set_baseN(A8W4_BASEN);
        tilingData.mmTilingData.set_baseK(A8W4_BASEK);
        tilingData.mmTilingData.set_dbL0B(DOUBLE_BUFFER);
        tilingData.mmTilingData.set_stepKa(NUM_FOUR);
        tilingData.mmTilingData.set_stepKb(NUM_FOUR);
        tilingData.mmTilingData.set_depthA1(NUM_EIGHT);
        tilingData.mmTilingData.set_depthB1(NUM_EIGHT);
        tilingData.mmTilingData.set_stepM(1);
        tilingData.mmTilingData.set_stepN(1);
    }
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    int64_t usrWorkspaceLimit = USER_WORKSPACE_LIMIT;
    int64_t mLimit = 0;
    if (isA8W4MSD) {
        mLimit = ((usrWorkspaceLimit / DOUBLE_WORKSPACE_SPLIT) / (k * sizeof(int8_t) + DOUBLE_ROW * n * sizeof(half)));
    } else {
        mLimit = ((usrWorkspaceLimit / DOUBLE_WORKSPACE_SPLIT) / INT32_DTYPE_SIZE) / n;
    }
    OP_CHECK_IF(mLimit <= 0,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "mLimit is %ld must over then 0.", mLimit),
                return GRAPH_FAILED);
    tilingData.gmmSwigluBaseParams.set_mLimit(mLimit);
    if (isA8W4MSD) {
        int workSpaceMTemp = mLimit * DOUBLE_WORKSPACE_SPLIT;
        tilingData.gmmSwigluBaseParams.set_workSpaceOffset1(workSpaceMTemp * k * sizeof(int8_t));
        tilingData.gmmSwigluBaseParams.set_workSpaceOffset2(2 * workSpaceMTemp * n * sizeof(half));
        workspaceSizes[0] =
            SYS_WORKSPACE_SIZE +                    // 系统预留16MB
            (workSpaceMTemp * k * sizeof(int8_t)) + // 第一阶段 预处理左矩阵 (mLimit, K) * int8 * 2(double WorkSpace)
            (DOUBLE_ROW * workSpaceMTemp * n *
             sizeof(half)); // 第二阶段 矩阵乘结果 (2 * mLimit, N) * fp16 * 2(double WorkSpace)
    } else {
        int workSpaceMTemp = (mLimit * DOUBLE_WORKSPACE_SPLIT > m ? m : mLimit * DOUBLE_WORKSPACE_SPLIT);
        tilingData.gmmSwigluBaseParams.set_workSpaceOffset1(0);
        tilingData.gmmSwigluBaseParams.set_workSpaceOffset2(0);
        workspaceSizes[0] = SYS_WORKSPACE_SIZE + (workSpaceMTemp * n * sizeof(int32_t));
    }
    bool isSplitWorkSpace = m > mLimit * DOUBLE_WORKSPACE_SPLIT;
    OP_LOGD(context->GetNodeName(), "grouped_matmul_swiglu_quant_tiling.");
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.groupNum:      %ld", groupNum);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.coreNum:       %u ", compileInfoPtr->aicNum_);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.M:             %ld", m);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.K:             %ld", k);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.N:             %ld", n);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.baseM:         %ld", A8W4_BASEM);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.baseN:         %ld", A8W4_BASEN);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.mLimit:        %ld", mLimit);
    OP_LOGD(context->GetNodeName(), "gmmSwigluBaseParams.quantGroupNum: %ld", quantGroupNum);
    OP_LOGD(context->GetNodeName(), "gmmSwiglu.maxProcessRowNum:        %ld", row);
    OP_LOGD(context->GetNodeName(), "gmmSwiglu.groupListLen:            %ld", groupNum);
    OP_LOGD(context->GetNodeName(), "gmmSwiglu.tokenLen:                %ld", n);
    OP_LOGD(context->GetNodeName(), "USER_WORKSPACE_LIMIT:              %ld", usrWorkspaceLimit);
    OP_LOGD(context->GetNodeName(), "workspaceSizes:                    %lu", workspaceSizes[0]);
    OP_LOGD(context->GetNodeName(), "isSplitWorkSpace:                  %s", isSplitWorkSpace ? "true" : "false");
    OP_LOGD(context->GetNodeName(), "GMMSWIGLUQUANT_TILING: baseM is %u, baseK is %u, baseN is %u.", A8W4_BASEM, A8W4_BASEK, A8W4_BASEN);
    SetTilingKey(context, isSplitWorkSpace, isA8W4MSD);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->SetBlockDim(compileInfoPtr->aicNum_); // block dim is the number of aicube
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGD(context->GetNodeName(), "End Run GMM Swiglu Tiling.");
    return GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C graphStatus TilingPrepareForGMMSwigluQuant(gert::TilingParseContext *context)
{
    // get info
    fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto compileInfoPtr = context->GetCompiledInfo<GMMSwigluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aicNum_ = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize_);
    OP_LOGD(context->GetNodeName(), "ubSize is %lu, aicNum is %u.", compileInfoPtr->ubSize_, compileInfoPtr->aicNum_);
    return GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulSwigluQuantClamp)
    .Tiling(TilingGMMSwigluQuant)
    .TilingParse<GMMSwigluCompileInfo>(TilingPrepareForGMMSwigluQuant);
} // namespace optiling
