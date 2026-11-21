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
 * \file grouped_matmul_situ_quant_arch35_tiling.cpp
 * \brief host tiling for the arch35 fused GroupedMatmul(MX A8W4) + SiTU + MXQuant op.
 *
 * 与拆链（npu_grouped_matmul + situ_mx_quant）相比，本 tiling 只做三件事：
 *   1) 校验融合路径的 shape/dtype/attr 契约（K 32 对齐、N 128 对齐、groupListType 0/1）；
 *   2) 填 GMMSQWeightQuantTilingData（与 grouped_matmul_swiglu_quant_v2 MxA8W4 共用同一调度契约），
 *      额外写入 SiTU 的 beta/linearBeta；
 *   3) 产出静态 grid（= 全 AIC 核数）与模板 tilingKey。
 * 组表不进 tiling blob：group_list 以 device 张量下发，kernel 侧每个 basic block 直接读取，
 * 因此同一份 tiling 在图捕获下可被任意次 replay 复用（不冻结 host 侧的 token 分布）。
 *
 * 输入/属性序号按本仓库算子原型（weight/weight_scale 在 group_list 之前，
 * group_list_type/beta/linear_beta 是最前面的三个属性）；arch35 只实际使用
 * x / x_scale / weight / weight_scale / group_list 五个输入。
 */

#include <algorithm>
#include <cstring>

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/grouped_matmul_situ_quant_arch35_tiling_data.h"
#include "../op_kernel/grouped_matmul_situ_quant_tiling_key.h"
#include "grouped_matmul_situ_quant_arch35_tiling.h"

namespace optiling {
namespace {

constexpr size_t INPUT_X_INDEX = 0;
constexpr size_t INPUT_X_SCALE_INDEX = 1;
constexpr size_t INPUT_WEIGHT_INDEX = 2;
constexpr size_t INPUT_WEIGHT_SCALE_INDEX = 3;
constexpr size_t INPUT_GROUP_LIST_INDEX = 4;

constexpr size_t ATTR_GROUP_LIST_TYPE_INDEX = 0;
constexpr size_t ATTR_BETA_INDEX = 1;
constexpr size_t ATTR_LINEAR_BETA_INDEX = 2;

// weight_scale 形状：TensorList 为 (N, K/32, 2)，单张量堆叠为 (E, N, K/32, 2)
constexpr size_t WEIGHT_SCALE_DIM_LIST = 3;
constexpr size_t WEIGHT_SCALE_DIM_STACKED = 4;
// MX 量化块大小（e8m0 per-32）
constexpr uint32_t MX_BLOCK_SIZE = 32;
// K 按 32 对齐（MX 分组），N 按 128 对齐（gate/up 各 64 列）
constexpr int64_t ALIGN_SIZE_K = 32;
constexpr int64_t ALIGN_SIZE_N = 128;
constexpr int64_t MIN_K_SIZE = 32;
constexpr int64_t MIN_N_SIZE = 128;
// 专家数上限：物理上最多 896（总专家 896 / EP>=1），1024 是留余量的通用上界。
// 注意 weight 走 TensorList 形态时，GE 框架另有 256 的列表长度硬限制，会先于本校验报 161002。
constexpr int64_t MAX_GROUP_NUM = 1024;
constexpr uint64_t DIM_NUM_X = 2;
constexpr uint64_t DIM_NUM_GROUP_LIST = 1;
constexpr uint32_t MAX_GROUP_SCAN = 1024;

uint32_t GetDynamicInputListSize(gert::TilingContext &context, size_t inputIndex)
{
    uint32_t count = 0;
    for (; count < MAX_GROUP_SCAN; ++count) {
        if (context.GetDynamicInputShape(inputIndex, count) == nullptr) {
            break;
        }
    }
    return count;
}

} // namespace

ge::graphStatus TilingForGMMSituQuantArch35(gert::TilingContext *context)
{
    const char *nodeName = context->GetNodeName();

    // ------------------------------ 输入 shape ------------------------------
    // 注意：GetInputShape 按「扁平化后的输入槽位」索引，TensorList 长度 > 1 时
    // weight / weight_scale 会占掉多个槽位，导致 group_list 的静态索引落到 weight 实例上。
    // 这里统一用 GetDynamicInputShape(descIdx, 0)：它按算子原型里的输入序号索引，与列表长度无关。
    const auto *xShapePtr = context->GetDynamicInputShape(INPUT_X_INDEX, 0);
    const auto *xScaleShapePtr = context->GetDynamicInputShape(INPUT_X_SCALE_INDEX, 0);
    const auto *weightScaleShapePtr = context->GetDynamicInputShape(INPUT_WEIGHT_SCALE_INDEX, 0);
    const auto *groupListShapePtr = context->GetDynamicInputShape(INPUT_GROUP_LIST_INDEX, 0);
    const auto *weightShapePtr = context->GetDynamicInputShape(INPUT_WEIGHT_INDEX, 0);
    if (xShapePtr == nullptr || xScaleShapePtr == nullptr || groupListShapePtr == nullptr ||
        weightScaleShapePtr == nullptr || weightShapePtr == nullptr) {
        OP_LOGE(nodeName, "[gmsq_tiling] input shape is nullptr.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &xShape = xShapePtr->GetOriginShape();
    const gert::Shape &groupListShape = groupListShapePtr->GetOriginShape();
    const gert::Shape &weightScaleShape = weightScaleShapePtr->GetOriginShape();
    if (xShape.GetDimNum() != DIM_NUM_X || groupListShape.GetDimNum() != DIM_NUM_GROUP_LIST) {
        OP_LOGE(nodeName, "[gmsq_tiling] rank mismatch: x=%zu groupList=%zu", xShape.GetDimNum(),
                groupListShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const int64_t mSize = xShape.GetDim(0);
    const int64_t kSize = xShape.GetDim(1);
    const int64_t groupNum = groupListShape.GetDim(0);

    // N 从 weight_scale 读取：与 weight 的 NZ 物理布局解耦，两种 weight 形态共用同一套推导。
    const size_t weightScaleDimNum = weightScaleShape.GetDimNum();
    if (weightScaleDimNum != WEIGHT_SCALE_DIM_LIST && weightScaleDimNum != WEIGHT_SCALE_DIM_STACKED) {
        OP_LOGE(nodeName, "[gmsq_tiling] weight_scale rank %zu is invalid (expect 3 or 4).", weightScaleDimNum);
        return ge::GRAPH_FAILED;
    }
    const int64_t nSize = weightScaleShape.GetDim(weightScaleDimNum == WEIGHT_SCALE_DIM_STACKED ? 1 : 0);

    // ------------------------------ 输入 dtype ------------------------------
    const auto *xDesc = context->GetDynamicInputDesc(INPUT_X_INDEX, 0);
    const auto *xScaleDesc = context->GetDynamicInputDesc(INPUT_X_SCALE_INDEX, 0);
    const auto *weightDesc = context->GetDynamicInputDesc(INPUT_WEIGHT_INDEX, 0);
    const auto *weightScaleDesc = context->GetDynamicInputDesc(INPUT_WEIGHT_SCALE_INDEX, 0);
    const auto *groupListDesc = context->GetDynamicInputDesc(INPUT_GROUP_LIST_INDEX, 0);
    if (xDesc == nullptr || xScaleDesc == nullptr || groupListDesc == nullptr || weightDesc == nullptr ||
        weightScaleDesc == nullptr) {
        OP_LOGE(nodeName, "[gmsq_tiling] input desc is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetDataType() != ge::DT_FLOAT8_E4M3FN || xScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0 ||
        weightDesc->GetDataType() != ge::DT_FLOAT4_E2M1 || weightScaleDesc->GetDataType() != ge::DT_FLOAT8_E8M0 ||
        groupListDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE(nodeName, "[gmsq_tiling] dtype mismatch: x=%d x_scale=%d weight=%d weight_scale=%d group_list=%d.",
                static_cast<int>(xDesc->GetDataType()), static_cast<int>(xScaleDesc->GetDataType()),
                static_cast<int>(weightDesc->GetDataType()), static_cast<int>(weightScaleDesc->GetDataType()),
                static_cast<int>(groupListDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    // ------------------------------ 形状合法域 ------------------------------
    if (mSize < 1 || kSize < MIN_K_SIZE || nSize < MIN_N_SIZE || kSize % ALIGN_SIZE_K != 0 ||
        nSize % ALIGN_SIZE_N != 0 || groupNum < 1 || groupNum > MAX_GROUP_NUM) {
        OP_LOGE(nodeName, "[gmsq_tiling] invalid shape: M=%ld K=%ld N=%ld E=%ld.", mSize, kSize, nSize, groupNum);
        return ge::GRAPH_FAILED;
    }

    // ------------------------------ weight 形态 ------------------------------
    const uint32_t weightListSize = GetDynamicInputListSize(*context, INPUT_WEIGHT_INDEX);
    const uint32_t weightScaleListSize = GetDynamicInputListSize(*context, INPUT_WEIGHT_SCALE_INDEX);
    if (weightListSize == 0 || weightScaleListSize == 0) {
        OP_LOGE(nodeName, "[gmsq_tiling] empty weight list.");
        return ge::GRAPH_FAILED;
    }
    const bool isSingleMultiSingle = weightListSize > 1;
    if (isSingleMultiSingle && (weightListSize != weightScaleListSize ||
                                static_cast<int64_t>(weightListSize) != groupNum)) {
        OP_LOGE(nodeName, "[gmsq_tiling] tensor list size mismatch: weight=%u weight_scale=%u groupNum=%ld.",
                weightListSize, weightScaleListSize, groupNum);
        return ge::GRAPH_FAILED;
    }

    // ------------------------------ 属性 ------------------------------
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(nodeName, "[gmsq_tiling] attrs is nullptr.");
        return ge::GRAPH_FAILED;
    }
    const auto *groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_LIST_TYPE_INDEX);
    const auto *betaPtr = attrs->GetAttrPointer<float>(ATTR_BETA_INDEX);
    const auto *linearBetaPtr = attrs->GetAttrPointer<float>(ATTR_LINEAR_BETA_INDEX);
    if (groupListTypePtr == nullptr || betaPtr == nullptr || linearBetaPtr == nullptr) {
        OP_LOGE(nodeName, "[gmsq_tiling] attrs group_list_type/beta/linear_beta is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (*groupListTypePtr != 0 && *groupListTypePtr != 1) {
        OP_LOGE(nodeName, "[gmsq_tiling] group_list_type=%ld is invalid (expect 0 or 1).", *groupListTypePtr);
        return ge::GRAPH_FAILED;
    }
    if (*betaPtr <= 0.0f) {
        OP_LOGE(nodeName, "[gmsq_tiling] beta=%f must be > 0.", *betaPtr);
        return ge::GRAPH_FAILED;
    }
    if (*linearBetaPtr <= 0.0f) {
        OP_LOGE(nodeName, "[gmsq_tiling] linear_beta=%f must be > 0.", *linearBetaPtr);
        return ge::GRAPH_FAILED;
    }

    // ------------------------------ tiling 数据 ------------------------------
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = std::max<uint32_t>(1, platform.GetCoreNumAic());

    gmm_situ::SituTilingData tiling = {};
    tiling.groupListType = static_cast<uint8_t>(*groupListTypePtr);
    tiling.groupNum = static_cast<uint32_t>(groupNum);
    tiling.coreNum = coreNum;
    tiling.groupSize = MX_BLOCK_SIZE;
    tiling.kSize = static_cast<uint64_t>(kSize);
    tiling.nSize = static_cast<uint64_t>(nSize);
    tiling.mSize = static_cast<uint64_t>(mSize);
    tiling.beta = *betaPtr;
    tiling.invBeta = 1.0f / *betaPtr;
    tiling.linearBeta = *linearBetaPtr;
    tiling.invLinearBeta = (*linearBetaPtr == 0.0f) ? 0.0f : (1.0f / *linearBetaPtr);

    auto *rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr) {
        OP_LOGE(nodeName, "[gmsq_tiling] raw tiling data is nullptr.");
        return ge::GRAPH_FAILED;
    }
    if (rawTiling->GetCapacity() < sizeof(tiling)) {
        OP_LOGE(nodeName, "[gmsq_tiling] raw tiling capacity %zu < %zu.", static_cast<size_t>(rawTiling->GetCapacity()),
                sizeof(tiling));
        return ge::GRAPH_FAILED;
    }
    errno_t ret = memcpy_s(rawTiling->GetData(), rawTiling->GetCapacity(), &tiling, sizeof(tiling));
    if (ret != EOK) {
        OP_LOGE(nodeName, "[gmsq_tiling] memcpy_s failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(sizeof(tiling));

    // 单配方：arch35 MX A8W4 一条数据通路，模板维度只区分 weight 是否转置与 weight 形态。
    context->SetTilingKey(GET_TPL_TILING_KEY(GMSQ_TRANS, GMSQ_NO_TRANS, static_cast<uint64_t>(isSingleMultiSingle)));
    // 静态 grid = 全 AIC 核数：图捕获下每次 replay 的 launch 配置一致，无 block 的核在调度阶段直接退出。
    context->SetBlockDim(coreNum);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();

    OP_LOGI(nodeName, "[gmsq_tiling] OK: M=%ld K=%ld N=%ld E=%ld groupListType=%ld list=%d beta=%f lbeta=%f cores=%u",
            mSize, kSize, nSize, groupNum, *groupListTypePtr, static_cast<int>(isSingleMultiSingle), *betaPtr,
            *linearBetaPtr, coreNum);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
