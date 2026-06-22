/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */
/*!
 * \file fused_gdn_gating_tiling.cpp
 * \brief Tiling implementation for FusedGdnGating.
 */
#include "fused_gdn_gating_tiling.h"
#include "fused_gdn_gating_tiling_utils.h"
#include "register/op_impl_registry.h"
#include "securec.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <limits>
#include <algorithm>
#include <cmath>

using namespace optiling;

namespace {

// --------------------------------------------------------
// 910/910B 架构专用常量
// --------------------------------------------------------
constexpr uint64_t TILING_KEY_BF16 = 1;
constexpr uint64_t TILING_KEY_FP16 = 2;
constexpr uint64_t TILING_KEY_PARAM_BF16_OFFSET = 2;
constexpr uint64_t TILING_KEY_PARAM_FP16_OFFSET = 4;
constexpr size_t INPUT_INDEX_A_LOG = 0;
constexpr size_t INPUT_INDEX_A = 1;
constexpr size_t INPUT_INDEX_DT_BIAS = 3;

// --------------------------------------------------------
// 310P 架构专用常量
// --------------------------------------------------------
constexpr uint32_t HW_ALIGN_BYTES = 16;
constexpr uint32_t DEFAULT_CORE_NUM = 1;
constexpr uint32_t TARGET_CORE_NUM_SMALL = 1;
constexpr uint32_t TARGET_CORE_NUM_MID = 4;
constexpr uint32_t ELEMENT_THRESHOLD_SMALL = 2048;
constexpr uint32_t ELEMENT_THRESHOLD_MID = 16384;
constexpr uint32_t TARGET_TILE_ELEMENTS = 3072;
constexpr uint64_t TILING_KEY_310P_DEFAULT = 200000;

// 辅助函数：求最大公约数，用于 310P 的数据对齐计算
uint32_t gcd(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t t = b;
        b = a % b;
        a = t;
    }
    return (a == 0) ? 1 : a;
}

} // namespace

ge::graphStatus optiling::FusedGdnGatingTilingFunc(gert::TilingContext *context)
{
    // 1. 获取基础上下文与平台信息
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    
    // 2. 核心修复：安全判断当前是否为 310P 芯片及其子型号
    bool isAscend310P = false;

    auto socVersion = ascendcPlatform.GetSocVersion();
    isAscend310P = (socVersion == platform_ascendc::SocVersion::ASCEND310P);

    // 3. 解析输入 Shape 信息
    auto *shapeA = context->GetInputShape(INPUT_INDEX_A);
    if (shapeA == nullptr || shapeA->GetStorageShape().GetDimNum() < 2) {
        return ge::GRAPH_FAILED;
    }
    const auto &storageShape = shapeA->GetStorageShape();
    int64_t numBatches = storageShape.GetDim(0);
    int64_t numHeads   = storageShape.GetDim(1);
    if (numBatches <= 0 || numHeads <= 0) {
        return ge::GRAPH_FAILED;
    }

    // 4. 解析公共属性参数
    float beta = 1.0f;
    float threshold = 20.0f;
    auto *attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float *betaAttr = attrs->GetAttrPointer<float>(0);
        if (betaAttr != nullptr) { beta = *betaAttr; }
        const float *thresholdAttr = attrs->GetAttrPointer<float>(1);
        if (thresholdAttr != nullptr) { threshold = *thresholdAttr; }
    }

    // 5. 初始化统一的 Tiling 数据结构
    FusedGdnGatingTilingData td{};
    td.set_numHeads(static_cast<uint32_t>(numHeads));
    td.set_beta(beta);
    td.set_threshold(threshold);

    uint32_t blockDim = 1;
    uint64_t tilingKey = 0;

    // --------------------------------------------------------
    // 分支 A：Ascend 310P 芯片 Tiling 逻辑 (1D 扁平切分)
    // --------------------------------------------------------
    if (isAscend310P) {
        uint32_t totalElements = static_cast<uint32_t>(numBatches * numHeads);
        uint32_t hwCoreNum = ascendcPlatform.GetCoreNum();
        if (hwCoreNum == 0) {
            hwCoreNum = DEFAULT_CORE_NUM;
        }

        uint32_t baseBlock = (HW_ALIGN_BYTES * numHeads) / gcd(HW_ALIGN_BYTES, numHeads);
        uint32_t targetCoreNum = hwCoreNum;
        
        // 依据数据量动态决定使用核心数
        if (totalElements <= ELEMENT_THRESHOLD_SMALL) {
            targetCoreNum = TARGET_CORE_NUM_SMALL;
        } else if (totalElements <= ELEMENT_THRESHOLD_MID) {
            targetCoreNum = TARGET_CORE_NUM_MID;
        }
        if (totalElements <= baseBlock) {
            targetCoreNum = TARGET_CORE_NUM_SMALL;
        }

        // 计算单核处理的数据量及其尾部数据
        uint32_t elementsPerCore = (totalElements + targetCoreNum - 1) / targetCoreNum;
        elementsPerCore = ((elementsPerCore + baseBlock - 1) / baseBlock) * baseBlock;
        if (elementsPerCore == 0) {
            elementsPerCore = baseBlock;
        }

        uint32_t usedCores = (totalElements + elementsPerCore - 1) / elementsPerCore;
        if (usedCores > targetCoreNum) {
            usedCores = targetCoreNum;
        }

        uint32_t tailElements = totalElements - (usedCores - 1) * elementsPerCore;
        uint32_t tileElements = TARGET_TILE_ELEMENTS;
        tileElements = (tileElements / baseBlock) * baseBlock;
        if (tileElements == 0) {
            tileElements = baseBlock;
        }
        if (tileElements > elementsPerCore) {
            tileElements = elementsPerCore;
        }

        // 计算 inv_beta 防除零，并修复浮点数比对安全性
        float inv_beta = (std::abs(beta) < 1e-6f) ? std::numeric_limits<float>::infinity() : (1.0f / beta);

        // 填充 310P 专属字段
        td.set_usedCoreNum(usedCores);
        td.set_alignedLength(elementsPerCore);
        td.set_tailLength(tailElements);
        td.set_tileRows(tileElements);
        td.set_inv_beta(inv_beta);

        blockDim = usedCores;
        tilingKey = TILING_KEY_310P_DEFAULT; // 对应 Kernel 层的 200000 路由
    } 
    // --------------------------------------------------------
    // 分支 B：Ascend 910/910B 芯片 Tiling 逻辑 (2D Chunk 切分)
    // --------------------------------------------------------
    else {
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
        if (aivNum == 0) {
            aivNum = 1;
        }

        // 数据类型解析与 Tiling Key 生成
        auto *aDesc = context->GetInputDesc(INPUT_INDEX_A);
        auto *aLogDesc = context->GetInputDesc(INPUT_INDEX_A_LOG);
        auto *dtBiasDesc = context->GetInputDesc(INPUT_INDEX_DT_BIAS);
        if (aDesc == nullptr || aLogDesc == nullptr || dtBiasDesc == nullptr) {
            return ge::GRAPH_FAILED;
        }
        
        ge::DataType aDtype = aDesc->GetDataType();
        ge::DataType aLogDtype = aLogDesc->GetDataType();
        ge::DataType dtBiasDtype = dtBiasDesc->GetDataType();
        if (aLogDtype != dtBiasDtype) {
            return ge::GRAPH_FAILED;
        }

        tilingKey = (aDtype == ge::DT_FLOAT16) ? TILING_KEY_FP16 : TILING_KEY_BF16;
        if (aLogDtype == ge::DT_BF16) {
            tilingKey += TILING_KEY_PARAM_BF16_OFFSET;
        } else if (aLogDtype == ge::DT_FLOAT16) {
            tilingKey += TILING_KEY_PARAM_FP16_OFFSET;
        }

        uint32_t numBatchesU32 = static_cast<uint32_t>(numBatches);
        uint32_t numHeadsU32 = static_cast<uint32_t>(numHeads);
        
        blockDim = std::min(numBatchesU32, aivNum);

        // 计算安全的流水线步长 (rowsPerIter)
        uint32_t rowsConservative = FusedGdnGating::ComputeRowsPerIter(numHeadsU32, ubSize);
        uint32_t rowsPerIter = rowsConservative;

        uint32_t totalChunksForRPI = (numBatchesU32 + rowsPerIter - 1) / rowsPerIter;
        if (numBatchesU32 <= rowsPerIter || totalChunksForRPI < blockDim) {
            uint32_t maxRPI = std::max(1u, numBatchesU32 / blockDim);
            if      (maxRPI >= 128) { rowsPerIter = 128; }
            else if (maxRPI >= 64)  { rowsPerIter = 64;  }
            else if (maxRPI >= 32)  { rowsPerIter = 32;  }
            else if (maxRPI >= 16)  { rowsPerIter = 16;  }
            else if (maxRPI >= 8)   { rowsPerIter = 8;   }
            else if (maxRPI >= 4)   { rowsPerIter = 4;   }
            else if (maxRPI >= 2)   { rowsPerIter = 2;   }
            else                    { rowsPerIter = 1;   }
            if (rowsPerIter > rowsConservative) { rowsPerIter = rowsConservative; }
        }

        const bool bulkDmaBatchOk = (numBatchesU32 > blockDim * rowsPerIter);
        bool useBulkDma = bulkDmaBatchOk && FusedGdnGating::CanUseBulkDma(numHeadsU32, rowsPerIter);

        // 填充 910/910B 专属字段
        td.set_numBatches(numBatchesU32);
        td.set_rowsPerIter(rowsPerIter);
        td.set_useBulkDma(useBulkDma ? 1u : 0u);
    }

    // 6. 将 Tiling 结构体序列化并下发给 Device 端
    const size_t tilingSize = td.GetDataSize();
    auto *rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetCapacity() < tilingSize) {
        return ge::GRAPH_FAILED;
    }
    
    td.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingSize);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);

    // 清空 Workspace 请求
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

// 7. 编译期解析配置入口
ge::graphStatus optiling::TilingPrepareForFusedGdnGating(gert::TilingParseContext *context)
{
    // 获取底层系统平台信息
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 将编译期的硬件能力写进 CompileInfo
    auto compileInfo = context->GetCompiledInfo<FusedGdnGatingCompileInfo>();
    if (compileInfo != nullptr) {
        compileInfo->coreNum = ascendcPlatform.GetCoreNum();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        compileInfo->ubSizePlatForm = ubSize;
    }
    
    return ge::GRAPH_SUCCESS;
}

// 8. 绑定 Tiling 函数到算子实体
IMPL_OP_OPTILING(FusedGdnGating)
    .Tiling(optiling::FusedGdnGatingTilingFunc)
    .TilingParse<optiling::FusedGdnGatingCompileInfo>(optiling::TilingPrepareForFusedGdnGating);
