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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_tiling.h"

#include "../weight_quant_batch_matmul_v2_tiling_key.h"
#include "common/op_host/math_util.h"

using namespace platform_ascendc;

namespace optiling {
namespace Mc2weight_quant_batch_matmul_v2 {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t DB_SIZE = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t ASW_PRIORITY = 9;

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingASW::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of adaptive sliding window tiling strategy.");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    AnalyseSlidingWinInfo();
    CalL1Tiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingASW::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(WeightQuantBatchMatmulV2ASWTilingData);
    if (tilingData_ == nullptr) {
        try {
            // make_unique不会返回空指针，只会返回异常，无需在后面加空指针校验
            tilingData_ = std::make_unique<WeightQuantBatchMatmulV2ASWTilingData>();
        } catch (std::bad_alloc &) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void Mc2WeightQuantBatchMatmulV2TilingASW::AnalyseSlidingWinInfo()
{
    CalcBasicBlock();
    adaptiveWin_.mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    adaptiveWin_.totalBlockCnt = adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = matmulInfoPtr_->mSize - (adaptiveWin_.mBlockCnt - 1) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = matmulInfoPtr_->nSize - (adaptiveWin_.nBlockCnt - 1) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, static_cast<uint64_t>(compileInfoPtr_->aicNum));
    adaptiveWin_.tailWinBlockCnt = (adaptiveWin_.totalBlockCnt) % compileInfoPtr_->aicNum;
    if (adaptiveWin_.useTailWinLogic) {
        CalcTailBasicBlock();
    }
}

void Mc2WeightQuantBatchMatmulV2TilingASW::CalcBasicBlock()
{
    // baseM/baseN/baseK对齐原则
    // 原则1：L1上x1、x2都是NZ排布，转置可选          须满足内轴32B对齐，外轴16对齐
    // 原则2：L0上x1是Zz排布，x2是Zn排布，非转置      须满足baseM、baseN关于16对齐，baseK关于32B对齐
    // 理论上需要同时满足原则1和原则2,但伪量化的x1Dtype可取{DT_FLOAT16, DT_BF16},
    // x2Dtype可取{DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_HIFLOAT8}
    // sizeof(x1Dtype) <= 2, sizeof(x2Dtype) <= 1
    // baseM满足原则1时，可能是关于32B对齐或关于16对齐，间接满足原则2
    // baseN满足原则1时，可能是关于32B对齐或关于16对齐，间接满足原则2
    // baseK满足32B对齐时，同时满足原则1和原则2
    // 所以下列代码中的对齐计算方式合理
    adaptiveWin_.baseM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    adaptiveWin_.baseM =
        !matmulInfoPtr_->transA ?
            ops::CeilAlign(adaptiveWin_.baseM, CUBE_BLOCK) :
            ops::CeilAlign(adaptiveWin_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype));
    adaptiveWin_.baseN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    adaptiveWin_.baseN =
        matmulInfoPtr_->transB ?
            ops::CeilAlign(adaptiveWin_.baseN, CUBE_BLOCK) :
            ops::CeilAlign(adaptiveWin_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype));

    uint64_t minBaseK =
        std::min(std::min(GetShapeWithDataType(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->aDtype),
                          GetShapeWithDataType(static_cast<uint64_t>(BASIC_BLOCK_SIZE_128), matmulInfoPtr_->bDtype)),
                 matmulInfoPtr_->kSize);
    uint64_t maxAlignSize =
        std::max(static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, matmulInfoPtr_->aDtype)),
                 static_cast<uint64_t>(GetShapeWithDataType(CUBE_REDUCE_BLOCK, matmulInfoPtr_->bDtype)));
    adaptiveWin_.baseK = ops::CeilAlign(minBaseK, maxAlignSize);
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingASW::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingASW::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    bool is4BitInput = (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4);
    if (is4BitInput) {
        // 2: 判断是否是偶数
        OP_TILING_CHECK(
            shapeSize % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "shapeSize", std::to_string(shapeSize).c_str(),
                                                  "The value of shapeSize must be even when dtype is FLOAT4/INT4."),
            return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

void Mc2WeightQuantBatchMatmulV2TilingASW::CalcTailBasicBlock()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    auto &preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto &secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;
    while (CalUsedCoreNum(preSplit + 1, secSplit) <= compileInfoPtr_->aicNum) {
        preSplit += 1UL;
        if (IsValidWeighNzTailSplit(preSplit, true)) {
            preSplitValid = preSplit;
        }

        if (CalUsedCoreNum(preSplit, secSplit + 1) <= compileInfoPtr_->aicNum) {
            secSplit += 1UL;
            if (IsValidWeighNzTailSplit(secSplit, false)) {
                secSplitValid = secSplit;
            }
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

bool Mc2WeightQuantBatchMatmulV2TilingASW::IsValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ ||
        (((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
          (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)))) {
        return true;
    }
    uint64_t tailN = adaptiveWin_.baseN / splitCnt;
    return tailN % GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype) == 0;
}

uint32_t Mc2WeightQuantBatchMatmulV2TilingASW::CalUsedCoreNum() const
{
    if (adaptiveWin_.totalWinCnt > 1 || adaptiveWin_.tailWinBlockCnt == 0) {
        return compileInfoPtr_->aicNum;
    }

    return adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile;
}

uint32_t Mc2WeightQuantBatchMatmulV2TilingASW::CalUsedCoreNum(uint32_t mTile, uint32_t nTile) const
{
    return mTile * nTile * adaptiveWin_.tailWinBlockCnt;
}

void Mc2WeightQuantBatchMatmulV2TilingASW::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(opName_, "coreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;

    basicTiling_.stepM = 1;
    basicTiling_.stepN = 1;
    basicTiling_.singleCoreM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = matmulInfoPtr_->kSize;

    uint64_t biasDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(matmulInfoPtr_->antiQuantScaleDtype);
    uint64_t totalL1Size = compileInfoPtr_->l1Size;

    basicTiling_.iterateOrder = 0;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= compileInfoPtr_->l0cSize) &&
         CheckAntiQuantScale(basicTiling_.baseN, DB_SIZE)) ?
            DB_SIZE :
            1;
    uint64_t singleCoreBiasSize = matmulInfoPtr_->hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize =
        matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_CHANNEL ? basicTiling_.baseN * scaleDtypeSize : 0;
    uint64_t leftL1Size = totalL1Size - singleCoreBiasSize - singleCoreScaleSize;
    CalL1TilingDepth(leftL1Size);
}

void Mc2WeightQuantBatchMatmulV2TilingASW::CalL1TilingDepth(uint64_t leftL1Size)
{
    // depthA1/B1恒>=2, 证明如下：
    // 已知baseM/N/K <= 128, sizeof(x1Dtype) <= 2, sizeof(x2Dtype) <=1
    // 则baseM * baseK * sizeof(x1Dtype) <= 32K, baseN * baseK * sizeof(x2Dtype) <= 16K
    // 又因为 L1 bias和scale所需最大空间分别是baseN * sizeof(biasDtype) = 0.5K, baseN * sizeof(scaleDtype) = 1K
    // 所以depth >= (L1Size - bias最大空间 - scale最大空间） / NUM_HALF / max(左基本块最大空间, 右基本块最大空间)
    //           >=  (L1Size - 1.5K - 1K) / 2 / max(32K, 16K)
    // 进一步计算，只要L1Size >= 129.5K, 则 depthA1/B1恒>=2，硬件规格显然满足
    basicTiling_.depthA1 =
        leftL1Size / NUM_HALF /
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM * basicTiling_.baseK), matmulInfoPtr_->aDtype);
    basicTiling_.depthB1 =
        leftL1Size / NUM_HALF /
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN * basicTiling_.baseK), matmulInfoPtr_->bDtype);
    CalStepKs();
}

void Mc2WeightQuantBatchMatmulV2TilingASW::CalStepKs()
{
    basicTiling_.stepKa = basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 / DB_SIZE;
    if (basicTiling_.stepKa * basicTiling_.baseK > matmulInfoPtr_->kSize) {
        basicTiling_.stepKa = ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (basicTiling_.stepKb * basicTiling_.baseK > matmulInfoPtr_->kSize) {
        basicTiling_.stepKb = ops::CeilDiv(matmulInfoPtr_->kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (basicTiling_.stepKa > basicTiling_.stepKb) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }

    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

bool Mc2WeightQuantBatchMatmulV2TilingASW::CheckAntiQuantScale(uint64_t baseN, uint64_t dbL0c) const
{
    uint64_t maxScaleBaseN = dbL0c == 1 ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_256;
    bool isScaleInvalid = baseN > maxScaleBaseN;
    return !isScaleInvalid;
}

void Mc2WeightQuantBatchMatmulV2TilingASW::SetTilingData()
{
    tilingData_->hasBias = 0U;
    tilingData_->mTailTile = adaptiveWin_.mTailTile;
    tilingData_->nTailTile = adaptiveWin_.nTailTile;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingASW::DoLibApiTiling()
{
    tilingData_->matmulTiling.M = matmulInfoPtr_->mSize;
    tilingData_->matmulTiling.N = matmulInfoPtr_->nSize;
    tilingData_->matmulTiling.Ka = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.Kb = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_->matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_->matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_->matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_->matmulTiling.baseM = basicTiling_.baseM;
    tilingData_->matmulTiling.baseN = basicTiling_.baseN;
    tilingData_->matmulTiling.baseK = basicTiling_.baseK;
    tilingData_->matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_->matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_->matmulTiling.stepM = basicTiling_.stepM;
    tilingData_->matmulTiling.stepN = basicTiling_.stepN;
    tilingData_->matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_->matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_->matmulTiling.isBias = 0;
    tilingData_->matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_->matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_->matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_->matmulTiling.dbL0C = basicTiling_.dbL0c;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingASW::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "failed to get workspace size"), return ge::GRAPH_FAILED);
    workspaces[0] = WORKSPACE_SIZE; // asc要求workspace最低需要16 * 1024 * 1024 Byte
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingASW::PostTiling()
{
    size_t tilingDataSize = sizeof(WeightQuantBatchMatmulV2ASWTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] is not aligned to 8.", tilingDataSize),
                    return ge::GRAPH_FAILED);
    errno_t ret =
        memcpy_s(context_->GetTilingData<WeightQuantBatchMatmulV2ASWTilingData>(),
                 context_->GetRawTilingData()->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingASW::GetTilingKey() const
{
    Mc2TilingKeyConfigure tilingKeyConfigure;
    // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
    tilingKeyConfigure.socVersionType = static_cast<uint8_t>(Mc2SocVersionType::SUPPORT_MMAD_S8S4) * 10;
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(Mc2QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    tilingKeyConfigure.algorithm = static_cast<uint8_t>(Mc2OptimizationAlgorithmCategory::FIXPIPE_ANTIQUANT) * 10 +
                                   static_cast<uint8_t>(algorithmSubCategory_);
    tilingKeyConfigure.transposeSituation =
        (static_cast<uint16_t>(matmulInfoPtr_->transA) << 1) | static_cast<uint16_t>(matmulInfoPtr_->transB);
    tilingKeyConfigure.antiquantType = static_cast<uint8_t>(matmulInfoPtr_->antiQuantType);
    tilingKeyConfigure.quantType = static_cast<uint8_t>(matmulInfoPtr_->quantType);

    if (matmulInfoPtr_->biasDtype == ge::DT_FLOAT && matmulInfoPtr_->hasBias) {
        // 按照tilingKey的定义，bias为float场景在原有tiling key的基础上加4做区分
        tilingKeyConfigure.optionInputSituation = (static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1) + 4;
    } else {
        tilingKeyConfigure.optionInputSituation = static_cast<uint16_t>(matmulInfoPtr_->hasAntiQuantOffset) << 1;
    }

    tilingKeyConfigure.weightFormat = static_cast<uint8_t>(Mc2WeightFormat::ND);

    tilingKeyConfigure.templateCustom = static_cast<uint8_t>(mte2Config_);
    tilingKeyConfigure.apiConstexpr = 0;
    return tilingKeyConfigure.GenTilingKey();
}
REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2TilingASW, ASW_PRIORITY);
} // namespace Mc2weight_quant_batch_matmul_v2
} // namespace optiling
