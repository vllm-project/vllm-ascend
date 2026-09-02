/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_basic_aswt_tiling.cpp
 * \brief
 */
#include "matmul_v3_basic_aswt_tiling.h"
#include "matmul_v3_tiling_strategy.h"
#include "./matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"

using Ops::NN::MathUtil;

namespace optiling {
namespace matmul_v3_advanced {
using namespace strategy;

// 注册BASIC_ASWT作为基础API实现的模板策略
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicAswtTiling, DAV_3510, BASIC_ASWT);
MM_REGISTER_TILING_TEMPLATE(MatMulV3, MatMulV3BasicAswtTiling, DAV_RESV, BASIC_ASWT); // supportMmadS8S4平台

bool MatMulV3BasicAswtTiling::IsCapable()
{
    OP_LOGI(args_.opName, "operator launched with BasicAswt module.");
    return true;
}

void MatMulV3BasicAswtTiling::ResetFullLoadLoadBalance()
{
    // 全载模板需重置负载均衡计算
    runInfo_.mBaseTailSplitCnt = 1UL;
    runInfo_.nBaseTailSplitCnt = 1UL;
    runInfo_.tailInfo.mTailMain = 0UL;
    runInfo_.tailInfo.nTailMain = 0UL;
}

bool MatMulV3BasicAswtTiling::CheckAL1FullLoad() const
{
    // 不支持Fixpipe优化
    if (l0C2Out_ != MatMulV3L0C2Out::ON_THE_FLY) {
        return false;
    }
    // 不支持CubeBound
    if (runInfo_.cubeBoundParam <= runInfo_.cubeBoundEdge) {
        OP_LOGD(args_.opName, "The shape already cubebound, no need do al1 full load.");
        return false;
    }
    // 不支持搬运量无减少
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    if (nCnt <= compileInfo_.aicNum) {
        return false;
    }
    // 不支持Fixp Bound多轮场景
    if (args_.kValue <= BASIC_BLOCK_SIZE_128 && mCnt != 1) {
        return false;
    }
    // 不超L1 Buffer
    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t aL1Size = mAlignedValue * kAlignedValue * args_.aDtypeSize;
    uint64_t biasSize = args_.hasBias ? runInfo_.baseN * DB_SIZE * GetSizeByDataType(args_.biasType) : 0;
    // 全载数据不超过3/4 L1 Buffer
    if ((aL1Size + biasSize) > compileInfo_.l1Size * 3UL / 4UL) {
        return false;
    }
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoAL1FullLoad.");
    return true;
}

bool MatMulV3BasicAswtTiling::CheckBL1FullLoad() const
{
    // 不支持CubeBound
    if (runInfo_.cubeBoundParam <= runInfo_.cubeBoundEdge) {
        OP_LOGD(args_.opName, "The shape already cubebound, no need do al1 full load.");
        return false;
    }
    // 不支持搬运量无减少
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.singleCoreM);
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.singleCoreN);
    if (mCnt <= compileInfo_.aicNum) {
        return false;
    }
    // 不超L1 Buffer
    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t bL1Size = nAlignedValue * kAlignedValue * args_.bDtypeSize;
    uint64_t biasSize = args_.hasBias ? nAlignedValue * GetSizeByDataType(args_.biasType) : 0;
    // 全载数据不超过3/4 L1 Buffer
    if ((bL1Size + biasSize) > compileInfo_.l1Size * 3UL / 4UL) {
        return false;
    }
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoBL1FullLoad.");
    return true;
}

void MatMulV3BasicAswtTiling::CalcTailBasicBlockAL1Full()
{
    uint64_t nCnt = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t tailCnt = nCnt <= compileInfo_.aicNum ? 0UL : nCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;

    if (tailCnt == 0UL) {
        return;
    }
    while ((runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum &&
           (args_.isBTrans || MathUtil::CeilDivision(runInfo_.baseN, runInfo_.tailInfo.nCnt) * args_.bDtypeSize >
                                  BASIC_BLOCK_K_128_BYTE)) {
        runInfo_.tailInfo.nCnt += 1UL;
    }
}

void MatMulV3BasicAswtTiling::CalcTailBasicBlockBL1Full()
{
    uint64_t mCnt = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t tailCnt = mCnt <= compileInfo_.aicNum ? 0UL : mCnt % compileInfo_.aicNum;
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    if (tailCnt == 0UL) {
        return;
    }
    while ((runInfo_.tailInfo.mCnt + 1UL) * tailCnt <= compileInfo_.aicNum &&
           (!args_.isATrans || MathUtil::CeilDivision(runInfo_.baseM, runInfo_.tailInfo.mCnt) * args_.aDtypeSize >
                                   BASIC_BLOCK_K_128_BYTE)) {
        runInfo_.tailInfo.mCnt += 1UL;
    }
}

void MatMulV3BasicAswtTiling::DoAL1FullLoad()
{
    ResetFullLoadLoadBalance();
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoAL1FullLoad.");

    uint64_t mAlignedValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    runInfo_.stepM = MathUtil::CeilDivision(mAlignedValue, runInfo_.baseM);
    runInfo_.stepKa = MathUtil::CeilDivision(kAlignedValue, runInfo_.baseK);
    uint64_t aL1Size = mAlignedValue * kAlignedValue * args_.aDtypeSize;
    uint64_t biasSize = args_.hasBias ? runInfo_.baseN * DB_SIZE * GetSizeByDataType(args_.biasType) : 0;
    uint64_t remainL1Size = compileInfo_.l1Size - (aL1Size + biasSize);
    uint64_t maxBaseNWithL1 = remainL1Size / (runInfo_.baseK * args_.bDtypeSize * DB_SIZE);
    uint64_t maxBaseNWithL0cDb = compileInfo_.l0CSize / (runInfo_.baseM * DATA_SIZE_FP32 * DB_SIZE);
    uint64_t maxBaseN = ops::FloorAlign(std::min(maxBaseNWithL1, maxBaseNWithL0cDb), BASIC_BLOCK_SIZE_16);
    uint64_t balanceBaseN = args_.batchInfo != nullptr ?
                                maxBaseN :
                                ops::CeilAlign(MathUtil::CeilDivision(args_.nValue, compileInfo_.aicNum),
                                               BASIC_BLOCK_SIZE_16);
    // 在已有baseN，最大baseN，负载均衡baseN中选择最小值
    runInfo_.baseN = std::min(runInfo_.baseN, std::min(maxBaseN, balanceBaseN));
    // N内轴时满足128B对齐
    uint64_t baseAlignUnit = BASIC_BLOCK_K_128_BYTE / args_.bDtypeSize;
    if (!args_.isBTrans && runInfo_.baseN > baseAlignUnit) {
        runInfo_.baseN = ops::FloorAlign(runInfo_.baseN, baseAlignUnit);
    }

    uint64_t maxStepKWithBufferLimit = remainL1Size / (runInfo_.baseK * runInfo_.baseN * DB_SIZE * args_.bDtypeSize);
    // stepK最大不超过4
    uint64_t maxStepK = std::min(
        std::min(MathUtil::CeilDivision(args_.kValue, runInfo_.baseK), maxStepKWithBufferLimit), 4UL);
    // B矩阵K为内轴，k_bl1不满足256B对齐，尝试baseK放大一倍， 提升B矩阵搬运效率
    if (args_.isBTrans && maxStepK == maxStepKWithBufferLimit &&
        maxStepK * runInfo_.baseK % BASIC_BLOCK_K_256_BYTE != 0 &&
        (runInfo_.baseK << 1) * runInfo_.baseM * args_.aDtypeSize * DB_SIZE <= compileInfo_.l0ASize) {
        runInfo_.baseK <<= 1;
        maxBaseNWithL1 = remainL1Size / (runInfo_.baseK * args_.bDtypeSize * DB_SIZE);
        uint64_t maxBaseNWithL0B = compileInfo_.l0BSize / (runInfo_.baseK * args_.bDtypeSize * DB_SIZE);
        runInfo_.baseN = std::min(runInfo_.baseN, maxBaseNWithL0B);
        runInfo_.stepKa = MathUtil::CeilDivision(kAlignedValue, runInfo_.baseK);
    }
    uint64_t stepK = 1;
    for (; stepK < maxStepK; stepK++) {
        uint64_t curKL1 = runInfo_.baseK * stepK;
        uint64_t usedL1Size = runInfo_.baseN * curKL1 * args_.bDtypeSize;
        // L1搬运量约束
        if (usedL1Size < L1_SINGLE_SIZE_LIMIT) {
            continue;
        }
        // K内轴时约束kL1 256B对齐，发挥带宽能力
        if (args_.isBTrans && curKL1 % (BASIC_BLOCK_K_256_BYTE / args_.bDtypeSize) != 0) {
            continue;
        }
        break;
    }
    runInfo_.stepKb = stepK;
    runInfo_.depthA1 = runInfo_.stepM * runInfo_.stepKa;
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
    runInfo_.singleCoreM = args_.mValue;
    runInfo_.singleCoreN = runInfo_.baseN;
    uint64_t bL14BufferSize = runInfo_.baseK * runInfo_.stepKb * runInfo_.baseN * args_.bDtypeSize *
                              BASIC_L1_BUFFER_NUM;
    uint64_t bias4BufferSize = biasSize / DB_SIZE * BASIC_L1_BUFFER_NUM;
    runInfo_.l1BufferNum = bL14BufferSize + aL1Size + bias4BufferSize > compileInfo_.l1Size ? DB_SIZE :
                                                                                              BASIC_L1_BUFFER_NUM;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    uint64_t nCore = MathUtil::CeilDivision(args_.nValue, runInfo_.baseN);
    uint64_t batchNum = args_.batchInfo == nullptr ? 1 : args_.batchInfo->batchC;
    runInfo_.usedCoreNum = std::min(nCore * batchNum, compileInfo_.aicNum);
    CalcTailBasicBlockAL1Full();
    fullLoad_ = MatMulV3FullLoad::A_FULL_LOAD;
    return;
}

void MatMulV3BasicAswtTiling::DoBL1FullLoad()
{
    // 负载均衡屏蔽全载模板
    ResetFullLoadLoadBalance();
    OP_LOGI(args_.opName, "MatMulV3 tiling enable state is DoBL1FullLoad.");

    uint64_t nAlignedValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t kAlignedValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    runInfo_.stepN = MathUtil::CeilDivision(nAlignedValue, runInfo_.baseN);
    runInfo_.stepKb = MathUtil::CeilDivision(kAlignedValue, runInfo_.baseK);
    uint64_t bL1Size = nAlignedValue * kAlignedValue * args_.bDtypeSize;
    uint64_t biasSize = args_.hasBias ? nAlignedValue * GetSizeByDataType(args_.biasType) : 0;
    uint64_t remainL1Size = compileInfo_.l1Size - (bL1Size + biasSize);
    uint64_t maxBaseMWithL1 = remainL1Size / (runInfo_.baseK * args_.aDtypeSize * DB_SIZE);
    uint64_t maxBaseMWithL0cDb = compileInfo_.l0CSize / (runInfo_.baseN * DATA_SIZE_FP32 * DB_SIZE);
    uint64_t maxBaseM = ops::FloorAlign(std::min(maxBaseMWithL1, maxBaseMWithL0cDb), BASIC_BLOCK_SIZE_16);
    uint64_t balanceBaseM = args_.batchInfo != nullptr ?
                                maxBaseM :
                                ops::CeilAlign(MathUtil::CeilDivision(args_.mValue, compileInfo_.aicNum),
                                               BASIC_BLOCK_SIZE_16);
    runInfo_.baseM = std::min(runInfo_.baseM, std::min(maxBaseM, balanceBaseM));
    uint64_t baseAlignUnit = BASIC_BLOCK_K_128_BYTE / args_.bDtypeSize;
    if (args_.isATrans && runInfo_.baseM > baseAlignUnit) {
        runInfo_.baseM = ops::FloorAlign(runInfo_.baseM, baseAlignUnit);
    }

    uint64_t maxStepKWithBufferLimit = remainL1Size / (runInfo_.baseK * runInfo_.baseM * DB_SIZE * args_.aDtypeSize);
    // stepK最大不超过4
    uint64_t maxStepK = std::min(
        std::min(MathUtil::CeilDivision(args_.kValue, runInfo_.baseK), maxStepKWithBufferLimit), 4UL);
    // A矩阵K为内轴，k_al1不满足256B对齐，尝试baseK放大一倍
    if (!args_.isATrans && maxStepK == maxStepKWithBufferLimit &&
        maxStepK * runInfo_.baseK % BASIC_BLOCK_K_256_BYTE != 0 &&
        (runInfo_.baseK << 1) * runInfo_.baseN * args_.bDtypeSize * DB_SIZE <= compileInfo_.l0BSize) {
        runInfo_.baseK <<= 1;
        maxBaseMWithL1 = remainL1Size / (runInfo_.baseK * args_.aDtypeSize * DB_SIZE);
        uint64_t maxBaseMWithL0A = compileInfo_.l0ASize / (runInfo_.baseK * args_.aDtypeSize * DB_SIZE);
        runInfo_.baseM = std::min(runInfo_.baseM, maxBaseMWithL0A);
        runInfo_.stepKb = MathUtil::CeilDivision(kAlignedValue, runInfo_.baseK);
    }
    uint64_t stepK = 1;
    for (; stepK < maxStepK; stepK++) {
        uint64_t curKL1 = runInfo_.baseK * stepK;
        uint64_t usedL1Size = runInfo_.baseM * curKL1 * args_.aDtypeSize;
        if (usedL1Size < L1_SINGLE_SIZE_LIMIT) {
            continue;
        }
        if (!args_.isATrans && curKL1 % (BASIC_BLOCK_K_256_BYTE / args_.aDtypeSize) != 0) {
            continue;
        }
        break;
    }
    runInfo_.stepKa = stepK;
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
    runInfo_.depthB1 = runInfo_.stepN * runInfo_.stepKb;
    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = args_.nValue;
    uint64_t aL14BufferSize = runInfo_.baseK * runInfo_.stepKa * runInfo_.baseM * args_.aDtypeSize *
                              BASIC_L1_BUFFER_NUM;
    uint64_t bias4BufferSize = biasSize / DB_SIZE * BASIC_L1_BUFFER_NUM;
    runInfo_.l1BufferNum = aL14BufferSize + bL1Size + bias4BufferSize > compileInfo_.l1Size ? DB_SIZE :
                                                                                              BASIC_L1_BUFFER_NUM;
    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 * DB_SIZE <= compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.mixInfo.ubDB = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 <= compileInfo_.ubSize ? DB_SIZE : 1UL;
    uint64_t mCore = MathUtil::CeilDivision(args_.mValue, runInfo_.baseM);
    uint64_t batchNum = args_.batchInfo == nullptr ? 1 : args_.batchInfo->batchC;
    runInfo_.usedCoreNum = std::min(mCore * batchNum, compileInfo_.aicNum);
    CalcTailBasicBlockBL1Full();
    fullLoad_ = MatMulV3FullLoad::B_FULL_LOAD;
    return;
}

void MatMulV3BasicAswtTiling::CheckFp32SplitK()
{
    // FP32切K判断
    bool isFp32 = (args_.aType == ge::DT_FLOAT && args_.bType == ge::DT_FLOAT);
    bool isNdFormat = (args_.aFormat == ge::FORMAT_ND && args_.bFormat == ge::FORMAT_ND);
    // 连续且非全载场景才支持切K
    if (!isSlice_ && isFp32 && !args_.isHf32 && isNdFormat && (args_.kValue >= FP32_SPLIT_K_THRESHOLD) &&
        fullLoad_ == MatMulV3FullLoad::NONE_FULL_LOAD) {
        model_ = MatMulV3Model::BASIC_SPLIT_K;
    }
}

void MatMulV3BasicAswtTiling::CheckApiLevelAndModel()
{
    bool isMatmul = strcmp(context_->GetNodeType(), "MatMulV3") == 0;
    apiLevel_ = (isMatmul && !args_.isAvoidTensorApi) ? MatMulV3ApiLevel::TENSOR_LEVEL : MatMulV3ApiLevel::BASIC_LEVEL;
    bool isTensorApiUnsupportedSlice = isSlice_ && (fullLoad_ != MatMulV3FullLoad::NONE_FULL_LOAD ||
                                                    l0C2Out_ != MatMulV3L0C2Out::ON_THE_FLY);
    // Tensor API仅支持非全载、OnTheFly的非连续Slice场景，其余场景回退基础API
    if (isTensorApiUnsupportedSlice) {
        apiLevel_ = MatMulV3ApiLevel::BASIC_LEVEL;
        model_ = MatMulV3Model::BASIC;
        OP_LOGD(args_.opName, "Non-contiguous slice with full load or Fixpipe falls back to Basic API.");
    } else if (isSlice_) {
        model_ = MatMulV3Model::SLICE;
    }

    // DAV_RESV当前只支持基础API
    if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
        apiLevel_ = MatMulV3ApiLevel::BASIC_LEVEL;
        model_ = MatMulV3Model::BASIC;
    }
}

ge::graphStatus MatMulV3BasicAswtTiling::DoOpTiling()
{
    MatMulV3AswTiling::DoOpTiling();
    // Slice标记需要在全载和Fixpipe分支选择前统一初始化，避免分支遗漏拦截
    isSlice_ = MatMulV3TilingHelper::IsSelfNonContiguous(context_);
    l0C2Out_ = MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_);
    if (CheckAL1FullLoad()) {
        DoAL1FullLoad();
        CheckFp32SplitK();
        CheckApiLevelAndModel();
    } else if (CheckBL1FullLoad()) {
        DoBL1FullLoad();
        CheckApiLevelAndModel();
    } else if (l0C2Out_ == MatMulV3L0C2Out::ON_THE_FLY) {
        // 非fixpipe优化场景
        uint64_t remainSizeForAL1BL1 = args_.hasBias ? (compileInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) :
                                                       compileInfo_.l1Size;
        runInfo_.stepKa = remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) /
                          args_.aDtypeSize;
        runInfo_.stepKb = runInfo_.stepKa; // has bias, adjust stepK to suitable value
        runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
        runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
        CheckFp32SplitK();
        CheckApiLevelAndModel();
    } else {
        // fixpipe优化场景
        CheckApiLevelAndModel();
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t MatMulV3BasicAswtTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetFullLoad(fullLoad_)
        .SetModel(model_)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(apiLevel_)
        .GetTilingKey();
}

ge::graphStatus MatMulV3BasicAswtTiling::GetTilingData(TilingResult& tiling) const
{
    return GetTilingDataImpl<MatMulV3BasicTilingData>(tiling);
}
} // namespace matmul_v3_advanced
} // namespace optiling
