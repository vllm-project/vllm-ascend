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
 * \file batch_mat_mul_v3_base_tiling.cc
 * \brief
 */
#include "batch_mat_mul_v3_base_tiling.h"
#include "batch_mat_mul_v3/op_kernel/batch_mat_mul_v3_tiling_key.h"
#include "util/math_util.h"
#include "mc2_log_compat.h"
#include "op_host/tiling_key.h"
#include "mc2_log.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "runtime_kb_api.h"
#include "common/op_host/math_util.h"
#include "common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"

using namespace optiling::Mc2batch_mat_mul_v3;
using Ops::Transformer::MathUtil;
using Ops::Transformer::OpTiling::GET_TILINGKEY;

namespace optiling {
namespace Mc2batch_mat_mul_v3 {
const std::vector<uint64_t> SUPPORT_ND2NZ_GM2L0_WITHOUT32B = {64, 96, 128, 160, 192, 224, 256, 384};
constexpr uint64_t BLOCK_CUBE = 16;
constexpr uint64_t NO_BATCH_SHAPE_DIM = 2;
constexpr uint64_t ONE_BATCH_SHAPE_DIM = 3;
constexpr uint64_t TWO_BATCH_SHAPE_DIM = 4;
constexpr uint64_t THREE_BATCH_SHAPE_DIM = 5;
constexpr uint64_t FOUR_BATCH_SHAPE_DIM = 6;
constexpr uint64_t ND_NZ_DIM_DIFF = 2;
constexpr uint64_t ALIGNMENT_32 = 32;
constexpr uint64_t DEFAULT_SIZE = 32;
constexpr uint64_t ND2NZ_ON_THE_FLY_LIMIT = 65535;
constexpr uint64_t LAST_DIM = 1;
constexpr uint64_t LAST_SECOND_DIM = 2;
constexpr uint64_t NUM_TWO = 2;
constexpr uint64_t NUM_ONE = 1;
constexpr uint64_t N_ALIGNED = 16;
constexpr uint64_t BLOCK_BYTE_SIZE = 32;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t RPC_WORKSIZE = 20;
constexpr uint64_t MB_SIZE = 1024UL * 1024UL;
constexpr uint64_t KB_SIZE = 1024;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t DOUBLE_BUFFER = 2UL;
constexpr uint64_t L2_SIZE_2 = 192UL * 1024UL * 1024UL;
constexpr uint64_t MAX_TRANS_CONFLICT = 6;
constexpr double TAIL_CONFLICT_RATIO = 0.5;
constexpr uint64_t MAX_INT32_VALUE = 2147483647uL;

static inline uint64_t LastPower2(uint64_t n)
{
    n |= n >> 1;  // 前2位为1
    n |= n >> 2;  // 前4位为1
    n |= n >> 4;  // 前8位为1
    n |= n >> 8;  // 前16位为1
    n |= n >> 16; // 前32位为1
    n |= n >> 32; // 前64位为1
    return (n & ~(n >> 1));
}

static inline uint64_t NextPower2(uint64_t n)
{
    uint64_t lastPower2 = LastPower2(n);
    return (n == lastPower2) ? lastPower2 : (lastPower2 << 1);
}

ge::graphStatus Mc2BatchMatmulV3BaseTiling::GetShapeAttrsInfo() // 检查输入属性是否支持
{
    auto ret = Mc2MatmulV3BaseTiling::GetShapeAttrsInfo();
    if (!GetBatchInfo()) {
        return ge::GRAPH_FAILED;
    }
    MergeBatchAndMAxis();
    return ret;
}

bool Mc2BatchMatmulV3BaseTiling::GetBatchInfo()
{
    auto aShape = context_->GetInputShape(0)->GetOriginShape();
    auto bShape = context_->GetInputShape(1)->GetOriginShape();
    auto cShape = context_->GetOutputShape(0)->GetOriginShape();

    size_t aDims = aShape.GetDimNum();
    size_t bDims = bShape.GetDimNum();
    size_t cDims = cShape.GetDimNum();
    if (args_.aFormat == ge::FORMAT_FRACTAL_NZ) {
        aDims = context_->GetInputShape(0)->GetStorageShape().GetDimNum() - ND_NZ_DIM_DIFF;
    }
    if (args_.bFormat == ge::FORMAT_FRACTAL_NZ) {
        bDims = context_->GetInputShape(1)->GetStorageShape().GetDimNum() - ND_NZ_DIM_DIFF;
    }
    if (args_.outFormat == ge::FORMAT_FRACTAL_NZ) {
        cDims = context_->GetOutputShape(0)->GetStorageShape().GetDimNum() - ND_NZ_DIM_DIFF;
    }
    batchInfo_.batchA3 = aDims > NO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - ONE_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchA2 = aDims > ONE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - TWO_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchA1 = aDims > TWO_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - THREE_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchA0 = aDims > THREE_BATCH_SHAPE_DIM ? aShape.GetDim(aDims - FOUR_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchB3 = bDims > NO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - ONE_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchB2 = bDims > ONE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - TWO_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchB1 = bDims > TWO_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - THREE_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchB0 = bDims > THREE_BATCH_SHAPE_DIM ? bShape.GetDim(bDims - FOUR_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchC3 = cDims > NO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - ONE_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchC2 = cDims > ONE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - TWO_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchC1 = cDims > TWO_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - THREE_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchC0 = cDims > THREE_BATCH_SHAPE_DIM ? cShape.GetDim(cDims - FOUR_BATCH_SHAPE_DIM) : 1;
    batchInfo_.batchA = batchInfo_.batchA0 * batchInfo_.batchA1 * batchInfo_.batchA2 * batchInfo_.batchA3;
    batchInfo_.batchB = batchInfo_.batchB0 * batchInfo_.batchB1 * batchInfo_.batchB2 * batchInfo_.batchB3;
    batchInfo_.batchC = batchInfo_.batchC0 * batchInfo_.batchC1 * batchInfo_.batchC2 * batchInfo_.batchC3;
    // Check if batch info is valid, if batch is M broadcast to N, return failed.
    bool batch3Invalid =
        batchInfo_.batchA3 != batchInfo_.batchB3 && batchInfo_.batchA3 != 1UL && batchInfo_.batchB3 != 1UL;
    bool batch2Invalid =
        batchInfo_.batchA2 != batchInfo_.batchB2 && batchInfo_.batchA2 != 1UL && batchInfo_.batchB2 != 1UL;
    bool batch1Invalid =
        batchInfo_.batchA1 != batchInfo_.batchB1 && batchInfo_.batchA1 != 1UL && batchInfo_.batchB1 != 1UL;
    bool batch0Invalid =
        batchInfo_.batchA0 != batchInfo_.batchB0 && batchInfo_.batchA0 != 1UL && batchInfo_.batchB0 != 1UL;
    if (batch3Invalid || batch2Invalid || batch1Invalid || batch0Invalid) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "[Mc2BatchMatMulV3]", "batch dims",
            (std::to_string(batchInfo_.batchA3) + ", " + std::to_string(batchInfo_.batchA2) + ", " +
             std::to_string(batchInfo_.batchA1) + " and " + std::to_string(batchInfo_.batchA0))
                .c_str(),
            "The batch dims must not use M broadcast to N mode.");
        return false;
    }
    return GetBiasWithBatchInfo();
}

bool Mc2BatchMatmulV3BaseTiling::GetBiasWithBatchInfo()
{
    batchInfo_.biasWithBatch = false;
    if (!args_.hasBias) {
        return true;
    }
    auto biasShape = context_->GetInputShape(2)->GetOriginShape();
    size_t biasDims = biasShape.GetDimNum();
    if (biasDims < NO_BATCH_SHAPE_DIM) {
        return true;
    }
    if (biasDims == NO_BATCH_SHAPE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("[Mc2BatchMatMulV3]", "bias", std::to_string(biasDims).c_str(),
                                                 "The shape dim of bias cannot be 2.");
        return false;
    }

    batchInfo_.biasWithBatch = true;
    uint64_t biasMValue = biasShape[biasDims - NO_BATCH_SHAPE_DIM];
    if (biasMValue != 1UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("[Mc2BatchMatMulV3]", "bias M", std::to_string(biasMValue).c_str(),
                                              "The value of bias M must be 1.");
        return false;
    }

    uint64_t batchBias3 =
        biasDims > NO_BATCH_SHAPE_DIM ? static_cast<uint64_t>(biasShape.GetDim(biasDims - ONE_BATCH_SHAPE_DIM)) : 1;
    uint64_t batchBias2 =
        biasDims > ONE_BATCH_SHAPE_DIM ? static_cast<uint64_t>(biasShape.GetDim(biasDims - TWO_BATCH_SHAPE_DIM)) : 1;
    uint64_t batchBias1 =
        biasDims > TWO_BATCH_SHAPE_DIM ? static_cast<uint64_t>(biasShape.GetDim(biasDims - THREE_BATCH_SHAPE_DIM)) : 1;
    uint64_t batchBias0 =
        biasDims > THREE_BATCH_SHAPE_DIM ? static_cast<uint64_t>(biasShape.GetDim(biasDims - FOUR_BATCH_SHAPE_DIM)) : 1;
    bool biasBatchValid = batchBias3 == batchInfo_.batchC3 && batchBias2 == batchInfo_.batchC2 &&
                          batchBias1 == batchInfo_.batchC1 && batchBias0 == batchInfo_.batchC0;
    if (batchInfo_.biasWithBatch && !biasBatchValid) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("[Mc2BatchMatMulV3]", "bias batch", std::to_string(batchBias3).c_str(),
                                              "The value of bias batch must be equal to batch of C.");
        return false;
    }
    return true;
}

void Mc2BatchMatmulV3BaseTiling::MergeBatchAndMAxis()
{
    if (!compileInfo_.supportL0c2out || batchInfo_.biasWithBatch) {
        return;
    }
    if (batchInfo_.batchB == 1 && !args_.isATrans) {
        // when BatchB == 1, adjust M = batchA * M, batchA = 1
        if (batchInfo_.batchA * args_.mValue > MAX_INT32_VALUE) {
            OP_LOGI("Mc2BatchMatMulV3", "m value will exceed int32 max value after merge axis, stop merging !");
            return;
        }
        args_.mValue = batchInfo_.batchA * args_.mValue;
        batchInfo_.batchA3 = 1UL;
        batchInfo_.batchA2 = 1UL;
        batchInfo_.batchA1 = 1UL;
        batchInfo_.batchA0 = 1UL;
        batchInfo_.batchA = 1UL;
        batchInfo_.batchC3 = 1UL;
        batchInfo_.batchC2 = 1UL;
        batchInfo_.batchC1 = 1UL;
        batchInfo_.batchC0 = 1UL;
        batchInfo_.batchC = 1UL;
    }
    return;
}

bool Mc2BatchMatmulV3BaseTiling::CheckBMMTilingDataIsVaild() const
{
    return (optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchA3, args_.opName, "batchInfo_.batchA3") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchA2, args_.opName, "batchInfo_.batchA2") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchA1, args_.opName, "batchInfo_.batchA1") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchA0, args_.opName, "batchInfo_.batchA0") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchB3, args_.opName, "batchInfo_.batchB3") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchB2, args_.opName, "batchInfo_.batchB2") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchB1, args_.opName, "batchInfo_.batchB1") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchB0, args_.opName, "batchInfo_.batchB0") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchC3, args_.opName, "batchInfo_.batchC3") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchC2, args_.opName, "batchInfo_.batchC2") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchC1, args_.opName, "batchInfo_.batchC1") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(batchInfo_.batchC0, args_.opName, "batchInfo_.batchC0") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(aBatchDimAll_, args_.opName, "batchInfo_.aBatchDimAll") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(bBatchDimAll_, args_.opName, "batchInfo_.bBatchDimAll") ||
            optiling::mc2_matmul_v3::CheckNumberIsValid(cBatchDimAll_, args_.opName, "batchInfo_.cBatchDimAll"));
}

void Mc2BatchMatmulV3BaseTiling::CheckandSetDiagonalConflict(uint64_t mCnt, uint64_t nCnt, uint64_t batch,
                                                             uint64_t usedCoreNum, uint64_t transConflict,
                                                             uint64_t newMcnt)
{
    // 行优先最大冲突数
    uint64_t oneCoreBlock = ops::CeilDiv(batch * mCnt * nCnt, usedCoreNum);
    uint64_t rowConflict = ops::CeilDiv(mCnt * nCnt, oneCoreBlock);
    // 错位分核冲突数大于行优先冲突数
    if (transConflict >= rowConflict) {
        return;
    }
    bmmTilingData_.matmulTiling.tileL2cacheTiling.mTileCntL2 =
        static_cast<uint32_t>(cBatchDimAll_ * ops::CeilDiv(mCnt, newMcnt));
    bmmTilingData_.matmulTiling.tileL2cacheTiling.mTileBlock = newMcnt;
    bmmTilingData_.matmulTiling.tileL2cacheTiling.nTileBlock = nCnt;
    bmmTilingData_.matmulTiling.tileL2cacheTiling.calOrder = 0;
}

void Mc2BatchMatmulV3BaseTiling::DoL2CacheAndCalOrderTiling()
{
    uint64_t usedCoreNum = bmmTilingData_.matmulTiling.matmulTiling.usedCoreNum;
    args_.l2Ratio = 1.0 * compileInfo_.l2Size / L2_SIZE_2;
    bool isL2Tile = (runInfo_.l2Info.mTile > 1 || runInfo_.l2Info.nTile > 1);
    bool isBigSize = false;
    if (!isL2Tile && (cBatchDimAll_ > 1UL)) {
        uint64_t totalSize = (aBatchDimAll_ * args_.mValue * args_.kValue * aDtypeSize_) +
                             (bBatchDimAll_ * args_.kValue * args_.nValue * bDtypeSize_) +
                             (cBatchDimAll_ * args_.mValue * args_.nValue * cDtypeSize_);
        isBigSize = totalSize > args_.l2Ratio * 100 * MB_SIZE; // 100 为 MMV3 L2cache暂定值
    }
    uint64_t mCnt =
        ops::CeilDiv(args_.mValue, static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.singleCoreM));
    uint64_t nCnt =
        ops::CeilDiv(args_.nValue, static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.singleCoreN));
    uint64_t transConflict = std::max(ops::CeilDiv(usedCoreNum, mCnt), ops::CeilDiv(usedCoreNum, nCnt));
    uint64_t tmpCnt = mCnt;
    if ((nCnt < usedCoreNum) && (mCnt > usedCoreNum)) {
        tmpCnt = ops::CeilDiv(usedCoreNum, nCnt) * nCnt; // 第一轮m方向没有冲突和复用
        for (; tmpCnt > 0UL; tmpCnt--) {
            uint64_t tailCnt = mCnt % tmpCnt;
            if (static_cast<double>(tailCnt) > TAIL_CONFLICT_RATIO * static_cast<double>(tmpCnt)) {
                break;
            }
        }
    }
    uint64_t newMcnt = tmpCnt;
    if (!isL2Tile && isBigSize && (cBatchDimAll_ < usedCoreNum) && (newMcnt > 0UL) &&
        (transConflict <= MAX_TRANS_CONFLICT)) {
        isL2Tile = true;
    }
    // 使用全载tilingkey作为判断标准
    bool isFullLoad =
        ((tilingEnable_.tilingEnableMultiBatchL1FullLoad == Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE) &&
         (tilingEnable_.tilingEnableMultiBatch == Mc2TilingEnableMultiBatch::IS_FALSE) &&
         ((tilingEnable_.tilingEnableLoadMode == Mc2TilingEnableLoadMode::AL1_FULL_LOAD) ||
          (tilingEnable_.tilingEnableLoadMode == Mc2TilingEnableLoadMode::BL1_FULL_LOAD)) &&
         (tilingEnable_.tilingEnableMultiBatchOut == Mc2TilingEnableMultiBatchOut::IS_FALSE) &&
         (tilingEnable_.tilingEnableMixNd2Nz == Mc2TilingEnableMixNd2Nz::IS_FALSE));
    if (!isFullLoad) {
        if (isL2Tile) {
            bmmTilingData_.matmulTiling.tileL2cacheTiling.mTileCntL2 =
                static_cast<uint32_t>(runInfo_.l2Info.mTile * cBatchDimAll_);
            if (isBigSize) {
                bmmTilingData_.matmulTiling.tileL2cacheTiling.mTileCntL2 =
                    static_cast<uint32_t>(cBatchDimAll_ * ops::CeilDiv(mCnt, newMcnt));
                bmmTilingData_.matmulTiling.tileL2cacheTiling.mTileBlock = newMcnt;
                bmmTilingData_.matmulTiling.tileL2cacheTiling.nTileBlock = nCnt;
            }
        } else {
            CheckandSetDiagonalConflict(mCnt, nCnt, cBatchDimAll_, usedCoreNum, transConflict, newMcnt);
        }
    } else {
        bmmTilingData_.matmulTiling.tileL2cacheTiling.mTileCntL2 = 1;
        bmmTilingData_.matmulTiling.tileL2cacheTiling.nTileCntL2 = 1;
    }
}

ge::graphStatus Mc2BatchMatmulV3BaseTiling::DoLibApiTiling()
{
    auto ret = Mc2MatmulV3BaseTiling::DoLibApiTiling();
    bmmTilingData_.Mc2multiBatchInfo.batchUsedCoreNum = bmmTilingData_.matmulTiling.matmulTiling.usedCoreNum;
    bmmTilingData_.Mc2multiBatchInfo.aBatchDim3 = static_cast<uint32_t>(batchInfo_.batchA3);
    bmmTilingData_.Mc2multiBatchInfo.aBatchDim2 = static_cast<uint32_t>(batchInfo_.batchA2);
    bmmTilingData_.Mc2multiBatchInfo.aBatchDim1 = static_cast<uint32_t>(batchInfo_.batchA1);
    bmmTilingData_.Mc2multiBatchInfo.aBatchDim0 = static_cast<uint32_t>(batchInfo_.batchA0);
    bmmTilingData_.Mc2multiBatchInfo.bBatchDim3 = static_cast<uint32_t>(batchInfo_.batchB3);
    bmmTilingData_.Mc2multiBatchInfo.bBatchDim2 = static_cast<uint32_t>(batchInfo_.batchB2);
    bmmTilingData_.Mc2multiBatchInfo.bBatchDim1 = static_cast<uint32_t>(batchInfo_.batchB1);
    bmmTilingData_.Mc2multiBatchInfo.bBatchDim0 = static_cast<uint32_t>(batchInfo_.batchB0);
    bmmTilingData_.Mc2multiBatchInfo.cBatchDim3 = static_cast<uint32_t>(batchInfo_.batchC3);
    bmmTilingData_.Mc2multiBatchInfo.cBatchDim2 = static_cast<uint32_t>(batchInfo_.batchC2);
    bmmTilingData_.Mc2multiBatchInfo.cBatchDim1 = static_cast<uint32_t>(batchInfo_.batchC1);
    bmmTilingData_.Mc2multiBatchInfo.cBatchDim0 = static_cast<uint32_t>(batchInfo_.batchC0);

    aBatchDimAll_ = batchInfo_.batchA0 * batchInfo_.batchA1 * batchInfo_.batchA2 * batchInfo_.batchA3;
    bBatchDimAll_ = batchInfo_.batchB0 * batchInfo_.batchB1 * batchInfo_.batchB2 * batchInfo_.batchB3;
    cBatchDimAll_ = batchInfo_.batchC0 * batchInfo_.batchC1 * batchInfo_.batchC2 * batchInfo_.batchC3;
    bmmTilingData_.Mc2multiBatchInfo.aBatchDimAll = static_cast<uint32_t>(aBatchDimAll_);
    bmmTilingData_.Mc2multiBatchInfo.bBatchDimAll = static_cast<uint32_t>(bBatchDimAll_);
    bmmTilingData_.Mc2multiBatchInfo.cBatchDimAll = static_cast<uint32_t>(cBatchDimAll_);
    bmmTilingData_.Mc2multiBatchInfo.batchTileBlock = static_cast<uint32_t>(cBatchDimAll_);
    if (CheckBMMTilingDataIsVaild()) {
        return ge::GRAPH_FAILED;
    }
    bmmTilingData_.Mc2multiBatchInfo.biasWithBatch = static_cast<uint32_t>(batchInfo_.biasWithBatch);
    bmmTilingData_.Mc2multiBatchInfo.mOri = static_cast<uint32_t>(args_.mOriValue);

    uint64_t innerSizeA = args_.isATrans ? args_.mValue : args_.kValue;
    uint64_t innerSizeB = args_.isBTrans ? args_.kValue : args_.nValue;
    if (innerSizeA > ND2NZ_ON_THE_FLY_LIMIT || innerSizeB > ND2NZ_ON_THE_FLY_LIMIT) {
        DoUnAlignCommonTiling();
        DoTilingKeyCustom();
        return ret;
    }

    DoCommonTiling();
    DoL1FullLoadTiling();
    DoL2CacheAndCalOrderTiling();
    if (compileInfo_.supportL0c2out && tilingSelect_ != Mc2TilingCalcSelect::COMMON &&
        std::string(context_->GetNodeType()) != "TransposeBatchMatMul" && args_.bFormat != ge::FORMAT_FRACTAL_NZ) {
        DoMultiBatchTiling();
        if (IsMultiBatchAL1FullLoad()) { // 多batch AL1全载
            DoMultiBatchL1FullLoadTiling();
        }
    }
    DoTilingKeyCustom();
    return ret;
}

/*
 * Algorithm to calculate the best (baseM, baseN) that gives even workload amongst iterations.
 * Parameter `divisor` is used to control the starting point of the algorithm.
 * Choosing different starting point can sometimes get better performance.
 * The starting point of `divisor = 2` is half of that of `divisor = 1`
 */
static void CalcBaseMN(uint64_t &baseM, uint64_t &baseN, const mc2_matmul_v3::Mc2MatmulV3Args &args,
                       uint64_t divisor = 1UL)
{
    uint64_t dtypeSize = GetSizeByDataType(args.aType);
    // step 1: calc baseM
    auto getBestBaseM = [dtypeSize, divisor](bool transX1, uint64_t m, uint64_t n, uint64_t k) -> uint64_t {
        if (!transX1) {
            uint64_t maxM =
                k >= 32UL ? 1024UL : 2048UL; // 防止baseK太小,限制baseM最大为2048B, k大于32时进一步限制为1024B
            maxM /= (dtypeSize * divisor);
            uint64_t mTimes = ops::CeilDiv(m, maxM);
            uint64_t singleTimeM = ops::CeilDiv(m, mTimes);
            return ops::CeilAlign(singleTimeM, BLOCK_CUBE);
        }
        uint64_t nCalc = std::min(n, 512UL);                             // baseN大小不会超过512
        uint64_t minBaseM = std::max(32768UL / dtypeSize / nCalc, 32UL); // 最小32, n小时限制baseM*n>=32768B(经验值)
        uint64_t maxBaseM = std::min(NextPower2(m), (2048UL / dtypeSize / divisor)); // L0限制最大2048B, fp16尝试1024B
        uint64_t bestBaseM = std::max(maxBaseM, 16UL);                               // 最小16
        for (uint64_t candidate = bestBaseM; candidate >= minBaseM; candidate >>= 1) {
            if (m % candidate == 0UL) {
                break;
            }
            if (ops::CeilAlign(m, candidate) < ops::CeilAlign(m, bestBaseM)) {
                bestBaseM = candidate;
            }
        }
        return bestBaseM;
    };
    baseM = getBestBaseM(args.isATrans, args.mValue, args.nValue, args.kValue);
    // step 2: calc baseN
    auto getBestBaseN = [dtypeSize, baseM, divisor](bool transX2, uint64_t n, uint64_t k) -> uint64_t {
        uint64_t minBaseN = std::max(std::min(32768UL / dtypeSize / baseM, 512UL / dtypeSize), 32UL); // 32768
        uint64_t maxN = (transX2 && k >= 32UL) ? 1024UL : 2048UL; // 最大2048，k为内轴且大于32时取1024
        uint64_t maxBaseN = maxN / dtypeSize / divisor;
        uint64_t bestBaseN = std::max(std::min(NextPower2(n), maxBaseN), 16UL); // 同时最小设为16
        for (uint64_t candidate = bestBaseN; candidate >= minBaseN; candidate >>= 1) {
            if (n % candidate == 0UL) {
                break;
            }
            if (ops::CeilAlign(n, candidate) < ops::CeilAlign(n, bestBaseN)) {
                bestBaseN = candidate;
            }
        }
        return std::min(LastPower2(32768UL / baseM), bestBaseN); // L0C大小限制baseM * baseN <= 32768;
    };
    baseN = getBestBaseN(args.isBTrans, args.nValue, args.kValue);
}

static void TuneBaseMN(mc2_matmul_v3::Mc2MatmulV3RunInfo &runInfo, const mc2_matmul_v3::Mc2MatmulV3Args &args,
                       uint64_t batchC, uint64_t aicNum)
{
    uint64_t &oriBaseM = runInfo.baseM;
    uint64_t &oriBaseN = runInfo.baseN;
    CalcBaseMN(oriBaseM, oriBaseN, args);
    if (args.aType == ge::DT_FLOAT) {
        return;
    }
    // for bf16 fp16 try a different set of thresholds
    uint64_t newBaseM;
    uint64_t newBaseN;
    CalcBaseMN(newBaseM, newBaseN, args, 2UL); // choose half of the starting point by setting divisor = 2
    // evaluate core utilization
    auto getCoreUtilization = [args, batchC, aicNum](uint64_t baseM, uint64_t baseN) -> double {
        uint64_t cnt = ops::CeilDiv(args.mValue, baseM) * ops::CeilDiv(args.nValue, baseN) * batchC;
        return static_cast<double>(cnt) / ops::CeilAlign(cnt, aicNum);
    };
    double oriCoreUtil = getCoreUtilization(oriBaseM, oriBaseN);
    if ((oriCoreUtil < 0.6) && (getCoreUtilization(newBaseM, newBaseN) > oriCoreUtil)) { // 0.6为经验值
        oriBaseM = newBaseM;
        oriBaseN = newBaseN;
        return;
    }
    // evaluate data copy workload
    auto getDataCopyRatio = [](uint64_t baseM, uint64_t baseN) -> double {
        return 1.0 / static_cast<double>(baseM) + 1.0 / static_cast<double>(baseN);
    };
    if (getDataCopyRatio(newBaseM, newBaseN) < getDataCopyRatio(oriBaseM, oriBaseN)) {
        oriBaseM = newBaseM;
        oriBaseN = newBaseN;
    }
}

static void TuneBaseMKN(mc2_matmul_v3::Mc2MatmulV3RunInfo &runInfo, const mc2_matmul_v3::Mc2MatmulV3Args &args,
                        uint64_t batchC, uint64_t aicNum)
{
    uint64_t &baseM = runInfo.baseM;
    uint64_t &baseN = runInfo.baseN;
    uint64_t &baseK = runInfo.baseK;
    OP_LOGD(args.opName, "before DoCommonTiling baseM, baseN, baseK[%lu, %lu, %lu]", baseM, baseN, baseK);
    bool isSmallShape = (args.mValue < BASIC_BLOCK_SIZE_256 || args.nValue <= BASIC_BLOCK_SIZE_256);
    bool useBaseBlock = ((baseM == BASIC_BLOCK_SIZE_128 || baseM == BASIC_BLOCK_SIZE_256) &&
                         (baseN == BASIC_BLOCK_SIZE_128 || baseN == BASIC_BLOCK_SIZE_256));
    bool isL2Tile = (runInfo.l2Info.mTile > 1 || runInfo.l2Info.nTile > 1);
    if ((!isSmallShape && useBaseBlock) || isL2Tile) {
        return;
    }

    TuneBaseMN(runInfo, args, batchC, aicNum);
    uint64_t maxBaseK = 32768UL / GetSizeByDataType(args.aType) / std::max(baseM, baseN); // L0AB大小限制32768B
    baseK = std::min(ops::FloorAlign(maxBaseK, BLOCK_CUBE), ops::CeilAlign(args.kValue, BLOCK_CUBE));
    OP_LOGD(args.opName, "after DoCommonTiling baseM, baseN, baseK[%lu, %lu, %lu]", baseM, baseN, baseK);
}

void Mc2BatchMatmulV3BaseTiling::DoCommonTiling()
{
    TuneBaseMKN(runInfo_, args_, cBatchDimAll_, compileInfo_.aicNum);

    uint64_t baseM = runInfo_.baseM;
    uint64_t baseN = runInfo_.baseN;
    uint64_t baseK = runInfo_.baseK;
    constexpr uint64_t reserveSize = 256;
    uint64_t totalL1Size = compileInfo_.l1Size + reserveSize; // 256B为预留给rpc使用，单算子不涉及
    if (args_.hasBias) {
        totalL1Size -= reserveSize * 4;       // 1024: 256 * 4, biasTable 空间
        baseN = std::min(reserveSize, baseN); // 带bias， baseN最大值为256
    }
    uint64_t depthA1 =
        (totalL1Size / NUM_TWO / aDtypeSize_ / (baseM * baseK) / 4UL) * 4UL; // DB后FP32下，L1可以存65536个数值
    uint64_t depthB1 =
        (totalL1Size / NUM_TWO / aDtypeSize_ / (baseN * baseK) / 4UL) * 4UL; // DB后FP32下，L1可以存65536个数值
    depthA1 = std::max(NUM_TWO, depthA1);
    depthB1 = std::max(NUM_TWO, depthB1);
    uint64_t stepKa = depthA1 / NUM_TWO;
    uint64_t stepKb = depthB1 / NUM_TWO;

    if (stepKa > stepKb) {
        stepKa = stepKa / stepKb * stepKb;
        depthA1 = stepKa * NUM_TWO;
    } else {
        stepKb = stepKb / stepKa * stepKa;
        depthB1 = stepKb * NUM_TWO;
    }
    bmmTilingData_.matmulTiling.matmulTiling.depthA1 = depthA1;
    bmmTilingData_.matmulTiling.matmulTiling.depthB1 = depthB1;
    bmmTilingData_.matmulTiling.matmulTiling.stepKa = stepKa;
    bmmTilingData_.matmulTiling.matmulTiling.stepKb = stepKb;
    bmmTilingData_.matmulTiling.matmulTiling.stepM = 1;
    bmmTilingData_.matmulTiling.matmulTiling.stepN = 1;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreM = std::min(args_.mValue, baseM);
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreN = std::min(args_.nValue, baseN);
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreK = args_.kValue;
    bmmTilingData_.matmulTiling.matmulTiling.baseM = baseM;
    bmmTilingData_.matmulTiling.matmulTiling.baseN = baseN;
    bmmTilingData_.matmulTiling.matmulTiling.baseK = baseK;

    tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
    tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_FALSE;
    tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::BASE;
    tilingEnable_.tilingEnableMultiBatchOut = Mc2TilingEnableMultiBatchOut::IS_FALSE;
    tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_FALSE;
}

void Mc2BatchMatmulV3BaseTiling::DoUnAlignCommonTiling()
{
    tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
    tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_FALSE;
    tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::BASE;
    tilingEnable_.tilingEnableMultiBatchOut = Mc2TilingEnableMultiBatchOut::IS_FALSE;
    tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_TRUE;
    uint64_t baseM = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.baseM);
    uint64_t baseN = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.baseN);
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreM = std::min(args_.mValue, baseM);
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreN = std::min(args_.nValue, baseN);
}

void Mc2BatchMatmulV3BaseTiling::DoMultiBatchTilingImpl()
{
    uint64_t m = args_.mValue;
    uint64_t n = args_.nValue;
    uint64_t k = args_.kValue;

    uint64_t baseM = 0;
    uint64_t baseN = 0;
    uint64_t baseK = 0;

    uint64_t mTimes = ops::CeilDiv(m, 2048UL / aDtypeSize_);
    uint64_t singleTimeM = ops::CeilDiv(m, mTimes);
    baseM = ops::CeilAlign(singleTimeM, BLOCK_CUBE);
    baseN = std::max(std::min(NextPower2(n), 2048UL / aDtypeSize_), 16UL); // 防止baseN太小
    uint64_t curBaseN = baseN;
    uint64_t bestBaseN = UINT64_MAX;
    while (n % curBaseN != 0UL &&
           curBaseN * baseM >= 32768UL / aDtypeSize_) { // 防止N太小导致计算效率低，限制BaseN * baseM >= 32768B (经验值)
        uint64_t nTimes = ops::CeilDiv(n, curBaseN);
        if (nTimes * curBaseN < bestBaseN) {
            bestBaseN = nTimes * curBaseN;
            baseN = curBaseN;
        }
        curBaseN /= NUM_TWO;
    }
    // 当有bias时，baseN最大不能超过256
    baseN = args_.hasBias ? std::min(baseN, 256UL) : baseN;

    while (baseM * baseN > 32768UL) { // L0C大小限制baseM * baseN <= 32768
        mTimes++;
        singleTimeM = ops::CeilDiv(m, mTimes);
        baseM = ops::CeilAlign(singleTimeM, BLOCK_CUBE);
    }
    uint64_t mnMax = std::max(baseM, baseN);
    baseK = ops::CeilAlign(k, BLOCK_CUBE);
    uint64_t validBaseK = (32768UL / aDtypeSize_ / mnMax / BLOCK_CUBE) * BLOCK_CUBE; // L0AB大小限制BaseK <= 32768
    baseK = std::min(validBaseK, baseK);

    bmmTilingData_.matmulTiling.matmulTiling.baseM = baseM;
    bmmTilingData_.matmulTiling.matmulTiling.baseN = baseN;
    bmmTilingData_.matmulTiling.matmulTiling.baseK = baseK;
}

bool Mc2BatchMatmulV3BaseTiling::DoMultiBatchOutTiling()
{
    uint64_t singleCoreN = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.singleCoreN);
    uint64_t singleCoreM = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.singleCoreM);
    uint64_t baseN = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.baseN);
    uint64_t baseM = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.baseM);
    uint64_t dbL0C = static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.dbL0C);
    uint64_t batchOutCnt = compileInfo_.l0CSize / (baseN * baseM * dbL0C * 4);
    bool isNBatchOut = batchOutCnt > 1UL && batchInfo_.batchC > 1000UL;

    // 多batch输出要求batch内无循环
    isNBatchOut = isNBatchOut && singleCoreM <= baseM && singleCoreN <= baseN;
    uint64_t n = bmmTilingData_.matmulTiling.matmulTiling.N;
    uint64_t m = bmmTilingData_.matmulTiling.matmulTiling.M;
    // 多batch输出适用于双边小矩阵场景，但限制mn不同时为1
    isNBatchOut = isNBatchOut && !(n == 1UL && m == 1UL);
    if (isNBatchOut) {
        return true;
    }
    return false;
}

void Mc2BatchMatmulV3BaseTiling::DoMultiBatchTiling()
{
    bool isEqualBatch = batchInfo_.batchA0 == batchInfo_.batchB0 && batchInfo_.batchA1 == batchInfo_.batchB1 &&
                        batchInfo_.batchA2 == batchInfo_.batchB2 && batchInfo_.batchA3 == batchInfo_.batchB3; // 广播
    if (!isEqualBatch || (args_.hasBias && !batchInfo_.biasWithBatch)) { // 不支持broadcast\非多batch bias
        return;
    }
    uint64_t shapeM = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.M), BLOCK_CUBE);
    uint64_t shapeN = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.N), BLOCK_CUBE);
    uint64_t shapeK = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.Ka), BLOCK_CUBE);
    uint64_t biasSize = 0;
    if (args_.hasBias) {
        biasSize = shapeN * GetSizeByDataType(args_.biasType); // MultiBatch场景,暂时不支持 不带batch的bias.
    }
    uint64_t iterBatch =
        ops::FloorDiv(compileInfo_.l1Size, ((shapeM * shapeK + shapeK * shapeN) * aDtypeSize_ + biasSize));
    if (optiling::mc2_matmul_v3::CheckNumberIsValid(iterBatch, args_.opName, "batchInfo_.iterBatch")) {
        return;
    }
    uint64_t preCoreBatch = ops::FloorDiv(batchInfo_.batchC, compileInfo_.aicNum);
    // if preCoreBatch < 2, no need use Multibatch
    iterBatch = std::max(std::min(iterBatch, preCoreBatch), 1UL);
    if (iterBatch <= 1UL) {
        return;
    }
    iterBatch = ops::FloorAlign(iterBatch, 2UL);
    uint64_t useCoreNum = std::min(ops::CeilDiv(batchInfo_.batchC, iterBatch), compileInfo_.aicNum);
    bmmTilingData_.Mc2multiBatchInfo.batchUsedCoreNum = useCoreNum;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreM = bmmTilingData_.matmulTiling.matmulTiling.M;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreN = bmmTilingData_.matmulTiling.matmulTiling.N;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreK = bmmTilingData_.matmulTiling.matmulTiling.Ka;
    bmmTilingData_.matmulTiling.matmulTiling.BatchNum = static_cast<uint32_t>(iterBatch);
    bmmTilingData_.Mc2multiBatchInfo.iterBatch = static_cast<uint32_t>(iterBatch);
    // multiBatch需要
    bmmTilingData_.matmulTiling.matmulTiling.usedCoreNum = useCoreNum;

    // 更新nd2nz的条件
    UpdateMultiBatchNd2nz();

    bmmTilingData_.matmulTiling.matmulRunInfo.nd2nzA = static_cast<uint32_t>(args_.nd2nzA);
    bmmTilingData_.matmulTiling.matmulRunInfo.nd2nzB = static_cast<uint32_t>(args_.nd2nzB);
    DoMultiBatchTilingImpl();

    // 多batch输出
    bool isMultiBatchOut = DoMultiBatchOutTiling();

    if (args_.nd2nzA || args_.nd2nzB) {
        tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
        tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_TRUE;
        tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::BASE;
        tilingEnable_.tilingEnableMultiBatchOut =
            isMultiBatchOut ? Mc2TilingEnableMultiBatchOut::IS_TRUE : Mc2TilingEnableMultiBatchOut::IS_FALSE;
        tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_TRUE;
        CalculateNd2nzWorkspaceSize();
        return;
    }

    tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
    tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_TRUE;
    tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::BASE;
    tilingEnable_.tilingEnableMultiBatchOut =
        isMultiBatchOut ? Mc2TilingEnableMultiBatchOut::IS_TRUE : Mc2TilingEnableMultiBatchOut::IS_FALSE;
    tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_FALSE;
    return;
}

bool Mc2BatchMatmulV3BaseTiling::IsMultiBatchAL1FullLoad()
{
    // 白名单:((1500,1,128),(1500,512,128)) || !(!args_.isATrans && args_.isBTrans)
    bool isEqualBatch = batchInfo_.batchA0 == batchInfo_.batchB0 && batchInfo_.batchA1 == batchInfo_.batchB1 &&
                        batchInfo_.batchA2 == batchInfo_.batchB2 && batchInfo_.batchA3 == batchInfo_.batchB3;
    constexpr uint64_t BATCH_DIM_ALL = 1500;
    constexpr uint64_t M_VALUE = 1;
    constexpr uint64_t K_VALUE_128 = 128;
    constexpr uint64_t N_VALUE_512 = 512;
    if (isEqualBatch && !args_.hasBias &&
        (aBatchDimAll_ == BATCH_DIM_ALL && args_.mValue == M_VALUE && args_.kValue == K_VALUE_128 &&
         args_.nValue == N_VALUE_512 && !args_.isATrans && args_.isBTrans && args_.aType == ge::DT_FLOAT)) {
        return true;
    }
    if ((!args_.isATrans && args_.isBTrans) || args_.aType != ge::DT_FLOAT) { // x1不转置、x2转置性能差
        return false;
    }
    return true;
}

void Mc2BatchMatmulV3BaseTiling::DoMultiBatchL1FullLoadTilingImpl()
{
    TuneBaseMKN(runInfo_, args_, cBatchDimAll_, compileInfo_.aicNum);

    uint64_t baseM = runInfo_.baseM;
    uint64_t baseN = runInfo_.baseN;
    uint64_t baseK = runInfo_.baseK;
    constexpr uint64_t reserveSize = 256;
    uint64_t totalL1Size = compileInfo_.l1Size + reserveSize; // 256B为预留给rpc使用，单算子不涉及
    if (args_.hasBias) {
        totalL1Size -= reserveSize * 4;       // 1024: 256 * 4, biasTable 空间
        baseN = std::min(reserveSize, baseN); // 带bias， baseN最大值为256
    }
    uint64_t shapeN = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.N), BLOCK_CUBE);
    uint64_t shapeK = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.Ka), BLOCK_CUBE);
    // BL1 batch=1
    uint64_t depthB1 = (shapeN * shapeK * bmmTilingData_.Mc2multiBatchInfo.bBatch / (baseN * baseK) / 4) * 4;
    depthB1 = std::max(NUM_TWO, depthB1);
    uint64_t stepKb = depthB1 / NUM_TWO;

    bmmTilingData_.matmulTiling.matmulTiling.depthB1 = depthB1;
    bmmTilingData_.matmulTiling.matmulTiling.stepKb = stepKb;
    bmmTilingData_.matmulTiling.matmulTiling.stepN = 1;
    bmmTilingData_.matmulTiling.matmulTiling.baseM = baseM;
    bmmTilingData_.matmulTiling.matmulTiling.baseN = baseN;
    bmmTilingData_.matmulTiling.matmulTiling.baseK = baseK;
}

void Mc2BatchMatmulV3BaseTiling::DoMultiBatchL1FullLoadTiling()
{
    bool isEqualBatch = batchInfo_.batchA0 == batchInfo_.batchB0 && batchInfo_.batchA1 == batchInfo_.batchB1 &&
                        batchInfo_.batchA2 == batchInfo_.batchB2 && batchInfo_.batchA3 == batchInfo_.batchB3; // 广播
    if (!isEqualBatch || args_.hasBias) { // 暂时不支持bias
        return;
    }
    uint64_t shapeM = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.M), BLOCK_CUBE);
    uint64_t shapeN = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.N), BLOCK_CUBE);
    uint64_t shapeK = ops::CeilAlign(static_cast<uint64_t>(bmmTilingData_.matmulTiling.matmulTiling.Ka), BLOCK_CUBE);
    uint64_t biasSize = 0;
    if (args_.hasBias) {
        biasSize = shapeN * GetSizeByDataType(args_.biasType); // MultiBatch场景,暂时不支持 不带batch的bias.
    }
    uint64_t bBatch =
        ops::FloorDiv(compileInfo_.l1Size, ((shapeM * shapeK + shapeK * shapeN) * aDtypeSize_ + biasSize));
    if (bBatch != 1UL) { // L1最多放1个B矩阵
        return;
    }
    uint64_t aBatch = ops::FloorDiv(compileInfo_.l1Size - ((shapeK * shapeN) * aDtypeSize_ + biasSize) * bBatch,
                                    (shapeM * shapeK) * aDtypeSize_);
    uint64_t preCoreBatch = ops::FloorDiv(batchInfo_.batchC, compileInfo_.aicNum);
    // if preCoreBatch < 2, no need use Multibatch
    aBatch = std::max(std::min(aBatch, preCoreBatch), 1UL);
    if (aBatch < 2UL) { // if aBatch < 2, no need use AL1 full load
        return;
    }
    bmmTilingData_.Mc2multiBatchInfo.aBatch = static_cast<uint32_t>(aBatch);
    bmmTilingData_.Mc2multiBatchInfo.bBatch = static_cast<uint32_t>(bBatch);
    uint64_t useCoreNum = std::min(ops::CeilDiv(batchInfo_.batchC, aBatch), compileInfo_.aicNum);
    bmmTilingData_.Mc2multiBatchInfo.batchUsedCoreNum = useCoreNum;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreM = bmmTilingData_.matmulTiling.matmulTiling.M;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreN = bmmTilingData_.matmulTiling.matmulTiling.N;
    bmmTilingData_.matmulTiling.matmulTiling.singleCoreK = bmmTilingData_.matmulTiling.matmulTiling.Ka;
    // multiBatch需要
    bmmTilingData_.matmulTiling.matmulTiling.usedCoreNum = useCoreNum;
    bmmTilingData_.matmulTiling.matmulTiling.BatchNum = static_cast<uint32_t>(bBatch);
    // baseM、baseN、baseK
    DoMultiBatchL1FullLoadTilingImpl();
    if (!args_.isATrans && args_.isBTrans) {
        // case ((1500,1,128),(1500,512,128)) tiling调整
        bmmTilingData_.matmulTiling.matmulTiling.depthA1 = 16; // 设置depthA1为16
        bmmTilingData_.matmulTiling.matmulTiling.depthB1 = 8;  // 设置depthB1为8
        bmmTilingData_.matmulTiling.matmulTiling.stepKa = 8;   // 设置stepKa为8
        bmmTilingData_.matmulTiling.matmulTiling.stepKb = 4;   // 设置stepKb为4
        bmmTilingData_.matmulTiling.matmulTiling.stepM = 1;    // 设置stepM为1
        bmmTilingData_.matmulTiling.matmulTiling.stepN = 1;    // 设置stepN为1
        bmmTilingData_.matmulTiling.matmulTiling.baseM = 16;   // 设置baseM为16
        bmmTilingData_.matmulTiling.matmulTiling.baseN = 64;   // 设置baseN为64
        bmmTilingData_.matmulTiling.matmulTiling.baseK = 128;  // 设置baseK为128
    }
    tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_TRUE;
    tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_FALSE;
    tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::BASE;
    tilingEnable_.tilingEnableMultiBatchOut = Mc2TilingEnableMultiBatchOut::IS_FALSE;
    tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_FALSE;
    return;
}


void Mc2BatchMatmulV3BaseTiling::UpdateMultiBatchNd2nz()
{
    uint64_t innerSizeA = args_.isATrans ? args_.mValue : args_.kValue;
    uint64_t innerSizeB = args_.isBTrans ? args_.kValue : args_.nValue;
    bool supportNd2NzOnTheWayA = IsOnTheWay(args_.aFormat, innerSizeA, aDtypeSize_, SUPPORT_ND2NZ_GM2L0_WITHOUT32B);
    bool supportNd2NzOnTheWayB = IsOnTheWay(args_.bFormat, innerSizeB, bDtypeSize_, SUPPORT_ND2NZ_GM2L0_WITHOUT32B);
    bool innerAlignA = args_.isATrans ? m256Align_ : kA256Align_;
    bool innerAlignB = args_.isBTrans ? kB256Align_ : n256Align_;
    uint64_t outerSizeA = args_.isATrans ? args_.kValue : args_.mValue;
    uint64_t outerSizeB = args_.isBTrans ? args_.nValue : args_.kValue;
    args_.nd2nzA = !innerAlignA && !supportNd2NzOnTheWayA && outerSizeA > 4; // 外轴<=4会导致数据量增大减慢搬运
    args_.nd2nzA = args_.nd2nzA && (innerSizeA > 1) &&
                   (innerSizeA * aDtypeSize_ <= 192 ||                          // 192B为最大奇数内轴长度
                    (innerSizeA * aDtypeSize_ <= 384 && innerSizeA % 2 == 0) || // 384B为最大偶数内轴长度
                    (innerSizeA * aDtypeSize_ <= CACHELINE && innerSizeA % 4 == 0));
    args_.nd2nzB = !innerAlignB && !supportNd2NzOnTheWayB && outerSizeB > 4; // 外轴<=4会导致数据量增大减慢搬运
    args_.nd2nzB = args_.nd2nzB && (innerSizeB > 1) &&
                   (innerSizeB * bDtypeSize_ <= 192 ||                          // 192B为最大奇数内轴长度
                    (innerSizeB * bDtypeSize_ <= 384 && innerSizeB % 2 == 0) || // 384B为最大偶数内轴长度
                    (innerSizeB * bDtypeSize_ <= CACHELINE && innerSizeB % 4 == 0));
}

void Mc2BatchMatmulV3BaseTiling::CalculateNd2nzWorkspaceSize()
{
    workspaceSize_ = 0;
    uint64_t alignedSize = 0;
    uint64_t c0 = BLOCK_BYTE_SIZE / aDtypeSize_;
    uint64_t alignedM = args_.isATrans ? ops::CeilAlign(args_.mValue, c0) : ops::CeilAlign(args_.mValue, N_ALIGNED);
    uint64_t alignedKa = args_.isATrans ? ops::CeilAlign(args_.kValue, N_ALIGNED) : ops::CeilAlign(args_.kValue, c0);
    uint64_t alignedKb = args_.isBTrans ? ops::CeilAlign(args_.kValue, c0) : ops::CeilAlign(args_.kValue, N_ALIGNED);
    uint64_t alignedN = args_.isBTrans ? ops::CeilAlign(args_.nValue, N_ALIGNED) : ops::CeilAlign(args_.nValue, c0);
    if (args_.nd2nzA) {
        workspaceSize_ += alignedM * alignedKa * aDtypeSize_ * aBatchDimAll_;
        alignedSize += alignedM * alignedKa * aDtypeSize_;
    }

    if (args_.nd2nzB) {
        workspaceSize_ += alignedKb * alignedN * bDtypeSize_ * bBatchDimAll_;
        alignedSize += alignedKb * alignedN * bDtypeSize_;
    }
    // workspace超L2切分batch轴,单个batch取四分之一L2
    if (workspaceSize_ > compileInfo_.l2Size && alignedSize != 0) {
        uint64_t batchTileBlock = compileInfo_.l2Size / 4 / alignedSize;
        if (batchTileBlock > bmmTilingData_.Mc2multiBatchInfo.iterBatch) {
            bmmTilingData_.Mc2multiBatchInfo.batchTileBlock = batchTileBlock;
            workspaceSize_ = batchTileBlock * alignedSize * DOUBLE_BUFFER;
        }
    }
}

/*
 * Func:    tune down parameter x until either y(x) is below target value, or x has reached its minimum
 * Args:    `target` - target value
 *          `y` - current value of y(x)
 *          `x` - the parameter being tuned
 *          `dydx` - the slope, dy/dx
 *          `step` - the step length (i.e. the minimum variation value) of x. Default is 1.
 * Note:    `x` is assumed to be aligned to `step`, that makes `step` the minimum of `x`.
 */
static void TuneDownParam(uint64_t target, uint64_t &y, uint64_t &x, uint64_t dydx, uint64_t step = 1UL)
{
    if ((y <= target) || (x <= step)) {
        return;
    }

    uint64_t dx = ops::CeilDiv(y - target, dydx);
    dx = ops::CeilAlign(dx, step);
    if (x >= dx + step) {
        y -= dx * dydx;
        x -= dx;
    } else {
        y -= (x - step) * dydx;
        x = step;
    }
}

static void TuneDownBaseBlock(bool isTensorA, const mc2_matmul_v3::Mc2MatmulV3Args &args, uint64_t l1Size,
                              uint64_t &baseMN, uint64_t &baseK)
{
    const uint64_t mnDim = isTensorA ? args.mValue : args.nValue;
    const uint64_t kDim = args.kValue;
    const uint64_t dtypeSize = ge::GetSizeByDataType(isTensorA ? args.aType : args.bType);
    auto tuneBaseK = [mnDim, kDim, baseMN, baseK]() -> bool {
        // if kDim has more unused space than mDim/nDim
        return ops::CeilAlign(kDim, baseK) * mnDim >= ops::CeilAlign(mnDim, baseMN) * kDim;
    };
    auto needToTune = [mnDim, kDim, baseMN, baseK, l1Size, dtypeSize]() -> bool {
        uint64_t loadSize = ops::CeilAlign(mnDim, baseMN) * ops::CeilAlign(kDim, baseK) * dtypeSize;
        return loadSize + 64UL * KB_SIZE > l1Size; // 64 KB is L0A/B size
    };
    while (needToTune()) {
        if (tuneBaseK()) {
            baseK = ops::CeilAlign(baseK / NUM_TWO, BLOCK_CUBE);
            if (baseK <= BLOCK_CUBE) {
                return;
            }
        } else {
            baseMN = ops::CeilAlign(baseMN / NUM_TWO, BLOCK_CUBE);
            if (baseMN <= BLOCK_CUBE) {
                return;
            }
        }
    }
}

static void AL1FullLoadTiling(const mc2_matmul_v3::Mc2MatmulV3Args &args, uint64_t l1Size, uint64_t l0CSize,
                              AscendC::tiling::TCubeTiling &tilingData)
{
    uint64_t baseK = static_cast<uint64_t>(tilingData.baseK);
    uint64_t baseM = static_cast<uint64_t>(tilingData.baseM);
    TuneDownBaseBlock(true, args, l1Size, baseM, baseK);
    const uint64_t stepM = ops::CeilDiv(args.mValue, baseM);
    const uint64_t stepKa = ops::CeilDiv(args.kValue, baseK);
    const uint64_t aSize = stepKa * stepM * baseK * baseM * ge::GetSizeByDataType(args.aType);
    tilingData.baseM = baseM;
    tilingData.baseK = baseK;
    tilingData.stepKa = stepKa;
    tilingData.stepM = stepM;
    tilingData.depthA1 = stepKa * stepM;
    tilingData.singleCoreM = args.mValue;

    uint64_t baseN = static_cast<uint64_t>(tilingData.baseN);
    uint64_t stepN = static_cast<uint64_t>(tilingData.stepN);
    uint64_t stepKb = static_cast<uint64_t>(tilingData.stepKb);
    const uint64_t bSizePerStepN = stepKb * baseK * baseN * ge::GetSizeByDataType(args.bType) * NUM_TWO;
    const uint64_t biasSizePerStepN = args.hasBias ? baseN * ge::GetSizeByDataType(args.biasType) * NUM_TWO : 0UL;
    uint64_t loadSize = aSize + (bSizePerStepN + biasSizePerStepN) * stepN;
    // Tune down loadSize until it is fully loaded in L1
    // Stage 1: try tunning stepN
    TuneDownParam(l1Size, loadSize, stepN, bSizePerStepN + biasSizePerStepN);
    // Stage 2: stepN has reached 1 yet loadSize's still too big for L1, tune down stepKb
    const uint64_t bSizePerStepKb = bSizePerStepN / stepKb;
    TuneDownParam(l1Size, loadSize, stepKb, bSizePerStepKb);
    // Stage 3: stepN & stepKb have both reached 1, tune down baseN
    const uint64_t bSizePerBaseN = (bSizePerStepKb + biasSizePerStepN) / baseN;
    TuneDownParam(l1Size, loadSize, baseN, bSizePerBaseN, BLOCK_CUBE);
    // Last stage: update tiling data
    tilingData.stepN = stepN;
    tilingData.stepKb = stepKb;
    tilingData.baseN = baseN;
    tilingData.depthB1 = stepKb * stepN * NUM_TWO;
    const uint64_t singleCoreN = (stepKb * baseK < args.kValue) ? stepN * baseN : stepN * baseN * NUM_TWO;
    tilingData.singleCoreN = std::min(singleCoreN, args.nValue);
    tilingData.dbL0C = (baseM * baseN * 4 * NUM_TWO > l0CSize ? 1 : NUM_TWO); // 4 is data size within l0c
}

static void BL1FullLoadTiling(const mc2_matmul_v3::Mc2MatmulV3Args &args, uint64_t l1Size, uint64_t l0CSize,
                              AscendC::tiling::TCubeTiling &tilingData)
{
    uint64_t baseK = static_cast<uint64_t>(tilingData.baseK);
    uint64_t baseN = static_cast<uint64_t>(tilingData.baseN);
    TuneDownBaseBlock(false, args, l1Size, baseN, baseK);
    const uint64_t stepN = ops::CeilDiv(args.nValue, baseN);
    const uint64_t stepKb = ops::CeilDiv(args.kValue, baseK);
    const uint64_t bSize = stepKb * stepN * baseK * baseN * ge::GetSizeByDataType(args.bType);
    tilingData.baseN = baseN;
    tilingData.baseK = baseK;
    tilingData.stepKb = stepKb;
    tilingData.stepN = stepN;
    tilingData.depthB1 = stepKb * stepN;
    tilingData.singleCoreN = args.nValue;

    uint64_t baseM = static_cast<uint64_t>(tilingData.baseM);
    uint64_t stepM = static_cast<uint64_t>(tilingData.stepM);
    uint64_t stepKa = static_cast<uint64_t>(tilingData.stepKa);
    const uint64_t biasSize = args.hasBias ? baseN * stepN * ge::GetSizeByDataType(args.biasType) * NUM_TWO : 0UL;
    const uint64_t aSizePerStepM = stepKa * baseK * baseM * ge::GetSizeByDataType(args.aType) * NUM_TWO;
    uint64_t loadSize = bSize + biasSize + aSizePerStepM * stepM;
    // Tune down loadSize util it is fully loaded in L1
    // Stage 1: try tunning stepM
    TuneDownParam(l1Size, loadSize, stepM, aSizePerStepM);
    // Stage 2: stepM has reached 1 yet loadSize's still too big for L1, tune down stepKa
    const uint64_t aSizePerStepKa = aSizePerStepM / stepKa;
    TuneDownParam(l1Size, loadSize, stepKa, aSizePerStepKa);
    // Stage 3: stepM & stepKa have both reached 1, tune down baseM
    const uint64_t aSizePerBaseM = aSizePerStepKa / baseM;
    TuneDownParam(l1Size, loadSize, baseM, aSizePerBaseM, BLOCK_CUBE);
    // Last stage: update tiling data
    tilingData.stepM = stepM;
    tilingData.stepKa = stepKa;
    tilingData.baseM = baseM;
    tilingData.depthA1 = stepKa * stepM * NUM_TWO;
    const uint64_t singleCoreM = (stepKa * baseK < args.kValue) ? stepM * baseM : stepM * baseM * NUM_TWO;
    tilingData.singleCoreM = std::min(singleCoreM, args.mValue);
    tilingData.dbL0C = (baseM * baseN * 4 * NUM_TWO > l0CSize ? 1 : NUM_TWO); // 4 is data size within l0c
}

static void UpdateUsedCoreNum(uint64_t batchC, uint64_t aicNum, Mc2BatchMatmulTilingData &tilingData)
{
    AscendC::tiling::TCubeTiling &matmulTiling = tilingData.matmulTiling.matmulTiling;
    uint64_t mWideNum = static_cast<uint64_t>(ops::CeilDiv(matmulTiling.M, matmulTiling.singleCoreM));
    uint64_t nWideNum = static_cast<uint64_t>(ops::CeilDiv(matmulTiling.N, matmulTiling.singleCoreN));
    uint64_t usedCoreNum = std::min(batchC * mWideNum * nWideNum, aicNum);
    matmulTiling.usedCoreNum = usedCoreNum;
    tilingData.Mc2multiBatchInfo.batchUsedCoreNum = usedCoreNum;
}

void Mc2BatchMatmulV3BaseTiling::DoL1FullLoadTiling()
{
    if (compileInfo_.socVersion == platform_ascendc::SocVersion::ASCEND310P ||
        std::string(context_->GetNodeType()) == "TransposeBatchMatMul") {
        return; // currently not support weight NZ
    }

    enum class L1FullLoad {
        NONE,
        AL1,
        BL1
    };
    const uint64_t totalL1Size = compileInfo_.l1Size + 256; // 256B为预留给rpc使用，单算子不涉及
    auto getL1FullLoad = [totalL1Size](uint64_t aBatchDim, uint64_t bBatchDim,
                                       const Mc2MatmulV3CompileInfo &compileInfo,
                                       const mc2_matmul_v3::Mc2MatmulV3Args &args) -> enum L1FullLoad {
        const bool aNoBatch = aBatchDim <= 1UL; const bool bNoBatch = bBatchDim <= 1UL;
            if ((aNoBatch == bNoBatch) && !(bNoBatch && !args.isATrans)){
                // do l1 full-load if and only if exactly 1 matrix has no batch
                // except for 1 case: (bBatch==1 && !aTrans), where aBatch will be merged into m
                return L1FullLoad::NONE;
    }
    const uint64_t aicoreNum = compileInfo.aicNum;
    auto getMatrixArea = [](uint64_t batch, uint64_t outer, uint64_t inner, int32_t dtypeSize) -> uint64_t {
        const uint64_t c0 = BLOCK_BYTE_SIZE / dtypeSize;
        return std::max(batch, 1UL) * ops::CeilAlign(outer, BLOCK_CUBE) * ops::CeilAlign(inner, c0) * dtypeSize;
    };
    uint64_t aSize = getMatrixArea(aBatchDim, args.isATrans ? args.kValue : args.mValue,
                                   args.isATrans ? args.mValue : args.kValue, ge::GetSizeByDataType(args.aType));
    uint64_t bSize = getMatrixArea(bBatchDim, args.isBTrans ? args.nValue : args.kValue,
                                   args.isBTrans ? args.kValue : args.nValue, ge::GetSizeByDataType(args.bType));
    if (aNoBatch && (aSize * NUM_TWO <= totalL1Size) &&
        (bSize >= totalL1Size * aicoreNum || bBatchDim >= 4UL * aicoreNum)) { // batch 单核上循环4轮
        return L1FullLoad::AL1;
    }
    if (bNoBatch && (bSize * NUM_TWO <= totalL1Size) &&
        (aSize >= totalL1Size * aicoreNum || aBatchDim >= 4UL * aicoreNum)) { // batch 单核上循环4轮
        return L1FullLoad::BL1;
    }
    return L1FullLoad::NONE;
};
switch (getL1FullLoad(aBatchDimAll_, bBatchDimAll_, compileInfo_, args_)) {
    case L1FullLoad::AL1:
        tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
        tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_FALSE;
        tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::AL1_FULL_LOAD;
        tilingEnable_.tilingEnableMultiBatchOut = Mc2TilingEnableMultiBatchOut::IS_FALSE;
        tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_FALSE;
        AL1FullLoadTiling(args_, totalL1Size, compileInfo_.l0CSize, bmmTilingData_.matmulTiling.matmulTiling);
        return UpdateUsedCoreNum(batchInfo_.batchC, compileInfo_.aicNum, bmmTilingData_);
    case L1FullLoad::BL1:
        tilingEnable_.tilingEnableMultiBatchL1FullLoad = Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE;
        tilingEnable_.tilingEnableMultiBatch = Mc2TilingEnableMultiBatch::IS_FALSE;
        tilingEnable_.tilingEnableLoadMode = Mc2TilingEnableLoadMode::BL1_FULL_LOAD;
        tilingEnable_.tilingEnableMultiBatchOut = Mc2TilingEnableMultiBatchOut::IS_FALSE;
        tilingEnable_.tilingEnableMixNd2Nz = Mc2TilingEnableMixNd2Nz::IS_FALSE;
        BL1FullLoadTiling(args_, totalL1Size, compileInfo_.l0CSize, bmmTilingData_.matmulTiling.matmulTiling);
        return UpdateUsedCoreNum(batchInfo_.batchC, compileInfo_.aicNum, bmmTilingData_);
    case L1FullLoad::NONE:
    default:
        return;
}
} // namespace Mc2batch_mat_mul_v3

ge::graphStatus Mc2BatchMatmulV3BaseTiling::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2BatchMatmulTilingData);
    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(args_.opName, "tiling data size[%zu] is not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&bmmTilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    context_->SetBlockDim(compileInfo_.aicNum);
    if ((tilingEnable_.tilingEnableMultiBatchL1FullLoad == Mc2TilingEnableMultiBatchL1FullLoad::IS_FALSE) &&
        (tilingEnable_.tilingEnableMultiBatch == Mc2TilingEnableMultiBatch::IS_TRUE) &&
        (tilingEnable_.tilingEnableLoadMode == Mc2TilingEnableLoadMode::BASE) &&
        (tilingEnable_.tilingEnableMixNd2Nz == Mc2TilingEnableMixNd2Nz::IS_TRUE)) {
        CalculateNd2nzWorkspaceSize();
        workspaceSize_ += RPC_WORKSIZE * MB_SIZE;
    }
    workspaceSize_ = std::max(workspaceSize_, DEFAULT_SIZE);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspaces"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2BatchMatmulV3BaseTiling::GetTilingKey() const
{
    return tilingKey_;
}

void Mc2BatchMatmulV3BaseTiling::DoTilingKeyCustom()
{
    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint64_t>(tilingEnable_.tilingEnableMultiBatchL1FullLoad),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableMultiBatch),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableLoadMode),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableMultiBatchOut),
                                    static_cast<uint64_t>(tilingEnable_.tilingEnableMixNd2Nz), );
    OP_LOGI(args_.opName, "Tiling Key is 0x%x", tilingKey_);
}

} // namespace optiling
}
