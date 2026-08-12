/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_matmul_v3_iterbatch_tiling.cc
 * \brief
 */

#include "batch_matmul_v3_iterbatch_tiling.h"
#include "batch_matmul_v3_tiling_strategy.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace optiling {
namespace Mc2batch_matmul_v3_advanced {
using namespace strategy;
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2BatchMatMulV3, Mc2BatchMatMulV3IterBatchTiling, DAV_3510, ITER_BATCH);
// supportMmadS8S4平台
MC2_MM_REGISTER_TILING_TEMPLATE(Mc2BatchMatMulV3, Mc2BatchMatMulV3IterBatchTiling, DAV_RESV, ITER_BATCH);

bool Mc2BatchMatMulV3IterBatchTiling::IsCapable()
{
    bool isNotEqualBatch = batchInfo_->batchA0 != batchInfo_->batchB0 || batchInfo_->batchA1 != batchInfo_->batchB1 ||
                           batchInfo_->batchA2 != batchInfo_->batchB2 || batchInfo_->batchA3 != batchInfo_->batchB3;
    if (isNotEqualBatch) {
        return false;
    }
    // get align m,k,n value
    uint64_t alignMValue = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignKValue = ops::CeilAlign(args_.kValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignNValue = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
    uint64_t alignBiasSize = 0;
    if (args_.hasBias) {
        alignBiasSize = alignNValue * GetSizeByDataType(args_.biasType);
    }
    inputSizeOneBatch_ = (alignMValue * alignKValue + alignKValue * alignNValue) * args_.aDtypeSize + alignBiasSize;
    iterBatch_ = ops::FloorDiv(compileInfo_.l1Size, inputSizeOneBatch_);
    if (iterBatch_ > 1UL) {
        preCoreBatch_ = ops::CeilDiv(batchInfo_->batchC, compileInfo_.aicNum);
        // if preCoreBatch_ < iterBatch, use preCoreBatch_ for batch
        iterBatch_ = std::max(std::min(iterBatch_, preCoreBatch_), 1UL);
    }
    if (iterBatch_ <= 1UL) {
        return false;
    }
    return true;
}

ge::graphStatus Mc2BatchMatMulV3IterBatchTiling::DoOpTiling()
{
    Mc2MatMulV3TilingHelper::ResetBase(context_, compileInfo_, args_, runInfo_);
    Mc2MatMulV3TilingHelper::CalL1Tiling(compileInfo_, args_, runInfo_);
    runInfo_.singleCoreM = args_.mValue;
    runInfo_.singleCoreN = args_.nValue;
    runInfo_.singleCoreK = args_.kValue;
    bool isEnableMultiBatch = false;
    if (runInfo_.baseM >= runInfo_.singleCoreM && runInfo_.baseN >= runInfo_.singleCoreN) {
        runInfo_.baseM = ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16);
        runInfo_.baseN = ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16);
        isEnableMultiBatch = true;
    }
    uint64_t baseKa = compileInfo_.l0ASize / DB_SIZE / runInfo_.baseM / args_.aDtypeSize;
    uint64_t baseKb = compileInfo_.l0BSize / DB_SIZE / runInfo_.baseN / args_.bDtypeSize;
    uint64_t baseK = std::min(baseKa, baseKb) / BASIC_BLOCK_SIZE_16 * BASIC_BLOCK_SIZE_16;
    uint64_t singleCoreK = (baseK < runInfo_.singleCoreK || iterBatch_ <= 4UL) ? // 4 avoid issueque
                               runInfo_.singleCoreK / DB_SIZE :
                               runInfo_.singleCoreK;
    runInfo_.baseK = std::max(ops::CeilAlign(std::min(singleCoreK, baseK), BASIC_BLOCK_SIZE_16), runInfo_.baseK);
    runInfo_.stepKa = ops::CeilDiv(runInfo_.singleCoreK, runInfo_.baseK);
    runInfo_.stepKb = runInfo_.stepKa;
    runInfo_.stepM = ops::CeilDiv(runInfo_.singleCoreM, runInfo_.baseM);
    runInfo_.stepN = ops::CeilDiv(runInfo_.singleCoreN, runInfo_.baseN);

    runInfo_.depthA1 = runInfo_.stepKa * runInfo_.stepM; // L1 mm fullLoad, batch 2 DB
    runInfo_.depthB1 = runInfo_.stepKb * runInfo_.stepN;

    // need align to 2 for db in api
    iterBatch_ = ops::FloorAlign(iterBatch_, 2UL);
    batchOutNum_ = isEnableMultiBatch ?
                       std::min(ops::FloorDiv(compileInfo_.l0CSize,
                                              (runInfo_.baseM * runInfo_.baseN * runInfo_.dbL0C * sizeof(float))),
                                iterBatch_) :
                       1UL;
    if (iterBatch_ == batchOutNum_) {
        batchOutNum_ = iterBatch_ >> 1;
    }
    runInfo_.bmmRunInfo.iterBatch = iterBatch_;
    runInfo_.bmmRunInfo.batchOutNum = batchOutNum_;
    iterBatchBiasModel_ = (args_.hasBias && (args_.batchInfo->batchBias == 1UL)) ?
                              Mc2MatMulV3Model::ITER_BATCH_SINGLE_BIAS :
                              Mc2MatMulV3Model::ITER_BATCH_BATCH_BIAS;
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2BatchMatMulV3IterBatchTiling::GetTilingKey() const
{
    return Mc2MatMulV3TilingKey().SetTrans(args_.isATrans, args_.isBTrans).SetModel(iterBatchBiasModel_).GetTilingKey();
}

uint64_t Mc2BatchMatMulV3IterBatchTiling::GetBlockDim() const
{
    return compileInfo_.aicNum;
}
} // namespace Mc2batch_matmul_v3_advanced
} // namespace optiling
