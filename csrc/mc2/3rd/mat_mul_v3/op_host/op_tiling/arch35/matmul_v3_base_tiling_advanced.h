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
 * \file matmul_v3_base_tiling_advanced.h
 * \brief
 */
#ifndef __OP_HOST_MATMUL_V3_BASE_TILING_ADVANCED_H__
#define __OP_HOST_MATMUL_V3_BASE_TILING_ADVANCED_H__

#include "common/op_host/math_util.h"
#include "matmul_base_tiling.h"
#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_helper.h"
#include "matmul_v3_tiling_key.h"
#include "matmul_v3_tiling_data.h"

namespace optiling {
namespace mc2_matmul_v3_advanced {

class Mc2MatMulV3BaseTiling : public Mc2MatMulBaseTiling {
public:
    Mc2MatMulV3BaseTiling(gert::TilingContext *context, Mc2MatMulTilingCfg &cfg)
        : Mc2MatMulBaseTiling(context, cfg),
          compileInfo_(*static_cast<const Mc2MatmulV3CompileInfo *>(cfg.compileInfo)),
          args_(*static_cast<const Mc2MatMulV3Args *>(cfg.args)),
          batchInfo_(static_cast<const Mc2MatMulV3BatchInfo *>(args_.batchInfo))
    {
    }
    ~Mc2MatMulV3BaseTiling() override
    {
    }

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        if (context_ == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT("Mc2MatMulV3", "context");
            return ge::GRAPH_FAILED;
        }
        auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
        if (!isValidDimValue(args_.mValue) || !isValidDimValue(args_.kValue) || !isValidDimValue(args_.nValue)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args_.opName, "m, k and n",
                                                  (std::to_string(args_.mValue) + ", " + std::to_string(args_.kValue) +
                                                   " and " + std::to_string(args_.nValue))
                                                      .c_str(),
                                                  "The value of m, k and n must be positive and not exceed INT32_MAX.");
            return ge::GRAPH_FAILED;
        }
        if (compileInfo_.aicNum == 0UL) {
            OP_LOGE(context_->GetNodeName(), "compileInfo.aicNum is 0");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus AdjustOpTiling() override
    {
        if (args_.hasBias) {
            // 有bias时 baseN 小于btsize
            runInfo_.baseN = std::min(std::max(compileInfo_.btSize, BASIC_BLOCK_SIZE_256), runInfo_.baseN);
        }
        return ge::GRAPH_SUCCESS;
    };

    std::vector<size_t> GetWorkspaceSize() const override
    {
        std::vector<size_t> workspaceSize{RPC_WORKSIZE * MB_SIZE};
        return workspaceSize; // 20MB workspace for RPC
    };

    uint64_t GetBlockDim() const override
    {
        return runInfo_.usedCoreNum;
    };

    bool CheckBasicApiTilingKey(uint64_t tilingkey) const
    {
        return Mc2MatMulV3TilingKey().GetApiLevel(tilingkey) == Mc2MatMulV3ApiLevel::BASIC_LEVEL;
    }

    bool CheckIterBatchBasicApi(uint64_t tilingkey) const
    {
        return (Mc2MatMulV3TilingKey().GetApiLevel(tilingkey) == Mc2MatMulV3ApiLevel::BASIC_LEVEL &&
                Mc2MatMulV3TilingKey().GetModel(tilingkey) == Mc2MatMulV3Model::ITER_BATCH_BATCH_BIAS);
    }

    ge::graphStatus PostTiling() override
    {
        Mc2TilingResult tiling;
        tiling.tilingKey = GetTilingKey();
        tiling.workspaceSize = GetWorkspaceSize();
        tiling.blockDim = GetBlockDim();
        Mc2MatMulV3TilingData tilingData;
        Mc2BatchMatMulV3TilingData batchTilingData;
        Mc2BatchMatMulV3BasicTilingData batchBasicTilingData;
        Mc2MatMulV3BasicTilingData tilingBasicData;
        Mc2BatchMatMulV3IterBatchBasicTilingData iterbatchTilingBasicData;
        ge::graphStatus getTilingRet = ge::GRAPH_SUCCESS;
        if (CheckBasicApiTilingKey(tiling.tilingKey) && batchInfo_ == nullptr) {
            getTilingRet = GetTilingData(tilingBasicData);
            tiling.tilingData = static_cast<void *>(&tilingBasicData);
            tiling.tilingDataSize = sizeof(Mc2MatMulV3BasicTilingData);
        } else if (CheckIterBatchBasicApi(tiling.tilingKey)) {
            getTilingRet = GetTilingData(iterbatchTilingBasicData);
            tiling.tilingData = static_cast<void *>(&iterbatchTilingBasicData);
            tiling.tilingDataSize = sizeof(Mc2BatchMatMulV3IterBatchBasicTilingData);
        } else if (batchInfo_ == nullptr) {
            getTilingRet = GetTilingData(tilingData);
            tiling.tilingData = static_cast<void *>(&tilingData);
            tiling.tilingDataSize = sizeof(Mc2MatMulV3TilingData);
        } else if (CheckBasicApiTilingKey(tiling.tilingKey)) {
            GetTilingData(batchBasicTilingData);
            tiling.tilingData = static_cast<void *>(&batchBasicTilingData);
            tiling.tilingDataSize = sizeof(Mc2BatchMatMulV3BasicTilingData);
        } else {
            getTilingRet = GetTilingData(batchTilingData);
            tiling.tilingData = static_cast<void *>(&batchTilingData);
            tiling.tilingDataSize = sizeof(Mc2BatchMatMulV3TilingData);
        }
        if (getTilingRet == ge::GRAPH_FAILED) {
            OP_LOGE(context_->GetNodeName(), "Get tiling data from api failed");
            return ge::GRAPH_FAILED;
        }
        if (cfg_.needUpdate) {
            cfg_.Update(tiling);
            return ge::GRAPH_SUCCESS;
        }
        if (SetTilingData(tiling) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        context_->SetBlockDim(tiling.blockDim);
        context_->SetTilingKey(tiling.tilingKey);
        if (tiling.workspaceSize.size() > 0) {
            size_t *workspaces = context_->GetWorkspaceSizes(1); // set workapce
            OP_TILING_CHECK(workspaces == nullptr, OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "workspace"),
                            return ge::GRAPH_FAILED);
            workspaces[0] = tiling.workspaceSize[0];
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus SetTilingData(const Mc2TilingResult &tiling) const
    {
        if ((strcmp(context_->GetNodeType(), "Mc2MatMulV3") == 0) && (tiling.tilingDataSize <= TILINGDATA_OFFSET) &&
            (!CheckBasicApiTilingKey(tiling.tilingKey)) && (!CheckIterBatchBasicApi(tiling.tilingKey))) {
            for (uint64_t i = 0; i < TILINGDATA_SPLIT_NUM; ++i) {
                errno_t ret = memcpy_s((uint8_t *)context_->GetRawTilingData()->GetData() + i * TILINGDATA_OFFSET,
                                       context_->GetRawTilingData()->GetCapacity() - i * TILINGDATA_OFFSET,
                                       tiling.tilingData, tiling.tilingDataSize);
                if (ret != EOK) {
                    OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
                    return ge::GRAPH_FAILED;
                }
            }
            context_->GetRawTilingData()->SetDataSize(ops::CeilAlign(tiling.tilingDataSize, TILINGDATA_OFFSET) +
                                                      tiling.tilingDataSize);
        } else {
            errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                   tiling.tilingData, tiling.tilingDataSize);
            if (ret != EOK) {
                OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
                return ge::GRAPH_FAILED;
            }
            context_->GetRawTilingData()->SetDataSize(tiling.tilingDataSize);
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus InitTCubeTilingData(::TCubeTiling &tCubeTiling) const
    {
        matmul_tiling::MultiCoreMatmulTiling mm;
        auto aFormat = args_.aFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
        auto bFormat = args_.bFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
        auto cFormat = args_.outFormat == ge::FORMAT_ND ? matmul_tiling::CubeFormat::ND : matmul_tiling::CubeFormat::NZ;
        try {
            mm.SetAType(matmul_tiling::TPosition::GM, aFormat, dtypeMap_.at(args_.aType), args_.isATrans);
            mm.SetBType(matmul_tiling::TPosition::GM, bFormat, dtypeMap_.at(args_.bType), args_.isBTrans);
            mm.SetCType(matmul_tiling::TPosition::GM, cFormat, dtypeMap_.at(args_.cType));
            mm.SetDim(compileInfo_.aicNum);
            mm.SetShape(args_.mValue, args_.nValue, args_.kValue);
            mm.SetOrgShape(args_.mValue, args_.nValue, args_.kValue);
            if (args_.hasBias) {
                mm.SetBias(true);
                mm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                               dtypeMap_.at(args_.biasType));
            }
        } catch (const std::out_of_range &e) {
            OP_LOGE(args_.opName, "Mc2MatMulV3 Set Type Failed! %d, %d, %d, %d", static_cast<int32_t>(args_.aType),
                    static_cast<int32_t>(args_.bType), static_cast<int32_t>(args_.cType),
                    static_cast<int32_t>(args_.biasType));
            return ge::GRAPH_FAILED;
        }

        mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0CSize, compileInfo_.ubSize);
        if (mm.GetTiling(tCubeTiling) == -1) {
            OP_LOGE(args_.opName, "Mc2MatMulV3 Get Tiling Failed!");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    uint64_t GetAswWindowLen() const
    {
        uint64_t sqrtNum = static_cast<uint64_t>(sqrt(compileInfo_.aicNum));
        for (uint64_t factor = sqrtNum; factor >= 1UL; --factor) {
            if (compileInfo_.aicNum % factor == 0UL) {
                return factor;
            }
        }
        return 1UL;
    }

    virtual ge::graphStatus GetTilingData(Mc2MatMulV3TilingData &tilingData) const
    {
        ge::graphStatus ret = InitTCubeTilingData(tilingData.tCubeTiling);
        tilingData.tCubeTiling.usedCoreNum = runInfo_.usedCoreNum;
        tilingData.tCubeTiling.singleCoreM = runInfo_.singleCoreM;
        tilingData.tCubeTiling.singleCoreN = runInfo_.singleCoreN;
        tilingData.tCubeTiling.singleCoreK = runInfo_.singleCoreK;
        tilingData.tCubeTiling.baseM = runInfo_.baseM;
        tilingData.tCubeTiling.baseN = runInfo_.baseN;
        tilingData.tCubeTiling.baseK = runInfo_.baseK;
        tilingData.tCubeTiling.depthA1 = runInfo_.depthA1;
        tilingData.tCubeTiling.depthB1 = runInfo_.depthB1;
        tilingData.tCubeTiling.stepM = runInfo_.stepM;
        tilingData.tCubeTiling.stepN = runInfo_.stepN;
        tilingData.tCubeTiling.stepKa = runInfo_.stepKa;
        tilingData.tCubeTiling.stepKb = runInfo_.stepKb;
        tilingData.tCubeTiling.iterateOrder = runInfo_.iterateOrder;
        tilingData.tCubeTiling.dbL0C = runInfo_.dbL0C;
        tilingData.tCubeTiling.BatchNum = runInfo_.bmmRunInfo.iterBatch;
        tilingData.mTailCnt = runInfo_.tailInfo.mCnt;
        tilingData.nTailCnt = runInfo_.tailInfo.nCnt;
        tilingData.kTailCnt = runInfo_.tailInfo.kCnt;
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = runInfo_.tailInfo.mTailMain;
        tilingData.nTailMain = runInfo_.tailInfo.nTailMain;
        tilingData.isHf32 = args_.isHf32;
        tilingData.aswWindowLen = GetAswWindowLen();
        return ret;
    };

    virtual ge::graphStatus GetTilingData(Mc2BatchMatMulV3TilingData &tilingData) const
    {
        tilingData.aBatchDimAll = batchInfo_->batchA;
        tilingData.bBatchDimAll = batchInfo_->batchB;
        tilingData.cBatchDimAll = batchInfo_->batchC;
        tilingData.biasBatchDimAll = batchInfo_->batchBias;
        tilingData.aBatchDim0 = batchInfo_->batchA0;
        tilingData.aBatchDim1 = batchInfo_->batchA1;
        tilingData.aBatchDim2 = batchInfo_->batchA2;
        tilingData.aBatchDim3 = batchInfo_->batchA3;
        tilingData.bBatchDim0 = batchInfo_->batchB0;
        tilingData.bBatchDim1 = batchInfo_->batchB1;
        tilingData.bBatchDim2 = batchInfo_->batchB2;
        tilingData.bBatchDim3 = batchInfo_->batchB3;
        tilingData.cBatchDim0 = batchInfo_->batchC0;
        tilingData.cBatchDim1 = batchInfo_->batchC1;
        tilingData.cBatchDim2 = batchInfo_->batchC2;
        tilingData.cBatchDim3 = batchInfo_->batchC3;
        tilingData.iterBatch = runInfo_.bmmRunInfo.iterBatch;
        tilingData.batchOutNum = runInfo_.bmmRunInfo.batchOutNum;

        return GetTilingData(tilingData.matMulTilingData);
    };

    virtual ge::graphStatus GetTilingData(Mc2BatchMatMulV3BasicTilingData &tilingData) const
    {
        tilingData.batchDimAll = batchInfo_->batchA;
        return GetTilingData(tilingData.matMulTilingData);
    };

    virtual ge::graphStatus GetTilingData(Mc2MatMulV3BasicTilingData &tilingData) const
    {
        tilingData.usedCoreNum = runInfo_.usedCoreNum;
        tilingData.m = args_.mValue;
        tilingData.n = args_.nValue;
        tilingData.k = args_.kValue;
        tilingData.mL1 = runInfo_.baseM;
        tilingData.nL1 = std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN * runInfo_.stepN);
        int32_t stepKa = std::min(runInfo_.stepKb, runInfo_.stepKa);
        int32_t STEPKA_THERSHOLD = 4;
        stepKa = std::min(STEPKA_THERSHOLD, stepKa);
        tilingData.kL1 = runInfo_.baseK * static_cast<uint32_t>(stepKa);
        tilingData.skSingleCoreK = runInfo_.singleCoreK;
        tilingData.baseM = runInfo_.baseM;
        tilingData.baseN = runInfo_.baseN;
        tilingData.baseK = runInfo_.baseK;
        tilingData.mTailCnt = runInfo_.tailInfo.mCnt;
        tilingData.nTailCnt = runInfo_.tailInfo.nCnt;
        tilingData.isHf32 = static_cast<uint8_t>(args_.isHf32);
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.l0cDB = static_cast<uint8_t>(runInfo_.dbL0C);
        tilingData.ubDB = static_cast<uint8_t>(runInfo_.mixInfo.ubDB);
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = runInfo_.tailInfo.mTailMain;
        tilingData.nTailMain = runInfo_.tailInfo.nTailMain;
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingData(Mc2BatchMatMulV3IterBatchBasicTilingData &iterbatchTilingBasicData) const
    {
        iterbatchTilingBasicData.m = args_.mValue;
        iterbatchTilingBasicData.n = args_.nValue;
        iterbatchTilingBasicData.k = args_.kValue;
        iterbatchTilingBasicData.b = batchInfo_->batchC;
        iterbatchTilingBasicData.iterBatchL1 = runInfo_.iterBatchL1;
        iterbatchTilingBasicData.iterBatchL0 = runInfo_.iterBatchL0;
        iterbatchTilingBasicData.isHf32 = args_.isHf32;
        iterbatchTilingBasicData.baseM = runInfo_.baseM;
        iterbatchTilingBasicData.baseN = runInfo_.baseN;
        iterbatchTilingBasicData.baseK = BASIC_BLOCK_SIZE_16;
        return ge::GRAPH_SUCCESS;
    };

protected:
    const Mc2MatmulV3CompileInfo &compileInfo_;
    const Mc2MatMulV3Args &args_;
    const Mc2MatMulV3BatchInfo *batchInfo_;
    Mc2MatMulV3RunInfo runInfo_;

private:
    const std::map<ge::DataType, matmul_tiling::DataType> dtypeMap_ = {
        {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
        {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
        {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
    };
};
} // namespace mc2_matmul_v3_advanced
} // namespace optiling
#endif // __OP_HOST_MATMUL_V3_BASE_TILING_ADVANCED_H__
