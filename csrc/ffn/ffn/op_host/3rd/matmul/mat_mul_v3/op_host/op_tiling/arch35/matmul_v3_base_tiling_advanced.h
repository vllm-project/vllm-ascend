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
 * \file matmul_v3_base_tiling_advanced.h
 * \brief
 */
#pragma once

#include "matmul/common/op_host/math_util.h"
#include "matmul_base_tiling.h"
#include "matmul_v3_common_advanced.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_helper.h"
#include "matmul_v3_tiling_data.h"
#include "error_util.h"
#include "matmul/common/op_host/log_format_util.h"

namespace optiling {
namespace matmul_v3_advanced {

class MatMulV3BaseTiling : public MatMulBaseTiling {
public:
    MatMulV3BaseTiling(gert::TilingContext* context, MatMulTilingCfg& cfg)
        : MatMulBaseTiling(context, cfg),
          compileInfo_(*static_cast<const MatmulV3CompileInfo*>(cfg.compileInfo)),
          args_(*static_cast<const MatMulV3Args*>(cfg.args)),
          batchInfo_(static_cast<const MatMulV3BatchInfo*>(args_.batchInfo)),
          tilingKeyObj(cfg.tilingKeyObj)
    {}
    ~MatMulV3BaseTiling() override = default;

protected:
    ge::graphStatus GetShapeAttrsInfo() override
    {
        if (context_ == nullptr) {
            OP_LOGE("MatMulV3", "context_ is nullptr");
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& selfShape = context_->GetInputShape(0)->GetOriginShape();
        const gert::Shape& mat2Shape = context_->GetInputShape(1)->GetOriginShape();
        if (args_.kValue == 0UL && args_.hasBias) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "a", Ops::Base::ToString(selfShape).c_str(),
                Ops::NN::FormatString("When optional parameter %s exists, %s of %s must be a positive number", "bias",
                                      "k-axis", "a")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
        auto isValidDimValueK = [](int64_t dim) -> bool { return (dim >= 0) && (dim <= INT32_MAX); };
        if (!isValidDimValue(args_.mValue) || !isValidDimValueK(args_.kValue) || !isValidDimValue(args_.nValue)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                args_.opName, "a, b",
                Ops::NN::FormatString("%s, %s", Ops::Base::ToString(selfShape).c_str(),
                                      Ops::Base::ToString(mat2Shape).c_str())
                    .c_str(),
                Ops::NN::FormatString("%s of %s must be within the range %s, %s of %s must be within the range %s",
                                      "m, n", "a, b", "(0, INT32_MAX]", "k", "a, b", "[0, INT32_MAX]")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        if (compileInfo_.aicNum == 0UL) {
            OP_LOGE(context_->GetNodeName(), "compileInfo.aicNum is 0");
            return ge::GRAPH_FAILED;
        }
        // 为 DAV_RESV 获取私有属性
        if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
            auto attrs = context_->GetAttrs();
            if (attrs == nullptr) {
                OP_LOGE(context_->GetNodeName(), "attrs is nullptr");
                return ge::GRAPH_FAILED;
            }
            auto attrsNum = attrs->GetAttrNum();
            // 防止越界
            if (attrsNum <= 2) {
                OP_LOGE(context_->GetNodeName(), "enable_uncache attr not registered, use default.");
                return ge::GRAPH_FAILED;
            }
            // 倒数第2个属性
            auto enableUncacheAttr = attrs->GetAttrPointer<int64_t>(attrsNum - 2);
            enableUncache_ = (enableUncacheAttr != nullptr && *enableUncacheAttr != 0);

            auto shiftValuePtr = attrs->GetAttrPointer<int64_t>(attrsNum - 1);
            shiftValue_ = shiftValuePtr ? *shiftValuePtr : 0;
            shiftValue_ = shiftValue_ == 0 ? ORI_SHIFT_VALUE : shiftValue_;
            supportMmadS8S4_ = true;
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

    uint64_t GetNumBlocks() const override { return runInfo_.usedCoreNum; };

    bool CheckBasicApiTilingKey(uint64_t tilingkey) const
    {
        return MatMulV3TilingKey().GetApiLevel(tilingkey) != MatMulV3ApiLevel::HIGH_LEVEL;
    }

    bool CheckIterBatchBasicApi(uint64_t tilingkey) const
    {
        return (MatMulV3TilingKey().GetApiLevel(tilingkey) != MatMulV3ApiLevel::HIGH_LEVEL &&
                MatMulV3TilingKey().GetBatchModel(tilingkey) == MatMulV3BatchModel::SINGLE_BIAS_MODEL);
    }

    bool CheckMatMulStreamK(uint64_t tilingkey) const
    {
        return (MatMulV3TilingKey().GetModel(tilingkey) == MatMulV3Model::STREAM_K ||
                MatMulV3TilingKey().GetModel(tilingkey) == MatMulV3Model::SK_SPLIT_K);
    }

    // 针对非全载右矩阵是否进入L2条件
    L2CacheMode SetDisableL2cache(uint32_t mL1, uint32_t kaL1, uint32_t kbL1, uint32_t nL1) const
    {
        L2CacheMode cacheMode = L2CacheMode::L2_CACHE_DEFAULT;
        uint64_t innerA = args_.isATrans ? args_.mValue : args_.kValue;
        uint64_t innerB = args_.isBTrans ? args_.kValue : args_.nValue;
        // 判断切分部分是不是128B对齐
        bool flagA = args_.isATrans ? (mL1 * args_.aDtypeSize % ALIGN_128 == 0) :
                                      (kaL1 * args_.aDtypeSize % ALIGN_128 == 0);
        bool flagB = args_.isBTrans ? (kbL1 * args_.bDtypeSize % ALIGN_128 == 0) :
                                      (nL1 * args_.bDtypeSize % ALIGN_128 == 0);

        uint64_t totalSize = args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.cType) +
                             args_.mValue * args_.kValue * args_.aDtypeSize +
                             args_.kValue * args_.nValue * args_.bDtypeSize;
        if (batchInfo_ != nullptr) {
            totalSize = args_.mValue * args_.nValue * ge::GetSizeByDataType(args_.cType) * batchInfo_->batchC +
                        args_.mValue * args_.kValue * args_.aDtypeSize * batchInfo_->batchA +
                        args_.kValue * args_.nValue * args_.bDtypeSize * batchInfo_->batchB;
        }

        OP_LOGD("MatMulV3", "Input + Output totalSize: %lu, l2Size:%lu.", totalSize, compileInfo_.l2Size);
        if (compileInfo_.npuArch == NpuArch::DAV_RESV) {
            if (!enableUncache_) {
                OP_LOGD("MatMulV3", "enable_uncache is not set to 1, L2 uncache disabled.");
                return L2CacheMode::L2_CACHE_DEFAULT;
            }
            if (totalSize < compileInfo_.l2Size) {
                bool rightNotL2Cache = runInfo_.baseM >= args_.mValue && runInfo_.tailInfo.mCnt <= 1 &&
                                       innerB * args_.bDtypeSize % ALIGN_128 == 0 && flagB;
                L2CacheMode result = rightNotL2Cache ? L2CacheMode::B_L2_CACHE_DISABLE : L2CacheMode::L2_CACHE_DEFAULT;
                OP_LOGD("MatMulV3", "DAV_RESV L2 cache: flagB:%d, rightNotL2Cache:%d, cacheMode:%d.",
                        static_cast<int32_t>(flagB), static_cast<int32_t>(rightNotL2Cache),
                        static_cast<int32_t>(result));
                return result;
            }
        } else if (totalSize < compileInfo_.l2Size) {
            return cacheMode;
        }
        // 左矩阵UNCACHE
        bool leftNotL2Cache = runInfo_.baseN >= args_.nValue && runInfo_.tailInfo.nCnt <= 1 &&
                              innerA * args_.aDtypeSize % ALIGN_128 == 0 && flagA;
        // 右矩阵UNCACHE
        bool rightNotL2Cache = runInfo_.baseM >= args_.mValue && runInfo_.tailInfo.mCnt <= 1 &&
                               innerB * args_.bDtypeSize % ALIGN_128 == 0 && flagB;

        if (leftNotL2Cache && rightNotL2Cache) {
            cacheMode = L2CacheMode::ALL_L2_CACHE_DISABLE;
        } else if (leftNotL2Cache) {
            cacheMode = L2CacheMode::A_L2_CACHE_DISABLE;
        } else if (rightNotL2Cache) {
            cacheMode = L2CacheMode::B_L2_CACHE_DISABLE;
        }
        OP_LOGD("MatMulV3", "L2 cache params: flagA:%d, flagB:%d, leftNotL2Cache:%d, rightNotL2Cache:%d, cacheMode:%d.",
                static_cast<int32_t>(flagA), static_cast<int32_t>(flagB), static_cast<int32_t>(leftNotL2Cache),
                static_cast<int32_t>(rightNotL2Cache), static_cast<int32_t>(cacheMode));

        return cacheMode;
    }

    // 子类中指定模版
    virtual ge::graphStatus GetTilingData(TilingResult& tiling) const = 0;

    // 根据模版赋值TilingData
    template <typename TilingDataType>
    ge::graphStatus GetTilingDataImpl(TilingResult& tiling) const
    {
        std::shared_ptr<TilingDataType> tilingDataPtr;
        try {
            tilingDataPtr = std::make_shared<TilingDataType>();
        } catch (const std::bad_alloc& e) {
            OP_LOGE(context_->GetNodeName(), "Failed to allocate memory for tilingData ");
            return ge::GRAPH_FAILED;
        }
        ge::graphStatus getTilingRet = GetTilingDataProcess(*tilingDataPtr);
        tiling.tilingData = tilingDataPtr;
        tiling.tilingDataSize = sizeof(TilingDataType);
        return getTilingRet;
    }

    ge::graphStatus PostTiling() override
    {
        TilingResult tiling;
        tiling.tilingKey = GetTilingKey();
        tiling.workspaceSize = GetWorkspaceSize();
        tiling.numBlocks = GetNumBlocks();
        ge::graphStatus getTilingRet = ge::GRAPH_SUCCESS;
        getTilingRet = GetTilingData(tiling);
        if (getTilingRet == ge::GRAPH_FAILED) {
            OP_LOGE(context_->GetNodeName(), "Get tiling data from api failed");
            return ge::GRAPH_FAILED;
        }
        if (cfg_.needUpdate) {
            OP_TILING_CHECK(cfg_.Update(tiling) == ge::GRAPH_FAILED,
                            CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "tiling update failed"),
                            return ge::GRAPH_FAILED);
            return ge::GRAPH_SUCCESS;
        }
        if (SetTilingData(tiling) == ge::GRAPH_FAILED) {
            return ge::GRAPH_FAILED;
        }
        context_->SetBlockDim(tiling.numBlocks);
        context_->SetTilingKey(tiling.tilingKey);
        if (CheckMatMulStreamK(tiling.tilingKey)) {
            context_->SetScheduleMode(1);
        }
        if (tiling.workspaceSize.size() > 0) {
            size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
            OP_TILING_CHECK(workspaces == nullptr,
                            CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspace is nullptr"),
                            return ge::GRAPH_FAILED);
            workspaces[0] = tiling.workspaceSize[0];
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus InitTCubeTilingData(::TCubeTiling& tCubeTiling) const
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
        } catch (const std::out_of_range& e) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                args_.opName, "a, b, c, bias",
                Ops::NN::FormatString("%s, %s, %s, %s", Ops::Base::ToString(args_.aType).c_str(),
                                      Ops::Base::ToString(args_.bType).c_str(),
                                      Ops::Base::ToString(args_.cType).c_str(),
                                      Ops::Base::ToString(args_.biasType).c_str())
                    .c_str(),
                Ops::NN::FormatString("The dtypes of %s must be within the range %s", "a, b, c, bias",
                                      "matmul_tiling dtypeMap")
                    .c_str());
            return ge::GRAPH_FAILED;
        }

        mm.SetBufferSpace(compileInfo_.l1Size, compileInfo_.l0CSize, compileInfo_.ubSize);
        if (mm.GetTiling(tCubeTiling) == -1) {
            OP_LOGE(args_.opName, "MatMulV3 Get Tiling Failed!");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    };

    ge::graphStatus SetTilingData(const TilingResult& tiling) const
    {
        if ((strcmp(context_->GetNodeType(), "MatMulV3") == 0) && (tiling.tilingDataSize <= TILINGDATA_OFFSET) &&
            (!CheckBasicApiTilingKey(tiling.tilingKey)) && (!CheckIterBatchBasicApi(tiling.tilingKey))) {
            for (uint64_t i = 0; i < TILINGDATA_SPLIT_NUM; ++i) {
                errno_t ret = memcpy_s((uint8_t*)context_->GetRawTilingData()->GetData() + i * TILINGDATA_OFFSET,
                                       context_->GetRawTilingData()->GetCapacity() - i * TILINGDATA_OFFSET,
                                       tiling.tilingData.get(), tiling.tilingDataSize);
                if (ret != EOK) {
                    OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
                    return ge::GRAPH_FAILED;
                }
            }
            context_->GetRawTilingData()->SetDataSize(ops::CeilAlign(tiling.tilingDataSize, TILINGDATA_OFFSET) +
                                                      tiling.tilingDataSize);
        } else {
            errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                                   tiling.tilingData.get(), tiling.tilingDataSize);
            if (ret != EOK) {
                OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
                return ge::GRAPH_FAILED;
            }
            context_->GetRawTilingData()->SetDataSize(tiling.tilingDataSize);
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

    // MM高阶API模板GetTilingDataProcess
    virtual ge::graphStatus GetTilingDataProcess(MatMulV3TilingData& tilingData) const
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
        tilingData.mmadParam = supportMmadS8S4_ ? static_cast<uint8_t>(shiftValue_) : args_.isHf32;
        tilingData.aswWindowLen = GetAswWindowLen();
        tilingData.l2CacheDisable = SetDisableL2cache(tilingData.tCubeTiling.baseM * tilingData.tCubeTiling.stepM,
                                                      tilingData.tCubeTiling.baseK * tilingData.tCubeTiling.stepKa,
                                                      tilingData.tCubeTiling.baseK * tilingData.tCubeTiling.stepKb,
                                                      tilingData.tCubeTiling.baseN * tilingData.tCubeTiling.stepN);
        return ret;
    };

    virtual ge::graphStatus GetTilingDataProcess(BatchMatMulV3TilingData& tilingData) const
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
        tilingData.mL1 = static_cast<uint32_t>(
            std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM * runInfo_.stepM));
        tilingData.nL1 = static_cast<uint32_t>(
            std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN * runInfo_.stepN));
        int32_t stepKa = std::min(runInfo_.stepKb, runInfo_.stepKa);
        int32_t STEPKA_THERSHOLD = 4;
        stepKa = std::min(STEPKA_THERSHOLD, stepKa);
        tilingData.kL1 = runInfo_.baseK * static_cast<uint32_t>(stepKa);
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.ubDB = static_cast<uint8_t>(runInfo_.mixInfo.ubDB);

        auto selfViewShape = context_->GetInputShape(0)->GetOriginShape();
        auto mat2Shape = context_->GetInputShape(1)->GetOriginShape();
        auto selfStorageShape = context_->GetInputShape(0)->GetStorageShape();
        // 非连续Slice校验
        // TensorV2 & 3d && storageShape 1d
        if (context_->InputIsView(0) && selfViewShape.GetDimNum() == 3 && mat2Shape.GetDimNum() == 2 &&
            selfStorageShape.GetDimNum() == 1) {
            auto selfViewStride = context_->GetInputStride(0);
            tilingData.sliceM = static_cast<uint32_t>(selfViewShape[1]); // sliceM=self[1], ndNum = baseM/sliceM
            tilingData.srcNdStride = static_cast<uint32_t>(selfViewStride->GetStride(0)); // oriM * srcK
        } else {
            tilingData.sliceM = runInfo_.baseM;
            tilingData.srcNdStride = 1;
        }
        tilingData.innerBatch = runInfo_.innerBatch;
        tilingData.iterBatchL1 = static_cast<uint32_t>(runInfo_.iterBatchL1);
        tilingData.iterBatchL0 = static_cast<uint32_t>(runInfo_.iterBatchL0);
        tilingData.broadcastAxisA = runInfo_.bmmRunInfo.broadcastAxisA;
        tilingData.broadcastAxisB = runInfo_.bmmRunInfo.broadcastAxisB;
        tilingData.mmadParam = supportMmadS8S4_ ? static_cast<uint8_t>(shiftValue_) : args_.isHf32;
        return GetTilingDataProcess(tilingData.matMulTilingData);
    };

    virtual ge::graphStatus GetTilingDataProcess(BatchMatMulV3BasicTilingData& tilingData) const
    {
        // A全载和B全载基础API当前只支持单边batch, 后续放开后不能再用batchC
        tilingData.batchDimAll = batchInfo_->batchC;
        tilingData.batchX3 = args_.batchX3;
        return GetTilingDataProcess(tilingData.matMulTilingData);
    };

    // MM基础API模板GetTilingDataProcess
    virtual ge::graphStatus GetTilingDataProcess(MatMulV3BasicTilingData& tilingData) const
    {
        tilingData.usedCoreNum = runInfo_.usedCoreNum;
        tilingData.m = args_.mValue;
        tilingData.n = args_.nValue;
        tilingData.k = args_.kValue;
        tilingData.mL1 = static_cast<uint32_t>(
            std::min(ops::CeilAlign(args_.mValue, BASIC_BLOCK_SIZE_16), runInfo_.baseM * runInfo_.stepM));
        tilingData.nL1 = static_cast<uint32_t>(
            std::min(ops::CeilAlign(args_.nValue, BASIC_BLOCK_SIZE_16), runInfo_.baseN * runInfo_.stepN));
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
        tilingData.mmadParam = supportMmadS8S4_ ? static_cast<uint8_t>(shiftValue_) : args_.isHf32;
        tilingData.l1BufferNum = static_cast<uint8_t>(runInfo_.l1BufferNum);
        tilingData.l0cDB = static_cast<uint8_t>(runInfo_.dbL0C);
        tilingData.ubDB = static_cast<uint8_t>(runInfo_.mixInfo.ubDB);
        tilingData.mBaseTailSplitCnt = runInfo_.mBaseTailSplitCnt;
        tilingData.nBaseTailSplitCnt = runInfo_.nBaseTailSplitCnt;
        tilingData.mTailMain = runInfo_.tailInfo.mTailMain;
        tilingData.nTailMain = runInfo_.tailInfo.nTailMain;
        tilingData.l2CacheDisable = SetDisableL2cache(tilingData.mL1, tilingData.kL1, tilingData.kL1, tilingData.nL1);
        auto selfInputShape = context_->GetInputShape(0);
        auto mat2InputShape = context_->GetInputShape(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, selfInputShape);
        OP_CHECK_NULL_WITH_CONTEXT(context_, mat2InputShape);
        auto selfViewShape = selfInputShape->GetOriginShape();
        auto mat2Shape = mat2InputShape->GetOriginShape();
        auto selfStorageShape = selfInputShape->GetStorageShape();
        // 非连续Slice校验
        // TensorV2 & 3d && storageShape 1d
        if (context_->InputIsView(0) && selfViewShape.GetDimNum() == 3 && mat2Shape.GetDimNum() == 2 &&
            selfStorageShape.GetDimNum() == 1) {
            auto selfViewStride = context_->GetInputStride(0);
            tilingData.sliceM = static_cast<uint32_t>(selfViewShape[1]); // sliceM=self[1], ndNum = baseM/sliceM
            tilingData.srcNdStride = static_cast<uint32_t>(selfViewStride->GetStride(0)); // oriM * srcK
        } else {
            tilingData.sliceM = runInfo_.baseM;
            tilingData.srcNdStride = 1;
        }
        tilingData.innerBatch = runInfo_.innerBatch;
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingDataProcess(BatchMatMulV3IterBatchBasicTilingData& iterbatchTilingBasicData) const
    {
        iterbatchTilingBasicData.m = args_.mValue;
        iterbatchTilingBasicData.n = args_.nValue;
        iterbatchTilingBasicData.k = args_.kValue;
        iterbatchTilingBasicData.b = batchInfo_->batchC;
        iterbatchTilingBasicData.iterBatchL1 = runInfo_.iterBatchL1;
        iterbatchTilingBasicData.iterBatchL0 = runInfo_.iterBatchL0;
        iterbatchTilingBasicData.mmadParam = supportMmadS8S4_ ? static_cast<uint8_t>(shiftValue_) : args_.isHf32;
        iterbatchTilingBasicData.baseM = runInfo_.baseM;
        iterbatchTilingBasicData.baseN = runInfo_.baseN;
        iterbatchTilingBasicData.baseK = runInfo_.baseK;
        iterbatchTilingBasicData.innerBatch = runInfo_.innerBatch;
        iterbatchTilingBasicData.batchX3 = args_.batchX3;
        iterbatchTilingBasicData.needNdDma = runInfo_.needNdDma;
        iterbatchTilingBasicData.l2CacheDisable = SetDisableL2cache(args_.mValue, args_.kValue, args_.kValue,
                                                                    args_.nValue);
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingDataProcess(
        BatchMatMulV3MergeBatchBasicTilingData& mergebatchTilingBasicData) const
    {
        mergebatchTilingBasicData.m = args_.mValue;
        mergebatchTilingBasicData.n = args_.nValue;
        mergebatchTilingBasicData.k = args_.kValue;
        mergebatchTilingBasicData.b = batchInfo_->batchC;
        mergebatchTilingBasicData.batchAL1 = runInfo_.mergeBatchAL1;
        mergebatchTilingBasicData.batchBL1 = runInfo_.mergeBatchBL1;
        mergebatchTilingBasicData.batchL0 = runInfo_.mergeBatchL0;
        mergebatchTilingBasicData.baseK = runInfo_.baseK;
        mergebatchTilingBasicData.kL1 = runInfo_.stepKa * runInfo_.baseK;
        mergebatchTilingBasicData.mmadParam = supportMmadS8S4_ ? static_cast<uint8_t>(shiftValue_) : args_.isHf32;
        mergebatchTilingBasicData.batchX3 = args_.batchX3;
        mergebatchTilingBasicData.l2CacheDisable = SetDisableL2cache(args_.mValue, mergebatchTilingBasicData.kL1,
                                                                     mergebatchTilingBasicData.kL1, args_.nValue);
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingDataProcess(BatchMatMulToMulBasicTilingData& bmmToMulBasicData) const
    {
        bmmToMulBasicData.m = runInfo_.bmmToMulInfo.m;
        bmmToMulBasicData.n = runInfo_.bmmToMulInfo.n;
        bmmToMulBasicData.b = runInfo_.bmmToMulInfo.b;
        bmmToMulBasicData.usedCoreNum = runInfo_.bmmToMulInfo.usedCoreNum;
        bmmToMulBasicData.singleCoreBatch = runInfo_.bmmToMulInfo.singleCoreBatch;
        bmmToMulBasicData.batchNum = runInfo_.bmmToMulInfo.batchNum;
        bmmToMulBasicData.batchNumLastRound = runInfo_.bmmToMulInfo.batchNumLastRound;
        bmmToMulBasicData.batchNumLastRoundTail = runInfo_.bmmToMulInfo.batchNumLastRoundTail;
        bmmToMulBasicData.lastCoreNum = runInfo_.bmmToMulInfo.lastCoreNum;
        bmmToMulBasicData.alignNum = runInfo_.bmmToMulInfo.alignNum;
        return ge::GRAPH_SUCCESS;
    };

    virtual ge::graphStatus GetTilingDataProcess(MatMulV3KEqZeroBasicTilingData& kEqZeroBasicTilingData) const
    {
        kEqZeroBasicTilingData.totalDataAmount = runInfo_.totalDataAmount;
        kEqZeroBasicTilingData.aivNum = runInfo_.usedCoreNum;
        return ge::GRAPH_SUCCESS;
    }

    virtual ge::graphStatus GetTilingDataProcess(MatMulToMulBasicTilingData& matmulToMulBasicData) const
    {
        matmulToMulBasicData.usedCoreNum = runInfo_.usedCoreNum;
        matmulToMulBasicData.tileNum = runInfo_.mmToMulInfo.tileNum;
        matmulToMulBasicData.m = args_.mValue;
        matmulToMulBasicData.n = args_.nValue;
        matmulToMulBasicData.k = args_.kValue;
        matmulToMulBasicData.baseMN = runInfo_.mmToMulInfo.baseMN;
        matmulToMulBasicData.tailMN = runInfo_.mmToMulInfo.tailMN;
        matmulToMulBasicData.baseK = runInfo_.mmToMulInfo.baseK;
        matmulToMulBasicData.tailK = runInfo_.mmToMulInfo.tailK;
        matmulToMulBasicData.loopK = runInfo_.mmToMulInfo.loopK;
        matmulToMulBasicData.dataCopyMode = runInfo_.mmToMulInfo.dataCopyMode;
        return ge::GRAPH_SUCCESS;
    }

    virtual ge::graphStatus GetTilingDataProcess(MatMulToVectorBasicTilingData& matmulToVectorBasicData) const
    {
        matmulToVectorBasicData.usedCoreNum = runInfo_.usedCoreNum;
        matmulToVectorBasicData.m = args_.mValue;
        matmulToVectorBasicData.n = args_.nValue;
        matmulToVectorBasicData.k = args_.kValue;
        matmulToVectorBasicData.baseM = runInfo_.baseM;
        matmulToVectorBasicData.baseN = runInfo_.baseN;
        matmulToVectorBasicData.baseK = runInfo_.baseK;
        return ge::GRAPH_SUCCESS;
    }

protected:
    const MatmulV3CompileInfo& compileInfo_;
    const MatMulV3Args& args_;
    const MatMulV3BatchInfo* batchInfo_;
    MatMulV3RunInfo runInfo_;
    MatMulV3TilingKey* tilingKeyObj;

private:
    bool enableUncache_ = false;
    static constexpr int64_t ORI_SHIFT_VALUE = 42; // 未设置shiftValue或设置为0，则使用默认值进行计算
    int64_t shiftValue_ = 0;
    bool supportMmadS8S4_ = false;
    const std::map<ge::DataType, matmul_tiling::DataType> dtypeMap_ = {
        {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
        {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
        {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
        {ge::DT_FLOAT8_E5M2, matmul_tiling::DataType::DT_FLOAT8_E5M2},
        {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_FLOAT8_E4M3FN},
        {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_HIFLOAT8},
        {ge::DT_INT8, matmul_tiling::DataType::DT_INT8}};
};
} // namespace matmul_v3_advanced
} // namespace optiling
