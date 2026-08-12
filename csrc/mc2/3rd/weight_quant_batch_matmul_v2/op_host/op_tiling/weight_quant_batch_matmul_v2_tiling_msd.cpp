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
 * \file weight_quant_batch_matmul_v2_tiling_msd.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_msd.h"

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_white_list.h"
#include "weight_quant_batch_matmul_v2_tiling_key.h"
#include "op_host/tiling_key.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "ops_legacy/op_tiling/op_cache_def_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

using Ops::Transformer::OpTiling::RecursiveSum;

namespace optiling {

constexpr uint64_t MSD_M_LIMIT = 64UL;
constexpr uint64_t MSD_PERCHANNEL_MAX_K = 13696UL;
constexpr uint64_t MSD_PERCHANNEL_MAX_N = 32000UL;
constexpr uint64_t INT4_BLK_SIZE = 64UL;
constexpr uint64_t DB_BUFFER = 2UL;

const std::map<Mc2WhiteListShape, uint32_t> MM_PRELOAD_TIME_MAP = {
    {{1, 5568, 6656, false, false, true, 1}, 1},  {{1, 8192, 3072, false, false, true, 1}, 3},
    {{1, 8192, 6144, false, false, true, 1}, 3},  {{1, 1024, 8192, false, false, true, 1}, 3},
    {{16, 8192, 5504, false, false, true, 1}, 3}, {{16, 8192, 7168, false, false, true, 1}, 2}};

const std::set<Mc2WhiteListShape> MSD_HIGH_PRECISION_LIST = {
    // llama3-70B
    {36, 8192, 1280, false, false, true, 1},
};

void Mc2WeightQuantBatchMatmulV2Msd::Reset()
{
    Mc2WeightQuantBatchMatmulV2Tiling::Reset();
    splitKFlag_ = false;
    highPrecision_ = false;

    OP_TILING_CHECK(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                             context_->GetRawTilingData()->GetCapacity()) != EOK,
                    OP_LOGE(opName_, "fail to memset tiling data"), return;);
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2MsdTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    // 设置 block dim
    uint32_t usedAicNum = tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN;
    uint32_t usedAivNum = usedAicNum * 2;
    uint32_t blkDim = 0;
    blkDim = CalcTschBlockDim(std::max(usedAivNum, blkDim_), compileInfoPtr_->aicNum, compileInfoPtr_->aivNum);
    context_->SetBlockDim(blkDim);
    OP_LOGD(opName_, "set blkDim %d", blkDim);
    errno_t ret =
        memcpy_s(context_->GetTilingData<Mc2WeightQuantBatchMatmulV2MsdTilingData>(),
                 context_->GetRawTilingData()->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

/*
The function is limite of msd
1. not trans_a, c_dtype!=int8, antiquantscale_dtype!=uint64
2. perchannel:
   1) m <= 64, k % 32=0, n % 32=0
   2) int4: trans_b, ND format, n >= 2*m
   3) int8: not (splitK and NZ format) when splitK = k > 13696 && n > 32000
3. pergroup:
   1) int4, not trans_b
   2) groupsize: 64 or 128
   3) m <= groupsize/8, k % groupsize =0, n % 64 = 0
*/
bool Mc2WeightQuantBatchMatmulV2Msd::IsCapable()
{
    OP_LOGI(opName_, "Begin check msd");
    OP_TILING_CHECK(matmulInfoPtr_->transA || matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64 ||
                        matmulInfoPtr_->cDtype == ge::DT_INT8 || matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64 ||
                        (matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_CHANNEL &&
                         matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_GROUP),
                    OP_LOGI(opName_, "MSD not support trans_a, quant, int64 antiquant or pertsor"), return false);
    if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_CHANNEL) {
        OP_TILING_CHECK(
            matmulInfoPtr_->mSize > MSD_M_LIMIT || matmulInfoPtr_->kSize % 32 != 0 || matmulInfoPtr_->nSize % 32 != 0,
            OP_LOGI(opName_, "Perchannel must m <= 64, k and n align to 32, while m, k, n are [%lu], [%lu] and [%lu]",
                    matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize),
            return false);
        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            OP_TILING_CHECK(!matmulInfoPtr_->transB || matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ,
                            OP_LOGI(opName_, "Perchannel int4 must trans_b or WeightND"), return false);
            OP_TILING_CHECK(matmulInfoPtr_->nSize < 2 * matmulInfoPtr_->mSize,
                            OP_LOGI(opName_, "Perchannel int4 must n >= 2*m, while m and n are [%lu] and [%lu]",
                                    matmulInfoPtr_->mSize, matmulInfoPtr_->nSize),
                            return false);
            // Expansion times of w4 is 3
            order_ = 3;
            splitKFlag_ = true;
        } else {
            splitKFlag_ = matmulInfoPtr_->kSize > MSD_PERCHANNEL_MAX_K || matmulInfoPtr_->nSize > MSD_PERCHANNEL_MAX_N;
            Mc2WhiteListShape shape({matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
                                     matmulInfoPtr_->hasBias, matmulInfoPtr_->transA, matmulInfoPtr_->transB, 1});
            if (MSD_HIGH_PRECISION_LIST.find(shape) != MSD_HIGH_PRECISION_LIST.end()) {
                OP_LOGI(opName_, "The case matched msd high precison");
                highPrecision_ = true;
            }
        }
        OP_LOGI(opName_, "Check msd fo perchannel succ");
    }
    if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP) {
        OP_TILING_CHECK(matmulInfoPtr_->transB || matmulInfoPtr_->bDtype != ge::DT_INT4,
                        OP_LOGI(opName_, "Pergroup not support trans_b or W8"), return false);

        OP_TILING_CHECK(
            (matmulInfoPtr_->groupSize != 64 && matmulInfoPtr_->groupSize != 128) ||
                matmulInfoPtr_->mSize > matmulInfoPtr_->groupSize / 8 ||
                matmulInfoPtr_->kSize % matmulInfoPtr_->groupSize != 0 || matmulInfoPtr_->nSize % 64 != 0,
            OP_LOGI(opName_,
                    "Pergroup must groupsize is 64/128, m <= groupsize/8, k align to groupsize, n align to 64 "
                    "while m, k, n and groupsize are [%lu], [%lu], [%lu] and [%lu]",
                    matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->groupSize),
            return false);
        // Expansion times of w4 is 3
        order_ = 3;
        splitKFlag_ = true;
        OP_LOGI(opName_, "Check msd fo pergroup succ");
    }
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2MsdTilingData);
    if (tilingData_ == nullptr) {
        tilingData_ = std::unique_ptr<Mc2WeightQuantBatchMatmulV2MsdTilingData>(
            new (std::nothrow) Mc2WeightQuantBatchMatmulV2MsdTilingData());
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->hasBias = matmulInfoPtr_->hasBias;

    if (!splitKFlag_) {
        tilingData_->v1BaseK = matmulInfoPtr_->kSize;
        tilingData_->v1BaseM = 1;
        OP_TILING_CHECK(!GetMatMulTiling(),
                        OP_LOGE(opName_, "failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
                                matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
                        return ge::GRAPH_FAILED);
        uint32_t preloadTimes = 3;
        uint32_t bL1KSize = tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.stepKb;
        preloadTimes = std::min(preloadTimes, static_cast<uint32_t>(matmulInfoPtr_->kSize / bL1KSize));
        Mc2WhiteListShape shape({matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
                                 matmulInfoPtr_->hasBias, matmulInfoPtr_->transA, matmulInfoPtr_->transB, 1});
        auto it = MM_PRELOAD_TIME_MAP.find(shape);
        if (it != MM_PRELOAD_TIME_MAP.end()) {
            preloadTimes = it->second;
        }
        tilingData_->preloadTimes = preloadTimes;
        uint32_t preprocessUsedAivNum = matmulInfoPtr_->mSize;
        if (preprocessUsedAivNum > compileInfoPtr_->aivNum) {
            OP_LOGE_IF(compileInfoPtr_->aivNum == 0, ge::GRAPH_FAILED, context_->GetNodeName(), "aivNum is 0");
            uint32_t divNum =
                (matmulInfoPtr_->mSize + compileInfoPtr_->aivNum - 1) / compileInfoPtr_->aivNum; // 计算需要分几份
            preprocessUsedAivNum = (matmulInfoPtr_->mSize + divNum - 1) / divNum;                // 需要用几个 core
        }
        tilingData_->preProcessUsedVecNum = preprocessUsedAivNum;
        blkDim_ = preprocessUsedAivNum;
    } else {
        if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP) {
            return DoMSDGroupSplitKOpTiling();
        }
        return DoMSDGeneralOpTiling();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::DoMSDGeneralOpTiling()
{
    uint64_t kBlockNum = 1;
    // nk差距超过30倍，需要考虑对k切多份, 该值为经验值
    if (matmulInfoPtr_->kSize / matmulInfoPtr_->nSize >= 30 && matmulInfoPtr_->transB == false) {
        // 根据经验，当weight不转置时，k切4份及以上，可以增加n方向并行度以提高性能
        kBlockNum = 4;
    }
    // 65535最大规格满足切分6份，继续增加切分份数导致后处理数据量过多
    for (; kBlockNum <= 6; kBlockNum++) {
        uint64_t singleCoreK = SplitKByKBlock(kBlockNum);
        // 单core容忍的k范围有限，根据ub切分，最大支持的规格为12 * 1024
        if (singleCoreK <= 0 || singleCoreK > 12 * 1024) {
            continue;
        }
        // 前处理基本块大小为12*1024
        uint64_t v1BaseM = 12 * 1024 / singleCoreK;
        // 根据workspace空间反算N轴的切分，
        // C1C2在workspace上多份，避免同步开销引起的性能裂化
        uint64_t singleNSize = 16 * 1024 * 1024 / (kBlockNum * order_ * matmulInfoPtr_->mSize * sizeof(int32_t));
        singleNSize = singleNSize / 256 * 256; // 向下对齐到256，保证非尾块处理效率
        // vec一次处理的标准块是64*128。按照n=128划分n方向计算一轮cube的n最大切分
        uint64_t aivBaseN = 128;
        if (singleNSize > aivBaseN * compileInfoPtr_->aivNum) {
            singleNSize = aivBaseN * compileInfoPtr_->aivNum;
        }
        if (singleNSize > matmulInfoPtr_->nSize) {
            singleNSize = matmulInfoPtr_->nSize;
        }
        uint64_t singleCoreNSize = ops::CeilAlign(
            ops::CeilDiv(singleNSize, static_cast<uint64_t>(compileInfoPtr_->aivNum)), static_cast<uint64_t>(aivBaseN));

        // 后处理的n方向切分数量
        uint64_t postProcessNBlockNum = singleCoreNSize > singleNSize ? 1 : ops::CeilDiv(singleNSize, singleCoreNSize);
        uint64_t postProcessMBlockNum = compileInfoPtr_->aivNum / postProcessNBlockNum;
        uint64_t postProcessSingleCoreM = ops::CeilDiv(matmulInfoPtr_->mSize, postProcessMBlockNum);
        if (postProcessSingleCoreM <= 0) {
            continue;
        }
        uint64_t postProcessBaseM = std::min(32 * 256 / singleCoreNSize, postProcessSingleCoreM);
        tilingData_->v1BaseK = singleCoreK;
        // 前处理一次最多处理8行数据
        tilingData_->v1BaseM = std::min(8UL, v1BaseM);
        tilingData_->taskNSize = singleNSize;
        tilingData_->taskSingleCoreNSize = singleCoreNSize;
        tilingData_->postProcessSingleCoreM = postProcessSingleCoreM;
        tilingData_->postProcessBaseM = postProcessBaseM;

        OP_TILING_CHECK(!GetMatMulTiling(),
                        OP_LOGE(opName_, "failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
                                matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
                        return ge::GRAPH_FAILED);
        blkDim_ = std::max(
            std::min(kBlockNum * ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->v1BaseM)),
                     static_cast<uint64_t>(compileInfoPtr_->aivNum)),
            postProcessMBlockNum * postProcessNBlockNum);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

uint64_t Mc2WeightQuantBatchMatmulV2Msd::SplitKByKBlock(uint64_t kBlockNum) const
{
    // 默认k切分粒度希望尽量小，使分核均匀
    uint64_t kAlignSize = 64;
    if (matmulInfoPtr_->transB) {
        // 转置场景优先保证cache line，使mte2更好
        kAlignSize = matmulInfoPtr_->bDtype == ge::DT_INT4 ? 1024UL : 512UL;
    }
    return kBlockNum == 0 ? 0 : ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->kSize, kBlockNum), kAlignSize);
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::DoMSDGroupSplitKOpTiling()
{
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    uint64_t kBlockNum = 1;
    uint64_t v1BaseM = 1;
    // 确保groupPack * matmulInfoPtr_->groupSize的值为1024
    uint64_t groupPack = matmulInfoPtr_->groupSize == 128 ? 8 : 16;
    uint64_t v1BaseK =
        ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->kSize, kBlockNum), groupPack * matmulInfoPtr_->groupSize);
    // 65535最大规格满足切分6份，继续增加切分份数导致后处理数据量过多，根据ub切分，最大支持的规格为12
    // * 1024
    for (; kBlockNum <= 6 && v1BaseK * v1BaseM > 12 * 1024; kBlockNum++) {
        v1BaseK = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->kSize, kBlockNum), groupPack * matmulInfoPtr_->groupSize);
    }

    // cube N方向切分固定位128
    uint64_t singleCoreNSize = 128;
    uint64_t singleNSize = std::min(singleCoreNSize * compileInfoPtr_->aivNum, matmulInfoPtr_->nSize);

    // 后处理
    uint64_t postProcessNBlockNum = ops::CeilDiv(singleNSize, singleCoreNSize);
    uint64_t postProcessMBlockNum = std::min(compileInfoPtr_->aivNum / postProcessNBlockNum,
                                             matmulInfoPtr_->mSize); // 避免m方向上多分核
    uint64_t postProcessSingleCoreM = ops::CeilDiv(matmulInfoPtr_->mSize, postProcessMBlockNum);

    // 后处理buffer分配的内存为32k，数据类型为int32，所以单次处理的数据,32 * 256
    uint64_t postProcessBaseM = std::min(32 * 256 / singleCoreNSize, postProcessSingleCoreM);
    tilingData_->v1BaseK = v1BaseK;
    tilingData_->v1BaseM = v1BaseM;
    tilingData_->taskNSize = singleNSize;
    tilingData_->taskSingleCoreNSize = singleCoreNSize;
    tilingData_->postProcessSingleCoreM = postProcessSingleCoreM;
    tilingData_->postProcessBaseM = postProcessBaseM;
    tilingData_->groupPack = groupPack;

    OP_TILING_CHECK(!GetMatMulTiling(),
                    OP_LOGE(opName_, "failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
                            matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
                    return ge::GRAPH_FAILED);
    // cube开db，需要乘以2
    tilingData_->matmulTiling.depthB1 = DB_BUFFER * tilingData_->matmulTiling.stepKb * tilingData_->matmulTiling.stepN;
    blkDim_ =
        std::max(std::min(kBlockNum * ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->v1BaseM)),
                          static_cast<uint64_t>(compileInfoPtr_->aivNum)),
                 postProcessMBlockNum * postProcessNBlockNum);
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2WeightQuantBatchMatmulV2Msd::GetInnerPreciseTilingKey() const
{
    Mc2TilingKeyConfigure tilingKeyConfigure;
    SetCommonTilingKeyElement(tilingKeyConfigure);

    uint64_t socVersionType = tilingKeyConfigure.socVersionType / 10UL;
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = tilingKeyConfigure.quantizationScenario;
    uint64_t algorithm = static_cast<uint64_t>(
        Mc2OptimizationAlgorithmCategory::MULTI_SCALE_DEQUANT); // 3 means CUSTOM tilingkey algorithm
    uint64_t subAlgorithm = static_cast<uint64_t>(Mc2OptimizationAlgorithmSubCategory::SPLIT_K);
    uint64_t subAlgorithmCustom = 0UL;
    uint64_t innerPrecise = matmulInfoPtr_->innerPrecise;
    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;
    bool transA = ((tilingKeyConfigure.transposeSituation >> 1) & 1) != 0;
    bool transB = (tilingKeyConfigure.transposeSituation & 1) != 0;
    uint64_t antiquantType = tilingKeyConfigure.antiquantType;
    uint64_t quantType = tilingKeyConfigure.quantType;
    bool hasAntiquantOffset = ((tilingKeyConfigure.optionInputSituation >> 1) & 1) != 0;
    bool hasBias = false;
    bool isBiasFp32 = false;
    bool isWeightNz = (tilingKeyConfigure.weightFormat == 1UL) ? true : false; // Mc2WeightFormat::ND
    uint64_t templateExtra = 3UL;                                              // 3 means TEMPLATE_EXTRA_NOT_USED
    uint64_t fullLoadMode = 5UL;                                               // 5 means FULL_LOAD_MODE_NOT_USED
    uint64_t batch = 0UL;
    uint64_t tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                             subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                             apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                             hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
    return tilingKey_;
}

// 4、计算高阶API的TilingData
ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

// 5、计算TilingKey
uint64_t Mc2WeightQuantBatchMatmulV2Msd::GetTilingKey() const
{
    // 在A16W4 pergroup切K 下才有效
    if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP && splitKFlag_ &&
        matmulInfoPtr_->bDtype == ge::DT_INT4 &&
        (matmulInfoPtr_->innerPrecise != 0 || matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ)) {
        return GetInnerPreciseTilingKey();
    }
    uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = 0UL;
    uint64_t algorithm = 3UL; // 3 means CUSTOM tilingkey algorithm
    uint64_t subAlgorithm = 0UL;
    uint64_t subAlgorithmCustom = static_cast<uint64_t>(Mc2KernelTemplateType::MSD_MULTI_CORE);
    uint64_t innerPrecise = 0UL;
    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = static_cast<uint64_t>(matmulInfoPtr_->quantType);
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = false;
    bool isBiasFp32 = false;
    bool isWeightNz;
    uint64_t templateExtra;
    uint64_t fullLoadMode = 5UL; // 5 means FULL_LOAD_MODE_NOT_USED
    uint64_t batch = 0UL;
    uint64_t tilingKey_;

    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        if (highPrecision_) {
            isWeightNz = true; // Mc2KernelTemplateType::WEIGHT_NZ
            templateExtra = static_cast<uint64_t>(Mc2KernelTemplateTypeExtra::HIGH_PRECISION);
            tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                            apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                            hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        } else {
            if (splitKFlag_) {
                isWeightNz = true; // Mc2KernelTemplateType::WEIGHT_NZ
                templateExtra = static_cast<uint64_t>(Mc2KernelTemplateTypeExtra::MSD_GENERAL);
                tilingKey_ = GET_TPL_TILING_KEY(
                    socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm, subAlgorithmCustom,
                    innerPrecise, templateCustom, apiConstexpr, transA, transB, antiquantType, quantType,
                    hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
                return tilingKey_;
            }
            isWeightNz = true;   // Mc2KernelTemplateType::WEIGHT_NZ
            templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
            tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                            apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                            hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        }
    }

    if (splitKFlag_) {
        isWeightNz = false; // Mc2WeightFormat::ND
        templateExtra = static_cast<uint64_t>(Mc2KernelTemplateTypeExtra::MSD_GENERAL);
        tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm, subAlgorithm,
                                        subAlgorithmCustom, innerPrecise, templateCustom, apiConstexpr, transA, transB,
                                        antiquantType, quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz,
                                        templateExtra, fullLoadMode, batch);
        return tilingKey_;
    } else {
        if (highPrecision_) {
            isWeightNz = false; // Mc2WeightFormat::ND
            templateExtra = static_cast<uint64_t>(Mc2KernelTemplateTypeExtra::HIGH_PRECISION);
            tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                            apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                            hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        } else {
            isWeightNz = false;
            templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
            tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                            apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                            hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
            return tilingKey_;
        }
    }
}

// 6、计算Workspace 大小
ge::graphStatus Mc2WeightQuantBatchMatmulV2Msd::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "failed to get workspace size"), return ge::GRAPH_FAILED);
    workspaces[0] = 64 * 1024 * 1024; // workspace 固定使用 64 * 1024 * 1024
    return ge::GRAPH_SUCCESS;
}

bool Mc2WeightQuantBatchMatmulV2Msd::CheckCacheTiling()
{
    if (tilingData_->cubeBlockDimM != 1) {
        // 切k模板未实现m分核逻辑，导致性能裂化。mult core模板不支持m分核场景
        OP_LOGW(opName_, "cubeBlockDimM must be 1");
        return false;
    }
    // 为了单独适配 multi core 下，transW 对齐了 Kb 的场景
    if (tilingData_->matmulTiling.Ka != tilingData_->matmulTiling.Kb) {
        OP_LOGW(opName_, "param Ka Kb in matmulTiling should be equal");
        return false;
    }

    int kaStepIter = ops::CeilDiv(tilingData_->matmulTiling.singleCoreK,
                                  tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.stepKa);
    int kbStepIter = ops::CeilDiv(tilingData_->matmulTiling.singleCoreK,
                                  tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.stepKb);
    OP_TILING_CHECK(!splitKFlag_ && kaStepIter % kbStepIter != 0 && kbStepIter % kaStepIter != 0,
                    OP_LOGW(opName_,
                            "(kaStepIter %% kbStepIter) or (kbStepIter %% kaStepIter) should "
                            "be 0. kaStepIter(%d) kbStepIter(%d)",
                            kaStepIter, kbStepIter),
                    return false);

    OP_TILING_CHECK(
        tilingData_->matmulTiling.singleCoreM % order_ != 0,
        OP_LOGW(opName_, "singleCoreM must %%  %d = 0, actual is %d", order_, tilingData_->matmulTiling.singleCoreM),
        return false);

    // int8场景：ND-NK 和 NZ-KN singleCoreN允许产生非256对齐的解; int4场景：singleCoreN允许产生非256对齐的解
    OP_TILING_CHECK(
        !CheckInt8MatmulTiling(tilingData_->matmulTiling.singleCoreN),
        OP_LOGW(opName_, "singleCoreN must %% 256 = 0, actual is %d", tilingData_->matmulTiling.singleCoreN),
        return false);

    OP_TILING_CHECK(!CheckInt4MatmulTiling(), OP_LOGW(opName_, "in int4 scenario, msd matmul tiling check failed"),
                    return false);
    OP_LOGD(opName_, "get and convert cache tiling success");
    return true;
}

bool Mc2WeightQuantBatchMatmulV2Msd::CheckInt8MatmulTiling(uint64_t singleCoreNCalc) const
{
    if (matmulInfoPtr_->bDtype != ge::DT_INT8) {
        return true;
    }

    if ((matmulInfoPtr_->transB) && (matmulInfoPtr_->bFormat == ge::FORMAT_ND)) {
        return true;
    }
    if ((!matmulInfoPtr_->transB) && (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ)) {
        return true;
    }

    // 部分场景kernel只支持singleCoreN为256对齐，否则会有精度问题
    if ((singleCoreNCalc % 256) != 0) {
        return false;
    }

    return true;
}

bool Mc2WeightQuantBatchMatmulV2Msd::CheckInt4MatmulTiling() const
{
    if (matmulInfoPtr_->bDtype != ge::DT_INT4) {
        return true;
    }

    // b转置时，baseK应大于64
    OP_TILING_CHECK(static_cast<uint64_t>(tilingData_->matmulTiling.baseK) < INT4_BLK_SIZE,
                    OP_LOGW(opName_,
                            "in int4 scenario, baseK must greater than %lu. baseK:[%d], "
                            "matmulInfoPtr_->bDtype:[%s]",
                            INT4_BLK_SIZE, tilingData_->matmulTiling.baseK,
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString()),
                    return false);

    // b不转置时，baseN应大于64
    OP_TILING_CHECK(!matmulInfoPtr_->transB && static_cast<uint64_t>(tilingData_->matmulTiling.baseN) < INT4_BLK_SIZE,
                    OP_LOGW(opName_,
                            "in int4 scenario, baseN must greater than %lu, baseM:[%d], "
                            "matmulInfoPtr_->bDtype:[%s], transB[%s]",
                            INT4_BLK_SIZE, tilingData_->matmulTiling.baseN,
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString(),
                            matmulInfoPtr_->transB ? "true" : "false"),
                    return false);

    // int4场景切分的内轴应该为偶数
    OP_TILING_CHECK(matmulInfoPtr_->transB && (tilingData_->matmulTiling.singleCoreK & 1) != 0,
                    OP_LOGW(opName_,
                            "in int4 scenario, singleCoreK must %% 2 = 0, singleCoreK:[%d], "
                            "matmulInfoPtr_->bDtype:[%s], transB[%s]",
                            tilingData_->matmulTiling.singleCoreK,
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString(),
                            matmulInfoPtr_->transB ? "true" : "false"),
                    return false);

    OP_TILING_CHECK(!matmulInfoPtr_->transB && (tilingData_->matmulTiling.singleCoreN & 1) != 0,
                    OP_LOGW(opName_,
                            "in int4 scenario, singleCoreN must %% 2 = 0, singleCoreN:[%d], "
                            "matmulInfoPtr_->bDtype:[%s], transB[%s]",
                            tilingData_->matmulTiling.singleCoreN,
                            ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->bDtype).GetString(),
                            matmulInfoPtr_->transB ? "true" : "false"),
                    return false);
    return true;
}

bool Mc2WeightQuantBatchMatmulV2Msd::InvokeCacheTiling()
{
    Mc2MatmulMultiCoreResult multiCoreResult;
    if (!splitKFlag_) {
        bool result = Mc2ComputeMatmulTiling::GetTiling(
            tilingData_->matmulTiling, multiCoreResult,
            {order_ * matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, ge::DT_INT8, ge::DT_INT8,
             ge::DT_INT32, ge::DT_INT32, matmulInfoPtr_->transA, matmulInfoPtr_->transB, false, matmulInfoPtr_->aFormat,
             matmulInfoPtr_->bFormat, ge::FORMAT_ND, matmulInfoPtr_->quantType, true},
            aicoreParams_, context_);
        OP_LOGI_IF_RETURN(!result, false, opName_, "cannot get tiling from cachetiling, mnk[%lu, %lu, %lu]",
                          matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);
    } else {
        bool result = Mc2ComputeMatmulTiling::GetTiling(
            tilingData_->matmulTiling, multiCoreResult,
            {order_ * matmulInfoPtr_->mSize,
             std::min(matmulInfoPtr_->kSize, static_cast<uint64_t>(tilingData_->v1BaseK)),
             std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->taskNSize)), matmulInfoPtr_->bDtype,
             matmulInfoPtr_->bDtype, ge::DT_INT32, ge::DT_INT32, matmulInfoPtr_->transA, matmulInfoPtr_->transB, false,
             matmulInfoPtr_->aFormat, matmulInfoPtr_->bFormat, ge::FORMAT_ND, matmulInfoPtr_->quantType, false},
            aicoreParams_, context_);
        OP_LOGI_IF_RETURN(!result, false, opName_, "cannot get tiling from cachetiling, mnk[%lu, %lu, %u]",
                          matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, tilingData_->v1BaseK);
    }

    tilingData_->cubeBlockDimM = static_cast<uint8_t>(multiCoreResult.mDim);
    tilingData_->cubeBlockDimN = static_cast<uint8_t>(multiCoreResult.nDim);

    return CheckCacheTiling();
}

bool Mc2WeightQuantBatchMatmulV2Msd::GetMatMulTiling()
{
    if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP || (!GetTilingFromCache() && !InvokeCacheTiling())) {
        uint64_t mSize = order_ * matmulInfoPtr_->mSize;
        uint64_t nSize = matmulInfoPtr_->nSize;
        uint64_t kSize = matmulInfoPtr_->kSize;
        uint32_t cubeSingleCoreN;
        if (splitKFlag_) {
            nSize = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->taskNSize));
            kSize = std::min(matmulInfoPtr_->kSize, static_cast<uint64_t>(tilingData_->v1BaseK));
        }
        uint32_t maxCubeSingleCoreN = ops::FloorDiv(nSize, static_cast<uint64_t>(compileInfoPtr_->aicNum));
        if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ && !matmulInfoPtr_->transB) {
            // 为保证带宽利用率和mmad效率，设置singleCoreN的对齐基数为256
            cubeSingleCoreN = ops::CeilAlign(maxCubeSingleCoreN, static_cast<uint32_t>(256));
        } else {
            // 为保证带宽利用率和mmad效率，设置singleCoreN的对齐基数为32
            cubeSingleCoreN = ops::CeilAlign(maxCubeSingleCoreN, static_cast<uint32_t>(32));
        }
        tilingData_->cubeBlockDimM = static_cast<uint8_t>(1);
        tilingData_->cubeBlockDimN = static_cast<uint8_t>(ops::CeilDiv(nSize, static_cast<uint64_t>(cubeSingleCoreN)));
        matmul_tiling::CubeFormat bCube = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ ?
                                              matmul_tiling::CubeFormat::NZ :
                                              matmul_tiling::CubeFormat::ND;
        matmul_tiling::MatmulApiTiling mmTiling;
        mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                          Mc2GetMatmulTilingDtype(matmulInfoPtr_->bDtype), matmulInfoPtr_->transA);
        mmTiling.SetBType(matmul_tiling::TPosition::GM, bCube, Mc2GetMatmulTilingDtype(matmulInfoPtr_->bDtype),
                          matmulInfoPtr_->transB);
        mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                          matmul_tiling::DataType::DT_INT32);
        mmTiling.SetBias(false);
        if (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP) {
            mmTiling.SetOrgShape(order_ * matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);
            // N方向cube上切分SingleCoreN固定为1024
            mmTiling.SetShape(order_ * matmulInfoPtr_->mSize, 1024, matmulInfoPtr_->groupSize);
            mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
            mmTiling.SetFixSplit(ops::CeilAlign(static_cast<uint32_t>(order_ * matmulInfoPtr_->mSize), BLOCK_CUBE),
                                 BASIC_BLOCK, matmulInfoPtr_->groupSize);
        } else {
            mmTiling.SetOrgShape(mSize, nSize, kSize);
            mmTiling.SetShape(mSize, cubeSingleCoreN, kSize);
            mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        }
        OP_TILING_CHECK(mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
                        OP_LOGE(opName_, "failed to get matmul tiling"), return false);
    }
    if (splitKFlag_) {
        ReviseMMTiling();
    }
    return true;
}

void Mc2WeightQuantBatchMatmulV2Msd::ReviseMMTiling() const
{
    uint64_t stepKb = tilingData_->matmulTiling.stepKb;
    uint64_t stepKa = tilingData_->matmulTiling.stepKa;
    if (stepKa < stepKb) {
        stepKb = stepKa;
    }

    // stepKb大于4的导致scalar阻塞
    if (stepKb > 4) {
        stepKb = stepKb >> 1;
    }
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        uint32_t baseK = tilingData_->matmulTiling.baseK;
        tilingData_->matmulTiling.baseK = ops::FloorAlign(static_cast<uint64_t>(baseK), INT4_BLK_SIZE);
        if (!matmulInfoPtr_->transB) {
            tilingData_->matmulTiling.baseN =
                ops::FloorAlign(static_cast<uint64_t>(tilingData_->matmulTiling.baseN), INT4_BLK_SIZE);
        }
    }
    tilingData_->matmulTiling.Ka = tilingData_->v1BaseK;
    tilingData_->matmulTiling.stepKb = stepKb;
    tilingData_->matmulTiling.shareL1Size = 0;
}

bool Mc2WeightQuantBatchMatmulV2Msd::GetTilingFromCache()
{
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        return false;
    }
    uint64_t mMatchSize = ops::CeilDiv(matmulInfoPtr_->mSize * order_, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t nSize = matmulInfoPtr_->nSize;
    uint64_t kSize = matmulInfoPtr_->kSize;
    if (splitKFlag_) {
        nSize = tilingData_->taskNSize;
        kSize = tilingData_->v1BaseK;
    }

    bool isNzFormat = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ;
    WeightQuantBatchMatmulCacheTilingParas paras({mMatchSize, kSize, nSize, matmulInfoPtr_->hasBias,
                                                  matmulInfoPtr_->transA, matmulInfoPtr_->transB, isNzFormat,
                                                  compileInfoPtr_->aicNum});

    WeightQuantBatchMatmulCacheTilingData matmulTilingCache;
    if (!GenWqbmmTiling(WQBMM_MSD, paras, matmulTilingCache)) {
        OP_LOGD(opName_, "the Msd template not find mm tiling from cache");
        return false;
    }

    OP_LOGD(opName_, "the Msd template get mm tiling from cache");
    Mc2SetMatmulTilingFromCacheData(matmulTilingCache, tilingData_->matmulTiling, matmulInfoPtr_->mSize * order_, nSize,
                                    0);

    tilingData_->cubeBlockDimM = matmulTilingCache.mDim_;
    tilingData_->cubeBlockDimN = matmulTilingCache.nDim_;
    return true;
}
} // namespace optiling
