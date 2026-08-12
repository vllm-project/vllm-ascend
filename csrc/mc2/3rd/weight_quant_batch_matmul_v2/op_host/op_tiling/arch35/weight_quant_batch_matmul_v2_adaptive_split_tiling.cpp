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
 * \file weight_quant_batch_matmul_v2_adaptive_split_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_adaptive_split_tiling.h"

#include <array>

#include "register/op_impl_registry.h"
#include "common/op_host/math_util.h"

using namespace platform_ascendc;

namespace optiling {
namespace Mc2weight_quant_batch_matmul_v2 {
struct CubeSplitResult {
    uint64_t mte2Cost;
    uint64_t cubeBlockDimM;
    uint64_t cubeBlockDimN;
};

constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t QUARTER_BUFFER_NUM = 4;
constexpr uint64_t SINGLE_BUFFER_NUM = 1;
constexpr uint64_t M_MAX_SIZE = 256UL;
constexpr uint64_t L1_N1_MAX_SIZE = 16UL;
constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
constexpr uint64_t M_MAX_SIZE_WITH_BIAS_QUANT = 240UL;
constexpr uint64_t K_L1_SIZE = 256UL;
constexpr uint64_t BIT4_CORRECTION_FACTOR = 2UL;
constexpr uint64_t KILOBYTE = 1024UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint64_t UB_SIZE_V100 = 248UL * 1024UL; // 当前框架获取的UB空间为240KB，问题解决后删除
constexpr uint64_t A_L1_MAX_SIZE = 256UL * 1024UL;
constexpr uint64_t A_L1_MAX_SIZE_WITH_BIAS_QUANT = 240UL * 1024UL;
constexpr uint64_t L1_MAX_SIZE = 512UL * 1024UL;
constexpr uint64_t L1_MAX_SIZE_WITH_BIAS_QUANT = 496UL * 1024UL;
constexpr int32_t ADAPTIVE_SPLIT_PRIORITY = 7;
constexpr std::array<int64_t, 4> SUPPORTED_GROUP_SIZE{32L, 64L, 128L, 256L};

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingAS::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2ASTilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);

    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    // 计算aic num n方向分核*m方向分核
    context_->SetBlockDim(tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN);
    errno_t ret =
        memcpy_s(context_->GetTilingData<Mc2WeightQuantBatchMatmulV2ASTilingData>(),
                 context_->GetRawTilingData()->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool Mc2WeightQuantBatchMatmulV2TilingAS::IsCapable()
{
    OP_TILING_CHECK(matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "antiQuantScaleDtype",
                        ge::TypeUtils::DataTypeToAscendString(matmulInfoPtr_->antiQuantScaleDtype).GetString(),
                        "The value of antiQuantScaleDtype is not supported as uint64 on NpuArch 3510."),
                    return false);
    OP_TILING_CHECK(
        (matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) &&
            (matmulInfoPtr_->transA || matmulInfoPtr_->transB),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "transA, transB",
            (std::string(matmulInfoPtr_->transA ? "true" : "false") + ", " +
             std::string(matmulInfoPtr_->transB ? "true" : "false"))
                .c_str(),
            "The value of transA and transB must be false when weight layout is FRACTAL_NZ on NpuArch 3510."),
        return false);

    // PS 从RegBase模板迁移的场景: pergroup int4 Nz groupsize(32, 64, 128, 256)
    bool isMigrationScenario = matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP;
    bool isSupported = matmulInfoPtr_->bDtype == ge::DT_INT4 && matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ &&
                       std::find(SUPPORTED_GROUP_SIZE.begin(), SUPPORTED_GROUP_SIZE.end(), matmulInfoPtr_->groupSize) !=
                           SUPPORTED_GROUP_SIZE.end();
    OP_TILING_CHECK(
        isMigrationScenario && !isSupported,
        OP_LOGI(opName_,
                "only support weight's dtype is int4, format is FRACTAL_NZ and group_size is 32, 64, 128, 256"),
        return false);
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingAS::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2TilingData);
    if (tilingData_ == nullptr) {
        try {
            tilingData_ = std::make_unique<Mc2WeightQuantBatchMatmulV2ASTilingData>();
        } catch (std::bad_alloc &) {
            tilingData_ = nullptr;
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingAS::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    if (compileInfoPtr_->socVersion != SocVersion::ASCEND910_55) {
        // 950上默认给L1的n轴大小为256
        l1NMaxSize_ = 256UL;
    } else {
        // 910_55上默认给L1的n轴大小转置情况下为128，非转置情况下为256
        l1NMaxSize_ = matmulInfoPtr_->transB ? 128UL : 256UL;
    }
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->hasBias = matmulInfoPtr_->hasBias;
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    weightMxFp4Flag_ = CheckWeightMicroscalingFp4Scene();
    weightInt4Flag_ = matmulInfoPtr_->bDtype == ge::DT_INT4;
    nzSceneFlag_ = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ && !matmulInfoPtr_->transB;
    bool highPerfFlag = CheckHighPerfScene();
    ComputeCubeTiling(highPerfFlag);
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55) {
        ComputeBasicTiling();
    } else {
        if (highPerfFlag) {
            ComputeTailResplitTiling();
        } else {
            ComputeBasicTiling();
        }
    }

    if (highPerfFlag) {
        // 设置A的提前载入
        SetPreLoad();
    }

    return ge::GRAPH_SUCCESS;
}

bool Mc2WeightQuantBatchMatmulV2TilingAS::IsWeight4Nz() const
{
    return ((weightInt4Flag_ || weightMxFp4Flag_) && nzSceneFlag_);
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ComputeCubeTiling(bool highPerfFlag)
{
    SetDefaultMatmulTiling();
    ComputeCubeSplit(highPerfFlag);
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55) {
        SetAttrs();
    }
    OptimizeMatmulTiling();
}

void Mc2WeightQuantBatchMatmulV2TilingAS::SetAttrs()
{
    // 910_55的tiling分核属性设置
    uint64_t mainBlockL1SizeDefault = l1NMaxSize_;
    uint64_t mainBlockCountDefault = ops::CeilDiv(matmulInfoPtr_->nSize, mainBlockL1SizeDefault);
    uint64_t cubeBlockDimMMax =
        ops::CeilDiv(static_cast<uint64_t>(matmulInfoPtr_->mSize), static_cast<uint64_t>(M_MAX_SIZE));
    tilingData_->cubeBlockDimN =
        (static_cast<uint32_t>(std::min(static_cast<uint64_t>(compileInfoPtr_->aicNum), mainBlockCountDefault)));
    tilingData_->cubeBlockDimM = (static_cast<uint32_t>(
        std::min(cubeBlockDimMMax, static_cast<uint64_t>(compileInfoPtr_->aicNum / tilingData_->cubeBlockDimN))));
}

void Mc2WeightQuantBatchMatmulV2TilingAS::SetDefaultMatmulTiling()
{
    tilingData_->matmulTiling.M = matmulInfoPtr_->mSize;
    tilingData_->matmulTiling.Ka = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.N = matmulInfoPtr_->nSize;
    tilingData_->matmulTiling.Kb = matmulInfoPtr_->kSize;
    if (matmulInfoPtr_->hasBias) {
        tilingData_->matmulTiling.singleCoreM = M_MAX_SIZE_WITH_BIAS_QUANT;
    } else {
        tilingData_->matmulTiling.singleCoreM = M_MAX_SIZE;
    }
    tilingData_->matmulTiling.singleCoreK = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.singleCoreN = l1NMaxSize_;

    tilingData_->matmulTiling.baseM = tilingData_->matmulTiling.singleCoreM;
    tilingData_->matmulTiling.baseN = tilingData_->matmulTiling.singleCoreN;
    tilingData_->matmulTiling.baseK = ((compileInfoPtr_->l0aSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                                       tilingData_->matmulTiling.singleCoreN);
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55 && matmulInfoPtr_->transB) {
        tilingData_->matmulTiling.baseK =
            ((compileInfoPtr_->l0aSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
             // 910_55上转置情况下singleCoreN为128，会导致算出的baseK增大为128，超L0A，故此处baseK除以2缩小为64
             tilingData_->matmulTiling.singleCoreN / 2);
    }
    tilingData_->matmulTiling.dbL0A = DOUBLE_BUFFER_NUM;
    tilingData_->matmulTiling.dbL0B = DOUBLE_BUFFER_NUM;
    tilingData_->matmulTiling.dbL0C = SINGLE_BUFFER_NUM;

    tilingData_->matmulTiling.stepM = 1;
    tilingData_->matmulTiling.stepN = 1;

    tilingData_->matmulTiling.stepKa =
        (ops::CeilDiv(K_L1_SIZE, static_cast<uint64_t>(tilingData_->matmulTiling.baseK)));
    tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.stepKa * DOUBLE_BUFFER_NUM;

    tilingData_->matmulTiling.stepKb = tilingData_->matmulTiling.stepKa;
    tilingData_->matmulTiling.depthB1 = tilingData_->matmulTiling.depthA1;
    tilingData_->matmulTiling.iterateOrder = 0;

    tilingData_->matmulTiling.isBias = static_cast<int32_t>(matmulInfoPtr_->hasBias);
    if (matmulInfoPtr_->cDtype == ge::DT_INT8) {
        tilingData_->matmulTiling.shareL1Size =
            (DOUBLE_BUFFER_NUM * tilingData_->matmulTiling.baseN * sizeof(uint64_t));
    }
    tilingData_->matmulTiling.shareL0CSize =
        (tilingData_->matmulTiling.baseM * tilingData_->matmulTiling.baseN * sizeof(float));
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ComputeCubeSplit(bool highPerfFlag)
{
    tilingData_->aPreloadSize = 0;
    if (highPerfFlag) {
        ComputeHighPerfSceneCubeSplit();
        // m方向有切分场景或非cacheline对齐，启用缓存
        int32_t minCacheLine = 128; // 缓存大小128B，对应4bit为128个元素
        if (weightMxFp4Flag_ || weightInt4Flag_) {
            minCacheLine = 256; // 缓存大小128B，对应4bit为256个元素
        }
        tilingData_->weightL2Cacheable = (tilingData_->cubeBlockDimM > 1 || tilingData_->kSize % minCacheLine != 0);
        return;
    }

    tilingData_->weightL2Cacheable = 1; // 默认开启l2缓存功能
    uint64_t mainBlockL1SizeDefault = l1NMaxSize_;
    uint32_t mainBlockCountDefault = ops::CeilDiv(matmulInfoPtr_->nSize, mainBlockL1SizeDefault);

    uint32_t cubeBlockDimMMax = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(M_MAX_SIZE));
    if (matmulInfoPtr_->transB) {
        uint32_t halfAicNum = compileInfoPtr_->aicNum >> 1;
        if (mainBlockCountDefault <= halfAicNum) {
            // 分核只有一半，尝试切分粒度缩减后再重新切分
            mainBlockL1SizeDefault = mainBlockL1SizeDefault >> 1;
            mainBlockCountDefault = ops::CeilDiv(matmulInfoPtr_->nSize, mainBlockL1SizeDefault);
        }

        tilingData_->cubeBlockDimN = std::min(compileInfoPtr_->aicNum, mainBlockCountDefault);
        if (cubeBlockDimMMax <= (compileInfoPtr_->aicNum / tilingData_->cubeBlockDimN) >> 1) {
            // 分核只够剩下可用核数量一半，尝试切分粒度缩减后再重新切分
            tilingData_->cubeBlockDimM = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(M_MAX_SIZE >> 1));
        } else {
            tilingData_->cubeBlockDimM = (std::min(cubeBlockDimMMax, static_cast<uint32_t>(compileInfoPtr_->aicNum) /
                                                                         tilingData_->cubeBlockDimN));
        }
    } else {
        tilingData_->cubeBlockDimN = std::min(compileInfoPtr_->aicNum, mainBlockCountDefault);
        tilingData_->cubeBlockDimM = (std::min(cubeBlockDimMMax, compileInfoPtr_->aicNum / tilingData_->cubeBlockDimN));
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ComputeHighPerfSceneCubeSplit()
{
    if ((matmulInfoPtr_->mSize <= M_MAX_SIZE && !matmulInfoPtr_->hasBias) ||
        (matmulInfoPtr_->mSize <= M_MAX_SIZE_WITH_BIAS_QUANT && matmulInfoPtr_->hasBias)) {
        /*
         * 增量场景下需要讨论cube以何种方式切分
         * 根据实验数据，weight切分后n=128的处理数据量在大部分场景下可以发挥带宽优势。
         * n切分后小于该阈值，或者m大于该阈值的场景，只在n方向分核可能导致整体搬运量过大(A矩阵的占比较高)。
         * 此时需要遍历可能存在的切分方式，找到理论最小搬运量的解。
         * 在其他场景下，默认只在n方向分核即可。
         */
        static const uint64_t TAIL_L1_SIZE_THRESHOLD = 128UL;
        bool needPollingFlag = matmulInfoPtr_->nSize < TAIL_L1_SIZE_THRESHOLD * compileInfoPtr_->aicNum ||
                               (matmulInfoPtr_->nSize >= TAIL_L1_SIZE_THRESHOLD * compileInfoPtr_->aicNum &&
                                matmulInfoPtr_->mSize > TAIL_L1_SIZE_THRESHOLD);
        if (!needPollingFlag) {
            tilingData_->cubeBlockDimM = 1;
            tilingData_->cubeBlockDimN = compileInfoPtr_->aicNum;
            return;
        }
        // 开始轮询
        CubeSplitResult cubeSplitResult = {UINT64_MAX, 1, compileInfoPtr_->aicNum};
        for (uint64_t mBlkNum = 1UL;
             mBlkNum <= std::min(static_cast<uint64_t>(compileInfoPtr_->aicNum),
                                 ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(BLOCK_CUBE)));
             mBlkNum++) {
            uint64_t nBlkNumMax = compileInfoPtr_->aicNum / mBlkNum;
            uint64_t nL1SizeMin =
                ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nBlkNumMax), static_cast<uint64_t>(BLOCK_CUBE));
            uint64_t nBlkNum = std::min(ops::CeilDiv(matmulInfoPtr_->nSize, nL1SizeMin), nBlkNumMax);

            // k轴的大小对mte2的搬运量无影响，此处无需计算
            uint64_t mte2Cost = GetSizeByDataType(matmulInfoPtr_->aDtype) * matmulInfoPtr_->mSize * nBlkNum +
                                GetSizeByDataType(matmulInfoPtr_->bDtype) * matmulInfoPtr_->nSize * mBlkNum;
            if (matmulInfoPtr_->bDtype == ge::DT_INT4 || matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 ||
                matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E1M2) {
                mte2Cost = GetSizeByDataType(matmulInfoPtr_->aDtype) * matmulInfoPtr_->mSize * nBlkNum +
                           matmulInfoPtr_->nSize * mBlkNum / BIT4_CORRECTION_FACTOR;
            }
            // n较小场景才会进入轮询，此时应该在保证mte2Cost最小的前提下，尽可能提高m的切分粒度，保证n在单核上不会太小
            if (mte2Cost <= cubeSplitResult.mte2Cost) {
                cubeSplitResult.mte2Cost = mte2Cost;
                cubeSplitResult.cubeBlockDimM = mBlkNum;
                cubeSplitResult.cubeBlockDimN = nBlkNum;
            }
        }
        // 根据轮询结果设置cube切分
        tilingData_->cubeBlockDimM = cubeSplitResult.cubeBlockDimM;
        tilingData_->cubeBlockDimN = cubeSplitResult.cubeBlockDimN;
    } else {
        // 全量尝试4 : 8的切分设置
        static const uint64_t BLOCK_DIM_M_MAX = std::min(static_cast<uint64_t>(compileInfoPtr_->aicNum), 4UL);
        uint64_t realBlockDimM = matmulInfoPtr_->hasBias ?
                                     ops::CeilDiv(matmulInfoPtr_->mSize, M_MAX_SIZE_WITH_BIAS_QUANT) :
                                     ops::CeilDiv(matmulInfoPtr_->mSize, M_MAX_SIZE);
        tilingData_->cubeBlockDimM = std::min(BLOCK_DIM_M_MAX, realBlockDimM);
        uint64_t nBlkNumMax = compileInfoPtr_->aicNum / tilingData_->cubeBlockDimM;
        uint64_t nL1SizeMin =
            ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nBlkNumMax), static_cast<uint64_t>(BLOCK_CUBE));
        tilingData_->cubeBlockDimN = std::min(ops::CeilDiv(matmulInfoPtr_->nSize, nL1SizeMin), nBlkNumMax);
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::EnlargeBaseK(uint64_t l0aMaxBaseK)
{
    // 910_55不走basek放大的优化方案
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55) {
        return;
    }
    // 根据理论需要处理的N推算L0B上K的最大值
    uint64_t l0bMaxBaseK = (compileInfoPtr_->l0bSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                           tilingData_->matmulTiling.singleCoreN / BLOCK_CUBE * BLOCK_CUBE;
    // l0b上K的取值可以分成三档，即64,128,256，默认设置为64。当前处理的singleCoreN不会超过256,因此暂不考虑baseK=32。
    static const std::vector<uint64_t> L0_BASE_K_LIST = (weightMxFp4Flag_ && nzSceneFlag_) ?
                                                            std::vector<uint64_t>{128UL, 64UL} :
                                                            std::vector<uint64_t>{256UL, 128UL, 64UL};
    tilingData_->matmulTiling.baseK = L0_BASE_K_LIST.back(); // 初始化赋值
    if ((matmulInfoPtr_->transB && matmulInfoPtr_->antiQuantType != Mc2QuantType::PER_TENSOR) ||
        (weightMxFp4Flag_ && nzSceneFlag_)) { // 小N，非转置场景不做优化，非主要性能场景，避免tilingkey膨胀
        for (size_t listId = 0; listId < L0_BASE_K_LIST.size(); listId++) {
            if (std::min(l0bMaxBaseK, l0aMaxBaseK) >= L0_BASE_K_LIST[listId]) {
                tilingData_->matmulTiling.baseK = L0_BASE_K_LIST[listId];
                break;
            }
        }
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::EnlargeBaseKInFullloadA(uint64_t maxBaseK, uint64_t minKL1, uint64_t maxL1,
                                                                  const std::vector<uint64_t> &l0BaseKList,
                                                                  const std::vector<uint64_t> &l1KList)
{
    tilingData_->matmulTiling.baseK = l0BaseKList.back(); // 初始化赋值
    for (auto baseK : l0BaseKList) {
        if (maxBaseK < baseK) {
            continue;
        }
        uint64_t stepKa = ops::CeilDiv(minKL1, baseK);
        uint64_t maxKbL1 =
            ((maxL1 - baseK * stepKa * tilingData_->matmulTiling.baseM) >> 1) / tilingData_->matmulTiling.baseN;
        if (maxKbL1 < baseK) {
            continue;
        }
        tilingData_->matmulTiling.baseK = baseK;
        for (auto kL1 : l1KList) {
            if (maxKbL1 < kL1) {
                continue;
            }
            tilingData_->matmulTiling.stepKa = stepKa;
            tilingData_->matmulTiling.stepKb = std::min(kL1 / baseK, 4UL);
            tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.stepKa;
            tilingData_->matmulTiling.depthB1 = tilingData_->matmulTiling.stepKb << 1;
            break;
        }
        break;
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::EnlargeBaseKNotFullloadA(uint64_t maxBaseK, uint64_t maxKL1,
                                                                   const std::vector<uint64_t> &l0BaseKList,
                                                                   const std::vector<uint64_t> &l1KList)
{
    tilingData_->matmulTiling.baseK = l0BaseKList.back(); // 初始化赋值
    for (auto baseK : l0BaseKList) {
        if (maxBaseK < baseK || maxKL1 < baseK) {
            continue;
        }
        tilingData_->matmulTiling.baseK = baseK;
        for (auto kL1 : l1KList) {
            if (maxKL1 < kL1) {
                continue;
            }
            tilingData_->matmulTiling.stepKa = std::min(kL1 / baseK, 4UL);
            tilingData_->matmulTiling.stepKb = tilingData_->matmulTiling.stepKa;
            tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.stepKa << 1;
            tilingData_->matmulTiling.depthB1 = tilingData_->matmulTiling.depthA1;
            break;
        }
        break;
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::OptimizeMatmulTiling()
{
    // 修正m方向的实际大小
    tilingData_->matmulTiling.singleCoreM =
        (ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->cubeBlockDimM)));

    tilingData_->matmulTiling.baseM =
        (std::min(tilingData_->matmulTiling.baseM,
                  ops::CeilAlign(tilingData_->matmulTiling.singleCoreM, static_cast<int32_t>(BLOCK_CUBE))));

    // 计算N分核后，单core上理论需要处理的最大N，并往16对齐
    uint64_t singleN =
        ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->cubeBlockDimN)), 16UL);
    tilingData_->matmulTiling.singleCoreN = std::min(l1NMaxSize_, singleN);
    tilingData_->matmulTiling.baseN = tilingData_->matmulTiling.singleCoreN;

    if (IsWeight4Nz() && weightInt4Flag_ && nzSceneFlag_ && !matmulInfoPtr_->transA && !matmulInfoPtr_->transB &&
        matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP && matmulInfoPtr_->cDtype != ge::DT_INT8) {
        OptimizeMatmulTilingInA16W4Nz();
        return;
    }

    // 根据M值计算L0A上K的最大值
    uint64_t l0aMaxBaseK = (compileInfoPtr_->l0aSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                           tilingData_->matmulTiling.singleCoreM / BLOCK_CUBE * BLOCK_CUBE;
    // 根据l1的情况反推L0A上K的最大值
    if (matmulInfoPtr_->hasBias) {
        l0aMaxBaseK =
            std::min(l0aMaxBaseK, (A_L1_MAX_SIZE_WITH_BIAS_QUANT >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                                      tilingData_->matmulTiling.baseM / tilingData_->matmulTiling.stepKa / BLOCK_CUBE *
                                      BLOCK_CUBE);
    } else {
        l0aMaxBaseK = std::min(l0aMaxBaseK, (A_L1_MAX_SIZE >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                                                tilingData_->matmulTiling.baseM / tilingData_->matmulTiling.stepKa /
                                                BLOCK_CUBE * BLOCK_CUBE);
    }
    EnlargeBaseK(l0aMaxBaseK);

    // 判断L1上的A矩阵是否可以全载
    uint64_t aL1MaxSize = (matmulInfoPtr_->hasBias || matmulInfoPtr_->cDtype == ge::DT_INT8) ?
                              A_L1_MAX_SIZE_WITH_BIAS_QUANT :
                              A_L1_MAX_SIZE;
    uint64_t kAlign = ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    if (tilingData_->matmulTiling.baseM * kAlign * GetSizeByDataType(matmulInfoPtr_->aDtype) <= aL1MaxSize) {
        tilingData_->matmulTiling.stepKa =
            (ops::CeilDiv(kAlign, static_cast<uint64_t>(tilingData_->matmulTiling.baseK)));
        tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.stepKa;
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::OptimizeMatmulTilingInA16W4Nz()
{
    static const std::vector<uint64_t> L0_BASE_K_LIST = {1024UL, 512UL, 256UL, 128UL, 64UL};
    static const std::vector<uint64_t> L1_K_LIST = {1024UL, 512UL, 256UL};

    // 根据M值计算L0A上K的最大值
    uint64_t l0aMaxBaseK = (compileInfoPtr_->l0aSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                           tilingData_->matmulTiling.singleCoreM / BLOCK_CUBE * BLOCK_CUBE;
    // 根据理论需要处理的N推算L0B上K的最大值
    uint64_t l0bMaxBaseK = (compileInfoPtr_->l0bSize >> 1) / GetSizeByDataType(matmulInfoPtr_->aDtype) /
                           tilingData_->matmulTiling.singleCoreN / BLOCK_CUBE * BLOCK_CUBE;
    uint64_t bL1 =
        tilingData_->matmulTiling.depthB1 * tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.baseN;

    uint64_t kAlign = ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t l1Max = ((matmulInfoPtr_->hasBias || matmulInfoPtr_->cDtype == ge::DT_INT8) ? L1_MAX_SIZE_WITH_BIAS_QUANT :
                                                                                           L1_MAX_SIZE) /
                     GetSizeByDataType(matmulInfoPtr_->aDtype);
    bool fullLoadA = tilingData_->matmulTiling.baseM * kAlign <= (l1Max - bL1);
    // 根据l1的情况反推L0A上K的最大值
    if (fullLoadA) {
        EnlargeBaseKInFullloadA(std::min(l0aMaxBaseK, l0bMaxBaseK), kAlign, l1Max, L0_BASE_K_LIST, L1_K_LIST);
    } else {
        uint64_t l1MaxK = (l1Max / (tilingData_->matmulTiling.baseN + tilingData_->matmulTiling.baseM)) >> 1;
        EnlargeBaseKNotFullloadA(std::min(l0aMaxBaseK, l0bMaxBaseK), l1MaxK, L0_BASE_K_LIST, L1_K_LIST);
    }
}

/*
 * 高性能场景准入条件：
 * 非int8输出
 * 1. a不转置  b转置    per channel           A16W8/A16W4       ND
 * 2. a不转置  b转置    mx                    A16MxFp4          ND
 * 3. a不转置  b不转置  per tensor/channel/mx A16W4/A16MxFp4    NZ
 * 4. a不转置  b不转置  per group             A16W4(int4)       NZ
 */
bool Mc2WeightQuantBatchMatmulV2TilingAS::CheckHighPerfScene() const
{
    bool a16w8w4NdHighPerfScene = matmulInfoPtr_->transB && matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_CHANNEL;

    bool a16MxFp4NdHighPerfScene =
        weightMxFp4Flag_ && matmulInfoPtr_->transB && matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ;

    bool transACDtypeRestriction = !matmulInfoPtr_->transA && matmulInfoPtr_->cDtype != ge::DT_INT8;

    if (compileInfoPtr_->socVersion != SocVersion::ASCEND910_55 && transACDtypeRestriction &&
        (a16w8w4NdHighPerfScene || a16MxFp4NdHighPerfScene || IsWeight4Nz())) {
        OP_LOGD(opName_, "current shape match Adaptive tiling high perf scene.");
        return true;
    }

    return false;
}

bool Mc2WeightQuantBatchMatmulV2TilingAS::CheckWeightMicroscalingFp4Scene() const
{
    if (matmulInfoPtr_->antiQuantScaleDtype == ge::DT_FLOAT8_E8M0 &&
        (matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E2M1 || matmulInfoPtr_->bDtype == ge::DT_FLOAT4_E1M2)) {
        OP_LOGD(opName_, "current shape match microscaling fromats scene, bDtype is FP4.");
        return true;
    }
    return false;
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ComputeTailResplitTiling()
{
    algorithmSubCategory_ = Mc2OptimizationAlgorithmSubCategory::N_FIRST_TAIL_RESPLIT;

    // step 1. 尾块重切分
    ResplitTail();

    // step 2. 判定内轴(k)的核内切分规则
    SetInnerSize();

    // step 3. 重计算baseN baseK
    RecalculateBaseBlockSize();
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ResplitTail()
{
    uint64_t mainL1SizeDefault = l1NMaxSize_;
    // nSize，ND场景为n，Nz场景为n1
    uint64_t nSize = matmulInfoPtr_->nSize;
    // A16W4/A16MxFp4 Nz weight不转置场景，(n1,k1,k0,n0)，n1按16的粒度切分
    if (IsWeight4Nz()) {
        mainL1SizeDefault = l1NMaxSize_ / static_cast<uint64_t>(BLOCK_CUBE);
        nSize = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
    }
    uint64_t mainBlockCountDefault = nSize / mainL1SizeDefault;

    // 主块数量较多，则匀出一个主块给尾块
    if (mainBlockCountDefault >= tilingData_->cubeBlockDimN ||
        nSize % (mainL1SizeDefault * tilingData_->cubeBlockDimN) == 0) {
        tilingData_->mainBlockL1Size = mainL1SizeDefault;
        if (nSize % (mainL1SizeDefault * tilingData_->cubeBlockDimN) == 0) {
            tilingData_->mainBlockCount = mainBlockCountDefault;
        } else {
            uint64_t blockCountFloorAlign =
                mainBlockCountDefault / tilingData_->cubeBlockDimN * tilingData_->cubeBlockDimN;
            tilingData_->mainBlockCount = blockCountFloorAlign - tilingData_->cubeBlockDimN;
        }
    } else {
        tilingData_->mainBlockL1Size = 0;
        tilingData_->mainBlockCount = 0;
    }
    uint64_t tailSize = nSize - tilingData_->mainBlockCount * tilingData_->mainBlockL1Size;

    if (IsWeight4Nz()) {
        tilingData_->mainBlockL1Size = tilingData_->mainBlockL1Size * static_cast<uint64_t>(BLOCK_CUBE);
    }

    if (tailSize == 0UL) {
        tilingData_->firstTailBlockL1Size = 0;
        tilingData_->firstTailBlockCount = 0;
        tilingData_->secondTailBlockL1Size = 0;
        tilingData_->secondTailBlockCount = 0;
        return;
    }
    /* 一个主块+一个尾块，需要多核做2轮才能保证单次不超过基本块的最大规格。
     * 一个尾块，多核做1轮将可以保证单次不超过基本块的最大规格。
     */
    uint64_t tailBlockCount = mainBlockCountDefault >= tilingData_->cubeBlockDimN ? 2 * tilingData_->cubeBlockDimN :
                                                                                    tilingData_->cubeBlockDimN;
    tilingData_->firstTailBlockCount = tailBlockCount - tailSize % tailBlockCount;
    tilingData_->firstTailBlockL1Size = tailSize / tailBlockCount;
    tilingData_->secondTailBlockCount = tailSize % tailBlockCount;
    tilingData_->secondTailBlockL1Size = 0;
    if (tailSize % tailBlockCount > 0UL) {
        tilingData_->secondTailBlockL1Size = tilingData_->firstTailBlockL1Size + 1;
    }
    if (IsWeight4Nz()) {
        tilingData_->firstTailBlockL1Size = (tilingData_->firstTailBlockL1Size * static_cast<uint64_t>(BLOCK_CUBE));
        tilingData_->secondTailBlockL1Size = (tilingData_->secondTailBlockL1Size * static_cast<uint64_t>(BLOCK_CUBE));
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::SetInnerSize()
{
    static constexpr uint64_t MTE2_K_256 = 256UL;
    static constexpr uint64_t MTE2_K_512 = 512UL;
    static constexpr uint64_t MTE2_K_1024 = 1024UL;
    uint64_t weightL1K = tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.stepKb;

    if (IsWeight4Nz()) {
        if (weightInt4Flag_ && matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP) {
            if (weightL1K >= MTE2_K_1024) {
                mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_4;
            } else if (weightL1K >= MTE2_K_512) {
                mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4;
            } else {
                mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2;
            }
        } else if (weightMxFp4Flag_ && weightL1K >= MTE2_K_512) {
            mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4;
        } else {
            mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4;
        }
        return;
    }

    if (weightMxFp4Flag_ && matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2;
        return;
    }

    // 内轴切分仅支持512、1024切分,若weightL1K已超过这些配置，则直接调整内轴配置即可，无需进一步调整
    if (weightL1K >= MTE2_K_1024) {
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2;
        return;
    } else if (weightL1K >= MTE2_K_512) {
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4;
        return;
    }

    uint64_t blockL1MaxSize = std::max(tilingData_->mainBlockL1Size,
                                       std::max(tilingData_->firstTailBlockL1Size, tilingData_->secondTailBlockL1Size));
    if (matmulInfoPtr_->kSize <= MTE2_K_256 * QUARTER_BUFFER_NUM) {
        // k轴较小场景，mte2内轴较大会造成较高的头开销，缩短内轴增加ub上的buffer数量
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4;
    } else if (blockL1MaxSize > (l1NMaxSize_ >> 1)) {
        // 实际处理的n轴大于ub上限的一半，ub的buffer数量无法进一步增加
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2;
    } else if (matmulInfoPtr_->kSize <= MTE2_K_512 * QUARTER_BUFFER_NUM) {
        // 实际处理的n轴较小，可以适当增加ub载入量或载入频率。考虑到内轴较小，通过增加ub上的buffer数量提高载入频率
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4;
    } else {
        // 实际处理的n轴较小，可以适当增加ub载入量或载入频率。考虑到内轴较大，通作增加内轴提高载入量
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2;
    }
}

/***
 * 在 8192<n<=8192*2的场景下，会存在尾块N方向较小的情况，此时需要重新计算基本块大小，尝试扩大baseN.
 * baseN扩大后，相应带来UB、L1、L0大小的变化需要重新计算。
 * 其中当把baseN扩大时，需要将baseK减半，而baseN最多扩张1倍，所以L0无需重新计算。
 */
void Mc2WeightQuantBatchMatmulV2TilingAS::RecalculateBaseBlockSize()
{
    // A16MxFp4场景，不做baskN放大
    if (weightMxFp4Flag_) {
        return;
    }
    if (compileInfoPtr_->aicNum * l1NMaxSize_ < matmulInfoPtr_->nSize &&
        compileInfoPtr_->aicNum * l1NMaxSize_ * NUM_TWO > matmulInfoPtr_->nSize &&
        (matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_CHANNEL ||
         matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP)) {
        uint64_t tailSize = matmulInfoPtr_->nSize - tilingData_->mainBlockCount * tilingData_->mainBlockL1Size;

        if (weightInt4Flag_ && nzSceneFlag_) {
            // A16W4 Nz weight不转置场景，n方向按照n1相对于16的粒度切分
            tailSize = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE)) -
                       tilingData_->mainBlockCount * tilingData_->mainBlockL1Size / static_cast<uint64_t>(BLOCK_CUBE);
        }

        uint64_t tailBlockCount = tilingData_->cubeBlockDimN;
        if (tailBlockCount == 0UL) {
            return;
        }
        uint64_t firstTailBlockL1Size = tailSize / tailBlockCount;
        uint64_t firstTailBlockCount = tailBlockCount - tailSize % tailBlockCount;
        uint64_t secondTailBlockL1Size = 0;
        uint64_t secondTailBlockCount = 0;
        if (tailSize % tailBlockCount > 0UL) {
            secondTailBlockL1Size = firstTailBlockL1Size + 1UL;
            secondTailBlockCount = tailSize % tailBlockCount;
        }
        uint64_t maxBlockL1Size = std::max(firstTailBlockL1Size, secondTailBlockL1Size);
        uint64_t extendedWeightMte2N = nzSceneFlag_ ?
                                           ops::CeilDiv(maxBlockL1Size, NUM_TWO) * static_cast<uint64_t>(BLOCK_CUBE) :
                                           ops::CeilDiv(maxBlockL1Size, NUM_TWO);
        uint64_t extendedBaseN = nzSceneFlag_ ? maxBlockL1Size * static_cast<uint64_t>(BLOCK_CUBE) :
                                                ops::CeilAlign(maxBlockL1Size, static_cast<uint64_t>(BLOCK_CUBE));
        if (!CheckUbSizeAfterExtending(extendedWeightMte2N) || !CheckL1SizeAfterExtending(extendedBaseN) ||
            !CheckL0CSizeAfterExtending(extendedBaseN)) {
            return;
        }
        // 满足空间要求后，修正tiling参数
        ReSetTilingAfterExtendedBaseN(extendedBaseN, firstTailBlockL1Size, firstTailBlockCount, secondTailBlockL1Size,
                                      secondTailBlockCount);
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ReSetTilingAfterExtendedBaseN(uint64_t extendedBaseN,
                                                                        uint64_t firstTailBlockL1Size,
                                                                        uint64_t firstTailBlockCount,
                                                                        uint64_t secondTailBlockL1Size,
                                                                        uint64_t secondTailBlockCount) const
{
    tilingData_->matmulTiling.baseN = extendedBaseN;
    tilingData_->matmulTiling.singleCoreN = extendedBaseN;
    tilingData_->matmulTiling.baseK = tilingData_->matmulTiling.baseK / NUM_TWO;     // 将 baseK 减半
    tilingData_->matmulTiling.stepKa = tilingData_->matmulTiling.stepKa * NUM_TWO;   // 将 stepKa 翻倍
    tilingData_->matmulTiling.stepKb = tilingData_->matmulTiling.stepKb * NUM_TWO;   // 将 stepKb 翻倍
    tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.depthA1 * NUM_TWO; // 将 depthA1 翻倍
    tilingData_->matmulTiling.depthB1 = tilingData_->matmulTiling.depthB1 * NUM_TWO; // 将 depthB1 翻倍
    tilingData_->matmulTiling.shareL0CSize =
        (tilingData_->matmulTiling.baseM * tilingData_->matmulTiling.baseN * sizeof(float));
    tilingData_->firstTailBlockL1Size = firstTailBlockL1Size;
    tilingData_->firstTailBlockCount = firstTailBlockCount;
    tilingData_->secondTailBlockL1Size = secondTailBlockL1Size;
    tilingData_->secondTailBlockCount = secondTailBlockCount;
    if (nzSceneFlag_) {
        tilingData_->firstTailBlockL1Size = firstTailBlockL1Size * static_cast<uint64_t>(BLOCK_CUBE);
        tilingData_->secondTailBlockL1Size = secondTailBlockL1Size * static_cast<uint64_t>(BLOCK_CUBE);
    }
}

bool Mc2WeightQuantBatchMatmulV2TilingAS::CheckL1SizeAfterExtending(uint64_t extendedBaseN) const
{
    uint64_t al1Size = tilingData_->matmulTiling.baseM * tilingData_->matmulTiling.depthA1 *
                       tilingData_->matmulTiling.baseK * GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t extendedBl1Size = extendedBaseN * tilingData_->matmulTiling.depthB1 * tilingData_->matmulTiling.baseK *
                               GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t aL1MaxSize = (matmulInfoPtr_->hasBias || matmulInfoPtr_->cDtype == ge::DT_INT8) ?
                              A_L1_MAX_SIZE_WITH_BIAS_QUANT :
                              A_L1_MAX_SIZE;
    if (aL1MaxSize - al1Size < extendedBl1Size - (compileInfoPtr_->l1Size >> 1)) {
        return false;
    }
    return true;
}

bool Mc2WeightQuantBatchMatmulV2TilingAS::CheckL0CSizeAfterExtending(uint64_t extendedBaseN) const
{
    return (tilingData_->matmulTiling.baseM * extendedBaseN * sizeof(float) <= compileInfoPtr_->l0cSize);
}

bool Mc2WeightQuantBatchMatmulV2TilingAS::CheckUbSizeAfterExtending(uint64_t extendedWeightMte2N) const
{
    uint64_t antiquantParamsBufferSize = 8 * KILOBYTE; // antiquant参数占用8KB
    uint64_t weightLowBitBufferSize = extendedWeightMte2N * KILOBYTE;
    // A16W4NZ场景128KB，其他66KB
    uint64_t weightF16BufferSize = (weightInt4Flag_ && nzSceneFlag_) ? 128UL * KILOBYTE : 66UL * KILOBYTE;
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        weightLowBitBufferSize = weightLowBitBufferSize / BIT4_CORRECTION_FACTOR;
    }
    if (mte2Config_ == Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_4 ||
        mte2Config_ == Mc2Mte2Configuration::MTE2_INNER_SIZE_1024_BUF_NUM_2) {
        weightLowBitBufferSize = weightLowBitBufferSize * NUM_TWO;
    }

    if (weightLowBitBufferSize + weightF16BufferSize + antiquantParamsBufferSize > UB_SIZE_V100) {
        return false;
    }
    return true;
}

void Mc2WeightQuantBatchMatmulV2TilingAS::SetPreLoad()
{
    uint64_t aSize = matmulInfoPtr_->mSize * matmulInfoPtr_->kSize * GetSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t aL1MaxSize = (matmulInfoPtr_->hasBias || matmulInfoPtr_->cDtype == ge::DT_INT8) ?
                              A_L1_MAX_SIZE_WITH_BIAS_QUANT :
                              A_L1_MAX_SIZE;
    /*
     * 准入条件：
     * m < 512 && a矩阵size < cubeDimM * cubeDimN * aL1MaxSize
     */
    if (matmulInfoPtr_->mSize < 512 && aSize < tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN * aL1MaxSize) {
        tilingData_->aPreloadSize = (ops::CeilAlign(
            ops::CeilDiv(static_cast<uint64_t>(matmulInfoPtr_->mSize * matmulInfoPtr_->kSize),
                         static_cast<uint64_t>(tilingData_->cubeBlockDimN * tilingData_->cubeBlockDimM)),
            static_cast<uint64_t>(64))); // 64 表示128B的cacheline对齐
    }
}

void Mc2WeightQuantBatchMatmulV2TilingAS::ComputeBasicTiling()
{
    algorithmSubCategory_ = Mc2OptimizationAlgorithmSubCategory::N_FIRST_BASIC_BLOCK;
    if (matmulInfoPtr_->transB && compileInfoPtr_->socVersion != SocVersion::ASCEND910_55) {
        // step 1. n方向做尾块重切分
        ResplitTail();

        // step 2. 判定内轴(k)的核内切分规则
        mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT;
    } else {
        // step 1. n方向为内轴，无法做重切分
        tilingData_->mainBlockL1Size = tilingData_->matmulTiling.singleCoreN;
        tilingData_->mainBlockCount = (matmulInfoPtr_->nSize / tilingData_->matmulTiling.singleCoreN /
                                       tilingData_->cubeBlockDimN * tilingData_->cubeBlockDimN);
        tilingData_->firstTailBlockL1Size =
            (tilingData_->matmulTiling.singleCoreN); // n为内轴，细粒度切分影响搬运效率，此处暂不考虑

        uint64_t tailSize = matmulInfoPtr_->nSize - tilingData_->mainBlockCount * tilingData_->mainBlockL1Size;
        tilingData_->firstTailBlockCount =
            (ops::CeilDiv(tailSize, static_cast<uint64_t>(tilingData_->matmulTiling.singleCoreN)));
        tilingData_->secondTailBlockL1Size = 0;
        tilingData_->secondTailBlockCount = 0;

        // step 2. 判定内轴(k)的核内切分规则
        if (compileInfoPtr_->socVersion != SocVersion::ASCEND910_55) {
            mte2Config_ = Mc2Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_4;
        } else {
            mte2Config_ = matmulInfoPtr_->transB ? Mc2Mte2Configuration::MTE2_INNER_SIZE_512_BUF_NUM_2 :
                                                   Mc2Mte2Configuration::MTE2_INNER_SIZE_256_BUF_NUM_2;
        }
    }
}

// 5、计算TilingKey
uint64_t Mc2WeightQuantBatchMatmulV2TilingAS::GetTilingKey() const
{
    Mc2TilingKeyConfigure tilingKeyConfigure;
    if (compileInfoPtr_->socVersion == SocVersion::ASCEND910_55) {
        // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
        tilingKeyConfigure.socVersionType = static_cast<uint8_t>(SocVersion::ASCEND910_55) * 10;
    } else {
        // 平台类型占2位(平台大类， 平台小类)，平台大类在高位，需要乘10
        tilingKeyConfigure.socVersionType = static_cast<uint8_t>(Mc2SocVersionType::SUPPORT_L1_TO_BT_BF16) * 10;
    }
    tilingKeyConfigure.quantizationScenario = static_cast<uint8_t>(Mc2QuantizationScenario::DEFAULT);
    // 算法类型占2位(算法大类，算法小类)，算法大类在高位，需要乘10
    tilingKeyConfigure.algorithm = static_cast<uint8_t>(Mc2OptimizationAlgorithmCategory::VECTOR_ANTIQUANT) * 10 +
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
    tilingKeyConfigure.weightFormat = (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ?
                                          static_cast<uint8_t>(Mc2WeightFormat::FRACTAL_NZ) :
                                          static_cast<uint8_t>(Mc2WeightFormat::ND);
    tilingKeyConfigure.templateCustom = static_cast<uint8_t>(mte2Config_);
    tilingKeyConfigure.apiConstexpr = 0;
    return tilingKeyConfigure.GenTilingKey();
}

// 6、计算Workspace 大小
ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingAS::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, OP_LOGE(opName_, "failed to get workspace size"), return ge::GRAPH_FAILED);
    workspaces[0] = static_cast<size_t>(16 * 1024 * 1024); // asc要求workspace最低需要16 * 1024 * 1024 Byte
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(Mc2WeightQuantBatchMatmulV2, Mc2WeightQuantBatchMatmulV2TilingAS, ADAPTIVE_SPLIT_PRIORITY);
} // namespace Mc2weight_quant_batch_matmul_v2
} // namespace optiling
