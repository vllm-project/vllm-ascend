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
 * \file weight_quant_batch_matmul_v2_tiling_custom.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_tiling_custom.h"

#include "weight_quant_batch_matmul_v2_compute_matmul_tiling.h"
#include "weight_quant_batch_matmul_v2_white_list.h"
#include "op_host/tiling_key.h"
#include "ops_legacy/op_tiling/op_cache_tiling.h"
#include "ops_legacy/op_tiling/op_cache_def_tiling.h"
#include "../../op_kernel/weight_quant_batch_matmul_v2_kernel_tiling_key.h"

using Ops::Transformer::OpTiling::RecursiveSum;

namespace optiling {

constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr int32_t MAX_REPEAT_TIMES = 255;
constexpr uint32_t CUSTOM_NZ_TRANS_BASE_N = 64;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_BASE_N = 32;
constexpr uint32_t CUSTOM_NZ_TRANS_BF16_BASE_K = 256;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_BF16_BASE_N = 544;
constexpr uint32_t CUSTOM_NZ_TRANS_FP16_BASE_K = 384;
constexpr uint32_t CUSTOM_NZ_NO_TRANS_FP16_BASE_K = 864;
constexpr int32_t TILING_COMPENSATION_FACTOR = 2;
constexpr uint32_t CUSTOM_NZ_GROUP_BASE_N = 48U;

void Mc2WeightQuantBatchMatmulV2TilingCustom::Reset()
{
    cubeBaseN_ = static_cast<uint64_t>(BLOCK_CUBE);
}

/*
The function is limite of custom
1. not support antiquant scale dtype is uint64/int64
*/
bool Mc2WeightQuantBatchMatmulV2TilingCustom::IsCapable()
{
    OP_LOGI(opName_, "Begin check custom");
    OP_TILING_CHECK(((matmulInfoPtr_->antiQuantScaleDtype == ge::DT_UINT64) ||
                     (matmulInfoPtr_->antiQuantScaleDtype == ge::DT_INT64)),
                    OP_LOGI(opName_, "Custom do not support antiquant scale dtype is uint64 and int64"), return false);
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ && matmulInfoPtr_->antiQuantType == Mc2QuantType::PER_GROUP) {
        OP_TILING_CHECK(matmulInfoPtr_->groupSize != 64 && matmulInfoPtr_->groupSize != 128,
                        OP_LOGI(opName_,
                                "Custom Nz only support group_size = 64 or 128 for per-group scene, but is [%lu]",
                                matmulInfoPtr_->groupSize),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->kSize % matmulInfoPtr_->groupSize != 0,
                        OP_LOGI(opName_,
                                "Custom Nz only support kSize align to group_size for per-group scene, "
                                "but kSize is [%lu], group_size is [%lu]",
                                matmulInfoPtr_->kSize, matmulInfoPtr_->groupSize),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->kSize % 64 != 0 && matmulInfoPtr_->nSize % 64 != 0,
                        OP_LOGI(opName_,
                                "Custom Nz only support kSize and nSize align to 64 for per-group scene, "
                                "but kSize is [%lu], nSize is [%lu]",
                                matmulInfoPtr_->kSize, matmulInfoPtr_->nSize),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->transB,
                        OP_LOGI(opName_, "Custom Nz cannot support weight transpose for per-group scene"),
                        return false);
        OP_TILING_CHECK(matmulInfoPtr_->kSize > MAX_SHAPE_DIM || matmulInfoPtr_->nSize > MAX_SHAPE_DIM,
                        OP_LOGI(opName_, "Custom Nz only support and n < 65536 and k < 65536"), return false);
    }
    OP_LOGI(opName_, "Check custom succ");
    return true;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingCustom::DoOpTiling()
{
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    OP_LOGE(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);
    // Set shape dim and pad of tiling date
    SetShapeSize();
    OP_TILING_CHECK(!GetMatMulTiling(),
                    OP_LOGE(opName_, "failed to get mm tiling for mnk[%lu, %lu, %lu]", matmulInfoPtr_->mSize,
                            matmulInfoPtr_->nSize, matmulInfoPtr_->kSize),
                    return ge::GRAPH_FAILED);

    uint64_t defaultVecSingleN = 0;
    uint64_t defaultVecSingleK = 0;
    if (matmulInfoPtr_->groupSize > 0) {
        ComputeGroupDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    } else {
        ComputeDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }

    uint64_t vecSingleN = std::min(defaultVecSingleN, tilingData_->nAlign);
    uint64_t vecSingleK = std::min(defaultVecSingleK, tilingData_->kAlign);

    tilingData_->vecSingleN = static_cast<uint32_t>(vecSingleN);
    tilingData_->vecSingleK = static_cast<uint32_t>(vecSingleK);

    uint64_t totalCubeSingleN = cubeBaseN_ * tilingData_->cubeBlockDimN;
    totalCubeSingleN = std::min(totalCubeSingleN, tilingData_->nAlign);
    tilingData_->vecSingleNLoop = ops::CeilDiv(totalCubeSingleN, vecSingleN);
    tilingData_->vecSingleNTailLoop =
        ops::CeilDiv(Mc2CalcTailSize(matmulInfoPtr_->nSize, cubeBaseN_ * tilingData_->cubeBlockDimN), vecSingleN);
    tilingData_->vecSingleKLoop = ops::CeilDiv(matmulInfoPtr_->kSize, vecSingleK);

    tilingData_->vecBlockDimK = 1;
    uint64_t taskNum = tilingData_->vecSingleNLoop * tilingData_->vecSingleKLoop;
    uint64_t singleCoreVecLoop = ops::CeilDiv(taskNum, static_cast<uint64_t>(compileInfoPtr_->aivNum));
    tilingData_->vecBlockDimN = ops::CeilDiv(taskNum, singleCoreVecLoop);
    return ge::GRAPH_SUCCESS;
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::SetShapeSize()
{
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    uint64_t weightBlockAlignSize = Mc2GetBlockAlignSizeByDataType(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->transB) {
        tilingData_->kAlign = ops::CeilAlign(matmulInfoPtr_->kSize, weightBlockAlignSize);
        if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
            tilingData_->nAlign = ops::CeilAlign(matmulInfoPtr_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
        } else {
            tilingData_->nAlign = matmulInfoPtr_->nSize;
        }
        tilingData_->kPadSize = static_cast<uint8_t>(tilingData_->kAlign - matmulInfoPtr_->kSize);
    } else {
        if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
            tilingData_->kAlign = ops::CeilAlign(matmulInfoPtr_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
        } else {
            tilingData_->kAlign = matmulInfoPtr_->kSize;
        }
        tilingData_->nAlign = ops::CeilAlign(matmulInfoPtr_->nSize, weightBlockAlignSize);
        tilingData_->nPadSize = static_cast<uint8_t>(tilingData_->nAlign - matmulInfoPtr_->nSize);
    }
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->kSize = matmulInfoPtr_->kSize;
    // weightquantbmmv2 not support batch dims
    tilingData_->haveBatchA = 0;
    tilingData_->haveBatchB = 0;
    tilingData_->shapeBatch = 1;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingCustom::InstantiateTilingData()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2TilingData);
    if (tilingData_ == nullptr) {
        OP_TILING_CHECK(isOutTilingData_, OP_LOGE_WITH_INVALID_INPUT(opName_, "out tilingData"),
                        return ge::GRAPH_FAILED);
        tilingDataManager_ = std::unique_ptr<Mc2WeightQuantBatchMatmulV2TilingData>(
            new (std::nothrow) Mc2WeightQuantBatchMatmulV2TilingData());
        tilingData_ = tilingDataManager_.get();
    }
    OP_TILING_CHECK(tilingData_ == nullptr, OP_LOGE(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize,
                    OP_LOGE(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                            context_->GetRawTilingData()->GetCapacity(), tilingDataSize),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool Mc2WeightQuantBatchMatmulV2TilingCustom::GetMatMulTiling()
{
    if (!GetTilingFromCache() && !InvokeCacheTiling()) {
        auto mmInputDtype = Mc2GetMatmulTilingDtype(matmulInfoPtr_->aDtype);
        auto mmOutputDtype = Mc2GetMatmulTilingDtype(matmulInfoPtr_->cDtype);
        matmul_tiling::MultiCoreMatmulTiling mmTiling;
        matmul_tiling::CubeFormat bCubeFormat = (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) ?
                                                    matmul_tiling::CubeFormat::NZ :
                                                    matmul_tiling::CubeFormat::ND;
        mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmInputDtype,
                          matmulInfoPtr_->transA);
        mmTiling.SetBType(matmul_tiling::TPosition::GM, bCubeFormat, mmInputDtype, matmulInfoPtr_->transB);
        mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmOutputDtype);
        mmTiling.SetBias(matmulInfoPtr_->hasBias);
        if (matmulInfoPtr_->hasBias) {
            auto mmBiasDtype = Mc2GetMatmulTilingDtype(matmulInfoPtr_->biasDtype);
            mmTiling.SetBiasType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, mmBiasDtype);
        }
        mmTiling.SetDim(compileInfoPtr_->aicNum);
        // 转置场景内轴256对齐
        uint64_t kAlignSize = !matmulInfoPtr_->transB ? tilingData_->kAlign :
                                                        ops::CeilAlign(tilingData_->kSize, static_cast<uint64_t>(256));
        if (kAlignSize >= MAX_SHAPE_DIM) {
            kAlignSize = tilingData_->kSize;
        }
        mmTiling.SetOrgShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize, kAlignSize);
        mmTiling.SetShape(matmulInfoPtr_->mSize, matmulInfoPtr_->nSize, matmulInfoPtr_->kSize);
        mmTiling.SetSingleRange(-1, -1, -1, -1, -1, matmulInfoPtr_->kSize);
        mmTiling.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        OP_TILING_CHECK(mmTiling.GetTiling(tilingData_->matmulTiling) == -1,
                        OP_LOGE(matmulInfoPtr_->opName, "failed to get matmul tiling"), return false);

        auto mDim = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.singleCoreM));
        auto nDim = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.singleCoreN));
        OP_TILING_CHECK(mDim * nDim != static_cast<uint64_t>(tilingData_->matmulTiling.usedCoreNum),
                        OP_LOGE(matmulInfoPtr_->opName, "mDim(%lu) * nDim(%lu) != usedCoreNum(%d)", mDim, nDim,
                                tilingData_->matmulTiling.usedCoreNum),
                        return false);
        tilingData_->cubeBlockDimN = static_cast<uint8_t>(nDim);
        tilingData_->cubeBlockDimM = static_cast<uint8_t>(mDim);
    }
    AdjustMatmulTiling();

    uint64_t singleCoreN = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->cubeBlockDimN));
    tilingData_->matmulTiling.singleCoreN = tilingData_->matmulTiling.baseN;
    cubeBaseN_ = static_cast<uint64_t>(tilingData_->matmulTiling.baseN);
    auto nDim = ops::CeilDiv(matmulInfoPtr_->nSize, ops::CeilAlign(singleCoreN, cubeBaseN_));
    tilingData_->cubeBlockDimN = static_cast<uint8_t>(nDim);
    return true;
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::AdjustMatmulTiling() const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ || matmulInfoPtr_->transB) {
        return;
    }
    int32_t baseN = tilingData_->matmulTiling.baseN;
    int32_t minCubeBaseN = ONE_BLK_SIZE;
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        minCubeBaseN = minCubeBaseN << 1;
    }
    if ((baseN * tilingData_->cubeBlockDimN) % minCubeBaseN != 0) {
        tilingData_->matmulTiling.baseN = std::max(ops::FloorAlign(baseN, minCubeBaseN), minCubeBaseN);
        int32_t baseK = tilingData_->matmulTiling.baseK;
        if (tilingData_->matmulTiling.baseN > baseN) {
            // baseN小于32，被向上对齐了，K要相应缩小并且向下对齐到16
            tilingData_->matmulTiling.baseK =
                std::max(ops::FloorAlign(tilingData_->matmulTiling.baseK / TILING_COMPENSATION_FACTOR,
                                         static_cast<int32_t>(BLOCK_CUBE)),
                         static_cast<int32_t>(BLOCK_CUBE));
        }
        if (baseK == tilingData_->matmulTiling.baseK) {
            // kl0没有缩小，就要缩小kL1; 如果stepKb为1时无法调整stepKb，改成调整stepN
            if (tilingData_->matmulTiling.stepKb == 1) {
                tilingData_->matmulTiling.stepN =
                    std::max(tilingData_->matmulTiling.stepN / TILING_COMPENSATION_FACTOR, 1);
            } else {
                tilingData_->matmulTiling.stepKb =
                    std::max(tilingData_->matmulTiling.stepKb / TILING_COMPENSATION_FACTOR, 1);
            }
            tilingData_->matmulTiling.depthB1 =
                std::max(tilingData_->matmulTiling.depthB1 / TILING_COMPENSATION_FACTOR, 1);
            if (tilingData_->matmulTiling.stepKb > tilingData_->matmulTiling.stepKa &&
                tilingData_->matmulTiling.stepKb % tilingData_->matmulTiling.stepKa != 0 &&
                tilingData_->matmulTiling.stepKb * baseK < static_cast<int32_t>(tilingData_->kSize)) {
                tilingData_->matmulTiling.stepKb =
                    ops::FloorAlign(tilingData_->matmulTiling.stepKb, tilingData_->matmulTiling.stepKa);
            }
            if (tilingData_->matmulTiling.stepKa > tilingData_->matmulTiling.stepKb &&
                tilingData_->matmulTiling.stepKa % tilingData_->matmulTiling.stepKb != 0 &&
                tilingData_->matmulTiling.stepKa * baseK < static_cast<int32_t>(tilingData_->kSize)) {
                tilingData_->matmulTiling.stepKa =
                    ops::FloorAlign(tilingData_->matmulTiling.stepKa, tilingData_->matmulTiling.stepKb);
            }
        } else {
            // kl0缩小了，相应的L1上k一定没全载，stepM和stepN只能为1
            tilingData_->matmulTiling.depthB1 = tilingData_->matmulTiling.depthB1 / tilingData_->matmulTiling.stepN;
            tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.depthA1 / tilingData_->matmulTiling.stepM;
            tilingData_->matmulTiling.stepM = 1;
            tilingData_->matmulTiling.stepN = 1;
        }
        AdjustL1Size();
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::AdjustL1Size() const
{
    // 如果调整完之后l1size还是大于l1空间，则缩小stepM和depthA1
    uint64_t a1Length = static_cast<uint64_t>(Mc2GetShapeSizeWithDataType(
        tilingData_->matmulTiling.baseM * tilingData_->matmulTiling.baseK, matmulInfoPtr_->aDtype));
    uint64_t b1Length = static_cast<uint64_t>(Mc2GetShapeSizeWithDataType(
        tilingData_->matmulTiling.baseN * tilingData_->matmulTiling.baseK, matmulInfoPtr_->aDtype));
    uint64_t aL1Size = a1Length * tilingData_->matmulTiling.depthA1;
    uint64_t bL1Size = b1Length * tilingData_->matmulTiling.depthB1;
    uint64_t biasL1Size = matmulInfoPtr_->hasBias ?
                              Mc2GetShapeSizeWithDataType(tilingData_->matmulTiling.baseN, matmulInfoPtr_->biasDtype) :
                              0;
    uint64_t l1Size = aL1Size + bL1Size + biasL1Size;
    if (l1Size > aicoreParams_.l1Size) {
        tilingData_->matmulTiling.stepM = tilingData_->matmulTiling.stepM / TILING_COMPENSATION_FACTOR;
        tilingData_->matmulTiling.depthA1 = tilingData_->matmulTiling.depthA1 / TILING_COMPENSATION_FACTOR;
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeDefaultBlock(uint64_t &defaultVecSingleK,
                                                                  uint64_t &defaultVecSingleN)
{
    uint64_t defaultInnerAxis = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    uint64_t defaultOutterAxis = 32;

    // 非group场景，一次求解即可
    if (matmulInfoPtr_->transB) {
        defaultVecSingleN = defaultOutterAxis;
        // 保证mte2的带宽，根据weight的数据类型，默认载入量取512和1024
        defaultVecSingleK = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    } else {
        // weight不转置场景，n轴取值为cube一轮计算的n轴
        uint64_t weightInnerAxisAlignSize = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);

        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            // int4场景, 内轴shape按照2倍的ONE_BLK_SIZE对齐
            weightInnerAxisAlignSize = ONE_BLK_SIZE * 2;
        }
        defaultVecSingleN = std::min(defaultInnerAxis,
                                     ops::CeilAlign(cubeBaseN_ * tilingData_->cubeBlockDimN, weightInnerAxisAlignSize));
        defaultVecSingleK = defaultOutterAxis;
    }
    ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeGroupDefaultBlock(uint64_t &defaultVecSingleK,
                                                                       uint64_t &defaultVecSingleN)
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        defaultVecSingleK = CUSTOM_NZ_TRANS_BF16_BASE_K;
        if (!matmulInfoPtr_->transB && tilingData_->nSize > INT16_MAX) {
            defaultVecSingleK = matmulInfoPtr_->groupSize;
        }
        if (matmulInfoPtr_->aDtype == ge::DT_BF16 && matmulInfoPtr_->transB) {
            defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
        } else {
            defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
        }
        return;
    }
    uint64_t defaultInnerAxis = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
    uint64_t defaultOutterAxis = 32;
    if (matmulInfoPtr_->transB) {
        uint32_t repeatStrideMax = 255;
        uint32_t repeatAxisMax = repeatStrideMax * (ONE_BLK_SIZE / sizeof(matmulInfoPtr_->aDtype));
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            repeatAxisMax = repeatStrideMax * (ONE_BLK_SIZE / sizeof(float));
        }
        tilingData_->repeatAxisMax = repeatAxisMax;
        if (tilingData_->kAlign <= repeatAxisMax ||
            (tilingData_->kAlign > repeatAxisMax && matmulInfoPtr_->groupSize <= repeatAxisMax &&
             tilingData_->kAlign % matmulInfoPtr_->groupSize == 0)) {
            // k轴不会导致repeatStride超过限制，或者kAlign满足groupSize对齐的限制。考虑k全载，避免复杂尾块处理
            defaultVecSingleK = tilingData_->kAlign;
            ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        }
        if (defaultVecSingleN == 0) {
            OP_LOGD(opName_, "the K axis cannot full load, current defaultVecSingleK: [%lu], groupSize: [%lu].",
                    defaultVecSingleK, matmulInfoPtr_->groupSize);
            // k无法全载的情况下，需重新设置k轴载入量, 同时保证mte2的带宽，根据weight的数据类型，默认载入量取512和1024
            defaultVecSingleK = matmulInfoPtr_->bDtype == ge::DT_INT8 ? 512 : 1024;
            if (defaultVecSingleK >= matmulInfoPtr_->groupSize) {
                defaultVecSingleK = defaultVecSingleK / matmulInfoPtr_->groupSize * matmulInfoPtr_->groupSize;
            }
            ReviseGroupDefaultBlockWithTrans(defaultVecSingleK, defaultVecSingleN);
        }
    } else {
        // weight不转置场景，n轴取值为cube一轮计算的n轴
        uint64_t weightInnerAxisAlignSize = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);

        if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
            // int4场景, 内轴shape按照32Byte的2倍对齐
            weightInnerAxisAlignSize = ONE_BLK_SIZE * 2;
        }
        defaultVecSingleN = std::min(defaultInnerAxis,
                                     ops::CeilAlign(cubeBaseN_ * tilingData_->cubeBlockDimN, weightInnerAxisAlignSize));
        defaultVecSingleK = ops::CeilDiv(defaultOutterAxis, matmulInfoPtr_->groupSize) * matmulInfoPtr_->groupSize;
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        ReviseGroupDefaultBlockWithoutTrans(defaultVecSingleK, defaultVecSingleN);
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ReviseGroupDefaultBlockWithTrans(uint64_t &defaultVecSingleK,
                                                                               uint64_t &defaultVecSingleN)
{
    for (; defaultVecSingleK > matmulInfoPtr_->groupSize; defaultVecSingleK -= matmulInfoPtr_->groupSize) {
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        if (defaultVecSingleN > 0) {
            // n轴大于0,表明该解为合法解，提前退出
            return;
        }
    }

    for (; defaultVecSingleK >= MIN_GROUP_SIZE; defaultVecSingleK -= MIN_GROUP_SIZE) {
        if (matmulInfoPtr_->groupSize % defaultVecSingleK != 0) {
            // 合适的k轴必须满足groupSize_因子的关系
            continue;
        }
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
        if (defaultVecSingleN > 0) {
            // 求得一个合法解，提前退出
            return;
        }
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ReviseGroupDefaultBlockWithoutTrans(uint64_t &defaultVecSingleK,
                                                                                  uint64_t &defaultVecSingleN)
{
    while (defaultVecSingleN > 0) {
        // 若groupSize比MAX_REPEAT_TIMES大，则k对齐到groupSize后必定不满足小于MAX_REPEAT_TIMES的要求，
        // 因此排除这种情况下对k的修正
        if (matmulInfoPtr_->groupSize < MAX_REPEAT_TIMES && defaultVecSingleK >= matmulInfoPtr_->groupSize) {
            // 不转置场景下，k在向groupSize取整后应保证小于MAX_REPEAT_TIMES
            defaultVecSingleK =
                std::min(MAX_REPEAT_TIMES / matmulInfoPtr_->groupSize, defaultVecSingleK / matmulInfoPtr_->groupSize) *
                matmulInfoPtr_->groupSize;
            return;
        }
        for (uint32_t targetK = matmulInfoPtr_->groupSize; targetK >= MIN_GROUP_SIZE; targetK -= MIN_GROUP_SIZE) {
            // 合法的k值在不转置场景下应满足小于MAX_REPEAT_TIMES的限制
            if (targetK > MAX_REPEAT_TIMES) {
                continue;
            }

            // 合法的k值需要满足为groupSize的因子
            if (matmulInfoPtr_->groupSize % targetK != 0) {
                continue;
            }
            if (targetK <= defaultVecSingleK) {
                defaultVecSingleK = targetK;
                return;
            }
        }

        // 无法搜索到满足条件的k值，尝试缩小n重新搜索
        defaultVecSingleN = defaultVecSingleN >> 1;
        ComputeVectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeVectorDefaultBlock(uint64_t &defaultVecSingleK,
                                                                        uint64_t &defaultVecSingleN)
{
    /*
        整体vec处理的基本块推导应该满足如下公式：antiquantBufferSize + weightBufferSize < ubSize
        group场景，固定k轴求n，基本公式化简为：n = ubSize * gs / (antiquantCoefficient * k + weightCoefficient * k *
        gs) 非group场景，固定k轴求n，基本公式进一步化简为：n = ub / (antiquantCoefficient + weightCoefficient * k)
        int4场景，weightCoefficient涉及除2操作，因此先放大weightCoefficient 2倍再除2。避免浮点数的系数影响
    */
    if (matmulInfoPtr_->groupSize > 0 || matmulInfoPtr_->bDtype == ge::DT_INT4) {
        ComputeInt4VectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    } else {
        ComputeInt8VectorDefaultBlock(defaultVecSingleK, defaultVecSingleN);
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeInt4VectorDefaultBlock(uint64_t &defaultVecSingleK,
                                                                            uint64_t &defaultVecSingleN)
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            defaultVecSingleN = 64;  // 固定切分，n取64
            defaultVecSingleK = 256; // 固定切分，k取256
        } else {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = 64;  // 固定切分，n取64
                defaultVecSingleK = 384; // 固定切分，k取384
            } else {
                defaultVecSingleN = 64;  // 固定切分，n取64
                defaultVecSingleK = 256; // 固定切分，k取256
            }
        }
        return;
    }
    uint64_t start = 0;
    uint64_t length = matmulInfoPtr_->kSize + 1;
    if (matmulInfoPtr_->transB) {
        length = matmulInfoPtr_->nSize + 1;
    }

    // 固定内轴的情况下，二分求解最大的外轴是多少
    while (length > 0) {
        uint64_t mid = start + (length >> 1);
        uint64_t antiquantBuffer;
        uint64_t weightBuffer;
        if (matmulInfoPtr_->transB) {
            antiquantBuffer = ComputeAntiquantBuffer(defaultVecSingleK, mid);
            weightBuffer = ComputeWeightBuffer(defaultVecSingleK, mid);
        } else {
            antiquantBuffer = ComputeAntiquantBuffer(mid, defaultVecSingleN);
            weightBuffer = ComputeWeightBuffer(mid, defaultVecSingleN);
        }

        if (aicoreParams_.ubSize < antiquantBuffer + weightBuffer) {
            length = length >> 1;
        } else {
            start = mid + 1;
            length = length - (length >> 1) - 1;
        }
    }

    // start是不满足条件的最小值，因此最终结果需要-1
    if (matmulInfoPtr_->transB) {
        defaultVecSingleN = start - 1;
    } else {
        defaultVecSingleK = start - 1;
    }
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeAntiquantBuffer(uint64_t &defaultVecSingleK,
                                                                         uint64_t &defaultVecSingleN)
{
    uint64_t aDtypeBlockSize = Mc2GetBlockAlignSizeByDataType(matmulInfoPtr_->aDtype);
    uint64_t antiquantSize = ops::CeilAlign(defaultVecSingleN, aDtypeBlockSize);
    if (matmulInfoPtr_->groupSize > 0) {
        if (matmulInfoPtr_->transB) {
            if (defaultVecSingleK >= matmulInfoPtr_->kSize) {
                // 全载场景，antiquant的n*gourpCount合并成一根轴计算
                antiquantSize = ops::CeilAlign(
                    ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize) * defaultVecSingleN, aDtypeBlockSize);
            } else {
                // 非全载场景，antiquant的shape只能当作(n, gourpCount)计算，同时考虑内轴对齐
                antiquantSize =
                    defaultVecSingleN *
                    ops::CeilAlign(ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize), aDtypeBlockSize);
            }
        } else {
            // 不转置场景，antiquant的shape只能当作(gourpCount，n)计算，同时考虑内轴对齐
            antiquantSize = ops::CeilDiv(defaultVecSingleK, matmulInfoPtr_->groupSize) *
                            ops::CeilAlign(defaultVecSingleN, aDtypeBlockSize);
        }
    }

    // scale和offset两个入参，需要占用2份空间
    uint64_t antiquantParamsCount = 2;
    uint64_t antiquantInQueSize = antiquantParamsCount * antiquantSize * sizeof(matmulInfoPtr_->aDtype);
    if (matmulInfoPtr_->transB) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            return antiquantInQueSize + antiquantSize * sizeof(float) +
                   antiquantParamsCount * antiquantSize * ONE_BLK_SIZE;
        } else {
            return antiquantInQueSize + antiquantParamsCount * antiquantSize * ONE_BLK_SIZE;
        }
    } else {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            return antiquantInQueSize + antiquantSize * sizeof(float) +
                   antiquantParamsCount * antiquantSize * sizeof(float);
        } else {
            return antiquantInQueSize + antiquantParamsCount * antiquantSize * sizeof(matmulInfoPtr_->aDtype);
        }
    }
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeWeightBuffer(uint64_t defaultVecSingleK,
                                                                      uint64_t defaultVecSingleN)
{
    uint64_t originWeightAlignAxis = ONE_BLK_SIZE / sizeof(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        // int4场景，内轴的长度是32Byte的2倍
        originWeightAlignAxis = ONE_BLK_SIZE * 2;
    }

    uint64_t weightShape;
    if (matmulInfoPtr_->transB) {
        weightShape = defaultVecSingleN * ops::CeilAlign(defaultVecSingleK, originWeightAlignAxis);
    } else {
        weightShape = defaultVecSingleK * ops::CeilAlign(defaultVecSingleN, originWeightAlignAxis);
    }
    uint64_t originWeightSize = weightShape * sizeof(matmulInfoPtr_->bDtype);
    if (matmulInfoPtr_->bDtype == ge::DT_INT4) {
        originWeightSize = originWeightSize >> 1;
    }
    uint64_t weight16Size = weightShape * sizeof(matmulInfoPtr_->aDtype);
    uint64_t weight32Size = weightShape * sizeof(float);
    // 输出的buffer共有2份，方便开db
    uint64_t weightOutSize = 2 * weight16Size;

    if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
        return originWeightSize + weight16Size + weight32Size + weightOutSize;
    } else {
        return originWeightSize + weight16Size + weightOutSize;
    }
}

void Mc2WeightQuantBatchMatmulV2TilingCustom::ComputeInt8VectorDefaultBlock(uint64_t &defaultVecSingleK,
                                                                            uint64_t &defaultVecSingleN) const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ) {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            if (matmulInfoPtr_->transB) {
                // 需要11nk + 76n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=32
                defaultVecSingleN = 32;
                defaultVecSingleK = BASIC_BLOCK;
            } else {
                // 需要11nk + 12n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=32
                defaultVecSingleK = 32;
                defaultVecSingleN = BASIC_BLOCK;
            }
        } else {
            if (matmulInfoPtr_->transB) {
                // 需要7nk + 68n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=40
                defaultVecSingleN = 40;
                defaultVecSingleK = BASIC_BLOCK;
            } else {
                // 需要7nk + 4n空间，内轴优先512B对齐，根据k=512和总UB空间，计算出n=48
                defaultVecSingleK = 48;
                defaultVecSingleN = BASIC_BLOCK;
            }
        }
    } else {
        if (matmulInfoPtr_->aDtype == ge::DT_BF16) {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_TRANS_BF16_BASE_K;
            } else {
                defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_NO_TRANS_BF16_BASE_N;
            }
        } else {
            if (matmulInfoPtr_->transB) {
                defaultVecSingleN = CUSTOM_NZ_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_TRANS_FP16_BASE_K;
            } else {
                defaultVecSingleN = CUSTOM_NZ_NO_TRANS_BASE_N;
                defaultVecSingleK = CUSTOM_NZ_NO_TRANS_FP16_BASE_K;
            }
        }
    }
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingCustom::DoLibApiTiling()
{
    uint64_t cubeBlockDimN = static_cast<uint64_t>(tilingData_->cubeBlockDimN);
    uint64_t cubeEachCoreN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, cubeBlockDimN), cubeBaseN_);
    tilingData_->cubeSingleNLoop = ops::CeilDiv(cubeEachCoreN, cubeBaseN_);
    tilingData_->cubeSingleNTailLoop =
        ops::CeilDiv(matmulInfoPtr_->nSize - cubeEachCoreN * (cubeBlockDimN - 1), cubeBaseN_);
    tilingData_->cubeTailM =
        Mc2CalcTailSize(matmulInfoPtr_->mSize, static_cast<uint64_t>(tilingData_->matmulTiling.singleCoreM));
    tilingData_->cubeTailN =
        Mc2CalcTailSize(matmulInfoPtr_->nSize, static_cast<uint64_t>(tilingData_->matmulTiling.baseN));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingCustom::GetWorkspaceSize()
{
    // weight的缓存最多使用3份空间，实际划分少于3时以实际划分为准
    uint64_t weightCacheCount = std::min(static_cast<uint32_t>(3), tilingData_->cubeSingleNLoop);
    uint64_t weightCacheNSize = tilingData_->matmulTiling.singleCoreN * tilingData_->cubeBlockDimN;
    if (!matmulInfoPtr_->transB && matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        weightCacheNSize = ops::CeilAlign(weightCacheNSize, static_cast<uint64_t>(ONE_BLK_SIZE));
    }
    uint64_t weightCacheSize = tilingData_->kAlign * weightCacheNSize;
    if (matmulInfoPtr_->transB) {
        // 内轴需256对齐以提高nd2nz效率
        weightCacheSize = ops::CeilAlign(tilingData_->kSize, static_cast<uint64_t>(256)) *
                          tilingData_->matmulTiling.singleCoreN * tilingData_->cubeBlockDimN;
    }
    // 向256对齐，可以保证workspace起始地址保证512B对齐，提升mte3性能
    uint64_t weightCacheAlignSize = ops::CeilDiv(weightCacheSize, static_cast<uint64_t>(256)) * 256;
    workspaceSize_ = weightCacheAlignSize * weightCacheCount * ge::GetSizeByDataType(matmulInfoPtr_->aDtype) +
                     compileInfoPtr_->workspaceNum;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Mc2WeightQuantBatchMatmulV2TilingCustom::PostTiling()
{
    size_t tilingDataSize = sizeof(Mc2WeightQuantBatchMatmulV2TilingData);
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize);

    OP_TILING_CHECK(tilingDataSize % sizeof(uint64_t) != 0,
                    OP_LOGE(opName_, "tiling data size[%zu] not aligned to 8", tilingDataSize),
                    return ge::GRAPH_FAILED);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    uint32_t usedAicNum = tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN;
    uint32_t usedAivNum = tilingData_->vecBlockDimK * tilingData_->vecBlockDimN;
    context_->SetBlockDim(
        std::max(usedAicNum, CalcTschBlockDim(usedAivNum, compileInfoPtr_->aicNum, compileInfoPtr_->aivNum)));

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    workspaces[0] = workspaceSize_;

    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           tilingData_, tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t Mc2WeightQuantBatchMatmulV2TilingCustom::GetTilingKey() const
{
    Mc2KernelTemplateType templateType = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ ?
                                             Mc2KernelTemplateType::WEIGHT_NZ :
                                             Mc2KernelTemplateType::CUSTOM_ANTIQUANT;
    uint64_t socVersionType = 1UL; // 1 means SUPPORT_L0C_TO_OUT
    uint64_t subSocVersionType = 0UL;
    uint64_t antiquantScenario = 0UL;
    uint64_t algorithm = 3UL; // 3 means CUSTOM tilingkey algorithm
    uint64_t subAlgorithm = 0UL;
    uint64_t subAlgorithmCustom = static_cast<uint64_t>(templateType);
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
    bool isWeightNz = false;      // weightNz according to subAlgorithmCustom templateType
    uint64_t templateExtra = 3UL; // 3 means TEMPLATE_EXTRA_NOT_USED
    uint64_t fullLoadMode = 5UL;  // 5 means FULL_LOAD_MODE_NOT_USED
    uint64_t batch = 0UL;
    uint64_t tilingKey_ = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                             subAlgorithm, subAlgorithmCustom, innerPrecise, templateCustom,
                                             apiConstexpr, transA, transB, antiquantType, quantType, hasAntiquantOffset,
                                             hasBias, isBiasFp32, isWeightNz, templateExtra, fullLoadMode, batch);
    return tilingKey_;
}

bool Mc2WeightQuantBatchMatmulV2TilingCustom::GetTilingFromCache()
{
    bool isNzFormat = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ;
    uint64_t mMatchSize = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(BLOCK_CUBE));
    WeightQuantBatchMatmulCacheTilingParas paras({mMatchSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize,
                                                  matmulInfoPtr_->hasBias, matmulInfoPtr_->transA,
                                                  matmulInfoPtr_->transB, isNzFormat, compileInfoPtr_->aicNum});
    WeightQuantBatchMatmulCacheTilingData matmulTilingCache;
    if (!GenWqbmmTiling(WQBMM_CUSTOM, paras, matmulTilingCache)) {
        OP_LOGD(opName_, "not find mm tiling from cache");
        return false;
    }

    OP_LOGD(opName_, "get mm tiling from cache");
    Mc2SetMatmulTilingFromCacheData(matmulTilingCache, tilingData_->matmulTiling, matmulInfoPtr_->mSize,
                                    matmulInfoPtr_->nSize, static_cast<int32_t>(matmulInfoPtr_->hasBias));

    tilingData_->cubeBlockDimM = matmulTilingCache.mDim_;
    tilingData_->cubeBlockDimN = matmulTilingCache.nDim_;
    return true;
}

bool Mc2WeightQuantBatchMatmulV2TilingCustom::CheckCacheTiling()
{
    if (matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ) {
        int32_t kAL1Loop = ops::CeilDiv(tilingData_->matmulTiling.singleCoreK,
                                        tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.stepKa);
        int32_t kBL1Loop = ops::CeilDiv(tilingData_->matmulTiling.singleCoreK,
                                        tilingData_->matmulTiling.baseK * tilingData_->matmulTiling.stepKb);
        if (kAL1Loop == 0 || kBL1Loop == 0) {
            return false;
        }
        if (kAL1Loop % kBL1Loop != 0 && kBL1Loop % kAL1Loop != 0) {
            return false;
        }
    }
    // 拦截分核数小于0.5倍总核数的解
    OP_TILING_CHECK(tilingData_->cubeBlockDimM * tilingData_->cubeBlockDimN < 0.5 * compileInfoPtr_->aicNum,
                    OP_LOGI(opName_, "Current cache tiling result is aborted for insufficient core use"), return false);

    OP_LOGD(opName_, "get and convert cache tiling success");
    return true;
}

bool Mc2WeightQuantBatchMatmulV2TilingCustom::InvokeCacheTiling()
{
    Mc2MatmulMultiCoreResult multiCoreResult;
    bool result = Mc2ComputeMatmulTiling::GetTiling(
        tilingData_->matmulTiling, multiCoreResult,
        {matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize, matmulInfoPtr_->aDtype,
         matmulInfoPtr_->bDtype, matmulInfoPtr_->cDtype, matmulInfoPtr_->biasDtype, matmulInfoPtr_->transA,
         matmulInfoPtr_->transB, matmulInfoPtr_->hasBias, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
         matmulInfoPtr_->quantType, true},
        aicoreParams_, context_);

    OP_LOGI_IF_RETURN(!result, false, opName_, "cannot get tiling from cachetiling, mnk[%lu, %lu, %lu]",
                      matmulInfoPtr_->mSize, matmulInfoPtr_->kSize, matmulInfoPtr_->nSize);

    tilingData_->cubeBlockDimM = static_cast<uint8_t>(multiCoreResult.mDim);
    tilingData_->cubeBlockDimN = static_cast<uint8_t>(multiCoreResult.nDim);
    tilingData_->blockBatch = static_cast<uint8_t>(multiCoreResult.batchDim);

    return CheckCacheTiling();
}
} // namespace optiling
