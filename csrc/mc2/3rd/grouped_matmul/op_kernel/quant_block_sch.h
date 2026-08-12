/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file quant_block_sch.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_QUANT_BLOCK_SCH_H
#define MC2_GROUPED_MATMUL_QUANT_BLOCK_SCH_H

#include "quant_utils.h"

namespace Mc2GroupedMatmul {
struct ASWTilingParam {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t singleCoreM;
    uint64_t singleCoreN;
    uint64_t mCnt;
    uint64_t nCnt;
    uint64_t totalCnt;
    uint64_t mCoreNum;
    uint64_t mTailCoreNum;
    uint64_t mBaseTail;
    uint64_t nBaseTail;
    uint64_t mIndex;
    uint64_t nIndex;
    uint64_t mSplitAddrOffset;
    uint64_t nSplitAddrOffset;
    uint64_t aGroupAddrOffset;
    uint64_t bGroupAddrOffset;
    uint64_t cGroupAddrOffset;
    uint64_t xScaleGroupAddrOffset;
    uint64_t wScaleGroupAddrOffset;
    uint64_t biasGroupAddrOffset;
    uint64_t mTailTile;
    uint64_t nTailTile;
    uint64_t mainRow;
    uint64_t round;
    uint64_t index;
};

// 对于每一个GlobalTensor的偏移
struct ASWOffsetParam {
    uint64_t offsetA;
    uint64_t offsetB;
    uint64_t offsetC;
    uint64_t offsetScale;
    uint64_t offsetBias;
    uint64_t offsetPerTokenScale;
};

class QuantASWBlockSch {
public:
    __aicore__ inline QuantASWBlockSch() {}
    template <bool isGmm> __aicore__ inline void Init(const TCubeTiling *__restrict &tilingData, uint32_t blockIdx);
    __aicore__ inline void SetRankFirstLayout(uint32_t rankDim, const __gm__ uint64_t *cGroupOffsetTableGm)
    {
        rankDim_ = rankDim;
        cGroupOffsetTableGm_ = cGroupOffsetTableGm;
    }
    __aicore__ inline void SetAGroupOffsetTable(const __gm__ uint64_t *aGroupOffsetTableGm)
    {
        aGroupOffsetTableGm_ = aGroupOffsetTableGm;
    }
    __aicore__ inline void SetXScaleGroupOffsetTable(const __gm__ uint64_t *xScaleGroupOffsetTableGm)
    {
        xScaleGroupOffsetTableGm_ = xScaleGroupOffsetTableGm;
    }
    template <bool aTrans, bool bTrans, class xType, class scaleType, CubeFormat wFormat = CubeFormat::ND>
    __aicore__ inline void UpdateGroupOffset(int32_t m, int32_t n, int32_t k, uint32_t groupIdx);
    template <bool isGmm> __aicore__ inline void UpdateGroupParams();
    __aicore__ inline void UpdateTailTile();
    template <bool isGmm>
    __aicore__ inline void UpdateBasicIndex(uint64_t roundIdx, bool isLastGroupRound);
    template <bool aTrans, bool bTrans, CubeFormat wFormat = CubeFormat::ND>
    __aicore__ inline void UpdateBlockParams(uint64_t roundIdx, bool isLastGroupRound = true);
    template <bool aTrans, bool bTrans, class scaleType = AscendC::fp8_e8m0_t, CubeFormat wFormat = CubeFormat::ND>
    __aicore__ inline void CalcGMOffset();
    __aicore__ inline void ResetAddressOffsets();
    __aicore__ inline uint32_t GetStartBlockIdx() const;
    __aicore__ inline uint32_t GetEndBlockIdx() const;

public:
    ASWTilingParam params_;
    ASWOffsetParam offset_;
    const TCubeTiling *__restrict tilingData_;

private:
    const uint64_t WINDOW_LEN = 4;
    uint32_t blockIdx_;
    uint32_t startBlockIdx_;
    uint32_t endBlockIdx_;
    uint32_t rankDim_ = 1U;
    const __gm__ uint64_t *cGroupOffsetTableGm_ = nullptr;
    const __gm__ uint64_t *aGroupOffsetTableGm_ = nullptr;
    const __gm__ uint64_t *xScaleGroupOffsetTableGm_ = nullptr;
};

template <bool isGmm>
__aicore__ inline void QuantASWBlockSch::Init(const TCubeTiling *__restrict &tilingData, uint32_t blockIdx)
{
    blockIdx_ = blockIdx;
    tilingData_ = tilingData;
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
    params_.xScaleGroupAddrOffset = 0; // xScale is optional, mm不需要，与GMM代码归一
    params_.mTailTile = 1; // 1 说明不切分, mm可直接从mm tiling里获取，但目前仓不同，无法拿到
    params_.nTailTile = 1; // 1 说明不切分
    startBlockIdx_ = 0;    // 每个group核使用开始的索引, 单mm默认从0开始

    if constexpr (isGmm) { // GMM
        params_.m = 0;
        params_.n = 0;
        params_.k = 0;
        // 不同group的偏移量, 在dynamic输入输出的取二级指针偏移所需，对于globalTensor偏移不需要
        params_.aGroupAddrOffset = 0;
        params_.bGroupAddrOffset = 0;
        params_.cGroupAddrOffset = 0;
        params_.wScaleGroupAddrOffset = 0;
        params_.biasGroupAddrOffset = 0;
        // 标记每个group的结束核
        endBlockIdx_ = tilingData_->usedCoreNum - 1; // 上个group核使用结束的索引
    } else {                                         // MM
        params_.m = tilingData_->M;                  // mm可以直接从tilingData里取值
        params_.n = tilingData_->N;
        params_.k = tilingData_->Ka;
        UpdateGroupParams<false>();
    }
}

template <bool aTrans, bool bTrans, class xType, class scaleType, CubeFormat wFormat>
__aicore__ inline void QuantASWBlockSch::UpdateGroupOffset(int32_t m, int32_t n, int32_t k, uint32_t groupIdx)
{
    if (groupIdx > 0) {
        if (aGroupOffsetTableGm_ != nullptr) {
            params_.aGroupAddrOffset = aGroupOffsetTableGm_[groupIdx];
        } else {
            if constexpr (Mc2QuantUtils::IsFp4<xType>()) {
                params_.aGroupAddrOffset += params_.m * params_.k / 2;
            } else {
                params_.aGroupAddrOffset += params_.m * params_.k;
            }
        }

        bool isExpertBoundary = (rankDim_ <= 1U) || (groupIdx % rankDim_ == 0U);
        if constexpr (Mc2QuantUtils::IsFp4<xType>()) {
            if (isExpertBoundary) {
                params_.bGroupAddrOffset += params_.n * params_.k / 2;
            }
        } else {
            if constexpr (wFormat == CubeFormat::NZ) {
                if (isExpertBoundary) {
                    if constexpr (bTrans) {
                        params_.bGroupAddrOffset +=
                            Mc2QuantUtils::CeilDiv(params_.k, Mc2QuantUtils::WEIGHTNZ_K0_32) *
                            Mc2QuantUtils::CeilDiv(params_.n, Mc2QuantUtils::WEIGHTNZ_N0_16) *
                            Mc2QuantUtils::WEIGHTNZ_N0_K0;
                    } else {
                        params_.bGroupAddrOffset +=
                            Mc2QuantUtils::CeilDiv(params_.n, Mc2QuantUtils::WEIGHTNZ_N0_32) *
                            Mc2QuantUtils::CeilDiv(params_.k, Mc2QuantUtils::WEIGHTNZ_K0_16) *
                            Mc2QuantUtils::WEIGHTNZ_N0_K0;
                    }
                }
            } else {
                if (isExpertBoundary) {
                    params_.bGroupAddrOffset += params_.n * params_.k;
                }
            }
        }

        if (rankDim_ > 1U && cGroupOffsetTableGm_ != nullptr) {
            params_.cGroupAddrOffset = cGroupOffsetTableGm_[groupIdx];
        } else {
            params_.cGroupAddrOffset += params_.m * params_.n;
        }

        if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
            uint64_t scaleK = Mc2QuantUtils::MXFP_MULTI_BASE_SIZE;
            if constexpr (!aTrans) {
                scaleK *= Mc2QuantUtils::CeilDiv(params_.k, Mc2QuantUtils::MXFP_DIVISOR_SIZE);
                if (xScaleGroupOffsetTableGm_ != nullptr) {
                    params_.xScaleGroupAddrOffset = xScaleGroupOffsetTableGm_[groupIdx];
                } else {
                    params_.xScaleGroupAddrOffset += params_.m * scaleK;
                }
                if (isExpertBoundary) {
                    params_.wScaleGroupAddrOffset += params_.n * scaleK;
                }
            } else if constexpr (aTrans && !bTrans) {
                scaleK *= (params_.bGroupAddrOffset / params_.n / Mc2QuantUtils::MXFP_DIVISOR_SIZE + groupIdx);
                params_.xScaleGroupAddrOffset = params_.m * scaleK;
                params_.wScaleGroupAddrOffset = params_.n * scaleK;
            }
        } else {
            params_.xScaleGroupAddrOffset += params_.m;
            if (isExpertBoundary) {
                params_.wScaleGroupAddrOffset += params_.n;
            }
        }
        if (isExpertBoundary) {
            params_.biasGroupAddrOffset += params_.n;
        }
    } else {
        if (rankDim_ > 1U && cGroupOffsetTableGm_ != nullptr) {
            params_.cGroupAddrOffset = cGroupOffsetTableGm_[0];
        }
        if (aGroupOffsetTableGm_ != nullptr) {
            params_.aGroupAddrOffset = aGroupOffsetTableGm_[0];
        }
        if (xScaleGroupOffsetTableGm_ != nullptr) {
            params_.xScaleGroupAddrOffset = xScaleGroupOffsetTableGm_[0];
        }
    }

    params_.m = m;
    params_.n = n;
    params_.k = k;
}

// 兼容GMM和MM的更新
template <bool isGmm>
__aicore__ inline void QuantASWBlockSch::UpdateGroupParams()
{
    params_.mCnt = Mc2QuantUtils::CeilDiv(params_.m, tilingData_->baseM);
    params_.nCnt = Mc2QuantUtils::CeilDiv(params_.n, tilingData_->baseN);
    params_.totalCnt = params_.mCnt * params_.nCnt;
    params_.mBaseTail = params_.m - (params_.mCnt - 1) * tilingData_->baseM;
    params_.nBaseTail = params_.n - (params_.nCnt - 1) * tilingData_->baseN;
    params_.mCoreNum = Mc2QuantUtils::Min(WINDOW_LEN, params_.mCnt);
    // 计算round数还是按照实际，使用核数按照startBlockIdx开始
    params_.round = Mc2QuantUtils::CeilDiv(params_.totalCnt, tilingData_->usedCoreNum);
    params_.mainRow = params_.mCnt / params_.mCoreNum - 1;
    params_.mTailCoreNum = params_.mCnt - params_.mCoreNum * params_.mainRow;
    if constexpr (isGmm) {
        // 计算当前group的mm最后一轮计算空闲的核的索引
        // 新group开始的空闲的核索引
        startBlockIdx_ = endBlockIdx_ == tilingData_->usedCoreNum - 1 ? 0 : (endBlockIdx_ + 1);
        // 当前group结束的空闲的核索引
        endBlockIdx_ = (params_.totalCnt + startBlockIdx_ - 1) % tilingData_->usedCoreNum;
        // 如果当前group不是最后一个group，则空闲核不参与最后一轮计算，需要留给下一个group使用
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            params_.round -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            params_.round -= 1;
        }
    }
}

// 尾块切分后，更新结束的核索引和round数
__aicore__ inline void QuantASWBlockSch::UpdateTailTile()
{
    uint64_t newEndBlockIdx = params_.mTailTile * params_.nTailTile * (endBlockIdx_ + 1) - 1;
    if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
        params_.round += 1;
    }
    endBlockIdx_ = newEndBlockIdx;
}

template <bool isGmm>
__aicore__ inline void QuantASWBlockSch::UpdateBasicIndex(uint64_t roundIdx, bool isLastGroupRound)
{
    uint64_t newBlockIdx = isLastGroupRound ? (blockIdx_ / (params_.mTailTile * params_.nTailTile)) : blockIdx_;
    params_.index = newBlockIdx + roundIdx * tilingData_->usedCoreNum;
    // GMM当前group的startBlockIdx_不一定从0开始，要进行计算过的base块数修正
    if constexpr (isGmm) {
        if (blockIdx_ < startBlockIdx_) {
            params_.index += tilingData_->usedCoreNum - startBlockIdx_; // 加上最开始从startBlockIdx_开始的base数
        } else {
            params_.index -= startBlockIdx_; // 减去roundIdx * CoreNum的未包含的最开始0-(startBlockIdx_-1)的base数
        }
    }
    uint64_t rowIdx = params_.index / params_.nCnt / params_.mCoreNum;
    if (rowIdx < params_.mainRow) {
        params_.mIndex = rowIdx * params_.mCoreNum + params_.index % params_.mCoreNum;
        params_.nIndex = (params_.index / params_.mCoreNum) % params_.nCnt;
    } else {
        rowIdx = params_.mainRow;
        uint64_t tailIndex = params_.index - params_.mainRow * params_.mCoreNum * params_.nCnt;
        params_.mIndex = params_.mainRow * params_.mCoreNum + tailIndex % params_.mTailCoreNum;
        params_.nIndex = (tailIndex / params_.mTailCoreNum) % params_.nCnt;
    }

    if (rowIdx & 1) {
        params_.nIndex = params_.nCnt - 1 - params_.nIndex;
    }
}

// 返回给kernel的开始空闲核索引，帮助最后一个group的尾块细分
__aicore__ inline uint32_t QuantASWBlockSch::GetStartBlockIdx() const
{
    return startBlockIdx_;
}

// 返回给kernel的开始空闲核索引，帮助最后一个group的尾块细分
__aicore__ inline uint32_t QuantASWBlockSch::GetEndBlockIdx() const
{
    return endBlockIdx_;
}

template <bool aTrans, bool bTrans, CubeFormat wFormat>
__aicore__ inline void QuantASWBlockSch::UpdateBlockParams(uint64_t roundIdx, bool isLastGroupRound)
{
    params_.singleCoreM = params_.mIndex != (params_.mCnt - 1) ? tilingData_->baseM : params_.mBaseTail;
    params_.singleCoreN = params_.nIndex != (params_.nCnt - 1) ? tilingData_->baseN : params_.nBaseTail;
    if (!isLastGroupRound || (params_.mTailTile == 1 && params_.nTailTile == 1)) {
        return;
    }

    if (roundIdx == params_.round - 1) {
        uint64_t singleCoreMSplit = (params_.singleCoreM + params_.mTailTile - 1) / params_.mTailTile;
        uint64_t singleCoreNSplit = (params_.singleCoreN + params_.nTailTile - 1) / params_.nTailTile;
        if constexpr (aTrans) { // (k, m)
            singleCoreMSplit = Mc2QuantUtils::Align(singleCoreMSplit, Mc2QuantUtils::INNER_AXIS_MIN_SPLIT_VAL);
        }
        if constexpr (!bTrans) { // (k, n)
            singleCoreNSplit = Mc2QuantUtils::Align(singleCoreNSplit, Mc2QuantUtils::INNER_AXIS_MIN_SPLIT_VAL);
        } else {
            if constexpr (wFormat == CubeFormat::NZ) {
                singleCoreNSplit = Mc2QuantUtils::Align(singleCoreNSplit, Mc2QuantUtils::WEIGHTNZ_N0_16);
            }
        }

        uint64_t totalTailTile = params_.mTailTile * params_.nTailTile;
        uint64_t mSplitIdx = (blockIdx_ % totalTailTile) % params_.mTailTile;
        uint64_t nSplitIdx = (blockIdx_ % totalTailTile) / params_.mTailTile;
        params_.mSplitAddrOffset = mSplitIdx * singleCoreMSplit;
        params_.nSplitAddrOffset = nSplitIdx * singleCoreNSplit;
        if (params_.mSplitAddrOffset >= params_.singleCoreM || params_.nSplitAddrOffset >= params_.singleCoreN) {
            params_.singleCoreM = 0;
            params_.singleCoreN = 0;
            return;
        }
        if (params_.mSplitAddrOffset + singleCoreMSplit > params_.singleCoreM) {
            params_.singleCoreM = params_.singleCoreM - singleCoreMSplit * mSplitIdx;
        } else {
            params_.singleCoreM = singleCoreMSplit;
        }
        if (params_.nSplitAddrOffset + singleCoreNSplit > params_.singleCoreN) {
            params_.singleCoreN = params_.singleCoreN - singleCoreNSplit * nSplitIdx;
        } else {
            params_.singleCoreN = singleCoreNSplit;
        }
    }
}

__aicore__ inline void QuantASWBlockSch::ResetAddressOffsets()
{
    // 尾块细分的m方向偏移量和n方向偏移量需要重置
    params_.mSplitAddrOffset = 0;
    params_.nSplitAddrOffset = 0;
}

template <bool aTrans, bool bTrans, class scaleType, CubeFormat wFormat>
__aicore__ inline void QuantASWBlockSch::CalcGMOffset()
{
    uint64_t mOffset = params_.mIndex * tilingData_->baseM + params_.mSplitAddrOffset;
    uint64_t nOffset = params_.nIndex * tilingData_->baseN + params_.nSplitAddrOffset;
    if constexpr (aTrans) {
        offset_.offsetA = mOffset;
    } else {
        offset_.offsetA = mOffset * params_.k;
    }

    if constexpr (wFormat == CubeFormat::NZ) {
        if constexpr (bTrans) {
            offset_.offsetB = nOffset * Mc2QuantUtils::WEIGHTNZ_K0_32;
        } else {
            offset_.offsetB = nOffset * Mc2QuantUtils::CeilDiv(params_.k, Mc2QuantUtils::WEIGHTNZ_K0_16) *
                              Mc2QuantUtils::WEIGHTNZ_K0_16;
        }
    } else {
        if constexpr (bTrans) {
            offset_.offsetB = nOffset * params_.k;
        } else {
            offset_.offsetB = nOffset;
        }
    }

    offset_.offsetC = mOffset * params_.n + nOffset;

    if constexpr (Mc2QuantUtils::IsMxType<scaleType>()) {
        uint64_t pertokenScaleK = Mc2QuantUtils::MXFP_MULTI_BASE_SIZE;
        uint64_t scaleK = Mc2QuantUtils::MXFP_MULTI_BASE_SIZE;
        if constexpr (!aTrans) { // mx (m, ceil(k / 64), 2)
            pertokenScaleK *= Mc2QuantUtils::CeilDiv(params_.k, Mc2QuantUtils::MXFP_DIVISOR_SIZE);
        }
        if constexpr (bTrans) { // mx (n, ceil(k / 64), 2)
            scaleK *= Mc2QuantUtils::CeilDiv(params_.k, Mc2QuantUtils::MXFP_DIVISOR_SIZE);
        }
        offset_.offsetPerTokenScale = mOffset * pertokenScaleK;
        offset_.offsetScale = nOffset * scaleK;
    } else {
        offset_.offsetPerTokenScale = mOffset;
        offset_.offsetScale = nOffset;
    }
    // pertoken是optional，不是dynamic，要累加分组的偏移量
    offset_.offsetPerTokenScale += params_.xScaleGroupAddrOffset;
    offset_.offsetBias = nOffset;
}
} // namespace Mc2GroupedMatmul
#endif // MC2_GROUPED_MATMUL_QUANT_BLOCK_SCH_H
