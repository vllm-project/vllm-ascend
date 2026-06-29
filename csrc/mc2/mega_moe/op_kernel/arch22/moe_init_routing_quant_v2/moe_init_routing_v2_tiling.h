/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v2_tiling.h
 * \brief
 */

#ifndef MC2_MOE_INIT_ROUTING_V2_QUANT_TILING_H
#define MC2_MOE_INIT_ROUTING_V2_QUANT_TILING_H

#include "../moe_init_routing_v2/moe_init_routing_v2_tiling.h"
#include <cmath>

namespace Mc2Tiling {

// Quant-specific inner types wrapping the canonical Moe* definitions
// from moe_init_routing_v2_tiling.h

struct InnerMoeV2VBSComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElements = 0;
};

struct InnerMoeV2VMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};

struct InnerMoeV2SortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};

struct InnerMoeV2GatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t activateRows = 0;
    int64_t perCoreRows = 0;
    int64_t perCorePerLoopRows = 0;
    int64_t perCoreLastLoopRows = 0;
    int64_t lastCoreRows = 0;
    int64_t lastCorePerLoopRows = 0;
    int64_t lastCoreLastLoopRows = 0;
    int64_t perCoreLoops = 0;
    int64_t lastCoreLoops = 0;
    int64_t perLoopCols = 0;
    int64_t lastLoopCols = 0;
    int64_t colLoops = 0;
};

struct InnerMoeInitRoutingV2TilingData {
    int64_t coreNum;
    int64_t n;
    int64_t cols;
    int64_t k;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    int64_t expertTokensBeforeCapacityFlag;
    InnerMoeV2VBSComputeTilingData vbsComputeParamsOp;
    InnerMoeV2VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    InnerMoeV2SortOutComputeTilingData sortOutComputeParamsOp;
    InnerMoeV2GatherOutComputeTilingData srcToDstComputeParamsOp;
    InnerMoeV2GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    InnerMoeV2GatherOutComputeTilingData gatherOutComputeParamsOp;
};


class InnerMoeInitRoutingV2TilingBase : public TilingBaseClass {
protected:
    inline bool GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) override;
    inline bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, int64_t expertNum,
                                  int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
                                  bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode,
                                  int64_t scaleDim0) override;

    inline bool DoOpTiling() override;
    inline uint64_t GetTilingKey() const override;
    inline bool GetWorkspaceSize() override;

protected:
    bool CheckTokenCount(int64_t num, const char *tag);

    virtual void Tiling4GatherOutCompute() = 0;
    inline void Tiling4SrcToDstCompute();
    virtual void Tiling4SrcToDstCapacityCompute();
    inline void Tiling4SortOutCompute();
    inline void Tiling4VMSMiddleCompute();
    inline void Tiling4VBSCompute();
    void ShowTilingData();
    inline void Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData *tilingData);
    inline void Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData *tilingData);
    virtual bool IsFullLoad() = 0;

    int64_t aivNum = 0;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 2040;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = 0;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t inuptXDtypeSize_ = 0;
    bool isFullLoad = false;

    InnerMoeInitRoutingV2TilingData moeInitRoutingTilingData;
};


inline bool InnerMoeInitRoutingV2TilingBase::DoOpTiling()
{
    sortLoopMaxElement =
        (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    isFullLoad = IsFullLoad();
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4SrcToDstCapacityCompute();
    Tiling4GatherOutCompute();
    return true;
};

inline uint64_t InnerMoeInitRoutingV2TilingBase::GetTilingKey() const
{
    if (isFullLoad) {
        return TILING_KEY_HIGH_PERFORMANCE;
    }
    if (dropPadMode == 0) {
        if (totalLength <= sortLoopMaxElement) {
            return TILING_KEY_DROPLESS_SORT_ONE_CORE;
        } else {
            return TILING_KEY_DROPLESS_SORT_MULTI_CORE;
        }
    } else {
        if (totalLength <= sortLoopMaxElement) {
            return TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;
        } else {
            return TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE;
        }
    }
    return tilingKey_;
}

inline bool InnerMoeInitRoutingV2TilingBase::GetShapeAttrsInfo(
    int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, int64_t expertNum, int64_t activateNum,
    int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag,
    int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0)
{
    this->activateNum = activateNum;
    this->expertCapacity = expertCapacity;
    this->expertNum = expertNum;
    this->dropPadMode = dropPadMode;
    this->expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
    if (dropPadMode == 1) {
        expertTokensCountOrCumsumFlag = 0;
    } else {
        expertTokensBeforeCapacityFlag = false;
    }
    moeInitRoutingTilingData.cols = cols;
    moeInitRoutingTilingData.n = m;
    moeInitRoutingTilingData.k = topK;
    moeInitRoutingTilingData.expertCapacity = expertCapacity;
    moeInitRoutingTilingData.expertNum = expertNum;
    moeInitRoutingTilingData.dropPadMode = dropPadMode;
    moeInitRoutingTilingData.expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
    moeInitRoutingTilingData.expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
    totalLength = moeInitRoutingTilingData.n * moeInitRoutingTilingData.k;
    inuptXDtypeSize_ = inuptXDtypeSize;
    return true;
}

inline bool InnerMoeInitRoutingV2TilingBase::GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm)
{
    aivNum = aivCoreNum;
    aicoreParams_.blockDim = aivCoreNum;
    aicoreParams_.ubSize = ubSizePlatForm;
    moeInitRoutingTilingData.coreNum = aivCoreNum;
    return true;
}


inline bool InnerMoeInitRoutingV2TilingBase::GetWorkspaceSize()
{
    size_t sortWorkspaceSize = totalLength * sizeof(float) * NUM_TWO * NUM_THREE;
    size_t scatterWorkspaceSize = totalLength * sizeof(int32_t) * NUM_TWO;
    size_t expertTokenFlagSize = aivNum * 2 * sizeof(int32_t);
    workspaceSize_ =
        sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return true;
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData *tilingData)
{
    tilingData->needCoreNum = 1;
    tilingData->perCoreElements = totalLength;
    tilingData->perCoreLoops = 1;
    tilingData->perCorePerLoopElements = tilingData->perCoreElements;
    tilingData->perCoreLastLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreElements = tilingData->perCoreElements;
    tilingData->lastCoreLoops = 1;
    tilingData->lastCorePerLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreLastLoopElements = tilingData->perCoreElements;
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData *tilingData)
{
    int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, aivNum);
    if (needCoreNum > 0) {
        int64_t perCoreElements = totalLength / needCoreNum;
        int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
        int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
        int64_t alineCeilPerCoreElements =
            perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
        if (lastCoreElement > alineCeilPerCoreElements) {
            perCoreElements = alineCeilPerCoreElements;
            needCoreNum = CeilDiv(totalLength, perCoreElements);
        } else {
            perCoreElements = alineFloorPerCoreElements;
        }
        tilingData->needCoreNum = needCoreNum;
        do {
            tilingData->perCoreElements = perCoreElements;
            tilingData->perCoreLoops = CeilDiv(tilingData->perCoreElements, sortLoopMaxElement);
            tilingData->perCorePerLoopElements = std::min(tilingData->perCoreElements, sortLoopMaxElement);
            tilingData->perCoreLastLoopElements =
                tilingData->perCoreElements - (tilingData->perCoreLoops - 1) * tilingData->perCorePerLoopElements;
            tilingData->lastCoreElements = totalLength - (tilingData->needCoreNum - 1) * tilingData->perCoreElements;
            tilingData->lastCoreLoops = tilingData->perCoreLoops;
            int64_t tmp = CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops);
            int64_t lastCorePerLoopElements =
                CeilDiv(CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops), SORT32_ALIGN_ELEMENT) *
                SORT32_ALIGN_ELEMENT;
            tilingData->lastCorePerLoopElements = lastCorePerLoopElements;
            tilingData->lastCoreLastLoopElements =
                tilingData->lastCoreElements - (tilingData->lastCoreLoops - 1) * tilingData->lastCorePerLoopElements;
            perCoreElements -= SORT32_ALIGN_ELEMENT;
        } while (tilingData->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
    }
}


inline void InnerMoeInitRoutingV2TilingBase::Tiling4VBSCompute()
{
    auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->oneLoopMaxElements = sortLoopMaxElement;
    if (totalLength <= sortLoopMaxElement) {
        Tiling4VBSOneCoreCompute(tilingData);
        return;
    }
    Tiling4VBSMultiCoreCompute(tilingData);
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->needCoreNum <= MRG_LIST_NUM) {
        tilingData->needCoreNum = 0;
    } else {
        int64_t needCoreNum = CeilDiv(vbsComputeTilingData->needCoreNum, MRG_LIST_NUM);
        tilingData->needCoreNum = needCoreNum;
    }
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->oneLoopMaxElements = mrgSortListMaxElement;
}


inline void InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

    int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                             (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->needCoreNum = needCoreNum;
    int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->perCoreRows = perCoreRows;
    if (perLoopMaxRows >= tilingData->perCoreRows) {
        tilingData->perCorePerLoopRows = tilingData->perCoreRows;
        tilingData->perCoreLastLoopRows = tilingData->perCoreRows;
    } else {
        tilingData->perCorePerLoopRows = perLoopMaxRows;
        tilingData->perCoreLastLoopRows =
            tilingData->perCoreRows - (CeilDiv(tilingData->perCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
    }
    tilingData->lastCoreRows = lastCoreNum;
    if (perLoopMaxRows >= tilingData->lastCoreRows) {
        tilingData->lastCorePerLoopRows = tilingData->lastCoreRows;
        tilingData->lastCoreLastLoopRows = tilingData->lastCoreRows;
    } else {
        tilingData->lastCorePerLoopRows = perLoopMaxRows;
        tilingData->lastCoreLastLoopRows =
            tilingData->lastCoreRows - (CeilDiv(tilingData->lastCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
    }
}


inline void InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute()
{
    auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->needCoreNum = needCoreNum;
    int64_t cols = moeInitRoutingTilingData.cols;
    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;
    int64_t rowSize =
        (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        tilingData->perCorePerLoopRows = perCoreRows;
        tilingData->perCoreLastLoopRows = perCoreRows;
        tilingData->lastCorePerLoopRows = lastCoreRows;
        tilingData->lastCoreLastLoopRows = lastCoreRows;
        tilingData->perCoreLoops = 1;
        tilingData->lastCoreLoops = 1;
        tilingData->perLoopCols = cols;
        tilingData->lastLoopCols = cols;
        tilingData->colLoops = 1;
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                     static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE *
                          ONE_BLOCK_BYTE;
        }
        tilingData->perLoopCols = (std::min(baseMaxCols, cols));
        tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
        tilingData->colLoops = ((cols + baseMaxCols - 1) / baseMaxCols);
        tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLoops = ((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
        tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLoops = ((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}

} // namespace Mc2Tiling
#endif // MC2_MOE_INIT_ROUTING_V2_QUANT_TILING_H
