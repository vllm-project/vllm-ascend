/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_tiling.h
 * \brief scatter_nd_update arch32 tiling data definition
 */

#ifndef SCATTER_ND_UPDATE_ARCH32_TILING_H
#define SCATTER_ND_UPDATE_ARCH32_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

constexpr uint64_t MAX_DIM_NUM = 8;

namespace optiling {

BEGIN_TILING_DATA_DEF(ScatterNdUpdateScatterTiling)
TILING_DATA_FIELD_DEF(uint64_t, scatterLength)
TILING_DATA_FIELD_DEF(uint64_t, tailRow)
TILING_DATA_FIELD_DEF(uint64_t, frontRow)
TILING_DATA_FIELD_DEF(uint64_t, frontNum)
TILING_DATA_FIELD_DEF(uint64_t, tailNum)
TILING_DATA_FIELD_DEF(uint64_t, ubLengthForUpdates)
TILING_DATA_FIELD_DEF(uint64_t, scatterAlignLength)
TILING_DATA_FIELD_DEF(uint64_t, formDim)
TILING_DATA_FIELD_DEF(uint64_t, copyRow)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileNum)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileLength)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileTail)
TILING_DATA_FIELD_DEF(uint64_t, scatterTileAlignLength)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateScatterTilingOp, ScatterNdUpdateScatterTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateLinearIndexTiling)
TILING_DATA_FIELD_DEF(uint64_t, coreNum)
TILING_DATA_FIELD_DEF(uint64_t, ubSize)
TILING_DATA_FIELD_DEF(uint64_t, indexDim)
TILING_DATA_FIELD_DEF(uint64_t, blockLength)
TILING_DATA_FIELD_DEF(uint64_t, blockNum)
TILING_DATA_FIELD_DEF(uint64_t, blockRemainLength)
TILING_DATA_FIELD_DEF(uint64_t, tailBlockNum)
TILING_DATA_FIELD_DEF(uint64_t, frontBlockNum)
TILING_DATA_FIELD_DEF(uint64_t, frontCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, tailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, sortWorkspace)
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_DIM_NUM, indicesMask)
TILING_DATA_FIELD_DEF(uint64_t, isInt64Indices)
TILING_DATA_FIELD_DEF(uint64_t, needLargeIndexKernel)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateLinearIndexTilingOp, ScatterNdUpdateLinearIndexTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateViewTiling)
TILING_DATA_FIELD_DEF(uint64_t, isViewStride0)
TILING_DATA_FIELD_DEF(uint64_t, varStride0Elements)
TILING_DATA_FIELD_DEF(uint64_t, firstDimStrideRows)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateViewTilingOp, ScatterNdUpdateViewTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateHpTiling)
TILING_DATA_FIELD_DEF(uint64_t, hpCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, hpFrontIndexNum)
TILING_DATA_FIELD_DEF(uint64_t, hpTailIndexNum)
TILING_DATA_FIELD_DEF(uint64_t, hpFrontCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, hpTailCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, hpIndexTileLength)
TILING_DATA_FIELD_DEF(uint64_t, hpScatterTileLength)
TILING_DATA_FIELD_DEF(uint64_t, hpScatterTileNum)
TILING_DATA_FIELD_DEF(uint64_t, hpScatterTileTail)
TILING_DATA_FIELD_DEF(uint64_t, hpRowBytesAligned)
TILING_DATA_FIELD_DEF(uint64_t, hpRowsPerBatch)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdateHpTilingOp, ScatterNdUpdateHpTiling)

BEGIN_TILING_DATA_DEF(ScatterNdUpdateArch32TilingData)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateScatterTiling, scatterTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateLinearIndexTiling, linearIndexTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateHpTiling, hpTiling)
TILING_DATA_FIELD_DEF_STRUCT(ScatterNdUpdateViewTiling, viewTiling)
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(ScatterNdUpdate, ScatterNdUpdateArch32TilingData)
REGISTER_TILING_DATA_CLASS(ScatterNdUpdateTilingDataOp, ScatterNdUpdateArch32TilingData)

struct ScatterNdUpdateArch32CompileInfo {
    uint32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};

class ScatterNdUpdateArch32Tiling {
public:
    explicit ScatterNdUpdateArch32Tiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline bool IsSort(uint64_t totalLength) const;
    inline bool IsLinearIndex(uint64_t totalLength) const;
    inline size_t CalcWorkSpaceSize(uint64_t indexRow);
    inline void SetTilingKeyMode();
    inline void GetDtypeSize();
    inline void Tiling4Scatter(uint64_t totalLength);
    inline void Tiling4LinearIndex(uint64_t indexRow, uint64_t indexDim);
    inline void Tiling4Hp(uint64_t indexRow);
    inline uint64_t Tiling4HpScatterShape();
    inline void Tiling4HpIndexTile(uint64_t updateUbBytes);
    inline void Tiling4HpCorePartition(uint64_t indexRow);
    inline ge::graphStatus HandleViewStride();

    ScatterNdUpdateArch32TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;

    uint64_t coreNum_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t isLinearIndex_ = false;
    uint64_t isSort_ = false;
    uint64_t sortWorkspace_ = 0;
    uint64_t dataTypeSize_ = 0;
    uint64_t isInt64Indices_ = false;
    uint64_t needLargeIndexKernel_ = false;
    uint64_t isViewStride0_ = 0;
    uint64_t varStride0Elements_ = 0;
    uint64_t firstDimStrideRows_ = 1;

private:
    // LinearIndex
    uint64_t indexDim_ = 0;
    uint64_t blockLength_ = 0;
    uint64_t blockNum_ = 0;
    uint64_t blockRemainLength_ = 0;
    uint64_t tailBlockNum_ = 0;
    uint64_t frontBlockNum_ = 0;
    uint64_t frontCoreNum_ = 0;
    uint64_t tailCoreNum_ = 0;
    uint64_t indicesMask_[MAX_DIM_NUM] = {0};

    // Scatter
    uint64_t scatterLength_ = 1;
    uint64_t tailRow_ = 0;
    uint64_t frontRow_ = 0;
    uint64_t frontNum_ = 0;
    uint64_t tailNum_ = 0;
    uint64_t ubLengthForUpdates_ = 0;
    uint64_t scatterAlignLength_ = 0;
    uint64_t formDim_ = 0;
    uint64_t copyRow_ = 0;
    uint64_t scatterTileNum_ = 1;
    uint64_t scatterTileLength_ = 0;
    uint64_t scatterTileTail_ = 0;
    uint64_t scatterTileAlignLength_ = 0;

    // HighPerformance
    uint64_t hpCoreNum_ = 0;
    uint64_t hpFrontIndexNum_ = 0;
    uint64_t hpTailIndexNum_ = 0;
    uint64_t hpFrontCoreNum_ = 0;
    uint64_t hpTailCoreNum_ = 0;
    uint64_t hpIndexTileLength_ = 0;
    uint64_t hpScatterTileLength_ = 0;
    uint64_t hpScatterTileNum_ = 1;
    uint64_t hpScatterTileTail_ = 0;
    uint64_t hpRowBytesAligned_ = 0;
    uint64_t hpRowsPerBatch_ = 1;
};
} // namespace optiling
#endif // SCATTER_ND_UPDATE_ARCH32_TILING_H
