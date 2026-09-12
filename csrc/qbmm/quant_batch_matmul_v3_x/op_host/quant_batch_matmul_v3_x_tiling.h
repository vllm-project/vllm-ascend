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
 * \file quant_batch_matmul_v3_x_tiling.h
 * \brief TilingData declarations for QuantBatchMatmulV3X (Ascend 310P).
 * \author Feodor Pisnitchenko
 */
#ifndef QUANT_BATCH_MATMUL_V3_X_TILING_H
#define QUANT_BATCH_MATMUL_V3_X_TILING_H

#include <exe_graph/runtime/tiling_context.h>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

// NOTE: the tiling data struct must be FLAT. The tbe opc compile extracts the
// struct definition from REGISTER_TILING_DATA_CLASS to generate the
// GET_TILING_DATA_WITH_STRUCT codegen; nested TILING_DATA_FIELD_DEF_STRUCT
// entries are not exported and the kernel compile then lacks the tiling
// macros entirely.
BEGIN_TILING_DATA_DEF(QBMTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, M);
TILING_DATA_FIELD_DEF(uint32_t, K);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, hasPertoken);
TILING_DATA_FIELD_DEF(uint32_t, quantGroupNum);
TILING_DATA_FIELD_DEF(uint32_t, maxOneTurnToken);
TILING_DATA_FIELD_DEF(uint32_t, maxWeightColOneTurn);
TILING_DATA_FIELD_DEF(uint32_t, baseK);
TILING_DATA_FIELD_DEF(uint32_t, scaleType);
TILING_DATA_FIELD_DEF(uint32_t, isPerTensor);
TILING_DATA_FIELD_DEF(uint32_t, phaseXChunk);
TILING_DATA_FIELD_DEF(uint32_t, phaseXFullPairs);
TILING_DATA_FIELD_DEF(uint32_t, phaseXTail);
TILING_DATA_FIELD_DEF(uint32_t, MCoreNum);      // M-axis cores in the 2-D grid
TILING_DATA_FIELD_DEF(uint32_t, NCoreNum);      // N-axis cores in the 2-D grid
TILING_DATA_FIELD_DEF(uint32_t, wChunkKPasses); // kPasses fetched per MTE2 into wL1
TILING_DATA_FIELD_DEF(uint32_t, scaleCoalesce); // 0 = per-ch scale MTE2, 1 = one-per-cb coalesced
TILING_DATA_FIELD_DEF(uint32_t, l2HintMode);    // CacheMode int (0=DISABLE,1=NORMAL,4=PERSISTENT)
TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);    // # M-axis super-tiles (1 = no split)
TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);    // # N-axis super-tiles (1 = no split)
TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);    // baseM blocks per M super-tile
TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);    // baseN blocks per N super-tile
TILING_DATA_FIELD_DEF(uint32_t, dbL0c);         // 1 = L0C ping-pong (2*baseM*baseN*4 <= L0C); 0 = single
TILING_DATA_FIELD_DEF(uint32_t, pertokenCoalesce); // 1 = single B*M MTE2 at Process entry; 0 = per-mTile
TILING_DATA_FIELD_DEF(uint32_t, hasBias);          // 1 = bias[N] int32 broadcast into L0C before Mmad
TILING_DATA_FIELD_DEF(uint32_t, kTail);            // K % 32; non-zero requires partial-K0-group zero-pad in Phase X
// ubCalcM: inner-tile Phase D control (Q == 1 only).
//   0                       -> disabled (single drain of full mAligned).
//   0 < ubCalcM < mAligned   -> wrap the cb-loop tail in
//                                mUbLoops = ceildiv(mAligned, ubCalcM)
//                                slices; per-mu pertoken Mul writes
//                                copyOutBuf in-place so the next mu's
//                                partial VDEQ16 does not race with the
//                                still-in-flight MTE3 CopyOut.
TILING_DATA_FIELD_DEF(uint32_t, ubCalcM);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatmulV3X, QBMTilingData)

struct QBMCompileInfo {
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
};

// Extract N from x2's shape. Works with gert::Shape and ge::Shape (both
// expose GetDimNum and GetDim). FRACTAL_NZ (dimNum>=4): [..., K1, N1, N0, K0]
// so N = N1 * N0. ND (dimNum<4): [..., K, N] or [..., N, K] if transposed.
template <typename ShapeT>
inline int64_t DeriveN(const ShapeT& x2Shape, bool transposeX2)
{
    auto dimNum = x2Shape.GetDimNum();
    if (dimNum >= 4) {
        return static_cast<int64_t>(x2Shape.GetDim(dimNum - 2)) *
               static_cast<int64_t>(x2Shape.GetDim(dimNum - 3));
    }
    return transposeX2
        ? x2Shape.GetDim(dimNum - 2)
        : x2Shape.GetDim(dimNum - 1);
}

}  // namespace optiling

#endif  // QUANT_BATCH_MATMUL_V3_X_TILING_H
