/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #ifndef SOLVE_TRI_TILING_H
 #define SOLVE_TRI_TILING_H

 #include "register/tilingdata_base.h"

 namespace optiling {

 BEGIN_TILING_DATA_DEF(SolveTriTilingData)
     TILING_DATA_FIELD_DEF(int64_t, totalTiles);
     TILING_DATA_FIELD_DEF(int64_t, matrixSize);
     TILING_DATA_FIELD_DEF(int64_t, numHeads);
     TILING_DATA_FIELD_DEF(int64_t, seqLen);
     TILING_DATA_FIELD_DEF(int64_t, batchSize);
     TILING_DATA_FIELD_DEF(int64_t, isLower);
     TILING_DATA_FIELD_DEF(int64_t, hasCuSeqlens);
     TILING_DATA_FIELD_DEF(int64_t, tilesPerCore);
     TILING_DATA_FIELD_DEF(int64_t, chunkSize);
     TILING_DATA_FIELD_DEF(int64_t, numChunks);
     TILING_DATA_FIELD_DEF(int64_t, lastChunkValidSize);
     TILING_DATA_FIELD_DEF(int64_t, isVarlen);
     TILING_DATA_FIELD_DEF(int64_t, totalChunks);
     TILING_DATA_FIELD_DEF(int64_t, layoutMode);
     TILING_DATA_FIELD_DEF(int64_t, dtypeMode);  // 0=fp16, 1=bf16
 END_TILING_DATA_DEF;

 REGISTER_TILING_DATA_CLASS(SolveTri, SolveTriTilingData)

 }  // namespace optiling

 #endif
