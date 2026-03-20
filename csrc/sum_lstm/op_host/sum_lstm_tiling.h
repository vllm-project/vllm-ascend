#ifndef SUM_LSTM_TILING_H
#define SUM_LSTM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SumLstmTilingData)
    // Basic parameters
    TILING_DATA_FIELD_DEF(uint32_t, totalSamples);      // Total samples (batch * ...)
    TILING_DATA_FIELD_DEF(uint32_t, hiddenDim);         // Hidden dimension D
    TILING_DATA_FIELD_DEF(uint32_t, gatedDim);          // 4D (4 * hiddenDim)

    // Core allocation parameters
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);           // Number of cores used
    TILING_DATA_FIELD_DEF(uint32_t, samplesPerCore);    // Samples per core
    TILING_DATA_FIELD_DEF(uint32_t, remainSamples);     // Remaining samples

    // Tile parameters
    TILING_DATA_FIELD_DEF(uint32_t, tileNumPerCore);    // Tiles per core
    TILING_DATA_FIELD_DEF(uint32_t, tileSamples);       // Samples per tile
    TILING_DATA_FIELD_DEF(uint32_t, lastTileSamples);   // Samples in last tile

    // Alignment parameters
    TILING_DATA_FIELD_DEF(uint32_t, hiddenDimAligned);  // D aligned to 32B (half)
    TILING_DATA_FIELD_DEF(uint32_t, gatedDimAligned);   // 4D aligned to 32B (half)
    TILING_DATA_FIELD_DEF(uint32_t, floatHiddenDimAligned); // D aligned to 32B (float)

    // Operator attributes
    TILING_DATA_FIELD_DEF(float, alpha);                // z tensor scaling factor
    TILING_DATA_FIELD_DEF(float, epsCell);              // Cell RMSNorm epsilon
    TILING_DATA_FIELD_DEF(float, epsState);             // State RMSNorm epsilon
    TILING_DATA_FIELD_DEF(uint32_t, useFastGelu);       // Use fast GELU

    // Optional input flags
    TILING_DATA_FIELD_DEF(uint32_t, hasWCell);          // Has w_cell
    TILING_DATA_FIELD_DEF(uint32_t, hasBCell);          // Has b_cell
    TILING_DATA_FIELD_DEF(uint32_t, hasWState);         // Has w_state
    TILING_DATA_FIELD_DEF(uint32_t, hasBState);         // Has b_state

    // Data type info
    TILING_DATA_FIELD_DEF(uint32_t, dataTypeSize);      // Data type bytes (2 for fp16/bf16)

    // UB memory parameters
    TILING_DATA_FIELD_DEF(uint32_t, ubBufferSize);      // Single buffer size
    TILING_DATA_FIELD_DEF(uint32_t, bufferCount);       // I/O queue buffer depth (1=single, 2=double)
    TILING_DATA_FIELD_DEF(uint32_t, preloadWeights);    // Preload weights (0=per-sample, 1=preload)

END_TILING_DATA_DEF;

// Register Tiling data class
REGISTER_TILING_DATA_CLASS(SumLstm, SumLstmTilingData)

}  // namespace optiling

#endif  // SUM_LSTM_TILING_H
