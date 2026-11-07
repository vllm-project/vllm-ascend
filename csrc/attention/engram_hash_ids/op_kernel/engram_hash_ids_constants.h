// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <cstdint>

namespace EngramHashIds {
enum InputIndex {
    INPUT_IDS,
    TOKEN_MAP,
    DEAD_MASK,
    POSITIONS,
    BLOCK_TABLE,
    QUERY_START_LOC,
    REQ_IDS,
    MULTIPLIERS,
    PRIMES,
    OFFSETS,
    CACHE,
    LOOKBACK_TOKEN_IDS,
    LOOKBACK_DEAD_MASK,
};

enum AttrIndex { PARAMETERS };
// Scalar attributes are packed in the public interface order.
enum ParameterIndex {
    NUM_TOKENS,
    NUM_SLOTS,
    PAD_ID,
    INPUT_STRIDE,
    MASK_STRIDE,
    POSITION_STRIDE,
    TABLE_STRIDE,
    TABLE_COL_STRIDE,
    QUERY_STRIDE,
    NUM_TABLE_ROWS,
    MAX_BLOCKS,
    CACHE_BLOCK_SIZE,
    MAX_NGRAM,
    NUM_HEADS,
    DEAD_ID,
    LOOKBACK_DEPTH,
    LOOKBACK_ROW_STRIDE,
    LOOKBACK_COL_STRIDE,
    LOOKBACK_MASK_ROW_STRIDE,
    LOOKBACK_MASK_COL_STRIDE,
    NUM_LAYERS,
    PARAMETER_COUNT,
};

constexpr int64_t TOKEN_DIM = 0;
constexpr int64_t LAYER_DIM = 0;
constexpr int64_t NGRAM_DIM = 1;
constexpr int64_t HEAD_DIM = 2;
constexpr int64_t ROW_DIM = 0;
constexpr int64_t COL_DIM = 1;
constexpr int64_t VECTOR_RANK = 1;
constexpr int64_t MATRIX_RANK = 2;
constexpr int64_t PARAM_RANK = 3;
constexpr int64_t UNKNOWN_TOKEN_ID = -1;
// UB capacity stays fixed; host selects the effective task size from these tiers.
constexpr uint32_t TOKEN_TILE_ELEMENTS = 128;
constexpr uint32_t MEDIUM_TOKEN_TILE_ELEMENTS = 64;
constexpr uint32_t SMALL_TOKEN_TILE_ELEMENTS = 32;
constexpr uint32_t HEAD_TILE_ELEMENTS = 8;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t OUTPUT_BLOCK_ELEMENTS = DATA_BLOCK_BYTES / sizeof(int32_t);
constexpr uint64_t MAX_DMA_STRIDE_BYTES = UINT32_MAX;
constexpr uint32_t COPY_BLOCK_COUNT = 1;
constexpr uint32_t COPY_EVENT_ID = 0;
constexpr uint64_t TILING_KEY = 0;
} // namespace EngramHashIds
