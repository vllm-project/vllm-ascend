#include <pto/pto-inst.hpp>

#include "int4_cvt.hpp"

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// Keep some slack below PTO's TMP_UB_OFFSET (184KB) while increasing the
// batched x tile a bit to improve samples_per_load for smaller block widths.
constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t UB_HALF_BYTES = X_BUFFER_BYTES / 2;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t Y_BUFFER_BYTES = ELEMENTS_PER_TILE / 2;
constexpr uint32_t UB_USABLE_BYTES = 184 * 1024;
// The public dynamic fused wrapper routes NPU execution only for n >= 64, so
// the largest reachable batched row count per load is 16384 / 64 = 256.
constexpr uint32_t MAX_DYNAMIC_SAMPLES_PER_LOAD = ELEMENTS_PER_TILE / 64;
constexpr unsigned X_PING = 0x00000;
constexpr unsigned Y_PING = X_PING + X_BUFFER_BYTES + 0x100;
constexpr unsigned X_PONG = Y_PING + Y_BUFFER_BYTES + 0x100;
constexpr unsigned Y_PONG = X_PONG + X_BUFFER_BYTES + 0x100;
constexpr unsigned EVEN_BASE = Y_PONG + Y_BUFFER_BYTES + 0x100;
constexpr unsigned ODD_BASE = EVEN_BASE + UB_HALF_BYTES + 0x100;
constexpr unsigned SCALE_BASE = ODD_BASE + UB_HALF_BYTES + 0x100;
constexpr unsigned REDUCE_TMP_BASE = SCALE_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(float) + 0x100;
constexpr unsigned ROWMAX_BASE = REDUCE_TMP_BASE + X_BUFFER_BYTES + 0x100;
constexpr unsigned ROWMIN_BASE =
    ROWMAX_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(half) + 0x100;
static_assert(ODD_BASE + UB_HALF_BYTES <= UB_USABLE_BYTES,
              "Fused Hadamard+quantize UB layout exceeds usable UB.");
static_assert(SCALE_BASE +
                  MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(float) <=
              UB_USABLE_BYTES,
              "Dynamic quant scale UB layout exceeds usable UB.");
static_assert(REDUCE_TMP_BASE + X_BUFFER_BYTES <= UB_USABLE_BYTES,
              "Dynamic quant reduce-temp UB layout exceeds usable UB.");
static_assert(ROWMAX_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(half) <=
                  UB_USABLE_BYTES,
              "Dynamic quant row-max UB layout exceeds usable UB.");
static_assert(ROWMIN_BASE + MAX_DYNAMIC_SAMPLES_PER_LOAD * sizeof(half) <=
                  UB_USABLE_BYTES,
              "Dynamic quant row-min UB layout exceeds usable UB.");

#define FAST_HADAMARD_BATCHED_CASES(X) \
  X(64, 6)                             \
  X(128, 7)                            \
  X(256, 8)                            \
  X(512, 9)                            \
  X(1024, 10)                          \
  X(2048, 11)                          \
  X(4096, 12)                          \
  X(8192, 13)                          \
  X(16384, 14)

namespace {

struct TileWork {
  uint32_t gm_offset, sample_count, elements;
};

template <typename InputT, uint32_t kN, uint32_t kLog2N>
AICORE void runBatchedHadamardInPlace(unsigned x_base, uint32_t sample_count) {
  constexpr uint32_t kNHalf = kN >> 1;
  constexpr uint32_t kSamplesPerLoad = ELEMENTS_PER_TILE / kN;

  using FullTile = Tile<TileType::Vec, InputT, kSamplesPerLoad, kN,
                        BLayout::RowMajor, DYNAMIC, kN>;
  using HalfTile = Tile<TileType::Vec, InputT, kSamplesPerLoad, kNHalf,
                        BLayout::RowMajor, DYNAMIC, kNHalf>;
  using RowHalfTile =
      Tile<TileType::Vec, InputT, 1, kNHalf, BLayout::RowMajor, 1, kNHalf>;

  FullTile xBulkTile(sample_count);
  HalfTile evenTile(sample_count);
  HalfTile oddTile(sample_count);
  TASSIGN(xBulkTile, x_base);
  TASSIGN(evenTile, EVEN_BASE);
  TASSIGN(oddTile, ODD_BASE);

  for (uint32_t iter_m = 0; iter_m < kLog2N; ++iter_m) {
    TGATHER<HalfTile, FullTile, MaskPattern::P0101>(evenTile, xBulkTile);
    TGATHER<HalfTile, FullTile, MaskPattern::P1010>(oddTile, xBulkTile);

    pipe_barrier(PIPE_V);

    for (uint32_t s = 0; s < sample_count; ++s) {
      const unsigned row_base = x_base + s * kN * sizeof(InputT);
      const unsigned even_row_base = EVEN_BASE + s * kNHalf * sizeof(InputT);
      const unsigned odd_row_base = ODD_BASE + s * kNHalf * sizeof(InputT);

      RowHalfTile evenRow;
      RowHalfTile oddRow;
      RowHalfTile xFirstHalf;
      RowHalfTile xSecondHalf;
      TASSIGN(evenRow, even_row_base);
      TASSIGN(oddRow, odd_row_base);
      TASSIGN(xFirstHalf, row_base);
      TASSIGN(xSecondHalf, row_base + kNHalf * sizeof(InputT));

      TADD(xFirstHalf, evenRow, oddRow);
      TSUB(xSecondHalf, evenRow, oddRow);
    }

    pipe_barrier(PIPE_V);
  }
}

template <typename InputT>
AICORE void issueTLoad(__gm__ InputT *x, const TileWork &tile, unsigned x_base,
                       event_t ev) {
  using InShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using InGlobal = pto::GlobalTensor<InputT, InShapeDim5, StridDim5>;
  using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                        BLayout::RowMajor, -1, -1>;

  FullTile xBulkTile(1, tile.elements);
  TASSIGN(xBulkTile, x_base);

  InGlobal xGlobal(x + tile.gm_offset);
  TASSIGN(xGlobal, (x + tile.gm_offset));

  wait_flag(PIPE_V, PIPE_MTE2, ev);
  TLOAD(xBulkTile, xGlobal);
  set_flag(PIPE_MTE2, PIPE_V, ev);
}

template <typename InputT>
AICORE void issueSerializedTLoad(__gm__ InputT *x, const TileWork &tile,
                                 unsigned x_base, event_t ev) {
  // The validated small-row path avoids UB x-buffer reuse by alternating
  // X_PING/X_PONG. The oversized-row helper reuses a single x slot, so it must
  // explicitly wait until vector compute has released that slot back to MTE2
  // before issuing the next TLOAD into the same address range.
  wait_flag(PIPE_V, PIPE_MTE2, ev);
  issueTLoad(x, tile, x_base, ev);
}

AICORE bool nextTile(uint32_t &sample_done, uint32_t gm_offset_base,
                     uint32_t samples_to_process, uint32_t samples_per_load,
                     uint32_t n, TileWork &tile) {
  if (sample_done >= samples_to_process) {
    return false;
  }

  tile.sample_count = min(samples_per_load, samples_to_process - sample_done);
  tile.elements = tile.sample_count * n;
  tile.gm_offset = gm_offset_base + sample_done * n;
  sample_done += tile.sample_count;
  return true;
}

template <typename InputT>
AICORE bool tryRunBatchedHadamard(unsigned x_base, uint32_t sample_count,
                                  uint32_t n, uint32_t log2_n) {
  switch (n) {
#define FAST_HADAMARD_BATCHED_DISPATCH_CASE(N, LOG2)                    \
  case N:                                                               \
    if (log2_n == LOG2) {                                               \
      runBatchedHadamardInPlace<InputT, N, LOG2>(x_base, sample_count); \
      return true;                                                      \
    }                                                                   \
    break;
    FAST_HADAMARD_BATCHED_CASES(FAST_HADAMARD_BATCHED_DISPATCH_CASE)
#undef FAST_HADAMARD_BATCHED_DISPATCH_CASE
    default:
      break;
  }
  return false;
}

template <typename InputT>
AICORE void runSingleHadamardRow(unsigned row_base, uint32_t n,
                                 uint32_t log2_n) {
  const uint32_t n_half = n >> 1;
  using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                        BLayout::RowMajor, -1, -1>;
  using HalfTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE / 2,
                        BLayout::RowMajor, -1, -1>;

  FullTile xRowTile(1, n);
  HalfTile xFirstHalf(1, n_half);
  HalfTile xSecondHalf(1, n_half);
  HalfTile evenTile(1, n_half);
  HalfTile oddTile(1, n_half);
  TASSIGN(xRowTile, row_base);
  TASSIGN(xFirstHalf, row_base);
  TASSIGN(xSecondHalf, row_base + n_half * sizeof(InputT));
  TASSIGN(evenTile, EVEN_BASE);
  TASSIGN(oddTile, ODD_BASE);

  for (uint32_t iter_m = 0; iter_m < log2_n; ++iter_m) {
    TGATHER<HalfTile, FullTile, MaskPattern::P0101>(evenTile, xRowTile);
    TGATHER<HalfTile, FullTile, MaskPattern::P1010>(oddTile, xRowTile);

    pipe_barrier(PIPE_V);

    TADD(xFirstHalf, evenTile, oddTile);
    TSUB(xSecondHalf, evenTile, oddTile);

    pipe_barrier(PIPE_V);
  }
}

template <typename InputT>
AICORE void runRowBlockwiseHadamardInPlace(unsigned row_base, uint32_t full_n,
                                           uint32_t hadamard_n,
                                           uint32_t log2_hadamard_n) {
  if (full_n == hadamard_n &&
      tryRunBatchedHadamard<InputT>(row_base, 1, full_n, log2_hadamard_n)) {
    return;
  }

  const uint32_t num_blocks = full_n / hadamard_n;
  for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const unsigned block_base =
        row_base + block_idx * hadamard_n * sizeof(InputT);
    if (!tryRunBatchedHadamard<InputT>(block_base, 1, hadamard_n,
                                       log2_hadamard_n)) {
      runSingleHadamardRow<InputT>(block_base, hadamard_n, log2_hadamard_n);
    }
  }
}

template <typename InputT>
AICORE void runTileBlockwiseHadamardInPlace(unsigned tile_base,
                                            uint32_t sample_count,
                                            uint32_t full_n,
                                            uint32_t hadamard_n,
                                            uint32_t log2_hadamard_n) {
  if (full_n == hadamard_n &&
      tryRunBatchedHadamard<InputT>(tile_base, sample_count, full_n,
                                    log2_hadamard_n)) {
    return;
  }

  for (uint32_t s = 0; s < sample_count; ++s) {
    const unsigned row_base = tile_base + s * full_n * sizeof(InputT);
    runRowBlockwiseHadamardInPlace<InputT>(row_base, full_n, hadamard_n,
                                           log2_hadamard_n);
  }
}

template <typename InputT, typename BulkTileT, typename ReduceTmpTileT,
          typename ReduceTileColMajorT, typename ReduceTileRowMajorT>
AICORE void reduceTileMaxAbs(BulkTileT &xTile, ReduceTmpTileT &reduceTmpTile,
                             ReduceTileColMajorT &rowMaxTile,
                             ReduceTileColMajorT &rowMinTile,
                             ReduceTileRowMajorT &rowMaxTileRm,
                             ReduceTileRowMajorT &rowMinTileRm) {
  TROWMAX(rowMaxTile, xTile, reduceTmpTile);
  TROWMIN(rowMinTile, xTile, reduceTmpTile);
  pipe_barrier(PIPE_V);

  TRESHAPE(rowMaxTileRm, rowMaxTile);
  TRESHAPE(rowMinTileRm, rowMinTile);
  pipe_barrier(PIPE_V);

  TMULS(rowMinTileRm, rowMinTileRm, (InputT)-1.0f);
  pipe_barrier(PIPE_V);
  TMAX(rowMaxTileRm, rowMaxTileRm, rowMinTileRm);
  pipe_barrier(PIPE_V);
}

template <typename InputT, typename BulkTileT, typename ReduceTileColMajorT,
          typename ReduceTileRowMajorT>
AICORE void normalizeTileFromRowMax(BulkTileT &xTile,
                                    ReduceTileColMajorT &rowMaxTile,
                                    ReduceTileRowMajorT &rowMaxTileRm,
                                    float scale_divisor) {
  TRESHAPE(rowMaxTile, rowMaxTileRm);
  pipe_barrier(PIPE_V);
  TROWEXPANDDIV(xTile, xTile, rowMaxTile);
  pipe_barrier(PIPE_V);
  TMULS(xTile, xTile, (InputT)scale_divisor);
  pipe_barrier(PIPE_V);
}

template <typename InputT, typename OutputT>
AICORE void runLargeRowDynamicQuantTile(__gm__ InputT *x, __gm__ OutputT *y,
                                        __gm__ float *row_scales,
                                        const TileWork &tile,
                                        uint32_t full_n,
                                        uint32_t hadamard_n,
                                        uint32_t log2_hadamard_n,
                                        float inv_sqrt_hadamard_n,
                                        unsigned x_base,
                                        unsigned y_base,
                                        event_t ev) {
  using BulkTile =
      Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD,
           ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
  using BulkQuantTile =
      Tile<TileType::Vec, OutputT, MAX_DYNAMIC_SAMPLES_PER_LOAD,
           ELEMENTS_PER_TILE / 2, BLayout::RowMajor, -1, -1>;
  using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2,
                         BLayout::RowMajor, -1, -1>;
  using OutShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE / 2>;
  using ScaleShapeDim5 =
      pto::Shape<1, 1, 1, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using OutGlobal = pto::GlobalTensor<OutputT, OutShapeDim5, StridDim5>;
  using ScaleGlobal = pto::GlobalTensor<float, ScaleShapeDim5, StridDim5>;
  using ScaleTile = Tile<TileType::Vec, float, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD,
                         BLayout::RowMajor, -1, -1>;
  using ReduceTmpTile =
      Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD,
           ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
  using ReduceTileColMajor =
      Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD, 1,
           BLayout::ColMajor, -1, -1>;
  using ReduceTileRowMajor =
      Tile<TileType::Vec, InputT, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD,
           BLayout::RowMajor, -1, -1>;

  const uint32_t row_index = tile.gm_offset / full_n;
  const uint32_t blocks_per_segment = ELEMENTS_PER_TILE / hadamard_n;
  const uint32_t segment_n = blocks_per_segment * hadamard_n;
  const float scale_divisor = 7.0f;

  ReduceTileRowMajor rowAccumTile(1, 1);
  ReduceTileRowMajor rowFloorTile(1, 1);
  TASSIGN(rowAccumTile, SCALE_BASE);
  TASSIGN(rowFloorTile, ODD_BASE);

  bool has_row_accum = false;
  for (uint32_t segment_offset = 0; segment_offset < full_n;
       segment_offset += segment_n) {
    const uint32_t segment_elements = min(segment_n, full_n - segment_offset);
    TileWork segment_tile = {tile.gm_offset + segment_offset, 1,
                             segment_elements};
    issueSerializedTLoad(x, segment_tile, x_base, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);

    BulkTile xSegmentTile(1, segment_elements);
    TASSIGN(xSegmentTile, x_base);
    runTileBlockwiseHadamardInPlace<InputT>(x_base, 1, segment_elements,
                                            hadamard_n, log2_hadamard_n);
    pipe_barrier(PIPE_V);

    ReduceTmpTile reduceTmpTile(1, segment_elements);
    ReduceTileColMajor rowMaxTile(1, 1);
    ReduceTileColMajor rowMinTile(1, 1);
    ReduceTileRowMajor rowMaxTileRm(1, 1);
    ReduceTileRowMajor rowMinTileRm(1, 1);
    TASSIGN(reduceTmpTile, REDUCE_TMP_BASE);
    TASSIGN(rowMaxTile, ROWMAX_BASE);
    TASSIGN(rowMinTile, ROWMIN_BASE);
    TASSIGN(rowMaxTileRm, EVEN_BASE);
    TASSIGN(rowMinTileRm, ODD_BASE);

    reduceTileMaxAbs<InputT>(xSegmentTile, reduceTmpTile, rowMaxTile,
                             rowMinTile, rowMaxTileRm, rowMinTileRm);
    if (!has_row_accum) {
      TMULS(rowAccumTile, rowMaxTileRm, (InputT)1.0f);
      pipe_barrier(PIPE_V);
      has_row_accum = true;
    } else {
      TMAX(rowAccumTile, rowAccumTile, rowMaxTileRm);
      pipe_barrier(PIPE_V);
    }
    set_flag(PIPE_V, PIPE_MTE2, ev);
  }

  TMULS(rowFloorTile, rowAccumTile, (InputT)0.0f);
  pipe_barrier(PIPE_V);
  TADDS(rowFloorTile, rowFloorTile, (InputT)1e-6f);
  pipe_barrier(PIPE_V);
  TMAX(rowAccumTile, rowAccumTile, rowFloorTile);
  pipe_barrier(PIPE_V);

  for (uint32_t segment_offset = 0; segment_offset < full_n;
       segment_offset += segment_n) {
    const uint32_t segment_elements = min(segment_n, full_n - segment_offset);
    TileWork segment_tile = {tile.gm_offset + segment_offset, 1,
                             segment_elements};
    issueSerializedTLoad(x, segment_tile, x_base, ev);
    wait_flag(PIPE_MTE2, PIPE_V, ev);

    BulkTile xSegmentTile(1, segment_elements);
    BulkQuantTile ySegmentTile2D(1, segment_elements >> 1);
    QuantTile ySegmentTile(1, segment_elements >> 1);
    ReduceTileColMajor rowMaxTile(1, 1);
    ReduceTileRowMajor rowMaxTileRm(1, 1);
    TASSIGN(xSegmentTile, x_base);
    TASSIGN(ySegmentTile2D, y_base);
    TASSIGN(ySegmentTile, y_base);
    TASSIGN(rowMaxTile, ROWMAX_BASE);
    TASSIGN(rowMaxTileRm, SCALE_BASE);
    runTileBlockwiseHadamardInPlace<InputT>(x_base, 1, segment_elements,
                                            hadamard_n, log2_hadamard_n);
    pipe_barrier(PIPE_V);
    normalizeTileFromRowMax<InputT>(xSegmentTile, rowMaxTile, rowMaxTileRm,
                                    scale_divisor);

    wait_flag(PIPE_MTE3, PIPE_V, ev);
    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(ySegmentTile2D, xSegmentTile,
                                                 RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE3, ev);
    wait_flag(PIPE_V, PIPE_MTE3, ev);
    OutGlobal yGlobal(y + ((tile.gm_offset + segment_offset) >> 1));
    TASSIGN(yGlobal, (y + ((tile.gm_offset + segment_offset) >> 1)));
    TSTORE(yGlobal, ySegmentTile);
    set_flag(PIPE_MTE3, PIPE_V, ev);
    set_flag(PIPE_V, PIPE_MTE2, ev);
  }

  ScaleTile scaleTile(1, 1);
  ScaleTile scaleFloorTile(1, 1);
  // The serialized large-row path only uses X_PING/Y_PING for load+pack.
  // Stage the final scale store in the unused pong buffers so the next row can
  // immediately reuse EVEN/ODD/SCALE/REDUCE scratch after we return.
  TASSIGN(scaleTile, X_PONG);
  TASSIGN(scaleFloorTile, Y_PONG);

  TCVT(scaleTile, rowAccumTile, RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
  TMULS(scaleTile, scaleTile, inv_sqrt_hadamard_n / scale_divisor);
  pipe_barrier(PIPE_V);

  TMULS(scaleFloorTile, scaleTile, 0.0f);
  pipe_barrier(PIPE_V);
  TADDS(scaleFloorTile, scaleFloorTile, 1e-6f);
  pipe_barrier(PIPE_V);
  TMAX(scaleTile, scaleTile, scaleFloorTile);
  pipe_barrier(PIPE_V);

  ScaleGlobal scaleGlobal(row_scales + row_index);
  TASSIGN(scaleGlobal, (row_scales + row_index));
  wait_flag(PIPE_MTE3, PIPE_V, ev);
  set_flag(PIPE_V, PIPE_MTE3, ev);
  wait_flag(PIPE_V, PIPE_MTE3, ev);
  TSTORE(scaleGlobal, scaleTile);
  set_flag(PIPE_MTE3, PIPE_V, ev);
  set_flag(PIPE_V, PIPE_MTE2, ev);
}

template <typename InputT, typename OutputT>
AICORE void runTFastHadamardQuant(__gm__ InputT *x, __gm__ OutputT *y,
                                  __gm__ InputT *group_scales,
                                  __gm__ InputT *group_offsets,
                                  uint32_t scale_group_stride,
                                  uint32_t offset_group_stride, uint32_t batch,
                                  uint32_t n, uint32_t log2_n,
                                  uint32_t num_cores, uint32_t vid, float scale,
                                  uint32_t group_size, float q_offset) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (n == 0 || (n & 1U) != 0 || n > ELEMENTS_PER_TILE) {
    return;
  }

  const uint32_t samples_per_core = DIV_ROUNDUP(batch, num_cores);
  const uint32_t sample_offset = samples_per_core * vid;
  if (sample_offset >= batch) {
    return;
  }

  uint32_t samples_to_process = samples_per_core;
  if (sample_offset + samples_to_process > batch) {
    samples_to_process = batch - sample_offset;
  }
  if (samples_to_process == 0) {
    return;
  }

  using InShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using OutShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE / 2>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using OutGlobal = pto::GlobalTensor<OutputT, OutShapeDim5, StridDim5>;

  using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                        BLayout::RowMajor, -1, -1>;
  using HalfTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE / 2,
                        BLayout::RowMajor, -1, -1>;
  using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2,
                         BLayout::RowMajor, -1, -1>;

  const uint32_t samples_per_load =
      (n < ELEMENTS_PER_TILE) ? (ELEMENTS_PER_TILE / n) : 1;
  const uint32_t n_half = n >> 1;
  const uint32_t packed_n = n >> 1;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  HalfTile evenTile(1, n_half);
  HalfTile oddTile(1, n_half);
  TASSIGN(evenTile, EVEN_BASE);
  TASSIGN(oddTile, ODD_BASE);

  uint32_t sample_done = 0;
  TileWork current_tile;
  const uint32_t gm_offset_base = sample_offset * n;
  if (!nextTile(sample_done, gm_offset_base, samples_to_process,
                samples_per_load, n, current_tile)) {
    return;
  }

  bool ping = true;
  issueTLoad(x, current_tile, X_PING, EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned current_x_base = ping ? X_PING : X_PONG;
    const unsigned current_y_base = ping ? Y_PING : Y_PONG;

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    TileWork next_tile;
    const bool has_next =
        nextTile(sample_done, gm_offset_base, samples_to_process,
                 samples_per_load, n, next_tile);
    if (has_next) {
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x_base = ping ? X_PONG : X_PING;
      issueTLoad(x, next_tile, next_x_base, next_ev);
    }

    FullTile xBulkTile(1, current_tile.elements);
    QuantTile yBulkTile(1, current_tile.elements >> 1);
    TASSIGN(xBulkTile, current_x_base);
    TASSIGN(yBulkTile, current_y_base);

    OutGlobal yGlobal(y + (current_tile.gm_offset >> 1));
    TASSIGN(yGlobal, (y + (current_tile.gm_offset >> 1)));

    if (!tryRunBatchedHadamard<InputT>(current_x_base,
                                       current_tile.sample_count, n, log2_n)) {
      for (uint32_t s = 0; s < current_tile.sample_count; ++s) {
        const unsigned row_base = current_x_base + s * n * sizeof(InputT);

        FullTile xRowTile(1, n);
        HalfTile xFirstHalf(1, n_half);
        HalfTile xSecondHalf(1, n_half);
        TASSIGN(xRowTile, row_base);
        TASSIGN(xFirstHalf, row_base);
        TASSIGN(xSecondHalf, row_base + n_half * sizeof(InputT));

        for (uint32_t iter_m = 0; iter_m < log2_n; ++iter_m) {
          TGATHER<HalfTile, FullTile, MaskPattern::P0101>(evenTile, xRowTile);
          TGATHER<HalfTile, FullTile, MaskPattern::P1010>(oddTile, xRowTile);

          pipe_barrier(PIPE_V);

          TADD(xFirstHalf, evenTile, oddTile);
          TSUB(xSecondHalf, evenTile, oddTile);

          pipe_barrier(PIPE_V);
        }
      }
    }
    const bool has_group_scales = group_scales != nullptr;
    const bool has_group_offsets = group_offsets != nullptr;
    if (!has_group_scales && !has_group_offsets) {
      // Uniform scale/offset is equivalent for any group_size, so keep the
      // whole-tile path and overlap the scale/add with the previous store on
      // the opposite ping-pong buffer before we touch yBulkTile again.
      TMULS(xBulkTile, xBulkTile, (InputT)scale);
      pipe_barrier(PIPE_V);
      if (q_offset != 0.0f) {
        TADDS(xBulkTile, xBulkTile, (InputT)q_offset);
        pipe_barrier(PIPE_V);
      }
      wait_flag(PIPE_MTE3, PIPE_V, current_ev);
      fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(yBulkTile, xBulkTile,
                                                   RoundMode::CAST_NONE);
      pipe_barrier(PIPE_V);
    } else {
      wait_flag(PIPE_MTE3, PIPE_V, current_ev);
      const uint32_t groups_per_row = n / group_size;
      const uint32_t row_index_base = current_tile.gm_offset / n;
      const uint32_t packed_group_size = group_size >> 1;
      for (uint32_t s = 0; s < current_tile.sample_count; ++s) {
        const uint32_t row_index = row_index_base + s;
        const unsigned row_x_base = current_x_base + s * n * sizeof(InputT);
        const unsigned row_y_base =
            current_y_base + s * packed_n * sizeof(OutputT);

        for (uint32_t g = 0; g < groups_per_row; ++g) {
          const unsigned group_x_base =
              row_x_base + g * group_size * sizeof(InputT);
          const unsigned group_y_base =
              row_y_base + g * packed_group_size * sizeof(OutputT);

          FullTile xGroupTile(1, group_size);
          QuantTile yGroupTile(1, packed_group_size);
          TASSIGN(xGroupTile, group_x_base);
          TASSIGN(yGroupTile, group_y_base);

          InputT group_scale = (InputT)scale;
          if (has_group_scales) {
            const uint32_t scale_index =
                (scale_group_stride == 0) ? g
                                          : row_index * scale_group_stride + g;
            group_scale = group_scales[scale_index];
          }

          TMULS(xGroupTile, xGroupTile, group_scale);
          pipe_barrier(PIPE_V);
          if (has_group_offsets || q_offset != 0.0f) {
            InputT group_offset = (InputT)q_offset;
            if (has_group_offsets) {
              const uint32_t offset_index =
                  (offset_group_stride == 0)
                      ? g
                      : row_index * offset_group_stride + g;
              group_offset = group_offsets[offset_index];
            }
            TADDS(xGroupTile, xGroupTile, group_offset);
            pipe_barrier(PIPE_V);
          }
          fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(yGroupTile, xGroupTile,
                                                       RoundMode::CAST_NONE);
          pipe_barrier(PIPE_V);
        }
      }
    }

    set_flag(PIPE_V, PIPE_MTE3, current_ev);
    wait_flag(PIPE_V, PIPE_MTE3, current_ev);
    TSTORE(yGlobal, yBulkTile);
    set_flag(PIPE_MTE3, PIPE_V, current_ev);
    set_flag(PIPE_V, PIPE_MTE2, current_ev);

    if (!has_next) {
      break;
    }

    current_tile = next_tile;
    ping = !ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

template <typename InputT, typename OutputT>
AICORE void runTFastHadamardDynamicQuant(__gm__ InputT *x, __gm__ OutputT *y,
                                         __gm__ float *row_scales,
                                         uint32_t batch, uint32_t full_n,
                                         uint32_t hadamard_n,
                                         uint32_t log2_hadamard_n,
                                         float inv_sqrt_hadamard_n,
                                         uint32_t num_cores, uint32_t vid) {
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (full_n == 0 || (full_n & 1U) != 0 ||
      hadamard_n > ELEMENTS_PER_TILE ||
      hadamard_n == 0 || full_n % hadamard_n != 0) {
    return;
  }

  const uint32_t samples_per_core = DIV_ROUNDUP(batch, num_cores);
  const uint32_t sample_offset = samples_per_core * vid;
  if (sample_offset >= batch) {
    return;
  }

  uint32_t samples_to_process = samples_per_core;
  if (sample_offset + samples_to_process > batch) {
    samples_to_process = batch - sample_offset;
  }
  if (samples_to_process == 0) {
    return;
  }

  using OutShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE / 2>;
  using ScaleShapeDim5 =
      pto::Shape<1, 1, 1, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using OutGlobal = pto::GlobalTensor<OutputT, OutShapeDim5, StridDim5>;
  using ScaleGlobal = pto::GlobalTensor<float, ScaleShapeDim5, StridDim5>;
  using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE / 2,
                         BLayout::RowMajor, -1, -1>;
  using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE,
                        BLayout::RowMajor, -1, -1>;
  using ScaleTile = Tile<TileType::Vec, float, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD,
                         BLayout::RowMajor, -1, -1>;

  const uint32_t samples_per_load =
      (full_n < ELEMENTS_PER_TILE) ? (ELEMENTS_PER_TILE / full_n) : 1;
  const uint32_t packed_n = full_n >> 1;
  const float scale_divisor = 7.0f;

  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  uint32_t sample_done = 0;
  TileWork current_tile;
  const uint32_t gm_offset_base = sample_offset * full_n;
  if (!nextTile(sample_done, gm_offset_base, samples_to_process,
                samples_per_load, full_n, current_tile)) {
    return;
  }

  if (full_n > ELEMENTS_PER_TILE) {
    while (true) {
      TileWork next_tile;
      const bool has_next =
          nextTile(sample_done, gm_offset_base, samples_to_process,
                   samples_per_load, full_n, next_tile);
      runLargeRowDynamicQuantTile<InputT, OutputT>(
          x, y, row_scales, current_tile, full_n, hadamard_n, log2_hadamard_n,
          inv_sqrt_hadamard_n, X_PING, Y_PING, EVENT_ID0);
      if (!has_next) {
        break;
      }
      current_tile = next_tile;
    }

    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    return;
  }

  bool ping = true;
  issueTLoad(x, current_tile, X_PING, EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned current_x_base = ping ? X_PING : X_PONG;
    const unsigned current_y_base = ping ? Y_PING : Y_PONG;

    TileWork next_tile;
    const bool has_next =
        nextTile(sample_done, gm_offset_base, samples_to_process,
                 samples_per_load, full_n, next_tile);
    wait_flag(PIPE_MTE2, PIPE_V, current_ev);
    if (has_next) {
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x_base = ping ? X_PONG : X_PING;
      issueTLoad(x, next_tile, next_x_base, next_ev);
    }

    QuantTile yBulkTile(1, current_tile.elements >> 1);
    TASSIGN(yBulkTile, current_y_base);
    OutGlobal yGlobal(y + (current_tile.gm_offset >> 1));
    TASSIGN(yGlobal, (y + (current_tile.gm_offset >> 1)));

    const uint32_t row_index_base = current_tile.gm_offset / full_n;
    using BulkTile =
        Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD,
             ELEMENTS_PER_TILE,
             BLayout::RowMajor, -1, -1>;
    using BulkQuantTile =
        Tile<TileType::Vec, OutputT, MAX_DYNAMIC_SAMPLES_PER_LOAD,
             ELEMENTS_PER_TILE / 2,
             BLayout::RowMajor, -1, -1>;
    using ReduceTmpTile =
        Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD,
             ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
    // The server PTO toolkit expects compact DN/ColMajor row vectors for
    // TROWEXPAND*, while generic elementwise ops like TMAX still want
    // row-major tiles. Keep both views and reshape between them.
    using ReduceTileColMajor =
        Tile<TileType::Vec, InputT, MAX_DYNAMIC_SAMPLES_PER_LOAD, 1,
             BLayout::ColMajor, -1, -1>;
    using ReduceTileRowMajor =
        Tile<TileType::Vec, InputT, 1, MAX_DYNAMIC_SAMPLES_PER_LOAD,
             BLayout::RowMajor, -1, -1>;

    BulkTile xBulkTile(current_tile.sample_count, full_n);
    BulkQuantTile yBulkTile2D(current_tile.sample_count, packed_n);
    TASSIGN(xBulkTile, current_x_base);
    TASSIGN(yBulkTile2D, current_y_base);

    ScaleTile scaleTile(1, current_tile.sample_count);
    ScaleTile scaleFloorTile(1, current_tile.sample_count);
    TASSIGN(scaleTile, SCALE_BASE);
    TASSIGN(scaleFloorTile, REDUCE_TMP_BASE);
    ScaleGlobal scaleGlobal(row_scales + row_index_base);
    TASSIGN(scaleGlobal, (row_scales + row_index_base));
    wait_flag(PIPE_MTE3, PIPE_V, current_ev);
    runTileBlockwiseHadamardInPlace<InputT>(current_x_base,
                                            current_tile.sample_count, full_n,
                                            hadamard_n, log2_hadamard_n);
    pipe_barrier(PIPE_V);
    ReduceTmpTile reduceTmpTile(current_tile.sample_count, full_n);
    ReduceTileColMajor rowMaxTile(current_tile.sample_count, 1);
    ReduceTileColMajor rowMinTile(current_tile.sample_count, 1);
    ReduceTileRowMajor rowMaxTileRm(1, current_tile.sample_count);
    ReduceTileRowMajor rowMinTileRm(1, current_tile.sample_count);
    TASSIGN(reduceTmpTile, REDUCE_TMP_BASE);
    TASSIGN(rowMaxTile, ROWMAX_BASE);
    TASSIGN(rowMinTile, ROWMIN_BASE);
    // EVEN/ODD scratch are free after the Hadamard pass completes.
    TASSIGN(rowMaxTileRm, EVEN_BASE);
    TASSIGN(rowMinTileRm, ODD_BASE);

    TROWMAX(rowMaxTile, xBulkTile, reduceTmpTile);
    TROWMIN(rowMinTile, xBulkTile, reduceTmpTile);
    pipe_barrier(PIPE_V);

    TRESHAPE(rowMaxTileRm, rowMaxTile);
    TRESHAPE(rowMinTileRm, rowMinTile);
    pipe_barrier(PIPE_V);

    TMULS(rowMinTileRm, rowMinTileRm, (InputT)-1.0f);
    pipe_barrier(PIPE_V);
    TMAX(rowMaxTileRm, rowMaxTileRm, rowMinTileRm);
    pipe_barrier(PIPE_V);

    TCVT(scaleTile, rowMaxTileRm, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);
    TMULS(scaleTile, scaleTile,
          inv_sqrt_hadamard_n / scale_divisor);
    pipe_barrier(PIPE_V);

    TMULS(rowMinTileRm, rowMaxTileRm, (InputT)0.0f);
    pipe_barrier(PIPE_V);
    TADDS(rowMinTileRm, rowMinTileRm, (InputT)1e-6f);
    pipe_barrier(PIPE_V);
    TMAX(rowMaxTileRm, rowMaxTileRm, rowMinTileRm);
    pipe_barrier(PIPE_V);

    TRESHAPE(rowMaxTile, rowMaxTileRm);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(xBulkTile, xBulkTile, rowMaxTile);
    pipe_barrier(PIPE_V);
    TMULS(xBulkTile, xBulkTile, (InputT)scale_divisor);
    pipe_barrier(PIPE_V);

    TMULS(scaleFloorTile, scaleTile, 0.0f);
    pipe_barrier(PIPE_V);
    TADDS(scaleFloorTile, scaleFloorTile, 1e-6f);
    pipe_barrier(PIPE_V);
    TMAX(scaleTile, scaleTile, scaleFloorTile);
    pipe_barrier(PIPE_V);

    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(yBulkTile2D, xBulkTile,
                                                 RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE3, current_ev);
    wait_flag(PIPE_V, PIPE_MTE3, current_ev);
    TSTORE(yGlobal, yBulkTile);
    TSTORE(scaleGlobal, scaleTile);
    set_flag(PIPE_MTE3, PIPE_V, current_ev);
    set_flag(PIPE_V, PIPE_MTE2, current_ev);

    if (!has_next) {
      break;
    }

    current_tile = next_tile;
    ping = !ping;
  }

  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

}  // namespace

__global__ AICORE void fast_hadamard_quant_fp16_to_int4(
    __gm__ void *x, __gm__ void *y, __gm__ void *group_scales,
    __gm__ void *group_offsets, uint32_t scale_group_stride,
    uint32_t offset_group_stride, uint32_t batch, uint32_t n, uint32_t log2_n,
    float scale, uint32_t group_size, float q_offset) {
#if defined(__DAV_VEC__)
  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  runTFastHadamardQuant<half, int8_t>(
      (__gm__ half *)x, (__gm__ int8_t *)y, (__gm__ half *)group_scales,
      (__gm__ half *)group_offsets, scale_group_stride, offset_group_stride,
      batch, n, log2_n, num_cores, vid, scale, group_size, q_offset);
#else
  (void)x;
  (void)y;
  (void)group_scales;
  (void)group_offsets;
  (void)scale_group_stride;
  (void)offset_group_stride;
  (void)batch;
  (void)n;
  (void)log2_n;
  (void)scale;
  (void)group_size;
  (void)q_offset;
#endif
}

__global__ AICORE void fast_hadamard_dynamic_quant_fp16_to_int4(
    __gm__ void *x, __gm__ void *y, __gm__ void *row_scales, uint32_t batch,
    uint32_t full_n, uint32_t hadamard_n, uint32_t log2_hadamard_n,
    float inv_sqrt_hadamard_n) {
#if defined(__DAV_VEC__)
  uint32_t num_cores = get_block_num() * get_subblockdim();
  uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  if (full_n > ELEMENTS_PER_TILE) {
    // Correctness-first path for oversized rows: keep execution on a single
    // logical worker until the helper is proven safe under wider parallelism.
    if (get_block_idx() != 0 || get_subblockid() != 0) {
      return;
    }
    num_cores = 1;
    vid = 0;
  }
  runTFastHadamardDynamicQuant<half, int8_t>(
      (__gm__ half *)x, (__gm__ int8_t *)y, (__gm__ float *)row_scales, batch,
      full_n, hadamard_n, log2_hadamard_n, inv_sqrt_hadamard_n, num_cores,
      vid);
#else
  (void)x;
  (void)y;
  (void)row_scales;
  (void)batch;
  (void)full_n;
  (void)hadamard_n;
  (void)log2_hadamard_n;
  (void)inv_sqrt_hadamard_n;
#endif
}

extern "C" void call_fused_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                                  uint8_t *y, uint8_t *group_scales,
                                  uint8_t *group_offsets,
                                  uint32_t scale_group_stride,
                                  uint32_t offset_group_stride, uint32_t batch,
                                  uint32_t n, uint32_t log2_n, float scale,
                                  uint32_t group_size, float q_offset) {
  blockDim = blockDim * 2;
  fast_hadamard_quant_fp16_to_int4<<<blockDim, nullptr, stream>>>(
      x, y, group_scales, group_offsets, scale_group_stride,
      offset_group_stride, batch, n, log2_n, scale, group_size, q_offset);
}

extern "C" void call_dynamic_quant_kernel(uint32_t blockDim, void *stream,
                                          uint8_t *x, uint8_t *y,
                                          float *row_scales, uint32_t batch,
                                          uint32_t full_n,
                                          uint32_t hadamard_n,
                                          uint32_t log2_hadamard_n,
                                          float inv_sqrt_hadamard_n) {
  if (full_n > ELEMENTS_PER_TILE) {
    // The oversized-row helper is serialized today. Launch it with a single
    // worker instead of the normal sub-block geometry so the runtime topology
    // matches the helper's correctness-first scheduling contract.
    blockDim = 1;
  } else {
    blockDim = blockDim * 2;
  }
  fast_hadamard_dynamic_quant_fp16_to_int4<<<blockDim, nullptr, stream>>>(
      x, y, row_scales, batch, full_n, hadamard_n, log2_hadamard_n,
      inv_sqrt_hadamard_n);
}
