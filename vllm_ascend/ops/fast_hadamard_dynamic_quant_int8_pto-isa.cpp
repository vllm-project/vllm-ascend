#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

// DeepSeek-V4 DSA indexer rotates 128-wide q/k rows.  Specializing the width
// keeps the fused FHT+dynamic-quant kernel branch-free in the hot path.
constexpr uint32_t N = 128;
constexpr uint32_t LOG2_N = 7;
constexpr uint32_t N_HALF = N / 2;

// 32KB is the pto-kernels tile size for this FHT schedule.  With N=128 and
// fp16 input, one tile holds 128 rows, so each worker amortizes launch and DMA
// overhead while still leaving UB room for ping-pong output and reductions.
constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t UB_HALF_BYTES = X_BUFFER_BYTES / 2;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t SAMPLES_PER_LOAD = ELEMENTS_PER_TILE / N;
constexpr uint32_t Y_BUFFER_BYTES = ELEMENTS_PER_TILE * sizeof(int8_t);
// 184KB matches the usable UB budget used by the pto-kernels FHT examples; the
// rest of the physical UB is reserved by the runtime/compiler.
constexpr uint32_t UB_USABLE_BYTES = 184 * 1024;
constexpr uint32_t DYNAMIC_SCALE_BYTES = SAMPLES_PER_LOAD * sizeof(float);
constexpr uint32_t DYNAMIC_REDUCE_BYTES = SAMPLES_PER_LOAD * sizeof(half);
// The Hadamard transform is normalized by folding 1/sqrt(128) into the stored
// dynamic quant scale instead of multiplying the transformed fp16 tile.
constexpr float INV_SQRT_N = 0.08838834764831845f;
// Symmetric int8 dynamic quantization uses [-127, 127] with zero point 0.
constexpr float QUANT_MAX_INT8 = 127.0f;
// Match torch_npu dynamic quant behavior for all-zero or tiny rows.
constexpr float SCALE_FLOOR = 1e-6f;
// MTE3 stores float scales in 32-byte blocks.  Eight fp32 rows keep each core's
// scale range 32-byte aligned and avoid cross-core overwrite on row_scales.
constexpr uint32_t SCALE_ALIGNMENT_ROWS = 8;

// PTO Tile objects are typed views over raw UB byte offsets.  The 0x100 guard
// spacing follows the pto-kernels examples and keeps independently addressed
// vector/MTE regions comfortably separated.
constexpr unsigned UB_GUARD_BYTES = 0x100;

// Ping-pong buffers let MTE2 load the next input tile and MTE3 store the current
// output tile without waiting for the vector pipe to finish every row block.
constexpr unsigned X_PING = 0x00000;
constexpr unsigned Y_PING = X_PING + X_BUFFER_BYTES + UB_GUARD_BYTES;
constexpr unsigned SCALE_PING = Y_PING + Y_BUFFER_BYTES + UB_GUARD_BYTES;
constexpr unsigned X_PONG = SCALE_PING + DYNAMIC_SCALE_BYTES + UB_GUARD_BYTES;
constexpr unsigned Y_PONG = X_PONG + X_BUFFER_BYTES + UB_GUARD_BYTES;
constexpr unsigned SCALE_PONG = Y_PONG + Y_BUFFER_BYTES + UB_GUARD_BYTES;

// Scratch/reduction buffers are shared by ping and pong.  Only one vector phase
// is active per worker, so double-buffering these would consume UB without
// increasing MTE overlap.
constexpr unsigned EVEN_BASE = SCALE_PONG + DYNAMIC_SCALE_BYTES + UB_GUARD_BYTES;
constexpr unsigned ODD_BASE = EVEN_BASE + UB_HALF_BYTES + UB_GUARD_BYTES;
constexpr unsigned REDUCE_TMP_BASE = ODD_BASE + UB_HALF_BYTES + UB_GUARD_BYTES;
constexpr unsigned SCALE_FLOOR_BASE = REDUCE_TMP_BASE + X_BUFFER_BYTES + UB_GUARD_BYTES;
constexpr unsigned ROWMAX_BASE = SCALE_FLOOR_BASE + DYNAMIC_SCALE_BYTES + UB_GUARD_BYTES;
constexpr unsigned ROWMIN_BASE = ROWMAX_BASE + DYNAMIC_REDUCE_BYTES + UB_GUARD_BYTES;
static_assert(ROWMIN_BASE + DYNAMIC_REDUCE_BYTES <= UB_USABLE_BYTES,
              "Fused Hadamard+int8 dynamic quant UB layout exceeds usable UB.");

namespace {

struct TileWork {
  uint32_t gm_offset, sample_count, elements;
};

// Produce the next contiguous GM row range for this vector sub-core.  gm_offset
// is measured in elements, not bytes, because GlobalTensor is already typed.
AICORE bool nextTile(uint32_t& sample_done, uint32_t gm_offset_base, uint32_t samples_to_process, TileWork& tile) {
  if (sample_done >= samples_to_process) {
    return false;
  }

  tile.sample_count = min(SAMPLES_PER_LOAD, samples_to_process - sample_done);
  tile.elements = tile.sample_count * N;
  tile.gm_offset = gm_offset_base + sample_done * N;
  sample_done += tile.sample_count;
  return true;
}

template <typename InputT>
AICORE void issueTLoad(__gm__ InputT* x, const TileWork& tile, unsigned x_base, event_t ev) {
  using InShape = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using Stride = pto::Stride<1, 1, 1, 1, 1>;
  using InGlobal = pto::GlobalTensor<InputT, InShape, Stride>;
  using FullTile = Tile<TileType::Vec, InputT, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;

  FullTile xBulkTile(1, tile.elements);
  TASSIGN(xBulkTile, x_base);
  InGlobal xGlobal(x + tile.gm_offset);
  TASSIGN(xGlobal, (x + tile.gm_offset));
  wait_flag(PIPE_V, PIPE_MTE2, ev);
  TLOAD(xBulkTile, xGlobal);
  set_flag(PIPE_MTE2, PIPE_V, ev);
}

template <typename InputT>
AICORE void runHadamard128InPlace(unsigned x_base, uint32_t sample_count) {
  using FullTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, DYNAMIC, N>;
  using ScratchTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N_HALF, BLayout::RowMajor, DYNAMIC, N_HALF>;
  using XHalfTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, DYNAMIC, N_HALF>;

  FullTile xBulkTile(sample_count);
  ScratchTile evenTile(sample_count);
  ScratchTile oddTile(sample_count);
  XHalfTile xFirstHalf(sample_count);
  XHalfTile xSecondHalf(sample_count);
  TASSIGN(xBulkTile, x_base);
  TASSIGN(evenTile, EVEN_BASE);
  TASSIGN(oddTile, ODD_BASE);
  TASSIGN(xFirstHalf, x_base);
  TASSIGN(xSecondHalf, x_base + N_HALF * sizeof(InputT));

// One radix-2 Hadamard stage:
//   1. split alternating values into even/odd scratch tiles,
//   2. write even + odd to the first half,
//   3. write even - odd to the second half.
// Repeating this LOG2_N times gives a 128-point Hadamard per row.  The
// pipe_barrier calls order vector instructions that reuse the same UB buffers.
#define FAST_HADAMARD_128_STAGE()                                            \
  do {                                                                       \
    TGATHER<ScratchTile, FullTile, MaskPattern::P0101>(evenTile, xBulkTile); \
    TGATHER<ScratchTile, FullTile, MaskPattern::P1010>(oddTile, xBulkTile);  \
    pipe_barrier(PIPE_V);                                                    \
    TADD(xFirstHalf, evenTile, oddTile);                                     \
    TSUB(xSecondHalf, evenTile, oddTile);                                    \
    pipe_barrier(PIPE_V);                                                    \
  } while (0)

  FAST_HADAMARD_128_STAGE();
  FAST_HADAMARD_128_STAGE();
  FAST_HADAMARD_128_STAGE();
  FAST_HADAMARD_128_STAGE();
  FAST_HADAMARD_128_STAGE();
  FAST_HADAMARD_128_STAGE();
  FAST_HADAMARD_128_STAGE();

#undef FAST_HADAMARD_128_STAGE
}

template <typename InputT>
AICORE void runHadamard128InPlaceFull(unsigned x_base) {
  using FullTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, SAMPLES_PER_LOAD, N>;
  using ScratchTile =
      Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N_HALF, BLayout::RowMajor, SAMPLES_PER_LOAD, N_HALF>;
  using XHalfTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, SAMPLES_PER_LOAD, N_HALF>;

  FullTile xBulkTile;
  ScratchTile evenTile;
  ScratchTile oddTile;
  XHalfTile xFirstHalf;
  XHalfTile xSecondHalf;
  TASSIGN(xBulkTile, x_base);
  TASSIGN(evenTile, EVEN_BASE);
  TASSIGN(oddTile, ODD_BASE);
  TASSIGN(xFirstHalf, x_base);
  TASSIGN(xSecondHalf, x_base + N_HALF * sizeof(InputT));

#define FAST_HADAMARD_128_FULL_STAGE()                                       \
  do {                                                                       \
    TGATHER<ScratchTile, FullTile, MaskPattern::P0101>(evenTile, xBulkTile); \
    TGATHER<ScratchTile, FullTile, MaskPattern::P1010>(oddTile, xBulkTile);  \
    pipe_barrier(PIPE_V);                                                    \
    TADD(xFirstHalf, evenTile, oddTile);                                     \
    TSUB(xSecondHalf, evenTile, oddTile);                                    \
    pipe_barrier(PIPE_V);                                                    \
  } while (0)

  FAST_HADAMARD_128_FULL_STAGE();
  FAST_HADAMARD_128_FULL_STAGE();
  FAST_HADAMARD_128_FULL_STAGE();
  FAST_HADAMARD_128_FULL_STAGE();
  FAST_HADAMARD_128_FULL_STAGE();
  FAST_HADAMARD_128_FULL_STAGE();
  FAST_HADAMARD_128_FULL_STAGE();

#undef FAST_HADAMARD_128_FULL_STAGE
}

template <typename InputT, typename OutputT>
AICORE void quantizeHadamard128Tile(unsigned x_base, unsigned y_base, uint32_t sample_count) {
  using XHalfTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, -1, N_HALF>;
  using QuantOutHalfTile = Tile<TileType::Vec, OutputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, -1, N_HALF>;
  using ReduceTileColMajor = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, 1, BLayout::ColMajor, -1, -1>;
  using ReduceTileRowMajor = Tile<TileType::Vec, InputT, 1, SAMPLES_PER_LOAD, BLayout::RowMajor, -1, -1>;

  XHalfTile xFirstHalf(sample_count);
  XHalfTile xSecondHalf(sample_count);
  QuantOutHalfTile yFirstHalf(sample_count);
  QuantOutHalfTile ySecondHalf(sample_count);
  ReduceTileColMajor rowMaxTile(1, sample_count);
  ReduceTileRowMajor rowMaxTileRm(1, sample_count);
  ReduceTileRowMajor rowFloorTile(1, sample_count);
  TASSIGN(xFirstHalf, x_base);
  TASSIGN(xSecondHalf, x_base + N_HALF * sizeof(InputT));
  TASSIGN(yFirstHalf, y_base);
  TASSIGN(ySecondHalf, y_base + N_HALF * sizeof(OutputT));
  TASSIGN(rowMaxTile, ROWMAX_BASE);
  TASSIGN(rowMaxTileRm, ROWMAX_BASE);
  TASSIGN(rowFloorTile, ODD_BASE);

  TMULS(rowFloorTile, rowMaxTileRm, (InputT)0.0f);
  pipe_barrier(PIPE_V);
  TADDS(rowFloorTile, rowFloorTile, (InputT)SCALE_FLOOR);
  pipe_barrier(PIPE_V);
  TMAX(rowMaxTileRm, rowMaxTileRm, rowFloorTile);
  pipe_barrier(PIPE_V);

  TRESHAPE(rowMaxTile, rowMaxTileRm);
  pipe_barrier(PIPE_V);
  TROWEXPANDDIV(xFirstHalf, xFirstHalf, rowMaxTile);
  TROWEXPANDDIV(xSecondHalf, xSecondHalf, rowMaxTile);
  pipe_barrier(PIPE_V);
  TMULS(xFirstHalf, xFirstHalf, (InputT)QUANT_MAX_INT8);
  TMULS(xSecondHalf, xSecondHalf, (InputT)QUANT_MAX_INT8);
  pipe_barrier(PIPE_V);
  TCVT(yFirstHalf, xFirstHalf, RoundMode::CAST_RINT);
  TCVT(ySecondHalf, xSecondHalf, RoundMode::CAST_RINT);
  pipe_barrier(PIPE_V);
}

template <typename InputT, typename OutputT>
AICORE void runTFastHadamardDynamicQuantInt8(__gm__ InputT* x, __gm__ OutputT* y, __gm__ float* row_scales,
                                             uint32_t batch, uint32_t num_cores, uint32_t vid) {
  // Configure the vector mask once for full-lane vector operations.  __DAV_VEC__
  // in the global entry point ensures this code is compiled for the vector side
  // of an Ascend AI Core.
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Each vid handles a contiguous band of rows; no inter-core synchronization is needed because
  // every row's Hadamard and dynamic quantization are independent.
  const uint32_t samples_per_core =
      DIV_ROUNDUP(DIV_ROUNDUP(batch, num_cores), SCALE_ALIGNMENT_ROWS) * SCALE_ALIGNMENT_ROWS;
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

  using Shape = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using ScaleShape = pto::Shape<1, 1, 1, 1, SAMPLES_PER_LOAD>;
  using Stride = pto::Stride<1, 1, 1, 1, 1>;
  using OutGlobal = pto::GlobalTensor<OutputT, Shape, Stride>;
  using ScaleGlobal = pto::GlobalTensor<float, ScaleShape, Stride>;
  using BulkTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, DYNAMIC, N>;
  using QuantTile = Tile<TileType::Vec, OutputT, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
  using ReduceTmpTile = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, N, BLayout::RowMajor, DYNAMIC, N>;
  using ReduceTileColMajor = Tile<TileType::Vec, InputT, SAMPLES_PER_LOAD, 1, BLayout::ColMajor, -1, -1>;
  using ReduceTileRowMajor = Tile<TileType::Vec, InputT, 1, SAMPLES_PER_LOAD, BLayout::RowMajor, -1, -1>;
  using ScaleTile = Tile<TileType::Vec, float, 1, SAMPLES_PER_LOAD, BLayout::RowMajor, -1, -1>;

  // Prime both ping-pong event slots.  EVENT_ID0 protects the ping buffers and
  // EVENT_ID1 protects the pong buffers across MTE2, vector, and MTE3 pipes.
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);

  const uint32_t gm_offset_base = sample_offset * N;
  uint32_t sample_done = 0;
  TileWork current_tile;
  if (!nextTile(sample_done, gm_offset_base, samples_to_process, current_tile)) {
    return;
  }

  bool ping = true;
  issueTLoad(x, current_tile, X_PING, EVENT_ID0);

  while (true) {
    const event_t current_ev = ping ? (event_t)EVENT_ID0 : (event_t)EVENT_ID1;
    const unsigned current_x_base = ping ? X_PING : X_PONG;
    const unsigned current_y_base = ping ? Y_PING : Y_PONG;
    const unsigned current_scale_base = ping ? SCALE_PING : SCALE_PONG;

    wait_flag(PIPE_MTE2, PIPE_V, current_ev);

    TileWork next_tile;
    const bool has_next = nextTile(sample_done, gm_offset_base, samples_to_process, next_tile);
    if (has_next) {
      const event_t next_ev = ping ? (event_t)EVENT_ID1 : (event_t)EVENT_ID0;
      const unsigned next_x_base = ping ? X_PONG : X_PING;
      issueTLoad(x, next_tile, next_x_base, next_ev);
    }

    const uint32_t row_index_base = current_tile.gm_offset / N;

    BulkTile xBulkTile(current_tile.sample_count);
    QuantTile yBulkTile(1, current_tile.elements);
    ReduceTmpTile reduceTmpTile(current_tile.sample_count);
    ReduceTileColMajor rowMaxTile(current_tile.sample_count, 1);
    ReduceTileColMajor rowMinTile(current_tile.sample_count, 1);
    ReduceTileRowMajor rowMaxTileRm(1, current_tile.sample_count);
    ReduceTileRowMajor rowMinTileRm(1, current_tile.sample_count);
    ScaleTile scaleTile(1, current_tile.sample_count);
    ScaleTile scaleFloorTile(1, current_tile.sample_count);
    TASSIGN(xBulkTile, current_x_base);
    TASSIGN(yBulkTile, current_y_base);
    TASSIGN(reduceTmpTile, REDUCE_TMP_BASE);
    TASSIGN(rowMaxTile, ROWMAX_BASE);
    TASSIGN(rowMinTile, ROWMIN_BASE);
    TASSIGN(rowMaxTileRm, EVEN_BASE);
    TASSIGN(rowMinTileRm, ODD_BASE);
    TASSIGN(scaleTile, current_scale_base);
    TASSIGN(scaleFloorTile, SCALE_FLOOR_BASE);
    wait_flag(PIPE_MTE3, PIPE_V, current_ev);

    // Phase 1: transform fp16 rows in UB.  The normalized Hadamard scale is not
    // applied to xBulkTile directly; it is folded into row_scales below.
    runHadamard128InPlace<InputT>(current_x_base, current_tile.sample_count);
    pipe_barrier(PIPE_V);

    // Phase 2: compute the quantization scale for each transformed row:
    //   row_scale = max(abs(Hx)) * (1 / sqrt(128)) / 127
    // The floor keeps all-zero rows from producing a zero scale.
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

    TCVT(scaleTile, rowMaxTileRm, RoundMode::CAST_NONE);
    pipe_barrier(PIPE_V);
    TMULS(scaleTile, scaleTile, INV_SQRT_N / QUANT_MAX_INT8);
    pipe_barrier(PIPE_V);

    TMULS(scaleFloorTile, scaleTile, 0.0f);
    pipe_barrier(PIPE_V);
    TADDS(scaleFloorTile, scaleFloorTile, SCALE_FLOOR);
    pipe_barrier(PIPE_V);
    TMAX(scaleTile, scaleTile, scaleFloorTile);
    pipe_barrier(PIPE_V);

    // Phase 3: produce int8 rows.  Use the same dynamic full-tile layout as
    // the pto-kernels dynamic-quant path for both full tiles and tails.
    quantizeHadamard128Tile<InputT, OutputT>(current_x_base, current_y_base, current_tile.sample_count);

    // Phase 4: store results.  MTE3 writes the current ping/pong Y/SCALE slots
    // to GM while the next loop iteration can compute into the opposite slots.
    set_flag(PIPE_V, PIPE_MTE3, current_ev);
    wait_flag(PIPE_V, PIPE_MTE3, current_ev);
    OutGlobal yGlobal(y + current_tile.gm_offset);
    TASSIGN(yGlobal, (y + current_tile.gm_offset));
    ScaleGlobal scaleGlobal(row_scales + row_index_base);
    TASSIGN(scaleGlobal, (row_scales + row_index_base));
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

__global__ AICORE void fast_hadamard_dynamic_quant_fp16_to_int8(__gm__ void* x, __gm__ void* y, __gm__ void* row_scales,
                                                                uint32_t batch) {
#if defined(__DAV_VEC__)
  // This branch is selected by compiling the PTO source for VecCore.
  // get_block_* identifies the AI Core block, while get_subblock* identifies
  // vector sub-cores inside that block.  Treating the pair as a flat vid gives
  // the total number of independent workers used to partition the batch rows.
  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  runTFastHadamardDynamicQuantInt8<half, int8_t>((__gm__ half*)x, (__gm__ int8_t*)y, (__gm__ float*)row_scales, batch,
                                                 num_cores, vid);
#else
  (void)x;
  (void)y;
  (void)row_scales;
  (void)batch;
#endif
}

extern "C" void call_dynamic_quant_int8_kernel(uint32_t blockDim, void* stream, uint8_t* x, uint8_t* y,
                                               float* row_scales, uint32_t batch) {
  // The Python wrapper passes the configured block count.  This C wrapper
  // doubles the launch grid to match the sub-block based worker indexing used
  // in fast_hadamard_dynamic_quant_fp16_to_int8.
  blockDim = blockDim * 2;
  fast_hadamard_dynamic_quant_fp16_to_int8<<<blockDim, nullptr, stream>>>(x, y, row_scales, batch);
}
