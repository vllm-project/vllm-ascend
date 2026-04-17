#include <pto/pto-inst.hpp>

using namespace pto;

#define DIV_ROUNDUP(x, y) (((x) + (y) - 1) / (y))

constexpr uint32_t X_BUFFER_BYTES = 32 * 1024;
constexpr uint32_t UB_HALF_BYTES = X_BUFFER_BYTES / 2;
constexpr uint32_t ELEMENTS_PER_TILE = X_BUFFER_BYTES / sizeof(half);
constexpr uint32_t UB_USABLE_BYTES = 184 * 1024;

// Double-buffered UB memory layout (ping/pong):
//   x buffer: 32KB (ELEMENTS_PER_TILE elements)
// Shared scratch for hadamard butterfly:
//   even buffer: 16KB (ELEMENTS_PER_TILE/2 elements)
//   odd buffer: 16KB (ELEMENTS_PER_TILE/2 elements)
constexpr unsigned X_PING = 0x00000;
constexpr unsigned X_PONG = X_PING + X_BUFFER_BYTES + 0x100;
constexpr unsigned EVEN_BASE = X_PONG + X_BUFFER_BYTES + 0x100;
constexpr unsigned ODD_BASE = EVEN_BASE + UB_HALF_BYTES + 0x100;
static_assert(ODD_BASE + UB_HALF_BYTES <= UB_USABLE_BYTES,
              "Fast Hadamard UB layout exceeds usable UB.");

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
  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using InGlobal = pto::GlobalTensor<InputT, ShapeDim5, StridDim5>;
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

template <typename T>
AICORE void runTFastHadamard(__gm__ T *x, uint32_t batch, uint32_t n,
                             uint32_t log2_n) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  if (n == 0 || n > ELEMENTS_PER_TILE) {
    return;
  }

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();

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

  using ShapeDim5 = pto::Shape<1, 1, 1, 1, ELEMENTS_PER_TILE>;
  using StridDim5 = pto::Stride<1, 1, 1, 1, 1>;
  using GlobalData = pto::GlobalTensor<T, ShapeDim5, StridDim5>;

  using FullTile =
      Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE, BLayout::RowMajor, -1, -1>;
  using HalfTile = Tile<TileType::Vec, T, 1, ELEMENTS_PER_TILE / 2,
                        BLayout::RowMajor, -1, -1>;

  const uint32_t samples_per_load =
      (n < ELEMENTS_PER_TILE) ? ELEMENTS_PER_TILE / n : 1;
  const uint32_t n_half = n >> 1;

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
    const unsigned x_base = ping ? X_PING : X_PONG;

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
    TASSIGN(xBulkTile, x_base);

    GlobalData xGlobal(x + current_tile.gm_offset);
    TASSIGN(xGlobal, (x + current_tile.gm_offset));

    if (!tryRunBatchedHadamard<T>(x_base, current_tile.sample_count, n,
                                  log2_n)) {
      for (uint32_t s = 0; s < current_tile.sample_count; ++s) {
        const unsigned row_base = x_base + s * n * sizeof(T);

        FullTile xRowTile(1, n);
        HalfTile xFirstHalf(1, n_half);
        HalfTile xSecondHalf(1, n_half);
        TASSIGN(xRowTile, row_base);
        TASSIGN(xFirstHalf, row_base);
        TASSIGN(xSecondHalf, row_base + n_half * sizeof(T));

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

    wait_flag(PIPE_MTE3, PIPE_V, current_ev);

    set_flag(PIPE_V, PIPE_MTE3, current_ev);
    wait_flag(PIPE_V, PIPE_MTE3, current_ev);

    TSTORE(xGlobal, xBulkTile);

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
#endif
}

}  // namespace

__global__ AICORE void fast_hadamard_fp16(__gm__ void *x, uint32_t batch,
                                          uint32_t n, uint32_t log2_n) {
  runTFastHadamard<half>((__gm__ half *)x, batch, n, log2_n);
}

extern "C" void call_kernel(uint32_t blockDim, void *stream, uint8_t *x,
                            uint32_t batch, uint32_t n, uint32_t log2_n) {
  blockDim = blockDim * 2;
  fast_hadamard_fp16<<<blockDim, nullptr, stream>>>(x, batch, n, log2_n);
}
