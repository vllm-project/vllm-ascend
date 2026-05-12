// ============================================================================
// wy_fast_kernel.cpp — WY representation for GatedDeltaNet chunk recurrence
//
// Computes the WY update matrices U and W for each chunk of C tokens:
//   U = A2 @ V     where A2 = A * beta_2d        (beta-scaled attention)
//   W = A1 @ K     where A1 = A * (exp(g)*beta)_2d (gate+beta-scaled attention)
//
// beta is the decay factor, g is the gate value, A is the triangular attention
// matrix (from the kkt kernel).  The column-broadcast notation x_2d means
// expanding a 1xC vector into a C/2 x C matrix by replicating across rows.
//
// Architecture: Vec+Cube cooperative kernel using cross-core synchronization.
//
//  Vec core (two sub-blocks for upper/lower C/2 rows):
//    For each chunk:
//      1. Load beta [H,T] and A [B,S,H,C], compute A2 = A * beta_2d -> ws
//      2. Load G [H,T], compute A1 = A * (exp(g)*beta)_2d -> ws
//      3. Signal Cube via cross-core flags when workspaces are ready
//
//  Cube core (waits for Vec signals):
//    For each chunk:
//      1. Load K, V from BSND layout into L1
//      2. Load A2 from workspace -> GEMM: U = A2 @ V
//      3. Load A1 from workspace -> GEMM: W = A1 @ K
//      4. Store U, W back to BSND layout
//
// NPU memory hierarchy used:
//   GM -> UB (Vec), GM -> L1 -> L0A/L0B -> L0C -> GM (Cube)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel uses BOTH the Cube engine (matrix multiply) and Vec engine
// (SIMD element-wise ops), running on SEPARATE physical cores that
// communicate via Global Memory (GM) + cross-core flags (FFTS).
//
// Execution flow:
//   Vec core:  load A,beta,G → compute A2,A1 → store to GM workspace
//   Cube core: wait for workspace → load A2/A1 + K/V → GEMM → store U,W
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(ub_tile, gm)      — ub_tile = gm[...]          (DMA: GM→UB, async MTE2)
//   TSTORE(gm, ub_tile)     — gm[...] = ub_tile          (DMA: UB→GM, async MTE3)
//   TCVT(dst, src, mode)    — dst = src.float() or .half() (type conversion)
//   TMOV(dst, src)          — dst = src.clone()
//   TMUL(d, a, b)           — d = a * b                   (element-wise)
//   TEXP(d, s)              — d = torch.exp(s)
//   TCOLEXPAND(2d, row)     — 2d[i,j] = row[j]  (broadcast row across all rows)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-block → L0A/L0B     (MTE1 for Cube GEMM)
//   TMATMUL(C, A, B)        — C = A @ B in Cube engine    (fp16→fp32 accumulate)
//   set_flag / wait_flag    — sync between pipes on SAME core
//   ffts_cross_core_sync    — signal ACROSS Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
#include <type_traits>
using namespace pto;

#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_HG
#define GDN_HG GDN_H
#endif

#ifndef GDN_D
#define GDN_D 128
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

#ifdef __CCE_AICORE__

namespace {

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512,
                              pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::RowMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int Rows, int Cols, int RowValid = Rows,
          int ColValid = Cols, pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN =
    pto::Tile<pto::TileType::Vec, T, Rows, Cols, pto::BLayout::ColMajor,
              RowValid, ColValid, pto::SLayout::NoneBox, 512, PadVal>;

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                           pto::BLayout::ColMajor, pto::DYNAMIC,
                           pto::DYNAMIC, pto::SLayout::RowMajor, 512,
                           pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                             pto::BLayout::RowMajor, pto::DYNAMIC,
                             pto::DYNAMIC, pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

// PTO cheat sheet for readers coming from PyTorch / NumPy:
//   - `GlobalTensor<T>` is a GM tensor view with explicit shape/stride metadata.
//   - `Tile<..., Mat, ...>` is an on-chip matrix tile used by Cube kernels.
//   - `Tile<..., Vec, ...>` is an on-chip UB tile used by SIMD vector kernels.
//   - `TileAcc<T, ...>` is the matmul accumulator tile.
//   - `TLOAD` / `TSTORE` are DMA copies between GM and local memory.
//   - `TCOLEXPAND` is broadcast like `x[None, :].expand(rows, -1)`.
//   - `TMUL`, `TEXP`, `TCVT` are vector ops on UB tiles.

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail, bool transpose_A = false, bool transpose_B = false>
AICORE PTO_INLINE void
gemm_v0(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                           TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                           TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C, bool clear)
{
  // Local K-sliced matmul helper:
  //   C = A @ B
  // PTO exposes the L1 -> L0 -> Cube movement explicitly, so keeping this tiny
  // helper local lets readers see the schedule without hiding it in a repo-wide
  // wrapper layer.
  //
  // PyTorch mental model:
  //   C = 0
  //   for k0 in range(0, K, kL0Size):
  //       C += A[:, k0:k1] @ B[k0:k1, :]
  constexpr uint32_t kL0Size = 128;
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

} // namespace

#endif

template <int32_t NumHeads, int32_t NumKeyHeads, int32_t HiddenSize,
          int32_t ChunkSize>
AICORE void wy_fast_kernel(
    __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *Beta_handle, __gm__ float *G_handle,
    __gm__ half *A_handle,
    __gm__ half *workspace_a1_handle, __gm__ half *workspace_a2_handle,
    __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  // WY recompute materializes two diagonal reweightings of the same A tile:
  //   A2[:, j] = A[:, j] * beta_j
  //   A1[:, j] = A[:, j] * exp(g_j) * beta_j
  // and then forms the two branch outputs
  //   U = A2 @ V,   W = A1 @ K.
  //
  // Shapes for one (sequence, head, chunk):
  //   A_chunk : [valid, valid]
  //   beta    : [valid]
  //   g       : [valid]
  //   K, V    : [valid, D]
  //
  // PyTorch / NumPy sketch:
  //   A2 = A_chunk * beta[None, :]
  //   A1 = A_chunk * (exp(g) * beta)[None, :]
  //   U  = A2 @ V_chunk
  //   W  = A1 @ K_chunk
  //
  // PTO split:
  //   Vec builds the two reweighted A tiles in workspace.
  //   Cube later consumes those workspaces in two GEMMs.
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  constexpr int32_t H = NumHeads;
  constexpr int32_t Hg = NumKeyHeads;
  static_assert(Hg > 0 && H % Hg == 0,
                "NumHeads must be divisible by NumKeyHeads");
  constexpr int32_t GROUP = H / Hg;
  constexpr int32_t BSND_V_STRIDE = H * HiddenSize;
  constexpr int32_t BSND_QK_STRIDE = Hg * HiddenSize;

  constexpr int32_t GHeadTileCols = ((NumHeads + 7) / 8) * 8;
  constexpr int32_t BetaHeadTileCols = ((NumHeads + 15) / 16) * 16;

  constexpr int32_t BetaHalfUbAddr = 0;
  constexpr int32_t A1HalfUbAddr   = 256;
  constexpr int32_t BetaUbAddr     = 16640;
  constexpr int32_t BetaRUbAddr    = 17152;
  constexpr int32_t Beta2dUbAddr   = 17664;
  constexpr int32_t TmpUbAddr      = 50432;
  constexpr int32_t A1UbAddr       = 75008;
  constexpr int32_t A2UbAddr       = 107776;
  constexpr int32_t A2HalfUbAddr   = 140544;
  constexpr int32_t GUbAddr        = 156928;
  constexpr int32_t GRUbAddr       = 157440;
  constexpr int32_t G2dUbAddr      = 157952;

  constexpr int32_t GBlockUbAddr    = TmpUbAddr;
  constexpr int32_t BetaBlockUbAddr = TmpUbAddr;

  constexpr int32_t WsA1Size = ChunkSize * ChunkSize;
  constexpr int32_t WsA2Size = ChunkSize * ChunkSize;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  TileUbDataND<half, 1, ChunkSize, 1, ChunkSize,
               pto::PadValue::Zero> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  TileUbDataND<half, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize, pto::PadValue::Zero> a1_ub_half;
  TASSIGN(a1_ub_half, A1HalfUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  TileUbDataND<uint8_t, 1, 24576, 1, 24576> tmp_ub;
  TASSIGN(tmp_ub, TmpUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> a1_ub;
  TASSIGN(a1_ub, A1UbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);
  TileUbDataND<half, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> a2_ub_half;
  TASSIGN(a2_ub_half, A2HalfUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize,
               pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, GUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize,
               HalfChunk, ChunkSize> g_2d_ub;
  TASSIGN(g_2d_ub, G2dUbAddr);

  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  TileMatL1<half, ChunkSize, HiddenSize,
            ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 32768);
  TileMatL1<half, ChunkSize, ChunkSize,
            ChunkSize, ChunkSize> a2_l1;
  TASSIGN(a2_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> u_l0;
  TASSIGN(u_l0, 0);
  TileMatL1<half, ChunkSize, ChunkSize,
            ChunkSize, ChunkSize> a1_l1;
  TASSIGN(a1_l1, 98304);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Vec prepares the two reweighted A workspaces (`A2` and `A1`) that the
  // Cube phase consumes later.
  if (cu_seqlens == nullptr) {
    bool first_iter = true;
    int64_t gi = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < NumHeads; ++head_idx) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            // Each Vec sub-block owns one HalfChunk-row stripe of the chunk.
            // For a tail chunk, the upper stripe (vid=0) may hold fewer than
            // 64 rows, and the lower stripe (vid=1) may hold only a suffix or
            // no rows at all.  `local_rows` is the exact number of live rows in
            // THIS sub-block's stripe.
            int32_t local_rows = valid_rows -
                static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;

            // Beta is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<half> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape, beta_stride);
              DynVecTile<half, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaHalfUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub_half, beta_load);
              }
            }

            // Load only the live rows for this sub-block, then zero-pad the
            // remainder of the HalfChunk tile.  The Cube phase always consumes
            // a full [HalfChunk, ChunkSize] workspace tile, so stale rows here
            // would leak garbage into ragged tails and cross-sequence boundaries.
            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start +
                    static_cast<int64_t>(vid) * HalfChunk) *
                   NumHeads + head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, ChunkSize);
              GmStride2D a_stride(NumHeads * ChunkSize);
              GmTensor2D<half> a_global(A_handle + a_gm_offset, a_shape,
                                        a_stride);
              DynVecTile<half, HalfChunk, ChunkSize, pto::PadValue::Zero> a_load(
                  local_rows, ChunkSize);
              TASSIGN(a_load, A1HalfUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk) {
                TFILLPAD_INPLACE(a1_ub_half, a_load);
              }
            } else {
              // Fully empty lower-half tail: materialize an all-zero tile so the
              // workspace still looks like a correctly padded HalfChunk block.
              TEXPANDS(a1_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            // Replicate beta_j across rows so every column j of A gets the same beta.
            // PyTorch-like:
            //   beta_2d = beta[None, :].expand(HalfChunk, ChunkSize)
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            // Form the beta-scaled tile that the later U = A2 * V matmul consumes.
            //   a2_ub = a1_ub * beta_2d_ub
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

            if (!first_iter) wait_flag_dev(3);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape, a2_stride);
              TSTORE(workspace_a2_global, a2_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

            // G is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D g_shape(1, valid_rows);
              GmStride2D g_stride(1);
              GmTensor2D<float> g_global(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  g_shape, g_stride);
              DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                  1, valid_rows);
              TASSIGN(g_load, GUbAddr);
              TLOAD(g_load, g_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(g_ub, g_load);
              }
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Build the g-based column weights before forming the W = A1 * K branch.
            // Torch-like:
            //   g_weight = exp(g) * beta
            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            // A1 keeps the same A columns but multiplies each one by exp(g_j) * beta_j.
            //   a1_ub = a1_ub * g_weight[None, :]
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

            if (!first_iter) wait_flag_dev(4);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a1_shape(HalfChunk, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<half> workspace_a1_global(
                  workspace_a1_handle +
                      static_cast<int64_t>(cid) * WsA1Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a1_shape, a1_stride);
              TSTORE(workspace_a1_global, a1_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
            first_iter = false;
          }
          gi++;
        }
      }
    }
  } else {
    // Same WY math as above; only the work enumeration changes for varlen input.
    int64_t gi = 0;
    bool first_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            // Same HalfChunk ownership rule as the fixed-length path above:
            // each Vec sub-block handles one 64-row stripe, and ragged varlen
            // tails may leave that stripe partially full or fully empty.
            int32_t local_rows = valid_rows -
                static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;
            int32_t head_idx = h;

            // Beta is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<half> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape, beta_stride);
              DynVecTile<half, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaHalfUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub_half, beta_load);
              }
            }

            // Tail-safe A loading is especially important in varlen mode because
            // the final chunk of one sequence may be immediately followed by the
            // first chunk of the next sequence in packed storage.
            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start +
                    static_cast<int64_t>(vid) * HalfChunk) *
                   NumHeads + head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, ChunkSize);
              GmStride2D a_stride(NumHeads * ChunkSize);
              GmTensor2D<half> a_global(A_handle + a_gm_offset, a_shape,
                                        a_stride);
              DynVecTile<half, HalfChunk, ChunkSize, pto::PadValue::Zero> a_load(
                  local_rows, ChunkSize);
              TASSIGN(a_load, A1HalfUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk) {
                TFILLPAD_INPLACE(a1_ub_half, a_load);
              }
            } else {
              // Empty stripe for this sub-block: write zeros so the downstream
              // full-tile Cube GEMM sees valid padding rather than old workspace.
              TEXPANDS(a1_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
            pipe_barrier(PIPE_V);
            TMOV(beta_r_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(beta_2d_ub, beta_r_ub);

            TCVT(a1_ub, a1_ub_half, pto::RoundMode::CAST_NONE);
            // Form the beta-scaled tile that the later U = A2 * V matmul consumes.
            TMUL(a2_ub, a1_ub, beta_2d_ub);
            TCVT(a2_ub_half, a2_ub, pto::RoundMode::CAST_NONE);

            if (!first_iter_v) wait_flag_dev(3);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape, a2_stride);
              TSTORE(workspace_a2_global, a2_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));

            // G is pre-transposed to [H, total_tokens] for contiguous loads.
            {
              GmShape2D g_shape(1, valid_rows);
              GmStride2D g_stride(1);
              GmTensor2D<float> g_global(
                  G_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  g_shape, g_stride);
              DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> g_load(
                  1, valid_rows);
              TASSIGN(g_load, GUbAddr);
              TLOAD(g_load, g_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(g_ub, g_load);
              }
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            // Build the g-based column weights before forming the W = A1 * K branch.
            TEXP(g_ub, g_ub);
            pipe_barrier(PIPE_V);
            TMUL(g_ub, g_ub, beta_ub);
            pipe_barrier(PIPE_V);
            TMOV(g_r_ub, g_ub);
            pipe_barrier(PIPE_V);
            TCOLEXPAND(g_2d_ub, g_r_ub);
            TMUL(a1_ub, a1_ub, g_2d_ub);
            TCVT(a1_ub_half, a1_ub, pto::RoundMode::CAST_NONE);

            if (!first_iter_v) wait_flag_dev(4);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a1_shape(HalfChunk, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<half> workspace_a1_global(
                  workspace_a1_handle +
                      static_cast<int64_t>(cid) * WsA1Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a1_shape, a1_stride);
              TSTORE(workspace_a1_global, a1_ub_half);
            }
            pipe_barrier(PIPE_ALL);
            ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
            first_iter_v = false;
          }
          gi++;
        }
      }
    }
  }
#endif

#if defined(__DAV_C220_CUBE__)
  // Cube consumes the two Vec-generated workspaces and turns them into the
  // branch outputs U and W.
  if (cu_seqlens == nullptr) {
    int64_t gi = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t head_idx = 0; head_idx < NumHeads; ++head_idx) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;

            int32_t head_g = head_idx / GROUP;
            int64_t k_off =
                (chunk_token_start * static_cast<int64_t>(Hg) +
                 static_cast<int64_t>(head_g)) *
                static_cast<int64_t>(HiddenSize);
            int64_t v_off =
                (chunk_token_start * static_cast<int64_t>(H) +
                 static_cast<int64_t>(head_idx)) *
                static_cast<int64_t>(HiddenSize);

            {
              GmShape2D k_shape(valid_rows, HiddenSize);
              GmStride2D k_stride(BSND_QK_STRIDE);
              GmTensor2D<half> k_global(K_handle + k_off, k_shape, k_stride);
              DynMatL1<half, ChunkSize, HiddenSize> k_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(k_l1_load, 0);
              TLOAD(k_l1_load, k_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(k_l1_load, k_l1_load);
              }
            }
            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_V_STRIDE);
              GmTensor2D<half> v_global(V_handle + v_off, v_shape, v_stride);
              DynMatL1<half, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(v_l1_load, 32768);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

            wait_flag_dev(2);
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape, a2_stride);
              // Load the Vec-prepared A2 tile:
              //   A2 = A * beta[None, :]
              TLOAD(a2_l1, workspace_a2_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // U = A2 * V keeps the beta-scaled path separate from the K-side update.
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_V_STRIDE);
              GmTensor2D<half> u_global(U_handle + v_off, u_shape, u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              // Store only the valid token rows even though the accumulator tile is
              // physically ChunkSize x HiddenSize.
              TSTORE(u_global, u_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));

            wait_flag_dev(1);
            {
              GmShape2D a1_shape(ChunkSize, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<half> workspace_a1_global(
                  workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size,
                  a1_shape, a1_stride);
              // Load the Vec-prepared A1 tile:
              //   A1 = A * (exp(g) * beta)[None, :]
              TLOAD(a1_l1, workspace_a1_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // W = A1 * K uses the g-reweighted path for the complementary WY factor.
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a1_l1, k_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_V_STRIDE);
              GmTensor2D<half> w_global(W_handle + v_off, w_shape, w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
          }
          gi++;
        }
      }
    }
  } else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;

            int32_t head_g = head_idx / GROUP;
            int64_t k_off =
                (chunk_token_start * static_cast<int64_t>(Hg) +
                 static_cast<int64_t>(head_g)) *
                static_cast<int64_t>(HiddenSize);
            int64_t v_off =
                (chunk_token_start * static_cast<int64_t>(H) +
                 static_cast<int64_t>(head_idx)) *
                static_cast<int64_t>(HiddenSize);

            {
              GmShape2D k_shape(valid_rows, HiddenSize);
              GmStride2D k_stride(BSND_QK_STRIDE);
              GmTensor2D<half> k_global(K_handle + k_off, k_shape,
                                        k_stride);
              DynMatL1<half, ChunkSize, HiddenSize> k_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(k_l1_load, 0);
              TLOAD(k_l1_load, k_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(k_l1_load, k_l1_load);
              }
            }
            {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_V_STRIDE);
              GmTensor2D<half> v_global(V_handle + v_off, v_shape,
                                        v_stride);
              DynMatL1<half, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                              HiddenSize);
              TASSIGN(v_l1_load, 32768);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

            wait_flag_dev(2);
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<half> workspace_a2_global(
                  workspace_a2_handle + static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape, a2_stride);
              TLOAD(a2_l1, workspace_a2_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // U = A2 * V keeps the beta-scaled path separate from the K-side update.
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a2_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_V_STRIDE);
              GmTensor2D<half> u_global(U_handle + v_off, u_shape,
                                        u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              TSTORE(u_global, u_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (3 << 8));

            wait_flag_dev(1);
            {
              GmShape2D a1_shape(ChunkSize, ChunkSize);
              GmStride2D a1_stride(ChunkSize);
              GmTensor2D<half> workspace_a1_global(
                  workspace_a1_handle + static_cast<int64_t>(cid) * WsA1Size,
                  a1_shape, a1_stride);
              TLOAD(a1_l1, workspace_a1_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            // W = A1 * K uses the g-reweighted path for the complementary WY factor.
            gemm_v0<half, float,
                ChunkSize, HiddenSize, ChunkSize,
                ChunkSize, HiddenSize, ChunkSize,
                KTail, false, false>(a1_l1, k_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_V_STRIDE);
              GmTensor2D<half> w_global(W_handle + v_off, w_shape,
                                        w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (4 << 8));
          }
          gi++;
        }
      }
    }
  }
#endif
}

extern "C" __global__ AICORE void launch_wy_fast(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *V_handle,
    __gm__ uint8_t *Beta_handle, __gm__ uint8_t *G_handle,
    __gm__ uint8_t *A_handle,
    __gm__ uint8_t *workspace_a1_handle, __gm__ uint8_t *workspace_a2_handle,
    __gm__ uint8_t *W_handle, __gm__ uint8_t *U_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  wy_fast_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ half *>(workspace_a1_handle),
      reinterpret_cast<__gm__ half *>(workspace_a2_handle),
      reinterpret_cast<__gm__ half *>(W_handle),
      reinterpret_cast<__gm__ half *>(U_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *k, uint8_t *v, uint8_t *beta, uint8_t *g_sum, uint8_t *A,
    uint8_t *workspace_a1, uint8_t *workspace_a2,
    uint8_t *w, uint8_t *u,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_wy_fast<<<block_dim, nullptr, stream>>>(
      k, v, beta, g_sum, A,
      workspace_a1, workspace_a2,
      w, u,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
