// ============================================================================
// chunk_h_kernel.cpp — Recurrent hidden state update for GatedDeltaNet
//
// Mathematical recurrence per chunk c:
//   S_{c+1} = exp(g_last) * S_c  +  K^T @ V
//
// where g_last = exp(g[valid-1]) is the chunk's final gate value, S is the
// D×D hidden state, K ∈ ℝ^{C×D}, V ∈ ℝ^{C×D}, and g ∈ ℝ^C is the per-token
// gate.
//
// ── Cube phase (two GEMMs per chunk, sequentially): ──────────────────────
//   1. WS = W @ S       project current state through W (wy_fast output)
//      W ∈ ℝ^{C×D}, S ∈ ℝ^{D×D}  →  WS ∈ ℝ^{C×D}
//   2. KV = K^T @ V     outer product of keys and values (transpose_A!)
//      K stored as D×C, V ∈ ℝ^{C×D}  →  KV ∈ ℝ^{D×D}
//
// ── Vec phase (two sub-blocks handle upper/lower C/2 rows): ─────────────
//   For each chunk:
//     1. Load K, G (pre-transposed), U (from wy_fast)
//     2. Compute coeff[i] = exp(g[i] - g[valid-1])  — time-decay scaling
//        Uses TROWEXPAND to broadcast coefficients across D columns
//     3. Scale K: K_scaled[i,:] = K[i,:] * coeff[i]
//     4. Load WS from Cube workspace, compute V_new = U - WS (residual)
//     5. Store V_new and K_scaled to workspace for Cube's next iteration
//     6. Update state: S = exp(g_last) * S + KV (from Cube workspace)
//     7. Store final state FS after last chunk
//
// Cross-core sync: Cube→Vec flags for WS/KV ready, Vec→Cube flags for
// K/S ready.
//
// Inputs:
//   K  [total_tokens, Hg, D] half   — keys (BSND layout; GQA/MQA group heads)
//   W  [total_tokens, H, D]  half   — wy_fast output (BSND layout)
//   U  [total_tokens, H, D]  half   — values pre-residual (BSND layout)
//   G  [H, total_tokens]     float  — pre-transposed cumulative gates
//   S  [total_chunks, H, D, D] half — per-chunk state snapshots (output)
//   V  [total_tokens, H, D]  half   — residual-corrected values (output)
//   FS [batch, H, D, D]      half   — final state per sequence (output)
//   workspace [per-core scratch]     — Cube↔Vec communication buffer
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B/L0C (Cube GEMM registers)
//   GM → UB (Vec-accessible, on-chip SRAM)
//   Cross-core sync via FFTS (Fast Fine-grained Task Synchronization)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This is the most complex kernel in the GDN suite. It implements the
// recurrent state update, requiring sequential chunk processing (chunks
// within a sequence CANNOT be parallelized — each depends on the previous).
//
// Key PTO APIs (numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data        (DMA: GM→L1 or GM→UB)
//   TSTORE(gm, src)         — gm_data = src        (DMA: UB/L0C→GM)
//   TASSIGN(tile, addr)     — tile = memory[addr]   (bind tile to buffer address)
//   TCVT(dst, src, mode)    — dst = src.float()/.half()
//   TMOV(dst, src)          — dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMULS(d, s, scalar)     — d = s * scalar       (scalar multiply)
//   TADDS(d, s, scalar)     — d = s + scalar       (scalar add)
//   TEXP(d, s)              — d = torch.exp(s)
//   TEXPANDS(tile, scalar)  — tile[:] = scalar     (fill with constant)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i]    (broadcast col across row dim)
//   TFILLPAD(dst, src)      — zero-fill L1 tile padding (for tail chunks)
//   TEXTRACT(l0, l1, r, c)  — L1 sub-tile → L0A/L0B
//   TRESHAPE(zn, nz)        — reinterpret layout NZ↔ZN (logical transpose, free)
//   TMATMUL(C, A, B)        — C = A @ B (Cube GEMM, fp16 inputs → fp32 accum)
//   set_flag/wait_flag      — pipe sync within same core
//   ffts_cross_core_sync    — cross-core signal Cube↔Vec
//   wait_flag_dev(flag)     — wait for cross-core signal
//   GetValue(idx)           — read a single scalar from a UB tile (slow, use sparingly)
//
// ── Workspace memory layout (shared between Cube and Vec via GM) ──────
// Each AI core has its own workspace region to avoid contention:
//   WS_WS [C×D]:  Cube writes WS = W @ S here → Vec reads it
//   WS_K  [D×C]:  Vec writes K_scaled here → Cube reads it for KV = K^T @ V
//   WS_S  [D×D]:  Vec writes current state S here → Cube reads it for GEMM 1
//   WS_KV [D×D]:  Cube writes KV = K^T @ V here → Vec reads it to update S
//
// Data flow per chunk (think of it as a ping-pong between Cube and Vec):
//   Vec: write S₀ to WS_S → signal Cube (flag 3)
//   Cube: read S from WS_S, load W → compute WS = W@S → write WS_WS → signal Vec (flag 0)
//   Vec: read WS, compute V_new = U - WS, compute K_scaled → write WS_K → signal Cube (flag 1)
//   Cube: read K from WS_K, load V → compute KV = K^T@V → write WS_KV → signal Vec (flag 2)
//   Vec: read KV, update S = exp(g_last)*S + KV → write S to WS_S → signal Cube (flag 3)
//   ... repeat for next chunk ...
// ============================================================================

#include <pto/pto-inst.hpp>
#include <type_traits>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

#ifdef __CCE_AICORE__

namespace {

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

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                            pto::BLayout::ColMajor, RowValid, ColValid,
                            pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat, T, Rows, Cols,
                              pto::BLayout::RowMajor, RowValid, ColValid,
                              pto::SLayout::ColMajor, 512,
                              pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0A = pto::Tile<pto::TileType::Left, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::RowMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols>
using TileMatL0B = pto::Tile<pto::TileType::Right, T, Rows, Cols,
                             pto::BLayout::RowMajor, RowValid, ColValid,
                             pto::SLayout::ColMajor, 512,
                             pto::PadValue::Zero>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::RowMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512, PadVal>;

template <typename T, int32_t Rows, int32_t Cols, int32_t RowValid = Rows,
          int32_t ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataDN = pto::Tile<pto::TileType::Vec, T, Rows, Cols,
                               pto::BLayout::ColMajor, RowValid, ColValid,
                               pto::SLayout::NoneBox, 512, PadVal>;

// PTO cheat sheet for the recurrent kernel:
//   - `GlobalTensor<T>` is a GM tensor view with explicit runtime shape/stride.
//   - `Tile<..., Mat, ...>` lives in L1 and feeds Cube matmul instructions.
//   - `Tile<..., Vec, ...>` lives in UB for elementwise vector work.
//   - `TileAcc<T, ...>` is a Cube accumulator tile.
//   - `TLOAD` / `TSTORE` are DMA copies between GM and on-chip memory.
//   - `TROWEXPAND` broadcasts a column vector across the feature dimension.
//   - `TFILLPAD(_INPLACE)` zero-pads tail rows so full-tile code can still run.

template <typename T1, typename T2, uint32_t M, uint32_t N, uint32_t K,
          uint32_t validM = M, uint32_t validN = N, uint32_t validK = K,
          uint32_t K_tail = K, bool transpose_A = false,
          bool transpose_B = false>
AICORE PTO_INLINE void
gemm_v0(std::conditional_t<transpose_A, TileMatL1<T1, K, M, validK, validM>,
                           TileMatL1<T1, M, K, validM, validK>> &A,
        std::conditional_t<transpose_B, TileMatL1<T1, N, K, validN, validK>,
                           TileMatL1<T1, K, N, validK, validN>> &B,
        pto::TileAcc<T2, M, N, validM, validN> &C, bool clear)
{
  // Local K-sliced matmul helper:
  //   C = A @ B
  // PTO exposes the L1/L0 staging explicitly, so this stays as a tiny file-
  // local helper instead of a shared wrapper.
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
AICORE void chunk_h_kernel(
    __gm__ half *K_handle, __gm__ half *W_handle, __gm__ half *U_handle,
    __gm__ float *G_handle,
    __gm__ half *S_handle, __gm__ half *V_handle, __gm__ half *FS_handle,
    __gm__ half *workspace_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  // chunk_h advances the recurrent hidden state chunk by chunk:
  //   ws_i      = W_i @ S_i
  //   v_i_new   = U_i - ws_i
  //   k_i_tilde = exp(g_last - g_i) * K_i
  //   S_{i+1}   = exp(g_last) * S_i + k_i_tilde^T @ v_i_new.
  //
  // Shapes for one (sequence, head, chunk):
  //   W_i, U_i, K_i, V_i_new : [valid, D]
  //   S_i, S_{i+1}           : [D, D]
  //
  // PyTorch / NumPy sketch:
  //   ws = W_i @ S_i
  //   v_new = U_i - ws
  //   decay = exp(g_last - g_i)[:, None]
  //   k_tilde = decay * K_i
  //   kv = k_tilde.T @ v_new
  //   S = exp(g_last) * S + kv
  //
  // PTO split:
  //   Cube forms the two matmuls (`W_i @ S_i` and `K_i^T @ V_i_new`).
  //   Vec does the elementwise gating/decay and carries the running state.
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t D = HiddenSize;
  constexpr int32_t C = ChunkSize;
  constexpr int32_t H = NumHeads;
  constexpr int32_t Hg = NumKeyHeads;
  static_assert(Hg > 0 && H % Hg == 0,
                "NumHeads must be divisible by NumKeyHeads");
  constexpr int32_t GROUP = H / Hg;
  constexpr int32_t HalfC = C / 2;
  constexpr int32_t BSND_QKV_STRIDE = H * D;
  constexpr int32_t BSND_K_STRIDE = Hg * D;
  constexpr int32_t DD = D * D;

  constexpr int32_t WS_WS = 0;
  constexpr int32_t WS_K  = DD;
  constexpr int32_t WS_S  = DD * 2;
  constexpr int32_t WS_KV = DD * 3;
  constexpr int32_t WS_PER_CORE = DD * 4;

  TileMatL1<half, D, D, D, D> s_l1;
  TASSIGN(s_l1, 0);
  TileMatL1<half, C, D, C, D> w_l1;
  TASSIGN(w_l1, D * D * sizeof(half));
  TileAcc<float, C, D, C, D> ws_l0;
  TASSIGN(ws_l0, 0);
  TileMatL1<half, D, C, D, C> k_l1;
  TASSIGN(k_l1, (DD + C * D) * sizeof(half));
  TileMatL1<half, C, D, C, D> v_l1;
  TASSIGN(v_l1, (DD + C * D + D * C) * sizeof(half));
  TileAcc<float, D, D, D, D> kv_l0;
  TASSIGN(kv_l0, C * D * sizeof(float));

  constexpr int32_t G_BLOCK_UB = 0;
  // Leading UB scratch: legacy kernels used ``C * NumHeads * sizeof(float)``, which overflows UB when
  // ``NumHeads`` is 32/48/64. Keep the same slack as the historical ``GDN_H=16`` build (8192 bytes).
  constexpr int32_t ZERO_UB =
      ChunkSize * 16 * static_cast<int32_t>(sizeof(float));
  constexpr int32_t S_UB = ZERO_UB + 64 * sizeof(float);
  constexpr int32_t K_UB_HALF = S_UB + HalfC * D * sizeof(float);
  constexpr int32_t G_UB = K_UB_HALF + HalfC * D * sizeof(half);
  constexpr int32_t U_UB_HALF = G_UB + C * sizeof(float);
  constexpr int32_t K_UB = U_UB_HALF + HalfC * D * sizeof(half);
  constexpr int32_t G_V_UB = K_UB + HalfC * D * sizeof(float);
  constexpr int32_t COEFF_UB = G_V_UB + 64 * sizeof(float);
  constexpr int32_t U_UB = COEFF_UB + 64 * sizeof(float);
  constexpr int32_t WS_UB = U_UB + HalfC * D * sizeof(float);
  constexpr int32_t KV_UB = U_UB_HALF;
  constexpr int32_t S_UB_HALF = WS_UB + HalfC * D * sizeof(float);

  TileUbDataND<float, 1, 64, 1, 64> zero_ub;
  TASSIGN(zero_ub, ZERO_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> s_ub;
  TASSIGN(s_ub, S_UB);
  TileUbDataND<half, HalfC, D, HalfC, D, pto::PadValue::Zero> k_ub_half;
  TASSIGN(k_ub_half, K_UB_HALF);
  TileUbDataND<float, 1, C, 1, C, pto::PadValue::Zero> g_ub;
  TASSIGN(g_ub, G_UB);
  TileUbDataND<half, HalfC, D, HalfC, D> s_ub_half;
  TASSIGN(s_ub_half, S_UB_HALF);
  TileUbDataND<half, HalfC, D, HalfC, D, pto::PadValue::Zero> u_ub_half;
  TASSIGN(u_ub_half, U_UB_HALF);
  TileUbDataND<float, HalfC, D, HalfC, D> k_ub;
  TASSIGN(k_ub, K_UB);
  TileUbDataND<float, 1, 64, 1, 64> g_v_ub;
  TASSIGN(g_v_ub, G_V_UB);
  TileUbDataND<float, 1, 64, 1, 64> coeff_ub;
  TASSIGN(coeff_ub, COEFF_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> u_ub;
  TASSIGN(u_ub, U_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> ws_ub;
  TASSIGN(ws_ub, WS_UB);
  TileUbDataND<float, HalfC, D, HalfC, D> kv_ub;
  TASSIGN(kv_ub, KV_UB);

  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * H;

#if defined(__DAV_C220_CUBE__)
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;
    // One per-core scratch region stores:
    //   WS_WS : ws = W_i @ S_i
    //   WS_K  : k_tilde
    //   WS_S  : running state S_i
    //   WS_KV : k_tilde^T @ v_i_new

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      wait_flag_dev(3);

      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;

      {
        GmShape2D s_shape(D, D);
        GmStride2D s_stride(D);
        GmTensor2D<half> s_global(workspace_handle + ws_base + WS_S, s_shape,
                                  s_stride);
        DynMatL1<half, D, D> s_l1_load(D, D);
        TASSIGN(s_l1_load, 0);
        // Load the previous recurrent state S_i from per-core workspace.
        TLOAD(s_l1_load, s_global);
      }

      int64_t w_offset = ((chunk_start) * H + head) * D;
      {
        GmShape2D w_shape(static_cast<int32_t>(valid), D);
        GmStride2D w_stride(BSND_QKV_STRIDE);
        GmTensor2D<half> w_global(W_handle + w_offset, w_shape, w_stride);
        DynMatL1<half, C, D> w_l1_load(static_cast<int32_t>(valid), D);
        TASSIGN(w_l1_load, D * D * static_cast<int32_t>(sizeof(half)));
        TLOAD(w_l1_load, w_global);
        if (valid != C) {
          TFILLPAD(w_l1_load, w_l1_load);
        }
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      // Apply the carried recurrent state to every token in this chunk.
      gemm_v0<half, float, C, D, D, C, D, D, D, false, false>(
          w_l1, s_l1, ws_l0, (bool)1);

      {
        GmShape2D ws_shape(C, D);
        GmStride2D ws_stride(D);
        GmTensor2D<half> ws_global(workspace_handle + ws_base + WS_WS,
                                   ws_shape, ws_stride);
        DynAccTile<float, C, D> ws_store(C, D);
        TASSIGN(ws_store, 0);
        // Save ws_i so the Vec phase can do `v_new = U_i - ws_i`.
        TSTORE(ws_global, ws_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      wait_flag_dev(1);

      {
        GmShape2D k_shape(D, C);
        GmStride2D k_stride(C);
        GmTensor2D<half> k_global(workspace_handle + ws_base + WS_K, k_shape,
                                  k_stride);
        DynMatL1<half, D, C> k_l1_load(D, C);
        TASSIGN(k_l1_load, (DD + C * D) * static_cast<int32_t>(sizeof(half)));
        TLOAD(k_l1_load, k_global);
      }

      int64_t v_offset = ((chunk_start) * H + head) * D;
      {
        GmShape2D v_shape(static_cast<int32_t>(valid), D);
        GmStride2D v_stride(BSND_QKV_STRIDE);
        GmTensor2D<half> v_global(V_handle + v_offset, v_shape, v_stride);
        DynMatL1<half, C, D> v_l1_load(static_cast<int32_t>(valid), D);
        TASSIGN(v_l1_load,
                (DD + C * D + D * C) * static_cast<int32_t>(sizeof(half)));
        TLOAD(v_l1_load, v_global);
        if (valid != C) {
          TFILLPAD(v_l1_load, v_l1_load);
        }
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      // This chunk contributes the additive update K_i^T V_i to the state recurrence.
      gemm_v0<half, float, D, D, C, D, D, C, C, true, false>(
          k_l1, v_l1, kv_l0, (bool)1);

      {
        GmShape2D kv_shape(D, D);
        GmStride2D kv_stride(D);
        GmTensor2D<half> kv_global(workspace_handle + ws_base + WS_KV,
                                   kv_shape, kv_stride);
        DynAccTile<float, D, D> kv_store(D, D);
        TASSIGN(kv_store, C * D * static_cast<int32_t>(sizeof(float)));
        // Save kv = k_tilde^T @ v_i_new so Vec can finish the state update.
        TSTORE(kv_global, kv_store);
      }
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
    }
  }
#endif
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Vec owns the running recurrent state S_i and updates it after every chunk.
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) break;

    int64_t head = pid % H;
    int64_t head_g = head / GROUP;
    int64_t seq_idx = pid / H;

    int64_t bos, slen;
    int64_t chunk_offset = 0;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
      slen = eos - bos;
      for (int64_t si = 0; si < seq_idx; ++si) {
        int64_t sb = static_cast<int64_t>(cu_seqlens[si]);
        int64_t se = static_cast<int64_t>(cu_seqlens[si + 1]);
        chunk_offset += (se - sb + C - 1) / C;
      }
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
      chunk_offset = seq_idx * ((seq_len + C - 1) / C);
    }
    int64_t num_chunks = (slen + C - 1) / C;
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;

    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    TEXPANDS(zero_ub, 0.0f);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    // Start each sequence/head recurrence from S_0 = 0.
    TEXPANDS(s_ub, 0.0f);

    TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    {
      // `workspace_handle` is a `half*`, so all offsets here are in half elements.
      GmShape2D s_shape(HalfC, D);
      GmStride2D s_stride(D);
      GmTensor2D<half> s_global(
          workspace_handle + ws_base + WS_S + vid * HalfC * D,
          s_shape, s_stride);
      DynVecTile<half, HalfC, D> s_store(HalfC, D);
      TASSIGN(s_store, S_UB_HALF);
      TSTORE(s_global, s_store);
    }
    ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

    int64_t chunk_start_0 = bos;
    int64_t valid0 = slen;
    if (valid0 > C) valid0 = C;
    // Vec work is split by row stripe, not by individual token.  For the first
    // chunk we compute exactly how many live rows belong to this sub-block's
    // HalfC stripe so short tails do not overrun the packed BSND input.
    int32_t valid_rows_0 =
        static_cast<int32_t>(valid0 - static_cast<int64_t>(vid) * HalfC);
    if (valid_rows_0 < 0) valid_rows_0 = 0;
    if (valid_rows_0 > HalfC) valid_rows_0 = HalfC;

    int64_t k_offset_0 =
        (chunk_start_0 * Hg + head_g) * D + vid * HalfC * BSND_K_STRIDE;
    if (valid_rows_0 > 0) {
      GmShape2D k_shape(valid_rows_0, D);
      GmStride2D k_stride(BSND_K_STRIDE);
      GmTensor2D<half> k_global(K_handle + k_offset_0, k_shape, k_stride);
      DynVecTile<half, HalfC, D, pto::PadValue::Zero> k_load(valid_rows_0, D);
      TASSIGN(k_load, K_UB_HALF);
      TLOAD(k_load, k_global);
      if (valid_rows_0 != HalfC) {
        TFILLPAD_INPLACE(k_ub_half, k_load);
      }
    } else {
      // Empty stripe (typically vid=1 on a very short tail chunk): synthesize
      // a zero tile so later full-width vector math and workspace stores still
      // observe proper padding semantics.
      TEXPANDS(k_ub, 0.0f);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
    }

    {
      GmShape2D g_shape(1, static_cast<int32_t>(valid0));
      GmStride2D g_stride(1);
      GmTensor2D<float> g_global(G_handle + head * total_tokens + chunk_start_0,
                                 g_shape, g_stride);
      DynVecTile<float, 1, C, pto::PadValue::Zero> g_load(
          1, static_cast<int32_t>(valid0));
      TASSIGN(g_load, G_UB);
      TLOAD(g_load, g_global);
      if (valid0 != C) {
        TFILLPAD_INPLACE(g_ub, g_load);
      }
    }

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (int32_t ci = 0; ci < static_cast<int32_t>(num_chunks); ++ci) {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) valid = C;
      int32_t valid_rows =
          static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows < 0) valid_rows = 0;
      if (valid_rows > HalfC) valid_rows = HalfC;
      // Each Vec subblock owns one contiguous HalfC-row stripe of the chunk.
      // For short tail chunks, `valid_rows` may be smaller or even zero.  This
      // is the key fix that keeps ragged tails and dense varlen boundary mixes
      // from reading or writing beyond the live rows in this stripe.

      int64_t u_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
      if (valid_rows > 0) {
        GmShape2D u_shape(valid_rows, D);
        GmStride2D u_stride(BSND_QKV_STRIDE);
        GmTensor2D<half> u_global(U_handle + u_offset, u_shape, u_stride);
        DynVecTile<half, HalfC, D, pto::PadValue::Zero> u_load(valid_rows, D);
        TASSIGN(u_load, U_UB_HALF);
        TLOAD(u_load, u_global);
        if (valid_rows != HalfC) {
          TFILLPAD_INPLACE(u_ub_half, u_load);
        }
      } else {
        // No live rows for this stripe in the current chunk; keep the tile
        // explicitly zero-padded so the remainder of the recurrence logic can
        // run in full-tile form without special-casing every later step.
        TEXPANDS(u_ub, 0.0f);
        TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      }

      TCVT(k_ub, k_ub_half, pto::RoundMode::CAST_NONE);

      TileUbDataND<float, 1, 64, 1, 64> g_ub_temp;
      TASSIGN(g_ub_temp, G_UB + vid * 64 * sizeof(float));
      TMOV(g_v_ub, g_ub_temp);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      float g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      // Rebase the chunk gate around g_last so the intra-chunk decay stays numerically local.
      // Torch-like:
      //   coeff = exp(g_last - g_rows_owned_by_this_subblock)
      TADDS(coeff_ub, g_v_ub, -g_last);
      pipe_barrier(PIPE_V);
      TSUB(coeff_ub, zero_ub, coeff_ub);
      pipe_barrier(PIPE_V);
      TEXP(coeff_ub, coeff_ub);

      TEXP(g_ub, g_ub);

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(u_ub, u_ub_half, pto::RoundMode::CAST_NONE);

      TileUbDataDN<float, HalfC, 1, HalfC, 1> coeff_col_ub;
      TASSIGN(coeff_col_ub, COEFF_UB);
      TileUbDataND<float, HalfC, D, HalfC, D> coeff_2d_ub;
      TASSIGN(coeff_2d_ub, WS_UB);
      // Broadcast one decay scalar per token row across the D feature columns:
      //   coeff_2d[row, :] = coeff[row]
      TROWEXPAND(coeff_2d_ub, coeff_col_ub);
      pipe_barrier(PIPE_V);
      // `k_ub` now holds k_tilde = exp(g_last - g_i) * K_i.
      TMUL(k_ub, k_ub, coeff_2d_ub);
      pipe_barrier(PIPE_V);

      wait_flag_dev(0);
      {
        GmShape2D ws_shape(HalfC, D);
        GmStride2D ws_stride(D);
        GmTensor2D<half> ws_global(
            workspace_handle + ws_base + WS_WS + vid * HalfC * D,
            ws_shape, ws_stride);
        DynVecTile<half, HalfC, D, pto::PadValue::Zero> ws_load(HalfC, D);
        TASSIGN(ws_load, U_UB_HALF);
        TLOAD(ws_load, ws_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(ws_ub, u_ub_half, pto::RoundMode::CAST_NONE);
      // v_i_new = U_i - W_i @ S_i.
      // In PyTorch notation:
      //   u_ub = u_ub - ws_ub
      TSUB(u_ub, u_ub, ws_ub);
      TCVT(u_ub_half, u_ub, pto::RoundMode::CAST_NONE);
      TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      int64_t v_offset = (chunk_start * H + head) * D + vid * HalfC * BSND_QKV_STRIDE;
      if (valid_rows > 0) {
        GmShape2D v_shape(valid_rows, D);
        GmStride2D v_stride(BSND_QKV_STRIDE);
        GmTensor2D<half> v_global(V_handle + v_offset, v_shape, v_stride);
        DynVecTile<half, HalfC, D> v_store(valid_rows, D);
        TASSIGN(v_store, U_UB_HALF);
        TSTORE(v_global, v_store);
      }

      // Spill both V_i_new and k_i_tilde so the Cube stage can form
      // k_i_tilde^T @ V_i_new for this chunk.
      {
        GmShape2D k_shape(HalfC, D);
        GmStride2D k_stride(D);
        GmTensor2D<half> k_global(
            workspace_handle + ws_base + WS_K + vid * HalfC * D,
            k_shape, k_stride);
        DynVecTile<half, HalfC, D> k_store(HalfC, D);
        TASSIGN(k_store, K_UB_HALF);
        TSTORE(k_global, k_store);
      }

      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      float exp_g_last = g_ub.GetValue(static_cast<int32_t>(valid) - 1);
      // Carry the recurrence across chunks: S_{i+1} = exp(g_last) * S_i + K_i^T V_i.
      TMULS(s_ub, s_ub, exp_g_last);

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        int64_t next_start = bos + static_cast<int64_t>(ci + 1) * C;
        int64_t next_valid = slen - static_cast<int64_t>(ci + 1) * C;
        if (next_valid > C) next_valid = C;
        int32_t next_valid_rows = static_cast<int32_t>(
            next_valid - static_cast<int64_t>(vid) * HalfC);
        if (next_valid_rows < 0) next_valid_rows = 0;
        if (next_valid_rows > HalfC) next_valid_rows = HalfC;

        int64_t nk_off =
            (next_start * Hg + head_g) * D + vid * HalfC * BSND_K_STRIDE;
        if (next_valid_rows > 0) {
          GmShape2D k_shape(next_valid_rows, D);
          GmStride2D k_stride(BSND_K_STRIDE);
          GmTensor2D<half> k_global(K_handle + nk_off, k_shape, k_stride);
          DynVecTile<half, HalfC, D, pto::PadValue::Zero> k_load(
              next_valid_rows, D);
          TASSIGN(k_load, K_UB_HALF);
          TLOAD(k_load, k_global);
          if (next_valid_rows != HalfC) {
            TFILLPAD_INPLACE(k_ub_half, k_load);
          }
        } else {
          // Same tail-safe zero materialization for the prefetch path: the next
          // chunk may have no rows in this stripe even though the other stripe
          // is still active.
          TEXPANDS(k_ub, 0.0f);
          TCVT(k_ub_half, k_ub, pto::RoundMode::CAST_NONE);
        }

        {
          GmShape2D g_shape(1, static_cast<int32_t>(next_valid));
          GmStride2D g_stride(1);
          GmTensor2D<float> g_global(G_handle + head * total_tokens + next_start,
                                     g_shape, g_stride);
          DynVecTile<float, 1, C, pto::PadValue::Zero> g_load(
              1, static_cast<int32_t>(next_valid));
          TASSIGN(g_load, G_UB);
          TLOAD(g_load, g_global);
          if (next_valid != C) {
            TFILLPAD_INPLACE(g_ub, g_load);
          }
        }
      }

      wait_flag_dev(2);
      {
        GmShape2D kv_shape(HalfC, D);
        GmStride2D kv_stride(D);
        GmTensor2D<half> kv_global(
            workspace_handle + ws_base + WS_KV + vid * HalfC * D,
            kv_shape, kv_stride);
        DynVecTile<half, HalfC, D, pto::PadValue::Zero> kv_load(HalfC, D);
        TASSIGN(kv_load, S_UB_HALF);
        TLOAD(kv_load, kv_global);
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(kv_ub, s_ub_half, pto::RoundMode::CAST_NONE);
      pipe_barrier(PIPE_ALL);
      // Finish S_{i+1} = exp(g_last) * S_i + k_i_tilde^T @ v_i_new.
      // Torch-like:
      //   s_ub = s_ub + kv_ub
      TADD(s_ub, s_ub, kv_ub);
      TCVT(s_ub_half, s_ub, pto::RoundMode::CAST_NONE);

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        {
          GmShape2D s_shape(HalfC, D);
          GmStride2D s_stride(D);
          GmTensor2D<half> s_global(
              workspace_handle + ws_base + WS_S + vid * HalfC * D,
              s_shape, s_stride);
          DynVecTile<half, HalfC, D> s_store(HalfC, D);
          TASSIGN(s_store, S_UB_HALF);
          TSTORE(s_global, s_store);
        }

        // Expose the post-chunk state so the next chunk (and debug/verification
        // outputs) can see S_{i+1}. Conceptually:
        //   S_handle[chunk_idx + 1, head] = S_{i+1}
        int64_t s_out_offset = ((chunk_offset + ci + 1) * H + head) * DD;
        {
          GmShape2D s_out_shape(HalfC, D);
          GmStride2D s_out_stride(D);
          GmTensor2D<half> s_out_global(
              S_handle + s_out_offset + vid * HalfC * D, s_out_shape,
              s_out_stride);
          DynVecTile<half, HalfC, D> s_out_store(HalfC, D);
          TASSIGN(s_out_store, S_UB_HALF);
          TSTORE(s_out_global, s_out_store);
        }
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
      }

      if (ci + 1 < static_cast<int32_t>(num_chunks)) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      }
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    int64_t fs_offset = (seq_idx * H + head) * DD;
    {
      GmShape2D fs_shape(HalfC, D);
      GmStride2D fs_stride(D);
      GmTensor2D<half> fs_global(FS_handle + fs_offset + vid * HalfC * D,
                                 fs_shape, fs_stride);
      DynVecTile<half, HalfC, D> fs_store(HalfC, D);
      TASSIGN(fs_store, S_UB_HALF);
      TSTORE(fs_global, fs_store);
    }
  }
#endif
}

#ifndef GDN_HG
#define GDN_HG GDN_H
#endif

extern "C" __global__ AICORE void launch_chunk_h(
    __gm__ uint8_t *K, __gm__ uint8_t *W, __gm__ uint8_t *U,
    __gm__ uint8_t *G,
    __gm__ uint8_t *S, __gm__ uint8_t *V, __gm__ uint8_t *FS,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens,
    uint64_t ffts_addr)
{
  chunk_h_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K),
      reinterpret_cast<__gm__ half *>(W),
      reinterpret_cast<__gm__ half *>(U),
      reinterpret_cast<__gm__ float *>(G),
      reinterpret_cast<__gm__ half *>(S),
      reinterpret_cast<__gm__ half *>(V),
      reinterpret_cast<__gm__ half *>(FS),
      reinterpret_cast<__gm__ half *>(workspace),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K, uint8_t *W, uint8_t *U, uint8_t *G,
    uint8_t *S, uint8_t *V, uint8_t *FS,
    uint8_t *workspace,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len, int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_h<<<block_dim, nullptr, stream>>>(
      K, W, U, G, S, V, FS, workspace, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
