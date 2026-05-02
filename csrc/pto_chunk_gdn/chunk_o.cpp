// ============================================================================
// chunk_o_kernel.cpp — Output computation for GatedDeltaNet (chunk-wise)
//
// Mathematical operation (per chunk of C tokens, per head h):
//
//   O = (QK_gated @ V) + exp(g) * (Q @ S)
//     = intra_chunk_attention + inter_chunk_state_contribution
//
// where:
//   Q, K, V ∈ ℝ^{C×D}    — query/key/value projections for this chunk
//   S ∈ ℝ^{D×D}           — accumulated hidden state entering this chunk
//   G ∈ ℝ^{C}             — cumulative gate values (pre-transposed [H,T])
//   Msk ∈ ℝ^{C×C}         — lower-triangular causal mask
//
// Cube phase (3 GEMMs per chunk):
//   1. QK   = Q @ K^T         — intra-chunk attention scores
//   2. QS   = Q @ S           — query applied to accumulated state
//   3. QKV  = QK_gated @ V    — gated attention applied to values
//
// Vec phase (two sub-blocks process upper/lower C/2 rows):
//   a. Load G → compute gating coefficients:
//        coeff[i,j] = exp(min(g[i] - g[j], 0)) * mask[i,j]
//   b. Apply gating to QK: QK_gated = QK * coeff
//   c. Scale QS by exp(g): QS_gated = QS * exp(g_row)
//   d. Combine: O = QS_gated + QKV
//   e. Store O to GM in BSND layout
//
// Cross-core sync protocol (Cube ↔ Vec via FFTS):
//   flag 0: Cube→Vec  — QK and QS results ready in workspace
//   flag 1: Vec→Cube  — QK_gated written back, Cube can proceed to GEMM 3
//   flag 2: Cube→Vec  — QKV result ready in workspace
//   flag 3: Vec→Cube  — Vec done with this chunk, Cube can reuse workspace
//
// NPU memory hierarchy used:
//   GM → L1 (Cube-accessible) → L0A/L0B (matrix engines) → L0C (accumulator)
//   GM → UB (Vec-accessible, on-chip SRAM)
//
// ── PTO / NPU Primer ──────────────────────────────────────────────────
// This kernel combines matrix multiplication (Cube) with element-wise gating
// (Vec) in a tightly coordinated 3-GEMM + gating pipeline per chunk.
//
// Execution timeline for one chunk:
//   Cube: GEMM1(Q@K^T) → GEMM2(Q@S) → store QK,QS → signal Vec ──────┐
//   Vec:  (meanwhile) load G, compute gating coefficients                │
//   Vec:  ←── wait for Cube signal ──── apply gating to QK → QK_gated  │
//   Vec:  store QK_gated → signal Cube ────────────────────────────────┐│
//   Cube: ←── wait for Vec signal ──── GEMM3(QK_gated@V) → store QKV ─┘│
//   Vec:  ←── wait for Cube signal ──── scale QS, combine O=QKV+QS_g   │
//   Vec:  store O → signal Cube "done" ─────────────────────────────────┘
//
// numpy pseudocode for the entire chunk computation:
//   QK = Q @ K.T                                          # GEMM 1
//   QS = Q @ S                                            # GEMM 2
//   coeff = exp(min(g_row - g_col, 0)) * mask             # gating (dynamic PTO)
//   (``static_baseline/run_chunk_o_static.py`` uses exp(g_row-g_col) without min.)
//   QK_gated = QK * coeff                                 # apply gating
//   QKV = QK_gated @ V                                    # GEMM 3
//   O = QKV + QS * np.exp(g_row).reshape(-1, 1)           # final output
//
// Key PTO APIs (with numpy/torch equivalents):
//   TLOAD(dst, gm)          — dst = gm_data      (DMA: GM→UB/L1, async)
//   TSTORE(gm, src)         — gm = src            (DMA: UB/L0C→GM, async)
//   TASSIGN(tile, addr)     — bind tile descriptor to buffer address
//   TCVT(dst, src, mode)    — type cast: dst = src.float() or .half()
//   TMOV(dst, src)          — copy: dst = src.clone()
//   TADD(d, a, b)           — d = a + b
//   TSUB(d, a, b)           — d = a - b
//   TMUL(d, a, b)           — d = a * b
//   TMINS(d, s, val)        — d = torch.clamp(s, max=val)
//   TEXP(d, s)              — d = torch.exp(s)
//   TROWEXPAND(2d, col)     — 2d[i,j] = col[i] (broadcast column→rows)
//   TCOLEXPAND(2d, row)     — 2d[i,j] = row[j] (broadcast row→columns)
//   TEXTRACT(l0, l1, r, c)  — copy L1 sub-tile → L0A/L0B (Cube input regs)
//   TRESHAPE(zn, nz)        — reinterpret L1 fractal layout (transpose, free)
//   TMATMUL(C, A, B)        — C = A @ B (Cube engine, fp16→fp32 accum)
//   set_flag / wait_flag    — synchronize pipes within same AI core
//   ffts_cross_core_sync    — signal across Cube↔Vec cores
//   wait_flag_dev(flag)     — wait for cross-core signal
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

// ── Compile-time configuration (overridable at build time via -D flags) ──
// GDN_H: number of attention heads (default 16)
// GDN_D: hidden dimension per head (default 128)
// GDN_C: chunk size in tokens (default 128)
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

// ── PTO type aliases (device-only, guarded for host pass safety) ────────────
// The bisheng compiler performs 3 passes: vec core, cube core (__CCE_AICORE__
// defined), and host (__CCE_AICORE__ NOT defined). Type aliases using PTO
// tile types must be guarded so the host pass never sees them.
#ifdef __CCE_AICORE__

// UbND = Unified Buffer tile, row-major (ND) layout, for Vec SIMD ops.
//   Like torch.empty((R, C), dtype=T) in fast on-chip SRAM (~256KB).
//   RV, CV = valid region (handles dynamic shapes, partial chunks).
//   PadValue::Zero = fill with 0 outside valid region during TLOAD.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad fill for TLOAD.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// UbDN = UB tile in column-major (DN) layout.
//   Needed as source for TROWEXPAND which requires column-format input.
//   TROWEXPAND takes a column vector and broadcasts it across all columns
//   of a destination ND tile: dst[i,j] = col[i] for all j.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;

// L1Mat = L1 cache tile in NZ fractal format — standard Cube GEMM input.
//   Data is loaded here from GM via TLOAD, then fed to L0A/L0B via TEXTRACT.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1MatZN = ZN fractal format — used for transposed GEMM operands.
//   TRESHAPE(l1_zn, l1_nz) converts NZ→ZN = logical matrix transpose (free, no data movement).
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                          RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

#endif  // __CCE_AICORE__

template <int32_t NumHeads, int32_t NumKeyHeads, int32_t HiddenSize,
          int32_t ChunkSize>
AICORE void chunk_o_kernel(
    __gm__ half *Q_handle, __gm__ half *K_handle, __gm__ half *V_handle,
    __gm__ half *S_handle, __gm__ float *G_handle,
    __gm__ float *Msk_handle,
    __gm__ half *workspace_qk_handle,
    __gm__ half *workspace_qs_qkv_handle,
    __gm__ half *workspace_qk_gated_handle,
    __gm__ half *O_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  // Half the chunk — each Vec sub-block handles C/2 rows independently.
  constexpr int32_t HalfChunk = ChunkSize / 2;
  // KTail / CTail: the number of valid elements in the last 128-element tile
  // when D or C isn't a multiple of 128. Used internally by PTO for partial tiles.
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);
  constexpr uint32_t CTail =
      (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  constexpr int32_t H = NumHeads;
  constexpr int32_t Hg = NumKeyHeads;
  static_assert(Hg > 0 && H % Hg == 0,
                "NumHeads must be divisible by NumKeyHeads");
  constexpr int32_t GROUP = H / Hg;
  constexpr int32_t BSND_V_STRIDE = H * HiddenSize;
  constexpr int32_t BSND_QK_STRIDE = Hg * HiddenSize;

  // Workspace sizes (in elements) shared between Cube and Vec via GM
  constexpr int32_t WsQKSize = ChunkSize * ChunkSize;
  constexpr int32_t WsQSSize = ChunkSize * HiddenSize;
  constexpr int32_t WsGatedSize = ChunkSize * ChunkSize;

  // ── UB memory map (byte addresses within Unified Buffer) ─────────────
  constexpr int32_t GUbAddr      = 0;
  constexpr int32_t MskUbAddr    = 512;
  constexpr int32_t QKUbAddr     = 33280;
  constexpr int32_t GvUbAddr     = 66048;
  constexpr int32_t CoeffUbAddr  = 66304;
  constexpr int32_t QKHalfUbAddr = 99072;
  constexpr int32_t QSHalfUbAddr = 115456;
  constexpr int32_t QSUbAddr     = 131840;
  constexpr int32_t OHalfUbAddr  = 164608;
  constexpr int32_t OUbAddr      = QKUbAddr;

  // Initialize the cross-core FFTS signaling base address for this AI core.
  set_ffts_base_addr(ffts_addr);
  // cid = which AI core am I? (0..block_num-1). Used to partition work items.
  auto cid = get_block_idx();
  // block_num = total number of AI cores running this kernel in parallel.
  auto block_num = get_block_num();
  // vid = Vec sub-block ID (0 or 1). Each Vec core has 2 sub-blocks that
  // process the upper (vid=0) and lower (vid=1) halves of C/2 rows.
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── L1 tiles for Cube GEMM operands ──────────────────────────────────
  // L1 holds matrices in NZ (col-major fractal) format for the matrix engine.
  // Each tile is assigned a fixed L1 byte address to avoid runtime allocation.
  //
  // ── L1 tile layout for Cube GEMMs ────────────────────────────────────
  // L1 cache (~1MB) is manually partitioned for the 3 GEMMs:
  //   q_l1   at 0:      Q [C×D]       — shared by GEMM 1 and GEMM 2
  //   k_l1   at 32768:  K [C×D]       — used in GEMM 1 (transposed via TRESHAPE)
  //   s_l1   at 65536:  S [D×D]       — accumulated state, used in GEMM 2
  //   qk_gated at 98304: QK_gated [C×C] — from Vec, used in GEMM 3
  //   v_l1   at 131072: V [C×D]       — values, used in GEMM 3
  L1Mat<half, ChunkSize, HiddenSize> q_l1;
  TASSIGN(q_l1, 0);
  L1Mat<half, ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 32768);
  TileAcc<float, ChunkSize, ChunkSize,
          ChunkSize, ChunkSize> qk_l0;
  TASSIGN(qk_l0, 0);
  L1Mat<half, HiddenSize, HiddenSize> s_l1;
  TASSIGN(s_l1, 65536);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> qs_l0;
  TASSIGN(qs_l0, 65536);
  L1Mat<half, ChunkSize, ChunkSize> qk_gated_l1;
  TASSIGN(qk_gated_l1, 98304);
  L1Mat<half, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize,
          ChunkSize, HiddenSize> qkv_l0;
  TASSIGN(qkv_l0, 0);

  // ── UB tiles for Vec element-wise operations ─────────────────────────
  // UB (Unified Buffer) is on-chip SRAM accessible by the Vec engine.
  // Tiles here are row-major (ND) for standard element-wise ops.
  //
  // ── UB tile layout for Vec element-wise ops ──────────────────────────
  // Each Vec sub-block (vid=0 or vid=1) processes C/2 rows of the C×C or C×D
  // matrices. The UB layout (byte addresses) is designed so all needed tiles
  // fit simultaneously in the ~256KB UB without overlapping:
  //   g_ub:       gate values [1, C] float            @ 0
  //   msk_ub:     causal mask [C/2, C] float          @ 512     (loaded once, reused)
  //   qk_ub:      QK scores in float [C/2, C]         @ 33280   (after cast from half)
  //   g_v_ub:     this sub-block's gate slice [1, C/2] @ 66048
  //   coeff_ub:   gating coefficients [C/2, C] float  @ 66304
  //   qk_ub_half: QK in half [C/2, C]                @ 99072
  //   qs_ub_half: QS in half [C/2, D]                @ 115456
  //   qs_ub:      QS in float [C/2, D]               @ 131840
  //   o_ub_half:  output O in half [C/2, D]           @ 164608
  //   o_ub:       output O in float [C/2, D]          @ QKUbAddr (reuses qk_ub space)
  UbND<float, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<float, HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  UbND<float, HalfChunk, ChunkSize> qk_ub;
  TASSIGN(qk_ub, QKUbAddr);
  UbND<float, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  UbND<float, HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  UbND<half, HalfChunk, ChunkSize, HalfChunk, ChunkSize, PadValue::Zero> qk_ub_half;
  TASSIGN(qk_ub_half, QKHalfUbAddr);
  UbND<half, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero> qs_ub_half;
  TASSIGN(qs_ub_half, QSHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> qs_ub;
  TASSIGN(qs_ub, QSUbAddr);
  UbND<half, HalfChunk, HiddenSize, HalfChunk, HiddenSize, PadValue::Zero> o_ub_half;
  TASSIGN(o_ub_half, OHalfUbAddr);
  UbND<float, HalfChunk, HiddenSize> o_ub;
  TASSIGN(o_ub, OUbAddr);

  // Total work items = (batches * chunks_per_sequence * heads).
  // Each AI core (cid) picks every block_num-th work item (round-robin).
  int64_t total_work = 0;
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    total_work = num_seqs * chunks_per_seq * NumHeads;
  }

// =====================================================================
// CUBE CORE — Three GEMMs per chunk: QK, QS, QKV
// Each AI core processes a different (chunk, head) pair. The Cube engine
// performs the heavy matmuls, then writes results to GM workspace for
// the Vec engine to apply gating and produce the final output.
// =====================================================================
#if defined(__DAV_C220_CUBE__)
  if (cu_seqlens == nullptr) {
    // ── Fixed-length sequence path ──────────────────────────────────────
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t global_chunk_base = 0;
    bool first_cube_iter = true;

    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      // Wait for Vec to finish with previous chunk's workspace (flag 3)
      if (!first_cube_iter) wait_flag_dev(3);
      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int32_t head_g = head_idx / GROUP;
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) local_rows = 0;
      if (local_rows > HalfChunk) local_rows = HalfChunk;

      int64_t qk_off =
          (chunk_token_start * static_cast<int64_t>(Hg) +
           static_cast<int64_t>(head_g)) *
          static_cast<int64_t>(HiddenSize);
      int64_t v_off =
          (chunk_token_start * static_cast<int64_t>(H) +
           static_cast<int64_t>(head_idx)) *
          static_cast<int64_t>(HiddenSize);

      int64_t chunk_global_idx = seq_idx * chunks_per_seq + ci;
      int64_t s_offset =
          (chunk_global_idx * NumHeads + head_idx) *
          static_cast<int64_t>(HiddenSize) *
          static_cast<int64_t>(HiddenSize);

      // ── Load Q [valid_rows × D] from GM → L1 ────────────────────────
      // GlobalTensor describes the GM layout with BSND strides.
      // TLOAD performs DMA (MTE2 pipe). TFILLPAD zero-pads tail rows so
      // downstream GEMMs see a clean C×D matrix.
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QK_STRIDE, 1>> _gm(Q_handle + qk_off, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }
      // ── Load K [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 32768);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QK_STRIDE, 1>> _gm(K_handle + qk_off, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM 1: QK = Q @ K^T  (intra-chunk attention scores) ────────
      // ── GEMM 1: QK = Q @ K^T ─────────────────────────────────────────
      // numpy: QK = Q @ K.T  →  [C×D] @ [D×C] = [C×C]
      //
      // How transpose works on NPU:
      //   K is loaded into L1 in NZ (col-major fractal) format.
      //   TRESHAPE(l1_zn, k_l1) reinterprets it as ZN (row-major fractal) = K^T.
      //   This is a ZERO-COST operation — no data movement, just metadata change.
      //   TEXTRACT then loads the transposed view into L0B.
      //
      // Cube GEMM pipeline:
      //   TEXTRACT(l0a, q_l1, 0, 0)  — Q → L0A (left operand)
      //   TEXTRACT(l0b, k_zn, 0, 0)  — K^T → L0B (right operand)
      //   TMATMUL(qk_l0, l0a, l0b)   — QK = L0A × L0B → L0C accumulator
      //
      // transpose_B: TRESHAPE converts k_l1 from NZ → ZN fractal layout,
      // effectively transposing K before TEXTRACT loads it into L0B.
      {
        TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
        TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, q_l1, 0, 0);
        L1MatZN<half, HiddenSize, ChunkSize> _bzn; TRESHAPE(_bzn, k_l1); TEXTRACT(_l0b, _bzn, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qk_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Load S [D × D] from GM → L1  (accumulated hidden state) ─────
      {
        L1Mat<half, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(HiddenSize, HiddenSize);
        TASSIGN(_l1, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = HiddenSize; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(S_handle + s_offset, _gs);
        TLOAD(_l1, _gm);
      }

      // ── GEMM 2: QS = Q @ S  (query applied to accumulated state) ────
      {
        TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<half, HiddenSize, HiddenSize, HiddenSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, q_l1, 0, 0);
        TEXTRACT(_l0b, s_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qs_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store QK [C × C] from L0C → GM workspace (fp32→fp16 cast) ───
      // TSTORE on TileAcc triggers MTE3 DMA with implicit type conversion.
      {
        TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, ChunkSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_handle +
                static_cast<int64_t>(cid) * WsQKSize, _gs);
        TSTORE(_gm, _l0);
      }

      // ── Store QS [C × D] from L0C → GM workspace ────────────────────
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, HiddenSize);
        TASSIGN(_l0, 65536);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle +
                static_cast<int64_t>(cid) * WsQSSize, _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec: QK and QS are ready (flag 0, Cube→Vec)
      // ── Cross-core sync protocol ──────────────────────────────────────
      // Cube and Vec are SEPARATE physical cores. They exchange data through GM
      // and coordinate via FFTS flags. Think of it as two processes communicating
      // through shared memory with semaphores.
      //
      // ffts_cross_core_sync(PIPE_FIX, config):
      //   config = 1 | (mode << 4) | (flag_id << 8)
      //   mode=2: broadcast signal to all cores in this block
      //   flag_id: identifies which signal (0, 1, 2, 3)
      //
      // Protocol for this kernel:
      //   flag 0: Cube→Vec "QK and QS are ready in workspace"
      //   flag 1: Vec→Cube "QK_gated is ready for GEMM 3"
      //   flag 2: Cube→Vec "QKV (GEMM 3 result) is ready"
      //   flag 3: Vec→Cube "I'm done with this chunk, you can reuse workspace"
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

      // Wait for Vec to write QK_gated back (flag 1, Vec→Cube)
      wait_flag_dev(1);

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // ── Load QK_gated [C × C] from GM workspace → L1 ────────────────
      {
        L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
        TASSIGN(_l1, 98304);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_gated_handle +
                static_cast<int64_t>(cid) * WsGatedSize, _gs);
        TLOAD(_l1, _gm);
      }
      // ── Load V [valid_rows × D] from GM → L1 ────────────────────────
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 131072);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_V_STRIDE, 1>> _gm(V_handle + v_off, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM 3: QKV = QK_gated @ V  (gated attention → values) ──────
      {
        TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
        TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
        TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
        TEXTRACT(_l0a, qk_gated_l1, 0, 0);
        TEXTRACT(_l0b, v_l1, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(qkv_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store QKV [C × D] from L0C → GM workspace ───────────────────
      // ── Workspace buffer reuse ────────────────────────────────────────
      // workspace_qs_qkv_handle is shared between QS (GEMM 2 output) and QKV
      // (GEMM 3 output). This is safe because:
      //   1. Vec reads QS BEFORE Cube writes QKV to the same buffer
      //   2. The cross-core flags ensure proper ordering:
      //      - flag 0: QS ready (Vec reads QS)
      //      - flag 1: QK_gated ready (Vec done reading QS, Cube can write QKV)
      //      - flag 2: QKV ready (Vec reads QKV from same buffer)
      {
        TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, HiddenSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle +
                static_cast<int64_t>(cid) * WsQSSize, _gs);
        TSTORE(_gm, _l0);
      }

      // Signal Vec: QKV is ready (flag 2, Cube→Vec)
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
      first_cube_iter = false;
    }
  } else {
    // ── Variable-length sequence path (cu_seqlens != nullptr) ──────────
    int64_t gi = 0;
    int64_t chunk_global_idx = 0;
    bool first_cube_iter_v = true;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            if (!first_cube_iter_v) wait_flag_dev(3);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int32_t head_g = head_idx / GROUP;

            int64_t qk_off =
                (chunk_token_start * static_cast<int64_t>(Hg) +
                 static_cast<int64_t>(head_g)) *
                static_cast<int64_t>(HiddenSize);
            int64_t v_off =
                (chunk_token_start * static_cast<int64_t>(H) +
                 static_cast<int64_t>(head_idx)) *
                static_cast<int64_t>(HiddenSize);
            int64_t s_offset =
                (chunk_global_idx * NumHeads + head_idx) *
                static_cast<int64_t>(HiddenSize) *
                static_cast<int64_t>(HiddenSize);

            // Load Q
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QK_STRIDE, 1>> _gm(Q_handle + qk_off, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }
            // Load K
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 32768);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_QK_STRIDE, 1>> _gm(K_handle + qk_off, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // GEMM 1: QK = Q @ K^T (transpose_B via TRESHAPE NZ→ZN)
            {
              TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
              TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
              TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, q_l1, 0, 0);
              L1MatZN<half, HiddenSize, ChunkSize> _bzn; TRESHAPE(_bzn, k_l1); TEXTRACT(_l0b, _bzn, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qk_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Load S
            {
              L1Mat<half, HiddenSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(HiddenSize, HiddenSize);
              TASSIGN(_l1, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = HiddenSize; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(S_handle + s_offset, _gs);
              TLOAD(_l1, _gm);
            }

            // GEMM 2: QS = Q @ S
            {
              TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
              TileRight<half, HiddenSize, HiddenSize, HiddenSize, HiddenSize> _l0b;
              TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, q_l1, 0, 0);
              TEXTRACT(_l0b, s_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qs_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            // Store QK → workspace
            {
              TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, ChunkSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_qk_handle +
                      static_cast<int64_t>(cid) * WsQKSize, _gs);
              TSTORE(_gm, _l0);
            }

            // Store QS → workspace
            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, HiddenSize);
              TASSIGN(_l0, 65536);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
                  workspace_qs_qkv_handle +
                      static_cast<int64_t>(cid) * WsQSSize, _gs);
              TSTORE(_gm, _l0);
            }

            // Cube→Vec: QK & QS ready (flag 0)
            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (0 << 8));

            // Wait Vec→Cube: QK_gated ready (flag 1)
            wait_flag_dev(1);

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

            // Load QK_gated
            {
              L1Mat<half, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l1(ChunkSize, ChunkSize);
              TASSIGN(_l1, 98304);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                  workspace_qk_gated_handle +
                      static_cast<int64_t>(cid) * WsGatedSize, _gs);
              TLOAD(_l1, _gm);
            }
            // Load V
            {
              L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
              TASSIGN(_l1, 131072);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_V_STRIDE, 1>> _gm(V_handle + v_off, _gs);
              TLOAD(_l1, _gm);
              if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
            }

            // GEMM 3: QKV = QK_gated @ V
            {
              TileLeft<half, ChunkSize, ChunkSize, ChunkSize, ChunkSize> _l0a;
              TileRight<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0b;
              TASSIGN(_l0a, 0x0); TASSIGN(_l0b, 0x0);
              auto _we = EVENT_ID1;
              set_flag(PIPE_MTE2, PIPE_MTE1, _we); wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
              set_flag(PIPE_M, PIPE_MTE1, _we); wait_flag(PIPE_M, PIPE_MTE1, _we);
              TEXTRACT(_l0a, qk_gated_l1, 0, 0);
              TEXTRACT(_l0b, v_l1, 0, 0);
              set_flag(PIPE_MTE1, PIPE_M, _we); wait_flag(PIPE_MTE1, PIPE_M, _we);
              TMATMUL(qkv_l0, _l0a, _l0b);
              set_flag(PIPE_MTE1, PIPE_MTE2, _we); wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
              set_flag(PIPE_M, PIPE_FIX, _we); wait_flag(PIPE_M, PIPE_FIX, _we);
            }

            {
              TileAcc<float, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, HiddenSize);
              TASSIGN(_l0, 0);
              Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
              _gs.shape[3] = ChunkSize; _gs.shape[4] = HiddenSize;
              GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
                  workspace_qs_qkv_handle +
                      static_cast<int64_t>(cid) * WsQSSize, _gs);
              TSTORE(_gm, _l0);
            }

            ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (2 << 8));
            first_cube_iter_v = false;
          }
          gi++;
        }
        chunk_global_idx++;
      }
    }
  }
#endif

// =====================================================================
// VEC CORE — Gating, element-wise ops, output assembly
// Two Vec sub-blocks (vid=0,1) process upper/lower C/2 rows in parallel.
// Each sub-block independently:
//   1. Computes gating coefficients from G and the causal mask
//   2. Applies gating to the Cube's QK result → QK_gated
//   3. Scales the Cube's QS result by exp(g)
//   4. Combines QKV + scaled QS → final output O
// =====================================================================
#if defined(__DAV_C220_VEC__)
  // Vec engine initialization: set_mask_norm selects "normal" masking mode,
  // and set_vector_mask(-1, -1) enables ALL SIMD lanes (no masking).
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Load causal mask once (reused across all chunks) ─────────────────
  // ── Causal mask (loaded once, reused) ─────────────────────────────────
  // The causal mask is a C×C lower-triangular matrix of 0s and 1s:
  //   mask[i,j] = 1 if i >= j else 0
  // Each sub-block loads its C/2 rows. Applied via TMUL to zero out
  // non-causal (future) attention scores.
  //
  // Each sub-block (vid=0,1) loads its C/2 rows of the C×C lower-tri mask.
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
    GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        Msk_handle +
            static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
    TASSIGN(_ld, MskUbAddr);
    TLOAD(_ld, _gm);
  }
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  if (cu_seqlens == nullptr) {
    // ── Fixed-length sequence path ──────────────────────────────────────
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;

    for (int64_t work_idx = static_cast<int64_t>(cid);
         work_idx < total_work;
         work_idx += static_cast<int64_t>(block_num)) {
      int32_t head_idx = static_cast<int32_t>(work_idx % NumHeads);
      int64_t chunk_head_idx = work_idx / NumHeads;
      int64_t seq_idx = chunk_head_idx / chunks_per_seq;
      int64_t ci = chunk_head_idx % chunks_per_seq;

      int64_t bos = seq_idx * seq_len;
      int64_t slen = seq_len;
      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      int64_t chunk_token_start = bos + chunk_start;
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      int32_t local_rows = valid_rows - row_offset;
      if (local_rows < 0) local_rows = 0;
      if (local_rows > HalfChunk) local_rows = HalfChunk;

      if (local_rows > 0) {
        // ── Load G [1 × valid_rows] — gate values for this chunk ────────
        // G is pre-transposed to [H, total_tokens], contiguous per head.
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
          GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
              G_handle + static_cast<int64_t>(head_idx) * total_tokens
                       + chunk_token_start, _gs);
          UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
          TASSIGN(_ld, GUbAddr);
          TLOAD(_ld, _gm);
          if (valid_rows != ChunkSize) {
            UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
            TASSIGN(_pd, GUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // ── Compute gating coefficients ──────────────────────────────────
        // ── Gating coefficient computation (numpy pseudocode) ─────────────
        // For this sub-block's rows (vid=0: rows 0..C/2-1, vid=1: rows C/2..C-1):
        //
        //   g_row = g[my_start:my_start+C/2]    # my gates (shape [C/2])
        //   g_col = g[0:C]                       # full chunk gates (shape [C])
        //
        //   # Broadcast to 2D matrices:
        //   g_r_2d = g_row[:, None] * np.ones((1, C))    # TROWEXPAND: [C/2, C]
        //   g_c_2d = np.ones((C/2, 1)) * g_col[None, :]  # TCOLEXPAND: [C/2, C]
        //   coeff = exp(min(g_r_2d - g_c_2d, 0)) * mask
        //
        //   # Also compute exp(g_row) for QS scaling:
        //   exp_g_row = np.exp(g_row)                     # TEXP
        UbND<float, 1, HalfChunk> g_ub_temp_0;
        TASSIGN(g_ub_temp_0,
                GUbAddr + static_cast<int32_t>(vid) * HalfChunk *
                              static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp_0);

        // Broadcast g_row into [C/2 × C] and g_col into [C/2 × C]
        UbND<float, HalfChunk, ChunkSize> g_r_2d;
        TASSIGN(g_r_2d, QSUbAddr);
        UbDN<float, HalfChunk, 1> g_v_col;
        TASSIGN(g_v_col, GvUbAddr);
        TROWEXPAND(g_r_2d, g_v_col);       // g_r_2d[i,j] = g_row[i]
        TCOLEXPAND(coeff_ub, g_ub);        // coeff[i,j] = g_col[j]
        TSUB(coeff_ub, g_r_2d, coeff_ub);  // d = g_row - g_col
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);
        pipe_barrier(PIPE_V);
        TMUL(coeff_ub, coeff_ub, msk_ub);
        pipe_barrier(PIPE_V);
        TEXP(g_v_ub, g_v_ub);              // exp(g_row) for QS scaling
      }

      // ── Wait for Cube→Vec flag 0: QK & QS ready ─────────────────────
      wait_flag_dev(0);
      if (local_rows == 0) {
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
        wait_flag_dev(2);
        ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
        continue;
      }

      // ── Load QK [C/2 × C] from workspace → UB ───────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_handle +
                static_cast<int64_t>(cid) * WsQKSize +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(local_rows, ChunkSize);
        TASSIGN(_ld, QKHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qk_ub_half, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

      // ── Load QS [C/2 × D] from workspace → UB ───────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle +
                static_cast<int64_t>(cid) * WsQSSize +
                static_cast<int64_t>(vid) * HalfChunk * HiddenSize, _gs);
        UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(local_rows, HiddenSize);
        TASSIGN(_ld, QSHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(qs_ub_half, _ld);
        }
      }

      // ── Apply gating: QK_gated = QK * exp(d*mask)*mask
      TMUL(qk_ub, qk_ub, coeff_ub);
      TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);

      // ── Store QK_gated [C/2 × C] → workspace for Cube's GEMM 3 ─────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_qk_gated_handle +
                static_cast<int64_t>(cid) * WsGatedSize +
                static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
        UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_rows, ChunkSize);
        TASSIGN(_st, QKHalfUbAddr);
        TSTORE(_gm, _st);
      }
      // Vec→Cube: QK_gated ready (flag 1)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

      // ── Scale QS by exp(g): QS_gated = QS * exp(g_row) ──────────────
      // ── Scale QS by exp(g): inter-chunk state contribution ────────────
      // numpy: QS_scaled = QS * np.exp(g_row)[:, None]   (broadcast across D columns)
      // TROWEXPAND broadcasts the scalar exp(g[i]) for each row i across all D columns,
      // then TMUL applies it element-wise. This gates how much the accumulated state
      // contributes to each token's output.
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);
      UbND<float, HalfChunk, HiddenSize> g_exp_2d;
      TASSIGN(g_exp_2d, CoeffUbAddr);
      UbDN<float, HalfChunk, 1> g_v_col2;
      TASSIGN(g_v_col2, GvUbAddr);
      TROWEXPAND(g_exp_2d, g_v_col2);    // broadcast exp(g_row) across columns
      pipe_barrier(PIPE_V);
      TMUL(qs_ub, qs_ub, g_exp_2d);      // QS_gated = QS * exp(g_row)

      // ── Wait for Cube→Vec flag 2: QKV ready ─────────────────────────
      wait_flag_dev(2);

      // ── Load QKV [C/2 × D] from workspace → UB ──────────────────────
      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
            workspace_qs_qkv_handle +
                static_cast<int64_t>(cid) * WsQSSize +
                static_cast<int64_t>(vid) * HalfChunk * HiddenSize, _gs);
        UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(local_rows, HiddenSize);
        TASSIGN(_ld, OHalfUbAddr);
        TLOAD(_ld, _gm);
        if (local_rows != HalfChunk) {
          TFILLPAD_INPLACE(o_ub_half, _ld);
        }
      }

      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── Combine: O = QS_gated + QKV ─────────────────────────────────
      // ── Final output: O = QKV + QS_scaled ─────────────────────────────
      // numpy: O = (QK_gated @ V) + (Q @ S) * exp(g)[:, None]
      //       = intra_chunk_attention + inter_chunk_state_contribution
      // TCVT half→float for QKV, then TADD, then TCVT float→half for output.
      TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);
      TADD(o_ub, qs_ub, o_ub);
      TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);

      // ── Store O [C/2 × D] → GM in BSND layout ───────────────────────
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      int64_t o_offset =
          (chunk_token_start * static_cast<int64_t>(H) +
           static_cast<int64_t>(head_idx)) *
              static_cast<int64_t>(HiddenSize) +
          static_cast<int64_t>(vid) * HalfChunk *
              static_cast<int64_t>(BSND_V_STRIDE);

      {
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = local_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_V_STRIDE, 1>> _gm(
            O_handle + o_offset, _gs);
        UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC> _st(local_rows, HiddenSize);
        TASSIGN(_st, OHalfUbAddr);
        TSTORE(_gm, _st);
      }

      // Vec→Cube: done with this chunk (flag 3)
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
    }
  } else {
    // ── Variable-length sequence path (cu_seqlens != nullptr) ──────────
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
            int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
            int32_t local_rows = valid_rows - row_offset;
            if (local_rows < 0) local_rows = 0;
            if (local_rows > HalfChunk) local_rows = HalfChunk;

            if (local_rows > 0) {
              // Load G
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
                GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
                    G_handle + static_cast<int64_t>(head_idx) * total_tokens
                             + chunk_token_start, _gs);
                UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
                TASSIGN(_ld, GUbAddr);
                TLOAD(_ld, _gm);
                if (valid_rows != ChunkSize) {
                  UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
                  TASSIGN(_pd, GUbAddr);
                  TFILLPAD_INPLACE(_pd, _ld);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

              // Compute gating coefficients (same math as fixed-length path — see detailed pseudocode above)
              UbND<float, 1, HalfChunk> g_ub_temp_v;
              TASSIGN(g_ub_temp_v,
                      GUbAddr +
                          static_cast<int32_t>(vid) * HalfChunk *
                              static_cast<int32_t>(sizeof(float)));
              TMOV(g_v_ub, g_ub_temp_v);

              UbND<float, HalfChunk, ChunkSize> g_r_2d_v;
              TASSIGN(g_r_2d_v, QSUbAddr);
              UbDN<float, HalfChunk, 1> g_v_col_v;
              TASSIGN(g_v_col_v, GvUbAddr);
              TROWEXPAND(g_r_2d_v, g_v_col_v);
              TCOLEXPAND(coeff_ub, g_ub);
              TSUB(coeff_ub, g_r_2d_v, coeff_ub);  // d = g_row - g_col
              pipe_barrier(PIPE_V);
              TMINS(coeff_ub, coeff_ub, 0.0f);
              pipe_barrier(PIPE_V);
              TEXP(coeff_ub, coeff_ub);
              pipe_barrier(PIPE_V);
              TMUL(coeff_ub, coeff_ub, msk_ub);
              pipe_barrier(PIPE_V);
              TEXP(g_v_ub, g_v_ub);
            }

            wait_flag_dev(0);
            if (local_rows == 0) {
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));
              wait_flag_dev(2);
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
            } else {
              // Load QK from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows; _gs.shape[4] = ChunkSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                    workspace_qk_handle +
                        static_cast<int64_t>(cid) * WsQKSize +
                        static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
                UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(local_rows, ChunkSize);
                TASSIGN(_ld, QKHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(qk_ub_half, _ld);
                }
              }

              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              TCVT(qk_ub, qk_ub_half, pto::RoundMode::CAST_NONE);

              set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

              // Load QS from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows; _gs.shape[4] = HiddenSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
                    workspace_qs_qkv_handle +
                        static_cast<int64_t>(cid) * WsQSSize +
                        static_cast<int64_t>(vid) * HalfChunk * HiddenSize, _gs);
                UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(local_rows, HiddenSize);
                TASSIGN(_ld, QSHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(qs_ub_half, _ld);
                }
              }

              TMUL(qk_ub, qk_ub, coeff_ub);
              TCVT(qk_ub_half, qk_ub, pto::RoundMode::CAST_NONE);  // float→half for GM store

              // Store QK_gated → workspace
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows; _gs.shape[4] = ChunkSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
                    workspace_qk_gated_handle +
                        static_cast<int64_t>(cid) * WsGatedSize +
                        static_cast<int64_t>(vid) * HalfChunk * ChunkSize, _gs);
                UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_rows, ChunkSize);
                TASSIGN(_st, QKHalfUbAddr);
                TSTORE(_gm, _st);
              }
              // Vec→Cube: QK_gated ready (flag 1)
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (1 << 8));

              // Scale QS by exp(g): QS_scaled = QS * exp(g_row)[:, None]
              // (same inter-chunk state scaling as fixed-length path)
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              TCVT(qs_ub, qs_ub_half, pto::RoundMode::CAST_NONE);  // half→float for Vec math

              UbND<float, HalfChunk, HiddenSize> g_exp_2d_v;
              TASSIGN(g_exp_2d_v, CoeffUbAddr);
              UbDN<float, HalfChunk, 1> g_v_col2_v;
              TASSIGN(g_v_col2_v, GvUbAddr);
              TROWEXPAND(g_exp_2d_v, g_v_col2_v);
              pipe_barrier(PIPE_V);
              TMUL(qs_ub, qs_ub, g_exp_2d_v);

              wait_flag_dev(2);

              // Load QKV from workspace
              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows; _gs.shape[4] = HiddenSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, HiddenSize, 1>> _gm(
                    workspace_qs_qkv_handle +
                        static_cast<int64_t>(cid) * WsQSSize +
                        static_cast<int64_t>(vid) * HalfChunk * HiddenSize, _gs);
                UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(local_rows, HiddenSize);
                TASSIGN(_ld, OHalfUbAddr);
                TLOAD(_ld, _gm);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(o_ub_half, _ld);
                }
              }

              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

              // O = QS_gated + QKV  (final output: intra-chunk attention + inter-chunk state)
              TCVT(o_ub, o_ub_half, pto::RoundMode::CAST_NONE);  // half→float
              TADD(o_ub, qs_ub, o_ub);                            // O = QS_scaled + QKV
              TCVT(o_ub_half, o_ub, pto::RoundMode::CAST_NONE);  // float→half for GM store

              // Store O → GM
              set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
              wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

              int64_t o_offset =
                  (chunk_token_start * static_cast<int64_t>(H) +
                   static_cast<int64_t>(head_idx)) *
                      static_cast<int64_t>(HiddenSize) +
                  static_cast<int64_t>(vid) * HalfChunk *
                      static_cast<int64_t>(BSND_V_STRIDE);

              {
                Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
                _gs.shape[3] = local_rows; _gs.shape[4] = HiddenSize;
                GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, BSND_V_STRIDE, 1>> _gm(
                    O_handle + o_offset, _gs);
                UbND<half, HalfChunk, HiddenSize, DYNAMIC, DYNAMIC> _st(local_rows, HiddenSize);
                TASSIGN(_st, OHalfUbAddr);
                TSTORE(_gm, _st);
              }

              // Vec→Cube: done with this chunk (flag 3)
              ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));
            }
          }
          gi++;
        }
      }
    }
  }
#endif
}

// ── Device kernel entry point ─────────────────────────────────────────
// extern "C" __global__ AICORE: NPU kernel function.
// Runs on each AI core independently. Args are uint8_t* (type-erased)
// because the NPU launch ABI passes all pointers as raw bytes; we
// reinterpret_cast them to the correct types before calling the template.
extern "C" __global__ AICORE void launch_chunk_o(
    __gm__ uint8_t *Q_handle, __gm__ uint8_t *K_handle,
    __gm__ uint8_t *V_handle, __gm__ uint8_t *S_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_qk, __gm__ uint8_t *workspace_qs_qkv,
    __gm__ uint8_t *workspace_qk_gated,
    __gm__ uint8_t *O_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  chunk_o_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(Q_handle),
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(V_handle),
      reinterpret_cast<__gm__ half *>(S_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_qk),
      reinterpret_cast<__gm__ half *>(workspace_qs_qkv),
      reinterpret_cast<__gm__ half *>(workspace_qk_gated),
      reinterpret_cast<__gm__ half *>(O_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

// ── Host launcher (called from Python ctypes) ─────────────────────────
// Launches kernel on block_dim AI cores via NPU stream.
// rtGetC2cCtrlAddr obtains the FFTS (cross-core sync) control address that
// the kernel needs for Cube↔Vec flag signaling.
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *q, uint8_t *k, uint8_t *v, uint8_t *s, uint8_t *g_sum,
    uint8_t *mask,
    uint8_t *workspace_qk, uint8_t *workspace_qs_qkv,
    uint8_t *workspace_qk_gated,
    uint8_t *o,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_chunk_o<<<block_dim, nullptr, stream>>>(
      q, k, v, s, g_sum, mask,
      workspace_qk, workspace_qs_qkv, workspace_qk_gated,
      o,
      cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
