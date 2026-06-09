// W4A4 MoE HYBRID mega kernel — PTO vec stages + AscendC MatmulImpl<int4b_t> cube stages,
// ONE fused __global__ MIX kernel, raw-FFTS synced.
//
// This is the production mega_kernel.cpp hybridized: the two PTO int4 cube matmuls
// (Stage 2 gate_up, Stage 4 down) are replaced by AscendC `MatmulImpl<int4b_t>` (the
// validated grouped single-Init path, ~5x faster at decode than the raw-PTO mad_s4).
// The PTO vec stages (Stage 1 quant+scatter, Stage 3 swiglu+quant, Stage 5 combine) and
// the SAFESYNC B0..B5 barrier schedule are UNCHANGED.
//
// Coexistence recipe (proven by hybrid_proof.cpp + the pass-split probe):
//   - `using namespace pto`     ONLY in the __DAV_C220_VEC__ compile pass.
//   - `using namespace AscendC` + `using namespace matmul` ONLY in __DAV_C220_CUBE__.
//     (file-global `using namespace pto` would make TPipe / DYNAMIC ambiguous against
//      AscendC; the bisheng compiler runs two passes, one per target, so a pass-guarded
//      using-directive cleanly separates the two worlds.)
//   - FFTS sync constants/helpers renamed Hx*/HX_* to dodge AscendC's global
//     SYNC_AIV_FLAG / SyncAllImpl.
//   - int4_cvt.hpp (does its own unconditional `using namespace pto`) included VEC-only.
//
// STAGE A (this build, default): NO in-kernel Stage 0. The block-diag Hadamard is applied
// by Python before the kernel (the existing pyhadamard mode), so the cube branch does ONLY
// the two int4 matmuls. Build WITHOUT -DMEGA_CUBE_HADAMARD; the kernel bakes in
// MEGA_HADAMARD_KERNEL_SKIP semantics for the Stage 1 quant scale (kInvSqrtN=1).
//
// Weights in HBM are FRACTAL_NZ int4 (the make_nz_weight_int4 layout from
// mm_impl_int4_grouped / hybrid_proof), NOT the RowMajor packing the PTO cube path used.
// Two TCubeTiling structs are computed on host (gate_up: K=H_DIM,N=N_GU; down:
// K=I_DIM,N=H_DIM) and passed in.
//
// NPU 4 only.

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include "runtime/rt_ffts.h"

#if defined(__DAV_C220_CUBE__)
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
#endif
#if defined(__DAV_C220_VEC__)
#include "int4_cvt.hpp"  // does `using namespace pto`
using namespace pto;
#endif

// ===== Shape constants (Qwen3.6-35B-A3B via I_DIM_OVERRIDE; default I=512) =====
constexpr uint32_t H_DIM = 2048;
#ifdef I_DIM_OVERRIDE
constexpr uint32_t I_DIM = I_DIM_OVERRIDE;
#define mega_kernel_hybrid       mega_kernel_hybrid_q36
#define call_mega_kernel_hybrid  call_mega_kernel_hybrid_qwen36
#else
constexpr uint32_t I_DIM = 512;
#endif
constexpr uint32_t N_GU  = 2 * I_DIM;
constexpr uint32_t M_TILE_CUBE = 32;   // M-pad granularity (matches mega_kernel.cpp workspaces)

// HADAMARD_N selects the Stage 1 quant scale normalization. Production uses 64. With the
// pyhadamard / cube-Hadamard (kernel-skip) mode the rotation is pre-applied, so the scale
// uses kInvSqrtN=1, but HADAMARD_N is still kept for build parity.
#ifndef MEGA_HADAMARD_N
#define MEGA_HADAMARD_N 64
#endif
constexpr uint32_t HADAMARD_N = MEGA_HADAMARD_N;

// ISOLATION GATE: -DMEGA_STOP_AFTER_N=k makes the kernel execute only stages 1..k
// (vec) and the cube stages they need, replacing the later stage BODIES with no-ops
// while keeping every HxSyncAllImpl barrier so the AIV/AIC FFTS rendezvous stays
// balanced. Used to pin which stage first faults under the vLLM profile_run. Default
// (undefined) = full kernel. Stage numbering: 1=stage1 quant+scatter (vec),
// 2=gate_up matmul (cube), 3=swiglu+quant (vec), 4=down matmul (cube), 5=combine (vec).
#ifndef MEGA_STOP_AFTER_N
#define MEGA_STOP_AFTER_N 5
#endif

#define UB_ALIGN(x) (((unsigned)(x) + 255u) & ~255u)

// ---------------- raw FFTS sync (HX_-renamed to dodge AscendC's global SYNC_*) ----------------
constexpr uint32_t HX_SYNC_AIV_ONLY_ALL     = 0x6;
constexpr uint32_t HX_SYNC_AIV_FLAG         = 0x4;
constexpr uint32_t HX_SYNC_AIC_FLAG         = 0x5;
constexpr uint32_t HX_SYNC_AIC_AIV_FLAG     = 3;
constexpr uint32_t HX_SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint32_t HX_SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t HxGetffstMsg(uint16_t mode, uint16_t flagId) {
  return (0x1 + ((mode & 0x3) << HX_SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << HX_SYNC_FLAG_SHIFT_VALUE));
}
template <bool isAIVOnly = true>
AICORE inline void HxSyncAllImpl() {
  pipe_barrier(PIPE_ALL);
  if constexpr (isAIVOnly) {
    ffts_cross_core_sync(PIPE_MTE3, HxGetffstMsg(0x0, HX_SYNC_AIV_ONLY_ALL));
    wait_flag_dev(HX_SYNC_AIV_ONLY_ALL);
    return;
  }
#if defined(__DAV_C220_CUBE__)
  wait_flag_dev(HX_SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, HxGetffstMsg(0x0, HX_SYNC_AIC_FLAG));
  wait_flag_dev(HX_SYNC_AIC_FLAG);
  ffts_cross_core_sync(PIPE_MTE3, HxGetffstMsg(0x02, HX_SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_C220_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, HxGetffstMsg(0x02, HX_SYNC_AIV_FLAG));
  wait_flag_dev(HX_SYNC_AIC_AIV_FLAG);
#endif
}

#if defined(__DAV_C220_VEC__)
// ub_ptr helper (PTO Tile -> raw UB pointer). Vec-only.
template <typename T, typename TileT>
__tf__ AICORE inline __ubuf__ T* ub_ptr(TileT &tile) {
  return reinterpret_cast<__ubuf__ T*>(__cce_get_tile_ptr(tile.data()));
}
#endif

// ====================================================================================
//  VEC STAGES — copied verbatim from mega_kernel.cpp (production clean), VEC-pass only.
//  Stage 1 (quant + routing scatter), Stage 3 (swiglu + quant, vec-batched v2),
//  Stage 5 (dequant + scatter-add combine). They read/write the same GM workspaces the
//  cube MatmulImpl produces/consumes (xq_ws -> gu_ws -> iq_ws -> d_ws).
// ====================================================================================
#if defined(__DAV_C220_VEC__)

// ---- STAGE 1 INT4 + SCATTER ----
//
// At prefill Stage 1 is the single biggest vec stage (~796us at m_per=256, ~50% of
// the dbuf-S5 kernel): each expanded row gathers x[orig_t] (H fp16), row-max-quants
// it to int4, and stores the packed bytes + scale. The ORIGINAL loop fully serializes
// MTE2(gather)/V(reduce+quant)/MTE3(store) per row and reuses ONE xTile/qTile, so
// consecutive rows can't overlap.
//
// MEGA_S1_DBUF: software-pipeline the row loop with TWO x/q buffers (ping/pong, exact
// flag balancing — same pattern that fixed Stage 5). The x gather (MTE2) of row m+1
// overlaps the reduce+quant (V) of row m and the q/scale store (MTE3) of row m-1. The
// per-row scale (rmax->fa) still needs a V->S->V round-trip *within* a row (the scale
// depends on that row's own data), but that is now hidden behind the neighbouring
// row's MTE2/MTE3. Math is bit-identical to the serial path (cos preserved).
//
// MEGA_S1_FAST (LEVER 1): multi-row BATCH + ALL-VECTOR scale. The DBUF path still pays
// a per-row V->S->V drain (reads rmax[0] to a scalar to build fa=7/rmax) which breaks
// vector-engine continuity (~337 GB/s vs the ~1 TB/s the gather should hit). FAST kills
// it entirely, using the exact pattern Stage 3 already proves:
//   - BLOCK distribution: each core owns a CONTIGUOUS range of expanded slots [base,end)
//     so the int4-q store of S1_R consecutive rows is a single contiguous [R, H/2] TSTORE.
//   - Gather S1_R rows of x[orig_t] (R back-to-back MTE2, no V op between -> continuous).
//   - One TROWMAX/TROWMIN over the whole [R,H] tile -> [R,1] col abs-max (vectorized).
//   - Scale stays a VECTOR: scCol = rmax/7 (TMULS on the [1,R] reshape); apply the quant
//     scale via TROWEXPANDDIV(x, x, scCol) = x*(7/rmax). NO rmax[0] scalar read, so the
//     V engine never stalls on a scalar handshake; MTE2 gather and MTE3 stores stay
//     vector-continuous across the whole batch.
//   - int4 cast of the full [R,H] tile in one TCVT_FP16_TO_INT4_PACKED (validRow=R loop).
//   - Scale store: strided [R,1] TSTORE into s_gm (row-stride 32 floats), no scalar drain.
// Math is bit-identical to the serial/DBUF scale (rmax/7 -> int4 round). Runtime gate:
// experts with <2 rows on this core fall to a tiny serial tail (one row at a time) so
// decode (tiny per-core M) is unaffected. cos must stay >= 0.99.
#if defined(MEGA_S1_FAST)
// R rows per batch. R=16 measured best at H=2048 (S1 499->193us at m_per=256): amortizes the
// single reduce/scale over 16 rows while [16,2048]fp16 x + [16,2048]fp16 tmp + [16,1024]int4 q
// (~145KB) fit UB. R=32 overflows UB (507015). R=8 is slightly slower (less amortization).
#ifndef MEGA_S1_FAST_R
#define MEGA_S1_FAST_R 16
#endif
constexpr int32_t S1_R = MEGA_S1_FAST_R;
AICORE void stage1_int4_routed(
    __gm__ half* x_gm, __gm__ int32_t* eri_gm, __gm__ int8_t* q_gm, __gm__ float* s_gm,
    uint32_t M_total, uint32_t top_k, uint32_t t_lo = 0, uint32_t t_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  constexpr float kScaleDivisor = 7.0f;
  constexpr float kInvSqrtN = 1.0f;   // Hadamard pre-applied (kernel-skip) -> 1/sqrtN folded out.

  // Reduce/scale col/row tiles need a static dim that is 32-byte aligned (ColMajor NoneBox:
  // Rows*sizeof % 32 == 0). Pad the static rows of those tiny tiles to S1_PAD=32 (like S3's
  // S3_ROWS=32); only the VALID rows (=rows) are used. The wide x/q/tmp tiles keep S1_R rows.
  constexpr int32_t S1_PAD = 32;
  // Wide [R,H] batch tiles. x (gather) | tmp (TROWMAX scratch, same [R,H]) | q (packed int4)
  // | rmax/rmin col [R,1] fp16 | scale col [R,1] fp32 (row-major view for the /7 + reshape).
  using TF16   = Tile<TileType::Vec, half,   S1_R, H_DIM,     BLayout::RowMajor, DYNAMIC, H_DIM>;
  using TI4P   = Tile<TileType::Vec, int8_t, S1_R, H_DIM / 2, BLayout::RowMajor, DYNAMIC, H_DIM / 2>;
  using TRowF16= Tile<TileType::Vec, half,   1,    H_DIM,     BLayout::RowMajor, 1, H_DIM>;
  using TMaxCm = Tile<TileType::Vec, half,   S1_PAD, 1, BLayout::ColMajor, DYNAMIC, 1>;
  using TMaxRm = Tile<TileType::Vec, half,   1, S1_PAD, BLayout::RowMajor, 1, DYNAMIC>;
  using TScRm  = Tile<TileType::Vec, float,  1, S1_PAD, BLayout::RowMajor, 1, DYNAMIC>;

  constexpr unsigned X_BASE  = 0;
  constexpr unsigned TMP_BASE= UB_ALIGN(X_BASE   + S1_R * H_DIM * sizeof(half));
  constexpr unsigned Q_BASE  = UB_ALIGN(TMP_BASE + S1_R * H_DIM * sizeof(half));
  constexpr unsigned RMAX_CM = UB_ALIGN(Q_BASE   + S1_R * (H_DIM / 2) * sizeof(int8_t));
  constexpr unsigned RMIN_CM = UB_ALIGN(RMAX_CM  + S1_PAD * sizeof(half));
  constexpr unsigned SC_CM   = UB_ALIGN(RMIN_CM  + S1_PAD * sizeof(float));

  using GmF = GlobalTensor<half,   TileShape2D<half, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmI8P_2D = GlobalTensor<int8_t, TileShape2D<int8_t, DYNAMIC, DYNAMIC, Layout::ND>,
                                Stride<1,1,1,DYNAMIC,1>, Layout::ND>;
  const TileShape2D<half, 1, DYNAMIC, Layout::ND> fsh(1, H_DIM);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t m_lo = t_lo;
  const uint32_t m_end = (t_hi == 0) ? M_total : t_hi;
  if (m_end <= m_lo) return;

  // Block partition the [m_lo, m_end) rows across cores: contiguous chunk per core so the
  // int4-q store per batch is contiguous. chunk = ceil(total / num_cores).
  const uint32_t total = m_end - m_lo;
  const uint32_t chunk = (total + num_cores - 1) / num_cores;
  const uint32_t my_lo = m_lo + vid * chunk;
  if (my_lo >= m_end) return;
  const uint32_t my_hi = (my_lo + chunk < m_end) ? (my_lo + chunk) : m_end;

  for (uint32_t base = my_lo; base < my_hi; base += S1_R) {
    const int32_t rows = (base + S1_R <= my_hi) ? S1_R : (int32_t)(my_hi - base);

    // ---- gather R rows: R back-to-back TLOADs into sub-rows of the [R,H] x tile ----
    TF16 xT(rows); TASSIGN(xT, X_BASE);
    {
      for (int32_t r = 0; r < rows; ++r) {
        const int32_t oi = eri_gm[base + r];
        const uint32_t ot = (uint32_t)(oi / (int32_t)top_k);
        TRowF16 xrow; TASSIGN(xrow, X_BASE + (unsigned)r * H_DIM * sizeof(half));
        GmF xg(x_gm + (int64_t)ot * H_DIM, fsh);
        TLOAD(xrow, xg);
      }
    }
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // ---- abs-max over each row -> [R,1] col, then RowMajor for the negate/max (S3 idiom) ----
    TF16   tmpT(rows);    TASSIGN(tmpT,   TMP_BASE);
    TMaxCm rmaxCm(rows);  TASSIGN(rmaxCm, RMAX_CM);
    TMaxCm rminCm(rows);  TASSIGN(rminCm, RMIN_CM);
    TROWMAX(rmaxCm, xT, tmpT); pipe_barrier(PIPE_V);
    TROWMIN(rminCm, xT, tmpT); pipe_barrier(PIPE_V);
    TMaxRm rmaxRm(rows);  TASSIGN(rmaxRm, RMAX_CM);
    TMaxRm rminRm(rows);  TASSIGN(rminRm, RMIN_CM);
    TRESHAPE(rmaxRm, rmaxCm);
    TRESHAPE(rminRm, rminCm);
    pipe_barrier(PIPE_V);
    TMULS(rminRm, rminRm, (half)-1.0f); pipe_barrier(PIPE_V);
    TMAX(rmaxRm, rmaxRm, rminRm); pipe_barrier(PIPE_V);
    // rmaxRm now holds the per-row abs-max (RowMajor [1,R]).

    // ---- stored scale = abs-max * (1/sqrtN / 7), as a fp32 row tile (scalar-loop store, like S3) ----
    TScRm scRm(rows);  TASSIGN(scRm, SC_CM);
    TCVT(scRm, rmaxRm, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
    TMULS(scRm, scRm, (float)(kInvSqrtN / kScaleDivisor)); pipe_barrier(PIPE_V);

    // ---- apply int4 quant scale ALL-VECTOR: divide each row by its abs-max, then *7 (S3 idiom).
    // Divisor = the RAW abs-max column (reshape rmaxRm back to ColMajor). No rmax[0] scalar read,
    // so the V engine never stalls; x = x / rmax * 7 = x * 7/rmax. ----
    TRESHAPE(rmaxCm, rmaxRm);
    pipe_barrier(PIPE_V);
#ifdef MEGA_SCALE_FOLD
    // Fold the *7 quant constant into the tiny [R,1] divisor: x/(rmax/7) == x*7/rmax. Deletes
    // the wide [R,H] TMULS (S1 is vec-bound). Math-identical mod fp16 rounding order.
    TMULS(rmaxCm, rmaxCm, (half)(1.0f / kScaleDivisor)); pipe_barrier(PIPE_V);
    TROWEXPANDDIV(xT, xT, rmaxCm); pipe_barrier(PIPE_V);
#else
    TROWEXPANDDIV(xT, xT, rmaxCm); pipe_barrier(PIPE_V);
    TMULS(xT, xT, (half)kScaleDivisor); pipe_barrier(PIPE_V);
#endif

    // ---- int4 cast the whole [R,H] tile (validRow=rows loop inside the helper) ----
    TI4P qT(rows); TASSIGN(qT, Q_BASE);
    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(qT, xT, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);

    // ---- store: contiguous [R, H/2] int4 (single TSTORE) + scalar-loop scale (row-stride 32) ----
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TileShape2D<int8_t, DYNAMIC, DYNAMIC, Layout::ND> qShape(rows, H_DIM / 2);
    GmI8P_2D qg(q_gm + (int64_t)base * (H_DIM / 2), qShape,
                Stride<1,1,1,DYNAMIC,1>(1,1,1, H_DIM / 2, 1));
    TSTORE(qg, qT);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    // scale: make scRm (V-produced) visible to S, then scalar-loop store (rows <= S1_R, cheap).
    set_flag(PIPE_V, PIPE_S, EVENT_ID0); wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    {
      __ubuf__ float* sc_ub = ub_ptr<float>(scRm);
      for (int32_t r = 0; r < rows; ++r) s_gm[(int64_t)(base + r) * 32] = sc_ub[r];
    }
    set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
  }
}
#elif defined(MEGA_S1_DBUF)
AICORE void stage1_int4_routed(
    __gm__ half* x_gm, __gm__ int32_t* eri_gm, __gm__ int8_t* q_gm, __gm__ float* s_gm,
    uint32_t M_total, uint32_t top_k, uint32_t t_lo = 0, uint32_t t_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  constexpr float kScaleDivisor = 7.0f;
  constexpr float kInvSqrtN = 1.0f;

  using TF16   = Tile<TileType::Vec, half,   1, H_DIM,     BLayout::RowMajor, 1, H_DIM>;
  using TI4P   = Tile<TileType::Vec, int8_t, 1, H_DIM / 2, BLayout::RowMajor, 1, H_DIM / 2>;
  using TMax   = Tile<TileType::Vec, half,   1, 16,        BLayout::RowMajor, 1, 16>;
  using TScale1 = Tile<TileType::Vec, float, 1, 8,         BLayout::RowMajor, 1, 8>;

  // UB: x0|x1 (gather), red (shared reduce scratch), rmax0|rmin0|rmax1|rmin1,
  //     q0|q1 (packed int4), s0|s1 (scale). Two of each data buffer for ping/pong.
  constexpr unsigned X0   = 0;
  constexpr unsigned X1   = UB_ALIGN(X0   + H_DIM * sizeof(half));
  constexpr unsigned RED  = UB_ALIGN(X1   + H_DIM * sizeof(half));
  constexpr unsigned RMX0 = UB_ALIGN(RED  + H_DIM * sizeof(half));
  constexpr unsigned RMN0 = UB_ALIGN(RMX0 + 16 * sizeof(half));
  constexpr unsigned RMX1 = UB_ALIGN(RMN0 + 16 * sizeof(half));
  constexpr unsigned RMN1 = UB_ALIGN(RMX1 + 16 * sizeof(half));
  constexpr unsigned Q0   = UB_ALIGN(RMN1 + 16 * sizeof(half));
  constexpr unsigned Q1   = UB_ALIGN(Q0   + (H_DIM / 2) * sizeof(int8_t));
  constexpr unsigned S0   = UB_ALIGN(Q1   + (H_DIM / 2) * sizeof(int8_t));
  constexpr unsigned S1B  = UB_ALIGN(S0   + 8 * sizeof(float));

  TF16 x0, x1, redS; TMax rmax0, rmin0, rmax1, rmin1; TI4P q0, q1; TScale1 s0, s1;
  TASSIGN(x0, X0); TASSIGN(x1, X1); TASSIGN(redS, RED);
  TASSIGN(rmax0, RMX0); TASSIGN(rmin0, RMN0); TASSIGN(rmax1, RMX1); TASSIGN(rmin1, RMN1);
  TASSIGN(q0, Q0); TASSIGN(q1, Q1); TASSIGN(s0, S0); TASSIGN(s1, S1B);

  using GmF = GlobalTensor<half,   TileShape2D<half, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmI = GlobalTensor<int8_t, TileShape2D<int8_t, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmS = GlobalTensor<float,  TileShape2D<float, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  const TileShape2D<half,   1, DYNAMIC, Layout::ND> fsh(1, H_DIM);
  const TileShape2D<int8_t, 1, DYNAMIC, Layout::ND> ish(1, H_DIM / 2);
  const TileShape2D<float,  1, DYNAMIC, Layout::ND> ssh(1, 8);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t m_end = (t_hi == 0) ? M_total : t_hi;

  // Per-buffer gather: scalar orig_t (from eri) + TLOAD x; signal x-loaded (MTE2->V, id B).
  // cur_M##B records the expanded slot for the row loaded into buffer B (used at store).
#define S1_LOAD(B, MM)                                                        \
  do {                                                                        \
    cur_M##B = (uint32_t)(MM);                                                \
    int32_t _oi = eri_gm[(MM)];                                               \
    uint32_t _ot = (uint32_t)(_oi / (int32_t)top_k);                          \
    GmF _xg(x_gm + (int64_t)_ot * H_DIM, fsh);                                \
    TLOAD(x##B, _xg);                                                         \
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                                \
  } while (0)
  // Per-buffer compute+store: wait x-loaded; row-max reduce -> scalar scale -> quant ->
  // store packed int4 + scale. Signals x-free (V->MTE2, id 2/3) after the scaling TMULS
  // (last read of x##B) and q/s-free (MTE3->V, id 4/5) after the stores. NEEDFREE waits
  // this buffer's prior q/s-free (WAR on q##B/s##B vs the prior store).
#define S1_COMPUTE(B, NEEDFREE)                                               \
  do {                                                                        \
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                               \
    if (NEEDFREE) wait_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B)));  \
    TROWMAX(rmax##B, x##B, redS); pipe_barrier(PIPE_V);                       \
    TROWMIN(rmin##B, x##B, redS); pipe_barrier(PIPE_V);                       \
    TMULS(rmin##B, rmin##B, (half)-1.0f); pipe_barrier(PIPE_V);               \
    TMAX(rmax##B, rmax##B, rmin##B); pipe_barrier(PIPE_V);                    \
    set_flag(PIPE_V, PIPE_S, (event_t)(EVENT_ID6 + (B)));                    \
    wait_flag(PIPE_V, PIPE_S, (event_t)(EVENT_ID6 + (B)));                   \
    __ubuf__ half* _rm = ub_ptr<half>(rmax##B);                              \
    const float _rmv = (float)_rm[0];                                         \
    const float _sc = (_rmv == 0.0f) ? 1e-6f : (_rmv * (kInvSqrtN / kScaleDivisor)); \
    const float _fa = (_rmv == 0.0f) ? 0.0f  : (kScaleDivisor / _rmv);        \
    set_flag(PIPE_S, PIPE_V, (event_t)(EVENT_ID6 + (B)));                    \
    wait_flag(PIPE_S, PIPE_V, (event_t)(EVENT_ID6 + (B)));                   \
    TMULS(x##B, x##B, (half)_fa); pipe_barrier(PIPE_V);                       \
    set_flag(PIPE_V, PIPE_MTE2, (event_t)(EVENT_ID2 + (B)));                 \
    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(q##B, x##B, RoundMode::CAST_RINT); \
    pipe_barrier(PIPE_V);                                                     \
    __ubuf__ float* _su = ub_ptr<float>(s##B);                               \
    _su[0] = _sc;                                                             \
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); \
    GmI _qg(q_gm + (int64_t)cur_M##B * (H_DIM / 2), ish);                     \
    TSTORE(_qg, q##B);                                                        \
    GmS _sg(s_gm + (int64_t)cur_M##B * 32, ssh);                             \
    TSTORE(_sg, s##B);                                                        \
    set_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B)));                 \
  } while (0)

  // This core's assigned rows are M = (t_lo+vid) + j*num_cores, j=0,1,2,... Pipeline them
  // with ping/pong on j&1 (same exact-flag-balancing as Stage 5). cur_M0/1 carry the
  // expanded slot for the row currently in each buffer (needed at store).
  uint32_t cur_M0 = 0, cur_M1 = 0;
  // Build the row list bounds.
  const uint32_t m_first = t_lo + vid;
  if (m_first >= m_end) return;
  // Count rows for this core.
  int64_t set_difS[2] = {0, 0};   // x-free (V->MTE2) pending per buffer
  int64_t set_qsS[2]  = {0, 0};   // q/s-free (MTE3->V) pending per buffer

  // Prologue: load row 0 into buffer 0.
  S1_LOAD(0, m_first);
  int64_t j = 0;
  for (uint32_t M = m_first; M < m_end; M += num_cores, ++j) {
    const bool cb0 = ((j & 1) == 0);
    const uint32_t Mnext = M + num_cores;
    // Prefetch next row (other buffer); wait that buffer's x-free if pending (WAR on x).
    if (Mnext < m_end) {
      if (cb0) { if (set_difS[1]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3); set_difS[1]=0; } S1_LOAD(1, Mnext); }
      else     { if (set_difS[0]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2); set_difS[0]=0; } S1_LOAD(0, Mnext); }
    }
    // Compute+store this row from buffer cb. Wait q/s-free if this buffer was stored before.
    if (cb0) { S1_COMPUTE(0, (set_qsS[0] != 0)); set_qsS[0]=1; set_difS[0]=1; }
    else     { S1_COMPUTE(1, (set_qsS[1] != 0)); set_qsS[1]=1; set_difS[1]=1; }
  }
  // Drain trailing pending flags (1:1 with the unwaited sets).
  if (set_difS[0]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID2);
  if (set_difS[1]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID3);
  if (set_qsS[0])  wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID4);
  if (set_qsS[1])  wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID5);
#undef S1_LOAD
#undef S1_COMPUTE
}
#else
AICORE void stage1_int4_routed(
    __gm__ half* x_gm, __gm__ int32_t* eri_gm, __gm__ int8_t* q_gm, __gm__ float* s_gm,
    uint32_t M_total, uint32_t top_k, uint32_t t_lo = 0, uint32_t t_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  constexpr float kScaleDivisor = 7.0f;
  // Stage A: Hadamard applied by Python before the kernel (kernel-skip), scale uses 1/sqrtN=1.
  constexpr float kInvSqrtN = 1.0f;

  using TF16   = Tile<TileType::Vec, half,   1, H_DIM,     BLayout::RowMajor, 1, H_DIM>;
  using TI4P   = Tile<TileType::Vec, int8_t, 1, H_DIM / 2, BLayout::RowMajor, 1, H_DIM / 2>;
  using TMax   = Tile<TileType::Vec, half,   1, 16,        BLayout::RowMajor, 1, 16>;
  using TScale1 = Tile<TileType::Vec, float, 1, 8,         BLayout::RowMajor, 1, 8>;

  constexpr unsigned X_BASE = 0;
  constexpr unsigned REDUCE = UB_ALIGN(X_BASE + H_DIM * sizeof(half));
  constexpr unsigned RMAX   = UB_ALIGN(REDUCE + H_DIM * sizeof(half));
  constexpr unsigned RMIN   = UB_ALIGN(RMAX   + 16 * sizeof(half));
  constexpr unsigned Q_BASE = UB_ALIGN(RMIN   + 16 * sizeof(half));
  constexpr unsigned S_BASE = UB_ALIGN(Q_BASE + (H_DIM / 2) * sizeof(int8_t));

  TF16 xTile, redS;  TMax rMax, rMin;  TI4P qTile;  TScale1 sTile;
  TASSIGN(xTile, X_BASE);  TASSIGN(redS, REDUCE);
  TASSIGN(rMax, RMAX);     TASSIGN(rMin, RMIN);
  TASSIGN(qTile, Q_BASE);  TASSIGN(sTile, S_BASE);

  using GmF = GlobalTensor<half,
        TileShape2D<half, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmI = GlobalTensor<int8_t,
        TileShape2D<int8_t, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmS = GlobalTensor<float,
        TileShape2D<float, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  const TileShape2D<half,   1, DYNAMIC, Layout::ND> fsh(1, H_DIM);
  const TileShape2D<int8_t, 1, DYNAMIC, Layout::ND> ish(1, H_DIM / 2);
  const TileShape2D<float,  1, DYNAMIC, Layout::ND> ssh(1, 8);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t m_end = (t_hi == 0) ? M_total : t_hi;

  for (uint32_t M = t_lo + vid; M < m_end; M += num_cores) {
    set_flag(PIPE_V, PIPE_S, EVENT_ID0); wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const int32_t orig_idx = eri_gm[M];
    const uint32_t orig_t = (uint32_t)(orig_idx / (int32_t)top_k);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);

    GmF xg(x_gm + (int64_t)orig_t * H_DIM, fsh);
    TLOAD(xTile, xg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TROWMAX(rMax, xTile, redS); pipe_barrier(PIPE_V);
    TROWMIN(rMin, xTile, redS); pipe_barrier(PIPE_V);
    TMULS(rMin, rMin, (half)-1.0f); pipe_barrier(PIPE_V);
    TMAX(rMax, rMax, rMin); pipe_barrier(PIPE_V);
    set_flag(PIPE_V, PIPE_S, EVENT_ID0); wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    __ubuf__ half* rm_ub = ub_ptr<half>(rMax);
    const float rmv = (float)rm_ub[0];
    const float sc = (rmv == 0.0f) ? 1e-6f : (rmv * (kInvSqrtN / kScaleDivisor));
    const float fa = (rmv == 0.0f) ? 0.0f  : (kScaleDivisor / rmv);
    set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    TMULS(xTile, xTile, (half)fa); pipe_barrier(PIPE_V);

    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(qTile, xTile, RoundMode::CAST_RINT);
    pipe_barrier(PIPE_V);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GmI qg(q_gm + (int64_t)M * (H_DIM / 2), ish);
    TSTORE(qg, qTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    __ubuf__ float* s_ub = ub_ptr<float>(sTile);
    s_ub[0] = sc;
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
    GmS sg(s_gm + (int64_t)M * 32, ssh);
    TSTORE(sg, sTile);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  }
}
#endif  // MEGA_S1_DBUF

// ---- STAGE 3 INT4 v2 — vec-batched dequant + swiglu + int4 quant ----
constexpr uint32_t S3_ROWS = 32;

// MEGA_S3_DBUF: software-pipeline the per-tile loop with TWO input/output buffers
// (ping/pong on the tile index, exact 1:1 flag balancing — same pattern that fixed
// Stage 1 & Stage 5). Stage 3 was the only remaining un-pipelined vec stage at
// prefill (~207us at m_per=256). The serial version drains the pipe ~4x per tile
// (MTE2->V load, V->S scalar xs read, S->V, V->MTE3 iq store, MTE3->S scalar is
// write, S->V) and reuses ONE set of gI32/uI32/iq tiles, so consecutive tiles have
// a hard WAR dependency. The DBUF version prefetches tile t+1's gI32/uI32 (MTE2) and
// its scalar xs metadata while the vector engine computes tile t, and stores tile
// t's iq (MTE3) + scalar is afterwards. The big compute-scratch tiles (gF32/uF32/
// tmp/swF16/reductions) stay single (consumed within one tile's compute, no overlap
// needed). Per-tile vector math is bit-identical to the serial version (cos preserved).
#ifdef MEGA_S3_DBUF
// S3FP16 VARIANT: the gate_up cube now folds the per-channel w13 scale into the FIXPIPE
// (SetQuantVector) and emits fp16 gu_ws (HALF the int32 bytes). S3 loads fp16 gate/up,
// casts to fp32, applies only the per-token xs scalar — the per-channel wsg/wsu TCOLEXPANDMUL
// is gone (folded into the cube). gu_gm is half here (was int32 in mega_kernel_hybrid.cpp).
AICORE void stage3_int4_swiglu_quant_grouped_v2(
    __gm__ half* gu_gm, __gm__ float* xs_gm, __gm__ float* ws_gm, __gm__ int64_t* gl_gm,
    __gm__ int8_t* iq_gm, __gm__ float* is_gm, uint32_t T, uint32_t E,
    uint32_t e_lo = 0, uint32_t e_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  constexpr float kScaleDivisor = 7.0f;
  constexpr int32_t S3_STEP = (I_DIM <= 128) ? 7 : (int32_t)S3_ROWS;

  using TI32   = Tile<TileType::Vec, int32_t, S3_ROWS, I_DIM, BLayout::RowMajor, DYNAMIC, I_DIM>;
  using TF32   = Tile<TileType::Vec, float,   S3_ROWS, I_DIM, BLayout::RowMajor, DYNAMIC, I_DIM>;
  using TF16   = Tile<TileType::Vec, half,    S3_ROWS, I_DIM, BLayout::RowMajor, DYNAMIC, I_DIM>;
  using TI4P   = Tile<TileType::Vec, int8_t,  S3_ROWS, I_DIM/2, BLayout::RowMajor, DYNAMIC, I_DIM/2>;
  using TWsRow = Tile<TileType::Vec, float,   1, I_DIM, BLayout::RowMajor, 1, I_DIM>;
  using TXs    = Tile<TileType::Vec, float,   S3_ROWS, 1, BLayout::ColMajor, DYNAMIC, 1>;
  using TRMaxCm = Tile<TileType::Vec, half,   S3_ROWS, 1, BLayout::ColMajor, DYNAMIC, 1>;
  using TRMaxRm = Tile<TileType::Vec, half,   1, S3_ROWS, BLayout::RowMajor, 1, DYNAMIC>;
  using TScaleR = Tile<TileType::Vec, float,  1, S3_ROWS, BLayout::RowMajor, 1, DYNAMIC>;

  // Two input buffers (gI32/uI32), two iq output buffers, two xs + two is scalar tiles.
  // Single compute scratch (gF32/uF32/tmp/swF16/reductions).
  constexpr unsigned GATE_I32_0 = 0x0;
  constexpr unsigned UP_I32_0   = GATE_I32_0 + S3_ROWS * I_DIM * 4;
  constexpr unsigned GATE_I32_1 = UP_I32_0   + S3_ROWS * I_DIM * 4;
  constexpr unsigned UP_I32_1   = GATE_I32_1 + S3_ROWS * I_DIM * 4;
  constexpr unsigned GATE_F32   = UP_I32_1   + S3_ROWS * I_DIM * 4;
  constexpr unsigned UP_F32     = GATE_F32   + S3_ROWS * I_DIM * 4;
  constexpr unsigned TMP_F32    = UP_F32     + S3_ROWS * I_DIM * 4;
  constexpr unsigned WSG_F32    = TMP_F32    + S3_ROWS * I_DIM * 4;
  constexpr unsigned WSU_F32    = WSG_F32    + I_DIM * 4;
  constexpr unsigned SWG_F16    = WSU_F32    + I_DIM * 4;
  constexpr unsigned IQ_0       = SWG_F16    + S3_ROWS * I_DIM * 2;
  constexpr unsigned IQ_1       = IQ_0       + S3_ROWS * (I_DIM / 2);
  constexpr unsigned RMAX_CM    = IQ_1       + S3_ROWS * (I_DIM / 2);
  constexpr unsigned RMIN_CM    = RMAX_CM    + S3_ROWS * 2;
  // XS / IS double-buffered (tiny, S3_ROWS*4 each): the scalar xs read and is write
  // touch a per-buffer slot so the next COMPUTE (other buffer) can't WAR-clobber.
  constexpr unsigned XS_0       = RMIN_CM    + S3_ROWS * 2;
  constexpr unsigned XS_1       = XS_0       + S3_ROWS * 4;
  constexpr unsigned IS_0       = XS_1       + S3_ROWS * 4;
  constexpr unsigned IS_1       = IS_0       + S3_ROWS * 4;

  TWsRow wsg;  TASSIGN(wsg, WSG_F32);
  TWsRow wsu;  TASSIGN(wsu, WSU_F32);

  using GmI32_2D = GlobalTensor<int32_t, TileShape2D<int32_t, DYNAMIC, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,DYNAMIC,1>, Layout::ND>;
  using GmF16_2D = GlobalTensor<half,    TileShape2D<half, DYNAMIC, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,DYNAMIC,1>, Layout::ND>;  // S3FP16: fp16 gate_up GM
  using GmF32_1D = GlobalTensor<float,   TileShape2D<float, 1, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,1,1>, Layout::ND>;
  using GmI8P_2D = GlobalTensor<int8_t,  TileShape2D<int8_t, DYNAMIC, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,DYNAMIC,1>, Layout::ND>;

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t e_end3 = (e_hi == 0) ? E : e_hi;

  // Per-buffer base addresses indexed by B in {0,1}, spelled out for CCE (compile-time
  // tile operands / event ids; a runtime-indexed array miscompiles the flags).
  // Event budget: ID0/1 MTE2->V gI/uI load-ready; ID2/3 V->MTE2 input-free (next reload
  // of this buffer waits, WAR on gI/uI vs the TCVT reads); ID4/5 MTE3->V iq-free (next
  // compute of this buffer waits, WAR on iq vs the prior store); ID6 ws load; ID7 scalar.
  // Only the big MTE2 loads (gI/uI) and the MTE3 iq store are double-buffered/overlapped.
  // The tiny per-tile scalar xs read and is write (<=S3_STEP elements) stay single-slot,
  // done just-in-time inside COMPUTE with short local S<->V / MTE3<->S handshakes — they
  // serialize only the scalar pipe, not the dominant MTE2/MTE3 traffic.
#define S3_LOAD(B, MT, ROWS)                                                  \
  do {                                                                        \
    const int32_t _rows = (ROWS);                                            \
    const int64_t _t0 = t_start + (MT);                                      \
    TileShape2D<half, DYNAMIC, DYNAMIC, Layout::ND> _gShape(_rows, I_DIM); \
    GmF16_2D _gGm(gu_gm + _t0 * N_GU,         _gShape, Stride<1,1,1,DYNAMIC,1>(1,1,1, N_GU, 1)); \
    GmF16_2D _uGm(gu_gm + _t0 * N_GU + I_DIM, _gShape, Stride<1,1,1,DYNAMIC,1>(1,1,1, N_GU, 1)); \
    TF16 _g(_rows); TASSIGN(_g, GATE_I32_##B);  /* fp16 tile in oversized int32 slot */     \
    TF16 _u(_rows); TASSIGN(_u, UP_I32_##B);                                 \
    TLOAD(_g, _gGm);                                                          \
    TLOAD(_u, _uGm);                                                          \
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                               \
  } while (0)

// Apply the int4 quant scale to D using the per-row abs-max RM. MEGA_SCALE_FOLD folds the *7
// constant into the tiny [R,1] divisor (x/(rmax/7)==x*7/rmax) to delete the wide [R,I_DIM] TMULS.
// (Defined as a macro so the #ifdef sits outside S3_COMPUTE's macro body.)
#ifdef MEGA_SCALE_FOLD
#define S3_QSCALE(D, RM) do { TMULS(RM, RM, (half)(1.0f / kScaleDivisor)); pipe_barrier(PIPE_V); \
                              TROWEXPANDDIV(D, D, RM); pipe_barrier(PIPE_V); } while (0)
#else
#define S3_QSCALE(D, RM) do { TROWEXPANDDIV(D, D, RM); pipe_barrier(PIPE_V); \
                              TMULS(D, D, (half)kScaleDivisor); pipe_barrier(PIPE_V); } while (0)
#endif

#define S3_COMPUTE(B, MT, ROWS, NEEDFREE)                                     \
  do {                                                                        \
    const int32_t _crows = (ROWS);                                           \
    const int64_t _ct0 = t_start + (MT);                                     \
    const int32_t rows = _crows;                                             \
    const int64_t t0 = _ct0;                                                 \
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                              \
    if (NEEDFREE) wait_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B))); \
    /* JIT scalar xs read into this buffer's XS slot (S pipe), made visible to V */ \
    TXs  xs_dyn(rows);   TASSIGN(xs_dyn,   XS_0 + (B) * (S3_ROWS * 4));      \
    {                                                                        \
      __ubuf__ float* _xs_ub = ub_ptr<float>(xs_dyn);                       \
      for (int32_t _r = 0; _r < rows; ++_r) _xs_ub[_r] = xs_gm[(t0 + _r) * 32]; \
    }                                                                        \
    set_flag(PIPE_S, PIPE_V, EVENT_ID7); wait_flag(PIPE_S, PIPE_V, EVENT_ID7); \
    TF16 gI32_dyn(rows); TASSIGN(gI32_dyn, GATE_I32_##B);  /* fp16: cube already dequant'd */ \
    TF16 uI32_dyn(rows); TASSIGN(uI32_dyn, UP_I32_##B);                      \
    TF32 gF32_dyn(rows); TASSIGN(gF32_dyn, GATE_F32);                        \
    TF32 uF32_dyn(rows); TASSIGN(uF32_dyn, UP_F32);                          \
    TCVT(gF32_dyn, gI32_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);    \
    TCVT(uF32_dyn, uI32_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);    \
    set_flag(PIPE_V, PIPE_MTE2, (event_t)(EVENT_ID2 + (B)));                \
    TROWEXPANDMUL(gF32_dyn, gF32_dyn, xs_dyn); pipe_barrier(PIPE_V);         \
    TROWEXPANDMUL(uF32_dyn, uF32_dyn, xs_dyn); pipe_barrier(PIPE_V);         \
    /* S3FP16: per-channel w13 scale folded into the gate_up cube FIXPIPE; no TCOLEXPANDMUL */ \
    TF32 tmp_dyn(rows); TASSIGN(tmp_dyn, TMP_F32);                           \
    TMULS(tmp_dyn, gF32_dyn, -1.0f); pipe_barrier(PIPE_V);                   \
    TEXP(tmp_dyn, tmp_dyn);          pipe_barrier(PIPE_V);                   \
    TADDS(tmp_dyn, tmp_dyn, 1.0f);   pipe_barrier(PIPE_V);                   \
    TDIV(gF32_dyn, gF32_dyn, tmp_dyn); pipe_barrier(PIPE_V);                 \
    TMUL(gF32_dyn, gF32_dyn, uF32_dyn); pipe_barrier(PIPE_V);                \
    TF16 swF16_dyn(rows); TASSIGN(swF16_dyn, SWG_F16);                       \
    TCVT(swF16_dyn, gF32_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);   \
    TF16 tmpRedF16(rows); TASSIGN(tmpRedF16, TMP_F32);                       \
    TRMaxCm rmaxCm_dyn(rows); TASSIGN(rmaxCm_dyn, RMAX_CM);                  \
    TRMaxCm rminCm_dyn(rows); TASSIGN(rminCm_dyn, RMIN_CM);                  \
    TROWMAX(rmaxCm_dyn, swF16_dyn, tmpRedF16); pipe_barrier(PIPE_V);         \
    TROWMIN(rminCm_dyn, swF16_dyn, tmpRedF16); pipe_barrier(PIPE_V);         \
    TRMaxRm rmaxRm_dyn(rows); TASSIGN(rmaxRm_dyn, RMAX_CM);                  \
    TRMaxRm rminRm_dyn(rows); TASSIGN(rminRm_dyn, RMIN_CM);                  \
    TRESHAPE(rmaxRm_dyn, rmaxCm_dyn);                                        \
    TRESHAPE(rminRm_dyn, rminCm_dyn);                                        \
    pipe_barrier(PIPE_V);                                                    \
    TMULS(rminRm_dyn, rminRm_dyn, (half)-1.0f); pipe_barrier(PIPE_V);        \
    TMAX(rmaxRm_dyn, rmaxRm_dyn, rminRm_dyn); pipe_barrier(PIPE_V);          \
    TScaleR isOut_dyn(rows); TASSIGN(isOut_dyn, IS_0 + (B) * (S3_ROWS * 4)); \
    TCVT(isOut_dyn, rmaxRm_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V); \
    TMULS(isOut_dyn, isOut_dyn, 1.0f / kScaleDivisor); pipe_barrier(PIPE_V); \
    TRESHAPE(rmaxCm_dyn, rmaxRm_dyn);                                        \
    pipe_barrier(PIPE_V);                                                    \
    S3_QSCALE(swF16_dyn, rmaxCm_dyn);                                        \
    TI4P iq_dyn(rows); TASSIGN(iq_dyn, IQ_##B);                              \
    fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(iq_dyn, swF16_dyn, RoundMode::CAST_RINT); \
    pipe_barrier(PIPE_V);                                                    \
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); \
    TileShape2D<int8_t, DYNAMIC, DYNAMIC, Layout::ND> iqShape(rows, I_DIM / 2); \
    GmI8P_2D iqGm(iq_gm + t0 * (I_DIM / 2), iqShape, Stride<1,1,1,DYNAMIC,1>(1,1,1, I_DIM / 2, 1)); \
    TSTORE(iqGm, iq_dyn);                                                    \
    set_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B)));                \
    /* scalar is write: isOut_dyn (per-buffer slot) produced by V above; make it       \
       visible to S, then write GM. Per-buffer slot => no cross-compute WAR. */         \
    set_flag(PIPE_V, PIPE_S, EVENT_ID7); wait_flag(PIPE_V, PIPE_S, EVENT_ID7); \
    {                                                                        \
      __ubuf__ float* is_ub = ub_ptr<float>(isOut_dyn);                     \
      for (int32_t r = 0; r < rows; ++r) is_gm[(t0 + r) * 32] = is_ub[r];   \
    }                                                                        \
  } while (0)

#ifdef MEGA_S3_ROWPART
  // ROW-partition + expert-walk (skew-immune): each core takes a contiguous row band of this
  // chunk and walks the experts overlapping it, reloading the per-expert weight-scale (wsg/wsu)
  // at each expert boundary. Removes the hot-expert straggler that the all-core barrier
  // otherwise serialized on. ws is tiny (I_DIM f32) so the per-segment reload is negligible vs
  // the gate/up tile traffic. The shared tile body below is reused VERBATIM: its n_tiles<2
  // `continue` advances the for-loop to the next expert in both partitionings, and `_cur` is
  // advanced before the body so the walk stays correct across that continue.
  int64_t _r_lo = (e_lo > 0) ? gl_gm[e_lo - 1] : 0;
  int64_t _r_hi = gl_gm[e_end3 - 1];
  int64_t _rtot = _r_hi - _r_lo;
  if (_rtot <= 0) return;
  int64_t _rchunk = (_rtot + (int64_t)num_cores - 1) / (int64_t)num_cores;
  int64_t _my_lo = _r_lo + (int64_t)vid * _rchunk;
  if (_my_lo >= _r_hi) return;
  int64_t _my_hi = (_my_lo + _rchunk < _r_hi) ? (_my_lo + _rchunk) : _r_hi;
  uint32_t _e0 = e_lo;
  while (_e0 < e_end3 && gl_gm[_e0] <= _my_lo) ++_e0;
  int64_t _cur = _my_lo;
  for (uint32_t e = _e0; _cur < _my_hi && e < e_end3; ++e) {
    int64_t _seg_end = (gl_gm[e] < _my_hi) ? gl_gm[e] : _my_hi;
    int64_t t_start = _cur;
    int32_t M_e = (int32_t)(_seg_end - _cur);
    _cur = _seg_end;
    if (M_e <= 0) continue;
#else
  for (uint32_t e = e_lo + vid; e < e_end3; e += num_cores) {
    int64_t t_end = gl_gm[e];
    int64_t t_start = (e > 0) ? gl_gm[e - 1] : 0;
    int32_t M_e = (int32_t)(t_end - t_start);
    if (M_e <= 0) continue;
#endif

    // S3FP16: the per-channel w13 scale is folded into the gate_up cube FIXPIPE
    // (SetQuantVector), so gu_ws arrives already dequant'd as fp16. No per-expert wsg/wsu
    // reload here, and S3_COMPUTE drops the TCOLEXPANDMUL — one fewer MTE2 + two fewer
    // wide vec passes per expert. (ws_gm arg is unused; WSG_F32/WSU_F32 slots left reserved.)

    // Number of tiles for this expert.
    const int32_t n_tiles = (M_e + S3_STEP - 1) / S3_STEP;

    // Runtime gate: tiny experts (decode, n_tiles<2) fall back to a simple serial loop
    // where the pipeline prologue/drain costs more than the overlap saves. (DBUF-SERIAL
    // env override forces this path for every expert when debugging the COMPUTE math.)
#ifdef MEGA_S3_DBUF_SERIAL_DEBUG
    if (1) {
#else
    if (n_tiles < 2) {
#endif
      for (int32_t ti = 0; ti < n_tiles; ++ti) {
        const int32_t mt = ti * S3_STEP;
        const int32_t rows = (mt + S3_STEP <= M_e) ? S3_STEP : (M_e - mt);
        S3_LOAD(0, mt, rows);
        S3_COMPUTE(0, mt, rows, (ti > 0) ? 1 : 0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      }
      if (n_tiles > 0) wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
      continue;
    }

#define S3_TILE_ROWS(TI) (((TI) * S3_STEP + S3_STEP <= M_e) ? S3_STEP : (M_e - (TI) * S3_STEP))

    // Depth-2 software pipeline over tiles. PROLOGUE loads tile 0; each step prefetches
    // tile ti+1 (other buffer) then computes+stores tile ti (buffer ti&1). Flag balance:
    // every set_flag matched 1:1 by a wait_flag (input-free / iq-free per buffer).
    S3_LOAD(0, 0, S3_TILE_ROWS(0));
    int64_t set_dif[2] = {0, 0};   // input-free pending per buffer (V->MTE2 ID2/3)
    int64_t set_buf[2] = {0, 0};   // iq-free pending per buffer (MTE3->V ID4/5)
    for (int32_t ti = 0; ti < n_tiles; ++ti) {
      const bool cb0 = ((ti & 1) == 0);
      const int32_t mt = ti * S3_STEP;
      const int32_t rows = S3_TILE_ROWS(ti);
      if (ti + 1 < n_tiles) {
        const int32_t nmt = (ti + 1) * S3_STEP;
        const int32_t nrows = S3_TILE_ROWS(ti + 1);
        if (cb0) { if (set_dif[1]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3); set_dif[1]=0; } S3_LOAD(1, nmt, nrows); }
        else     { if (set_dif[0]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2); set_dif[0]=0; } S3_LOAD(0, nmt, nrows); }
      }
      if (cb0) { S3_COMPUTE(0, mt, rows, (set_buf[0] != 0)); set_buf[0]=1; set_dif[0]=1; }
      else     { S3_COMPUTE(1, mt, rows, (set_buf[1] != 0)); set_buf[1]=1; set_dif[1]=1; }
    }
    if (set_dif[0]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID2);
    if (set_dif[1]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID3);
    if (set_buf[0]) wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID4);
    if (set_buf[1]) wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID5);
  }
#undef S3_TILE_ROWS
#undef S3_LOAD
#undef S3_COMPUTE
#undef S3_QSCALE
}
#else
AICORE void stage3_int4_swiglu_quant_grouped_v2(
    __gm__ int32_t* gu_gm, __gm__ float* xs_gm, __gm__ float* ws_gm, __gm__ int64_t* gl_gm,
    __gm__ int8_t* iq_gm, __gm__ float* is_gm, uint32_t T, uint32_t E,
    uint32_t e_lo = 0, uint32_t e_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  constexpr float kScaleDivisor = 7.0f;

  using TI32   = Tile<TileType::Vec, int32_t, S3_ROWS, I_DIM, BLayout::RowMajor, DYNAMIC, I_DIM>;
  using TF32   = Tile<TileType::Vec, float,   S3_ROWS, I_DIM, BLayout::RowMajor, DYNAMIC, I_DIM>;
  using TF16   = Tile<TileType::Vec, half,    S3_ROWS, I_DIM, BLayout::RowMajor, DYNAMIC, I_DIM>;
  using TI4P   = Tile<TileType::Vec, int8_t,  S3_ROWS, I_DIM/2, BLayout::RowMajor, DYNAMIC, I_DIM/2>;
  using TWsRow = Tile<TileType::Vec, float,   1, I_DIM, BLayout::RowMajor, 1, I_DIM>;
  using TXs    = Tile<TileType::Vec, float,   S3_ROWS, 1, BLayout::ColMajor, DYNAMIC, 1>;
  using TRMaxCm = Tile<TileType::Vec, half,   S3_ROWS, 1, BLayout::ColMajor, DYNAMIC, 1>;
  using TRMaxRm = Tile<TileType::Vec, half,   1, S3_ROWS, BLayout::RowMajor, 1, DYNAMIC>;
  using TScaleR = Tile<TileType::Vec, float,  1, S3_ROWS, BLayout::RowMajor, 1, DYNAMIC>;

  constexpr unsigned GATE_I32 = 0x0;
  constexpr unsigned UP_I32   = GATE_I32 + S3_ROWS * I_DIM * 4;
  constexpr unsigned GATE_F32 = UP_I32   + S3_ROWS * I_DIM * 4;
  constexpr unsigned UP_F32   = GATE_F32 + S3_ROWS * I_DIM * 4;
  constexpr unsigned TMP_F32  = UP_F32   + S3_ROWS * I_DIM * 4;
  constexpr unsigned WSG_F32  = TMP_F32  + S3_ROWS * I_DIM * 4;
  constexpr unsigned WSU_F32  = WSG_F32  + I_DIM * 4;
  constexpr unsigned XS_F32   = WSU_F32  + I_DIM * 4;
  constexpr unsigned SWG_F16  = XS_F32   + S3_ROWS * 4;
  constexpr unsigned IQ_BASE  = SWG_F16  + S3_ROWS * I_DIM * 2;
  constexpr unsigned RMAX_CM  = IQ_BASE  + S3_ROWS * (I_DIM / 2);
  constexpr unsigned RMIN_CM  = RMAX_CM  + S3_ROWS * 2;
  constexpr unsigned IS_OUT   = RMIN_CM  + S3_ROWS * 2;

  TI32   gI32; TASSIGN(gI32, GATE_I32);
  TI32   uI32; TASSIGN(uI32, UP_I32);
  TF32   gF32; TASSIGN(gF32, GATE_F32);
  TF32   uF32; TASSIGN(uF32, UP_F32);
  TF32   tmp;  TASSIGN(tmp,  TMP_F32);
  TWsRow wsg;  TASSIGN(wsg, WSG_F32);
  TWsRow wsu;  TASSIGN(wsu, WSU_F32);
  TXs    xs;   TASSIGN(xs,  XS_F32);
  TF16   swF16; TASSIGN(swF16, SWG_F16);
  TI4P   iq;   TASSIGN(iq,  IQ_BASE);
  TRMaxCm rmaxCm; TASSIGN(rmaxCm, RMAX_CM);
  TRMaxCm rminCm; TASSIGN(rminCm, RMIN_CM);
  TRMaxRm rmaxRm; TASSIGN(rmaxRm, RMAX_CM);
  TRMaxRm rminRm; TASSIGN(rminRm, RMIN_CM);
  TScaleR isOut;  TASSIGN(isOut, IS_OUT);

  using GmI32_2D = GlobalTensor<int32_t, TileShape2D<int32_t, DYNAMIC, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,DYNAMIC,1>, Layout::ND>;
  using GmF16_2D = GlobalTensor<half,    TileShape2D<half, DYNAMIC, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,DYNAMIC,1>, Layout::ND>;  // S3FP16: fp16 gate_up GM
  using GmF32_1D = GlobalTensor<float,   TileShape2D<float, 1, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,1,1>, Layout::ND>;
  using GmI8P_2D = GlobalTensor<int8_t,  TileShape2D<int8_t, DYNAMIC, DYNAMIC, Layout::ND>,
                                 Stride<1,1,1,DYNAMIC,1>, Layout::ND>;

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t e_end3 = (e_hi == 0) ? E : e_hi;

  for (uint32_t e = e_lo + vid; e < e_end3; e += num_cores) {
    int64_t t_end = gl_gm[e];
    int64_t t_start = (e > 0) ? gl_gm[e - 1] : 0;
    int32_t M_e = (int32_t)(t_end - t_start);
    if (M_e <= 0) continue;

    GmF32_1D wsgGm(ws_gm + (int64_t)e * 2 * I_DIM,
                    TileShape2D<float, 1, DYNAMIC, Layout::ND>(1, I_DIM));
    GmF32_1D wsuGm(ws_gm + (int64_t)e * 2 * I_DIM + I_DIM,
                    TileShape2D<float, 1, DYNAMIC, Layout::ND>(1, I_DIM));
    TLOAD(wsg, wsgGm);
    TLOAD(wsu, wsuGm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    constexpr int32_t S3_STEP = (I_DIM <= 128) ? 7 : (int32_t)S3_ROWS;
    for (int32_t mt = 0; mt < M_e; mt += S3_STEP) {
      const int32_t rows = (mt + S3_STEP <= M_e) ? S3_STEP : (M_e - mt);
      const int64_t t0 = t_start + mt;

      TileShape2D<int32_t, DYNAMIC, DYNAMIC, Layout::ND> gShape(rows, I_DIM);
      GmI32_2D gGm(gu_gm + t0 * N_GU,         gShape, Stride<1,1,1,DYNAMIC,1>(1,1,1, N_GU, 1));
      GmI32_2D uGm(gu_gm + t0 * N_GU + I_DIM, gShape, Stride<1,1,1,DYNAMIC,1>(1,1,1, N_GU, 1));
      TI32 gI32_dyn(rows); TASSIGN(gI32_dyn, GATE_I32);
      TI32 uI32_dyn(rows); TASSIGN(uI32_dyn, UP_I32);
      TLOAD(gI32_dyn, gGm);
      TLOAD(uI32_dyn, uGm);

      set_flag(PIPE_V, PIPE_S, EVENT_ID0); wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      TXs xs_dyn(rows); TASSIGN(xs_dyn, XS_F32);
      __ubuf__ float* xs_ub = ub_ptr<float>(xs_dyn);
      for (int32_t r = 0; r < rows; ++r) {
        xs_ub[r] = xs_gm[(t0 + r) * 32];
      }
      set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      TF32 gF32_dyn(rows); TASSIGN(gF32_dyn, GATE_F32);
      TF32 uF32_dyn(rows); TASSIGN(uF32_dyn, UP_F32);
      TCVT(gF32_dyn, gI32_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
      TCVT(uF32_dyn, uI32_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);

      TROWEXPANDMUL(gF32_dyn, gF32_dyn, xs_dyn); pipe_barrier(PIPE_V);
      TROWEXPANDMUL(uF32_dyn, uF32_dyn, xs_dyn); pipe_barrier(PIPE_V);

      TCOLEXPANDMUL(gF32_dyn, gF32_dyn, wsg); pipe_barrier(PIPE_V);
      TCOLEXPANDMUL(uF32_dyn, uF32_dyn, wsu); pipe_barrier(PIPE_V);

      TF32 tmp_dyn(rows); TASSIGN(tmp_dyn, TMP_F32);
      TMULS(tmp_dyn, gF32_dyn, -1.0f); pipe_barrier(PIPE_V);
      TEXP(tmp_dyn, tmp_dyn);          pipe_barrier(PIPE_V);
      TADDS(tmp_dyn, tmp_dyn, 1.0f);   pipe_barrier(PIPE_V);
      TDIV(gF32_dyn, gF32_dyn, tmp_dyn); pipe_barrier(PIPE_V);
      TMUL(gF32_dyn, gF32_dyn, uF32_dyn); pipe_barrier(PIPE_V);

      TF16 swF16_dyn(rows); TASSIGN(swF16_dyn, SWG_F16);
      TCVT(swF16_dyn, gF32_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);

      TF16 tmpRedF16(rows); TASSIGN(tmpRedF16, TMP_F32);
      TRMaxCm rmaxCm_dyn(rows); TASSIGN(rmaxCm_dyn, RMAX_CM);
      TRMaxCm rminCm_dyn(rows); TASSIGN(rminCm_dyn, RMIN_CM);
      TROWMAX(rmaxCm_dyn, swF16_dyn, tmpRedF16); pipe_barrier(PIPE_V);
      TROWMIN(rminCm_dyn, swF16_dyn, tmpRedF16); pipe_barrier(PIPE_V);
      TRMaxRm rmaxRm_dyn(rows); TASSIGN(rmaxRm_dyn, RMAX_CM);
      TRMaxRm rminRm_dyn(rows); TASSIGN(rminRm_dyn, RMIN_CM);
      TRESHAPE(rmaxRm_dyn, rmaxCm_dyn);
      TRESHAPE(rminRm_dyn, rminCm_dyn);
      pipe_barrier(PIPE_V);
      TMULS(rminRm_dyn, rminRm_dyn, (half)-1.0f); pipe_barrier(PIPE_V);
      TMAX(rmaxRm_dyn, rmaxRm_dyn, rminRm_dyn); pipe_barrier(PIPE_V);

      TScaleR isOut_dyn(rows); TASSIGN(isOut_dyn, IS_OUT);
      TCVT(isOut_dyn, rmaxRm_dyn, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
      TMULS(isOut_dyn, isOut_dyn, 1.0f / kScaleDivisor); pipe_barrier(PIPE_V);

      TRESHAPE(rmaxCm_dyn, rmaxRm_dyn);
      pipe_barrier(PIPE_V);
      TROWEXPANDDIV(swF16_dyn, swF16_dyn, rmaxCm_dyn); pipe_barrier(PIPE_V);
      TMULS(swF16_dyn, swF16_dyn, (half)kScaleDivisor); pipe_barrier(PIPE_V);

      TI4P iq_dyn(rows); TASSIGN(iq_dyn, IQ_BASE);
      fast_hadamard_int4::TCVT_FP16_TO_INT4_PACKED(iq_dyn, swF16_dyn, RoundMode::CAST_RINT);
      pipe_barrier(PIPE_V);

      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      TileShape2D<int8_t, DYNAMIC, DYNAMIC, Layout::ND> iqShape(rows, I_DIM / 2);
      GmI8P_2D iqGm(iq_gm + t0 * (I_DIM / 2), iqShape,
                     Stride<1,1,1,DYNAMIC,1>(1,1,1, I_DIM / 2, 1));
      TSTORE(iqGm, iq_dyn);

      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      __ubuf__ float* is_ub = ub_ptr<float>(isOut_dyn);
      for (int32_t r = 0; r < rows; ++r) {
        is_gm[(t0 + r) * 32] = is_ub[r];
      }
      set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
    }
  }
}
#endif  // MEGA_S3_DBUF

// ---- STAGE 5 SCATTER — chunk-by-expert atomic-add into y ----
//
// At prefill (m_per large) Stage 5 is the dominant vec stage (~752us at m_per=256
// in the per-stage microbench, 36% of the whole kernel): every expanded row loads
// H int32, casts, applies two scales, casts to fp16, and atomic-adds back. The
// ORIGINAL per-row loop fully serializes MTE2(load)/V(compute)/MTE3(store) — each
// row drains the pipe 5x (V->S scalar, S->V, MTE2->V, V->MTE3, MTE3->V) and reuses
// ONE set of UB tiles (dI/dF/y), so consecutive rows have a hard WAR dependency.
//
// MEGA_S5_DBUF: software-pipeline the row loop with TWO UB buffers (ping/pong on
// m&1, separate EVENT_IDs). While the vector engine computes row m, MTE2 prefetches
// row m+1's dI and MTE3 stores row m-1's y. The scalar metadata (orig_t / is*tw)
// for the next row is read one iteration ahead so the V<->S drain no longer gates
// the load. Per-row vector work, scales, atomic-add math are bit-identical to the
// serial version (cos preserved); only the pipe overlap changes.
//
// MEGA_S5_FP16 (LEVER 2): the down matmul (Stage 4) now folds the per-channel w2 dequant
// into the cube FIXPIPE (SetQuantVector) and emits HALF d_ws. So S5:
//   - loads fp16 d (HALF the bytes vs int32 -> half the MTE2 traffic, the S5 bottleneck),
//   - applies ONLY the per-token scalar (is * topk_w) (the per-channel w2_scale is already
//     baked into d by the cube), then atomic-adds. No ws load, no per-channel TMUL.
// d_gm is passed as int32_t* (uniform signature) but reinterpreted as half*. Same depth-2
// ping/pong pipeline as DBUF. w2s_gm is unused here (folded into the cube). cos preserved.
#if defined(MEGA_S5_FP16)
AICORE void stage5_scatter_combine_int4(
    __gm__ int32_t* d_gm_i32, __gm__ float* is_gm, __gm__ float* /*w2s_gm unused*/,
    __gm__ int32_t* sort_idx_gm, __gm__ half* topk_w_gm, __gm__ int64_t* gl_gm,
    __gm__ half* y_gm, uint32_t M_total, uint32_t E, uint32_t top_k, uint32_t T_orig,
    uint32_t e_lo = 0, uint32_t e_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  __gm__ half* d_gm = (__gm__ half*)d_gm_i32;   // cube emitted fp16 (w2_scale already applied)
  using TF16 = Tile<TileType::Vec, half, 1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  // Two fp16 in-buffers (d) + two fp16 out-buffers (y). int32->fp32 chain is gone; just a
  // scalar multiply by (is*tw) in fp16.
  TF16 d0; TASSIGN(d0, 0x0);              TF16 d1; TASSIGN(d1, H_DIM * 2);
  TF16 y0; TASSIGN(y0, H_DIM * 4);        TF16 y1; TASSIGN(y1, H_DIM * 6);
  using GmF16 = GlobalTensor<half, TileShape2D<half, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  const TileShape2D<half, 1, DYNAMIC, Layout::ND> f16sh(1, H_DIM);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t e_end = (e_hi == 0) ? E : e_hi;

  // Event-id budget: ID0/1 MTE2->V d load-ready; ID2/3 V->MTE2 d-free (reload WAR);
  // ID4/5 MTE3->V y-free (next compute WAR on y vs prior store).
#define S5F_LOAD(B, MROW)                                                    \
  do {                                                                       \
    int32_t _flat = sort_idx_gm[(MROW)];                                     \
    ld_orig##B = (uint32_t)_flat / top_k;                                    \
    ld_sc##B   = (half)(is_gm[(MROW) * 32] * (float)topk_w_gm[_flat]);       \
    GmF16 _dg(d_gm + (int64_t)(MROW) * H_DIM, f16sh);                        \
    TLOAD(d##B, _dg);                                                        \
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                               \
  } while (0)
#define S5F_COMPUTE(B, NEEDFREE)                                             \
  do {                                                                       \
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                              \
    if (NEEDFREE) wait_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B))); \
    TMULS(y##B, d##B, ld_sc##B); pipe_barrier(PIPE_V);                       \
    set_flag(PIPE_V, PIPE_MTE2, (event_t)(EVENT_ID2 + (B)));                \
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); \
    GmF16 _yg(y_gm + (int64_t)ld_orig##B * H_DIM, f16sh);                    \
    set_atomic_f16(); set_atomic_add();                                      \
    TSTORE(_yg, y##B);                                                       \
    set_atomic_none();                                                       \
    set_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B)));                \
  } while (0)

#ifdef MEGA_S5_ROWPART
  // ROW-partition (skew-immune): split this chunk's rows [r_lo,r_hi) evenly across the vec
  // cores instead of by expert. S5 has NO per-expert state (the per-channel w2 dequant is
  // folded into the cube FIXPIPE), so a row needs no expert lookup — each core streams a
  // contiguous row band. Under real prefill routing skew (one expert ~15-20x the mean) the
  // per-expert loop made the hot-expert core a straggler that stalled the all-core barrier;
  // row-partition removes the straggler. Same total work + same contiguous d/y access.
  {
    const int64_t r_lo = (e_lo > 0) ? gl_gm[e_lo - 1] : 0;
    const int64_t r_hi = gl_gm[e_end - 1];
    const int64_t rtotal = r_hi - r_lo;
    if (rtotal <= 0) return;
    const int64_t rchunk = (rtotal + (int64_t)num_cores - 1) / (int64_t)num_cores;
    const int64_t t_start = r_lo + (int64_t)vid * rchunk;
    if (t_start >= r_hi) return;
    const int64_t my_hi = (t_start + rchunk < r_hi) ? (t_start + rchunk) : r_hi;
    const int64_t n_rows = my_hi - t_start;
#else
  for (uint32_t e = e_lo + vid; e < e_end; e += num_cores) {
    const int64_t t_end = gl_gm[e];
    const int64_t t_start = (e > 0) ? gl_gm[e - 1] : 0;
    if (t_end <= t_start) continue;
    const int64_t n_rows = t_end - t_start;
#endif
    uint32_t ld_orig0 = 0, ld_orig1 = 0;
    half     ld_sc0 = (half)0.0f, ld_sc1 = (half)0.0f;

    if (n_rows < 4) {
      for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t mr = t_start + r;
        int32_t _flat = sort_idx_gm[mr];
        ld_orig0 = (uint32_t)_flat / top_k;
        ld_sc0   = (half)(is_gm[mr * 32] * (float)topk_w_gm[_flat]);
        GmF16 _dg(d_gm + (int64_t)mr * H_DIM, f16sh);
        TLOAD(d0, _dg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TMULS(y0, d0, ld_sc0); pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmF16 _yg(y_gm + (int64_t)ld_orig0 * H_DIM, f16sh);
        set_atomic_f16(); set_atomic_add();
        TSTORE(_yg, y0);
        set_atomic_none();
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
    } else {

    S5F_LOAD(0, t_start);
    int64_t set_dif[2] = {0, 0};
    int64_t set_buf[2] = {0, 0};
    for (int64_t r = 0; r < n_rows; ++r) {
      const bool cb0 = ((r & 1) == 0);
      if (r + 1 < n_rows) {
        const int64_t mi = t_start + r + 1;
        if (cb0) { if (set_dif[1]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3); set_dif[1]=0; } S5F_LOAD(1, mi); }
        else     { if (set_dif[0]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2); set_dif[0]=0; } S5F_LOAD(0, mi); }
      }
      if (cb0) { S5F_COMPUTE(0, (set_buf[0] != 0)); set_buf[0]=1; set_dif[0]=1; }
      else     { S5F_COMPUTE(1, (set_buf[1] != 0)); set_buf[1]=1; set_dif[1]=1; }
    }
    if (set_dif[0]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID2);
    if (set_dif[1]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID3);
    if (set_buf[0]) wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID4);
    if (set_buf[1]) wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID5);
    }   // close the n_rows>=4 else block (row-partition or expert-loop body)
  }
#undef S5F_LOAD
#undef S5F_COMPUTE
}
#elif defined(MEGA_S5_DBUF)
AICORE void stage5_scatter_combine_int4(
    __gm__ int32_t* d_gm, __gm__ float* is_gm, __gm__ float* w2s_gm,
    __gm__ int32_t* sort_idx_gm, __gm__ half* topk_w_gm, __gm__ int64_t* gl_gm,
    __gm__ half* y_gm, uint32_t M_total, uint32_t E, uint32_t top_k, uint32_t T_orig,
    uint32_t e_lo = 0, uint32_t e_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  using TI32 = Tile<TileType::Vec, int32_t, 1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  using TF32 = Tile<TileType::Vec, float,   1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  using TF16 = Tile<TileType::Vec, half,    1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  // UB layout (per buffer b in {0,1}):  dI[b] int32 | dF[b]+y[b] f32/f16 share | ws shared.
  //   dI0 @ 0                       (H*4)
  //   dI1 @ H*4                     (H*4)
  //   dF0 @ H*8                     (H*4)  (y0 packed fp16 overwrites dF0 region after cast)
  //   dF1 @ H*12                    (H*4)
  //   y0  @ H*16                    (H*2)
  //   y1  @ H*18                    (H*2)
  //   ws  @ H*20                    (H*4)
  TI32 dI0; TASSIGN(dI0, 0x0);          TI32 dI1; TASSIGN(dI1, H_DIM * 4);
  TF32 dF0; TASSIGN(dF0, H_DIM * 8);    TF32 dF1; TASSIGN(dF1, H_DIM * 12);
  TF16 y0;  TASSIGN(y0,  H_DIM * 16);   TF16 y1;  TASSIGN(y1,  H_DIM * 18);
  TF32 ws;  TASSIGN(ws,  H_DIM * 20);
  using GmI32 = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmF32 = GlobalTensor<float,   TileShape2D<float,   1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmF16 = GlobalTensor<half,    TileShape2D<half,    1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  const TileShape2D<int32_t, 1, DYNAMIC, Layout::ND> i32sh(1, H_DIM);
  const TileShape2D<float,   1, DYNAMIC, Layout::ND> f32sh(1, H_DIM);
  const TileShape2D<half,    1, DYNAMIC, Layout::ND> f16sh(1, H_DIM);

  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t e_end = (e_hi == 0) ? E : e_hi;

  // Event-id budget (8 hw flags): per-buffer B in {0,1}:
  //   ID0/1  MTE2->V  load-ready (reused later in the SAME compute for V->MTE3 store)
  //   ID2/3  V->MTE2  dI-free  (prefetch reload waits this: WAR on dI vs the TCVT read)
  //   ID4/5  MTE3->V  buf-free (next compute waits this: WAR on y/dF vs the prior store)
  //   ID6    MTE2->V  ws load
  // CCE requires compile-time event ids / tile operands, so both buffers are spelled
  // out explicitly (a runtime-indexed array miscompiles the flags -> every-other-row
  // corruption). MROW is the expanded-slot index.
#define S5_LOAD(B, MROW)                                                     \
  do {                                                                       \
    int32_t _flat = sort_idx_gm[(MROW)];                                     \
    ld_orig##B = (uint32_t)_flat / top_k;                                    \
    ld_sc##B   = is_gm[(MROW) * 32] * (float)topk_w_gm[_flat];               \
    GmI32 _dg(d_gm + (int64_t)(MROW) * H_DIM, i32sh);                        \
    TLOAD(dI##B, _dg);                                                       \
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                               \
  } while (0)
  // S5_COMPUTE waits its dI load (RAW), the buf-free flag (WAR on y/dF vs prior
  // store, when this buffer was used before), then: TCVT dI->dF (after which dI is
  // free -> signal V->MTE2 dI-free), dequant chain, TCVT->y, synchronous atomic store,
  // signal buf-free. NEEDFREE picks whether to wait the buf-free flag (skip on the
  // first two rows where the buffer hasn't been stored yet).
#define S5_COMPUTE(B, NEEDFREE)                                              \
  do {                                                                       \
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID##B);                              \
    if (NEEDFREE) wait_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B))); \
    TCVT(dF##B, dI##B, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);          \
    set_flag(PIPE_V, PIPE_MTE2, (event_t)(EVENT_ID2 + (B)));                \
    TMULS(dF##B, dF##B, ld_sc##B); pipe_barrier(PIPE_V);                     \
    TMUL(dF##B, dF##B, ws); pipe_barrier(PIPE_V);                            \
    TCVT(y##B, dF##B, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);           \
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID##B); \
    GmF16 _yg(y_gm + (int64_t)ld_orig##B * H_DIM, f16sh);                    \
    set_atomic_f16(); set_atomic_add();                                      \
    TSTORE(_yg, y##B);                                                       \
    set_atomic_none();                                                       \
    set_flag(PIPE_MTE3, PIPE_V, (event_t)(EVENT_ID4 + (B)));                \
  } while (0)

  for (uint32_t e = e_lo + vid; e < e_end; e += num_cores) {
    const int64_t t_end = gl_gm[e];
    const int64_t t_start = (e > 0) ? gl_gm[e - 1] : 0;
    if (t_end <= t_start) continue;
    GmF32 wsg(w2s_gm + (int64_t)e * H_DIM, f32sh);
    TLOAD(ws, wsg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID6); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID6);

    const int64_t n_rows = t_end - t_start;
    uint32_t ld_orig0 = 0, ld_orig1 = 0;
    float    ld_sc0 = 0.0f, ld_sc1 = 0.0f;

    // Runtime gate: at decode each core sees only 0-few rows per expert, where the
    // pipeline prologue/drain costs more than the overlap saves. Below S5_DBUF_MIN
    // rows fall back to the simple serial inner loop (buffer 0 only, no prefetch).
    if (n_rows < 4) {
      for (int64_t r = 0; r < n_rows; ++r) {
        const int64_t mr = t_start + r;
        int32_t _flat = sort_idx_gm[mr];
        ld_orig0 = (uint32_t)_flat / top_k;
        ld_sc0   = is_gm[mr * 32] * (float)topk_w_gm[_flat];
        GmI32 _dg(d_gm + (int64_t)mr * H_DIM, i32sh);
        TLOAD(dI0, _dg);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TCVT(dF0, dI0, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
        TMULS(dF0, dF0, ld_sc0); pipe_barrier(PIPE_V);
        TMUL(dF0, dF0, ws); pipe_barrier(PIPE_V);
        TCVT(y0, dF0, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmF16 _yg(y_gm + (int64_t)ld_orig0 * H_DIM, f16sh);
        set_atomic_f16(); set_atomic_add();
        TSTORE(_yg, y0);
        set_atomic_none();
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      }
      continue;
    }

    // Depth-2 software pipeline. Invariant kept SIMPLE for deadlock-freedom: every
    // set_flag is matched 1:1 by a wait_flag. The buffer used at step i (0-based row r=i)
    // is buffer (r&1). PROLOGUE loads row 0. Each step: prefetch row r+1 into buffer
    // ((r+1)&1) (overlaps), then compute+store row r from buffer (r&1). dI-free is set by
    // the compute and waited by THIS buffer's NEXT load (2 rows later); buf-free likewise
    // by the NEXT compute of this buffer. At end we wait the (at most 2) trailing flags.
    // To keep set/wait balanced regardless of n_rows parity, the prefetch of the buffer
    // happens BEFORE its compute, and we only ever wait a flag that was set.
    S5_LOAD(0, t_start);
    int64_t set_dif[2] = {0, 0};   // dI-free flag pending count per buffer
    int64_t set_buf[2] = {0, 0};   // buf-free flag pending count per buffer
    for (int64_t r = 0; r < n_rows; ++r) {
      const bool cb0 = ((r & 1) == 0);
      // Prefetch row r+1 (other buffer). Wait that buffer's dI-free if one is pending
      // (it was set when that buffer was last computed, 2 rows back).
      if (r + 1 < n_rows) {
        const int64_t mi = t_start + r + 1;
        if (cb0) { if (set_dif[1]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3); set_dif[1]=0; } S5_LOAD(1, mi); }
        else     { if (set_dif[0]) { wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2); set_dif[0]=0; } S5_LOAD(0, mi); }
      }
      // Compute+store row r (buffer cb). Wait this buffer's buf-free if pending.
      if (cb0) { S5_COMPUTE(0, (set_buf[0] != 0)); set_buf[0]=1; set_dif[0]=1; }
      else     { S5_COMPUTE(1, (set_buf[1] != 0)); set_buf[1]=1; set_dif[1]=1; }
    }
    // Drain trailing flags: each compute set dI-free + buf-free; the last use of each
    // buffer left both unwaited. Wait exactly the pending ones (1:1 with the sets).
    if (set_dif[0]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID2);
    if (set_dif[1]) wait_flag(PIPE_V,   PIPE_MTE2, EVENT_ID3);
    if (set_buf[0]) wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID4);
    if (set_buf[1]) wait_flag(PIPE_MTE3, PIPE_V,   EVENT_ID5);
  }
#undef S5_LOAD
#undef S5_COMPUTE
}
#else
AICORE void stage5_scatter_combine_int4(
    __gm__ int32_t* d_gm, __gm__ float* is_gm, __gm__ float* w2s_gm,
    __gm__ int32_t* sort_idx_gm, __gm__ half* topk_w_gm, __gm__ int64_t* gl_gm,
    __gm__ half* y_gm, uint32_t M_total, uint32_t E, uint32_t top_k, uint32_t T_orig,
    uint32_t e_lo = 0, uint32_t e_hi = 0)
{
  set_mask_norm(); set_vector_mask(-1, -1);
  using TI32 = Tile<TileType::Vec, int32_t, 1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  using TF32 = Tile<TileType::Vec, float,   1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  using TF16 = Tile<TileType::Vec, half,    1, H_DIM, BLayout::RowMajor, 1, H_DIM>;
  TI32 dI; TASSIGN(dI, 0x0);
  TF32 dF; TASSIGN(dF, H_DIM * 4);
  TF32 ws; TASSIGN(ws, H_DIM * 8);
  TF16 y;  TASSIGN(y,  H_DIM * 12);
  using GmI32 = GlobalTensor<int32_t, TileShape2D<int32_t, 1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmF32 = GlobalTensor<float,   TileShape2D<float,   1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  using GmF16 = GlobalTensor<half,    TileShape2D<half,    1, DYNAMIC, Layout::ND>, Stride<1,1,1,1,1>, Layout::ND>;
  const TileShape2D<int32_t, 1, DYNAMIC, Layout::ND> i32sh(1, H_DIM);
  const TileShape2D<float,   1, DYNAMIC, Layout::ND> f32sh(1, H_DIM);
  const TileShape2D<half,    1, DYNAMIC, Layout::ND> f16sh(1, H_DIM);
  const uint32_t num_cores = get_block_num() * get_subblockdim();
  const uint32_t vid = get_block_idx() * get_subblockdim() + get_subblockid();
  const uint32_t e_end = (e_hi == 0) ? E : e_hi;
  for (uint32_t e = e_lo + vid; e < e_end; e += num_cores) {
    const int64_t t_end = gl_gm[e];
    const int64_t t_start = (e > 0) ? gl_gm[e - 1] : 0;
    if (t_end <= t_start) continue;
    GmF32 wsg(w2s_gm + (int64_t)e * H_DIM, f32sh);
    TLOAD(ws, wsg);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    for (int64_t m = t_start; m < t_end; ++m) {
      set_flag(PIPE_V, PIPE_S, EVENT_ID0); wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
      const int32_t flat = sort_idx_gm[m];
      const uint32_t orig_t = (uint32_t)flat / top_k;
      const float is = is_gm[m * 32];
      const float tw = (float)topk_w_gm[flat];
      set_flag(PIPE_S, PIPE_V, EVENT_ID0); wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
      GmI32 dgm(d_gm + m * H_DIM, i32sh);
      TLOAD(dI, dgm);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      TCVT(dF, dI, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
      TMULS(dF, dF, is * tw); pipe_barrier(PIPE_V);
      TMUL(dF, dF, ws); pipe_barrier(PIPE_V);
      TCVT(y, dF, RoundMode::CAST_NONE); pipe_barrier(PIPE_V);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0); wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      GmF16 yg(y_gm + (int64_t)orig_t * H_DIM, f16sh);
      set_atomic_f16(); set_atomic_add();
      TSTORE(yg, y);
      set_atomic_none();
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0); wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  }
}
#endif  // MEGA_S5_DBUF
#endif  // __DAV_C220_VEC__

// ====================================================================================
//  CUBE STAGES — AscendC MatmulImpl<int4b_t> grouped int4xint4 -> int32. CUBE-pass only.
//  Replaces the PTO cube_grouped_int4_mm of mega_kernel.cpp. ONE TPipe / ONE MatmulImpl
//  object drives BOTH gate_up (K=H_DIM,N=N_GU) and down (K=I_DIM,N=H_DIM); re-Init per
//  stage with that stage's TCubeTiling. Per-expert loop = the validated grouped pattern.
// ====================================================================================
#if defined(__DAV_C220_CUBE__)
using aT_h    = MatmulType<TPosition::GM, CubeFormat::ND, int4b_t>;
using bT_h    = MatmulType<TPosition::GM, CubeFormat::NZ, int4b_t>;
using cT_h    = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using biasT_h = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
// isVecND2NZ (4th arg): false => MTE2 does ND->NZ inline on AIC (no AIV cooperation).
// In this fused MIX kernel the AIV cores are BUSY running the PTO vec stages, so they
// cannot service a vec-driven ND2NZ; force the pure-AIC path. (MEGA_HYBRID_VEC_ND2NZ=1
// flips it back to the original true for A/B testing.)
// LEVER 4 (default 0=off): doMTE2Preload (GetMDLConfig arg3) — prefetch A(M-dir=1)/B(N-dir=2)
// weights in MTE2 gaps. MDL-only (we use MDL ✓), needs K fully loaded (depthA1=depthB1=nKa ✓)
// + M/N double-buffer (dbL0A/B=2 ✓). Candidate for our MTE2/weight-load-bound cube. A/B via
// -DMEGA_CUBE_MTE2PRELOAD=1|2 (CANN Matmul perf-tuning case 3.10.4.14, found via npu-coding-mcp).
#ifndef MEGA_CUBE_MTE2PRELOAD
#define MEGA_CUBE_MTE2PRELOAD 0
#endif
#ifdef MEGA_HYBRID_VEC_ND2NZ
constexpr MatmulConfig MM_CFG_H = GetMDLConfig(false, false, MEGA_CUBE_MTE2PRELOAD, true, false, false, true);
#elif defined(MEGA_HYBRID_NO_UNITFLAG)
// isVecND2NZ=false (AIC MTE2 inline) + enUnitFlag=false: explicit set/wait between MTE1/MAD
// instead of the hardware unit-flag. Unit-flag mode races on a cold AIC pipeline when AIV
// runs concurrent PTO -> "L0A read/write conflict in MTE" (507015).
constexpr MatmulConfig MM_CFG_H = GetMDLConfig(false, false, MEGA_CUBE_MTE2PRELOAD, false, false, false, false);
#else
constexpr MatmulConfig MM_CFG_H = GetMDLConfig(false, false, MEGA_CUBE_MTE2PRELOAD, false, false, false, true);
#endif
using MMImpl_h = MatmulImpl<aT_h, bT_h, cT_h, biasT_h, MM_CFG_H>;

#ifdef MEGA_S5_FP16
// LEVER 2: down matmul emits fp16 with the per-channel w2 dequant folded into the FIXPIPE
// (SetQuantVector / GetTensorC half), exactly like vendor grouped_matmul_a4w4. The cube int4
// accumulator (int32) is multiplied by the per-N uint64 dequant scale on drain -> half d_ws.
// Halves S5's load (fp16 vs int32) AND removes S5's per-channel TMUL. The per-TOKEN activation
// scale `is` and the routing weight `topk_w` are per-row, so they stay in S5 (cube can only
// fold the per-N-channel weight scale). cT = half (the only difference vs MMImpl_h).
using cT_dq   = MatmulType<TPosition::GM, CubeFormat::ND, half>;
using MMImpl_dq = MatmulImpl<aT_h, bT_h, cT_dq, biasT_h, MM_CFG_H>;

// Grouped int4 matmul with per-channel fp16 dequant on drain. Mirrors grouped_matmul_int4_impl
// but: C is half, and before each Iterate it binds the per-(expert,N-channel) uint64 dequant
// scale via SetQuantVector(scaleGm[g*N + tailN]). scale_gm is the uint64-packed w2 scale [E,N].
AICORE inline void grouped_matmul_int4_dequant_impl(
    MMImpl_dq &mm, const TCubeTiling* tilingPtr,
    __gm__ int8_t* a_gm, __gm__ int8_t* w_gm, __gm__ half* c_gm,
    __gm__ uint64_t* scale_gm, __gm__ int64_t* gl_gm,
    uint32_t E, uint32_t e_lo, uint32_t e_hi)
{
  const int32_t K_RT = tilingPtr->Ka;
  const int32_t N_RT = tilingPtr->N;
  const uint32_t coreNum = GetBlockNum();
  const uint32_t coreIdx = GetBlockIdx();
  const int32_t singleN = tilingPtr->singleCoreN;
  const uint32_t blockDimN = (N_RT + singleN - 1) / singleN;
  const uint32_t e_end = (e_hi == 0) ? E : e_hi;

  GlobalTensor<int4b_t> xGm;  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(a_gm));
  GlobalTensor<int4b_t> wGm;  wGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(w_gm));
  GlobalTensor<half>     yGm; yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(c_gm));
  GlobalTensor<uint64_t> sGm; sGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(scale_gm));

  int32_t preOffset = (e_lo > 0) ? (int32_t)gl_gm[e_lo - 1] : 0;
  uint32_t offsetM = (uint32_t)preOffset;
  uint32_t globalBlk = 0;

  for (uint32_t g = e_lo; g < e_end; ++g) {
    int32_t cum = static_cast<int32_t>(gl_gm[g]);
    int32_t m = cum - preOffset;
    preOffset = cum;
    if (m <= 0) continue;

    const uint32_t blockDimM = (static_cast<uint32_t>(m) + tilingPtr->singleCoreM - 1) /
                               tilingPtr->singleCoreM;
    mm.SetOrgShape(m, N_RT, K_RT);

    for (uint32_t mIdx = 0; mIdx < blockDimM; ++mIdx) {
      for (uint32_t nIdx = 0; nIdx < blockDimN; ++nIdx, ++globalBlk) {
        if ((globalBlk % coreNum) != coreIdx) continue;
        uint32_t curSingleM = (mIdx == blockDimM - 1)
            ? (static_cast<uint32_t>(m) - mIdx * tilingPtr->singleCoreM)
            : tilingPtr->singleCoreM;
        uint32_t tailN = nIdx * singleN;
        uint32_t curSingleN = (nIdx == blockDimN - 1) ? (N_RT - tailN) : singleN;
        uint64_t xOffset = static_cast<uint64_t>(offsetM + mIdx * tilingPtr->singleCoreM) * K_RT;
        uint64_t wOffset = static_cast<uint64_t>(g) * N_RT * K_RT + static_cast<uint64_t>(tailN) * K_RT;
        uint64_t yOffset = static_cast<uint64_t>(offsetM + mIdx * tilingPtr->singleCoreM) * N_RT + tailN;
        uint64_t sOffset = static_cast<uint64_t>(g) * N_RT + tailN;
        mm.SetSingleShape(curSingleM, curSingleN, K_RT);
        mm.SetTensorA(xGm[xOffset]);
        mm.SetTensorB(wGm[wOffset]);
        mm.SetQuantVector(sGm[sOffset]);
        mm.Iterate();
        mm.GetTensorC(yGm[yOffset], 0, false);
      }
    }
    offsetM += static_cast<uint32_t>(m);
  }
}
#endif  // MEGA_S5_FP16

#ifdef MEGA_CUBE_HADAMARD
// ====================== STAGE 0 — block-diag Hadamard as fp16 PTO cube GEMM ======================
// Stage B: in-kernel block-diag Hadamard so the kernel is fully fused (no Python H). Ported
// verbatim from mega_kernel.cpp::stage0_hadamard, with every PTO TYPE name qualified pto::
// (the PTO macros TLOAD/TEXTRACT/TMATMUL/TSTORE/TASSIGN are unscoped, so unaffected). This
// PTO fp16 TMATMUL runs on the cube BEFORE the AscendC MatmulImpl Init (option a): the TPipe
// is constructed first, then S0 uses raw L0/L1 tiles, then mm.Init() lays out its own L1/L0.
// COEXISTENCE VALIDATED: S0's raw-tile fp16 TMATMUL + the int4 MatmulImpl share one TPipe
// sequentially (FFTS-separated) without clobbering each other.
//
// h = x @ B1, B1 block-diagonal: h[:, b*64:(b+1)*64] = x[:, b*64:+64] @ B1_block[b].
// B1_block[b] = normalized Hadamard-64 (H/sqrt(64)) so Stage 1 quant uses kInvSqrtN=1.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat_h = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                          RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatB_h = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                           RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;

AICORE void stage0_hadamard(__gm__ half* x_gm, __gm__ half* b1_gm, __gm__ half* xrot_gm,
                            uint32_t T_orig) {
  constexpr uint32_t HN0 = 64;
  constexpr uint32_t NB0 = H_DIM / HN0;        // 32 blocks for H=2048
  constexpr uint32_t M_T0 = 128;
  const uint32_t cid = get_block_idx();
  const uint32_t num_cores = get_block_num();
  const uint32_t M_TILES = (T_orig + M_T0 - 1) / M_T0;
  const uint32_t total = M_TILES * NB0;
  // EVENT_ID4: stage0 uses raw PTO set_flag/wait_flag on the (M,MTE1)/(MTE1,M)/(FIX,M)/(MTE1,MTE2)
  // pipe-pairs. The AscendC MatmulImpl that runs AFTER stage0 drives its L0A/L0B ping-pong buffers
  // with HardEvent M_MTE1 / MTE1_M on event IDs 0 and 1 (TBufPoolL0Base::Allocate/EnQue/DeQue/Free
  // use l0PingPongFlag_ in {0,1}; the UB TBufPoolL0 AllocEventID<M_MTE1> also returns 0,1 first).
  // stage0 originally used EVENT_ID1 here -> COLLISION on the (M,MTE1) id=1 hardware flag register:
  // stage0's epilogue leaves a residual M_MTE1(1) count that MatmulImpl's first pong-iteration
  // WaitFlag<M_MTE1>(1) consumes early -> cube reads L0A before MTE1 finishes loading ->
  // 507015 "L0A read/write conflict in MTE" (~70% cold-start). Stage A (no stage0) is 0/24; the
  // fault appears ONLY when stage0 precedes the matmul -> event-ID aliasing. Use a high event id
  // (>= the matmul's max) so the two never share a (pipe-pair,id) register.
  auto we = EVENT_ID4;

  pto::TileLeft<half,  M_T0, HN0, M_T0, HN0> a_l0; TASSIGN(a_l0, 0x0);
  pto::TileRight<half, HN0,  HN0, HN0,  HN0> b_l0; TASSIGN(b_l0, 0x0);
  pto::TileAcc<float,  M_T0, HN0, M_T0, HN0> c_l0; TASSIGN(c_l0, 0x0);

  set_flag(PIPE_MTE1, PIPE_MTE2, we);
  set_flag(PIPE_M,    PIPE_MTE1, we);
  set_flag(PIPE_FIX,  PIPE_M,    we);

  for (uint32_t pair = cid; pair < total; pair += num_cores) {
    const uint32_t mt = pair / NB0;
    const uint32_t b  = pair % NB0;
    const uint32_t m_base = mt * M_T0;
    const uint32_t k_base = b * HN0;
    const int32_t  M_e = (m_base + M_T0 <= T_orig) ? (int32_t)M_T0 : (int32_t)(T_orig - m_base);
    if (M_e <= 0) continue;

    wait_flag(PIPE_MTE1, PIPE_MTE2, we);
    L1Mat_h<half, M_T0, HN0, M_T0, HN0> a_l1; TASSIGN(a_l1, 0x0);
    pto::Shape<1,1,1,pto::DYNAMIC,HN0> aShape; aShape.shape[3] = M_e;
    pto::GlobalTensor<half, decltype(aShape), pto::Stride<1,1,1,H_DIM,1>>
        aGm(x_gm + (int64_t)m_base * H_DIM + k_base, aShape);
    TLOAD(a_l1, aGm);
    L1MatB_h<half, HN0, HN0, HN0, HN0> b_l1; TASSIGN(b_l1, M_T0 * HN0 * sizeof(half));
    pto::GlobalTensor<half, pto::TileShape2D<half, HN0, HN0, pto::Layout::DN>,
                      pto::Stride<1,1,1,1,pto::DYNAMIC>, pto::Layout::DN>
        bGm(b1_gm + (int64_t)b * HN0 * HN0, pto::TileShape2D<half, HN0, HN0, pto::Layout::DN>{},
            pto::Stride<1,1,1,1,pto::DYNAMIC>(1,1,1,1,(int32_t)HN0));
    TLOAD(b_l1, bGm);
    set_flag(PIPE_MTE2, PIPE_MTE1, we);

    wait_flag(PIPE_MTE2, PIPE_MTE1, we);
    wait_flag(PIPE_M,    PIPE_MTE1, we);
    TEXTRACT(a_l0, a_l1, 0, 0);
    TEXTRACT(b_l0, b_l1, 0, 0);
    set_flag(PIPE_MTE1, PIPE_MTE2, we);
    set_flag(PIPE_MTE1, PIPE_M,    we);

    wait_flag(PIPE_MTE1, PIPE_M, we);
    wait_flag(PIPE_FIX,  PIPE_M, we);
    TMATMUL(c_l0, a_l0, b_l0);
    set_flag(PIPE_M, PIPE_MTE1, we);
    set_flag(PIPE_M, PIPE_FIX,  we);

    wait_flag(PIPE_M, PIPE_FIX, we);
    pto::TileAcc<float, M_T0, HN0, pto::DYNAMIC, HN0> c_view(M_e); TASSIGN(c_view, 0x0);
    pto::Shape<1,1,1,pto::DYNAMIC,HN0> cShape; cShape.shape[3] = M_e;
    pto::GlobalTensor<half, decltype(cShape), pto::Stride<1,1,1,H_DIM,1>>
        cGm(xrot_gm + (int64_t)m_base * H_DIM + k_base, cShape,
            pto::Stride<1,1,1,H_DIM,1>(1,1,1,(int32_t)H_DIM,1));
    TSTORE(cGm, c_view);
    set_flag(PIPE_FIX, PIPE_M, we);
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, we);
  wait_flag(PIPE_M,    PIPE_MTE1, we);
  wait_flag(PIPE_FIX,  PIPE_M,    we);
}
#endif  // MEGA_CUBE_HADAMARD

// Grouped int4 matmul over experts [e_lo, e_hi), single Init by caller. Mirrors
// mm_impl_int4_grouped: flat (m_tile, n_tile) round-robin across cores, FRACTAL_NZ B
// (per-expert contiguous block of K*N int4), int32 ND C.
//   a_gm:  [M_total, K/2]      int4 packed ND   (xq_ws / iq_ws)
//   w_gm:  [E, N/64, K/16, 16, 64] int4 packed NZ  (weight)
//   c_gm:  [M_total, N]        int32 ND          (gu_ws / d_ws)
//   gl_gm: [E]                 int64 cumulative row counts
AICORE inline void grouped_matmul_int4_impl(
    MMImpl_h &mm, const TCubeTiling* tilingPtr,
    __gm__ int8_t* a_gm, __gm__ int8_t* w_gm, __gm__ int32_t* c_gm, __gm__ int64_t* gl_gm,
    uint32_t E, uint32_t e_lo, uint32_t e_hi)
{
  const int32_t K_RT = tilingPtr->Ka;
  const int32_t N_RT = tilingPtr->N;
  const uint32_t coreNum = GetBlockNum();
  const uint32_t coreIdx = GetBlockIdx();
  const int32_t singleN = tilingPtr->singleCoreN;
  const uint32_t blockDimN = (N_RT + singleN - 1) / singleN;
  const uint32_t e_end = (e_hi == 0) ? E : e_hi;

  GlobalTensor<int4b_t> xGm;  xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(a_gm));
  GlobalTensor<int4b_t> wGm;  wGm.SetGlobalBuffer(reinterpret_cast<__gm__ int4b_t*>(w_gm));
  GlobalTensor<int32_t>  yGm; yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(c_gm));

  // offsetM (global row base of expert e) = gl_gm[e-1]. preOffset tracks cum row count.
  int32_t preOffset = (e_lo > 0) ? (int32_t)gl_gm[e_lo - 1] : 0;
  uint32_t offsetM = (uint32_t)preOffset;
  uint32_t globalBlk = 0;

  for (uint32_t g = e_lo; g < e_end; ++g) {
    int32_t cum = static_cast<int32_t>(gl_gm[g]);
    int32_t m = cum - preOffset;
    preOffset = cum;
    if (m <= 0) continue;

    const uint32_t blockDimM = (static_cast<uint32_t>(m) + tilingPtr->singleCoreM - 1) /
                               tilingPtr->singleCoreM;
    mm.SetOrgShape(m, N_RT, K_RT);

    for (uint32_t mIdx = 0; mIdx < blockDimM; ++mIdx) {
      for (uint32_t nIdx = 0; nIdx < blockDimN; ++nIdx, ++globalBlk) {
        if ((globalBlk % coreNum) != coreIdx) continue;
        uint32_t curSingleM = (mIdx == blockDimM - 1)
            ? (static_cast<uint32_t>(m) - mIdx * tilingPtr->singleCoreM)
            : tilingPtr->singleCoreM;
        uint32_t tailN = nIdx * singleN;
        uint32_t curSingleN = (nIdx == blockDimN - 1) ? (N_RT - tailN) : singleN;
        uint64_t xOffset = static_cast<uint64_t>(offsetM + mIdx * tilingPtr->singleCoreM) * K_RT;
        uint64_t wOffset = static_cast<uint64_t>(g) * N_RT * K_RT + static_cast<uint64_t>(tailN) * K_RT;
        uint64_t yOffset = static_cast<uint64_t>(offsetM + mIdx * tilingPtr->singleCoreM) * N_RT + tailN;
        mm.SetSingleShape(curSingleM, curSingleN, K_RT);
        mm.SetTensorA(xGm[xOffset]);
        mm.SetTensorB(wGm[wOffset]);
        mm.Iterate();
        mm.GetTensorC(yGm[yOffset], 0, false);
      }
    }
    offsetM += static_cast<uint32_t>(m);
  }
}
#endif  // __DAV_C220_CUBE__

// ====================================================================================
//  HYBRID MEGA KERNEL — full W4A4 MoE chain, single launch, SAFESYNC overlap schedule.
//  Stage A: NO in-kernel Stage 0 (Python pre-applies the block-diag Hadamard). Cube branch
//  does ONLY the two int4 matmuls. Vec branch runs Stage 1/3/5. B0..B5 barrier sequence
//  matches mega_kernel.cpp (without the MEGA_CUBE_HADAMARD publish barrier).
//
//  Two tiling structs: tiling_gu (gate_up K=H_DIM,N=N_GU) and tiling_dn (down K=I_DIM,N=H_DIM).
// ====================================================================================
__global__ AICORE void mega_kernel_hybrid(
    __gm__ void* x_gm,
    __gm__ void* w13_gm, __gm__ void* w13_scale_gm,
    __gm__ void* w2_gm,  __gm__ void* w2_scale_gm,
    __gm__ void* group_list_gm,
    __gm__ void* eri_gm,
    __gm__ void* sort_idx_gm,
    __gm__ void* topk_w_gm,
    __gm__ void* xq_ws, __gm__ void* xs_ws, __gm__ void* gu_ws,
    __gm__ void* iq_ws, __gm__ void* is_ws, __gm__ void* d_ws,
    __gm__ void* y_gm,
    __gm__ void* tiling_gu_gm,   // TCubeTiling for gate_up (K=H_DIM, N=N_GU)
    __gm__ void* tiling_dn_gm,   // TCubeTiling for down    (K=I_DIM, N=H_DIM)
    __gm__ void* b1_gm,          // [NB,64,64] fp16 normalized H-64 blocks (Stage B cube-Hadamard)
    __gm__ void* xrot_ws,        // [T_orig, H] fp16 rotated-x workspace (Stage B)
    uint32_t M_total, uint32_t E, uint32_t top_k, uint32_t T_orig, uint64_t ffts_addr)
{
  set_ffts_base_addr(ffts_addr);

  const uint32_t eA = E / 2;   // SAFESYNC NC=2 expert-chunk boundary
  __gm__ int64_t* gl_ov = (__gm__ int64_t*)group_list_gm;

#if defined(__DAV_C220_VEC__)
#ifdef MEGA_CUBE_HADAMARD
  __gm__ half* s1x = (__gm__ half*)xrot_ws;  // Stage B: in-kernel cube-rotated x
#else
  __gm__ half* s1x = (__gm__ half*)x_gm;     // Stage A: Python pre-rotated x (pyhadamard)
#endif

#ifdef MEGA_CUBE_HADAMARD
  HxSyncAllImpl<false>();   // wait for cube Stage 0 -> xrot_ws
#endif
  HxSyncAllImpl<false>();                                                   // B0
  { pipe_barrier(PIPE_ALL); int64_t th = gl_ov[eA - 1]; pipe_barrier(PIPE_ALL);
    stage1_int4_routed(s1x, (__gm__ int32_t*)sort_idx_gm, (__gm__ int8_t*)xq_ws,
                       (__gm__ float*)xs_ws, M_total, top_k, 0u, (uint32_t)th); }
  HxSyncAllImpl<false>();                                                   // B1
  { pipe_barrier(PIPE_ALL); int64_t tl = gl_ov[eA-1], th = gl_ov[E-1]; pipe_barrier(PIPE_ALL);
    stage1_int4_routed(s1x, (__gm__ int32_t*)sort_idx_gm, (__gm__ int8_t*)xq_ws,
                       (__gm__ float*)xs_ws, M_total, top_k, (uint32_t)tl, (uint32_t)th); }
  HxSyncAllImpl<false>();                                                   // B2
#if MEGA_STOP_AFTER_N >= 3
  stage3_int4_swiglu_quant_grouped_v2((__gm__ half*)gu_ws, (__gm__ float*)xs_ws,
      (__gm__ float*)w13_scale_gm, gl_ov, (__gm__ int8_t*)iq_ws, (__gm__ float*)is_ws, M_total, E, 0u, eA);
#endif
  HxSyncAllImpl<false>();                                                   // B3
#if MEGA_STOP_AFTER_N >= 3
  stage3_int4_swiglu_quant_grouped_v2((__gm__ half*)gu_ws, (__gm__ float*)xs_ws,
      (__gm__ float*)w13_scale_gm, gl_ov, (__gm__ int8_t*)iq_ws, (__gm__ float*)is_ws, M_total, E, eA, E);
#endif
  HxSyncAllImpl<false>();                                                   // B4 (cube S4(0) done)
#if MEGA_STOP_AFTER_N >= 5
  stage5_scatter_combine_int4((__gm__ int32_t*)d_ws, (__gm__ float*)is_ws, (__gm__ float*)w2_scale_gm,
      (__gm__ int32_t*)sort_idx_gm, (__gm__ half*)topk_w_gm, gl_ov, (__gm__ half*)y_gm, M_total, E, top_k, T_orig, 0u, eA);
#endif
  HxSyncAllImpl<false>();                                                   // B5 (cube S4(1) done)
#if MEGA_STOP_AFTER_N >= 5
  stage5_scatter_combine_int4((__gm__ int32_t*)d_ws, (__gm__ float*)is_ws, (__gm__ float*)w2_scale_gm,
      (__gm__ int32_t*)sort_idx_gm, (__gm__ half*)topk_w_gm, gl_ov, (__gm__ half*)y_gm, M_total, E, top_k, T_orig, eA, E);
#endif

#elif defined(__DAV_C220_CUBE__)
#if MEGA_STOP_AFTER_N >= 2
  TPipe pipe;
  AscendCUtils::SetOverflow(1);

  // Load both tiling structs (GM -> local).
  TCubeTiling tiling_gu, tiling_dn;
  {
    __gm__ int32_t* src = reinterpret_cast<__gm__ int32_t*>(tiling_gu_gm);
    int32_t* dst = reinterpret_cast<int32_t*>(&tiling_gu);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i) dst[i] = src[i];
  }
  {
    __gm__ int32_t* src = reinterpret_cast<__gm__ int32_t*>(tiling_dn_gm);
    int32_t* dst = reinterpret_cast<int32_t*>(&tiling_dn);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i) dst[i] = src[i];
  }

  MMImpl_h mm;
  mm.SetSubBlockIdx(0);
#ifdef MEGA_S5_FP16
  // LEVER 2: separate fp16-output MatmulImpl for the down stage (per-channel w2 dequant
  // folded into the FIXPIPE). gate_up still uses int32 `mm` (S3 needs the raw int32 acc).
  MMImpl_dq mm_dq;
  mm_dq.SetSubBlockIdx(0);
#endif
#endif

#ifdef MEGA_CUBE_HADAMARD
  // Stage B: PTO fp16 block-diag Hadamard FIRST (raw L0/L1 tiles, EVENT_ID4), then publish
  // xrot_ws to vec. The AscendC MatmulImpl is Init'd only AFTER S0 finishes (option a).
  // NOTE: stage0 uses EVENT_ID4 (not 1) to avoid aliasing the matmul's L0 ping-pong M_MTE1
  // flags on ids 0/1 — see the comment in stage0_hadamard. That aliasing was the cold-start
  // 507015 "L0A read/write conflict in MTE" (Stage A with no stage0 was 0/24; Stage B 17/24).
  stage0_hadamard((__gm__ half*)x_gm, (__gm__ half*)b1_gm, (__gm__ half*)xrot_ws, T_orig);
  HxSyncAllImpl<false>();   // publish xrot_ws to vec
#endif

  // Cube schedule is SKEWED two barriers behind vec (matches mega_kernel.cpp): cube idles
  // at B0/B1 while vec does both Stage-1 chunks, so gate_up reads xq_ws AFTER it is fully
  // written. gate_up at B1->B2 overlaps vec Stage-3(chunk0) etc.
  //   B0 idle | B1 idle | gate_up(c0) [B1..B2] | gate_up(c1) [B2..B3] |
  //   down(c0) [B3..B4] | down(c1) [B4..B5]
  HxSyncAllImpl<false>();                                                   // B0
#if MEGA_STOP_AFTER_N >= 2
  mm_dq.Init(&tiling_gu, &pipe);   // S3FP16: gate_up emits fp16 via the dequant cube (per-channel w13 scale folded in)
#endif
  HxSyncAllImpl<false>();                                                   // B1
#if MEGA_STOP_AFTER_N >= 2
  grouped_matmul_int4_dequant_impl(mm_dq, &tiling_gu, (__gm__ int8_t*)xq_ws, (__gm__ int8_t*)w13_gm,
                           (__gm__ half*)gu_ws, (__gm__ uint64_t*)w13_scale_gm, gl_ov, E, 0u, eA);
#endif
  HxSyncAllImpl<false>();                                                   // B2
#if MEGA_STOP_AFTER_N >= 2
  grouped_matmul_int4_dequant_impl(mm_dq, &tiling_gu, (__gm__ int8_t*)xq_ws, (__gm__ int8_t*)w13_gm,
                           (__gm__ half*)gu_ws, (__gm__ uint64_t*)w13_scale_gm, gl_ov, E, eA, E);
  // Drain the MatmulImpl pipeline (L0A/L0B prefetch + unit-flag state) BEFORE the
  // re-Init for the down matmul. Without this, dbL0A=2 prefetch of gate_up can still be
  // outstanding when Init(tiling_dn) re-lays-out the L0 buffers -> intermittent
  // "L0A read/write conflict in MTE" (507015) on cold launches. mm.End() = Scheduler::End()
  // = CopyCubeInA/B Destroy + TBufPoolL0 ResetCache.
  mm_dq.End();   // S3FP16: gate_up ran on mm_dq (fp16 dequant cube)
  pipe_barrier(PIPE_ALL);
#endif
  HxSyncAllImpl<false>();                                                   // B3
#if MEGA_STOP_AFTER_N >= 4
#ifdef MEGA_S5_FP16
  // Down matmul emits fp16 with the per-channel w2 dequant (uint64 scale = w2_scale_gm)
  // folded into the FIXPIPE. d_ws is half. The per-token `is`/topk_w stay in S5.
  mm_dq.Init(&tiling_dn, &pipe);
  grouped_matmul_int4_dequant_impl(mm_dq, &tiling_dn, (__gm__ int8_t*)iq_ws, (__gm__ int8_t*)w2_gm,
                           (__gm__ half*)d_ws, (__gm__ uint64_t*)w2_scale_gm, gl_ov, E, 0u, eA);
#else
  mm.Init(&tiling_dn, &pipe);
  grouped_matmul_int4_impl(mm, &tiling_dn, (__gm__ int8_t*)iq_ws, (__gm__ int8_t*)w2_gm,
                           (__gm__ int32_t*)d_ws, gl_ov, E, 0u, eA);
#endif
#endif
  HxSyncAllImpl<false>();                                                   // B4
#if MEGA_STOP_AFTER_N >= 4
#ifdef MEGA_S5_FP16
  grouped_matmul_int4_dequant_impl(mm_dq, &tiling_dn, (__gm__ int8_t*)iq_ws, (__gm__ int8_t*)w2_gm,
                           (__gm__ half*)d_ws, (__gm__ uint64_t*)w2_scale_gm, gl_ov, E, eA, E);
  mm_dq.End();
#else
  grouped_matmul_int4_impl(mm, &tiling_dn, (__gm__ int8_t*)iq_ws, (__gm__ int8_t*)w2_gm,
                           (__gm__ int32_t*)d_ws, gl_ov, E, eA, E);
  mm.End();
#endif
  pipe_barrier(PIPE_ALL);
#endif
  HxSyncAllImpl<false>();                                                   // B5
#endif
}

// ---------------- host launcher ----------------
extern "C" void call_mega_kernel_hybrid(uint32_t block_dim, void* stream,
    uint8_t* x, uint8_t* w13, uint8_t* w13s,
    uint8_t* w2, uint8_t* w2s, uint8_t* group_list,
    uint8_t* eri, uint8_t* sort_idx, uint8_t* topk_w,
    uint8_t* xq_ws, uint8_t* xs_ws, uint8_t* gu_ws,
    uint8_t* iq_ws, uint8_t* is_ws, uint8_t* d_ws,
    uint8_t* y, uint8_t* tiling_gu, uint8_t* tiling_dn,
    uint8_t* b1, uint8_t* xrot_ws,
    uint32_t M_total, uint32_t E, uint32_t top_k, uint32_t T_orig)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  mega_kernel_hybrid<<<block_dim, nullptr, stream>>>(
      x, w13, w13s, w2, w2s, group_list, eri, sort_idx, topk_w,
      xq_ws, xs_ws, gu_ws, iq_ws, is_ws, d_ws, y, tiling_gu, tiling_dn,
      b1, xrot_ws, M_total, E, top_k, T_orig, fftsAddr);
}
