// ============================================================================
// scaled_dot_kkt_kernel.cpp — Intra-chunk attention matrix for GatedDeltaNet
//
// Computes A = mask(KK^T · gating_coeff) per chunk, where:
//   KK^T ∈ ℝ^{C×C} = K @ K^T                  (Cube engine, GEMM)
//   coeff[i,j] = exp(clamp(g[i]+log(β[i]) - g[j], max=0))  (Vec engine)
//   A[i,j] = KK^T[i,j] · coeff[i,j] · causal_mask[i,j]
//
// Inputs:
//   K       [total_tokens, Hg, D] half  — key vectors (BSND along seq; stride Hg * D)
//   Beta    [H, total_tokens]     half  — gate bias per **value** head (pre-transposed)
//   G       [H, total_tokens]     float — cumulative gate sum per **value** head
//   Msk     [C, C]                float — lower-triangular causal mask
//
// Output:
//   A       [total_tokens, H, C]  half  — gated attention matrix in BSND
//
// Architecture: Cube + Vec cross-core kernel.
//   Cube phase: K→L1, GEMM K@K^T→L0C, store to workspace (GM)
//   Vec phase:  load workspace KK^T, compute gating coefficients, apply mask
//
// Cross-core sync: Cube signals Vec via FFTS flag after each chunk's KK^T
// is written to workspace. Vec signals back when workspace buffer is free.
// Two workspace slots alternate (double-buffering via slot = ci & 1).
//
// Vec sub-blocks: Two sub-blocks (vid=0,1) process upper/lower halves of
// the C×C attention matrix in parallel (HalfChunk rows each).
//
// NPU memory hierarchy:
//   GM → L1 (Cube-accessible) → L0A/L0B (GEMM operands) → L0C (accumulator)
//   GM → UB (Vec-accessible SRAM)
//
// ── PTO / NPU Primer for This Kernel ──────────────────────────────────
// NPU Architecture (simplified):
//   Each "AI Core" (like a GPU SM) has:
//     - Cube engine: matrix multiply unit (like GPU Tensor Cores), works on L0A/L0B/L0C
//     - Vec engine: SIMD vector unit (like GPU CUDA cores), works on UB (Unified Buffer)
//     - MTE2: DMA engine for loading data: GM → L1 or GM → UB
//     - MTE3: DMA engine for storing data: UB → GM or L0C → GM
//     - MTE1: DMA engine for L1 → L0A/L0B transfers (internal to Cube pipeline)
//   Memory hierarchy (fast→slow): L0 registers > L1 cache > UB (SRAM) > GM (HBM)
//   Cube and Vec run on SEPARATE cores — they communicate via GM + cross-core flags.
//
// Key PTO APIs used in this kernel (with numpy/torch equivalents):
//   TASSIGN(tile, addr)     — Bind tile to UB/L1/L0 address (tile = memory[addr])
//   TLOAD(dst, gm_tensor)   — DMA load: dst = gm_tensor (async, MTE2 pipe)
//   TSTORE(gm, src)         — DMA store: gm = src (async, MTE3 pipe)
//   TFILLPAD(dst, src)      — Zero-fill padding: dst[outside valid] = 0
//   TFILLPAD_INPLACE(d, s)  — Same but in-place for UB tiles
//   TEXTRACT(l0, l1, r, c)  — Copy L1 sub-block → L0A or L0B (MTE1 pipe)
//   TRESHAPE(dst, src)      — Reinterpret L1 tile layout (NZ↔ZN for transpose)
//   TMATMUL(C, A, B)        — Matrix multiply: C = A @ B in Cube engine
//   TCVT(dst, src, mode)    — Type conversion: like dst = src.float() or src.half()
//   TMOV(dst, src)          — Copy: dst = src.clone()
//   TADD(d, a, b)           — Element-wise add: d = a + b
//   TSUB(d, a, b)           — Element-wise subtract: d = a - b
//   TMUL(d, a, b)           — Element-wise multiply: d = a * b
//   TMINS(d, s, val)        — Clamp max: d = torch.clamp(s, max=val)
//   TEXP(d, s)              — Element-wise exp: d = torch.exp(s)
//   TLOG(d, s)              — Element-wise log: d = torch.log(s)
//   TROWEXPAND(2d, col)     — Broadcast column → rows: 2d[i,j] = col[i]
//   TCOLEXPAND(2d, row)     — Broadcast row → cols: 2d[i,j] = row[j]
//   set_flag(P1, P2, EVT)   — Signal from pipe P1 to pipe P2 (like a semaphore post)
//   wait_flag(P1, P2, EVT)  — Wait for signal from P1 (like a semaphore wait)
//   pipe_barrier(PIPE_V)    — Local Vec barrier (ensure all Vec ops complete)
//   pipe_barrier(PIPE_ALL)  — Barrier for all local pipes
//   ffts_cross_core_sync()  — Cross-core signal (Cube↔Vec, different physical cores)
//   wait_flag_dev(flag)     — Wait for cross-core signal
// ============================================================================

#include <pto/pto-inst.hpp>   // PTO (Performance Tile Operator): NPU kernel API
#include "acl/acl.h"          // ACL (Ascend Computing Language): runtime API
#include <runtime/rt_ffts.h>  // FFTS: cross-core synchronization primitives
using namespace pto;

// ── Compile-time constants (set by the JIT compiler from Python) ──────
// These are typically passed as -DGDN_H=16 -DGDN_D=128 -DGDN_C=128 on the
// compiler command line. The #ifndef guards provide defaults for IDE tooling.
#ifndef GDN_H
#define GDN_H 16              // H = number of value heads (gates A β,g index here)
#endif

#ifndef GDN_HG
#define GDN_HG GDN_H          // Hg = shared key-query heads (GQA); default MHA
#endif

#ifndef GDN_D
#define GDN_D 128             // D = hidden dimension per head
#endif

#ifndef GDN_C
#define GDN_C 128             // C = chunk size (tokens processed per chunk)
#endif

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// These are only compiled for the NPU device compiler (__CCE_AICORE__ is defined
// when compiling for AI Core hardware, similar to __CUDA_ARCH__ in CUDA).
#ifdef __CCE_AICORE__
// UbND = UB tile in row-major (ND) layout for Vec engine.
//   Think of it as: torch.empty((R, C), dtype=T) in on-chip SRAM.
//   RV, CV = valid region (for dynamic shapes, like a[:valid_rows, :valid_cols])
//   The Vec engine (SIMD unit) reads/writes these tiles for element-wise ops.
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;

// UbDN = UB tile in column-major (DN) layout — needed for TROWEXPAND source.
//   TROWEXPAND requires its source vector in column-major (transposed) format.
//   Same physical memory (UB SRAM), just different indexing convention.
template <typename T, int R, int C, int RV = R, int CV = C>
using UbDN = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::ColMajor,
                       RV, CV, pto::SLayout::NoneBox, 512>;

// L1Mat = L1 cache tile in NZ fractal format (col-major blocks, row-major within).
//   This is the standard input format for the Cube matrix engine.
//   Think of it as a matrix in L1 cache ready for GEMM.
//   NZ = "Normal-Z": the default fractal layout that Cube expects for left/right operands.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1Mat = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::ColMajor,
                        RV, CV, pto::SLayout::RowMajor, 512, pto::PadValue::Zero>;

// L1MatZN = L1 tile in ZN fractal format (row-major blocks, col-major within).
//   Used when you need to transpose a matrix before GEMM:
//   TRESHAPE(l1_zn, l1_nz) reinterprets NZ→ZN layout = logical transpose.
//   This is FREE (no data movement) — it just changes how the Cube reads the bits.
template <typename T, int R, int C, int RV = R, int CV = C>
using L1MatZN = pto::Tile<pto::TileType::Mat, T, R, C, pto::BLayout::RowMajor,
                          RV, CV, pto::SLayout::ColMajor, 512, pto::PadValue::Zero>;
#endif

// ── Main kernel function (runs on each AI core) ──────────────────────
// Template parameters: NumHeads (H value), NumKeyHeads (Hg), HiddenSize, ChunkSize.
// GROUP = H/Hg; Cube loads K at head_g = head_idx / GROUP.
//
// __gm__: Marks pointers as Global Memory (HBM) — the NPU equivalent of
// CUDA's device memory. All input/output tensors live in GM.
template <int32_t NumHeads, int32_t NumKeyHeads, int32_t HiddenSize,
          int32_t ChunkSize>
AICORE void kkt_kernel(
    __gm__ half *K_handle, __gm__ half *Beta_handle,
    __gm__ float *G_handle, __gm__ float *Msk_handle,
    __gm__ half *workspace_handle, __gm__ half *A_handle,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  constexpr int32_t HalfChunk = ChunkSize / 2;
  constexpr int32_t ChunkSquare = ChunkSize * ChunkSize;
  static_assert(NumHeads % NumKeyHeads == 0,
                "NumHeads must be divisible by NumKeyHeads (GQA grouping)");
  constexpr int32_t GROUP = NumHeads / NumKeyHeads;
  constexpr int32_t BSND_QK_STRIDE = NumKeyHeads * HiddenSize;
  // KTail: number of valid columns in the last 128-wide fractal block of K.
  // If HiddenSize is a multiple of 128, the last block is fully used (128).
  // Otherwise it's the remainder. Used internally by TLOAD for partial blocks.
  constexpr uint32_t KTail =
      (HiddenSize % 128 == 0) ? 128 : (HiddenSize % 128);

  // ── UB address map (manual memory planning) ─────────────────────────
  // The UB is a flat SRAM; we manually assign byte offsets for each tile.
  // This is like malloc'ing fixed regions — no dynamic allocator on NPU.
  constexpr int32_t GUbAddr      = 0;       // g_ub: cumulative gates [1×C]
  constexpr int32_t BetaHalfUbAddr = 512;   // beta_ub_half: gate bias fp16 [1×C/2]
  constexpr int32_t BetaUbAddr   = 640;     // beta_ub: gate bias fp32 [1×C/2]
  constexpr int32_t GvUbAddr     = 896;     // g_v_ub: combined gate+bias [1×C/2]
  constexpr int32_t AUbAddr      = 1152;    // a_ub: attention sub-block fp32 [C/2×C]
  constexpr int32_t GRUbAddr     = 33920;   // g_r_ub: row gates [1×C/2]
  constexpr int32_t GCUbAddr     = 34176;   // g_c_ub: column gates [1×C]
  constexpr int32_t MskUbAddr    = 34688;   // msk_ub: causal mask [C/2×C]
  constexpr int32_t GR2dUbAddr   = 67456;   // g_r_2d_ub: broadcast row gates [C/2×C]
  constexpr int32_t GC2dUbAddr   = 124800;  // g_c_2d_ub: broadcast col gates [C/2×C]
  constexpr int32_t CoeffUbAddr  = 157568;  // coeff_ub: gating coefficient [C/2×C]
  // a_ub_half overlaps g_r_2d — safe because they're never live simultaneously
  constexpr int32_t AUbHalfAddr  = GR2dUbAddr;

  // set_ffts_base_addr: Tell the hardware where the cross-core flag table lives.
  // This is a one-time setup so ffts_cross_core_sync / wait_flag_dev know
  // which memory region to read/write for inter-core signaling.
  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();       // Which AI core am I? (like CUDA blockIdx.x)
  auto block_num = get_block_num();  // Total AI cores launched (like CUDA gridDim.x)
  // ── Vec sub-block parallelism ─────────────────────────────────────────
  // Each AI core has 2 Vec sub-blocks (vid=0 and vid=1).
  // They share the same UB memory but run independently in parallel.
  // Here, vid=0 processes rows [0, C/2) and vid=1 processes rows [C/2, C).
  // This halves the per-sub-block work and doubles Vec throughput.
  auto vid = get_subblockid();       // 0 or 1: which Vec sub-block am I?

  // Work distribution: each (sequence, head) pair is one "work item".
  // AI cores split work round-robin, just like CUDA blocks split a grid.
  int64_t num_seqs = batch_size;
  int64_t total_work = num_seqs * NumHeads;

  // ── Cube-side tile declarations ─────────────────────────────────────
  // Cube-side tiles: K in L1 (NZ format), accumulator in L0C
  L1Mat<half, ChunkSize, HiddenSize,
        ChunkSize, HiddenSize> k_l1;
  TASSIGN(k_l1, 0);
  // TileAcc<float, C, C>: L0C accumulator tile for GEMM results.
  // The Cube engine always accumulates in float32 for precision, even when
  // inputs are fp16. Think of it as: result = torch.matmul(a.half(), b.half()).float()
  // When stored to GM via TSTORE with a half GlobalTensor, automatic fp32→fp16 cast occurs.
  TileAcc<float, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a_l0;
  TASSIGN(a_l0, 0);

  // ── Vec-side UB tile declarations ────────────────────────────────────
  // These tiles live in UB (Unified Buffer, the Vec engine's SRAM scratchpad).
  // Each TASSIGN binds a tile handle to a fixed UB byte offset (our manual alloc).
  // Vec-side UB tiles for gating computation
  UbND<float, 1, ChunkSize, 1, ChunkSize> g_ub;
  TASSIGN(g_ub, GUbAddr);
  UbND<half, 1, HalfChunk, 1, HalfChunk> beta_ub_half;
  TASSIGN(beta_ub_half, BetaHalfUbAddr);
  UbND<float, 1, HalfChunk, 1, HalfChunk> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  UbND<float, 1, HalfChunk, 1, HalfChunk> g_v_ub;
  TASSIGN(g_v_ub, GvUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> a_ub;
  TASSIGN(a_ub, AUbAddr);
  UbND<float, 1, HalfChunk, 1, HalfChunk> g_r_ub;
  TASSIGN(g_r_ub, GRUbAddr);
  UbND<float, 1, ChunkSize, 1, ChunkSize> g_c_ub;
  TASSIGN(g_c_ub, GCUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> msk_ub;
  TASSIGN(msk_ub, MskUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> g_r_2d_ub;
  TASSIGN(g_r_2d_ub, GR2dUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> g_c_2d_ub;
  TASSIGN(g_c_2d_ub, GC2dUbAddr);
  UbND<float, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> coeff_ub;
  TASSIGN(coeff_ub, CoeffUbAddr);
  UbND<half, HalfChunk, ChunkSize,
       HalfChunk, ChunkSize> a_ub_half;
  TASSIGN(a_ub_half, AUbHalfAddr);

  // ========================================================================
  // CUBE PHASE: Compute KK^T = K @ K^T for each chunk via GEMM
  //
  // ── How GEMM works on NPU (the "Cube pipeline") ──────────────────────
  // The matrix multiply pipeline has 3 stages:
  //   Step 1: TLOAD loads data from GM → L1 (MTE2 pipe)
  //   Step 2: TEXTRACT copies sub-blocks from L1 → L0A/L0B (MTE1 pipe)
  //     L0A holds the left operand, L0B holds the right operand
  //   Step 3: TMATMUL multiplies L0A × L0B → L0C accumulator (M pipe)
  //
  // For K @ K^T:  (numpy: KK_T = K @ K.T)
  //   Left operand: K [C×D] loaded into L1 in NZ format
  //   Right operand: K^T — same data, but we TRESHAPE to ZN format
  //     (TRESHAPE is FREE — it just reinterprets the fractal layout as transposed)
  //   Result: KK^T [C×C] in L0C (float32 accumulator, even though inputs are fp16)
  // ========================================================================
  // __DAV_C220_CUBE__: This code only compiles for the Cube core.
  // On NPU, Cube and Vec are separate compilation targets (like two different GPUs).
#if defined(__DAV_C220_CUBE__)
  // Outer loop: iterate over all (sequence, head) work items assigned to this core
  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                  static_cast<int64_t>(cid);
    if (pid >= total_work) continue;

    // Map linear work index → (sequence, head) pair
    int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
    int64_t seq_idx = pid / NumHeads;

    // Resolve sequence boundaries: cu_seqlens for variable-length, else fixed stride
    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      // Variable-length sequences (packed tensor): cu_seqlens = [0, len0, len0+len1, ...]
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
    } else {
      // Fixed-length sequences: each is seq_len tokens starting at seq_idx*seq_len
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    // Ceiling division: how many ChunkSize-sized chunks cover this sequence
    int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

    // ── Double-buffering via workspace slots ──────────────────────────
    // slot = ci & 1: alternates between 0 and 1 each chunk iteration.
    // Cube writes KK^T to workspace[slot], then signals Vec.
    // While Vec processes slot[0], Cube can write slot[1] (next chunk).
    // This overlaps Cube computation with Vec computation for pipelining.
    for (int64_t ci = 0; ci < num_chunks; ++ci) {
      int32_t slot = static_cast<int32_t>(ci & 1);
      // Wait for Vec to finish reading the previous KK^T from this slot
      wait_flag_dev(2 + slot);
      pipe_barrier(PIPE_ALL);

      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      // BSND key layout [Seq, Hg, D]: token stride Hg * D (see BSND_QK_STRIDE).
      // Value head head_idx maps to head_g = head_idx / GROUP for shared K rows.
      int32_t head_g = head_idx / GROUP;
      int64_t k_offset =
          ((bos + chunk_start) * static_cast<int64_t>(NumKeyHeads) +
           static_cast<int64_t>(head_g)) *
          static_cast<int64_t>(HiddenSize);

      // ── Load K chunk from GM → L1 (MTE2 pipe) ──────────────────────
      // DYNAMIC shape: valid_rows may be < ChunkSize for the last chunk.
      // GlobalTensor describes the GM layout with strides (BSND interleaved).
      // TLOAD triggers the MTE2 DMA engine to copy from GM (HBM) → L1 (on-chip cache).
      // If the chunk is partial, TFILLPAD zero-fills the padding region
      // so the GEMM doesn't produce garbage from uninitialized memory.
      {
        L1Mat<half, ChunkSize, HiddenSize, DYNAMIC, DYNAMIC> _l1(valid_rows, HiddenSize);
        TASSIGN(_l1, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = valid_rows; _gs.shape[4] = HiddenSize;
        GlobalTensor<half, decltype(_gs),
                     Stride<1, 1, 1, BSND_QK_STRIDE, 1>>
            _gm(K_handle + k_offset, _gs);
        TLOAD(_l1, _gm);
        if (valid_rows != ChunkSize) TFILLPAD(_l1, _l1);
      }

      // ── GEMM: KK^T = K @ K^T (L1→L0A/L0B→L0C) ────────────────────
      // K is [C×D] in L1 NZ; K^T obtained via ZN reshape of same tile.
      //
      // ── WAR (Write-After-Read) synchronization ────────────────────────
      // Before TEXTRACT (MTE1) writes new data to L0A/L0B, we must ensure:
      //   1. MTE2 has finished loading L1 (MTE2→MTE1 sync)
      //   2. Cube M pipe has finished reading previous L0A/L0B data (M→MTE1 sync)
      // After TEXTRACT, before TMATMUL:
      //   3. MTE1→M sync ensures L0A/L0B data is ready for the matrix engine
      // After TMATMUL completes:
      //   4. M→FIX sync ensures the L0C accumulator can be read
      // This is like ensuring a producer-consumer chain is properly ordered.
      // WAR sync: MTE2→MTE1, M→MTE1 before extract; MTE1→M before matmul.
      {
        TileLeft<half, ChunkSize, HiddenSize, ChunkSize, HiddenSize> _l0a;
        TileRight<half, HiddenSize, ChunkSize, HiddenSize, ChunkSize> _l0b;
        TASSIGN(_l0a, 0x0);
        TASSIGN(_l0b, 0x0);
        auto _we = EVENT_ID1;
        set_flag(PIPE_MTE2, PIPE_MTE1, _we);
        wait_flag(PIPE_MTE2, PIPE_MTE1, _we);
        set_flag(PIPE_M, PIPE_MTE1, _we);
        wait_flag(PIPE_M, PIPE_MTE1, _we);
        // Left operand: K in NZ format, extract directly to L0A
        TEXTRACT(_l0a, k_l1, 0, 0);
        // Right operand: K^T via ZN reshape of same L1 tile, extract to L0B
        L1MatZN<half, HiddenSize, ChunkSize> _bzn;
        TRESHAPE(_bzn, k_l1);
        TEXTRACT(_l0b, _bzn, 0, 0);
        set_flag(PIPE_MTE1, PIPE_M, _we);
        wait_flag(PIPE_MTE1, PIPE_M, _we);
        TMATMUL(a_l0, _l0a, _l0b);
        set_flag(PIPE_MTE1, PIPE_MTE2, _we);
        wait_flag(PIPE_MTE1, PIPE_MTE2, _we);
        set_flag(PIPE_M, PIPE_FIX, _we);
        wait_flag(PIPE_M, PIPE_FIX, _we);
      }

      // ── Store KK^T from L0C → workspace GM (with fp32→fp16 cast) ───
      {
        TileAcc<float, ChunkSize, ChunkSize, DYNAMIC, DYNAMIC> _l0(ChunkSize, ChunkSize);
        TASSIGN(_l0, 0);
        Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
        _gs.shape[3] = ChunkSize; _gs.shape[4] = ChunkSize;
        GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
            workspace_handle +
                (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare,
            _gs);
        TSTORE(_gm, _l0);
      }

      // ── Cross-core synchronization (Cube → Vec) ──────────────────────
      // ffts_cross_core_sync(pipe, config): Signal across physical cores.
      // Unlike set_flag/wait_flag (which sync pipes within ONE core), this syncs
      // between the Cube core and Vec core (they are separate hardware units).
      //
      // Config encoding: 1 | (mode << 4) | (flag_id << 8)
      //   mode=2: broadcast to all cores on same block
      //   flag_id: which flag to set (0,1,2,3...)
      //
      // The receiving side calls wait_flag_dev(flag_id) to wait for this signal.
      //
      // In this kernel:
      //   Cube sets flag 0/1 → Vec waits on wait_flag_dev(0/1) (KK^T ready)
      //   Vec sets flag 2/3 → Cube waits on wait_flag_dev(2/3) (workspace free)
      //
      // Signal Vec that this slot's KK^T is ready
      ffts_cross_core_sync(PIPE_FIX, 1 | (2 << 4) | (slot << 8));
    }
  }
#endif

  // ========================================================================
  // VEC PHASE: Apply gating and causal mask to KK^T
  //   coeff[i,j] = exp(min(g[i]+log(β[i]) - g[j], 0))
  //   A[i,j] = KK^T[i,j] · coeff[i,j] · mask[i,j]
  // Each sub-block (vid=0,1) handles HalfChunk rows of the C×C matrix.
  //
  // ── Gating computation (numpy pseudocode) ─────────────────────────────
  // # For each sub-block's C/2 rows (vid selects upper or lower half):
  // g_row = g_sum[row_offset:row_offset+C/2]           # this sub-block's gates
  // g_v = g_row + np.log(beta[row_offset:row_offset+C/2])  # combined gate+bias
  // g_col = g_sum[0:C]                                  # full chunk gates
  //
  // # Broadcast to 2D matrices for element-wise ops:
  // g_r_2d = np.tile(g_v.reshape(-1, 1), (1, C))       # TROWEXPAND
  // g_c_2d = np.tile(g_col.reshape(1, -1), (C/2, 1))   # TCOLEXPAND
  //
  // # Gating coefficient: exponential decay, clamped to ≤ 1
  // coeff = np.exp(np.minimum(g_r_2d - g_c_2d, 0))     # TSUB → TMINS → TEXP
  //
  // # Final: A = KK_T * coeff * causal_mask
  // A = KK_T[my_rows] * coeff * mask[my_rows]           # TMUL × 2
  // ========================================================================
  // __DAV_C220_VEC__: This code only compiles for the Vec core.
#if defined(__DAV_C220_VEC__)
  // set_mask_norm / set_vector_mask: configure the SIMD mask for Vec ops.
  // (-1, -1) means "all lanes active" — process every element.
  // (Like CUDA's __activemask() returning all 1s for a full warp.)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // ── Load causal mask (lower triangular) once, reused across all chunks ──
  // vid=0 loads the top half (rows 0..C/2-1), vid=1 loads the bottom half.
  // The mask is [C×C] in GM; each sub-block loads its [C/2×C] portion.
  {
    Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
    _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
    GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
        Msk_handle +
            static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
        _gs);
    UbND<float, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
    TASSIGN(_ld, MskUbAddr);
    TLOAD(_ld, _gm);
  }
  // MTE2→V sync: ensure mask DMA is complete before Vec reads it
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

  // Initial cross-core sync: release both workspace slots so Cube can start.
  // Vec tells Cube "slots 0 and 1 are free" by setting flags 2 and 3.
  // Without this, Cube would hang on wait_flag_dev(2/3) at the first iteration.
  ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (2 << 8));
  ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (3 << 8));

  for (int64_t work_idx = 0;
       work_idx < (total_work + block_num - 1) / block_num; ++work_idx) {
    int64_t pid = work_idx * static_cast<int64_t>(block_num) +
                  static_cast<int64_t>(cid);
    if (pid >= total_work) continue;

    int32_t head_idx = static_cast<int32_t>(pid % NumHeads);
    int64_t seq_idx = pid / NumHeads;

    int64_t bos, slen;
    if (cu_seqlens != nullptr) {
      bos = static_cast<int64_t>(cu_seqlens[seq_idx]);
      slen = static_cast<int64_t>(cu_seqlens[seq_idx + 1]) - bos;
    } else {
      bos = seq_idx * seq_len;
      slen = seq_len;
    }
    int64_t num_chunks = (slen + ChunkSize - 1) / ChunkSize;

    for (int64_t ci = 0; ci < num_chunks; ++ci) {
      int32_t slot = static_cast<int32_t>(ci & 1);

      int64_t chunk_start = ci * ChunkSize;
      int64_t remaining = slen - chunk_start;
      int32_t valid_rows = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);
      // row_offset: which half of the C×C matrix this sub-block handles
      //   vid=0 → rows [0, C/2),  vid=1 → rows [C/2, C)
      int32_t row_offset = static_cast<int32_t>(vid) * HalfChunk;
      // local_valid: how many rows in this sub-block are real (not padding)
      //   Handles the case where the last chunk has fewer than C valid rows
      int32_t local_valid =
          valid_rows > row_offset
              ? (valid_rows - row_offset < HalfChunk
                     ? valid_rows - row_offset
                     : HalfChunk)
              : 0;

      if (local_valid > 0) {
        // ── Load G (full chunk, 1×C) and Beta (sub-block rows, 1×HalfC) ──
        // G is [H, total_tokens] float — contiguous per head
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1; _gs.shape[4] = valid_rows;
          GlobalTensor<float, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
              G_handle + static_cast<int64_t>(head_idx) * total_tokens
                       + (bos + chunk_start),
              _gs);
          UbND<float, 1, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, valid_rows);
          TASSIGN(_ld, GUbAddr);
          TLOAD(_ld, _gm);
          if (valid_rows != ChunkSize) {
            UbND<float, 1, ChunkSize, 1, ChunkSize, PadValue::Zero> _pd;
            TASSIGN(_pd, GUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }

        // Beta is [H, total_tokens] half — contiguous per head
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = 1; _gs.shape[4] = local_valid;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, 1, 1>> _gm(
              Beta_handle + static_cast<int64_t>(head_idx) * total_tokens
                          + (bos + chunk_start + row_offset),
              _gs);
          UbND<half, 1, HalfChunk, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(1, local_valid);
          TASSIGN(_ld, BetaHalfUbAddr);
          TLOAD(_ld, _gm);
          if (local_valid != HalfChunk) {
            UbND<half, 1, HalfChunk, 1, HalfChunk, PadValue::Zero> _pd;
            TASSIGN(_pd, BetaHalfUbAddr);
            TFILLPAD_INPLACE(_pd, _ld);
          }
        }
      }

      // Wait for Cube to finish writing KK^T for this slot
      wait_flag_dev(slot);
      pipe_barrier(PIPE_ALL);

      if (local_valid > 0) {
        // ── Compute gating coefficient ────────────────────────────────
        // Step 1: Convert beta from fp16→fp32 for precision
        // Step 2: g_v[i] = g[row_offset+i] + log(β[i])  — combined row gate
        // Step 3: Broadcast g_v (rows) and g (cols) to 2D matrices
        // Step 4: coeff = exp(min(g_v_2d - g_2d, 0)) — clamped exponential gating
        // g_v[i] = g[row_offset+i] + log(β[i])  — combined row gate
        TCVT(beta_ub, beta_ub_half, pto::RoundMode::CAST_NONE);
        // g_ub_temp points to the sub-block's portion of g within the full g_ub.
        // row_offset * sizeof(float) is the byte offset into the g_ub tile.
        UbND<float, 1, HalfChunk, 1, HalfChunk>
            g_ub_temp;
        TASSIGN(g_ub_temp,
                GUbAddr + row_offset *
                              static_cast<int32_t>(sizeof(float)));
        TMOV(g_v_ub, g_ub_temp);   // g_v = g[row_offset:row_offset+C/2]
        pipe_barrier(PIPE_V);       // Wait for TMOV to complete

        TLOG(beta_ub, beta_ub);     // beta_ub = log(beta) in-place
        pipe_barrier(PIPE_V);
        TADD(g_v_ub, g_v_ub, beta_ub);  // g_v = g_sub + log(beta) — the combined gate
        pipe_barrier(PIPE_V);
        TMOV(g_r_ub, g_v_ub);      // Copy to g_r for row-broadcast
        TMOV(g_c_ub, g_ub);        // Copy full g to g_c for col-broadcast
        pipe_barrier(PIPE_V);

        // Broadcast g_v to rows, g to columns → 2D gating matrix
        // coeff[i,j] = exp(min(g_v[i] - g[j], 0))
        //
        // g_r_ub_temp is a column-major (DN) alias of g_r_ub, required because
        // TROWEXPAND expects its source in column-major layout.
        UbDN<float, HalfChunk, 1,
             HalfChunk, 1> g_r_ub_temp;
        TASSIGN(g_r_ub_temp, GRUbAddr);
        TROWEXPAND(g_r_2d_ub, g_r_ub_temp);  // g_r_2d[i,j] = g_v[i] for all j
        TCOLEXPAND(g_c_2d_ub, g_c_ub);       // g_c_2d[i,j] = g[j] for all i
        pipe_barrier(PIPE_V);
        TSUB(coeff_ub, g_r_2d_ub, g_c_2d_ub);  // coeff[i,j] = g_v[i] - g[j]
        pipe_barrier(PIPE_V);
        TMINS(coeff_ub, coeff_ub, 0.0f);        // clamp to ≤ 0 (coeff will be ≤ 1 after exp)
        pipe_barrier(PIPE_V);
        TEXP(coeff_ub, coeff_ub);                // coeff = exp(clamped_diff) ∈ (0, 1]

        // V→MTE2 sync: ensure gating computation is done before we start
        // loading KK^T from workspace (we need coeff ready for the multiply later,
        // and we want to overlap the DMA load with the preceding Vec work).
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        // ── Load KK^T sub-block from workspace (fp16) ────────────────
        // workspace layout: [core_id * 2 + slot][C×C], we load our sub-block's
        // [C/2×C] portion (offset by vid * HalfChunk * ChunkSize elements).
        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = HalfChunk; _gs.shape[4] = ChunkSize;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, ChunkSize, 1>> _gm(
              workspace_handle +
                  (static_cast<int64_t>(cid) * 2 + slot) * ChunkSquare +
                  static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
              _gs);
          UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC, PadValue::Zero> _ld(HalfChunk, ChunkSize);
          TASSIGN(_ld, AUbHalfAddr);
          TLOAD(_ld, _gm);
        }

        // MTE2→V sync: KK^T data is now in UB, safe for Vec to read
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // ── Apply gating and mask: A = KK^T · coeff · mask ───────────
        // 1. Convert KK^T from fp16 → fp32 (Cube stored it as fp16 to save GM bandwidth)
        TCVT(a_ub, a_ub_half, pto::RoundMode::CAST_NONE);
        // 2. Element-wise multiply by gating coefficient
        TMUL(a_ub, a_ub, coeff_ub);
        // 3. Element-wise multiply by causal mask (lower triangular, zeros above diagonal)
        TMUL(a_ub, a_ub, msk_ub);
        // 4. Convert result back to fp16 for output
        TCVT(a_ub_half, a_ub, pto::RoundMode::CAST_NONE);

        // V→MTE3 sync: Vec computation done, safe for DMA store to begin
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // ── Store A sub-block to output GM ────────────────────────────
        // Output A is in BSND layout: [total_tokens, NumHeads, ChunkSize]
        // Each row of A corresponds to one token's attention weights for this head.
        // Stride between consecutive tokens = NumHeads * ChunkSize (BSND interleaved).
        int64_t a_gm_offset =
            ((bos + chunk_start + row_offset) * NumHeads +
             head_idx) *
            static_cast<int64_t>(ChunkSize);

        {
          Shape<1, 1, 1, DYNAMIC, DYNAMIC> _gs;
          _gs.shape[3] = local_valid; _gs.shape[4] = ChunkSize;
          GlobalTensor<half, decltype(_gs), Stride<1, 1, 1, NumHeads * ChunkSize, 1>> _gm(A_handle + a_gm_offset, _gs);
          UbND<half, HalfChunk, ChunkSize, DYNAMIC, DYNAMIC> _st(local_valid, ChunkSize);
          TASSIGN(_st, AUbHalfAddr);
          TSTORE(_gm, _st);
        }
      }

      pipe_barrier(PIPE_ALL);
      // Signal Cube that this workspace slot is free for reuse.
      // Flag (2+slot): slot 0 → flag 2, slot 1 → flag 3.
      // Cube is waiting on wait_flag_dev(2+slot) before writing the next chunk.
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | ((2 + slot) << 8));
    }
  }
#endif
}

// ── NPU kernel entry point ────────────────────────────────────────────
// extern "C" __global__ AICORE: NPU kernel entry point (like CUDA __global__).
// Parameters passed as uint8_t* and reinterpret_cast'd — standard NPU convention.
// The NPU runtime passes raw byte pointers; we cast them to typed pointers here.
// GDN_H, GDN_D, GDN_C are compile-time constants set by #define at the top.
extern "C" __global__ AICORE void launch_scaled_dot_kkt(
    __gm__ uint8_t *K_handle, __gm__ uint8_t *Beta_handle,
    __gm__ uint8_t *G_handle, __gm__ uint8_t *Msk_handle,
    __gm__ uint8_t *workspace_handle, __gm__ uint8_t *A_handle,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens,
    uint64_t ffts_addr)
{
  kkt_kernel<GDN_H, GDN_HG, GDN_D, GDN_C>(
      reinterpret_cast<__gm__ half *>(K_handle),
      reinterpret_cast<__gm__ half *>(Beta_handle),
      reinterpret_cast<__gm__ float *>(G_handle),
      reinterpret_cast<__gm__ float *>(Msk_handle),
      reinterpret_cast<__gm__ half *>(workspace_handle),
      reinterpret_cast<__gm__ half *>(A_handle),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, total_tokens, ffts_addr);
}

// ── Host-side launcher ────────────────────────────────────────────────
// call_kernel(): Host-side launcher invoked from Python via ctypes.
//   block_dim = number of AI cores (like CUDA grid size)
//   <<<block_dim, nullptr, stream>>>: NPU kernel launch syntax
//     - block_dim: how many AI cores to use (each runs kkt_kernel independently)
//     - nullptr: no shared memory (NPU doesn't have CUDA-style shared mem)
//     - stream: async execution stream (like CUDA streams)
//
// rtGetC2cCtrlAddr: Get the hardware address of the cross-core (Cube↔Vec) flag
// table. This address is passed to the kernel so it can call ffts_cross_core_sync.
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *K_handle, uint8_t *Beta_handle,
    uint8_t *G_handle, uint8_t *Msk_handle,
    uint8_t *workspace_handle, uint8_t *A_handle,
    uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    int64_t total_tokens)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_scaled_dot_kkt<<<block_dim, nullptr, stream>>>(
      K_handle, Beta_handle, G_handle, Msk_handle,
      workspace_handle, A_handle, cu_seqlens,
      batch_size, seq_len, total_tokens, fftsAddr);
}
