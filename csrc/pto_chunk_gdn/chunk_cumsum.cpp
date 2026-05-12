// ============================================================================
// chunk_cumsum_kernel.cpp — Prefix sum of gate values G along time dimension
//
// Mathematical operation (per chunk of C tokens, independently per head h):
//   g_sum[t, h] = Σ_{i=0}^{t} g[i, h]    for t = 0 .. valid-1
//
// Input:  g     [total_tokens, H]  float, BSND layout  — raw gate values
// Output: g_sum [total_tokens, H]  float               — cumulative sums
//
// The prefix sum enables downstream kernels to compute exponential decay
// coefficients:  exp(g_sum[i] - g_sum[j])  gives the cumulative gate
// from token j to token i within a chunk.
//
// Architecture: Vec-only kernel (no Cube/GEMM). Single Vec sub-block.
// Pipeline: MTE2(load) → Vec(compute) → MTE3(store), serialized per chunk.
//
// NPU memory hierarchy used:
//   GM (Global Memory) → UB (Unified Buffer, on-chip SRAM, Vec-accessible)
//
// ─── PTO / NPU Primer for This Kernel ──────────────────────────────────────
//
// AI Core: The basic processing unit of an NPU, analogous to a Streaming
//   Multiprocessor (SM) on a GPU. A single chip has many AI cores, and each
//   core runs the same kernel code on different data (SPMD model).
//
// Memory hierarchy (outer → inner):
//   GM  (Global Memory) — Off-chip DRAM, like GPU HBM. Large (several GB)
//       but high latency. All AI cores share GM.
//   UB  (Unified Buffer) — On-chip SRAM, ~256 KB per AI core. Like GPU
//       shared memory. Very fast, but small. The Vec engine can only operate
//       on data that lives in UB, so every tensor must be DMA'd in first.
//
// Hardware pipes (execute in parallel, like independent GPU warps):
//   Vec   — SIMD vector processor. Performs element-wise math (add, mul, etc.)
//           on data already in UB. Think of it as a wide SIMD ALU.
//   MTE2  — DMA engine for loads: copies data from GM → UB.
//   MTE3  — DMA engine for stores: copies data from UB → GM.
//   Cube  — Matrix engine for GEMMs (not used in this kernel).
//
// Synchronization (set_flag / wait_flag):
//   Because Vec, MTE2, and MTE3 run in parallel on separate hardware, you
//   must explicitly synchronize them to ensure data is ready:
//     set_flag(SRC_PIPE, DST_PIPE, event): SRC signals that it is done.
//     wait_flag(SRC_PIPE, DST_PIPE, event): DST blocks until the signal.
//   Example: After MTE2 loads data into UB, Vec must wait_flag before reading
//   it. This is like a fine-grained torch.cuda.synchronize() between pipes.
//   Events (EVENT_ID0 .. EVENT_ID7) are semaphore indices.
//
// ============================================================================

#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

// GDN_H, GDN_C: Compile-time constants injected by the build system.
//   GDN_H = number of attention heads (e.g., 16)
//   GDN_C = chunk size in tokens (e.g., 128)
// Using compile-time constants allows the compiler to optimize tile sizes,
// unroll loops, and compute UB addresses at compile time.
#ifndef GDN_H
#define GDN_H 16
#endif

#ifndef GDN_C
#define GDN_C 128
#endif

// ── PTO type aliases (device-only, guarded by __CCE_AICORE__) ───────────────
// UB tile in row-major (ND) layout, used by Vec engine.
// T=dtype, R×C=static shape, RV×CV=valid region, P=pad value for TLOAD.
//
// Think of UbND as: torch.empty((R, C), dtype=T) allocated in on-chip SRAM (UB).
//   - TileType::Vec  = this tile lives in UB, operated on by the Vec (SIMD) engine
//   - BLayout::RowMajor = row-major storage, like C arrays or numpy default
//   - RV, CV = "valid" region within the R×C buffer (for handling partial/tail chunks)
//   - PadValue = what to fill outside the valid region during TLOAD (Zero or Null)
//   - 512 = alignment in bytes (hardware requirement for efficient DMA)
#ifdef __CCE_AICORE__
template <typename T, int R, int C, int RV = R, int CV = C,
          pto::PadValue P = pto::PadValue::Null>
using UbND = pto::Tile<pto::TileType::Vec, T, R, C, pto::BLayout::RowMajor,
                       RV, CV, pto::SLayout::NoneBox, 512, P>;
#endif

template <int32_t NumHeads, int32_t ChunkSize>
AICORE void cumsum_kernel(
    __gm__ float *g_ptr, __gm__ float *g_sum_ptr,
    __gm__ int32_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  // get_block_idx(): Returns this AI core's index (0..block_num-1).
  //   Like blockIdx.x in CUDA — identifies which core this code runs on.
  // get_block_num(): Total number of AI cores launched (like gridDim.x in CUDA).
  // get_subblockid(): Returns 0 or 1 — selects which Vec sub-block within the core.
  //   Each AI core has 2 Vec sub-blocks that can run in parallel.
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();
  // set_ffts_base_addr(ffts_addr): Configure the base address for FFTS
  // (Fast Fine-grained Task Synchronization) — the cross-core signaling mechanism.
  // Required before any cross-core sync (ffts_cross_core_sync / wait_flag_dev).
  set_ffts_base_addr(ffts_addr);

// #if defined(__DAV_C220_VEC__): This block only compiles for the Vec core pass.
// The bisheng compiler makes 3 passes over the same source file:
//   Pass 1: __DAV_C220_VEC__  defined → compiles Vec (SIMD) code
//   Pass 2: __DAV_C220_CUBE__ defined → compiles Cube (matrix) code
//   Pass 3: neither defined → compiles host (CPU) launcher code
// Using these guards lets us put Vec, Cube, and host code in one file.
#if defined(__DAV_C220_VEC__)
  if (vid != 0) return;

  // set_mask_norm(): Reset Vec mask to normal mode (all lanes active).
  // set_vector_mask(-1, -1): Enable all SIMD lanes (128 lanes for fp32).
  //   The -1 sets all 64 bits to 1 in each of the two 64-bit mask registers.
  //   This is like setting torch's computation to operate on all elements.
  set_mask_norm();
  set_vector_mask(-1, -1);

  // HeadTileCols: NumHeads rounded up to 8-element alignment (32B for float)
  // HTC = NumHeads rounded up to nearest multiple of 8.
  // Why? The Vec engine processes data in 32-byte granularity.
  // For float (4 bytes), that's 8 elements per SIMD "word".
  // Rounding up ensures every row is a whole number of SIMD words,
  // avoiding partial-lane issues. The extra columns are zero-padded.
  // Example: NumHeads=16 → HTC=16 (already aligned), NumHeads=13 → HTC=16.
  constexpr int32_t HTC = ((NumHeads + 7) / 8) * 8;
  constexpr int32_t BlockBytes = ChunkSize * HTC *
                                 static_cast<int32_t>(sizeof(float));
  constexpr int32_t RowBytes = HTC * static_cast<int32_t>(sizeof(float));

  // ── UB memory layout ──────────────────────────────────────────────────
  //  [0            .. BlockBytes)     = g input  (ChunkSize × HTC floats)
  //  [BlockBytes   .. 2*BlockBytes)   = g_sum output
  //  [2*BlockBytes .. 2*BlockBytes+RowBytes) = row accumulator (1 × HTC)
  constexpr int32_t GUbAddr   = 0;
  constexpr int32_t SUbAddr   = BlockBytes;
  constexpr int32_t AccUbAddr = BlockBytes * 2;

  // GlobalTensor types for g/g_sum in [total_tokens, NumHeads] layout.
  // 5D shape with last two dims dynamic; stride encodes row pitch.
  //
  // GlobalTensor is a "view" into GM (Global Memory), like torch.as_strided().
  //   GlobalTensor<dtype, Shape, Stride>(base_ptr, shape)
  // Shape<1,1,1,DYNAMIC,DYNAMIC> = 5D shape where first 3 dims are 1 (unused),
  //   last 2 dims are set at runtime (valid rows × NumHeads).
  // Stride<1,1,1,NumHeads,1> = stride between elements. The 4th stride = NumHeads
  //   means consecutive rows in GM are NumHeads elements apart (BSND layout:
  //   token[t] at offset t*NumHeads, head[h] at offset h within that token).
  // This is equivalent to:
  //   g_gm = torch.as_strided(g_ptr, size=[valid, NumHeads], stride=[NumHeads, 1])
  using GmShape  = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
  using GmStride = Stride<1, 1, 1, NumHeads, 1>;
  using GmFloat  = GlobalTensor<float, GmShape, GmStride>;

  // Pre-assign row accumulator at fixed UB address
  // TASSIGN(tile, address): Binds a tile descriptor to a fixed byte address in UB.
  // Think of it as: tile = ub_memory[address:address+sizeof(tile)]
  // This does NOT allocate or move data — it just tells the hardware where the tile lives.
  // We manually manage UB memory layout (like a memory pool) via compile-time addresses.
  UbND<float, 1, HTC> acc_ub;
  TASSIGN(acc_ub, AccUbAddr);

  int64_t num_seqs = batch_size;

  // ── Fixed-length sequence path (cu_seqlens == nullptr) ────────────────
  if (cu_seqlens == nullptr) {
    int64_t chunks_per_seq = (seq_len + ChunkSize - 1) / ChunkSize;
    int64_t total_chunks = num_seqs * chunks_per_seq;

    // Work distribution: Each AI core processes chunks in a round-robin pattern.
    // Core `cid` handles chunks cid, cid+block_num, cid+2*block_num, ...
    // This is the NPU equivalent of CUDA's grid-stride loop:
    //   for (int i = blockIdx.x; i < total; i += gridDim.x)
    for (int64_t gi = static_cast<int64_t>(cid); gi < total_chunks;
         gi += static_cast<int64_t>(block_num)) {
      int64_t seq_idx = gi / chunks_per_seq;
      int64_t local_chunk = gi % chunks_per_seq;
      int64_t bos = seq_idx * seq_len;
      int64_t chunk_start = bos + local_chunk * ChunkSize;
      int64_t remaining = seq_len - local_chunk * ChunkSize;
      int32_t valid = static_cast<int32_t>(
          remaining < ChunkSize ? remaining : ChunkSize);

      // ── DMA: load g[chunk_start .. +valid] from GM → UB (MTE2 pipe) ──
      // Constructs a GlobalTensor view over the g array, loads into UB,
      // then zero-pads the tail region (rows beyond `valid`, cols beyond
      // NumHeads up to the 8-aligned HTC) so downstream Vec ops see zeros.
      {
        GmShape gs; gs.shape[3] = valid; gs.shape[4] = NumHeads;
        GmFloat g_gm(g_ptr + chunk_start * NumHeads, gs);
        UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
            g_load(valid, NumHeads);
        TASSIGN(g_load, GUbAddr);
        // TLOAD(ub_tile, gm_tensor): DMA transfer from GM → UB.
        // Equivalent to: ub_tile[:valid, :NumHeads] = gm_tensor[:valid, :NumHeads]
        // This is an ASYNC operation on the MTE2 pipe — the CPU/Vec engine can do
        // other work while DMA is in progress. You must call set_flag/wait_flag
        // before reading the loaded data.
        TLOAD(g_load, g_gm);
        if (valid != ChunkSize || NumHeads != HTC) {
          UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero> g_pad;
          TASSIGN(g_pad, GUbAddr);
          // TFILLPAD_INPLACE(full_tile, partial_tile): Zero-fills the region outside
          // the valid area of partial_tile.
          // Equivalent to:
          //   full_tile[valid:ChunkSize, :] = 0  # zero rows beyond valid
          //   full_tile[:, NumHeads:HTC] = 0     # zero cols beyond NumHeads (alignment padding)
          // This ensures downstream Vec operations see clean zeros in padded regions.
          TFILLPAD_INPLACE(g_pad, g_load);
        }
      }
      // ── Synchronization: MTE2 → Vec ────────────────────────────────────
      // set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0): Signal from MTE2 (DMA load
      //   engine) to Vec (SIMD engine) that the DMA transfer is complete.
      // wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0): Vec waits here until MTE2
      //   has set the flag. After this, UB data from TLOAD is safe to read.
      // Think of it as: torch.cuda.synchronize() but fine-grained per pipe.
      // EVENT_ID0 is a semaphore index (0-7 available).
      // MTE2 → Vec sync: wait for DMA load to finish before Vec reads UB
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

      // ── Vec compute: prefix sum over rows (all H heads in parallel) ───
      // Row 0: acc[h] = g[0,h];  g_sum[0,h] = acc[h]
      UbND<float, 1, HTC> g_row_0;
      TASSIGN(g_row_0, GUbAddr);
      // TMOV(dst, src): Element-wise copy, like dst = src.clone() in UB.
      TMOV(acc_ub, g_row_0);
      // pipe_barrier(PIPE_V): Ensures all pending Vec (SIMD) operations complete
      // before the next Vec instruction begins. Needed because Vec ops are pipelined
      // and may not finish in order. Think of it as a local __syncthreads() for the
      // Vec engine only. Much lighter than set_flag/wait_flag (which sync across
      // different hardware units).
      pipe_barrier(PIPE_V);

      UbND<float, 1, HTC> s_row_0;
      TASSIGN(s_row_0, SUbAddr);
      TMOV(s_row_0, acc_ub);
      pipe_barrier(PIPE_V);

      // Rows 1..valid-1:  acc[h] += g[i,h];  g_sum[i,h] = acc[h]
      for (int32_t i = 1; i < valid; ++i) {
        UbND<float, 1, HTC> g_row_i;
        TASSIGN(g_row_i, GUbAddr + i * RowBytes);
        // TADD(dst, a, b): Element-wise add, like dst = a + b. All in UB.
        // Operates on all HTC elements in parallel (SIMD).
        TADD(acc_ub, acc_ub, g_row_i);
        pipe_barrier(PIPE_V);

        UbND<float, 1, HTC> s_row_i;
        TASSIGN(s_row_i, SUbAddr + i * RowBytes);
        TMOV(s_row_i, acc_ub);
        pipe_barrier(PIPE_V);
      }

      // Zero-fill rows beyond valid (tail padding for downstream kernels)
      // TEXPANDS(tile, scalar): Fill entire tile with a scalar value.
      // Equivalent to: tile[:] = scalar  (like torch.full_like(tile, scalar))
      TEXPANDS(acc_ub, 0.0f);
      pipe_barrier(PIPE_V);
      for (int32_t i = valid; i < ChunkSize; ++i) {
        UbND<float, 1, HTC> s_row_i;
        TASSIGN(s_row_i, SUbAddr + i * RowBytes);
        TMOV(s_row_i, acc_ub);
        pipe_barrier(PIPE_V);
      }

      // ── DMA: store g_sum from UB → GM (MTE3 pipe) ────────────────────
      // ── Synchronization: Vec → MTE3 ───────────────────────────────────
      // Vec signals MTE3 that computation is done and UB data is ready to store.
      // MTE3 (DMA store engine) waits for this before reading UB for TSTORE.
      // Without this sync, MTE3 might read stale/partial data from UB.
      // Vec → MTE3 sync: ensure Vec writes to UB are visible before DMA
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

      {
        GmShape ss; ss.shape[3] = valid; ss.shape[4] = NumHeads;
        GmFloat gs_gm(g_sum_ptr + chunk_start * NumHeads, ss);
        UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC>
            s_store(valid, NumHeads);
        TASSIGN(s_store, SUbAddr);
        // TSTORE(gm_tensor, ub_tile): DMA transfer from UB → GM.
        // Equivalent to: gm_tensor[:valid, :NumHeads] = ub_tile[:valid, :NumHeads]
        // Async on MTE3 pipe. Must sync (Vec→MTE3) before calling, and sync
        // (MTE3→Vec) after if reusing the same UB region.
        TSTORE(gs_gm, s_store);
      }
      // ── Synchronization: MTE3 → Vec ───────────────────────────────────
      // MTE3 signals Vec that the DMA store is complete and UB can be reused.
      // Vec waits before starting the next iteration's TLOAD into the same UB region.
      // Without this, the next TLOAD could overwrite data still being stored.
      // MTE3 → Vec sync: wait for DMA store before reusing UB next iter
      set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
  }
  // ── Variable-length sequence path (cu_seqlens != nullptr) ─────────────
  else {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t c = 0; c < nc; ++c) {
        if (gi % static_cast<int64_t>(block_num) ==
            static_cast<int64_t>(cid)) {
          int64_t chunk_start = bos + c * ChunkSize;
          int64_t remaining = slen - c * ChunkSize;
          int32_t valid = static_cast<int32_t>(
              remaining < ChunkSize ? remaining : ChunkSize);

          // Load g chunk from GM → UB, zero-padded
          {
            GmShape gs; gs.shape[3] = valid; gs.shape[4] = NumHeads;
            GmFloat g_gm(g_ptr + chunk_start * NumHeads, gs);
            UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC, PadValue::Zero>
                g_load(valid, NumHeads);
            TASSIGN(g_load, GUbAddr);
            TLOAD(g_load, g_gm);
            if (valid != ChunkSize || NumHeads != HTC) {
              UbND<float, ChunkSize, HTC, ChunkSize, HTC, PadValue::Zero>
                  g_pad;
              TASSIGN(g_pad, GUbAddr);
              TFILLPAD_INPLACE(g_pad, g_load);
            }
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

          // Prefix sum: acc = g[0]; g_sum[0] = acc
          UbND<float, 1, HTC> g_row_0;
          TASSIGN(g_row_0, GUbAddr);
          TMOV(acc_ub, g_row_0);
          pipe_barrier(PIPE_V);

          UbND<float, 1, HTC> s_row_0;
          TASSIGN(s_row_0, SUbAddr);
          TMOV(s_row_0, acc_ub);
          pipe_barrier(PIPE_V);

          // acc += g[i]; g_sum[i] = acc
          for (int32_t i = 1; i < valid; ++i) {
            UbND<float, 1, HTC> g_row_i;
            TASSIGN(g_row_i, GUbAddr + i * RowBytes);
            TADD(acc_ub, acc_ub, g_row_i);
            pipe_barrier(PIPE_V);

            UbND<float, 1, HTC> s_row_i;
            TASSIGN(s_row_i, SUbAddr + i * RowBytes);
            TMOV(s_row_i, acc_ub);
            pipe_barrier(PIPE_V);
          }

          // Zero-fill padding rows
          TEXPANDS(acc_ub, 0.0f);
          pipe_barrier(PIPE_V);
          for (int32_t i = valid; i < ChunkSize; ++i) {
            UbND<float, 1, HTC> s_row_i;
            TASSIGN(s_row_i, SUbAddr + i * RowBytes);
            TMOV(s_row_i, acc_ub);
            pipe_barrier(PIPE_V);
          }

          // Store g_sum to GM
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

          {
            GmShape ss; ss.shape[3] = valid; ss.shape[4] = NumHeads;
            GmFloat gs_gm(g_sum_ptr + chunk_start * NumHeads, ss);
            UbND<float, ChunkSize, HTC, DYNAMIC, DYNAMIC>
                s_store(valid, NumHeads);
            TASSIGN(s_store, SUbAddr);
            TSTORE(gs_gm, s_store);
          }
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        gi++;
      }
    }
  }
#endif
}

// ── Device-side kernel entry point ─────────────────────────────────
// extern "C" __global__ AICORE: marks this as an NPU kernel function
//   (like __global__ in CUDA). Each AI core runs one instance of this function.
// Parameters are passed as uint8_t* (raw bytes) and reinterpret_cast'd to
// typed pointers — this is the standard NPU kernel calling convention.
extern "C" __global__ AICORE void launch_cumsum(
    __gm__ uint8_t *g_ptr, __gm__ uint8_t *g_sum_ptr,
    __gm__ uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len,
    uint64_t ffts_addr)
{
  cumsum_kernel<GDN_H, GDN_C>(
      reinterpret_cast<__gm__ float *>(g_ptr),
      reinterpret_cast<__gm__ float *>(g_sum_ptr),
      reinterpret_cast<__gm__ int32_t *>(cu_seqlens),
      batch_size, seq_len, ffts_addr);
}

// ── Host-side launcher (called from Python via ctypes) ────────────
// call_kernel(): CPU function that launches the NPU kernel.
//   block_dim = number of AI cores to use (like CUDA grid size)
//   stream = NPU stream for async execution (like CUDA stream)
//   rtGetC2cCtrlAddr: gets the FFTS control address for cross-core sync
//   <<<block_dim, nullptr, stream>>>: NPU kernel launch syntax (like CUDA <<<>>>)
extern "C" void call_kernel(
    uint32_t block_dim, void *stream,
    uint8_t *g_ptr, uint8_t *g_sum_ptr, uint8_t *cu_seqlens,
    int64_t batch_size, int64_t seq_len)
{
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  launch_cumsum<<<block_dim, nullptr, stream>>>(
      g_ptr, g_sum_ptr, cu_seqlens, batch_size, seq_len, fftsAddr);
}
