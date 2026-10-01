/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#define CATLASS_ARCH 2201
#define CATLASS_UNIFIED_CORE 1

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
#include "kernel_utils/gemm/hand_mmad_310p.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "../block/block_scheduler_gdn_fwd_h.hpp"

#include "kernel_operator.h"
using namespace Catlass;

namespace Catlass::Gemm::Kernel {

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE
>
class GDNFwdHKernel {
public:
    
    // NOTE on ArchTag: Catlass's Arch::AtlasA2 is the 910B descriptor and
    // understates this chip (UB 192K vs a real 248K usable, L1 512K vs 1M, L0C
    // 128K vs 256K -- Ascend310P3.ini + CANN's __NPU_ARCH__==2002 branch). It is
    // NOT swapped here: the vnew epilogue takes its ArchTag from
    // DispatchPolicy::ArchTag and its ctor takes Arch::Resource<ArchTag>&, so
    // changing it changes the Resource<> instantiation and breaks that call.
    // Fixing it properly means changing the epilogue policy too; the staging
    // below avoids needing the extra UB at all.
    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;

    // Only the vnew epilogue still takes GemmType wrappers; the hand mmads use
    // the element types directly.
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;

    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnew;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType>;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    
    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    uint32_t initalStateStride0;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vUpdateWorkspaceOffset;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;
    
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    
    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    CubeScheduler cubeBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline GDNFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, 
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        // Contiguous state layout: stride equals vHeadDim (initalStateStride0 attr removed).
        initalStateStride0 = gdnFwdHTilingData->vHeadDim;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;
        
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
    }
    
    __aicore__ inline void Process() {
        ProcessUnifiedCore();
    }

    // ---- hand-mmad scratch (unified core only) --------------------------------
    // The Catlass BlockMmadTla path is gone from ProcessUnifiedCore, so L1 and the
    // low UB are free outside the epilogues' own windows. Stage (NZ) and the ND
    // copy-out sit in [0, 128K): the epilogues' MTE3_MTE2 guards order their MTE2
    // loads after our MTE3 reads, V-pipe order covers the epilogues' calc buffer,
    // and MTE3-in-order covers overlap with their outstanding GM stores.
    static constexpr uint32_t HM_STAGE_OFFSET = 0;          // <=128x128 f32 = 64 KB
    static constexpr uint32_t HM_ND_OFFSET    = 64 * 1024;  // same max
    static constexpr uint32_t HM_L1A_OFFSET   = 0;
    static constexpr uint32_t HM_L1B_OFFSET   = 64 * 1024;
    // ---- L1-resident h ------------------------------------------------------
    // The recurrence state h[k x v] never leaves the chip between chunks: it
    // lives in L1 as zN f16, one bank per interleaved head (the scheduler
    // alternates two heads per core). Cube1 takes it as its B operand with no
    // load at all; the update phase pulls m-tiles L1->UB (MTE1), computes the
    // new state NZ-native against the cube staging buffer, writes it back
    // (MTE3) and deformats an f16 ND copy only for the gmH output store.
    // Chunk 0 of a task bootstraps the bank straight from gmH with the same
    // Nd2Nz the old GM path used.
    static constexpr uint32_t HRES_L1_SLOT   = 48 * 1024;   // zN(192,128) f16 max
    static constexpr uint32_t HRES_L1_OFFSET = 128 * 1024;
    // Update-phase UB map (all below vnew's pong region lifetimes):
    // Single-tile update (m = kHeadDim <= 192): the h_work stage grows to
    // 96 KB at [0,96K); the resident h tile sits above it. No f32 calc buffer:
    // Axpy fuses h*scale straight into the stage. [144K,192K) is scratch for
    // the (rare) final_state deformats.
    static constexpr uint32_t UB_UPD_H16   = 96 * 1024;   // f16 tile, <=48 KB
    static constexpr uint32_t UB_UPD_NDOUT = 144 * 1024;  // final-state + exp(g_last) scratch

    // ---- hand-rolled GM->L1 NZ staging --------------------------------------
    // Staged through the LOW half of the cube's own staging window, which is
    // dead at prefetch time: each prefetch runs immediately before the mmad that
    // writes HM_STAGE via its L0C->UB copy, and the previous task's staging was
    // already deformatted out to HM_ND and consumed by the epilogue. The closing
    // MTE3_V pair below orders our MTE3 reads of the scratch before that later
    // V-pipe staging write. Same trick chunk_fwd_o uses (it stages through
    // Vec1's own buffer, "dead at the beginning of a body").
    // Two 24KB slots (24KB = 64 rows x 192 cols f16, the largest tile here) so
    // back-to-back prefetches do not serialise on one scratch: 48K <= the 64K
    // HM_STAGE window.
    static constexpr uint32_t UB_PF_OFFSET = HM_STAGE_OFFSET;
    static constexpr uint32_t UB_PF_SLOT   = 24 * 1024;
    static_assert(UB_PF_OFFSET + 2 * UB_PF_SLOT <= HM_ND_OFFSET,
                  "fwd_h: prefetch staging must stay inside the HM_STAGE window");

    // Replaces `DataCopy(l1, gm, Nd2NzParams)`, which on m200 is NOT a DMA: the
    // dav_m200 library emulates it in 64Bx64B blocks, each a GM->UB read, a
    // masked-vadds transpose on V and a UB->L1 store on MTE3, serialised by
    // three flag pairs per block -- 26-42x a plain contiguous load
    // (yaml_spec B7.2). Hand-rolling is 4.1-4.3x faster and, more importantly
    // here, collapses ~340 scalar issue cyc per call to ~170-200: fwd_h is
    // SCALAR-bound (device: 239.5-269.8us of a 416.5us kernel) and intrinsic
    // issue cost is independent of transfer size, so call count is the budget.
    //
    // Same routine as chunk_fwd_o's PrefetchTileNZ. The pad rows are zero-filled
    // so a partial tile's NZ padding is deterministic rather than whatever the
    // previous task left in the slot.
    __aicore__ inline void PrefetchTileNZ(
        AscendC::GlobalTensor<half> src, uint32_t rows, uint32_t cols,
        uint32_t l1Offset, uint32_t pfSlot) {
        const uint32_t mAl = M200Gemm::HmRoundUp16(rows);
        auto scratch = resource.ubBuf.template GetBufferByByte<half>(
            UB_PF_OFFSET + pfSlot * UB_PF_SLOT);
        auto dst = resource.l1Buf.template GetBufferByByte<half>(l1Offset);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID6);
        // GM rows are contiguous (ld == cols), so the valid part is one burst.
        AscendC::DataCopy(scratch, src, rows * cols);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID6);
        if (mAl != rows) {
            AscendC::Duplicate<half>(scratch[rows * cols], static_cast<half>(0),
                                     (mAl - rows) * cols);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
        }
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(mAl);
        p.blockLen = 1;
        p.srcStride = static_cast<uint16_t>(cols / 16 - 1);
        p.dstStride = 0;
        for (uint32_t column = 0; column < cols / 16; ++column) {
            AscendC::DataCopy(dst[column * mAl * 16], scratch[column * 16], p);
        }
        // The consumer is MTE1 (HandMmad's L1->L0 load) and these call sites pass
        // NO_MTE1_MTE2, so HandMmad issues no MTE2_MTE1 of its own to chain
        // through: the MTE3->MTE1 order is ours to provide.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID6);
        // Protect the scratch against the next prefetch and any later V reader.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
    }

    // m-tile of the resident bank <-> UB, one strided descriptor each way.
    // zN(kR, v): fractal column nf stride kR*16 elems; a tile is nFracs runs of
    // mActual*16 elems starting at mOff*16.
    __aicore__ inline void ExtractResidentH(uint32_t slot, uint32_t kR,
                                            uint32_t mOff, uint32_t mActual, uint32_t nFracs) {
        auto src = resource.l1Buf.template GetBufferByByte<half>(
            HRES_L1_OFFSET + slot * HRES_L1_SLOT);
        auto dst = resource.ubBuf.template GetBufferByByte<half>(UB_UPD_H16);
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(nFracs);
        p.blockLen = static_cast<uint16_t>(mActual);            // mActual*16 f16 / 32B
        p.srcStride = static_cast<uint16_t>(kR - mActual);
        p.dstStride = 0;
        AscendC::DataCopy(dst, src[mOff * 16], p);
    }
    __aicore__ inline void WritebackResidentH(uint32_t slot, uint32_t kR,
                                              uint32_t mOff, uint32_t mActual, uint32_t nFracs) {
        auto src = resource.ubBuf.template GetBufferByByte<half>(UB_UPD_H16);
        auto dst = resource.l1Buf.template GetBufferByByte<half>(
            HRES_L1_OFFSET + slot * HRES_L1_SLOT);
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(nFracs);
        p.blockLen = static_cast<uint16_t>(mActual);
        p.srcStride = 0;
        p.dstStride = static_cast<uint16_t>(kR - mActual);
        AscendC::DataCopy(dst[mOff * 16], src, p);
    }
    // NZ -> ND for f16 (h_out store) or f32 (final_state store), same walk as
    // DeformatStagingToUb. UB->UB rides V; caller supplies src/dst offsets.
    template <typename T>
    __aicore__ inline void DeformatNzToNd(uint32_t dstOff, uint32_t srcOff,
                                          uint32_t mActual, uint32_t nActual) {
        auto src = resource.ubBuf.template GetBufferByByte<T>(srcOff);
        auto dst = resource.ubBuf.template GetBufferByByte<T>(dstOff);
        uint32_t mAligned = (mActual + 15) / 16 * 16;
        uint32_t nAligned = (nActual + 15) / 16 * 16;
        uint32_t mFracs = mAligned / 16;
        uint32_t nFracs = nAligned / 16;
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(mAligned);
        p.blockLen = static_cast<uint16_t>(16 * sizeof(T) / 32);
        p.srcStride = 0;
        p.dstStride = static_cast<uint16_t>((nAligned - 16) * sizeof(T) / 32);
        for (uint32_t nf = 0; nf < nFracs; ++nf) {
            AscendC::DataCopy(dst[nf * 16], src[nf * mFracs * 256], p);
        }
    }

    // NZ cube staging -> ND, one strided descriptor per Z-column (same move as
    // chunk_fwd_o's DeformatL0CStagingToUb; MTE3, reads after HandMmad's V_MTE3
    // drain, feeds the MTE3 GM store in pipe order).
    // v_work: NZ staging -> ND at HM_ND_OFFSET for the (still ND) vnew epilogue.
    // MTE3_V before: epilogue GM stores still read the dst window. V_MTE3 after:
    // the ND store... none remains -- kept so the next MTE3 reader (none today,
    // vnew consumes on V) stays ordered if one returns.
    __aicore__ inline void DeformatStagingToUb(uint32_t mActual, uint32_t nActual) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        DeformatNzToNd<float>(HM_ND_OFFSET, HM_STAGE_OFFSET, mActual, nActual);
        // No V_MTE3 tail: no MTE3 reader of the ND window remains, and
        // reduce_flags (fwd_h_v3.yaml) proves the epilogue's kept fences
        // order everything that follows.
    }

    __aicore__ inline void ProcessUnifiedCore() {
        // This integration branch still uses the legacy 310P generated wrapper
        // and KERNEL_TYPE_MIX_AIC_1_2.  Its two subblocks have separate UBs and
        // would otherwise duplicate every hand-mmad.  The newer upstream base
        // used by PR #17422 registers this binary as a plain launch and removed
        // the guard; retain it until this branch migrates that runtime ABI.
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();
            GDNFwdHOffsets& stage1Offsets = cubeBlockScheduler.GetStage1Offsets();

            // CUBE1: v_work = w @ h[i], hand mmad. h[i] and the workspaces are
            // MTE3-written (previous chunk's Vec2 / the initial-state pre-loop),
            // so drain MTE3 before the GM->L1 loads.
            if (cubeBlockScheduler.NeedProcessStage1()) {
                // Drain MTE3 first: the resident bank's last writeback and the
                // workspaces are MTE3-written; the MTE2 A load chains MTE1 after
                // it through HandMmad's MTE2_MTE1 pair.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                if (stage1Offsets.isInitialState) {
                    // Chunk 0: the host pre-fills h slot 0 with the zN image of
                    // initial_state (or leaves the zeros the binding allocates),
                    // so seeding the resident bank is one flat 48 KB burst.
                    auto bank = resource.l1Buf.template GetBufferByByte<half>(
                        HRES_L1_OFFSET + stage1Offsets.slot * HRES_L1_SLOT);
                    AscendC::DataCopy(bank, gmH[stage1Offsets.hSrcOffset],
                                      kHeadDim * vHeadDim);
                }
                // A (w) hand-rolled into L1 instead of HandMmad's Nd2Nz: the
                // library path is the 26-42x emulation (see PrefetchTileNZ).
                // C0Stride matches -- HandMmad would have used
                // l1aC0Stride = mR = HmRoundUp16(blockTokens), which is exactly
                // the blockCount PrefetchTileNZ writes.
                PrefetchTileNZ(gmW[stage1Offsets.wOffset],
                               stage1Offsets.blockTokens, kHeadDim,
                               HM_L1A_OFFSET, /*pfSlot=*/0);
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/true,
                                   /*A_COL_MAJOR=*/false, /*B_FROM_L1=*/true,
                                   /*LEAN_TAIL=*/true, /*NO_MTE1_MTE2=*/true, /*NO_M_MTE1=*/true>(
                    resource,
                    gmW[stage1Offsets.wOffset], kHeadDim,
                    gmH[stage1Offsets.hSrcOffset], vHeadDim,
                    stage1Offsets.blockTokens, vHeadDim, kHeadDim,
                    HM_L1A_OFFSET,
                    HRES_L1_OFFSET + stage1Offsets.slot * HRES_L1_SLOT,
                    HM_STAGE_OFFSET, 0);
                DeformatStagingToUb(stage1Offsets.blockTokens, vHeadDim);
                // v_work stays in UB at HM_ND_OFFSET; Vec1 consumes it in place.
            }

            // VEC1: v_new epilogue
            if (cubeBlockScheduler.NeedProcessStage1()) {
                epilogueGDNFwdHVnew(
                    gmV[stage1Offsets.uvOffset], gmVUpdateWorkspace[stage1Offsets.vWorkOffset],
                    gmG[stage1Offsets.gOffset], gmU[stage1Offsets.uvOffset], HM_ND_OFFSET,
                    stage1Offsets.blockTokens, kHeadDim, vHeadDim, cubeBlockScheduler.cube1Done
                );
            }

            if (cubeBlockScheduler.iterId > 1) {
                GDNFwdHOffsets& stage2Offsets = cubeBlockScheduler.GetStage2Offsets();

                // CUBE2 + VEC2, single fused m=192 tile. h_work never touches
                // GM: mmad -> NZ stage (96 KB) -> Axpy(stage += scale*h16) ->
                // cast -> bank writeback + strided zN h store. One tile means
                // half the fences and descriptors the m-loop paid.
                if (cubeBlockScheduler.NeedProcessStage2()) {
                    // v_update and the epilogue outputs are MTE3-written just
                    // above: drain MTE3 into MTE2 (loads) and V (our writes).
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                    // (MTE3_V drain dropped: reduce_flags proves the V stream is
                    // already ordered behind the epilogue's kept store fences.)
                    // exp(g_last) once per chunk (scalar dance; the GM read is
                    // L2-warm here -- an MTE2 hoist measured neutral-to-worse).
                    float hDecayScale;
                    {
                        AscendC::LocalTensor<float> gl =
                            resource.ubBuf.template GetBufferByByte<float>(UB_UPD_NDOUT);
                        // The previous head's final-state deformat writes this
                        // aliased UB window on V. Order that write before the
                        // scalar pipe reuses the first word for g_last.
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
                        gl.SetValue(0, gmG[stage2Offsets.gOffset].GetValue(stage2Offsets.blockTokens - 1));
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
                        AscendC::Exp(gl, gl, 1);
                        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID5);
                        hDecayScale = gl.GetValue(0);
                        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID5);
                    }
                    // v_update (B) loaded once. Hand-rolled: the Nd2NzParams form
                    // this replaced was the 26-42x dav_m200 emulation, and its
                    // dstNzC0Stride was (blockTokens+15)/16*16 == the mAl
                    // blockCount PrefetchTileNZ writes, so the L1 image is
                    // byte-identical.
                    PrefetchTileNZ(gmVUpdateWorkspace[stage2Offsets.vWorkOffset],
                                   stage2Offsets.blockTokens, vHeadDim,
                                   HM_L1B_OFFSET, /*pfSlot=*/1);
                    M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false,
                                       /*A_FROM_L1=*/false, /*A_COL_MAJOR=*/true,
                                       /*B_FROM_L1=*/true,
                                       /*LEAN_TAIL=*/true, /*NO_MTE1_MTE2=*/true, /*NO_M_MTE1=*/true>(
                        resource,
                        gmK[stage2Offsets.wkOffset], kHeadDim,
                        gmVUpdateWorkspace[stage2Offsets.vWorkOffset], vHeadDim,
                        kHeadDim, vHeadDim, stage2Offsets.blockTokens,
                        HM_L1A_OFFSET, HM_L1B_OFFSET, HM_STAGE_OFFSET, 0);
                    uint32_t kR = (kHeadDim + 15) / 16 * 16;
                    uint32_t nFr = vHeadDim / 16;
                    uint32_t elems = kHeadDim * vHeadDim;
                    uint32_t slot2 = stage2Offsets.slot;
                    // (V_MTE1 + MTE3_MTE1 pairs dropped: reduce_flags proves the
                    // extract is ordered through the mmad's kept M/V chain and
                    // the stage-top MTE3_MTE2 drain.)
                    ExtractResidentH(slot2, kR, 0, kHeadDim, nFr);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_V>(EVENT_ID5);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_V>(EVENT_ID5);
                    AscendC::LocalTensor<float> stageT =
                        resource.ubBuf.template GetBufferByByte<float>(HM_STAGE_OFFSET);
                    AscendC::LocalTensor<half> h16 =
                        resource.ubBuf.template GetBufferByByte<half>(UB_UPD_H16);
                    // h_new = h_work + scale * h  (f32 dst, f16 src, fused)
                    AscendC::Axpy(stageT, h16, (half)hDecayScale, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(h16, stageT, AscendC::RoundMode::CAST_NONE, elems);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                    WritebackResidentH(slot2, kR, 0, kHeadDim, nFr);
                    if (stage2Offsets.isFinalState && storeFinalState) {
                        if constexpr (std::is_same<ElementFinalState, float>::value) {
                            // ND f32 from the (still-live) f32 stage, staged over
                            // the h16 scratch after the writeback consumed it.
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            DeformatNzToNd<float>(UB_UPD_H16, HM_STAGE_OFFSET, kHeadDim, vHeadDim);
                            AscendC::LocalTensor<float> ndF =
                                resource.ubBuf.template GetBufferByByte<float>(UB_UPD_H16);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                            AscendC::DataCopy(gmFinalState[stage2Offsets.finalStateOffset], ndF, elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                        } else {
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            DeformatNzToNd<half>(UB_UPD_NDOUT, UB_UPD_H16, kHeadDim, vHeadDim);
                            AscendC::LocalTensor<ElementFinalState> ndH =
                                resource.ubBuf.template GetBufferByByte<ElementFinalState>(UB_UPD_NDOUT);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                            AscendC::DataCopy(gmFinalState[stage2Offsets.finalStateOffset], ndH, elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                        }
                    } else {
                        // gmH zN image: strided store straight from the tile.
                        AscendC::DataCopyParams hp;
                        hp.blockCount = static_cast<uint16_t>(nFr);
                        hp.blockLen = static_cast<uint16_t>(kHeadDim);
                        hp.srcStride = 0;
                        hp.dstStride = static_cast<uint16_t>(kR - kHeadDim);
                        AscendC::DataCopy(gmH[stage2Offsets.hDstOffset], h16, hp);
                    }
                }
            }
        }
    }

};

}
