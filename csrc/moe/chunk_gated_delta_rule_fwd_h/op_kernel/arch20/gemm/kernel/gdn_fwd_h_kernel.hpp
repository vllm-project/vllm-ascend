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
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_utils/gemm/hand_mmad_310p.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_h.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

using _0 = tla::Int<0>;
using _1 = tla::Int<1>;
using _2 = tla::Int<2>;
using _4 = tla::Int<4>;
using _8 = tla::Int<8>;
using _16 = tla::Int<16>;
using _32 = tla::Int<32>;
using _64 = tla::Int<64>;
using _128 = tla::Int<128>;
using _256 = tla::Int<256>;
using _512 = tla::Int<512>;
using _1024 = tla::Int<1024>;
using _2048 = tla::Int<2048>;
using _4096 = tla::Int<4096>;
using _8192 = tla::Int<8192>;
using _16384 = tla::Int<16384>;
using _32768 = tla::Int<32768>;
using _65536 = tla::Int<65536>;




#include "kernel_operator.h"
using namespace Catlass;
using namespace tla;

namespace Catlass::Gemm::Kernel {

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE
>
class GDNFwdHKernel {
public:
    
    using ArchTag = Arch::AtlasA2;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;

    // cube 1

    // cube 2

    // vec 1
    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnew;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType>;

    // vec 2

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;
    
    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;

    
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
    static constexpr uint32_t UB_UPD_NDOUT = 144 * 1024;  // final-state scratch
    // Scalar exp scratch: must NOT sit in vnew's pong region (a scalar write
    // there raced the outstanding v_update store -- fin came back undecayed).
    // The ND-out window is dead at hoist time: the previous body's h store was
    // drained by the MTE3_MTE2 guard just above.
    static constexpr uint32_t UB_GHOIST    = UB_UPD_NDOUT;

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
        // NOTE: the runtime registers this binary with mixType=0 (plain launch),
        // so exactly one instance runs per core and GetSubBlockIdx() is NOT
        // meaningful here -- an early-return gate on it killed every core once
        // the host stack was rebuilt. The epilogues are hardcoded to a single
        // subblock; no gate is needed.
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
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/false,
                                   /*A_COL_MAJOR=*/false, /*B_FROM_L1=*/true, /*B_NZ_GM=*/false,
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
                    // v_update (B) loaded once.
                    {
                        AscendC::Nd2NzParams pb;
                        pb.ndNum = 1;
                        pb.nValue = stage2Offsets.blockTokens;
                        pb.dValue = vHeadDim;
                        pb.srcNdMatrixStride = 0;
                        pb.srcDValue = vHeadDim;
                        pb.dstNzC0Stride = (stage2Offsets.blockTokens + 15) / 16 * 16;
                        pb.dstNzNStride = 1;
                        pb.dstNzMatrixStride = 0;
                        auto l1B = resource.l1Buf.template GetBufferByByte<half>(HM_L1B_OFFSET);
                        AscendC::DataCopy(l1B, gmVUpdateWorkspace[stage2Offsets.vWorkOffset], pb);
                    }
                    M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false,
                                       /*A_FROM_L1=*/false, /*A_COL_MAJOR=*/true,
                                       /*B_FROM_L1=*/true, /*B_NZ_GM=*/false,
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