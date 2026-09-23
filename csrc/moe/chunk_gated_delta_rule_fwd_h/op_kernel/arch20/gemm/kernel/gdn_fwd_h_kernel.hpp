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
    static constexpr uint32_t UB_UPD_CALC  = 64 * 1024;   // f32, <=64 KB
    static constexpr uint32_t UB_UPD_H16   = 128 * 1024;  // f16 tile, <=32 KB
    static constexpr uint32_t UB_UPD_NDOUT = 160 * 1024;  // f16 ND out, <=32 KB
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
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
    }

    __aicore__ inline void ProcessUnifiedCore() {
        // The 1:2 task type launches two subblock instances per core with
        // SEPARATE UBs: they duplicated every cube mmad and could only hand
        // data to the epilogues through GM (a UB-resident v_work reached
        // subblock 0 chunk-shifted). Run everything on instance 0; the
        // epilogues are hardcoded to a single subblock to match. Changing the
        // task type itself breaks the generated tilingkey wrapper, so gate at
        // runtime instead.
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);

        if (useInitialState) {
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
            AscendC::LocalTensor<ElementInitialState> stateUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
            AscendC::LocalTensor<ElementH> hUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
            uint32_t totalChunks = isVariedLen ? cubeBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
            uint32_t stateBlockSize = kHeadDim * vHeadDim;
            uint32_t pingpongFlag = 1;
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            for (uint32_t shapeBatchIdx = 0; shapeBatchIdx < shapeBatch; shapeBatchIdx++) {
                for (uint32_t vHeadIdx = 0; vHeadIdx < vNumHead; vHeadIdx++) {
                    for (uint32_t tokenBatchIdx = 0; tokenBatchIdx < cubeBlockScheduler.tokenBatch; tokenBatchIdx++) {
                        uint32_t batchIdx = isVariedLen ? tokenBatchIdx : shapeBatchIdx;
                        uint32_t chunkOffset = isVariedLen ? gmNumChunks.GetValue(tokenBatchIdx) : 0;
                        uint32_t initialStateOffset = (batchIdx * vNumHead + vHeadIdx) * stateBlockSize;
                        uint32_t hOffset = (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset) * stateBlockSize;
                        AscendC::LocalTensor<ElementInitialState> stateUbTensor = pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                        AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                        auto event_id = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                        if constexpr(!std::is_same<ElementInitialState, ElementH>::value) {
                            AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                            AscendC::Cast(hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_NONE, stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                            AscendC::DataCopy(gmH[hOffset], hUbTensor, stateBlockSize);
                        } else {
                            AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                            AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateBlockSize);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                        pingpongFlag = 1 - pingpongFlag;
                    }
                }
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }

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
                    // Chunk 0: seed the resident bank from gmH (same Nd2Nz the
                    // old GM path used, just landing in the bank).
                    AscendC::Nd2NzParams ph;
                    ph.ndNum = 1;
                    ph.nValue = kHeadDim;  ph.dValue = vHeadDim;
                    ph.srcNdMatrixStride = 0;
                    ph.srcDValue = vHeadDim;
                    ph.dstNzC0Stride = (kHeadDim + 15) / 16 * 16;
                    ph.dstNzNStride = 1;  ph.dstNzMatrixStride = 0;
                    auto bank = resource.l1Buf.template GetBufferByByte<half>(
                        HRES_L1_OFFSET + stage1Offsets.slot * HRES_L1_SLOT);
                    AscendC::DataCopy(bank, gmH[stage1Offsets.hSrcOffset], ph);
                }
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/false,
                                   /*A_COL_MAJOR=*/false, /*B_FROM_L1=*/true>(
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

                // CUBE2 + VEC2, fused per m-tile (m <= 128). h_work never
                // touches GM: mmad -> NZ stage -> deformat ND @HM_ND_OFFSET ->
                // update epilogue reads it in place, casts the h output over it
                // and stores straight to gmH/final_state. v_update was
                // MTE3-written by Vec1 just above: drain MTE3 before the loads.
                if (cubeBlockScheduler.NeedProcessStage2()) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
                    // exp(g_last) once per chunk: one GM scalar read + one
                    // 1-element vector Exp + S<->V handshake, shared by the tiles.
                    float hDecayScale;
                    {
                        AscendC::LocalTensor<float> gl =
                            resource.ubBuf.template GetBufferByByte<float>(UB_GHOIST);
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
                    // v_update (B) is identical for every m tile: load it once.
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
                    uint32_t mLoopC2 = (kHeadDim + 127) / 128;
                    for (uint32_t mIdx = 0; mIdx < mLoopC2; ++mIdx) {
                        uint32_t mOff = mIdx * 128;
                        uint32_t mTail = kHeadDim - mOff;
                        uint32_t mActual = (mTail < 128) ? mTail : 128;
                        M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false,
                                           /*A_FROM_L1=*/false, /*A_COL_MAJOR=*/true,
                                           /*B_FROM_L1=*/true>(
                            resource,
                            gmK[stage2Offsets.wkOffset + mOff], kHeadDim,
                            gmVUpdateWorkspace[stage2Offsets.vWorkOffset], vHeadDim,
                            mActual, vHeadDim, stage2Offsets.blockTokens,
                            HM_L1A_OFFSET, HM_L1B_OFFSET, HM_STAGE_OFFSET, 0);
                        // ---- NZ-native update on the resident bank ----
                        uint32_t kR = (kHeadDim + 15) / 16 * 16;
                        uint32_t nFr = vHeadDim / 16;
                        uint32_t elems = mActual * vHeadDim;
                        uint32_t slot2 = stage2Offsets.slot;
                        // Prior V readers (previous tile's casts/deformat) and MTE3
                        // readers (writeback, h store) of the H16 window must drain
                        // before MTE1 rewrites it.
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE1>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE1>(EVENT_ID5);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID5);
                        ExtractResidentH(slot2, kR, mOff, mActual, nFr);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_V>(EVENT_ID5);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_V>(EVENT_ID5);
                        AscendC::LocalTensor<float> calc =
                            resource.ubBuf.template GetBufferByByte<float>(UB_UPD_CALC);
                        AscendC::LocalTensor<half> h16 =
                            resource.ubBuf.template GetBufferByByte<half>(UB_UPD_H16);
                        AscendC::LocalTensor<float> stageT =
                            resource.ubBuf.template GetBufferByByte<float>(HM_STAGE_OFFSET);
                        AscendC::Cast(calc, h16, AscendC::RoundMode::CAST_NONE, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Muls(calc, calc, hDecayScale, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Add<float>(calc, calc, stageT, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Cast(h16, calc, AscendC::RoundMode::CAST_NONE, elems);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                        WritebackResidentH(slot2, kR, mOff, mActual, nFr);
                        if (stage2Offsets.isFinalState && storeFinalState) {
                            // final_state replaces the h store on the last chunk.
                            if constexpr (std::is_same<ElementFinalState, float>::value) {
                                // f32 ND from calc, staged over the (dead) cube
                                // staging window; MTE3_V after keeps the next
                                // tile's L0C->UB copy off the outstanding read.
                                DeformatNzToNd<float>(HM_STAGE_OFFSET, UB_UPD_CALC, mActual, vHeadDim);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                                AscendC::DataCopy(gmFinalState[stage2Offsets.finalStateOffset + mOff * vHeadDim],
                                                  stageT, elems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            } else {
                                DeformatNzToNd<half>(UB_UPD_NDOUT, UB_UPD_H16, mActual, vHeadDim);
                                AscendC::LocalTensor<ElementFinalState> ndOut =
                                    resource.ubBuf.template GetBufferByByte<ElementFinalState>(UB_UPD_NDOUT);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                                AscendC::DataCopy(gmFinalState[stage2Offsets.finalStateOffset + mOff * vHeadDim],
                                                  ndOut, elems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            }
                        } else {
                            DeformatNzToNd<half>(UB_UPD_NDOUT, UB_UPD_H16, mActual, vHeadDim);
                            AscendC::LocalTensor<ElementH> ndOut =
                                resource.ubBuf.template GetBufferByByte<ElementH>(UB_UPD_NDOUT);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
                            AscendC::DataCopy(gmH[stage2Offsets.hDstOffset + mOff * vHeadDim],
                                              ndOut, elems);
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
                        }
                    }
                }
            }
        }
    }

};

}