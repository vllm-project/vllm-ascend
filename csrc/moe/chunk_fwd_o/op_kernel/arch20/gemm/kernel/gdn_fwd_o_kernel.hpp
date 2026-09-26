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
#include "kernel_utils/gemm/hand_mmad_310p.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm_coord.hpp"
#include "../block/block_scheduler_gdn_fwd_o.hpp"

#include "kernel_operator.h"
using namespace Catlass;

namespace Catlass::Gemm::Kernel {

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename WORKSPACE_TYPE
>
class GDNFwdOKernel {
public:
    
    using ArchTag = Arch::AtlasA2;
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;

    // The Catlass BlockMmadTla / TileCopy stack is gone (hand mmads only);
    // the element types are the template inputs directly.
    using ElementQ = INPUT_TYPE;
    using ElementK = INPUT_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementVNEW = INPUT_TYPE;
    using ElementG = G_TYPE;

    uint32_t shapeBatch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    float scale;
    uint32_t numChunks;
    uint32_t isVariedLen;
    uint32_t tokenBatch;
    
    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementVNEW> gmV;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementVNEW> gmO;

    CubeScheduler cubeBlockScheduler;

    Arch::Resource<ArchTag> resource;

    __aicore__ inline GDNFwdOKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR h, GM_ADDR g, 
        GM_ADDR cu_seqlens, GM_ADDR chunk_offsets, GM_ADDR o, GM_ADDR tiling, GM_ADDR user) {
        
        __gm__ ChunkFwdOTilingData *__restrict gdnFwdOTilingData = reinterpret_cast<__gm__ ChunkFwdOTilingData *__restrict>(tiling);

        shapeBatch = gdnFwdOTilingData->shapeBatch;
        seqlen = gdnFwdOTilingData->seqlen;
        kNumHead = gdnFwdOTilingData->kNumHead;
        vNumHead = gdnFwdOTilingData->vNumHead;
        kHeadDim = gdnFwdOTilingData->kHeadDim;
        vHeadDim = gdnFwdOTilingData->vHeadDim;
        scale = gdnFwdOTilingData->scale;
        chunkSize = gdnFwdOTilingData->chunkSize;
        isVariedLen = gdnFwdOTilingData->isVariedLen;
        tokenBatch = gdnFwdOTilingData->tokenBatch;

        gmQ.SetGlobalBuffer((__gm__ ElementQ *)q);
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmV.SetGlobalBuffer((__gm__ ElementVNEW *)v);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmO.SetGlobalBuffer((__gm__ ElementVNEW *)o);

        cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
    }

    __aicore__ inline void Process() {
        ProcessUnifiedCore();
    }

    // ---- L1 ------------------------------------------------------------------
    // Software pipeline: every GM tile task i needs (Q, K, h, v) is issued into
    // its own double-buffered L1 bank at the top of body i, and the whole MTE2
    // stream runs under task i-1's compute (C2/C3/Vec2). All three mmads then
    // read L1 only. Banks are indexed by maskedParity like the masked slots.
    static constexpr uint32_t K_L1_SLOT   = 32 * 1024;   // 64x192 fp16 nZ = 24 KB
    static constexpr uint32_t K_L1_OFFSET = 256 * 1024;
    static constexpr uint32_t H_L1_SLOT   = 48 * 1024;   // 192x128 fp16 zN image
    static constexpr uint32_t H_L1_OFFSET = K_L1_OFFSET + 2 * K_L1_SLOT;
    static constexpr uint32_t V_L1_SLOT   = 16 * 1024;   // 64x128 fp16 zN
    static constexpr uint32_t V_L1_OFFSET = H_L1_OFFSET + 2 * H_L1_SLOT;
    // Masked attn tile handed UB->L1 (copy_ubuf_to_cbuf, MTE3) instead of through GM.
    // Two slots: Vec1[i] writes one while Cube3[i-1] still reads the other.
    static constexpr uint32_t MASKED_L1_SLOT   = 8 * 1024;            // 64x64 fp16
    static constexpr uint32_t MASKED_L1_OFFSET = V_L1_OFFSET + 2 * V_L1_SLOT;
    static_assert(MASKED_L1_OFFSET + 2 * MASKED_L1_SLOT <= ArchTag::L1_SIZE, "L1 overflow");
    // Q tile double buffer. Cube1 and Cube2 read the SAME Q tile (same qkOffset,
    // same [blockTokens, kHeadDim] shape) one body apart, so Cube1(i) lands Q in
    // slot[parity] and Cube2(i+1) takes it from L1 as its A operand instead of
    // re-reading GM: 16 KB less MTE2 per body. Two slots because Cube1(i+1) has
    // already loaded Q(i+1) before Cube2(i+1) consumes Q(i); rotation is gated on
    // vec1Ran exactly like MASKED_L1 so the drain body reads the right half.
    // The bottom 256 KB of L1 is free since ProcessSplitCore (the sole Catlass
    // BlockMmadTla user) was deleted. Cross-body MTE2-write-vs-MTE1-read on a slot
    // is ordered by each HandMmad's own MTE1_MTE2 fence.
    static constexpr uint32_t Q_L1_SLOT   = 32 * 1024;   // 64x192 fp16 max = 24 KB
    static constexpr uint32_t Q_L1_OFFSET = 0;
    static_assert(Q_L1_OFFSET + 2 * Q_L1_SLOT <= K_L1_OFFSET,
                  "chunk_fwd_o: Q slots overlap the prefetch banks");

    // ---- L0C -----------------------------------------------------------------
    // One region per cube so the mmads do not serialise on L0C reuse.
    static constexpr uint32_t UB_TILE_BYTES = 64 * 128 * sizeof(float);   // 32 KB
    static constexpr uint32_t L0C_C1_OFFSET = 0;                          // 64x64 fp32
    static constexpr uint32_t L0C_C2_OFFSET = 64 * 64 * sizeof(float);
    static constexpr uint32_t L0C_C3_OFFSET = L0C_C2_OFFSET + UB_TILE_BYTES;
    static_assert(L0C_C3_OFFSET + UB_TILE_BYTES <= ArchTag::L0C_SIZE, "L0C overflow");

    // ---- UB ------------------------------------------------------------------
    // One map, in address order. Three groups, each live in a different phase:
    //
    //   [0, VEC1_TOP)         Vec1 scratch. Vec2 reuses the bottom of it (UB_G /
    //                         UB_GBRC / UB_OUT) -- both are V-pipe and never run
    //                         concurrently.
    //   [MASK, MASK+32K)      persistent causal mask, built once per launch. Must be
    //                         above everything the cube touches.
    //   [HW, STAGE+32K)       cube-side: Cube2/Cube3 results plus the L0C->UB staging
    //                         buffer, which also hosts Vec2's Broadcast temp.
    //
    // AscendC sizes a Broadcast temp internally from the dst shape and its API takes
    // NO length, so an undersized window overruns silently -- that cost a long debug
    // once (mask corruption: output too large, error growing with chunk count, and
    // only once >2 tasks/core made Cube2/3 run). Hence the asserts below: every
    // region's end is checked against the next thing it must not reach.
    static constexpr uint32_t UB_LINE      = 512;                 // 128 * 2 * 2B
    static constexpr uint32_t UB_TILE_F32  = 32 * UB_LINE;        // 64x64 fp32 = 16 KB
    static constexpr uint32_t UB_TILE_F16  = 16 * UB_LINE;        // 64x64 fp16 =  8 KB
    static constexpr uint32_t UB_G_VEC     =  2 * UB_LINE;        // one g vector

    static constexpr uint32_t UB_BRC_A_OFFSET = 0;                       // decay, left
    static constexpr uint32_t UB_BRC_B_OFFSET = UB_BRC_A_OFFSET + UB_TILE_F32;
    static constexpr uint32_t UB_GCOMP_OFFSET = UB_BRC_B_OFFSET + UB_TILE_F32;
    static constexpr uint32_t UB_GEXP_OFFSET  = UB_GCOMP_OFFSET + UB_G_VEC;
    static constexpr uint32_t UB_GNEG_OFFSET  = UB_GEXP_OFFSET  + UB_G_VEC;
    static constexpr uint32_t UB_SHARE_OFFSET = UB_GNEG_OFFSET  + UB_G_VEC;  // Broadcast tmp
    // share -> [g f32][g f16][A f32] -> outH
    static constexpr uint32_t UB_OUTH_OFFSET  = UB_SHARE_OFFSET + UB_TILE_F32
                                              + 2 * UB_G_VEC + UB_TILE_F32;
    static constexpr uint32_t UB_VEC1_TOP     = UB_OUTH_OFFSET  + UB_TILE_F16;

    static constexpr uint32_t UB_MASK_OFFSET  = UB_VEC1_TOP;
    static constexpr uint32_t UB_MASK_SIZE    = UB_TILE_F32;

    static constexpr uint32_t UB_HW_OFFSET    = UB_MASK_OFFSET + UB_MASK_SIZE;
    static constexpr uint32_t UB_VW_OFFSET    = UB_HW_OFFSET + UB_TILE_BYTES;
    static constexpr uint32_t UB_STAGE_OFFSET = UB_VW_OFFSET + UB_TILE_BYTES;
    static_assert(UB_STAGE_OFFSET + UB_TILE_BYTES <= ArchTag::UB_SIZE, "UB overflow");
    static_assert(UB_BRC_A_OFFSET ==      0 && UB_BRC_B_OFFSET ==  16384
               && UB_GCOMP_OFFSET ==  32768 && UB_GEXP_OFFSET  ==  33792
               && UB_GNEG_OFFSET  ==  34816 && UB_SHARE_OFFSET ==  35840
               && UB_OUTH_OFFSET  ==  70656 && UB_VEC1_TOP     ==  78848
               && UB_MASK_OFFSET  ==  78848 && UB_HW_OFFSET    ==  95232
               && UB_VW_OFFSET    == 128000 && UB_STAGE_OFFSET == 160768,
                  "chunk_fwd_o: UB map moved -- these are the offsets the 24/24 "
                  "correctness sweep was measured at, re-verify before changing");

    // Vec2 scratch. NZ-native Vec2: h_work stages at UB_HW, v_work at UB_STAGE
    // (both straight from HandMmad, no deformat). gBrc aliases Vec1's brcA/brcB
    // (V pipe both, never concurrent); everything else lives in the dead UB_VW
    // window and the gap under the staging buffer, DISJOINT from Vec1's g
    // scratch and Broadcast temp -- so Vec1's MTE2 g load needs no fence
    // against Vec2's V stream or the o store's MTE3 read (auto_flag-verified,
    // fwd_o_v3.yaml).
    static constexpr uint32_t UB_GBRC_OFFSET   = 0;
    static constexpr uint32_t UB_G_OFFSET      = UB_VW_OFFSET;             // 256 B
    static constexpr uint32_t UB_BRCTMP_OFFSET = UB_VW_OFFSET + 512;
    static constexpr uint32_t UB_OUT_OFFSET    = UB_STAGE_OFFSET - 64 * 128 * sizeof(half);
    // Vec2's g vector, prefetched one body early; nothing else writes up here.
    static constexpr uint32_t UB_GPRE_OFFSET   = UB_OUT_OFFSET - 512;
    static_assert(UB_GBRC_OFFSET + UB_TILE_BYTES <= UB_GCOMP_OFFSET,
                  "chunk_fwd_o: gBrc runs into Vec1's g scratch");
    static_assert(UB_GPRE_OFFSET + 512 <= UB_OUT_OFFSET,
                  "chunk_fwd_o: gPre runs into the Vec2 out tile");

    // ---- NZ-native Vec1 -------------------------------------------------------
    // Causal mask in NZ fractal order, built once:
    //   maskNZ[(nf*4+mf)*256 + r*16 + c] = 1 iff (mf*16+r) >= (nf*16+c)
    // Whole fractals are all-ones (mf>nf) or all-zeros (mf<nf); only the diagonal
    // fractals need a per-row triangle.
    __aicore__ inline void InitCausalMaskNZ() {
        AscendC::LocalTensor<float> m =
            resource.ubBuf.template GetBufferByByte<float>(UB_MASK_OFFSET);
        constexpr uint32_t FR = 256, NF = 4, MF = 4;
        AscendC::Duplicate<float>(m, (float)0.0, NF * MF * FR);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t nf = 0; nf < NF; ++nf) {
            for (uint32_t mf = 0; mf < MF; ++mf) {
                uint32_t base = (nf * MF + mf) * FR;
                if (mf > nf) {
                    AscendC::Duplicate<float>(m[base], (float)1.0, FR);
                } else if (mf == nf) {
                    for (uint32_t r = 0; r < 16; ++r) {
                        uint32_t cnt = r + 1;
                        if (cnt >= 8) AscendC::Duplicate<float>(m[base + r * 16], (float)1.0, cnt);
                        else for (uint32_t c = 0; c < cnt; ++c) m.SetValue(base + r * 16 + c, (float)1.0);
                    }
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // Vec1, NZ-native: consumes the cube staging buffer IN PLACE -- no ND deformat.
    //   decay_nz = brcA * brcB * maskNZ
    //   brcA = exp(g[mf*16+r])  : Broadcast<2,1> of exp(g) over 16 cols, replicated per nf
    //   brcB = exp(-g[nf*16+c]) : Broadcast<2,0> per nf
    // Verified against the ND reference on host to 2e-16.
    __aicore__ inline void Vec1NZ(uint32_t gOffset, uint32_t bt) {
        auto stage = resource.ubBuf.template GetBufferByByte<float>(UB_STAGE_OFFSET);
        auto brcA  = resource.ubBuf.template GetBufferByByte<float>(UB_BRC_A_OFFSET);
        auto brcB  = resource.ubBuf.template GetBufferByByte<float>(UB_BRC_B_OFFSET);
        auto gUb   = resource.ubBuf.template GetBufferByByte<float>(UB_GCOMP_OFFSET);
        auto expg  = resource.ubBuf.template GetBufferByByte<float>(UB_GEXP_OFFSET);
        auto expmg = resource.ubBuf.template GetBufferByByte<float>(UB_GNEG_OFFSET);
        auto shareT= resource.ubBuf.template GetBufferByByte<uint8_t>(UB_SHARE_OFFSET);
        auto maskNZ= resource.ubBuf.template GetBufferByByte<float>(UB_MASK_OFFSET);
        auto outH  = resource.ubBuf.template GetBufferByByte<half>(UB_OUTH_OFFSET);

        if constexpr (std::is_same<ElementG, float>::value) {
            AscendC::DataCopy(gUb, gmG[gOffset], bt);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            AscendC::LocalTensor<ElementG> gTyped =
                resource.ubBuf.template GetBufferByByte<ElementG>(
                    UB_GCOMP_OFFSET + 256);
            AscendC::DataCopy(gTyped, gmG[gOffset], bt);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast(gUb, gTyped, AscendC::RoundMode::CAST_NONE, bt);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::Exp(expg, gUb, bt);
        AscendC::Muls(expmg, gUb, (float)-1.0, bt);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(expmg, expmg, bt);
        AscendC::PipeBarrier<PIPE_V>();

        const uint32_t NF = bt / 16, FRUN = NF * 256, N = bt * bt;
        { uint32_t d[2] = {bt, 16}, sr[2] = {bt, 1};
          AscendC::Broadcast<float, 2, 1>(brcA, expg, d, sr, shareT); }
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t nf = 1; nf < NF; ++nf) AscendC::DataCopy(brcA[nf * FRUN], brcA, FRUN);
        for (uint32_t nf = 0; nf < NF; ++nf) {
            uint32_t d[2] = {bt, 16}, sr[2] = {1, 16};
            AscendC::Broadcast<float, 2, 0>(brcB[nf * FRUN], expmg[nf * 16], d, sr, shareT);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(brcA, brcA, brcB, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(brcA, brcA, maskNZ, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(stage, stage, brcA, N);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(outH, stage, AscendC::RoundMode::CAST_NONE, N);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
    }

    // The masked tile goes UB->L1 directly (copy_ubuf_to_cbuf). No ND conversion is
    // needed and none is available -- DataCopyUB2L1ND2NZImpl is NOT_SUPPORT on m200 --
    // which is exactly why Vec1 had to become NZ-native first: the cube staging NZ order
    // and zN<half>(64,64) are the same layout, so the tile is already Cube3's A operand.
    // DataCopyUB2L1Impl is __inout_pipe__(MTE3); the handoff to Cube3's LoadData is
    // MTE3->MTE1 (kernel_event.h GetQueEvt, UB->L1).
    __aicore__ inline void MaskedNZToL1(uint32_t l1Off, uint32_t bt) {
        auto outH = resource.ubBuf.template GetBufferByByte<half>(UB_OUTH_OFFSET);
        auto l1M = resource.l1Buf.template GetBufferByByte<half>(l1Off);
        AscendC::DataCopyParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint16_t>(bt * bt * sizeof(half) / 32);
        p.srcStride = 0;
        p.dstStride = 0;
        AscendC::DataCopy(l1M, outH, p);
    }

    __aicore__ inline void ProcessUnifiedCore() {


        bool needRun = false;
        uint32_t maskedParity = 0;
        AscendC::LocalTensor<float> ubHwTensor = resource.ubBuf.template GetBufferByByte<float>(UB_HW_OFFSET);
        // v_work is HandMmad's C3 staging tile, consumed in place.
        AscendC::LocalTensor<float> ubVwTensor = resource.ubBuf.template GetBufferByByte<float>(UB_STAGE_OFFSET);

        // Persistent causal mask — built once, reused by every Vec1 invocation.
        InitCausalMaskNZ();


        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();

            // Whether Vec1 produced a masked tile this body. The drain body (isRunning
            // false, needRun true) runs Cube3 without a producer, so the L1 slot must
            // NOT rotate there or Cube3 would read the wrong half.
            const bool vec1Ran = cubeBlockScheduler.isRunning;

            if (vec1Ran) {
                // PREFETCH: issue every GM tile task i needs before task i-1's
                // compute, so the whole MTE2 stream (~112 KB/body) runs under it.
                // K/Q feed C1 at the end of THIS body; h/v/g feed C2/C3/Vec2 in
                // the NEXT body. Ordering against older readers of these banks is
                // the previous mmads' internal MTE1_MTE2 fences.
                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                const uint32_t pbt = cube1Offsets.blockTokens;
                M200Gemm::HmLoadGmToL1<ArchTag>(
                    resource, gmQ[cube1Offsets.qkOffset], kHeadDim,
                    pbt, kHeadDim, Q_L1_OFFSET + maskedParity * Q_L1_SLOT);
                M200Gemm::HmLoadGmToL1<ArchTag>(
                    resource, gmK[cube1Offsets.qkOffset], kHeadDim,
                    pbt, kHeadDim, K_L1_OFFSET + maskedParity * K_L1_SLOT);
                // C1's gate: only Q and K. h/v below carry their own flag, so
                // C1 does not stall on the 64 KB it never reads.
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);
                {   // h is already a zN image: one flat burst.
                    auto l1H = resource.l1Buf.template GetBufferByByte<half>(
                        H_L1_OFFSET + maskedParity * H_L1_SLOT);
                    AscendC::DataCopy(l1H, gmH[cube1Offsets.hOffset],
                                      M200Gemm::HmRoundUp16(kHeadDim) *
                                      M200Gemm::HmRoundUp16(vHeadDim));
                }
                M200Gemm::HmLoadGmToL1<ArchTag>(
                    resource, gmV[cube1Offsets.ovOffset], vHeadDim,
                    pbt, vHeadDim, V_L1_OFFSET + maskedParity * V_L1_SLOT);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID5);  // h/v -> next body's C2/C3
            }

            if (needRun) {
                GDNFwdOOffsets& prevOffsets = cubeBlockScheduler.GetCube23Offsets();

                // h/v banks for this task were prefetched at the previous body's
                // top; consume their flag here.
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID5);
                // CUBE2: h_work = q @ h. Both operands were prefetched at the top
                // of the previous body: Q in its slot, h's zN image in its bank.
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/true,
                                   /*A_COL_MAJOR=*/false, /*B_FROM_L1=*/true,
                                   /*LEAN_TAIL=*/true, /*NO_MTE1_MTE2=*/true>(
                    resource,
                    gmQ[prevOffsets.qkOffset], kHeadDim,
                    gmH[prevOffsets.hOffset], vHeadDim,
                    prevOffsets.blockTokens, vHeadDim, kHeadDim,
                    Q_L1_OFFSET + (maskedParity ^ 1u) * Q_L1_SLOT,
                    H_L1_OFFSET + (maskedParity ^ 1u) * H_L1_SLOT,
                    UB_HW_OFFSET, L0C_C2_OFFSET);

                // Deferred consume of the previous body's masked-tile UB->L1 store
                // (needRun implies that body ran Vec1, so the flag is outstanding).
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID3);

                // CUBE3: v_work = attn_masked @ v. A is the previous body's Vec1
                // tile, B was prefetched into its bank alongside it.
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/true,
                                   /*A_COL_MAJOR=*/false, /*B_FROM_L1=*/true,
                                   /*LEAN_TAIL=*/true, /*NO_MTE1_MTE2=*/true>(
                    resource,
                    gmV[prevOffsets.ovOffset], 0,
                    gmV[prevOffsets.ovOffset], vHeadDim,
                    prevOffsets.blockTokens, vHeadDim, prevOffsets.blockTokens,
                    MASKED_L1_OFFSET + (maskedParity ^ 1u) * MASKED_L1_SLOT,
                    V_L1_OFFSET + (maskedParity ^ 1u) * V_L1_SLOT,
                    UB_STAGE_OFFSET, L0C_C3_OFFSET);

                // VEC2 for 310P, NZ-native: o = scale * (v_work + exp(g) * h_work).
                // h_work sits in UB_HW and v_work in UB_STAGE exactly as HandMmad
                // staged them (NZ fractal order) -- both deformats are gone. exp(g)
                // depends only on the token row, so its NZ broadcast is one
                // [mAligned][16] row-broadcast block replicated per Z-column (the
                // Vec1NZ brcA pattern). o leaves UB as one strided zN->ND
                // descriptor per Z-column.
                {
                    uint32_t bt = prevOffsets.blockTokens;
                    uint32_t mAl = (bt + 15) / 16 * 16;
                    uint32_t nFr = (vHeadDim + 15) / 16;
                    uint32_t FRUN = mAl * 16;
                    uint32_t N = nFr * FRUN;
                    AscendC::LocalTensor<float> gUb =
                        resource.ubBuf.template GetBufferByByte<float>(UB_G_OFFSET);
                    AscendC::LocalTensor<float> gBrc =
                        resource.ubBuf.template GetBufferByByte<float>(UB_GBRC_OFFSET);
                    AscendC::LocalTensor<uint8_t> brcTmp =
                        resource.ubBuf.template GetBufferByByte<uint8_t>(UB_BRCTMP_OFFSET);
                    AscendC::LocalTensor<ElementVNEW> outUb =
                        resource.ubBuf.template GetBufferByByte<ElementVNEW>(UB_OUT_OFFSET);

                    // g was prefetched at the top of the previous body; the wait
                    // pairs with the MTE2_V set right after that issue.
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    if constexpr (std::is_same<ElementG, float>::value) {
                        AscendC::LocalTensor<float> gPre =
                            resource.ubBuf.template GetBufferByByte<float>(UB_GPRE_OFFSET);
                        AscendC::Exp(gUb, gPre, bt);
                    } else {
                        AscendC::LocalTensor<ElementG> gPre =
                            resource.ubBuf.template GetBufferByByte<ElementG>(UB_GPRE_OFFSET);
                        AscendC::Cast(gUb, gPre, AscendC::RoundMode::CAST_NONE, bt);
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Exp(gUb, gUb, bt);
                    }
                    // gPre is consumed; release it for this body's re-prefetch
                    // (auto_flag: WAR v2_exp -> gpre@MTE2). Narrow: only the g
                    // load waits, the big prefetch batch of the next body is
                    // already past its own fence by then.
                    if (vec1Ran) { AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4); }
                    AscendC::PipeBarrier<PIPE_V>();
                    {
                        uint32_t dstShape[2] = {bt, 16};
                        uint32_t srcShape[2] = {bt, 1};
                        AscendC::Broadcast<float, 2, 1>(gBrc, gUb, dstShape, srcShape, brcTmp);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    for (uint32_t nf = 1; nf < nFr; ++nf) {
                        AscendC::DataCopy(gBrc[nf * FRUN], gBrc, FRUN);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(ubHwTensor, ubHwTensor, gBrc, N);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(ubVwTensor, ubVwTensor, ubHwTensor, N);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(ubVwTensor, ubVwTensor, (float)scale, N);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(outUb, ubVwTensor, AscendC::RoundMode::CAST_NONE, N);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    // zN -> ND: per Z-column, rows are contiguous 16-element blocks;
                    // ND rows stride vHeadDim.
                    AscendC::DataCopyParams op;
                    op.blockCount = static_cast<uint16_t>(bt);
                    op.blockLen = static_cast<uint16_t>(16 * sizeof(ElementVNEW) / 32);
                    op.srcStride = 0;
                    op.dstStride = static_cast<uint16_t>((vHeadDim - 16) * sizeof(ElementVNEW) / 32);
                    for (uint32_t nf = 0; nf < nFr; ++nf) {
                        AscendC::DataCopy(gmO[prevOffsets.ovOffset + nf * 16], outUb[nf * FRUN], op);
                    }
                    // No MTE3_V drain after the store: v2out sits in its own
                    // window now and its only later writer is the next body's
                    // Vec2 cast, ordered through the kept V_MTE3 pair
                    // (reduce_flags: v2fin dropped, checker-verified).
                    // The old end-of-body V_MTE3 fence is gone with the deformats:
                    // the next body's only MTE3 readers (MaskedNZToL1, HandMmad
                    // internals) touch windows Vec2 never writes, and every V-write
                    // window Vec2 leaves behind is next written by the V pipe, which
                    // runs in order.
                }
            }

            if (vec1Ran) {
                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                // All four prefetched tiles are in by now -- the wait pairs with
                // the set at the top of this body.
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID6);

                // CUBE1: attn = q @ k^T.  B is ColumnMajor because k is stored
                // [seqlen][kHeadDim] row-major, so B[kk][j] == gm[j*kHeadDim + kk];
                // the prefetch's Nd2Nz already landed it in the nZ image this
                // path expects.
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/true, /*A_FROM_L1=*/true,
                                   /*A_COL_MAJOR=*/false, /*B_FROM_L1=*/true,
                                   /*LEAN_TAIL=*/true>(
                    resource,
                    gmQ[cube1Offsets.qkOffset], kHeadDim,
                    gmK[cube1Offsets.qkOffset], kHeadDim,
                    cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim,
                    Q_L1_OFFSET + maskedParity * Q_L1_SLOT,
                    K_L1_OFFSET + maskedParity * K_L1_SLOT,
                    UB_STAGE_OFFSET, L0C_C1_OFFSET);

                // VEC1 is NZ-native: it consumes the cube's staging buffer in place,
                // so there is no deformat here, and leaves the masked tile in L1 as
                // Cube3's A operand.
                Vec1NZ(cube1Offsets.gOffset, cube1Offsets.blockTokens);
                // Vec2(i)'s g, prefetched HERE and not in the top batch: Vec1NZ's
                // closing V_MTE3 drain just retired the previous Vec2's read of
                // gPre, so the single buffer is safe to overwrite (an MTE2 write
                // is not otherwise ordered against V reads).
                if (needRun) { AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4); }
                {
                    auto gPre = resource.ubBuf.template GetBufferByByte<ElementG>(UB_GPRE_OFFSET);
                    AscendC::DataCopy(gPre, gmG[cube1Offsets.gOffset], cube1Offsets.blockTokens);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);     // g -> next body's Vec2
                MaskedNZToL1(MASKED_L1_OFFSET + maskedParity * MASKED_L1_SLOT,
                             cube1Offsets.blockTokens);
                // Set here, wait at the NEXT body's Cube3: the next body's Cube2
                // runs underneath the outstanding MTE3.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID3);
            }
            if (vec1Ran) { maskedParity ^= 1u; }
            needRun = true;
        }

    }

    
};

}
