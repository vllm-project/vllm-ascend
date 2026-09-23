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
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "../block/block_scheduler_gdn_fwd_o.hpp"
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
    typename WORKSPACE_TYPE
>
class GDNFwdOKernel {
public:
    
    using ArchTag = Arch::AtlasA2;
    using GDNFwdOOffsets = Catlass::Gemm::Block::GDNFwdOOffsets;

    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdOVec;

    using DispatchPolicyTla = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using L1TileShapeTla = Shape<_128, _128, _128>;
    using L0TileShapeTla = L1TileShapeTla;
    using QType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using AttenType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using AttenMaskedType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using OinterType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VNEWType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;

    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using OType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using MaskType = Gemm::GemmType<bool, layout::RowMajor>;

    // cube 1
    using TileCopyQK = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::ColumnMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQK>;

    // cube 2
    using TileCopyQH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadQH = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyQH>;

    // cube 3
    using TileCopyAttenVNEW = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadAttenVNEW = Gemm::Block::BlockMmadTla<DispatchPolicyTla, L1TileShapeTla, L0TileShapeTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyAttenVNEW>;

    // vec 1

    // vec 2

    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = Catlass::layout::RowMajor;

    using ElementK =  typename BlockMmadQK::ElementB;
    using LayoutK = Catlass::layout::ColumnMajor;

    using ElementAtten = typename BlockMmadQK::ElementC;
    using LayoutAtten = Catlass::layout::RowMajor;
    
    using ElementAttenMasked = typename BlockMmadQH::ElementA;
    using LayoutAttenMasked = Catlass::layout::RowMajor;

    using ElementH = typename BlockMmadQH::ElementB;
    using LayoutH = Catlass::layout::RowMajor;

    using ElementOinter = typename BlockMmadQH::ElementC;
    using LayoutOinter = Catlass::layout::RowMajor;


    using ElementVNEW = typename BlockMmadAttenVNEW::ElementB; 
    using LayoutVNEW = Catlass::layout::RowMajor;


    using ElementG = G_TYPE;
    using ElementMask = bool;

    using L1TileShape = typename BlockMmadQK::L1TileShape;

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
    VecScheduler vecBlockScheduler;

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

    // Build the 64x64 lower-triangular causal mask ONCE per kernel launch.
    // It lives at UB_MASK_OFFSET, above every
    // ping-pong buffer, so the cube's L0C->UB staging at UB[0] cannot clobber it
    // and this does NOT have to be redone after each matmul.
    __aicore__ inline void InitCausalMask() {
        AscendC::LocalTensor<float> maskUbTensor =
            resource.ubBuf.template GetBufferByByte<float>(UB_MASK_OFFSET);
        // 310P: Duplicate count must be >= 8 (vector width = 8 floats).
        // Build lower-triangular mask: row i has 1.0 in cols [0..i], 0.0 elsewhere.
        // Fill all 1.0 first, then zero the upper triangle with count >= 8.
        AscendC::Duplicate<float>(maskUbTensor, (float)1.0, 64 * 64);
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t zeroStart = i + 1;
            uint32_t zeroLen = 64 - zeroStart;
            if (zeroLen >= 8) {
                AscendC::Duplicate<float>(maskUbTensor[i * 64 + zeroStart], (float)0.0, zeroLen);
            } else {
                for (uint32_t j = 0; j < zeroLen; ++j) {
                    maskUbTensor.SetValue(i * 64 + zeroStart + j, (float)0.0);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    // ---- L1 ------------------------------------------------------------------
    // The Catlass BlockMmadTla tiles used to own the bottom 256 KB; the hand mmad
    // keeps clear of them.
    static constexpr uint32_t HAND_L1A_OFFSET = 256 * 1024;
    static constexpr uint32_t HAND_L1B_OFFSET = HAND_L1A_OFFSET + 32 * 1024;
    // Masked attn tile handed UB->L1 (copy_ubuf_to_cbuf, MTE3) instead of through GM.
    // Two slots: Vec1[i] writes one while Cube3[i-1] still reads the other.
    static constexpr uint32_t MASKED_L1_SLOT   = 8 * 1024;            // 64x64 fp16
    static constexpr uint32_t MASKED_L1_OFFSET = HAND_L1B_OFFSET + 64 * 1024;
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
    static_assert(Q_L1_OFFSET + 2 * Q_L1_SLOT <= HAND_L1A_OFFSET,
                  "chunk_fwd_o: Q slots overlap the hand-mmad L1 region");

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

    // Vec2 scratch, aliasing Vec1's region; its Broadcast temp goes in the staging
    // window instead (live seq 16-22 vs 24, so lifetime-disjoint, and a full 32 KB).
    static constexpr uint32_t UB_G_OFFSET      = 0;
    static constexpr uint32_t UB_GBRC_OFFSET   = 512;
    static constexpr uint32_t UB_OUT_OFFSET    = UB_GBRC_OFFSET + UB_TILE_BYTES;
    static constexpr uint32_t UB_BRCTMP_OFFSET = UB_STAGE_OFFSET;
    static_assert(UB_OUT_OFFSET + 64 * 128 * sizeof(half) <= UB_VEC1_TOP,
                  "chunk_fwd_o: Vec2 scratch does not fit in Vec1's region");
    static_assert(UB_OUT_OFFSET + 64 * 128 * sizeof(half) <= UB_MASK_OFFSET,
                  "chunk_fwd_o: Vec2 scratch collides with the persistent causal mask");

    // block_mmad leaves the matmul result in UB at offset 0 in NZ fractal order
    // ([N/16 Z-col][M/16 frac][16 rows][16 cols]) before its own fractal loop pushes
    // it out to GM. On the unified core the very next consumer is a vector epilogue
    // on THIS core, so the GM round-trip is pure overhead: deformat NZ->ND straight
    // into a UB home instead. One burst per Z-column -- for a fixed nf the fractals
    // mf = 0..mFracs-1 are contiguous in UB and their ND rows mf*16 + r increase with
    // the UB linear index, so the whole column block is a single strided descriptor.
    __aicore__ inline void DeformatL0CStagingToUb(AscendC::LocalTensor<float> dst,
                                                  uint32_t mActual, uint32_t nActual,
                                                  uint32_t stageOff = 0) {
        AscendC::LocalTensor<float> co2Temp = resource.ubBuf.template GetBufferByByte<float>(stageOff);
        uint32_t mAligned = (mActual + 15) / 16 * 16;
        uint32_t nAligned = (nActual + 15) / 16 * 16;
        uint32_t mFracs = mAligned / 16;
        uint32_t nFracs = nAligned / 16;
        AscendC::DataCopyParams p;
        p.blockCount = static_cast<uint16_t>(mAligned);
        p.blockLen = static_cast<uint16_t>(16 * sizeof(float) / 32);
        p.srcStride = 0;
        p.dstStride = static_cast<uint16_t>((nAligned - 16) * sizeof(float) / 32);
        for (uint32_t nf = 0; nf < nFracs; ++nf) {
            AscendC::DataCopy(dst[nf * 16], co2Temp[nf * mFracs * 256], p);
        }
        // UB->UB move feeding a V-pipe consumer. Nothing here touches GM, so the
        // full PIPE_ALL that the old GM path needed is not required.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    }

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
        AscendC::LocalTensor<float> ubVwTensor = resource.ubBuf.template GetBufferByByte<float>(UB_VW_OFFSET);

        // Persistent causal mask — built once, reused by every Vec1 invocation.
        InitCausalMaskNZ();


        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();

            // Whether Vec1 produced a masked tile this body. The drain body (isRunning
            // false, needRun true) runs Cube3 without a producer, so the L1 slot must
            // NOT rotate there or Cube3 would read the wrong half.
            const bool vec1Ran = cubeBlockScheduler.isRunning;

            if (vec1Ran) {
                // CUBE1: attn = q @ k^T.  B is ColumnMajor because k is stored
                // [seqlen][kHeadDim] row-major, so B[kk][j] == gm[j*kHeadDim + kk].
                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/true>(
                    resource,
                    gmQ[cube1Offsets.qkOffset], kHeadDim,
                    gmK[cube1Offsets.qkOffset], kHeadDim,
                    cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim,
                    Q_L1_OFFSET + maskedParity * Q_L1_SLOT,
                    HAND_L1B_OFFSET, UB_STAGE_OFFSET, L0C_C1_OFFSET);

                // VEC1 is NZ-native: it consumes the cube's staging buffer in place,
                // so there is no deformat here, and leaves the masked tile in L1 as
                // Cube3's A operand.
                Vec1NZ(cube1Offsets.gOffset, cube1Offsets.blockTokens);
                MaskedNZToL1(MASKED_L1_OFFSET + maskedParity * MASKED_L1_SLOT,
                             cube1Offsets.blockTokens);
                // Set here, wait down at Cube3, so all of Cube2 runs underneath the
                // outstanding MTE3. That deferral is the point of the two L1 slots:
                // waiting here would drain MTE3 and the double buffer would be free.
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID3);
            }

            if (needRun) {
                GDNFwdOOffsets& prevOffsets = cubeBlockScheduler.GetCube23Offsets();

                // CUBE2: h_work = q @ h.  A (the Q tile) is already in L1 -- the
                // previous body's Cube1 loaded the identical tile -- so gmA/lda are
                // unused and no GM re-read happens.
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/true>(
                    resource,
                    gmQ[prevOffsets.qkOffset], kHeadDim,
                    gmH[prevOffsets.hOffset], vHeadDim,
                    prevOffsets.blockTokens, vHeadDim, kHeadDim,
                    Q_L1_OFFSET + (maskedParity ^ 1u) * Q_L1_SLOT,
                    HAND_L1B_OFFSET, UB_STAGE_OFFSET, L0C_C2_OFFSET);
                // h_work out of the staging buffer before Cube3 overwrites it.
                DeformatL0CStagingToUb(ubHwTensor, prevOffsets.blockTokens, vHeadDim, UB_STAGE_OFFSET);

                // Deferred consume of Vec1's UB->L1 store (set above, before Cube2).
                if (vec1Ran) { AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID3); }

                // CUBE3: v_work = attn_masked @ v.  A is already in L1 -- the previous
                // body's Vec1 put it there -- so gmA/lda are unused.
                M200Gemm::HandMmad<ArchTag, /*B_COL_MAJOR=*/false, /*A_FROM_L1=*/true>(
                    resource,
                    gmV[prevOffsets.ovOffset], 0,
                    gmV[prevOffsets.ovOffset], vHeadDim,
                    prevOffsets.blockTokens, vHeadDim, prevOffsets.blockTokens,
                    MASKED_L1_OFFSET + (maskedParity ^ 1u) * MASKED_L1_SLOT,
                    HAND_L1B_OFFSET, UB_STAGE_OFFSET, L0C_C3_OFFSET);
                DeformatL0CStagingToUb(ubVwTensor, prevOffsets.blockTokens, vHeadDim, UB_STAGE_OFFSET);

                // VEC2 for 310P: o = scale * (v_work + exp(g) * h_work).
                // h_work and v_work are already in UB (ND) from DeformatL0CStagingToUb,
                // so this reads nothing from GM and runs the whole 64-row tile in one
                // pass instead of two 32-row stages -- half the vector-op count, and
                // 64 KB/chunk-head of MTE2 reads removed outright.
                {
                    uint32_t bt = prevOffsets.blockTokens;
                    uint32_t elems = bt * vHeadDim;
                    // Scratch lives low in UB: Vec1's buffers down there
                    // are dead until the next iteration rewrites them, and the cube's
                    // UB[0] staging is finished for this iteration.
                    AscendC::LocalTensor<float> gUb =
                        resource.ubBuf.template GetBufferByByte<float>(UB_G_OFFSET);
                    AscendC::LocalTensor<float> gBrc =
                        resource.ubBuf.template GetBufferByByte<float>(UB_GBRC_OFFSET);
                    AscendC::LocalTensor<uint8_t> brcTmp =
                        resource.ubBuf.template GetBufferByByte<uint8_t>(UB_BRCTMP_OFFSET);
                    AscendC::LocalTensor<ElementVNEW> outUb =
                        resource.ubBuf.template GetBufferByByte<ElementVNEW>(UB_OUT_OFFSET);

                    if constexpr (std::is_same<ElementG, float>::value) {
                        AscendC::DataCopy(gUb, gmG[prevOffsets.gOffset], bt);
                    } else {
                        AscendC::LocalTensor<ElementG> gTyped =
                            resource.ubBuf.template GetBufferByByte<ElementG>(UB_G_OFFSET + 256);
                        AscendC::DataCopy(gTyped, gmG[prevOffsets.gOffset], bt);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                        AscendC::Cast(gUb, gTyped, AscendC::RoundMode::CAST_NONE, bt);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    AscendC::Exp(gUb, gUb, bt);
                    AscendC::PipeBarrier<PIPE_V>();
                    {
                        uint32_t dstShape[2] = {bt, vHeadDim};
                        uint32_t srcShape[2] = {bt, 1};
                        AscendC::Broadcast<float, 2, 1>(gBrc, gUb, dstShape, srcShape, brcTmp);
                    }
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Mul(ubHwTensor, ubHwTensor, gBrc, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add(ubVwTensor, ubVwTensor, ubHwTensor, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Muls(ubVwTensor, ubVwTensor, (float)scale, elems);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Cast(outUb, ubVwTensor, AscendC::RoundMode::CAST_NONE, elems);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::DataCopyParams cp{1, static_cast<uint16_t>(elems * sizeof(ElementVNEW) / 32), 0, 0};
                    AscendC::DataCopy(gmO[prevOffsets.ovOffset], outUb, cp);
                    // Next iteration's Cube1 reuses UB[0] via MTE2/MTE1 and its Vec1
                    // rewrites this scratch; only the outstanding MTE3 read of outUb
                    // has to complete first.
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
                    // End-of-body cross-iteration fence. Vec2's V writes (v_work,
                    // vec2_gbrc, vec2_out) land in UB windows that the NEXT body's
                    // MTE3 deformats and UB->UB moves overwrite (Mode B dep_graph:
                    // v2_fma@i0 -> v1_attn_in@i1, v2_brc@i0 -> c1_deformat@i1).
                    // The block_mmad PipeBarrier<PIPE_ALL> used to cover this by
                    // accident; now that it is a V_M edge, say it explicitly.
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
                }
            }

            if (vec1Ran && !needRun) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE1>(EVENT_ID3);
            }
            if (vec1Ran) { maskedParity ^= 1u; }
            needRun = true;
        }

    }

    
};

}
