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
#include "../../epilogue/block/block_epilogue_gdn_fwdo_qkmask.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdo_output.hpp"
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
    using DispatchPolicyGDNFwdOQkmask = Epilogue::EpilogueAtlasGDNFwdOQkmask;
    using EpilogueGDNFwdOQkmask = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOQkmask, AttenMaskedType, GType, AttenType, MaskType>;

    // vec 2
    using DispatchPolicyGDNFwdOOutput = Epilogue::EpilogueAtlasGDNFwdOOutput;
    using EpilogueGDNFwdOOutput = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdOOutput, OType, GType, OinterType, OinterType>;

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
    uint32_t vWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t attnWorkspaceOffset;
    uint32_t aftermaskWorkspaceOffset;
    uint32_t maskWorkspaceOffset;
    
    AscendC::GlobalTensor<ElementQ> gmQ;
    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementVNEW> gmV;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementVNEW> gmO;
    AscendC::GlobalTensor<ElementOinter> gmVWorkspace;
    AscendC::GlobalTensor<ElementOinter> gmHWorkspace;
    AscendC::GlobalTensor<ElementAtten> gmAttnWorkspace;
    AscendC::GlobalTensor<ElementAttenMasked> gmAftermaskWorkspace;
    AscendC::GlobalTensor<ElementMask> gmMask;

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
        vWorkspaceOffset = gdnFwdOTilingData->vWorkspaceOffset;
        hWorkspaceOffset = gdnFwdOTilingData->hWorkspaceOffset;
        attnWorkspaceOffset = gdnFwdOTilingData->attnWorkspaceOffset;
        aftermaskWorkspaceOffset = gdnFwdOTilingData->aftermaskWorkspaceOffset;
        maskWorkspaceOffset = gdnFwdOTilingData->maskWorkspaceOffset;

        gmQ.SetGlobalBuffer((__gm__ ElementQ *)q);
        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmV.SetGlobalBuffer((__gm__ ElementVNEW *)v);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmO.SetGlobalBuffer((__gm__ ElementVNEW *)o);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + vWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementOinter *)(user + hWorkspaceOffset));
        gmAttnWorkspace.SetGlobalBuffer((__gm__ ElementAtten *)(user + attnWorkspaceOffset));
        gmAftermaskWorkspace.SetGlobalBuffer((__gm__ ElementAttenMasked *)(user + aftermaskWorkspaceOffset));
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)(user + maskWorkspaceOffset));

        cubeBlockScheduler.Init(cu_seqlens, chunk_offsets, tiling);
    }

    __aicore__ inline void Process() {
        ProcessUnifiedCore();
    }

    __aicore__ inline void InitCausalMask() {
        AscendC::LocalTensor<float> maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(0);
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

    __aicore__ inline void ProcessUnifiedCore() {
        uint32_t coreNum = AscendC::GetBlockNum();

        BlockMmadQK blockMmadQK(resource);
        BlockMmadQH blockMmadQH(resource);
        BlockMmadAttenVNEW blockMmadAttenVNEW(resource);

        auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
        auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
        auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
        auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
        auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

        bool needRun = false;
        uint32_t pingpongFlag = 0;

        while (cubeBlockScheduler.isRunning) {
            cubeBlockScheduler.InitTask();

            if (cubeBlockScheduler.isRunning) {
                // CUBE1: attn = q @ k.T
                GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                auto tensorQ = tla::MakeTensor(gmQ[cube1Offsets.qkOffset], qLayout, Catlass::Arch::PositionGM{});
                auto tensorK = tla::MakeTensor(gmK[cube1Offsets.qkOffset], kLayout, Catlass::Arch::PositionGM{});
                auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1Offsets.attnWorkOffset], attenLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                blockMmadQK.preSetFlags();
                blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                blockMmadQK.finalWaitFlags();

                // Re-init causal mask after cube (cube overwrites UB[0])
                InitCausalMask();

                // VEC1: qkmask epilogue
                EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                epilogueGDNFwdOQkmask(
                    gmAftermaskWorkspace[cube1Offsets.attnWorkOffset],
                    gmG[cube1Offsets.gOffset], gmAttnWorkspace[cube1Offsets.attnWorkOffset], gmMask,
                    chunkSize, cube1Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag,
                    cube1Offsets.batchIdx, cube1Offsets.headIdx, cube1Offsets.chunkIdx
                );
            }

            // GM fence: ensure Vec1 MTE3 writes are committed before Cube3 MTE2 reads
            AscendC::PipeBarrier<PIPE_ALL>();

            if (needRun) {
                GDNFwdOOffsets& prevOffsets = cubeBlockScheduler.GetCube23Offsets();

                // CUBE2: h_work = q @ h
                auto tensorQ2 = tla::MakeTensor(gmQ[prevOffsets.qkOffset], qLayout, Catlass::Arch::PositionGM{});
                auto tensorH = tla::MakeTensor(gmH[prevOffsets.hOffset], hLayout, Catlass::Arch::PositionGM{});
                auto tensorHWork = tla::MakeTensor(gmHWorkspace[prevOffsets.hvWorkOffset], ointerLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube2Shape{prevOffsets.blockTokens, vHeadDim, kHeadDim};
                auto tensorBlockQ2 = GetTile(tensorQ2, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                blockMmadQH.preSetFlags();
                blockMmadQH(tensorBlockQ2, tensorBlockH, tensorBlockHWork, cube2Shape);
                blockMmadQH.finalWaitFlags();

                // CUBE3: v_work = attn_masked @ v
                auto attenLayout3 = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, prevOffsets.blockTokens);
                auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[prevOffsets.attnWorkOffset], attenLayout3, Catlass::Arch::PositionGM{});
                auto tensorV = tla::MakeTensor(gmV[prevOffsets.ovOffset], vnewLayout, Catlass::Arch::PositionGM{});
                auto tensorVWork = tla::MakeTensor(gmVWorkspace[prevOffsets.hvWorkOffset], ointerLayout, Catlass::Arch::PositionGM{});
                GemmCoord cube3Shape{prevOffsets.blockTokens, vHeadDim, prevOffsets.blockTokens};
                auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                blockMmadAttenVNEW.preSetFlags();
                blockMmadAttenVNEW(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                blockMmadAttenVNEW.finalWaitFlags();

                // GM fence: ensure Cube2/3 L0C→UB→MTE3→GM writes are committed
                AscendC::PipeBarrier<PIPE_ALL>();

                // VEC2 inline for 310P: o = scale * (v_work + exp(g) * h_work)
                // The epilogue class uses event-based MTE2 sync that breaks after cube matmul on 310P.
                {
                    constexpr uint32_t STAGE_ROWS = 32;
                    uint32_t bt = prevOffsets.blockTokens;
                    uint32_t stageCnt = STAGE_ROWS * vHeadDim;
                    // UB layout: vwUb[0..stageCnt), hwUb[stageCnt..2*stageCnt), gUb[2*stageCnt..+64)
                    AscendC::LocalTensor<float> vwUb = resource.ubBuf.template GetBufferByByte<float>(0);
                    AscendC::LocalTensor<float> hwUb = resource.ubBuf.template GetBufferByByte<float>(stageCnt * sizeof(float));
                    AscendC::LocalTensor<float> gUb  = resource.ubBuf.template GetBufferByByte<float>(stageCnt * sizeof(float) * 2);
                    // outUb (half) after gUb, aligned to 512B
                    constexpr uint32_t G_RESERVE = 512;
                    AscendC::LocalTensor<ElementVNEW> outUb = resource.ubBuf.template GetBufferByByte<ElementVNEW>(
                        stageCnt * sizeof(float) * 2 + G_RESERVE);

                    for (uint32_t row = 0; row < bt; row += STAGE_ROWS) {
                        uint32_t rows = (row + STAGE_ROWS <= bt) ? STAGE_ROWS : (bt - row);
                        uint32_t elems = rows * vHeadDim;
                        uint32_t gmOff = row * vHeadDim;

                        // Load v_work, h_work, g from GM
                        AscendC::DataCopy(vwUb, gmVWorkspace[prevOffsets.hvWorkOffset + gmOff], elems);
                        AscendC::DataCopy(hwUb, gmHWorkspace[prevOffsets.hvWorkOffset + gmOff], elems);
                        // Load g (may be float or half)
                        if constexpr (std::is_same<ElementG, float>::value) {
                            AscendC::DataCopy(gUb, gmG[prevOffsets.gOffset + row], rows);
                        } else {
                            AscendC::LocalTensor<ElementG> gTyped = resource.ubBuf.template GetBufferByByte<ElementG>(
                                stageCnt * sizeof(float) * 2 + 256);
                            AscendC::DataCopy(gTyped, gmG[prevOffsets.gOffset + row], rows);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::Cast(gUb, gTyped, AscendC::RoundMode::CAST_NONE, rows);
                        }
                        AscendC::PipeBarrier<PIPE_ALL>();

                        // exp(g)
                        AscendC::Exp(gUb, gUb, rows);
                        AscendC::PipeBarrier<PIPE_V>();

                        // Broadcast exp(g) into gBrc: each row r gets exp(g[r]) repeated Dv times
                        // gBrc lives after outUb in UB
                        AscendC::LocalTensor<float> gBrc = resource.ubBuf.template GetBufferByByte<float>(
                            stageCnt * sizeof(float) * 2 + G_RESERVE + stageCnt * sizeof(ElementVNEW));
                        {
                            uint32_t dstShape[2] = {rows, vHeadDim};
                            uint32_t srcShape[2] = {rows, 1};
                            // Broadcast needs a shared temp buffer — use space after gBrc
                            AscendC::LocalTensor<uint8_t> brcTmp = resource.ubBuf.template GetBufferByByte<uint8_t>(
                                stageCnt * sizeof(float) * 2 + G_RESERVE + stageCnt * sizeof(ElementVNEW) + elems * sizeof(float));
                            AscendC::Broadcast<float, 2, 1>(gBrc, gUb, dstShape, srcShape, brcTmp);
                        }
                        AscendC::PipeBarrier<PIPE_V>();
                        AscendC::Mul(hwUb, hwUb, gBrc, elems);
                        AscendC::PipeBarrier<PIPE_V>();

                        // v_work + exp(g)*h_work
                        AscendC::Add(vwUb, vwUb, hwUb, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        // * scale
                        AscendC::Muls(vwUb, vwUb, (float)scale, elems);
                        AscendC::PipeBarrier<PIPE_V>();
                        // Cast to output dtype
                        AscendC::Cast(outUb, vwUb, AscendC::RoundMode::CAST_NONE, elems);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                        AscendC::DataCopyParams cp{1, static_cast<uint16_t>(elems * sizeof(ElementVNEW) / 32), 0, 0};
                        AscendC::DataCopy(gmO[prevOffsets.ovOffset + gmOff], outUb, cp);
                        AscendC::PipeBarrier<PIPE_ALL>();
                    }
                }
            }

            needRun = true;
        }
    }

    __aicore__ inline void ProcessSplitCore() {
        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadQK blockMmadQK(resource);
            BlockMmadQH blockMmadQH(resource);
            BlockMmadAttenVNEW blockMmadAttenVNEW(resource);

            auto qLayout = tla::MakeLayout<ElementQ, LayoutQ>(shapeBatch * kNumHead * seqlen, kHeadDim);
            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * seqlen);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * seqlen * kHeadDim, vHeadDim);
            auto ointerLayout = tla::MakeLayout<ElementOinter, LayoutOinter>(coreNum * chunkSize * PING_PONG_STAGES, vHeadDim);
            auto vnewLayout = tla::MakeLayout<ElementVNEW, LayoutVNEW>(shapeBatch * vNumHead * seqlen, vHeadDim);

            bool needRun = false;
            bool isFirstC3 = true;

            while (cubeBlockScheduler.isRunning) {
                cubeBlockScheduler.InitTask();

                if (cubeBlockScheduler.isRunning && coreIdx < coreNum) {

                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);

                    GDNFwdOOffsets& cube1Offsets = cubeBlockScheduler.GetCube1Offsets();
                    int64_t cube1OffsetQ = cube1Offsets.qkOffset; 
                    int64_t cube1OffsetK = cube1Offsets.qkOffset; 
                    int64_t cube1OffsetAttn = cube1Offsets.attnWorkOffset; 
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube1Offsets.blockTokens);
                    auto tensorQ = tla::MakeTensor(gmQ[cube1OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorK = tla::MakeTensor(gmK[cube1OffsetK], kLayout, Catlass::Arch::PositionGM{});
                    auto tensorAttn = tla::MakeTensor(gmAttnWorkspace[cube1OffsetAttn], attenLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube1Shape{cube1Offsets.blockTokens, cube1Offsets.blockTokens, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                    auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                    auto tensorBlockAttn = GetTile(tensorAttn, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));
                    blockMmadQK.preSetFlags();
                    blockMmadQK(tensorBlockQ, tensorBlockK, tensorBlockAttn, cube1Shape);
                    blockMmadQK.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done);

                }
                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum) {
                    if(!cubeBlockScheduler.isRunning) Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                    Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);
                    GDNFwdOOffsets& cube2Offsets = cubeBlockScheduler.GetCube23Offsets();
                    int64_t cube2OffsetQ = cube2Offsets.qkOffset;
                    int64_t cube2OffsetH = cube2Offsets.hOffset;
                    int64_t cube2OffsetHWork = cube2Offsets.hvWorkOffset; 
                    auto tensorQ = tla::MakeTensor(gmQ[cube2OffsetQ], qLayout, Catlass::Arch::PositionGM{});
                    auto tensorH = tla::MakeTensor(gmH[cube2OffsetH], hLayout, Catlass::Arch::PositionGM{});
                    auto tensorHWork = tla::MakeTensor(gmHWorkspace[cube2OffsetHWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube2Shape{cube2Offsets.blockTokens, vHeadDim, kHeadDim};
                    auto tensorBlockQ = GetTile(tensorQ, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                    auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                    auto tensorBlockHWork = GetTile(tensorHWork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));
                    blockMmadQH.preSetFlags();
                    blockMmadQH(tensorBlockQ, tensorBlockH, tensorBlockHWork, cube2Shape);
                    blockMmadQH.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done);
                }

                if (needRun && coreIdx < coreNum) {
                    GDNFwdOOffsets& cube3Offsets = cubeBlockScheduler.GetCube23Offsets();
                    if(isFirstC3) Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done);
                    int64_t cube3OffsetAttnMask = cube3Offsets.attnWorkOffset; 
                    int64_t cube3OffsetV = cube3Offsets.ovOffset; 
                    int64_t cube3OffsetVWork = cube3Offsets.hvWorkOffset; 
                    auto attenLayout = tla::MakeLayout<ElementAtten, LayoutAtten>(coreNum * chunkSize * PING_PONG_STAGES, cube3Offsets.blockTokens);
                    auto tensorAttnMask = tla::MakeTensor(gmAftermaskWorkspace[cube3OffsetAttnMask], attenLayout, Catlass::Arch::PositionGM{});
                    auto tensorV = tla::MakeTensor(gmV[cube3OffsetV], vnewLayout, Catlass::Arch::PositionGM{});
                    auto tensorVWork = tla::MakeTensor(gmVWorkspace[cube3OffsetVWork], ointerLayout, Catlass::Arch::PositionGM{});
                    GemmCoord cube3Shape{cube3Offsets.blockTokens, vHeadDim, cube3Offsets.blockTokens};
                    auto tensorBlockAttnMask = GetTile(tensorAttnMask, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.k()));
                    auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.k(), cube3Shape.n()));
                    auto tensorBlockVWork = GetTile(tensorVWork, tla::MakeCoord(0, 0), tla::MakeShape(cube3Shape.m(), cube3Shape.n()));
                    blockMmadAttenVNEW.preSetFlags();
                    blockMmadAttenVNEW(tensorBlockAttnMask, tensorBlockV, tensorBlockVWork, cube3Shape);
                    blockMmadAttenVNEW.finalWaitFlags();
                    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube3Done);
                    isFirstC3 = false;
                }
                needRun = true;
                // AscendC::PipeBarrier<PIPE_ALL>();
            }
            if (coreIdx < coreNum) {
                Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done);

            }
        }

        if ASCEND_IS_AIV {

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
            uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
            uint32_t subBlockNum = AscendC::GetSubBlockNum();

            AscendC::LocalTensor<float> maskUbTensor = resource.ubBuf.template GetBufferByByte<float>(0);
            AscendC::Duplicate<float>(maskUbTensor, (float)0.0, 64*64);
            AscendC::PipeBarrier<PIPE_V>();
            for(uint32_t i = 0; i < 64; ++ i) AscendC::Duplicate<float>(maskUbTensor[i * 64], (float)1.0, i + 1);
            AscendC::PipeBarrier<PIPE_V>();

            bool needRun = false;
            uint32_t pingpongFlag = 0;

            if (coreIdx < coreNum * subBlockNum) {
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
            }

            while (vecBlockScheduler.isRunning) {
                vecBlockScheduler.InitTask();

                if (vecBlockScheduler.isRunning && coreIdx < coreNum * subBlockNum) {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube1Done);
                    GDNFwdOOffsets& vec1Offsets = vecBlockScheduler.GetVec1Offsets();
                    int64_t vec1OffsetAttnMask = vec1Offsets.attnWorkOffset;
                    int64_t vec1OffsetG = vec1Offsets.gOffset;
                    int64_t vec1OffsetAttn = vec1Offsets.attnWorkOffset;
                    EpilogueGDNFwdOQkmask epilogueGDNFwdOQkmask(resource);
                    epilogueGDNFwdOQkmask(
                        gmAftermaskWorkspace[vec1OffsetAttnMask], 
                        gmG[vec1OffsetG], gmAttnWorkspace[vec1OffsetAttn], gmMask,
                        chunkSize, vec1Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag, vec1Offsets.batchIdx, vec1Offsets.headIdx, vec1Offsets.chunkIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec1Done);
                }

                // AscendC::PipeBarrier<PIPE_ALL>();

                if (needRun && coreIdx < coreNum * subBlockNum) {
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done);
                    Arch::CrossCoreWaitFlag(vecBlockScheduler.cube3Done);
                    GDNFwdOOffsets& vec2Offsets = vecBlockScheduler.GetVec2Offsets();
                    int64_t vec2OffsetO = vec2Offsets.ovOffset;
                    int64_t vec2OffsetG = vec2Offsets.gOffset;
                    int64_t vec2OffsetVWork = vec2Offsets.hvWorkOffset;
                    int64_t vec2OffsetHWork = vec2Offsets.hvWorkOffset;
                    EpilogueGDNFwdOOutput epilogueGDNFwdOOutput(resource);
                    epilogueGDNFwdOOutput(
                        gmO[vec2OffsetO], 
                        gmG[vec2OffsetG], gmVWorkspace[vec2OffsetVWork], gmHWorkspace[vec2OffsetHWork], 
                        scale, vec2Offsets.blockTokens, kHeadDim, vHeadDim, pingpongFlag, vec2Offsets.batchIdx, vec2Offsets.headIdx, vec2Offsets.chunkIdx
                    );
                    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done);
                }
                
                // AscendC::PipeBarrier<PIPE_ALL>();

                needRun = true;
            }
        }
    }
    
};

}
