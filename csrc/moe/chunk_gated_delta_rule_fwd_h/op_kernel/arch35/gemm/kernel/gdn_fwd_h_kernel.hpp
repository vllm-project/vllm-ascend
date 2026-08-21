/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#define CATLASS_ARCH 3510

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/debug.hpp"
#include "../block/block_scheduler_gdn_fwd_h.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_update.hpp"
#include "../../epilogue/block/block_epilogue_gdn_fwdh_vnew.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_multi.hpp"
#include "kernel_utils/block/block_mmad_pingpong_tla_preloadA_l1B.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
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

struct GDNFwdHTileShapes128 {
    using L1TileShape = Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
};

struct GDNFwdHTileShapes256 {
    using L1TileShape = Shape<_128, _256, _128>;
    using L0TileShape = Shape<_128, _256, _64>;
};

template<
    typename INPUT_TYPE,
    typename G_TYPE,
    typename STATE_TYPE,
    typename WORKSPACE_TYPE,
    typename TileShapes = GDNFwdHTileShapes128,
    bool kGated = false
>
class GDNFwdHKernel {
public:

    using ArchTag = Arch::Ascend950;
    using CubeScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHCube;
    using VecScheduler = typename Catlass::Gemm::Block::BlockSchedulerGdnFwdHVec;

    using DispatchPolicyTlaMulti = Gemm::MmadPingpongTlaMulti<ArchTag, true, false>;
    using DispatchPolicyTlaPreloadAL1B = Gemm::MmadPingpongTlaPreloadAL1B<ArchTag, true>;
    using L1TileShapeVTla = typename TileShapes::L1TileShape;
    using L0TileShapeVTla = typename TileShapes::L0TileShape;

    using WType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using HType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using VworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using KType = Gemm::GemmType<INPUT_TYPE, layout::ColumnMajor>;
    using HworkType = Gemm::GemmType<WORKSPACE_TYPE, layout::RowMajor>;
    using VType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using GType = Gemm::GemmType<G_TYPE, layout::RowMajor>;
    using UType = Gemm::GemmType<INPUT_TYPE, layout::RowMajor>;
    using FinalStateType = Gemm::GemmType<STATE_TYPE, layout::RowMajor>;
    using VUpdateType = Gemm::GemmType<INPUT_TYPE, layout::zN>;

    // cube 1
    using TileCopyWH = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::RowMajor, INPUT_TYPE, layout::RowMajor, WORKSPACE_TYPE, layout::RowMajor>;
    using BlockMmadWH = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMulti, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyWH>;

    // cube 2
    using TileCopyKV = Catlass::Gemm::Tile::PackedTileCopyTla<ArchTag, INPUT_TYPE, layout::ColumnMajor, INPUT_TYPE, layout::zN, WORKSPACE_TYPE, layout::RowMajor>;
    using TileMmadKV = Gemm::Tile::TileMmadTla<ArchTag, INPUT_TYPE, typename TileCopyKV::LayoutTagL1A>;
    using BlockMmadKV = Gemm::Block::BlockMmadTla<DispatchPolicyTlaMulti, L1TileShapeVTla, L0TileShapeVTla, INPUT_TYPE, INPUT_TYPE, WORKSPACE_TYPE, void, TileCopyKV>;

    // vec 1
    using DispatchPolicyGDNFwdHVnew = Epilogue::EpilogueAtlasGDNFwdHVnew;
    using EpilogueGDNFwdHVnew = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHVnew, VType, GType, UType, VworkType, VUpdateType, FinalStateType, std::bool_constant<kGated>>;

    // vec 2
    using DispatchPolicyGDNFwdHUpdate = Epilogue::EpilogueAtlasGDNFwdHUpdate;
    using EpilogueGDNFwdHUpdate = Epilogue::Block::BlockEpilogue<DispatchPolicyGDNFwdHUpdate, HType, GType, HType, HworkType, FinalStateType, std::bool_constant<kGated>>;

    using GDNFwdHOffsets = Catlass::Gemm::Block::GDNFwdHOffsets;

    using ElementK = INPUT_TYPE;
    using ElementW = INPUT_TYPE;
    using ElementU = INPUT_TYPE;
    using ElementG = G_TYPE;
    using ElementH = INPUT_TYPE;
    using ElementV = INPUT_TYPE;
    using ElementVUpdate = INPUT_TYPE;
    using ElementVWork = WORKSPACE_TYPE;
    using ElementHWork = WORKSPACE_TYPE;
    using ElementInitialState = STATE_TYPE;
    using ElementFinalState = STATE_TYPE;

    using LayoutW = Catlass::layout::RowMajor;
    using LayoutH = Catlass::layout::RowMajor;
    using LayoutV = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutVUpdate = typename VUpdateType::Layout;


    uint32_t batch;
    uint32_t seqlen;
    uint32_t kNumHead;
    uint32_t vNumHead;
    uint32_t kHeadDim;
    uint32_t vHeadDim;
    uint32_t chunkSize;
    bool useInitialState;
    bool storeFinalState;
    uint32_t isVariedLen;
    uint32_t shapeBatch;
    uint32_t tokenBatch;
    uint32_t vWorkspaceOffset;
    uint32_t vUpdateWorkspaceOffset;
    uint32_t hWorkspaceOffset;
    uint32_t numSeqWorkspaceOffset;
    uint32_t numChunksWorkspaceOffset;
    uint32_t kDecayWorkspaceOffset;

    AscendC::GlobalTensor<ElementK> gmK;
    AscendC::GlobalTensor<ElementW> gmW;
    AscendC::GlobalTensor<ElementU> gmU;
    AscendC::GlobalTensor<ElementG> gmG;
    AscendC::GlobalTensor<ElementG> gmGk;
    AscendC::GlobalTensor<ElementInitialState> gmInitialState;
    AscendC::GlobalTensor<ElementH> gmH;
    AscendC::GlobalTensor<ElementV> gmV;
    AscendC::GlobalTensor<ElementFinalState> gmFinalState;
    AscendC::GlobalTensor<ElementVWork> gmVWorkspace;
    AscendC::GlobalTensor<ElementV> gmVUpdateWorkspace;
    AscendC::GlobalTensor<ElementHWork> gmHWorkspace;
    AscendC::GlobalTensor<ElementK> gmKDecayWorkspace;

    AscendC::GlobalTensor<int64_t> gmSeqlen;
    AscendC::GlobalTensor<int64_t> gmNumSeq;
    AscendC::GlobalTensor<int64_t> gmNumChunks;

    AscendC::LocalTensor<ElementHWork> ubHUpdatePing;
    AscendC::LocalTensor<ElementHWork> ubHUpdatePong;
    AscendC::LocalTensor<ElementVWork> ubVWorkPing;
    AscendC::LocalTensor<ElementVWork> ubVWorkPong;

    AscendC::LocalTensor<ElementV> l1VUpdatePing;
    AscendC::LocalTensor<ElementV> l1VUpdatePong;

    CubeScheduler cubeBlockScheduler;
    VecScheduler vecBlockScheduler;

    Arch::Resource<ArchTag> resource;


    __aicore__ inline GDNFwdHKernel() {}

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR gk, GM_ADDR inital_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices,
        GM_ADDR h, GM_ADDR v_new, GM_ADDR final_state, GM_ADDR tiling, GM_ADDR user) {

        __gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict gdnFwdHTilingData = reinterpret_cast<__gm__ ChunkGatedDeltaRuleFwdHTilingData *__restrict>(tiling);

        batch = gdnFwdHTilingData->batch;
        seqlen = gdnFwdHTilingData->seqlen;
        kNumHead = gdnFwdHTilingData->kNumHead;
        vNumHead = gdnFwdHTilingData->vNumHead;
        kHeadDim = gdnFwdHTilingData->kHeadDim;
        vHeadDim = gdnFwdHTilingData->vHeadDim;
        chunkSize = gdnFwdHTilingData->chunkSize;
        useInitialState = gdnFwdHTilingData->useInitialState;
        storeFinalState = gdnFwdHTilingData->storeFinalState;
        isVariedLen = gdnFwdHTilingData->isVariedLen;
        shapeBatch = gdnFwdHTilingData->shapeBatch;
        tokenBatch = gdnFwdHTilingData->tokenBatch;
        vWorkspaceOffset = gdnFwdHTilingData->vWorkspaceOffset;
        vUpdateWorkspaceOffset = gdnFwdHTilingData->vUpdateWorkspaceOffset;
        hWorkspaceOffset = gdnFwdHTilingData->hWorkspaceOffset;
        numSeqWorkspaceOffset = gdnFwdHTilingData->numSeqWorkspaceOffset;
        numChunksWorkspaceOffset = gdnFwdHTilingData->numChunksWorkspaceOffset;
        kDecayWorkspaceOffset = gdnFwdHTilingData->kDecayWorkspaceOffset;

        gmK.SetGlobalBuffer((__gm__ ElementK *)k);
        gmW.SetGlobalBuffer((__gm__ ElementW *)w);
        gmU.SetGlobalBuffer((__gm__ ElementU *)u);
        gmG.SetGlobalBuffer((__gm__ ElementG *)g);
        gmGk.SetGlobalBuffer((__gm__ ElementG *)gk);
        gmInitialState.SetGlobalBuffer((__gm__ ElementInitialState *)inital_state);
        gmH.SetGlobalBuffer((__gm__ ElementH *)h);
        gmV.SetGlobalBuffer((__gm__ ElementV *)v_new);
        gmFinalState.SetGlobalBuffer((__gm__ ElementFinalState *)final_state);
        gmVWorkspace.SetGlobalBuffer((__gm__ ElementVWork *)(user + vWorkspaceOffset));
        gmVUpdateWorkspace.SetGlobalBuffer((__gm__ ElementV *)(user + vUpdateWorkspaceOffset));
        gmHWorkspace.SetGlobalBuffer((__gm__ ElementHWork *)(user + hWorkspaceOffset));
        gmKDecayWorkspace.SetGlobalBuffer((__gm__ ElementK *)(user + kDecayWorkspaceOffset));

        gmSeqlen.SetGlobalBuffer((__gm__ int64_t *)cu_seqlens);
        gmNumSeq.SetGlobalBuffer((__gm__ int64_t *)(user + numSeqWorkspaceOffset));
        gmNumChunks.SetGlobalBuffer((__gm__ int64_t *)(user + numChunksWorkspaceOffset));

        ubHUpdatePing = resource.ubBuf.template GetBufferByByte<ElementHWork>(32 * 1024);
        ubHUpdatePong = resource.ubBuf.template GetBufferByByte<ElementHWork>(96 * 1024);
        ubVWorkPing = resource.ubBuf.template GetBufferByByte<ElementVWork>(32 * 1024);
        ubVWorkPong = resource.ubBuf.template GetBufferByByte<ElementVWork>(96 * 1024);

        l1VUpdatePing = resource.l1Buf.template GetBufferByByte<ElementV>(0);
        l1VUpdatePong = resource.l1Buf.template GetBufferByByte<ElementV>(chunkSize * vHeadDim * sizeof(ElementV));

        if ASCEND_IS_AIC {
            cubeBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }

        if ASCEND_IS_AIV {
            vecBlockScheduler.Init(cu_seqlens, chunk_indices, tiling, user);
        }
    }

    __aicore__ inline void Process() {
        if (isVariedLen) {
            AscendC::SyncAll<false>();
        }

        if ASCEND_IS_AIC {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            BlockMmadWH blockMmadWH(resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);
            BlockMmadKV blockMmadKV(resource, chunkSize * cubeBlockScheduler.vBlockSize * sizeof(ElementV) * PING_PONG_STAGES);

            auto wLayout = tla::MakeLayout<ElementW, LayoutW>(shapeBatch * kNumHead * cubeBlockScheduler.totalTokens, kHeadDim);
            auto hLayout = tla::MakeLayout<ElementH, LayoutH>(shapeBatch * vNumHead * cubeBlockScheduler.totalChunks * kHeadDim, vHeadDim);

            auto kLayout = tla::MakeLayout<ElementK, LayoutK>(kHeadDim, shapeBatch * kNumHead * cubeBlockScheduler.totalTokens);
            auto hworkLayout = tla::MakeLayout<ElementHWork, LayoutH>(kHeadDim, cubeBlockScheduler.vBlockSize);

            AscendC::SyncAll<false>();
            uint32_t currStage = 0; // 0: C1, 1: C2
            while (cubeBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* C1: v_work = w @ h[i] */
                    cubeBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }

                        const GDNFwdHOffsets& cube1Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[streamId]);
                        auto vLayout = tla::MakeLayout<ElementVWork, LayoutV>(cube1Offsets.blockTokens, cube1Offsets.vBlockDim);
                        int64_t cube1OffsetW = cube1Offsets.wOffset;
                        int64_t cube1OffsetH = cube1Offsets.hSrcOffset;
                        int64_t cube1OffsetVwork = cube1Offsets.vWorkOffset;
                        auto tensorW = tla::MakeTensor(gmW[cube1OffsetW], wLayout, Catlass::Arch::PositionGM{});
                        auto tensorH = tla::MakeTensor(gmH[cube1OffsetH], hLayout, Catlass::Arch::PositionGM{});
                        auto tensorV = tla::MakeTensor(gmVWorkspace[cube1OffsetVwork], vLayout, Catlass::Arch::PositionGM{});
                        GemmCoord cube1Shape {cube1Offsets.blockTokens, cube1Offsets.vBlockDim, kHeadDim};
                        auto tensorBlockW = GetTile(tensorW, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.k()));
                        auto tensorBlockH = GetTile(tensorH, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.k(), cube1Shape.n()));
                        auto tensorBlockV = GetTile(tensorV, tla::MakeCoord(0, 0), tla::MakeShape(cube1Shape.m(), cube1Shape.n()));

                        blockMmadWH.preSetFlags();
                        blockMmadWH(tensorBlockW, tensorBlockH, tensorBlockV, cube1Shape);
                        blockMmadWH.finalWaitFlags();
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube1Done[streamId]);
                    }
                } else {
                    /* C2: h[i+1] = k.T @ v_work */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = cubeBlockScheduler.GetStreamId(i);
                        const auto& stream = cubeBlockScheduler.GetStream(i);
                        if (cubeBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& cube2Offsets = cubeBlockScheduler.GetCurTaskOffsets(stream);
                        Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec1Done[streamId]);

                        if (cubeBlockScheduler.NeedProcessStage2(stream)) {
                            // step 3: h[i+1] = k.T @ v_work
                            int64_t cube2OffsetK = kGated ? cube2Offsets.kDecayWorkOffset : cube2Offsets.wkOffset;
                            int64_t cube2OffsetVwork = cube2Offsets.vWorkOffset;
                            auto tensorK = kGated
                                ? tla::MakeTensor(gmKDecayWorkspace[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{})
                                : tla::MakeTensor(gmK[cube2OffsetK], kLayout, Catlass::Arch::PositionGM{});
                            auto vUpdateLayout = tla::MakeLayout<ElementVUpdate, LayoutVUpdate>(cube2Offsets.blockTokens, cube2Offsets.vBlockDim);
                            auto tensorVwork = tla::MakeTensor(gmVUpdateWorkspace[cube2OffsetVwork], vUpdateLayout, Catlass::Arch::PositionGM{});
                            auto tensorHwork = tla::MakeTensor(gmHWorkspace[cube2Offsets.hWorkOffset], hworkLayout, Catlass::Arch::PositionGM{});
                            GemmCoord cube2Shape{kHeadDim, cube2Offsets.vBlockDim, cube2Offsets.blockTokens};
                            auto tensorBlockK = GetTile(tensorK, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.k()));
                            auto tensorBlockVwork = GetTile(tensorVwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.k(), cube2Shape.n()));
                            auto tensorBlockHwork = GetTile(tensorHwork, tla::MakeCoord(0, 0), tla::MakeShape(cube2Shape.m(), cube2Shape.n()));

                            blockMmadKV.preSetFlags();
                            blockMmadKV(tensorBlockK, tensorBlockVwork, tensorBlockHwork, cube2Shape);
                            blockMmadKV.finalWaitFlags();
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeBlockScheduler.cube2Done[streamId]);
                    }
                }
                currStage ^= 0x01;
            }
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[0]);
            Arch::CrossCoreWaitFlag(cubeBlockScheduler.vec2Done[1]);

        }

        if ASCEND_IS_AIV {
            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();

            if (useInitialState) {
                AscendC::LocalTensor<ElementInitialState> stateUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementInitialState>(0);
                AscendC::LocalTensor<ElementInitialState> stateUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementInitialState>(96 * 1024);
                AscendC::LocalTensor<ElementH> hUbTensorPing = resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
                AscendC::LocalTensor<ElementH> hUbTensorPong = resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
                uint32_t totalChunks = isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
                uint32_t transferCount = isVariedLen ? (vecBlockScheduler.tokenBatch * vNumHead / coreNum) : (shapeBatch * vNumHead / coreNum);
                uint32_t remainderFlag = isVariedLen ? (((vecBlockScheduler.tokenBatch * vNumHead) % coreNum) != 0): (((shapeBatch * vNumHead) % coreNum) != 0);
                uint32_t step = transferCount + remainderFlag;
                uint32_t stateBlockSize = kHeadDim * vHeadDim;
                uint32_t pingpongFlag = 1;
                uint32_t start = coreIdx * step;
                uint32_t end = start + step;
                uint32_t maxLimit = isVariedLen ? vecBlockScheduler.tokenBatch * vNumHead : shapeBatch * vNumHead;
                uint32_t realEnd = min(end, maxLimit);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                for (uint32_t initialStateBlockOffset = start; initialStateBlockOffset >= start && initialStateBlockOffset < realEnd; initialStateBlockOffset++) {
                    uint32_t batchIdx = initialStateBlockOffset / vNumHead;
                    uint32_t vHeadIdx = initialStateBlockOffset % vNumHead;
                    uint32_t chunkOffset = isVariedLen ? gmNumChunks.GetValue(batchIdx) : 0;
                    uint32_t initialStateBaseOffset = initialStateBlockOffset * stateBlockSize;
                    uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                    uint32_t hBaseOffset = (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset) * stateBlockSize;
                    if (vHeadDim <= 128) {
                        AscendC::LocalTensor<ElementInitialState> stateUbTensor = pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                        AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                        auto event_id = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                        if constexpr(!std::is_same<ElementInitialState, ElementH>::value) {
                            AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateBaseOffset], stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                            AscendC::Cast(hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT, stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                            AscendC::DataCopy(gmH[hBaseOffset], hUbTensor, stateBlockSize);
                        } else {
                            AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateBaseOffset], stateBlockSize);
                            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                            AscendC::DataCopy(gmH[hBaseOffset], stateUbTensor, stateBlockSize);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                        pingpongFlag = 1 - pingpongFlag;
                    } else {
                        uint32_t stateRowTile = (32 * 1024) / (vHeadDim * sizeof(ElementH));
                        for (uint32_t rowOffset = 0; rowOffset < kHeadDim; rowOffset += stateRowTile) {
                            uint32_t rowsThisTile = Min(stateRowTile, kHeadDim - rowOffset);
                            uint32_t stateTileElems = rowsThisTile * vHeadDim;
                            uint32_t initialStateOffset = initialStateBaseOffset + rowOffset * vHeadDim;
                            uint32_t hOffset = hBaseOffset + rowOffset * vHeadDim;
                            AscendC::LocalTensor<ElementInitialState> stateUbTensor = pingpongFlag ? stateUbTensorPing : stateUbTensorPong;
                            AscendC::LocalTensor<ElementH> hUbTensor = pingpongFlag ? hUbTensorPing : hUbTensorPong;
                            auto event_id = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                            if constexpr(!std::is_same<ElementInitialState, ElementH>::value) {
                                AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
                                AscendC::Cast(hUbTensor, stateUbTensor, AscendC::RoundMode::CAST_RINT, stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);
                                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
                                AscendC::DataCopy(gmH[hOffset], hUbTensor, stateTileElems);
                            } else {
                                AscendC::DataCopy(stateUbTensor, gmInitialState[initialStateOffset], stateTileElems);
                                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
                                AscendC::DataCopy(gmH[hOffset], stateUbTensor, stateTileElems);
                            }
                            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
                            pingpongFlag = 1 - pingpongFlag;
                        }
                    }
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            } else {
                uint32_t stateRowTile = (32 * 1024) / (vHeadDim * sizeof(ElementH));
                AscendC::LocalTensor<int64_t> chunkOffsetsUb =
                    resource.ubBuf.template GetBufferByByte<int64_t>(0);
                if (isVariedLen) {
                    uint32_t chunkOffsetBytes = (vecBlockScheduler.tokenBatch + 1) * sizeof(int64_t);
                    AscendC::DataCopyParams copyParams{
                        1, static_cast<uint16_t>(chunkOffsetBytes), 0, 0};
                    AscendC::DataCopyPadParams padParams{false, 0, 0, 0};
                    AscendC::DataCopyPad(chunkOffsetsUb, gmNumChunks[0], copyParams, padParams);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID3);
                }
                auto chunkOffsets = reinterpret_cast<__ubuf__ int64_t *>(chunkOffsetsUb.GetPhyAddr());
                AscendC::LocalTensor<ElementH> hUbTensorPing =
                    resource.ubBuf.template GetBufferByByte<ElementH>(64 * 1024);
                AscendC::LocalTensor<ElementH> hUbTensorPong =
                    resource.ubBuf.template GetBufferByByte<ElementH>(160 * 1024);
                uint32_t totalChunks =
                    isVariedLen ? vecBlockScheduler.totalChunks : ((seqlen + chunkSize - 1) / chunkSize);
                uint32_t taskCount =
                    (isVariedLen ? vecBlockScheduler.tokenBatch : shapeBatch) * vNumHead;
                uint32_t step = taskCount / coreNum + ((taskCount % coreNum) != 0);
                uint32_t start = coreIdx * step;
                uint32_t realEnd = Min(start + step, taskCount);
                uint32_t pingpongFlag = 1;
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                for (uint32_t taskIdx = start; taskIdx < realEnd; ++taskIdx) {
                    uint32_t batchIdx = taskIdx / vNumHead;
                    uint32_t vHeadIdx = taskIdx % vNumHead;
                    uint32_t chunkOffset = isVariedLen ? static_cast<uint32_t>(chunkOffsets[batchIdx]) : 0;
                    uint32_t shapeBatchIdx = isVariedLen ? 0 : batchIdx;
                    uint32_t hBaseOffset =
                        (shapeBatchIdx * vNumHead * totalChunks + vHeadIdx * totalChunks + chunkOffset) *
                        kHeadDim * vHeadDim;
                    for (uint32_t rowOffset = 0; rowOffset < kHeadDim; rowOffset += stateRowTile) {
                        uint32_t rowsThisTile = Min(stateRowTile, kHeadDim - rowOffset);
                        uint32_t stateTileElems = rowsThisTile * vHeadDim;
                        AscendC::LocalTensor<ElementH> hUbTensor =
                            pingpongFlag ? hUbTensorPing : hUbTensorPong;
                        auto eventId = pingpongFlag ? EVENT_ID1 : EVENT_ID0;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        AscendC::Duplicate(hUbTensor, static_cast<ElementH>(0), stateTileElems);
                        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventId);
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventId);
                        AscendC::DataCopy(gmH[hBaseOffset + rowOffset * vHeadDim], hUbTensor, stateTileElems);
                        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
                        pingpongFlag = 1 - pingpongFlag;
                    }
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            }

            AscendC::SyncAll<false>();

            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[0]);
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[1]);

            EpilogueGDNFwdHVnew epilogueGDNFwdHVnew(resource);
            EpilogueGDNFwdHUpdate epilogueGDNFwdHUpdate(resource);
            uint32_t pongBaseEvent = 4;

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset final_state
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset h_update
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);
            uint32_t currStage = 0; // 0: V1, 1: V2
            bool event0FromMte3[PING_PONG_STAGES] = {false, false};
            bool event2FromMte3[PING_PONG_STAGES] = {!(storeFinalState && std::is_same<ElementFinalState, float>::value),
                                                      !(storeFinalState && std::is_same<ElementFinalState, float>::value)};
            while (vecBlockScheduler.isRunning) {
                if (currStage == 0) {
                    /* V1:
                     * gmV = gmU - gmVWorkspace
                     * g_buf = gmG[-1] - gmG
                     * g_buf = exp(g_buf)
                     * gmVWorkspace = g_buf * gmV
                     */
                    vecBlockScheduler.InitTasks();
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec1Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        AscendC::LocalTensor<ElementV> l1VUpdate = (i == 0) ? l1VUpdatePing : l1VUpdatePong;
                        bool waitWsFromMte3 = storeFinalState && std::is_same<ElementFinalState, float>::value &&
                                              event0FromMte3[streamId];
                        epilogueGDNFwdHVnew(
                            gmV[vec1Offsets.uvOffset], gmVUpdateWorkspace[vec1Offsets.vWorkOffset], l1VUpdate,
                            gmG[vec1Offsets.gOffset], gmU[vec1Offsets.uvOffset], gmVWorkspace[vec1Offsets.vWorkOffset],
                            gmGk[vec1Offsets.gkOffset], gmK[vec1Offsets.wkOffset], gmKDecayWorkspace[vec1Offsets.kDecayWorkOffset],
                            vec1Offsets.blockTokens, kHeadDim, vec1Offsets.vBlockDim, vHeadDim,
                            vecBlockScheduler.cube1Done[streamId], vecBlockScheduler.vec1Done[streamId],
                            vec1Offsets.isInitialState, vec1Offsets.isFinalState, storeFinalState,
                            waitWsFromMte3, (i == 0)
                        );
                        if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                            event0FromMte3[streamId] = false;
                        }
                    }
                } else {
                    /* V2: h[i+1] += h_work if i < num_chunks - 1 else None */
                    for (uint32_t i = 0; i < PING_PONG_STAGES; ++i) {
                        uint32_t streamId = vecBlockScheduler.GetStreamId(i);
                        const auto& stream = vecBlockScheduler.GetStream(i);
                        if (vecBlockScheduler.StreamIsDone(stream)) {
                            continue;
                        }
                        const GDNFwdHOffsets& vec2Offsets = vecBlockScheduler.GetCurTaskOffsets(stream);
                        if (vecBlockScheduler.NeedProcessStage2(stream)) {
                            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                                event0FromMte3[streamId] = vec2Offsets.isFinalState;
                                event2FromMte3[streamId] = !vec2Offsets.isFinalState;
                            }
                            // step 4:  h[i+1] += h_work if i < num_chunks - 1 else None
                            epilogueGDNFwdHUpdate(
                                gmH[vec2Offsets.hDstOffset], gmFinalState[vec2Offsets.finalStateOffset],
                                gmG[vec2Offsets.gOffset],
                                gmH[vec2Offsets.hSrcOffset],
                                gmHWorkspace[vec2Offsets.hWorkOffset],
                                gmGk[vec2Offsets.gkOffset],
                                vec2Offsets.blockTokens, kHeadDim, vec2Offsets.vBlockDim, vHeadDim, vecBlockScheduler.cube2Done[streamId],
                                vec2Offsets.isInitialState, vec2Offsets.isFinalState, storeFinalState, (i == 0)
                            );
                        } else {
                            Arch::CrossCoreWaitFlag(vecBlockScheduler.cube2Done[streamId]);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(vecBlockScheduler.vec2Done[streamId]);
                    }
                }
                currStage ^= 0x01;
            }

            if (storeFinalState && std::is_same<ElementFinalState, float>::value) {
                if (event0FromMte3[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }
                if (event0FromMte3[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0 + pongBaseEvent);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                }
                if (event2FromMte3[0]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                if (event2FromMte3[1]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
                } else {
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2 + pongBaseEvent);
                }
            } else {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); // preset h_update
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0 + pongBaseEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); // preset h
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2 + pongBaseEvent);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1); // preset u
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1 + pongBaseEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); // preset g
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3 + pongBaseEvent);

        }
    }

};

}
