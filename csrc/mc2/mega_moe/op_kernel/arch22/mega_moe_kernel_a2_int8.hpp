/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_KERNEL_A2_INT8_HPP
#define MEGA_MOE_KERNEL_A2_INT8_HPP

#include "kernel_operator.h"

#include "utils/copy_l0c_to_gm_custom.hpp"

#include "template_linear_algebra_v2/catlass.hpp"
#include "template_linear_algebra_v2/arch/cross_core_sync.hpp"
#include "template_linear_algebra_v2/arch/resource.hpp"
#include "template_linear_algebra_v2/coord.hpp"
#include "template_linear_algebra_v2/detail/callback.hpp"
#include "template_linear_algebra_v2/gemm_coord.hpp"
#include "template_linear_algebra_v2/matrix_coord.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_copy.hpp"

#include "utilsA2/block_mmad_preload_async_fixpipe_quant_a2.hpp"
#include "utils/copy_gm_to_l1_custom.hpp"
#include "utilsA2/block_epilogue_pertoken_swiglu_a2_int8.hpp"
#include "utilsA2/block_epilogue_pertoken_v2_int8_a2.hpp"
#include "utils/hccl_shmem.hpp"
#include "utils/const_args.hpp"
#include "utils/layout3d.hpp"

#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2.hpp"
#include "moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h"
#include "unpermute/moe_token_unpermute.h"
#include "utils/get_tensor_addr.hpp"

namespace Catlass::Gemm::Kernel {

using namespace AscendC;
using namespace Mc2Tiling;

template <class BlockMmad_, class BlockScheduler_, class ElementGroupList_,
          class BlockEpilogue1_, class BlockEpilogue2_>
class MegaMoeKernelA2Int8 {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;
    using ElementScale = uint64_t;
    using LayoutScale = typename layout::VectorLayout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = typename layout::VectorLayout;
    using BlockScheduler = BlockScheduler_;
    using BlockEpilogue1 = BlockEpilogue1_;
    using BlockEpilogue2 = BlockEpilogue2_;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    using LayoutD1 = typename BlockEpilogue1::LayoutD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using LayoutD2 = typename BlockEpilogue2::LayoutD;
    using ElementABefore = int8_t;

    static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    struct Params {
        GemmCoord problemShape;
        __gm__ ElementABefore *ptrA;
        LayoutA layoutA;
        LayoutA layoutA2;
        __gm__ ElementB *ptrB1;
        LayoutB layoutB1;
        __gm__ float *ptrBias1;
        __gm__ ElementB *ptrB2;
        LayoutB layoutB2;
        __gm__ float *ptrBias2;
        __gm__ ElementScale *ptrScale1;
        LayoutScale layoutScale1;
        __gm__ ElementScale *ptrScale2;
        LayoutScale layoutScale2;
        __gm__ ElementD2 *ptrOutput;
        LayoutD1 layoutD1;
        LayoutD2 layoutD2;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrExpertTokenNums;
        int32_t EP;
        int32_t listLen;
        int32_t expertPerRank;
        uint32_t maxOutputSize;
        GM_ADDR expertIdx;
        GM_ADDR moeInitRoutingQuantV2Scale;
        GM_ADDR moeInitRoutingQuantV2Offset;
        GM_ADDR expandedX;
        GM_ADDR expandedRowIdx;
        GM_ADDR expertTokensCountOrCumsum;
        GM_ADDR expertTokensBeforeCapacity;
        GM_ADDR dynamicQuantScale;
        GM_ADDR probs;
        GM_ADDR ptrXActiveMask;
        GM_ADDR ptrScales;
        int64_t topK;
        uint64_t initRoutingQuantTilingKey;
        uint32_t epilogueCoreNum;
        uint32_t epilogueGranularity{0};
        float swigluLimit;
        GM_ADDR contextGM{nullptr};
        MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;

        CATLASS_HOST_DEVICE Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord problemShape_, uint32_t EP_, uint32_t listLen_,
               uint32_t expertPerRank_, uint32_t maxOutputSize_, int64_t topK_,
               uint64_t initRoutingQuantTilingKey_, uint32_t epilogueCoreNum_,
               GM_ADDR contextGM_, GM_ADDR ptrA_, LayoutA layoutA_, LayoutA layoutA2_, GM_ADDR ptrB1_,
               LayoutB layoutB1_, GM_ADDR ptrBias1_, GM_ADDR ptrB2_, LayoutB layoutB2_, GM_ADDR ptrBias2_,
               GM_ADDR ptrScale1_, LayoutScale layoutScale1_, GM_ADDR ptrScale2_, LayoutScale layoutScale2_,
               GM_ADDR ptrOutput_, LayoutD1 layoutD1_, LayoutD2 layoutD2_, GM_ADDR expertIdx_,
               GM_ADDR moeInitRoutingQuantV2Scale_, GM_ADDR moeInitRoutingQuantV2Offset_,
               GM_ADDR expertTokensBeforeCapacity_, GM_ADDR probs_, GM_ADDR ptrWorkspace_, GM_ADDR gmExpertTokenNums_,
               GM_ADDR ptrXActiveMask_, GM_ADDR ptrScales_,
               MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData_,
               uint32_t epilogueGranularity_ = 0,
               float swigluLimit_ = std::numeric_limits<float>::infinity())
            : problemShape(problemShape_), EP(EP_), listLen(listLen_),
              expertPerRank(expertPerRank_), maxOutputSize(maxOutputSize_), topK(topK_),
              initRoutingQuantTilingKey(initRoutingQuantTilingKey_), epilogueCoreNum(epilogueCoreNum_),
              epilogueGranularity(epilogueGranularity_), swigluLimit(swigluLimit_), contextGM(contextGM_),
              ptrA(reinterpret_cast<__gm__ ElementABefore *>(ptrA_)),
              layoutA(layoutA_), layoutA2(layoutA2_),
              ptrB1(reinterpret_cast<__gm__ ElementB *>(ptrB1_)), layoutB1(layoutB1_),
              ptrBias1(reinterpret_cast<__gm__ float *>(ptrBias1_)),
              ptrB2(reinterpret_cast<__gm__ ElementB *>(ptrB2_)), layoutB2(layoutB2_),
              ptrBias2(reinterpret_cast<__gm__ float *>(ptrBias2_)),
              ptrScale1(reinterpret_cast<__gm__ ElementScale *>(ptrScale1_)), layoutScale1(layoutScale1_),
              ptrScale2(reinterpret_cast<__gm__ ElementScale *>(ptrScale2_)), layoutScale2(layoutScale2_),
              ptrOutput(reinterpret_cast<__gm__ ElementD2 *>(ptrOutput_)), layoutD1(layoutD1_), layoutD2(layoutD2_),
              expertIdx(expertIdx_),
              moeInitRoutingQuantV2Scale(moeInitRoutingQuantV2Scale_), moeInitRoutingQuantV2Offset(moeInitRoutingQuantV2Offset_),
              expertTokensBeforeCapacity(expertTokensBeforeCapacity_), probs(probs_),
              ptrXActiveMask(ptrXActiveMask_), ptrScales(ptrScales_),
              ptrWorkspace(ptrWorkspace_), ptrExpertTokenNums(gmExpertTokenNums_),
              moeInitRoutingQuantV2TilingData(moeInitRoutingQuantV2TilingData_) {}
    };

    CATLASS_DEVICE
    MegaMoeKernelA2Int8(Params const &params)
    {
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            coreNum = AscendC::GetBlockNum();
        }

        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
            serverId_ = RuntimeRank(params) / SERVER_RANK_SIZE_A2;
        }
        initBuffer(params);
    }

    CATLASS_DEVICE
    ~MegaMoeKernelA2Int8()
    {
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        GMM1(params);
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
        GMM2(params);
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params)
    {
        DispatchAndCombine(params);
    }

private:
    CATLASS_DEVICE int32_t RuntimeRank(Params const &params) const
    {
        (void)params;
        return shmem.Rank();
    }

    CATLASS_DEVICE void initBuffer(Params const &params)
    {
        auto tmpContext = reinterpret_cast<__gm__ Mc2Aclnn::Mc2MoeContext *>(params.contextGM);
        shmem.initShmem(tmpContext);
        qp_info_ = reinterpret_cast<__gm__ HcclAiRMAInfo *>(
            reinterpret_cast<__gm__ HcclA2CombineOpParam *>(shmem.WinContext_)->aiRMAInfo);
        workspaceInfo = WorkspaceInfo(params);
        peermemInfo = PeermemInfo(params, shmem);
        cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrcumsumMM));
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(shmem() + peermemInfo.offsetA));
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC));
        gmPermutedToken.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1 *>(workspaceInfo.ptrPermutedToken));
        gmC2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(workspaceInfo.ptrC2));
        gmPerTokenScale1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(shmem() + peermemInfo.offsetPeerPerTokenScale));
        gmPerTokenScale2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale *>(workspaceInfo.ptrPerTokenScale2));
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert));
        tokenPerExpertLayout = Layout3D(AlignUp(params.EP * params.expertPerRank, ALIGN_128), params.expertPerRank);
        preSumBeforeRankForDispatch.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRankForDispatch));
        preSumBeforeRankForCombine.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspaceInfo.ptrSumBeforeRankForCombine));
    }

    template<typename T>
    CATLASS_DEVICE void CopyGMToGM(
        AscendC::GlobalTensor<T> dst,
        AscendC::GlobalTensor<T> src,
        int32_t elemNum,
        int32_t ubMoveNum
    )
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;
        constexpr int32_t BufferNum = 2;
        int tmpBufferSize = 32 * 1024 / sizeof(T);   // 32 KB
        AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
        tmpBuffer1.SetSize(tmpBufferSize);
        int tmpBufferOffset = 96 * 1024; // half of UB
        AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
        tmpBuffer2.SetSize(tmpBufferSize);

        int pingpongId = 0;
        auto processCount = CeilDiv(elemNum, ubMoveNum);
        for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
            uint32_t curProcessNum = (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
            auto processOffset = processIndex * ubMoveNum;

            auto inputOffset = processOffset;
            auto outputOffset = processOffset;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            copyGmToUb(buf, src[inputOffset], layout::RowMajor{ 1, curProcessNum}, layout::RowMajor{1, curProcessNum});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
            copyUbToGm(dst[outputOffset], buf, layout::RowMajor{ 1, curProcessNum}, layout::RowMajor{1, curProcessNum});

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            pingpongId = (pingpongId + 1) % BufferNum;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    // ============================================================
    // SendTokensV3：发送侧，三路径（self-rank / 同server / 跨server）
    //   - rows == 0 时提前 return（不写 flag，避免对端死等）
    //   - 路径 A（self-rank）：windowsOut→UB→本地 peer mem，inline 做 scale 分离，无需 flag
    //   - 路径 B（同 server）：IPC 直写远端 peer mem，完成后 DataCopyPad 写 epoch flag
    //   - 路径 C（跨 server）：批量 RDMA 数据，再 RDMA epoch flag
    // ============================================================
    template<typename T>
    CATLASS_DEVICE void SendTokensV3(
        AscendC::GlobalTensor<T> dst,   // 目标 rank peer mem（已含 rowStart * copyInNum 偏移）
        AscendC::GlobalTensor<T> src,   // 本 rank windowsOut（已含 rowSrc * copyInNum 偏移）
        int32_t rows,                   // 发送 token 数
        uint32_t hiddenSize,
        int32_t dstEpIdx,
        int32_t groupIdx,               // 当前专家组轮次（用于 epoch flag 值）
        int32_t rowStart,              // dst 起始行号（路径 A 写 scale 时使用）
        const Params & params
    )
    {
        // rows == 0：无 token 可发，不写 flag，对端 RecvTokensV3 同样 rows2==0 会提前 return
        if (rows == 0) return;

        uint32_t copyInNum = hiddenSize + ALIGN_512;

        // UB 布局：
        //   [0, copyInNum*sizeof(T))         : bufT（token 数据 + scale）
        //   [ubRdmaOffset, +UB_ALIGN)        : rdmaUbLocal（RDMA doorbell 用）
        //   [ubRdmaOffset+UB_ALIGN, +UB_ALIGN): rdmaUbLocalHead
        //   [ubFlagOffset, +UB_ALIGN)        : flagUb（epoch flag 值）
        uint64_t ubRdmaOffset = 2 * copyInNum * sizeof(uint32_t);
        AscendC::LocalTensor<uint64_t> rdmaUbLocal  = resource.ubBuf.template GetBufferByByte<uint64_t>(ubRdmaOffset);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead = resource.ubBuf.template GetBufferByByte<uint32_t>(ubRdmaOffset + UB_ALIGN);
        uint64_t ubFlagOffset = ubRdmaOffset + 2 * UB_ALIGN;
        AscendC::LocalTensor<int32_t> flagUb = resource.ubBuf.template GetBufferByByte<int32_t>(ubFlagOffset);

        AscendC::LocalTensor<T> bufT = resource.ubBuf.template GetBufferByByte<T>(0);
        AscendC::LocalTensor<float> bufScale = bufT[hiddenSize].template ReinterpretCast<float>();

        int32_t dstServerId = dstEpIdx / SERVER_RANK_SIZE_A2;

        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;

        // ── 路径 A：self-rank ──────────────────────────────────────────
        // windowsOut → UB → 本地 peer mem（只写 hiddenSize 有效数据）
        // scale inline 分离写入 gmScale，无需 epoch flag
        if (static_cast<int32_t>(RuntimeRank(params)) == dstEpIdx) {
            AscendC::GlobalTensor<ElementPerTokenScale> gmScale;
            gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale*>(
                shmem() + peermemInfo.offsetPeerPerTokenScale));

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            for (int32_t i = 0; i < rows; ++i) {
                auto offset = i * copyInNum;
                // windowsOut → UB (MTE2)
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                copyGmToUb(bufT, src[offset],
                    layout::RowMajor{1, copyInNum}, layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                // UB → 本地 peer mem（只写 hiddenSize 有效数据）
                copyUbToGm(dst[offset], bufT,
                    layout::RowMajor{1, hiddenSize}, layout::RowMajor{1, hiddenSize});
                // scale 分离写入独立 gmScale 区
                AscendC::DataCopyPad(gmScale[rowStart + i], bufScale, {1, 4, 0, 0, 0});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            // 自身 rank 无需写 flag，RecvTokensV3 中 srcEpIdx == rank 时直接 return
            return;
        }

        // ── 路径 B：同 server（IPC 直写远端 peer mem + IPC DataCopyPad flag）──
        if (dstServerId == serverId_) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            for (int32_t i = 0; i < rows; ++i) {
                auto offset = i * copyInNum;
                // windowsOut → UB (MTE2)
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                copyGmToUb(bufT, src[offset],
                    layout::RowMajor{1, copyInNum}, layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
                // UB → 远端 peer mem（IPC 直写，写完整 copyInNum，scale 由接收侧分离）
                copyUbToGm(dst[offset], bufT,
                    layout::RowMajor{1, copyInNum}, layout::RowMajor{1, copyInNum});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            // 写远端 epoch flag（IPC 直写，epoch = groupIdx + 1）
            // 接收侧用 gm_signal_wait_until_ne(flagAddr, 0) 等待不等于旧值
            AscendC::GlobalTensor<int32_t> remoteFlag;
            remoteFlag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                shmem(0, dstEpIdx) + peermemInfo.offsetFlag
                + (RuntimeRank(params) * params.expertPerRank + groupIdx) * sizeof(uint32_t)));
            flagUb.SetValue(0, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(remoteFlag, flagUb,
                AscendC::DataCopyParams{1, sizeof(int32_t), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();
            return;
        }

        // ── 路径 C：跨 server（批量 RDMA 数据 + flag RDMA）─────────────
        {
            // 1. 将 flag 值写入本地 windowsOut flag 区（RDMA-accessible），供 flag RDMA 使用
            AscendC::GlobalTensor<int32_t> localFlagGm;
            localFlagGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
                shmem.windowsOutAddr() + peermemInfo.offsetFlag
                + (RuntimeRank(params) * params.expertPerRank + groupIdx) * sizeof(uint32_t)));
            flagUb.SetValue(0, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(localFlagGm, flagUb,
                AscendC::DataCopyParams{1, sizeof(int32_t), 0, 0});
            AscendC::PipeBarrier<PIPE_ALL>();

            // 2. 批量 RDMA 数据（rows 个 token 在 src 中连续，一次完成）
            //    src 已指向 windowsOut 中 rowSrc 起始位置，RDMA-registered
            uint64_t totalBytes = rows * copyInNum * sizeof(T);
            AIVRDMAPostSend(
                (GM_ADDR)src.GetPhyAddr(),
                (GM_ADDR)dst.GetPhyAddr(),
                static_cast<uint64_t>(dstEpIdx), totalBytes,
                qp_info_, rdmaUbLocal, rdmaUbLocalHead);
            AscendC::PipeBarrier<PIPE_ALL>();

            // 3. flag RDMA（同 QP 顺序保证：flag 在数据之后到达接收侧）
            GM_ADDR remoteFlagAddr = reinterpret_cast<GM_ADDR>(
                reinterpret_cast<uintptr_t>(shmem(0, dstEpIdx))
                + static_cast<uintptr_t>(peermemInfo.offsetFlag)
                + static_cast<uintptr_t>(RuntimeRank(params) * params.expertPerRank + groupIdx) * sizeof(uint32_t));
            AIVRDMAPostSend(
                (GM_ADDR)localFlagGm.GetPhyAddr(), remoteFlagAddr,
                static_cast<uint64_t>(dstEpIdx), sizeof(int32_t),
                qp_info_, rdmaUbLocal, rdmaUbLocalHead);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ============================================================
    // RecvTokensV3：接收侧，等待 epoch flag 后做 scale 分离
    //   - srcEpIdx == rank：路径 A 已处理，直接 return
    //   - rows2 == 0：无 token 可接收，直接 return（避免等待永远不到的 flag）
    //   - 等 flag：gm_signal_wait_until_ne(addr, groupIdx)（等到值 != groupIdx，即 >= groupIdx+1）
    //   - 数据已就绪：逐 token CopyGm2Ub + CopyUb2Gm（只写 hiddenSize）+ DataCopyPad scale
    //   - 使用 catlass Tile copy，与 SendTokensV3 对齐一致性，正确处理非对齐边界
    // ============================================================
    template<typename T>
    CATLASS_DEVICE void RecvTokensV3(
        int32_t rows2,                  // 期望接收 token 数
        int32_t rowStart2,              // 在本 rank peer mem 中的起始行号
        uint32_t hiddenSize,
        int32_t srcEpIdx,               // 数据来源 rank
        int32_t groupIdx,               // 当前专家组轮次（用于 epoch flag 判定）
        const Params & params
    )
    {
        // 自身 rank 已在 SendTokensV3 路径 A 中完成搬运（含 scale 分离），跳过
        if (srcEpIdx == static_cast<int32_t>(RuntimeRank(params))) return;
        // 无 token 时跳过，避免等待永远不会到达的 flag
        if (rows2 == 0) return;

        uint32_t copyInNum = hiddenSize + ALIGN_512;

        AscendC::LocalTensor<T> bufT = resource.ubBuf.template GetBufferByByte<T>(0);
        AscendC::LocalTensor<float> bufScale = bufT[hiddenSize].template ReinterpretCast<float>();

        // 1. 等待 epoch flag：轮询直到 flag != groupIdx（即发送侧写入 groupIdx + 1）
        //    flag 地址：本 rank peer mem 的 flag 区，srcEpIdx 对应的 slot
        __gm__ int32_t* flagAddr = reinterpret_cast<__gm__ int32_t*>(
            shmem() + peermemInfo.offsetFlag + (srcEpIdx * params.expertPerRank + groupIdx) * sizeof(uint32_t));
        gm_signal_wait_until_ne(flagAddr, 0);

        // 2. flag 到达 ⇒ 全部 rows2 个 token 均已写入本地 peer mem
        //    使用 catlass CopyGm2Ub/CopyUb2Gm，与 SendTokensV3 保持一致：
        //    - 读入完整 copyInNum（有效数据 + scale 占位），layout 语义明确
        //    - 写回仅 hiddenSize 有效数据，catlass 内部处理非对齐 tail padding
        using TType = Gemm::GemmType<T, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;

        AscendC::GlobalTensor<T> gmDstA;
        gmDstA.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(shmem() + peermemInfo.offsetA));
        AscendC::GlobalTensor<ElementPerTokenScale> gmScale;
        gmScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementPerTokenScale*>(
            shmem() + peermemInfo.offsetPeerPerTokenScale));

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        for (int32_t i = 0; i < rows2; ++i) {
            auto offset = (rowStart2 + i) * copyInNum;
            // GM → UB（读完整 copyInNum：含有效数据 + scale 部分）
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            copyGmToUb(bufT, gmDstA[offset],
                layout::RowMajor{1, copyInNum}, layout::RowMajor{1, copyInNum});
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
            // UB → GM（只写回 hiddenSize 有效数据，tail padding 由 catlass 内部处理）
            copyUbToGm(gmDstA[offset], bufT,
                layout::RowMajor{1, hiddenSize}, layout::RowMajor{1, hiddenSize});
            // scale 分离写入独立 gmScale 区
            AscendC::DataCopyPad(gmScale[rowStart2 + i], bufScale, {1, 4, 0, 0, 0});
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void GetCumsumForMMAIV(AscendC::GlobalTensor<int32_t> & tokenPerExpert, AscendC::GlobalTensor<int32_t> & result, uint32_t expertPerRank, uint32_t EP)
    {
        int32_t expertPerRankAligned = (EP * expertPerRank + 8 - 1) / 8 * 8;
        AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> tmpResult = resource.ubBuf.template GetBufferByByte<int32_t>(EP * EP * expertPerRank * sizeof(int32_t));
#define U16(x) static_cast<uint16_t>(x)

        AscendC::DataCopyPad(
            tmpBuffer1,
            tokenPerExpert,
            {U16(EP), U16(EP * expertPerRank * sizeof(int32_t)),
                U16((AlignUp(EP * expertPerRank, 128) - EP * expertPerRank) * sizeof(int32_t)), 0},
            {}
        );

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[i * expertPerRankAligned], tmpBuffer1[(i - 1) * expertPerRankAligned], EP * expertPerRank);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(
            result,
            tmpBuffer1,
            {U16(EP), U16((EP * expertPerRank) * sizeof(int32_t)),
                0, U16((AlignUp(EP * expertPerRank, 128) - EP * expertPerRank) * sizeof(int32_t))}
        );
    }

    CATLASS_DEVICE
    void GMM1(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;
        uint32_t startCoreIdx = 0;
        uint32_t syncGroupIdx = 0;
        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;

        uint16_t syncgmmIdx = 0;

        AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT); // Wait for AIV to finish cumsum for matmul
        syncgmmIdx++;
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM >= params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }
            AscendC::GlobalTensor<ElementB> gmB1;
            AscendC::GlobalTensor<ElementScale> gmS;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB1.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(GetTensorAddr<int8_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB1))));
            gmS.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale1))));
            AscendC::PipeBarrier<PIPE_ALL>();

            if (currentM <= L1TileShape::M) {
                gmB1.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, params.problemShape.n(), params.problemShape.k()};
            // LayoutA layoutA = params.layoutA.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutA layoutA{inGroupProblemShape.m(), static_cast<uint32_t>(inGroupProblemShape.k() + ALIGN_512)};
            LayoutB layoutB1 = params.layoutB1;
            LayoutScale layoutScale = params.layoutScale1;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                for(;syncGroupIdx <= groupIdx; syncGroupIdx++) {
                    AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
                    syncgmmIdx ++;
                }

                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB1.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * params.problemShape.n() : 0);
                if (currentM > 0) {
                    blockMmad(
                        gmA[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmB1[gmGroupOffsetB + gmOffsetB], layoutB1,
                        gmC[gmGroupOffsetC + gmOffsetC], layoutC,
                        gmS[gmOffsetS], layoutScale, actualBlockShape);
                }
            }

            if ((groupIdx + 1) == params.epilogueGranularity  && (groupIdx < params.expertPerRank - 1)) {
                syncLoopIdx ++;
                if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
                    blockMmad.SynchronizeBlock();
                }
                // Synchronization signal: GMM1 notifies SwiGLU [1]
                blockMmad.Finalize(syncLoopIdx, SYNCFLAGC2V);
            }

            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * (inGroupProblemShape.k() + ALIGN_512);
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx  + coreLoops) % coreNum;
        }

        for(;syncGroupIdx < params.expertPerRank; syncGroupIdx++) {
            AscendC::CrossCoreWaitFlag<0x2>(syncgmmIdx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmmIdx ++;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        // Synchronization signal: GMM1 notifies SwiGLU [2]
        blockMmad.Finalize(syncLoopIdx + 1, SYNCFLAGC2V);
    }

    CATLASS_DEVICE
    void GMM2(Params const &params)
    {
        icache_preload(8);
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);

        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;

        int64_t gmGroupOffsetA = 0;
        int64_t gmGroupOffsetB = 0;
        int64_t gmGroupOffsetC = 0;

        uint32_t startCoreIdx = 0;

        AscendC::PipeBarrier<PIPE_ALL>();

        int64_t preCurrentmSum = 0;
        int32_t syncLoopIdx = -1;
        uint32_t lastDequantExpertNum = params.expertPerRank;

        if (params.epilogueGranularity < params.expertPerRank) {
            lastDequantExpertNum = params.expertPerRank - params.epilogueGranularity;
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentM = cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
            if (preCurrentmSum >= params.maxOutputSize) {
                currentM = 0;
            } else if (preCurrentmSum + currentM > params.maxOutputSize) {
                currentM = params.maxOutputSize - preCurrentmSum;
            }
            AscendC::GlobalTensor<ElementB> gmB2;
            AscendC::GlobalTensor<ElementScale> gmS2;
            int32_t arrayGroupIdx = params.listLen == 1 ? 0 : groupIdx;
            gmB2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(GetTensorAddr<int8_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrB2))));
            gmS2.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale *>(GetTensorAddr<int64_t>(arrayGroupIdx, reinterpret_cast<GM_ADDR>(params.ptrScale2))));
            if (currentM <= L1TileShape::M) {
                gmB2.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            GemmCoord inGroupProblemShape{currentM, n2, k2}; // M N K

            LayoutA layoutA = params.layoutA2.GetTileLayout(inGroupProblemShape.GetCoordMK());
            LayoutB layoutB2 = params.layoutB2;
            LayoutScale layoutScale = params.layoutScale2;
            LayoutC layoutC = LayoutC(inGroupProblemShape.m(), inGroupProblemShape.n());

            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();

            // Determine the starting loopIdx of the current core under the current groupIdx
            uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;
            // Loop through the matmul of each groupIdx
            if (params.expertPerRank > lastDequantExpertNum && groupIdx + 1 == params.expertPerRank - lastDequantExpertNum) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGV2C);
            }

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
                if (loopIdx + coreNum >= coreLoops) {
                    syncLoopIdx = groupIdx;
                }

                // Compute block location
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);

                // Compute initial location in logical coordinates
                MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
                MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
                MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};

                int64_t gmOffsetA = layoutA.GetOffset(offsetA);
                int64_t gmOffsetB = layoutB2.GetOffset(offsetB);
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetS = blockCoord.n() * L1TileShape::N + (params.listLen == 1 ? groupIdx * n2 : 0);   // One scale group per expert
                if (currentM > 0) {
                    blockMmad(
                        gmPermutedToken[gmGroupOffsetA + gmOffsetA], layoutA,
                        gmB2[gmGroupOffsetB + gmOffsetB], layoutB2,
                        gmC2[gmGroupOffsetC + gmOffsetC], layoutC,
                        gmS2[gmOffsetS], layoutScale,
                        actualBlockShape, syncLoopIdx, 0);
                }
            }
            preCurrentmSum += currentM;
            gmGroupOffsetA += inGroupProblemShape.m() * inGroupProblemShape.k();
            if (params.listLen == 1) {
                gmGroupOffsetB += inGroupProblemShape.k() * inGroupProblemShape.n();
            }
            gmGroupOffsetC += inGroupProblemShape.m() * inGroupProblemShape.n();
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }
        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
    }

    CATLASS_DEVICE
    void InitArithProgress(Params const &params)
    {
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::Duplicate(tmpBuffer1, 0.0f, (params.EP + 1) * FLAGSTRIDE);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::GlobalTensor<float> flagGlobalBase;
        flagGlobalBase.SetGlobalBuffer(workspaceInfo.ptrSoftFlagBase);
        AscendC::DataCopy(flagGlobalBase, tmpBuffer1, (params.EP + 1) * FLAGSTRIDE);
    }

    CATLASS_DEVICE
    void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params, int64_t localTokenPerExpertOffset)
    {
        uint32_t numPerCore = AlignUp(params.EP * params.expertPerRank, 128);
        AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];
        uint64_t ubOffet = 2 * numPerCore * sizeof(uint32_t);
        AscendC::LocalTensor<uint64_t> rdmaUbLocal = resource.ubBuf.template GetBufferByByte<uint64_t>(ubOffet);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead = resource.ubBuf.template GetBufferByByte<uint32_t>(ubOffet + UB_ALIGN);

        for(int32_t dstEpIdx = coreIdx, dstServerId = 0; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            dstServerId = dstEpIdx / SERVER_RANK_SIZE_A2;
            AscendC::GlobalTensor<int32_t> srcAddress;
            srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + localTokenPerExpertOffset));
            AscendC::GlobalTensor<int32_t> dstAddress;
            __gm__ void* dstPeermemPtr = shmem(localTokenPerExpertOffset, dstEpIdx);

            dstAddress.SetGlobalBuffer((__gm__ int32_t * )dstPeermemPtr);
            AscendC::GlobalTensor<int32_t> localDstAddress;
            GM_ADDR localDstPeermemPtr = shmem.windowsOutAddr() + peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(dstEpIdx, 0, 0) * sizeof(int32_t);
            localDstAddress.SetGlobalBuffer((__gm__ int32_t * )localDstPeermemPtr);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            using TType = Gemm::GemmType<int32_t, layout::RowMajor>;
            using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
            using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
            CopyGmToUb copyGmToUb;
            CopyUbToGm copyUbToGm;

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

            copyGmToUb(tmpBuffer, srcAddress[0],
                layout::RowMajor{ 1, numPerCore},
                layout::RowMajor{1, numPerCore});

            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if (dstServerId == serverId_) {
                copyUbToGm(dstAddress, tmpBuffer,
                    layout::RowMajor{ 1, numPerCore},
                    layout::RowMajor{1, numPerCore});
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            } else {
                // copy to windows out
                copyUbToGm(localDstAddress, tmpBuffer,
                    layout::RowMajor{ 1, numPerCore},
                    layout::RowMajor{1, numPerCore});
                AscendC::PipeBarrier<PIPE_ALL>();
                AIVRDMAPostSend((GM_ADDR)localDstAddress.GetPhyAddr(), (GM_ADDR)dstAddress.GetPhyAddr(), dstEpIdx, numPerCore * sizeof(int32_t),
                                qp_info_, rdmaUbLocal, rdmaUbLocalHead);
            }
        }

        for(int32_t dstEpIdx = coreIdx; dstEpIdx < params.EP; dstEpIdx += coreNum) {
            int32_t intPer512 = CACHE_LINE / sizeof(int);
            for(int32_t checkIdx = 0; checkIdx < AlignUp(params.EP * params.expertPerRank, 128); checkIdx += intPer512) {
                __gm__ int32_t* sync_check = reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetPeerTokenPerExpert) + tokenPerExpertLayout(dstEpIdx, 0, checkIdx);
                gm_signal_wait_until_ne(sync_check, 0);
            }
            AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], tmpBuffer, numPerCore);

            // 只有 dstEpIdx == RuntimeRank(params) 的 core 才拥有本 rank 自身的 token 分布，
            // 只允许该 core 写 preSumBeforeRank，避免其他 core 覆盖正确数据（写竞争修复）
            if (static_cast<uint32_t>(dstEpIdx) == RuntimeRank(params)) {
                for (uint32_t i = 0, currentSum = 0; i < numPerCore; i++) {
                    prevSumBuf(i) = currentSum;
                    currentSum += tmpBuffer(i);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::DataCopyPad(preSumBeforeRankForDispatch, prevSumBuf,
                    AscendC::DataCopyParams{1, static_cast<uint16_t>(numPerCore * sizeof(int32_t)), 0, 0});
            }
            for (int32_t i = 0, j = 0, prevSum = 0; i < (RuntimeRank(params) + 1) * params.expertPerRank; i++) {
                if (i >= RuntimeRank(params) * params.expertPerRank) {
                    prevSumBuf(j) = prevSum;
                    j++;
                }
                prevSum += tmpBuffer(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(preSumBeforeRankForCombine[dstEpIdx * params.expertPerRank], prevSumBuf,
                AscendC::DataCopyParams{1, static_cast<uint16_t>(params.expertPerRank * sizeof(int32_t)), 0, 0});
        }

        AscendC::SyncAll<true>();
    }

    CATLASS_DEVICE
    void ResetTokenPerExpert(int32_t num)
    {
        if (coreIdx != coreNum - 1) {
            return;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
        AscendC::Duplicate(tmp, 0, num);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(tokenPerExpert, tmp, num);
    }

    CATLASS_DEVICE
    void UpdateAicFlags(const Params &params)
    {
        float flagBase = 1.0f * params.expertPerRank;
        __gm__ float* aicFinishPtr = workspaceInfo.ptrSoftFlagBase + params.EP * FLAGSTRIDE;
        float flag = 0.0f;
        float lastflag = -1.0f;
        AscendC::LocalTensor<float> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<float>(0);
        __gm__ float* flagPtr = workspaceInfo.ptrSoftFlagBase;
        AscendC::GlobalTensor<float> flagGM;
        flagGM.SetGlobalBuffer(flagPtr);
        int32_t flagBufferSize = max(4, params.EP) * FLAGSTRIDE;
        AscendC::LocalTensor<float> dstValueBuffer = resource.ubBuf.template GetBufferByByte<float>(flagBufferSize);
        AscendC::LocalTensor<float> sharedTmpBuffer = resource.ubBuf.template GetBufferByByte<float>((flagBufferSize + 64));
        uint64_t mask[1] = {0};
        uint32_t repeatNum = (flagBufferSize / (4 * FLAGSTRIDE));
        for (int32_t i = 0; i < 4; i ++) {
            if (i < params.EP) {
                mask[0] |= 1ull * (1ull << (i * 16));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        while (flag < flagBase) {
            flag = flagBase;
            AscendC::DataCopy(tmpBuffer1, flagGM, params.EP * FLAGSTRIDE);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            AscendC::ReduceMin<float>(dstValueBuffer, tmpBuffer1, sharedTmpBuffer, mask, repeatNum, 8, false);

            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            flag = min(flag, dstValueBuffer.GetValue(0));
            if (flag > lastflag) {
                *aicFinishPtr = flag;
                gm_dcci(aicFinishPtr);
                lastflag = flag;
            }
        }
    }

    CATLASS_DEVICE
    void CombineSetFlag()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void DispatchAndCombine(Params const &params)
    {
        icache_preload(8);
        int64_t localTokenPerExpertOffset = peermemInfo.offsetPeerTokenPerExpert + tokenPerExpertLayout(RuntimeRank(params), 0, 0) * sizeof(int32_t);
        GM_ADDR localTokenPerExpert = shmem.windowsOutAddr() + localTokenPerExpertOffset;     // Place the entire communication matrix in peermem
        uint32_t expandedRowIdxOffset = AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);
        //---initRouting------
        moe_init_routing_quant_v2<ElementD2>(reinterpret_cast<GM_ADDR> (params.ptrA), params.expertIdx,
        params.moeInitRoutingQuantV2Scale, params.moeInitRoutingQuantV2Offset, shmem.windowsOutAddr() + peermemInfo.offsetA,
        workspaceInfo.expandedRowIdx, localTokenPerExpert, params.expertTokensBeforeCapacity,
        shmem.windowsOutAddr() + peermemInfo.offsetPeerPerTokenScale,
        params.ptrWorkspace + expandedRowIdxOffset,
        &params.moeInitRoutingQuantV2TilingData, params.initRoutingQuantTilingKey);

        AscendC::SyncAll<true>();
        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, localTokenPerExpertOffset);

        // ── V3 分发通信：epoch flag 替代 NaN in-band，批量 RDMA 替代逐 token RDMA ────────
        if (coreIdx == 0) {
            GetCumsumForMMAIV(tokenPerExpert, cumsumMM, params.expertPerRank, params.EP);
        }

        int32_t prevSum = 0;
        if (coreIdx < params.EP) {
            prevSum = preSumBeforeRankForDispatch(coreIdx * params.expertPerRank);
        }
        AscendC::SyncAll<true>();

        AscendC::GlobalTensor<int32_t> ExpertTokenNums;
        ExpertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.ptrExpertTokenNums));
        if(coreIdx == 0) {
            CopyGMToGM(ExpertTokenNums, cumsumMM[tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), 0)], params.expertPerRank, UB_MOVE_NUM);
        }

        uint16_t syncgmm1Idx = 0;
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
        syncgmm1Idx++;

        // prevGroupSum1:   本 core 所处理的 dstEpIdx 在 peer mem 中已接收的 token 基准偏移
        // prevGroupSum2:   RuntimeRank(params) 的 peer mem 中已接收 token 的基准偏移（所有 core 共享）
        uint32_t prevGroupSum1 = 0;
        uint32_t dequantSum1 = 0;
        uint32_t dequantSum2 = 0;
        uint32_t prevGroupSum2 = 0;

        icache_preload(8);
        for (int32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            // RuntimeRank(params) 本轮专家组接收的 token 总数（所有 source rank 之和）
            uint32_t currentRankM = static_cast<uint32_t>(
                cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx)));
            // ── SEND 阶段：各 core 并行发送到对应 dstEpIdx ───────────
            // currentMSend: 本 core 处理的 dstEpIdx 接收到的 token 总数
            // （用于更新 prevGroupSum1；EP ≤ coreNum 时每 core 恰好处理 1 个 dstEpIdx）
            uint32_t currentMSend = 0;

            for (int32_t dstEpIdx = static_cast<int32_t>(coreIdx); dstEpIdx < params.EP; dstEpIdx += static_cast<int32_t>(coreNum)) {
                uint32_t rowStart = prevGroupSum1 +
                    (RuntimeRank(params) == 0 ? 0u : static_cast<uint32_t>(
                        cumsumMM(tokenPerExpertLayout(RuntimeRank(params) - 1, dstEpIdx, groupIdx))));
                currentMSend = static_cast<uint32_t>(
                    cumsumMM(tokenPerExpertLayout(params.EP - 1, dstEpIdx, groupIdx)));
                if (rowStart < params.maxOutputSize) {
                    uint32_t rows = tokenPerExpert(tokenPerExpertLayout(RuntimeRank(params), dstEpIdx, groupIdx));
                    if (rowStart + rows > params.maxOutputSize) {
                        rows = params.maxOutputSize - rowStart;
                    }
                    uint32_t rowSrc = prevSum;
                    prevSum += rows;
                    AscendC::GlobalTensor<ElementA> gmSrcA;
                    gmSrcA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(
                        shmem.windowsOutAddr() + peermemInfo.offsetA));
                    int64_t gmSrcOffset = static_cast<int64_t>(rowSrc) * (params.problemShape.k() + ALIGN_512);

                    AscendC::GlobalTensor<ElementA> gmRemoteDstA;
                    gmRemoteDstA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(
                        shmem(0, dstEpIdx) + peermemInfo.offsetA));
                    int64_t gmDstOffset = static_cast<int64_t>(rowStart) * (params.problemShape.k() + ALIGN_512);

                    SendTokensV3<ElementA>(
                        gmRemoteDstA[gmDstOffset], gmSrcA[gmSrcOffset],
                        static_cast<int32_t>(rows), params.problemShape.k(),
                        dstEpIdx, groupIdx, static_cast<int32_t>(rowStart), params);
                }
            }
            AscendC::SyncAll<true>();  // 等待所有 core 发送完成

            // ── RECV 阶段：各 core 并行等 epoch flag 后做 scale 分离 ──
            for (int32_t srcEpIdx = static_cast<int32_t>(coreIdx); srcEpIdx < params.EP; srcEpIdx += static_cast<int32_t>(coreNum)) {
                // rowStart2：srcEpIdx 的 token 在本 rank peer mem 中的起始行号
                uint32_t rowStart2 = prevGroupSum2 +
                    (srcEpIdx == 0 ? 0u : static_cast<uint32_t>(
                        cumsumMM(tokenPerExpertLayout(srcEpIdx - 1, RuntimeRank(params), groupIdx))));
                if (rowStart2 < params.maxOutputSize) {
                    uint32_t rows2 = static_cast<uint32_t>(
                        tokenPerExpert(tokenPerExpertLayout(srcEpIdx, RuntimeRank(params), groupIdx)));
                    if (rows2 + rowStart2 > params.maxOutputSize) {
                        rows2 = params.maxOutputSize - rowStart2;
                    }
                    RecvTokensV3<ElementA>(
                        static_cast<int32_t>(rows2), static_cast<int32_t>(rowStart2),
                        params.problemShape.k(), srcEpIdx, groupIdx, params);
                }
            }
            AscendC::SyncAll<true>();  // 等待所有 core 接收完成，后续 GEMM 可用

            // 更新下一轮 groupIdx 的基准偏移
            prevGroupSum1 += currentMSend;  // dstEpIdx 累计收到的 token 数
            prevGroupSum2 += currentRankM;  // RuntimeRank(params) 累计收到的 token 数

            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(syncgmm1Idx / CROSS_CORE_FLAG_MAX_SET_COUNT);
            syncgmm1Idx++;

            // Token 计数（用于 epilogue SwiGLU 输入范围）
            if (groupIdx + 1 <= params.epilogueGranularity) {
                if (dequantSum1 + currentRankM <= params.maxOutputSize) {
                    dequantSum1 += currentRankM;
                } else if (dequantSum1 < params.maxOutputSize) {
                    dequantSum1 = params.maxOutputSize;
                }
            }
            if (groupIdx + 1 > params.epilogueGranularity && dequantSum1 < params.maxOutputSize) {
                if (dequantSum1 + dequantSum2 + currentRankM <= params.maxOutputSize) {
                    dequantSum2 += currentRankM;
                } else if (dequantSum1 + dequantSum2 < params.maxOutputSize) {
                    dequantSum2 += params.maxOutputSize - dequantSum1 - dequantSum2;
                }
            }
        }

        uint32_t n2 = params.problemShape.k();

        typename BlockEpilogue2::Params epilogueParams{
            static_cast<int32_t>(params.EP),
            static_cast<int32_t>(params.expertPerRank),
            static_cast<int32_t>(RuntimeRank(params)),
            reinterpret_cast<__gm__ int32_t *>(shmem() + peermemInfo.offsetPeerTokenPerExpert),
            params.layoutD2,
            static_cast<int32_t>(n2),
            static_cast<int32_t>(L1TileShape::N),
            shmem,
            static_cast<int32_t>(peermemInfo.offsetD),
            static_cast<int32_t>(serverId_)
        };

        uint32_t n = params.problemShape.n();
        BlockEpilogue2 blockEpilogue2(resource, epilogueParams);
        BlockEpilogue1 blockEpilogue1(resource, n);

        // Synchronous wait: SwiGLU waits for GMM1 [1]
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
        AscendC::SyncAll<true>();
        if (dequantSum1 > 0) {
            uint32_t rowStartThisCore = 0;
            MatrixCoord offsetC{0U, 0};
            MatrixCoord shapeC{dequantSum1, params.problemShape.n()};
            LayoutC layoutC{dequantSum1, params.problemShape.n()};
            int64_t gmOffsetC = layoutC.GetOffset(offsetC);
            int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
            blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], params.epilogueCoreNum);
        }
        AscendC::SyncAll<true>();
        // Synchronization signal: SwiGLU notifies GMM2 [1]
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);

        if ((params.epilogueGranularity < params.expertPerRank && params.epilogueGranularity > 0)) {
            // Synchronous wait: SwiGLU waits for GMM1 [2]
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAGC2V);
            AscendC::SyncAll<true>();
            if (dequantSum2 > 0) {
                uint32_t rowStartThisCore = dequantSum1;
                MatrixCoord offsetC{rowStartThisCore, 0};
                uint32_t dequantLen = dequantSum2;
                MatrixCoord shapeC{dequantLen, params.problemShape.n()};
                LayoutC layoutC{dequantLen, params.problemShape.n()};
                int64_t gmOffsetC = layoutC.GetOffset(offsetC);
                int64_t gmOffsetD = params.layoutD1.GetOffset(offsetC);
                blockEpilogue1(gmC[gmOffsetC], shapeC, gmPerTokenScale1[rowStartThisCore], gmPermutedToken[gmOffsetD], gmPerTokenScale2[rowStartThisCore], coreNum);
            }
            AscendC::SyncAll<true>();
            // Synchronization signal: SwiGLU notifies GMM2 [2]
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAGV2C);
        }

        blockEpilogue1.Finalize();

        CombineSetFlag();

        CombineV2(params, blockEpilogue2);

        AscendC::SyncAll<true>();
#ifndef __CROSSRANKSYNCANDALLGATHERV1__
        ResetTokenPerExpert(params.EP * AlignUp(params.EP * params.expertPerRank, 128));
#endif
        shmem.InitStatusTargetSum();
        if (get_subblockid() == 0) {
            AscendC::LocalTensor<int32_t> ctrBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
            shmem.CrossRankSyncV2Set(ctrBuffer);
        } else {
            uint32_t uboffset = 0;
            uint32_t aicCoreNum = coreNum / 2;
            uint32_t aicCoreIdx = get_block_idx();
            uint32_t sendRankNum_ = params.EP / aicCoreNum;
            uint32_t remainderRankNum = params.EP % aicCoreNum;
            if (aicCoreIdx < remainderRankNum) {
                sendRankNum_++;
            }
            AscendC::LocalTensor<float> statusTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += sendRankNum_ * UB_ALIGN;
            AscendC::LocalTensor<float> gatherMaskOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += AlignUp(params.EP * sizeof(float), 32);
            AscendC::LocalTensor<uint32_t> gatherTmpTensor = resource.ubBuf.template GetBufferByByte<uint32_t>(uboffset);
            uboffset += AlignUp(sizeof(uint32_t), 32);
            AscendC::LocalTensor<float> statusSumOutTensor = resource.ubBuf.template GetBufferByByte<float>(uboffset);
            uboffset += AlignUp(sizeof(float), 32);
            shmem.CrossRankSyncV2Wait(statusTensor, gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);

            MoeTokenUnpermuteTilingData tilingData;
            MoeTokenUnpermuteTiling(params.problemShape.m() * params.topK, n2, params.topK, tilingData, coreNum / 2);
            KernelMoeTokenUnpermute<ElementD2, int32_t, float, true> kernelMoeTokenUnpermuteOp;
            kernelMoeTokenUnpermuteOp.Init(shmem() + peermemInfo.offsetD, workspaceInfo.expandedRowIdx, params.probs, reinterpret_cast<GM_ADDR>(params.ptrOutput), &tilingData);
            kernelMoeTokenUnpermuteOp.Process();
        }
    }

    CATLASS_DEVICE
    void CombineV2(Params const &params, BlockEpilogue2 & blockEpilogue)
    {
        BlockScheduler blockScheduler;
        int32_t syncLoopIdx = 0;
        uint32_t startCoreIdx = 0;
        uint32_t aicCoreNum = coreNum / 2;
        uint32_t aicCoreIdx = get_block_idx();
        uint32_t aivSubCoreIdx = get_subblockid();
        uint32_t preSrcExpertSum = 0;
        uint32_t n2 = params.problemShape.k();
        uint32_t k2 = params.problemShape.n() / 2;
#ifdef HCCL_COMM
        AscendC::LocalTensor<uint64_t> rdmaUbLocal = resource.ubBuf.template GetBufferByByte<uint64_t>(128 * 1024);
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead = resource.ubBuf.template GetBufferByByte<uint32_t>(128 * 1024 + UB_ALIGN);
        AscendC::GlobalTensor<ElementD2> gmLocalWindowsOut;
        gmLocalWindowsOut.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(
            shmem.windowsOutAddr() + peermemInfo.offsetD));
#endif
        icache_preload(8);
        for (uint32_t groupIdx = 0; groupIdx < params.expertPerRank; ++groupIdx) {
            uint32_t currentExpertM = cumsumMM(tokenPerExpertLayout(params.EP - 1, RuntimeRank(params), groupIdx));
            if (preSrcExpertSum >= params.maxOutputSize) {
                currentExpertM = 0;
            } else if (preSrcExpertSum + currentExpertM > params.maxOutputSize) {
                currentExpertM = params.maxOutputSize - preSrcExpertSum;
            }
            GemmCoord inGroupProblemShape{currentExpertM, n2, k2}; // M N K
            blockScheduler.Update(inGroupProblemShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            uint32_t coreLoops = blockScheduler.GetCoreLoops();
            uint32_t startLoopIdx = ((aicCoreIdx < startCoreIdx) ? (aicCoreIdx + aicCoreNum) : aicCoreIdx) - startCoreIdx;

            for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += aicCoreNum) {
                GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
                GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
                int32_t m0 = 16;
                //  Block count, the shape of each block is (m0, actualBlockShape.n())
                int32_t m_rows = (actualBlockShape.m() + m0 - 1) / m0;
                int32_t aiv_m_rows = m_rows / 2;
                if (aivSubCoreIdx == 1 && aiv_m_rows * 2 < m_rows) {
                    aiv_m_rows += 1;
                }
                uint32_t m_offset = blockCoord.m() * L1TileShape::M; // blockOffset
                if (aivSubCoreIdx == 1) {
                    m_offset += (m_rows / 2) * m0;
                }

                for (;syncLoopIdx <= groupIdx; syncLoopIdx ++) {
                    int32_t flag_id = syncLoopIdx / CROSS_CORE_FLAG_MAX_SET_COUNT;
                    AscendC::CrossCoreWaitFlag<0x2>(flag_id);
                }

                for (int32_t cur_row = 0; cur_row < aiv_m_rows; cur_row ++) {
                    GemmCoord realTileCoord{m_offset, blockCoord.n() * L1TileShape::N, 1};
                    uint32_t actualm = m0;
                    if (aivSubCoreIdx == 1 && cur_row == aiv_m_rows - 1) {
                        actualm = actualBlockShape.m() - (m_rows / 2) * m0 - cur_row * m0;
                    }
                    GemmCoord realTileShape{actualm, actualBlockShape.n(), 1};
                    blockEpilogue(gmC2, gmPerTokenScale2, realTileCoord, realTileShape, groupIdx,
                        preSrcExpertSum, preSumBeforeRankForCombine);
                    m_offset += m0;
                }
            }

#ifdef HCCL_COMM
            AscendC::SyncAll<true>();
            int32_t preSumRankInExpert = 0;
            for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; ++dstEpIdx) {
                int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, RuntimeRank(params), groupIdx));
                int32_t stRankInExpert = preSumRankInExpert;
                int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                preSumRankInExpert += lenRankInExpert;
                if (stRankInExpert >= static_cast<int32_t>(currentExpertM)) {
                    break;
                }
                if ((dstEpIdx % coreNum) != static_cast<int32_t>(coreIdx)) {
                    continue;
                }
                if ((dstEpIdx / SERVER_RANK_SIZE_A2) == static_cast<int32_t>(serverId_)) {
                    continue;
                }

                int32_t edData = min(edRankInExpert, static_cast<int32_t>(currentExpertM));
                uint32_t lenData = edData - stRankInExpert;
                if (lenData == 0) {
                    continue;
                }

                AscendC::GlobalTensor<ElementD2> gmRemotePeer;
                __gm__ void* dstPeermemPtr = shmem(peermemInfo.offsetD, dstEpIdx);
                gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(dstPeermemPtr));

                int32_t dstExpertOffset = preSumBeforeRankForCombine(dstEpIdx * params.expertPerRank + groupIdx);
                MatrixCoord srcOffset{preSrcExpertSum + static_cast<uint32_t>(stRankInExpert), 0};
                MatrixCoord dstOffset{static_cast<uint32_t>(dstExpertOffset), 0};
                int64_t gmSrcOffset = params.layoutD2.GetOffset(srcOffset);
                int64_t gmDstOffset = params.layoutD2.GetOffset(dstOffset);
                uint64_t messageLen = static_cast<uint64_t>(lenData) * static_cast<uint64_t>(n2) * sizeof(ElementD2);
                AIVRDMAPostSend(
                    (GM_ADDR)gmLocalWindowsOut[gmSrcOffset].GetPhyAddr(),
                    (GM_ADDR)gmRemotePeer[gmDstOffset].GetPhyAddr(),
                    static_cast<uint64_t>(dstEpIdx),
                    messageLen,
                    qp_info_,
                    rdmaUbLocal,
                    rdmaUbLocalHead);
            }
#endif
            preSrcExpertSum += currentExpertM;
            startCoreIdx = (startCoreIdx + coreLoops) % aicCoreNum;
        }
        blockEpilogue.Finalize();
    }

private:
    struct WorkspaceInfo {
        GM_ADDR ptrA;
        GM_ADDR ptrPerTokenScale;
        GM_ADDR ptrcumsumMM;
        GM_ADDR ptrC;
        GM_ADDR ptrC2;
        GM_ADDR ptrPermutedToken;
        GM_ADDR ptrPerTokenScale2;
        GM_ADDR expandedRowIdx;
        GM_ADDR ptrTokenPerExpert;
        GM_ADDR ptrSumBeforeRankForDispatch;
        GM_ADDR ptrSumBeforeRankForCombine;
        __gm__ float* ptrSoftFlagBase;

        CATLASS_DEVICE
        WorkspaceInfo(){}

        CATLASS_DEVICE
        WorkspaceInfo(const Params & params) {
            uint32_t k2 = params.problemShape.n() / 2;
            uint32_t n2 = params.problemShape.k();
            int64_t workspaceOffset = 0;
            expandedRowIdx = params.ptrWorkspace;

            workspaceOffset += AlignUp(params.problemShape.m(), 256) * params.topK * sizeof(int32_t);
            ptrcumsumMM = params.ptrWorkspace + workspaceOffset;

            workspaceOffset +=  AlignUp(params.EP * params.expertPerRank, 128) * params.EP * sizeof(int32_t);

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrPerTokenScale = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrPerTokenScale2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * sizeof(ElementPerTokenScale);
            ptrTokenPerExpert =  params.ptrWorkspace + workspaceOffset;

            workspaceOffset += (params.EP * params.EP * params.expertPerRank) * sizeof(int32_t);
            ptrC = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * params.problemShape.n() * sizeof(ElementC);
            ptrC2 = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * n2 * sizeof(ElementC);
            ptrA = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * params.problemShape.k() * sizeof(ElementA);
            ptrPermutedToken = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.maxOutputSize * k2 * sizeof(ElementA);
            ptrSumBeforeRankForDispatch = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.EP * sizeof(int32_t) * AlignUp(params.expertPerRank, FLAGSTRIDE);
            ptrSumBeforeRankForCombine = params.ptrWorkspace + workspaceOffset;

            workspaceOffset += params.EP * sizeof(int32_t) * AlignUp(params.expertPerRank, FLAGSTRIDE);

            ptrSoftFlagBase = reinterpret_cast<__gm__ float*>(params.ptrWorkspace + workspaceOffset);
        }
    };

    struct PeermemInfo {
        int64_t offsetA;
        int64_t offsetPeerPerTokenScale;
        int64_t offsetPeerTokenPerExpert;
        int64_t offsetD;
        int64_t offsetFlag;  // EP × expertPerRank × 4 bytes；每个 sender rank 占 expertPerRank 个 uint64_t slot，存放 epoch 到达标志

        CATLASS_DEVICE
        PeermemInfo(){}

        CATLASS_DEVICE
        PeermemInfo(const Params & params, const HcclShmem<true> & shmem)
        {
            offsetA = 0;    // Occupies one third of BUFFSIZE
            offsetPeerPerTokenScale = offsetA + AlignUp(shmem.SegmentSize() / 3, 512); // Occupies 1 MB
            offsetD = offsetPeerPerTokenScale + MB_SIZE;    // Occupies the remaining space
            offsetPeerTokenPerExpert = shmem.SegmentSize() - 2 * MB_SIZE;     // Occupies the final 2 MB
            // Flag 区域：每个 sender rank expertPerRank 个 uint32_t slot，区域大小 512 字节对齐
            offsetFlag = offsetPeerTokenPerExpert - static_cast<int64_t>(
                AlignUp(static_cast<int64_t>(params.EP * params.expertPerRank * sizeof(uint32_t)), 512));
        }
    };

    Arch::Resource<ArchTag> resource;

    uint32_t coreIdx;
    uint32_t coreNum;
    uint32_t serverId_ = 0;

    WorkspaceInfo workspaceInfo;
    PeermemInfo peermemInfo;

    AscendC::GlobalTensor<ElementA> gmA;
    AscendC::GlobalTensor<ElementC> gmC;

    AscendC::GlobalTensor<ElementD1> gmPermutedToken;
    AscendC::GlobalTensor<ElementC> gmC2;

    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1;
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2;

    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    AscendC::GlobalTensor<int32_t> cumsumMM;
    AscendC::GlobalTensor<int32_t> preSumBeforeRankForDispatch;
    AscendC::GlobalTensor<int32_t> preSumBeforeRankForCombine;
    Layout3D tokenPerExpertLayout;
    HcclShmem<true> shmem;

    __gm__ HcclAiRMAInfo* qp_info_ = nullptr;
};
} // namespace Catlass::Gemm::Kernel

#endif // MEGA_MOE_KERNEL_A2_HPP