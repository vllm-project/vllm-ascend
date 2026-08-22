/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_GROUP_MATMUL_EPILOGUE_H
#define MEGA_MOE_GROUP_MATMUL_EPILOGUE_H

#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "mega_moe_combine_send.h"

namespace MegaMoeImpl {
namespace Detail {

// Prototype: the per-tile compute body in AivGmm1PostGeneric. Runs SwiGLU/quant for one
// generic GMM1 tile without owning its upstream wait. The wrapper notifies GMM2 immediately
// after each dependency-free SwiGLU call to preserve the original sub-tile overlap.
template <typename MakeLayoutC, typename ElementC, bool TopkWeightsPrefetch, typename Config, typename LayoutC,
          typename SwigluQuantOp, typename ActualShape>
__aicore__ inline void ComputeInterleavedSwigluTileGeneric(
    const Config &config, GM_ADDR gmm1OutputAddr, const LayoutC &layoutC, SwigluQuantOp &swigluQuantOp,
    uint32_t expertBeforeCnt, const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc, uint16_t pingpongIdx)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if constexpr (TopkWeightsPrefetch) {
        constexpr uint32_t subTileM = L1_TILE_M_128;
        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                      reinterpret_cast<__gm__ ElementC *>(gmm1OutputAddr)),
                                  layoutC);
        auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(0), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
            uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
            uint32_t subMLoc = mLoc + subOffset;
            uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::DataCopy(swigluQuantOp.weightUb_,
                              swigluQuantOp.metaInfoGm_[metaInfoRow * INT32_PER_256B],
                              subM * INT32_PER_256B);
            auto tensorBlockGm = gmC.Slice(
                Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUb, tensorBlockGm);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            uint32_t epilogueN = Get<N_VALUE>(actualShape) / SWIGLU_N_HALF;
            uint32_t epilogueNLoc = nLoc / SWIGLU_N_HALF;
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, epilogueN, 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                subMLoc * config.outputN + epilogueNLoc,
                subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                static_cast<int64_t>(subMLoc / L1_TILE_M_256),
                0,
                static_cast<int64_t>(metaInfoRow),
                0};
            AscendC::SetCtrlSpr<60, 60>(0);
            swigluQuantOp(epilogueShape, epilogueOffset, 0U);
            swigluQuantOp.NotifyCompletedTile(subMLoc / L1_TILE_M_256);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    } else {
        uint32_t epilogueN = Get<N_VALUE>(actualShape) / SWIGLU_N_HALF;
        uint32_t epilogueNLoc = nLoc / SWIGLU_N_HALF;
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
            Get<M_VALUE>(actualShape), epilogueN, 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * config.outputN + epilogueNLoc,
            mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(epilogueNLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            static_cast<int64_t>(mLoc / L1_TILE_M_256),
            0,
            0,
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        swigluQuantOp(epilogueShape, epilogueOffset, pingpongIdx);
        swigluQuantOp.NotifyCompletedTile(mLoc / L1_TILE_M_256);
    }
}

template <typename MakeLayoutC, typename ElementC, bool TopkWeightsPrefetch, bool IsGmm1Interleaved,
          typename Config, typename LayoutC,
          typename SwigluQuantOp, typename ActualShape>
__aicore__ inline void ComputeSwigluTileGeneric(
    const Config &config, GM_ADDR gmm1OutputAddr, const LayoutC &layoutC, SwigluQuantOp &swigluQuantOp,
    uint32_t expertBeforeCnt, const ActualShape &actualShape, uint32_t mLoc, uint32_t nLoc, uint16_t pingpongIdx)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }
    if constexpr (IsGmm1Interleaved) {
        ComputeInterleavedSwigluTileGeneric<MakeLayoutC, ElementC, TopkWeightsPrefetch>(
            config, gmm1OutputAddr, layoutC, swigluQuantOp, expertBeforeCnt, actualShape, mLoc, nLoc,
            pingpongIdx);
        return;
    }

    if constexpr (TopkWeightsPrefetch) {
        constexpr uint32_t subTileM = L1_TILE_M_128;
        auto &swigluOp = swigluQuantOp;
        uint64_t firstMetaInfoRow = expertBeforeCnt + mLoc;
        AscendC::DataCopy(swigluOp.weightUb_, swigluOp.metaInfoGm_[firstMetaInfoRow * INT32_PER_256B],
                          static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);

        auto gmC = Te::MakeTensor(Te::MakeMemPtr<Te::Location::GM>(
                                      reinterpret_cast<__gm__ ElementC *>(gmm1OutputAddr)),
                                  layoutC);
        auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
            uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
            uint32_t subMLoc = mLoc + subOffset;
            uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (subOffset != 0) {
                AscendC::DataCopy(swigluOp.weightUb_, swigluOp.metaInfoGm_[metaInfoRow * INT32_PER_256B],
                                  subM * INT32_PER_256B);
            }

            auto tensorBlockGmFirst =
                gmC.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
            auto tensorBlockGmSecond = gmC.Slice(Te::MakeCoord(subMLoc, nLoc + config.outputN),
                                                 Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                subMLoc * config.outputN + nLoc,
                subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                static_cast<int64_t>(metaInfoRow),
                0};
            AscendC::SetCtrlSpr<60, 60>(0);
            swigluOp(epilogueShape, epilogueOffset);
            swigluOp.NotifyCompletedTile(subMLoc / L1_TILE_M_256);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    } else {
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                     Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * config.outputN + nLoc,
            mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            0,
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        swigluQuantOp(epilogueShape, epilogueOffset);
        swigluQuantOp.NotifyCompletedTile(mLoc / L1_TILE_M_256);
    }
}

// Prototype: the per-tile compute body in AivGmm1PostA8W4. Runs SwiGLU/quant for one
// A8W4 GMM1 tile without owning its upstream handshake. The wrapper publishes each
// completed sub-tile at the same point as the original epilogue implementation.
template <typename ElementC, typename MakeLayoutC, typename TensorC, typename SwigluQuantOp, typename Config,
          bool TopkWeightsPrefetch, typename ActualShape>
__aicore__ inline void ComputeSwigluTileA8W4(
    SwigluQuantOp &swigluQuantOp, TensorC &l0cOutGm, const Config &config, const ActualShape &actualShape,
    uint32_t mLoc, uint32_t nLoc, uint32_t expertBeforeCnt)
{
    uint32_t mLen = Get<M_VALUE>(actualShape);
    if (mLen == 0U) {
        return;
    }

    if constexpr (TopkWeightsPrefetch) {
        constexpr uint32_t subTileM = L1_TILE_M_128;
        uint64_t firstMetaInfoRow = expertBeforeCnt + mLoc;
        AscendC::DataCopy(swigluQuantOp.weightUb_, swigluQuantOp.metaInfoGm_[firstMetaInfoRow * INT32_PER_256B],
                          static_cast<uint32_t>(mLen < subTileM ? mLen : subTileM) * INT32_PER_256B);
        auto layoutL0cUb = MakeLayoutC{}(subTileM, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        int64_t ubOffsetSecond = static_cast<int64_t>(MAX_SINGLE_MN_ALIGN32_NUM_128) * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        for (uint32_t subOffset = 0; subOffset < mLen; subOffset += subTileM) {
            uint32_t subM = mLen - subOffset < subTileM ? mLen - subOffset : subTileM;
            uint32_t subMLoc = mLoc + subOffset;
            uint64_t metaInfoRow = expertBeforeCnt + subMLoc;
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            if (subOffset != 0) {
                AscendC::DataCopy(swigluQuantOp.weightUb_, swigluQuantOp.metaInfoGm_[metaInfoRow * INT32_PER_256B],
                                  subM * INT32_PER_256B);
            }
            auto tensorBlockGmFirst =
                l0cOutGm.Slice(Te::MakeCoord(subMLoc, nLoc), Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
            auto tensorBlockGmSecond = l0cOutGm.Slice(Te::MakeCoord(subMLoc, nLoc + config.outputN),
                                                      Te::MakeShape(subM, Get<N_VALUE>(actualShape)));
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
            AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{subM, Get<N_VALUE>(actualShape), 0, 0};
            Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                subMLoc * config.outputN + nLoc,
                subMLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                    CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
                0,
                0,
                static_cast<int64_t>(metaInfoRow),
                0};
            AscendC::SetCtrlSpr<60, 60>(0);
            swigluQuantOp(epilogueShape, epilogueOffset);
            swigluQuantOp.NotifyCompletedTile();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
        auto tensorBlockGmFirst = l0cOutGm.Slice(
            Te::MakeCoord(mLoc, nLoc), Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto tensorBlockGmSecond =
            l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc + config.outputN),
                           Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto layoutL0cUb = MakeLayoutC{}(config.tileM, L1_TILE_N);
        int64_t ubOffsetFirst = 0;
        uint32_t ubBufferSize =
            config.tileM == L1_TILE_M_128 ? MAX_SINGLE_MN_ALIGN32_NUM_128 : MAX_SINGLE_MN_ALIGN32_NUM_256;
        int64_t ubOffsetSecond = static_cast<int64_t>(ubBufferSize) * sizeof(ElementC);
        auto tensorBlockUbFirst =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetFirst), layoutL0cUb);
        auto tensorBlockUbSecond =
            Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffsetSecond), layoutL0cUb);
        auto copyGm2Ub = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbFirst, tensorBlockGmFirst);
        AscendC::Te::Copy(copyGm2Ub, tensorBlockUbSecond, tensorBlockGmSecond);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{Get<M_VALUE>(actualShape),
                                                                     Get<N_VALUE>(actualShape), 0, 0};
        Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
            mLoc * config.outputN + nLoc,
            mLoc * CeilDiv(config.outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,
            0,
            0,
            0,
            0};
        AscendC::SetCtrlSpr<60, 60>(0);
        swigluQuantOp(epilogueShape, epilogueOffset);
        swigluQuantOp.NotifyCompletedTile();
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(0);
    }
}

// Prototype: AivGmm2PostGeneric. Runs generic GMM2 combine post-processing on one AIV.
template <typename ElementC, typename WorkSet, typename ExtraArgs>
__aicore__ inline void AivGmm2PostGeneric(WorkSet &workSet, ExtraArgs &args, uint32_t startLoopIdx, uint32_t tileNum)
{
    constexpr uint32_t ubBufSize = MAX_SINGLE_MN_ALIGN32_NUM_128;
    int64_t ubOffsetFirst = 0;
    int64_t ubOffsetSecond = static_cast<int64_t>(ubBufSize) * sizeof(ElementC);
    LocalTensor<ElementC> l0cOutUbGMM2First(TPosition::VECIN, ubOffsetFirst, L1_TILE_M_128 * L1_TILE_N);
    LocalTensor<ElementC> l0cOutUbGMM2Second(TPosition::VECIN, ubOffsetSecond, L1_TILE_M_128 * L1_TILE_N);

    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += workSet.config.blockNum) {
        auto blockCoord = workSet.scheduler.GetBlockCoord(loopIdx);
        auto actualShape = workSet.scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        auto l0cOutUbGMM2 = args.pingpongIdx == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
        WaitForCube(args.pingpongIdx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::GlobalTensor<int32_t> metaInfoGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            workSet.params.workspaceInfo.metaInfoPtr + (args.groupCnt + mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * META_INFO_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
            nLoc, workSet.config.n, metaInfoTensor, l0cOutUbGMM2, actualShape, L1_TILE_N, workSet.params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        NotifyCube(args.pingpongIdx);
        args.pingpongIdx = 1 - args.pingpongIdx;
    }
}

// Prototype: AivGmm2PostA8W4. Consumes AIC GM tiles and runs combine on AIV1.
template <typename ElementC, typename MakeLayoutC, typename Scheduler, typename TensorC, typename Config>
__aicore__ inline void AivGmm2PostA8W4(Scheduler &scheduler, TensorC &l0cOutGm, const Params &params,
                                       const GMMAddrInfo &gmmAddrInfo, int32_t &gmTileSequence, uint32_t groupCnt,
                                       const Config &config, uint32_t startLoopIdx, uint32_t tileNum)
{
    for (uint32_t loopIdx = startLoopIdx; loopIdx < tileNum; loopIdx += config.blockNum) {
        auto blockCoord = scheduler.GetBlockCoord(loopIdx);
        auto actualShape = scheduler.GetBlockShape(blockCoord);
        uint32_t mLoc = Get<M_VALUE>(blockCoord);
        uint32_t nLoc = Get<N_VALUE>(blockCoord);

        int32_t expectedReadySequence = gmTileSequence + 1;
        while (AscendC::ReadGmByPassDCache(gmmAddrInfo.gmmToEpilogueFlag) < expectedReadySequence) {
            int64_t startCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - startCycle < 100) {
            }
        }
        auto tensorBlockGm = l0cOutGm.Slice(Te::MakeCoord(mLoc, nLoc),
                                            Te::MakeShape(Get<M_VALUE>(actualShape), Get<N_VALUE>(actualShape)));
        auto layoutL0cUB = MakeLayoutC{}(config.tileM, L1_TILE_N);
        int64_t ubOffset = 0;
        auto tensorBlockUb = Te::MakeTensor(Te::MakeMemPtr<Te::Location::UB, ElementC>(ubOffset), layoutL0cUB);
        LocalTensor<ElementC> l0cOutUbGMM2 =
            LocalTensor<ElementC>(TPosition::VECIN, ubOffset, config.tileM * L1_TILE_N);
        auto copyGM2UB = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGM2UB, tensorBlockUb, tensorBlockGm);

        AscendC::GlobalTensor<int32_t> metaInfoGm;
        int32_t lenTile = Get<M_VALUE>(actualShape);
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECCALC, META_INFO_TENSOR_ADDR, lenTile * META_INFO_SIZE);
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params.workspaceInfo.metaInfoPtr + (groupCnt + mLoc) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, lenTile * META_INFO_SIZE);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        MegaMoeCombineImpl::CombineTokens<ElementC, decltype(actualShape)>(
            nLoc, config.n, metaInfoTensor, l0cOutUbGMM2, actualShape, L1_TILE_N, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        gmTileSequence = expectedReadySequence;
    }
}

} // namespace Detail
} // namespace MegaMoeImpl

#endif // MEGA_MOE_GROUP_MATMUL_EPILOGUE_H
