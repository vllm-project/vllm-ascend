/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_LAYERED_H
#define MEGA_MOE_LAYERED_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#if __has_include("../../common/mc2_kernel_utils.h")
#include "../../common/mc2_kernel_utils.h"
#else
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#if __has_include("../../moe_distribute_dispatch_v2/quantize_functions.h")
#include "../../moe_distribute_dispatch_v2/quantize_functions.h"
#else
#include "../../../moe_distribute_dispatch_v2/op_kernel/quantize_functions.h"
#endif

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int64_t, int64_t, int64_t>;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeLayeredTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeLayeredTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, \
                                       TopkWeightsPrefetch

template <TemplateMegaMoeLayeredTypeClass>
class MegaMoeLayered {
public:
    template <int32_t QM>
    struct QuantTraits {
        using OutType = fp8_e4m3fn_t;
    };
    template <>
    struct QuantTraits<E5M2_QUANT> {
        using OutType = fp8_e5m2_t;
    };
    template <>
    struct QuantTraits<E2M1_QUANT> {
        using OutType = fp4x2_e2m1_t;
    };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType =
        typename std::conditional<Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    struct ExpertLoopState {
        TupleShape problemShape;
        BlockOffset baseOffset;
        // Rows before the current expert, kept per cursor for dispatch/GMM prefetch state split.
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoeLayered(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void UnpermuteBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void ResetGmm2CombineSyncCounters();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendCntCal(int32_t localExpertId, uint64_t &sendCnt);
    __aicore__ inline void MetaInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt = 0);
    __aicore__ inline bool UpdateSharedGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
                                          uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx,
                                              int32_t copyNum);
    __aicore__ inline void ResetContiguousGm(GM_ADDR dstAddr, uint64_t sizeBytes);
    __aicore__ inline void ResetDispatchState();
    __aicore__ inline void DispatchTokenToRmtServer(int32_t localExpertId);
    __aicore__ inline void DispatchComputeKernel(int32_t localExpertId, uint32_t realComputeCoreNum, int32_t roundTag);
    __aicore__ inline void DispatchComputeToken(GlobalTensor<int32_t> &topkIdsGlobal, int32_t localExpertId,
                                                uint32_t tokenIdx, uint32_t tokenStart, uint32_t dedupWordsPerServer);
    __aicore__ inline void DispatchSendKernel(uint32_t senderCoreNum, uint32_t remoteServerNum,
                                              uint32_t realComputeCoreNum, int32_t roundTag);
    __aicore__ inline void DispatchToken(uint32_t targetServerForSender, uint32_t peerRank, ChannelHandle channel,
                                         int32_t slot);
    __aicore__ inline int32_t UpdateDispatchTokenCnt(uint32_t targetServerForSender, int32_t cursor, int32_t endCount);
    __aicore__ inline bool DispatchSyncWithCompute(uint32_t realComputeCoreNum, int32_t roundTag);
    __aicore__ inline void LoadTokenFromLocalRelay(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                                   uint32_t copyInNum);
    __aicore__ inline void CopyTokensFromLocalRelay(int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum,
                                                    int64_t widthA, int64_t widthAScale, uint32_t copyInNum);
    __aicore__ inline void CopyTokensFromRemoteRelay(int32_t rowDstOffsetInCore, uint32_t relayRank, uint32_t srcServer,
                                                     int32_t copyNum, int64_t widthA, int64_t widthAScale);
    __aicore__ inline void QuantTokenToLocalRelay(uint32_t tokenIdx);
    __aicore__ inline uint64_t SendWorkspaceServerOffset(uint32_t targetServer);
    __aicore__ inline uint64_t RelayTokenOffset(uint32_t sourceServer, uint32_t tokenId);
    __aicore__ inline uint64_t RelayFlagOffset(uint32_t sourceServer, uint32_t tokenId);
    __aicore__ inline void SharedExpertCopyInput();
    __aicore__ inline void ProcessSharedExpertGmm1(const TupleShape &initShape, const BlockOffset &initOffset,
                                                   int32_t &gmTileSequence);
    __aicore__ inline void ProcessSharedExpertGmm2(const TupleShape &initShape, const BlockOffset &initOffset,
                                                   int32_t &gmTileSequence);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx);
    __aicore__ inline void LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor, int32_t curentOffset,
                                               int32_t index, TEventID event);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithSwigluQuant(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                      uint32_t expertIdx, int32_t &vecSetSyncCom,
                                                      int32_t &gmTileSequence);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                  uint32_t expertIdx, int32_t &vecSetSyncCom, int32_t &gmTileSequence);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline bool BuildCombineRankInfo(uint32_t expertIdx, uint32_t mExpert, uint32_t startRankId,
                                                uint32_t processRankNum, int64_t &offset,
                                                LocalTensor<int32_t> &rankInfoTensor, uint32_t &totalTokensToProcess,
                                                uint32_t &batchBaseOffset);
    __aicore__ inline void ProcessCombineGroups(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
                                                uint32_t expertIdx, uint32_t &startRankId, uint32_t &endRankId);
    __aicore__ inline void ProcessCombineBatch(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
                                               uint32_t expertIdx, uint32_t rankIndex,
                                               LocalTensor<int32_t> &rankInfoTensor, uint32_t processRankNum,
                                               uint32_t batchBaseOffset, int64_t &offset, uint32_t &totalProcessed);
    __aicore__ inline void DrainCombineChannels(uint32_t startRankId, uint32_t endRankId);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    __gm__ int32_t *gmmToEpilogueFlag_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    Params params_{};

    GlobalTensor<int32_t> swigluToGmm2FlagGm_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> metaInfoGlobalTensor_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;
    // A8W4 路径下 GroupMatmulSwigluQuant 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。cumsumInfoGlobalTensor_ 作为 cumsum 数据的 GM 持久备份：
    // SendCntCal 中 Load → 计算 → Store；MetaInfoCalAndDispatch/ExpertTokenNumCopyOut 从 GM 恢复。
    GlobalTensor<int32_t> cumsumInfoGlobalTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t rankPerServer_ = 0;
    uint32_t serverNum_ = 0;
    uint32_t serverId_ = 0;
    uint32_t rankIdInServer_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNumPerRank_ = 2;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();
    uint32_t mxQuantScaleNumAlignPerToken_ = 0;
    uint32_t mxQuantTokenAlignBytes_ = 0;
    uint32_t mxQuantScaleAlignBytes_ = 0;
    uint32_t mxQuantTokenScaleAlignBytes_ = 0;
    uint32_t weightAlignBytes_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;               // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint32_t roundSendTotalNum_ = 0;          // 分轮次：每轮处理的 token*topK 数量（256 对齐保证 GM 写 32B 对齐）
    uint32_t roundCompareCount_ = 0;          // 分轮次：每轮 CompareScalar 的元素数
    uint32_t roundMaskAlignSize_ = 0;         // 分轮次：每轮部分 mask 的字节大小（32B 对齐）
    uint32_t roundMaskSlotSize_ = 0;          // 分轮次：每轮部分 [mask|count] 槽位字节大小
    uint32_t totalRounds_ = 0;                // 分轮次：总轮数 = CeilDiv(sendTotalNum, roundSendTotalNum)
    uint32_t dispatchRoundSendTotalNum_ = 0;  // dispatch分轮次：每轮处理的 token*topK 数量（256 对齐）
    uint32_t dispatchTotalRounds_ = 0;        // dispatch分轮次：总轮数
    uint32_t dispatchRoundMaskAlignSize_ = 0; // dispatch分轮次：每轮mask的字节大小（32B对齐）
    uint64_t maskWinOffset_ = 0;              // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;             // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t dispatchWinOffset_ = 0;          // peermemInfo dispatchRecivePtr 相对 URMA win 基址的偏移
    uint64_t dispatchFlagWinOffset_ = 0;      // peermemInfo dispatchFlagPtr 相对 URMA win 基址的偏移
    uint32_t relayRecordBytes_ = 0;
    uint64_t sendWorkspaceMetaBytes_ = ALIGN_32;
    uint64_t sendWorkspaceServerBytes_ = 0;
    uint64_t dispatchL2ScratchBytes_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    uint32_t topKWeightsChunkLen_ = 0;
    uint32_t topKWeightsTempAddr_ = 0;
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    static constexpr uint32_t SEND_DEDUP_MASK_UB_BYTES = 8U * 1024U;
    static constexpr uint32_t SEND_DEDUP_MASK_BITS_PER_WORD = 32U;
    static constexpr uint32_t SEND_SCAN_WINDOW = 32U;
    static constexpr uint32_t MX_QUANT_TEMP_UB_BYTES = 2U * 1024U;
    static constexpr uint32_t SEND_MASK_UB_LIMIT = 248U * 1024U;
    static constexpr uint32_t MAX_CORENUM_USE_SEND = 8U;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM]; // 6-buffer 软流水：占满 EVENT_ID0..EVENT_ID5。
    // 完整 [bs] relay flag 快照，复用 copyTmp 起始地址。
    LocalTensor<uint64_t> relayFlagTensor_;
    LocalTensor<int32_t> relayReceivedTensor_; // 当前 chunk 每个 copyIdx 的接收状态。
    LocalTensor<int32_t> metaInfoTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    int32_t resetBatchElementCount_ = 0;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER]; // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;           // SendMaskCal GatherMask 计 count 的废弃输出 scratch
    LocalTensor<uint32_t> sendDedupMaskTensor_;
    LocalTensor<int32_t> expertTokenNumsOutTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，SwigluQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using SwigluQuantOutType = typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<SwigluQuantOutType>::ELEMENTS_PER_BYTE;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t GMM1_TILE_M = MegaMoeImpl::L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;

    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EPILOGUE_TILE_M, MegaMoeImpl::L1_TILE_N, TopkWeightsPrefetch>;
    using SharedBlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = m_ * topK_;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(MegaMoeImpl::L1_TILE_M_256)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    rankPerServer_ = mc2Context_->rankSizePerServer;
    if (rankPerServer_ == 0 || rankPerServer_ > worldSize_) {
        rankPerServer_ = worldSize_;
    }
    serverNum_ = Ops::Base::CeilDiv(worldSize_, rankPerServer_);
    serverId_ = rankId_ / rankPerServer_;
    rankIdInServer_ = rankId_ % rankPerServer_;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);
    params_.sharedBGmAddr = GetTensorAddr(0, sharedWeight1);
    params_.sharedB2GmAddr = GetTensorAddr(0, sharedWeight2);
    params_.sharedBScaleGmAddr = GetTensorAddr(0, sharedWeightScales1);
    params_.sharedB2ScaleGmAddr = GetTensorAddr(0, sharedWeightScales2);
    params_.combineCommParams.rankId = rankId_;
    params_.combineCommParams.hcomm = &hcomm_;
    params_.combineCommParams.mc2Context = mc2Context_;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData, serverNum_);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE, serverNum_);
    params_.tilingData = tilingData;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmToEpilogueFlag_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                             static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    }
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t *)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expertRevTokenNumsPtr);
    metaInfoGlobalTensor_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.metaInfoPtr);
    // 每个 block 负责一个专家，cumsumInfo 中每个专家占 worldSize 个
    // int32_t 存 rank 维度的 cumsum 结果，blockIdx 决定了负责哪个专家。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumInfoGlobalTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_));
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
                      params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
                      params_.workspaceInfo.metaInfoPtr, tilingData->clampLimit, static_cast<uint8_t>(ActMode::SWIGLU),
                      static_cast<uint8_t>(ActSubMode::DEFAULT), 1.0f, 1.0f});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    dispatchWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.dispatchRecivePtr - params_.peermemInfo.rankSyncInWorldPtr);
    dispatchFlagWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.dispatchFlagPtr - params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ =
        Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256)) /
        sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    if constexpr (TopkWeightsPrefetch) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
    relayRecordBytes_ =
        Ops::Base::CeilAlign(static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_), static_cast<uint64_t>(ALIGN_512));
    sendWorkspaceServerBytes_ = static_cast<uint64_t>(ALIGN_32) + static_cast<uint64_t>(m_) * sendWorkspaceMetaBytes_;
    uint64_t flagSnapshotBytes = static_cast<uint64_t>(m_) * sizeof(uint64_t);
    dispatchL2ScratchBytes_ = Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<uint64_t>(ALIGN_512));
}

// =================================================================================================
// DispatchBuffInit：SendCntCal & MetaInfoCalAndDispatch & ExpertTokenNumCopyOut 中使用的buffer申请
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    LocalTensor<uint8_t> hcommTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
    hcomm_.Init(hcommTensor_, ALIGN_512 / sizeof(uint8_t));
    uint32_t expertTokenCntTensorAddr = ALIGN_512;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // 分轮次参数计算：validTopkIndexTensor_ + topkIndexTensor_ + gatherMaskTensor_ 均为 round-sized
    // SendCntCal 改为直接从 GM 读 count，不再加载完整 mask slot 到 UB
    // round 张量: validTopkIndex(R*4) + topkIndex(R*4) + gatherMask(R/8) = 8R + R/8 = 65R/8
    // 预留 metaInfoTensor_: MegaMoeImpl::L1_TILE_M_256 * ALIGN_32
    uint32_t dispatchFixedCost = 0;
    {
        uint32_t tokenScaleSize = mxQuantTokenScaleAlignBytes_;
        uint32_t copyTmpBytes = static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * tokenScaleSize;
        uint32_t relayFlagBytes =
            Ops::Base::CeilAlign(static_cast<uint32_t>(m_ * sizeof(uint64_t)), static_cast<uint32_t>(ALIGN_32));
        uint32_t relayReceivedBytes = Ops::Base::CeilAlign(
            static_cast<uint32_t>(MegaMoeImpl::L1_TILE_M_256 * sizeof(int32_t)), static_cast<uint32_t>(ALIGN_32));
        uint32_t relayAndCopyTmpBytes =
            (relayFlagBytes > copyTmpBytes ? relayFlagBytes : copyTmpBytes) + relayReceivedBytes;
        dispatchFixedCost = relayAndCopyTmpBytes + SEND_DEDUP_MASK_UB_BYTES + MX_QUANT_TEMP_UB_BYTES +
                            mxQuantTokenScaleAlignBytes_ +
                            Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t) +
                            Ops::Base::CeilAlign(static_cast<int64_t>(moeExpertPerRank_ * sizeof(int32_t)),
                                                 static_cast<int64_t>(ALIGN_32)) +
                            static_cast<uint32_t>(MegaMoeImpl::L1_TILE_M_256) * static_cast<uint32_t>(ALIGN_32);
    }
    uint32_t fixedBefore = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t dispatchAvailUb = 0;
    if (SEND_MASK_UB_LIMIT > fixedBefore && SEND_MASK_UB_LIMIT - fixedBefore > dispatchFixedCost) {
        dispatchAvailUb = SEND_MASK_UB_LIMIT - fixedBefore - dispatchFixedCost;
    } else {
        dispatchAvailUb = 0;
    }
    uint32_t maxDispatchRoundSize = dispatchAvailUb * 8U / 65U;
    maxDispatchRoundSize = (maxDispatchRoundSize / 256U) * 256U;
    if (maxDispatchRoundSize == 0) {
        maxDispatchRoundSize = 256U;
    }
    if (static_cast<uint64_t>(sendTotalNum_) <= static_cast<uint64_t>(maxDispatchRoundSize)) {
        dispatchRoundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minDispatchRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(maxDispatchRoundSize)));
        uint32_t evenDispatchRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(minDispatchRounds)),
            static_cast<int64_t>(ALIGN_256)));
        dispatchRoundSendTotalNum_ =
            (evenDispatchRoundSize <= maxDispatchRoundSize) ? evenDispatchRoundSize : maxDispatchRoundSize;
    }
    if (dispatchRoundSendTotalNum_ == 0) {
        dispatchRoundSendTotalNum_ = 256U;
    }
    dispatchTotalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(dispatchRoundSendTotalNum_)));
    dispatchRoundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(dispatchRoundSendTotalNum_) / 8, static_cast<int64_t>(ALIGN_32)));

    uint32_t validTopkIndexTensorAddr = fixedBefore;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // gatherMaskTensor_ 改为分轮次：仅容纳单轮 mask 位，SendCntCal 直接从 GM 读 count
    uint32_t gatherMaskTensorAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t gatherMaskTensorSize = dispatchRoundMaskAlignSize_;
    gatherMaskTensor_ =
        LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ =
        LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr, gatherMaskTensorSize / sizeof(uint32_t));
    uint32_t tokenScaleSize = mxQuantTokenScaleAlignBytes_;
    uint32_t COPY_TMP_BUFFER_SIZE = tokenScaleSize;
    uint32_t copyTmpBaseAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(
            TPosition::VECCALC, copyTmpBaseAddr + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }
    uint32_t relayFlagTensorSize =
        Ops::Base::CeilAlign(static_cast<uint32_t>(m_ * sizeof(uint64_t)), static_cast<uint32_t>(ALIGN_32));
    relayFlagTensor_ =
        LocalTensor<uint64_t>(TPosition::VECCALC, copyTmpBaseAddr, relayFlagTensorSize / sizeof(uint64_t));
    uint32_t relayReceivedTensorAddr =
        copyTmpBaseAddr + (relayFlagTensorSize > copyTmpTotalSize ? relayFlagTensorSize : copyTmpTotalSize);
    uint32_t relayReceivedTensorSize = Ops::Base::CeilAlign(
        static_cast<uint32_t>(MegaMoeImpl::L1_TILE_M_256 * sizeof(int32_t)), static_cast<uint32_t>(ALIGN_32));
    relayReceivedTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, relayReceivedTensorAddr, relayReceivedTensorSize / sizeof(int32_t));
    // Tensor用处：Level1DispatchUrma 中按 (targetServer, localTokenIdx) 去重。
    // 本 server 位同时表示 canonical 量化数据和 relay ready flag 已发布到本地 win。
    // 该状态跨 expert 保持，必须放在 relay flag/state 临时区之后，避免完整 flag 快照覆盖。
    uint32_t sendDedupMaskTensorAddr = relayReceivedTensorAddr + relayReceivedTensorSize;
    sendDedupMaskTensor_ =
        LocalTensor<uint32_t>(TPosition::VECCALC, sendDedupMaskTensorAddr, SEND_DEDUP_MASK_UB_BYTES / sizeof(uint32_t));
    Duplicate<uint32_t>(sendDedupMaskTensor_, 0, SEND_DEDUP_MASK_UB_BYTES / sizeof(uint32_t));
    // Tensor用处：QuantTokenToLocalRelay 函数中用于量化计算中间区域。
    uint32_t urmaMxTempTensorAddr = sendDedupMaskTensorAddr + SEND_DEDUP_MASK_UB_BYTES;
    mxTempTensor_ =
        LocalTensor<uint16_t>(TPosition::VECCALC, urmaMxTempTensorAddr, MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));
    // Tensor用处：QuantTokenToLocalRelay 函数中用于存储量化输出。
    uint32_t urmaXOutTensorAddr = urmaMxTempTensorAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t urmaXOutTensorSize = mxQuantTokenScaleAlignBytes_;
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, urmaXOutTensorAddr,
                                               urmaXOutTensorSize / sizeof(ActivationType));
    // Tensor用处：QuantTokenToLocalRelay 函数中用于存储输入 token。
    uint32_t urmaXInTensorAddr = urmaXOutTensorAddr + urmaXOutTensorSize;
    uint32_t urmaXInTensorSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    xInTensor1_ =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, urmaXInTensorAddr, urmaXInTensorSize / sizeof(bfloat16_t));
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的tokenCnt数；
    // Tensor大小：moeExpertPerRank_ * sizeof(int32_t) 对齐至32字节；
    uint32_t expertTokenNumsOutTensorAddr = urmaXInTensorAddr + urmaXInTensorSize;
    uint32_t expertTokenNumsOutTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(moeExpertPerRank_ * sizeof(int32_t)),
                             static_cast<int64_t>(ALIGN_32));
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                     expertTokenNumsOutTensorSize / sizeof(int32_t));
    // 记录当前已被使用的ub地址，用于后续MetaInfoCalAndDispatch函数中分核后神申请metaInfoTensor_
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList localTensor申请
// --------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 248KB 限制，因此按 roundSendTotalNum_ 分轮次
//   分配 UB 张量（topkIds / sendMask / sendGatherOut），每轮仅加载一部分 topkIds，
//   在 UB 中计算部分 mask 后拼写到 workspace 对应偏移，最终合并为完整 mask slot。
// ======================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    LocalTensor<uint8_t> hcommTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
    hcomm_.Init(hcommTensor_, ALIGN_512);

    // 计算 resetTensor 大小（与原逻辑一致）
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(moeExpertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
         static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_)); // 64 * (16 + 256 + 16 * 28) = 46080
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        totalFlagInt32 += static_cast<uint64_t>(aicNum_) * INT_CACHELINE;
    }
    int64_t tokenGroupResetSize = static_cast<int64_t>(moeExpertPerRank_) * blockAivNum_ * INT_CACHELINE;
    totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize) ?
                         static_cast<int64_t>(totalFlagInt32) :
                         tokenGroupResetSize;
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetNumPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetNumPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);
    resetBatchElementCount_ = resetTensorSize / sizeof(int32_t);

    // 分轮次参数计算：UB 预算扣除固定开销后，求解最大 roundSendTotalNum_
    // round 张量: topkIds(R*4) + 2*maskSlot(CeilAlign(R/8,32)+32) + gatherOut(R*4)
    //            = 8R + R/4 + 64  (R 为 256 对齐时 CeilAlign(R/8,32)=R/8)
    // 解: R <= (available - 64) * 4 / 33，再 floor-align 到 256（保证 GM 写 32B 对齐）
    uint32_t fixedUbCosts = ALIGN_512 + resetTensorSize;
    uint32_t availableUb = SEND_MASK_UB_LIMIT - fixedUbCosts;
    uint32_t maxRoundSize = (availableUb - 64U) * 4U / 33U;
    if (maxRoundSize == 0) {
        maxRoundSize = 256U;
    }
    maxRoundSize = (maxRoundSize / 256U) * 256U;
    if (static_cast<uint64_t>(sendTotalNum_) <= static_cast<uint64_t>(maxRoundSize)) {
        roundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(maxRoundSize)));
        uint32_t evenRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(minRounds)),
            static_cast<int64_t>(ALIGN_256)));
        roundSendTotalNum_ = (evenRoundSize <= maxRoundSize) ? evenRoundSize : maxRoundSize;
    }
    if (roundSendTotalNum_ == 0) {
        roundSendTotalNum_ = 256U;
    }
    roundCompareCount_ = roundSendTotalNum_; // 16K
    roundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(roundCompareCount_) / 8, static_cast<int64_t>(ALIGN_32)));
    roundMaskSlotSize_ = roundMaskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    totalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(roundSendTotalNum_)));

    // Tensor用处：SendMaskCal 每轮搬运本卡 topkIds 的一个子段；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = roundCompareCount_ * sizeof(int32_t);
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    // Tensor用处：SendMaskCal 每轮存储部分 mask 位；DOUBLE_BUFFER 双缓冲；
    // Tensor大小：roundMaskSlotSize_（每轮部分 [mask|count] 槽位大小）；
    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(
            TPosition::VECCALC, sendMaskAddr + static_cast<uint32_t>(index) * roundMaskSlotSize_, roundMaskSlotSize_);
    }

    // Tensor用处：SendMaskCal 每轮 GatherMask 计 count 的废弃输出 scratch；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * roundMaskSlotSize_;
    uint32_t sendGatherOutSize = roundCompareCount_ * sizeof(int32_t);
    sendGatherOutTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
}

// ===============================================================================================
// ResetFlagList：清理本卡 workspace 中连续排布的 GMM/dispatch flag 和 AIC-AIV1 ready sequence。
// ===============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetFlagList()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // workSpace Flag 清零
    // 总数 = SwiGluToGmm2(moeExpertPerRank * INT_CACHELINE)
    //        + DispatchToGmm1(moeExpertPerRank * dispatchFlagSlotsPerExpert_)
    //        + SendCntCalToUpdParams(moeExpertPerRank * aicNum_ * INT_CACHELINE)
    //        + GmmToEpilogue(aicNum_ * INT_CACHELINE, specialized A8W4/A4W4 only)
    swigluToGmm2FlagGm_.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum =
        static_cast<int32_t>(moeExpertPerRank_) * (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_ +
                                                   static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        flagNum += static_cast<int32_t>(aicNum_) * static_cast<int32_t>(INT_CACHELINE);
    }
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm_[coreOffset], resetTensor_, rankSyncCopyParams);
    }
    // combine量化模式下TokenGroupCompleteFlag清零
    ResetGmm2CombineSyncCounters();
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<int32_t> statusGm;
        statusGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr));
        int32_t statusElementCount =
            (static_cast<int32_t>(moeExpertPerRank_) *
                 static_cast<int32_t>(params_.tilingData->maxTilesPerExpert) +
             1) *
            INT_CACHELINE;
        int32_t statusCoreLen, statusCoreOffset;
        TilingByCore(statusElementCount, statusCoreLen, statusCoreOffset, 1);
        for (int32_t resetElementOffset = 0; resetElementOffset < statusCoreLen;
             resetElementOffset += resetBatchElementCount_) {
            int32_t currentBatchElementCount = statusCoreLen - resetElementOffset < resetBatchElementCount_ ?
                                                   statusCoreLen - resetElementOffset :
                                                   resetBatchElementCount_;
            DataCopyExtParams statusCopyParams{
                1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPad(statusGm[statusCoreOffset + resetElementOffset],
                        resetTensor_, statusCopyParams);
        }
    }
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各路由专家收到的token总数输出（不包含共享专家）
// ==================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ExpertTokenNumCopyOut()
{
    // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 恢复
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                    {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                    {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < moeExpertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// ======================================================================================================
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡
// ------------------------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 限制，因此按 roundSendTotalNum_ 分轮次计算：
//   Phase 1: 对每个全局专家，逐轮加载 topkIds 子段 → CompareScalar → 部分 mask；
//   Phase 2: 每轮部分 mask 拼写到 workspace/win 对应字节偏移，GatherMask 累加本轮 count；
//   Phase 3: 所有轮次完成后，写入 total count，URMA 发送完整 mask slot。
// ======================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendMaskCal()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_);
    // 按卡处理，只使用前 worldSize 个核。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts);
         curRankId += blockAivNum_) {
        for (uint32_t expertIdIndex = 0; expertIdIndex < moeExpertPerRank_; ++expertIdIndex) { // 按专家处理
            int32_t curExpertId = static_cast<int32_t>(curRankId * moeExpertPerRank_ + expertIdIndex);
            uint64_t srcOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(curRankId)) *
                                 static_cast<uint64_t>(maskSlotSize_);
            uint64_t dstOffset =
                maskWinOffset_ + static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(rankId_)) *
                                     static_cast<uint64_t>(maskSlotSize_);
            uint64_t totalSendCnt = 0;

            for (uint32_t roundIdx = 0; roundIdx < totalRounds_; ++roundIdx) {
                uint64_t roundStart = static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(roundSendTotalNum_);
                uint64_t roundLen64 = (roundIdx + 1 < totalRounds_) ? static_cast<uint64_t>(roundSendTotalNum_) :
                                                                      sendTotalNum_ - roundStart;
                uint32_t roundLen = static_cast<uint32_t>(roundLen64);
                uint32_t curRoundCompareCount =
                    (roundLen == roundSendTotalNum_) ?
                        roundCompareCount_ :
                        static_cast<uint32_t>(
                            Ops::Base::CeilAlign(static_cast<int64_t>(roundLen) * static_cast<int64_t>(sizeof(int32_t)),
                                                 static_cast<int64_t>(ALIGN_256)) /
                            static_cast<int64_t>(sizeof(int32_t)));
                uint32_t curRoundMaskAlignSize =
                    (roundLen == roundSendTotalNum_) ?
                        roundMaskAlignSize_ :
                        static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(curRoundCompareCount) / 8,
                                                                   static_cast<int64_t>(ALIGN_32)));

                // Phase 1: 加载本轮 topkIds 子段（零填充尾部）
                Duplicate<int32_t>(topkIdsTensor_, 0, roundCompareCount_);
                SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
                GlobalTensor<int32_t> roundSrcGlobal;
                roundSrcGlobal.SetGlobalBuffer(
                    (__gm__ int32_t *)(params_.expertIdxGmAddr + static_cast<uint64_t>(roundStart) * sizeof(int32_t)));
                DataCopyExtParams roundLoadParams{1U, static_cast<uint32_t>(roundLen * sizeof(int32_t)), 0U, 0U, 0U};
                DataCopyPadExtParams<int32_t> roundLoadPad{false, 0U, 0U, 0U};
                DataCopyPad(topkIdsTensor_, roundSrcGlobal, roundLoadParams, roundLoadPad);
                SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

                // Phase 2: CompareScalar → 本轮部分 mask + GatherMask 累加 count
                LocalTensor<uint8_t> maskBuf = sendMaskTensor_[roundIdx % DOUBLE_BUFFER];
                LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
                CompareScalar(maskBuf, topkIdsTensor_, curExpertId, AscendC::CMPMODE::EQ, curRoundCompareCount);
                PipeBarrier<PIPE_V>();
                uint64_t roundSendCnt = 0;
                GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true, static_cast<uint32_t>(roundLen),
                           {1, 1, 0, 0}, roundSendCnt);
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
                totalSendCnt += roundSendCnt;
                SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

                // Phase 2b: 将本轮部分 mask 写入 workspace 或 local win 对应字节偏移
                // roundByteOffset = roundStart/8，因 roundSendTotalNum_ 为 256 对齐，该偏移 32B 对齐
                uint64_t roundByteOffset = static_cast<uint64_t>(roundStart) / 8U;
                uint32_t writeBytes =
                    (static_cast<uint64_t>(curRoundMaskAlignSize) <=
                     static_cast<uint64_t>(maskAlignSize_) - roundByteOffset) ?
                        curRoundMaskAlignSize :
                        static_cast<uint32_t>(static_cast<uint64_t>(maskAlignSize_) - roundByteOffset);
                DataCopyExtParams partialMaskCopyParams{1U, writeBytes, 0U, 0U, 0U};

                if (curRankId == rankId_) {
                    GlobalTensor<uint8_t> winDstGlobal;
                    winDstGlobal.SetGlobalBuffer(
                        (__gm__ uint8_t *)(GetRankWinAddrWithOffset(rankId_, dstOffset + roundByteOffset)));
                    DataCopyPad(winDstGlobal, maskBuf, partialMaskCopyParams);
                } else {
                    GlobalTensor<uint8_t> wsDstGlobal;
                    wsDstGlobal.SetGlobalBuffer(
                        (__gm__ uint8_t *)(params_.workspaceInfo.maskSlotPtr + srcOffset + roundByteOffset));
                    DataCopyPad(wsDstGlobal, maskBuf, partialMaskCopyParams);
                }
                PipeBarrier<PIPE_ALL>();
            }

            // Phase 3: 写入 total count 并发送完整 mask slot
            if (curRankId == rankId_) {
                __gm__ int32_t *winCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(GetRankWinAddrWithOffset(rankId_, dstOffset + maskAlignSize_));
                WriteGmByPassDCache(winCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
            } else {
                __gm__ int32_t *wsCountPtr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.maskSlotPtr + srcOffset + maskAlignSize_);
                WriteGmByPassDCache(wsCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
                GM_ADDR remoteDataAddr = GetRankWinAddrWithOffset(curRankId, dstOffset);
                GM_ADDR localGmAddr = params_.workspaceInfo.maskSlotPtr + srcOffset;
                hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, curRankId, rankId_), remoteDataAddr, localGmAddr,
                                maskSlotSize_);
            }
        }
    }
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts); curRankId += blockAivNum_) {
        if (curRankId != rankId_) {
            hcomm_.Drain(GetUrmaCommHandle(mc2Context_, curRankId, rankId_));
        }
    }
}

// ======================================================================
// LoadTopkWeightsToUb：权重搬运到UB（TopkWeightsPrefetch=0 时仅做 MTE2_V 同步）
// ======================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void
MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor,
                                                                    int32_t curentOffset, int32_t index, TEventID event)
{
    uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<TopkWeightsType> weightGm;
        weightGm.SetGlobalBuffer(
            (__gm__ TopkWeightsType *)(params_.probsGmAddr + static_cast<int64_t>(curentOffset + index) *
                                                                 topK_ * sizeof(TopkWeightsType)));
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            LocalTensor<TopkWeightsType> weightBf16Tmp = mxTempTensor_.ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightBf16Tmp, weightGm,
                        {1U, static_cast<uint32_t>(topK_ * sizeof(TopkWeightsType)), 0U, 0U, 0U}, {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            LocalTensor<float> weightFp32Ub = xOutTensor[weightOffsetInUb].template ReinterpretCast<float>();
            Cast(weightFp32Ub, weightBf16Tmp, AscendC::RoundMode::CAST_NONE, topK_);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<TopkWeightsType> weightUb =
                xOutTensor[weightOffsetInUb].template ReinterpretCast<TopkWeightsType>();
            DataCopyPad(weightUb, weightGm, {1U, static_cast<uint32_t>(topK_ * sizeof(TopkWeightsType)), 0U, 0U, 0U},
                        {false, 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        }
    } else {
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendWorkspaceServerOffset(
    uint32_t targetServer)
{
    return static_cast<uint64_t>(targetServer) * static_cast<uint64_t>(sendWorkspaceServerBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayTokenOffset(uint32_t sourceServer,
                                                                                            uint32_t tokenId)
{
    return (static_cast<uint64_t>(sourceServer) * static_cast<uint64_t>(m_) + tokenId) *
           static_cast<uint64_t>(relayRecordBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayFlagOffset(uint32_t sourceServer,
                                                                                           uint32_t tokenId)
{
    return (static_cast<uint64_t>(sourceServer) * static_cast<uint64_t>(m_) + tokenId) * sizeof(uint64_t);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetContiguousGm(GM_ADDR dstAddr,
                                                                                         uint64_t sizeBytes)
{
    if (sizeBytes == 0 || resetBatchElementCount_ == 0) {
        return;
    }
    GlobalTensor<int32_t> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dstAddr));

    uint64_t totalElements = sizeBytes / sizeof(int32_t);
    uint64_t elementsPerCore = Ops::Base::CeilDiv(totalElements, static_cast<uint64_t>(blockAivNum_));
    elementsPerCore = Ops::Base::CeilAlign(elementsPerCore, static_cast<uint64_t>(INT32_PER_256B));
    uint64_t coreOffset = static_cast<uint64_t>(aivCoreIdx_) * elementsPerCore;
    if (coreOffset >= totalElements) {
        return;
    }

    uint64_t coreElements = totalElements - coreOffset;
    coreElements = coreElements < elementsPerCore ? coreElements : elementsPerCore;
    for (uint64_t resetOffset = 0; resetOffset < coreElements; resetOffset += resetBatchElementCount_) {
        uint64_t remainingElements = coreElements - resetOffset;
        uint32_t currentElements =
            static_cast<uint32_t>(remainingElements < static_cast<uint64_t>(resetBatchElementCount_) ?
                                      remainingElements :
                                      static_cast<uint64_t>(resetBatchElementCount_));
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(currentElements * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(dstGm[coreOffset + resetOffset], resetTensor_, copyParams);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetDispatchState()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // count 与 meta state 原本是离散位置；tokenId 和 padding 可一并清零后，整个 L1 comm 区连续。
    ResetContiguousGm(params_.workspaceInfo.dispatchL1CommPtr,
                      static_cast<uint64_t>(serverNum_) * sendWorkspaceServerBytes_);
    // cursor、done 和 relay ready flag 各自在独立地址区内连续，不跨逻辑分配合并。
    ResetContiguousGm(params_.workspaceInfo.dispatchCursorPtr, static_cast<uint64_t>(serverNum_) * sizeof(int32_t));
    ResetContiguousGm(params_.workspaceInfo.dispatchDonePtr, static_cast<uint64_t>(blockNum_) * sizeof(int32_t));
    ResetContiguousGm(params_.peermemInfo.dispatchFlagPtr,
                      static_cast<uint64_t>(serverNum_) * static_cast<uint64_t>(m_) * sizeof(uint64_t));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantTokenToLocalRelay(uint32_t tokenIdx)
{
    LocalTensor<uint8_t> xOutBytesTensor = xOutTensor1_.template ReinterpretCast<uint8_t>();
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
        params_.aGmAddr + static_cast<uint64_t>(tokenIdx) * k_ * sizeof(bfloat16_t)));
    uint64_t relayOffset = RelayTokenOffset(serverId_, tokenIdx);
    GM_ADDR recordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + relayOffset;
    GlobalTensor<uint8_t> workspaceDstGlobal;
    workspaceDstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(recordAddr));

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyPad(xInTensor1_, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    LoadTopkWeightsToUb(xOutTensor1_, 0, tokenIdx, EVENT_ID0);

    __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xInTensor1_.GetPhyAddr());
    __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTempTensor_.GetPhyAddr());
    __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
        mxTempTensor_[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))]
            .GetPhyAddr());
    __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutTensor1_.GetPhyAddr());
    __ubuf__ uint16_t *mxScaleAddr =
        reinterpret_cast<__ubuf__ uint16_t *>(xOutTensor1_[mxQuantTokenAlignBytes_].GetPhyAddr());

    Quant::ComputeMaxExp(srcAddr, maxExpAddr, k_);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, k_);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, k_);
    }

    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID1>();
    DataCopyPad(workspaceDstGlobal, xOutBytesTensor, {1U, mxQuantTokenScaleAlignBytes_, 0U, 0U, 0U});
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
    __gm__ uint64_t *readyFlag =
        reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr + RelayFlagOffset(serverId_, tokenIdx));
    WriteGmByPassDCache(readyFlag, uint64_t(1));
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchComputeToken(
    GlobalTensor<int32_t> &topkIdsGlobal, int32_t localExpertId, uint32_t tokenIdx, uint32_t tokenStart,
    uint32_t dedupWordsPerServer)
{
    for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
        int32_t globalExpertId = topkIdsGlobal.GetValue(tokenIdx * topK_ + topkIdx);
        if ((globalExpertId % static_cast<int32_t>(moeExpertPerRank_)) != localExpertId) {
            continue;
        }
        uint32_t targetRank = static_cast<uint32_t>(globalExpertId) / moeExpertPerRank_;
        uint32_t targetServer = targetRank / rankPerServer_;
        uint32_t localTokenIdx = tokenIdx - tokenStart;
        uint32_t targetDedupWordIdx =
            targetServer * dedupWordsPerServer + localTokenIdx / SEND_DEDUP_MASK_BITS_PER_WORD;
        uint32_t dedupBit = 1U << (localTokenIdx & (SEND_DEDUP_MASK_BITS_PER_WORD - 1U));
        uint32_t targetDedupWord = sendDedupMaskTensor_.GetValue(targetDedupWordIdx);
        if ((targetDedupWord & dedupBit) != 0U) {
            continue;
        }

        uint32_t localDedupWordIdx = serverId_ * dedupWordsPerServer + localTokenIdx / SEND_DEDUP_MASK_BITS_PER_WORD;
        uint32_t localDedupWord = sendDedupMaskTensor_.GetValue(localDedupWordIdx);
        if ((localDedupWord & dedupBit) == 0U) {
            QuantTokenToLocalRelay(tokenIdx);
            sendDedupMaskTensor_.SetValue(localDedupWordIdx, localDedupWord | dedupBit);
        }
        if (targetServer == serverId_) {
            continue;
        }

        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchL1CommPtr +
                                                                      SendWorkspaceServerOffset(targetServer));
        int32_t slotIdx = AtomicAdd(countPtr, int32_t(1));
        uint64_t metaOffset = SendWorkspaceServerOffset(targetServer) + ALIGN_32 +
                              static_cast<uint64_t>(slotIdx) * sendWorkspaceMetaBytes_;
        __gm__ int32_t *metaPtr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchL1CommPtr + metaOffset);
        WriteGmByPassDCache(metaPtr, static_cast<int32_t>(tokenIdx));
        // PipeBarrier<PIPE_ALL>();
        WriteGmByPassDCache(metaPtr + 1, int32_t(1));
        sendDedupMaskTensor_.SetValue(targetDedupWordIdx, targetDedupWord | dedupBit);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchComputeKernel(
    int32_t localExpertId, uint32_t realComputeCoreNum, int32_t roundTag)
{
    uint32_t computeIdx = blockIdx_;
    uint32_t baseTokenNum = m_ / realComputeCoreNum;
    uint32_t tokenRemainder = m_ % realComputeCoreNum;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(computeIdx < tokenRemainder);
    uint32_t tokenStart = computeIdx * baseTokenNum + ((computeIdx < tokenRemainder) ? computeIdx : tokenRemainder);
    uint32_t tokenEnd = tokenStart + tokenNumInCore;
    uint32_t maxTokenNumPerComputeCore = Ops::Base::CeilDiv(m_, realComputeCoreNum);
    uint32_t dedupWordsPerServer = Ops::Base::CeilDiv(maxTokenNumPerComputeCore, SEND_DEDUP_MASK_BITS_PER_WORD);

    GlobalTensor<int32_t> topkIdsGlobal;
    topkIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.expertIdxGmAddr));
    for (uint32_t tokenIdx = tokenStart; tokenIdx < tokenEnd; ++tokenIdx) {
        DispatchComputeToken(topkIdsGlobal, localExpertId, tokenIdx, tokenStart, dedupWordsPerServer);
    }
    __gm__ int32_t *donePtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchDonePtr);
    WriteGmByPassDCache(donePtr + computeIdx, roundTag);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchToken(uint32_t targetServerForSender,
                                                                                     uint32_t peerRank,
                                                                                     ChannelHandle channel,
                                                                                     int32_t slot)
{
    uint64_t metaOffset = SendWorkspaceServerOffset(targetServerForSender) + ALIGN_32 +
                          static_cast<uint64_t>(slot) * sendWorkspaceMetaBytes_;
    __gm__ int32_t *srcMetaPtr =
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchL1CommPtr + metaOffset);
    if (ReadGmByPassDCache(srcMetaPtr + 1) != int32_t(1)) {
        return;
    }
    int32_t tokenIdx = ReadGmByPassDCache(srcMetaPtr);
    uint64_t relayOffset = RelayTokenOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR srcAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + relayOffset;
    GM_ADDR dstAddr = GetRankWinAddrWithOffset(peerRank, dispatchWinOffset_) + relayOffset;
    GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(peerRank, dispatchFlagWinOffset_) +
                             RelayFlagOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    hcomm_.WriteWithNotifyNbi(channel, dstAddr, srcAddr, mxQuantTokenScaleAlignBytes_, remoteFlagAddr, uint64_t(1));
    WriteGmByPassDCache(srcMetaPtr + 1, int32_t(2));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline int32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateDispatchTokenCnt(
    uint32_t targetServerForSender, int32_t cursor, int32_t endCount)
{
    while (cursor < endCount) {
        uint64_t metaOffset = SendWorkspaceServerOffset(targetServerForSender) + ALIGN_32 +
                              static_cast<uint64_t>(cursor) * sendWorkspaceMetaBytes_;
        __gm__ int32_t *srcMetaPtr =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchL1CommPtr + metaOffset);
        if (ReadGmByPassDCache(srcMetaPtr + 1) != int32_t(2)) {
            break;
        }
        ++cursor;
    }
    return cursor;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchSyncWithCompute(
    uint32_t realComputeCoreNum, int32_t roundTag)
{
    __gm__ int32_t *donePtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchDonePtr);
    for (uint32_t idx = 0; idx < realComputeCoreNum; ++idx) {
        if (ReadGmByPassDCache(donePtr + idx) < roundTag) {
            return false;
        }
    }
    return true;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchSendKernel(uint32_t senderCoreNum,
                                                                                          uint32_t remoteServerNum,
                                                                                          uint32_t realComputeCoreNum,
                                                                                          int32_t roundTag)
{
    __gm__ int32_t *cursorPtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchCursorPtr);
    uint32_t senderIdx = blockIdx_ - realComputeCoreNum;
    for (uint32_t remoteServerIdx = senderIdx; remoteServerIdx < remoteServerNum; remoteServerIdx += senderCoreNum) {
        uint32_t targetServerForSender = (remoteServerIdx < serverId_) ? remoteServerIdx : remoteServerIdx + 1U;
        uint32_t peerRank = targetServerForSender * rankPerServer_ + rankIdInServer_;
        ChannelHandle channel = GetUrmaCommHandle(mc2Context_, peerRank, rankId_);
        __gm__ int32_t *countPtr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.dispatchL1CommPtr +
                                                                      SendWorkspaceServerOffset(targetServerForSender));
        int32_t cursor = ReadGmByPassDCache(cursorPtr + targetServerForSender);
        PipeBarrier<PIPE_ALL>();
        int32_t scanCursor = cursor;
        while (true) {
            int32_t endCount = ReadGmByPassDCache(countPtr);
            if (scanCursor < cursor || scanCursor >= endCount) {
                scanCursor = cursor;
            }
            int32_t scanEnd = scanCursor + static_cast<int32_t>(SEND_SCAN_WINDOW);
            if (scanEnd > endCount) {
                scanEnd = endCount;
            }
            for (int32_t slot = scanCursor; slot < scanEnd; ++slot) {
                DispatchToken(targetServerForSender, peerRank, channel, slot);
            }
            scanCursor = scanEnd;
            if (scanCursor >= endCount) {
                scanCursor = cursor;
            }
            int32_t oldCursor = cursor;
            cursor = UpdateDispatchTokenCnt(targetServerForSender, cursor, endCount);
            if (cursor != oldCursor) {
                WriteGmByPassDCache(cursorPtr + targetServerForSender, cursor);
            }
            if (DispatchSyncWithCompute(realComputeCoreNum, roundTag) && cursor == ReadGmByPassDCache(countPtr)) {
                break;
            }
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchTokenToRmtServer(int32_t localExpertId)
{
    uint32_t remoteServerNum = serverNum_ - 1U;
    uint32_t senderCoreNum = (remoteServerNum < MAX_CORENUM_USE_SEND) ? remoteServerNum : MAX_CORENUM_USE_SEND;
    uint32_t realComputeCoreNum = blockNum_ - senderCoreNum;
    int32_t roundTag = localExpertId + 1;

    if (blockIdx_ < realComputeCoreNum) {
        DispatchComputeKernel(localExpertId, realComputeCoreNum, roundTag);
    } else {
        DispatchSendKernel(senderCoreNum, remoteServerNum, realComputeCoreNum, roundTag);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTokenFromLocalRelay(uint32_t srcServer,
                                                                                               uint32_t tokenIndex,
                                                                                               int32_t bufferIdx,
                                                                                               uint32_t copyInNum)
{
    __gm__ uint64_t *readyFlag = reinterpret_cast<__gm__ uint64_t *>(params_.peermemInfo.dispatchFlagPtr +
                                                                     RelayFlagOffset(srcServer, tokenIndex));
    while (ReadGmByPassDCache(readyFlag) != uint64_t(1)) {
    }

    uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, tokenIndex);
    GM_ADDR localRecordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) + remoteCopyOffset;
    GlobalTensor<ActivationType> relayGlobalTensor;
    relayGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(localRecordAddr));
    DataCopy(copyTmpTensors_[bufferIdx], relayGlobalTensor, copyInNum);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
}

// ==================================================================================================
// SendCntCal：目标专家卡读 count 计数，得到当前专家Id收到的token总数
// --------------------------------------------------------------------------------------------------
//   Phase 1: 逐 rank 从本卡 win 加载单个 [mask|count] 槽位，提取 count 并累加 cumsum；
//   Phase 2: 写 expertRevNumsGlobalTensor_ + AtomicAdd 通知 AIC;
// ==================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendCntCal(int32_t localExpertId,
                                                                                  uint64_t &sendCnt)
{
    sendCnt = 0;

    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 加载前序 expert 的 cumsum
            DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                        {1U, static_cast<uint32_t>(worldSize_ * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                        {true, 0U, 0U, 0U});
        }
    }

    // Phase 1: 逐 rank 直接从 GM 读 count，累计 cumsum（不再加载完整 mask slot 到 UB）
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        __gm__ int32_t *rankCountPtr = reinterpret_cast<__gm__ int32_t *>(
            params_.peermemInfo.maskRecvPtr + static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ +
            static_cast<uint64_t>(calRankId) * maskSlotSize_ + maskAlignSize_);
        int32_t perRankCnt = ReadGmByPassDCache(rankCountPtr);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(perRankCnt);
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_));
    }

    // Phase 2: 写到 gm 上，并通知 AIC
    expertTokenCntTensor_.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(expertRevNumsGlobalTensor_[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
                      expertTokenCntTensor_, INT32_PER_256B);
    if constexpr (ENABLE_A8W4) {
        // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，更新后写回 GM
        DataCopyPad(cumsumInfoGlobalTensor_, cumsumInfoTensor_,
                    {1U, static_cast<uint32_t>(worldSize_ * (localExpertId + 1) * sizeof(int32_t)), 0U, 0U, 0U});
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (TopkWeightsPrefetch) {
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }

    __gm__ int32_t *sendCntFlag = (__gm__ int32_t *)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                                  static_cast<uint64_t>(localExpertId) * aicNum_ * INT_CACHELINE +
                                  static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// ============================================================================
// CopyTokensFromLocalRelay：本卡中转使用 UB 多 buffer 搬运 token 与 scale
// ----------------------------------------------------------------------------
//   prime: 发出前 BufferNum 个 token 的 MTE2。
//   steady: 每轮执行 MTE3_out[i] + MTE2_in[i + BufferNum]，循环复用槽位。
//   drain: 收尾不再发新 MTE2，只等待 MTE3 完成。
// ============================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyTokensFromLocalRelay(
    int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum, int64_t widthA, int64_t widthAScale,
    uint32_t copyInNum)
{
    constexpr int32_t BufferNum = 5;
    constexpr TEventID kBufEvents[BufferNum] = {EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));

    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }

    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = metaInfoTensor_[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        TEventID eventId = kBufEvents[primeIdx];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        LoadTokenFromLocalRelay(srcServer, static_cast<uint32_t>(tokenIndex), primeIdx, copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        if constexpr (TopkWeightsPrefetch) {
            SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
        }
    }

    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % BufferNum;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        if constexpr (TopkWeightsPrefetch) {
            WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
            LocalTensor<ActivationType> weightBuf = copyTmpTensors_[outIdx];
            uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
            LocalTensor<int32_t> bufWeightsInt32 =
                weightBuf[weightOffsetInUb].template ReinterpretCast<int32_t>();
            int32_t topkIndex = metaInfoTensor_[copyIdx * INT32_PER_256B].GetValue(TOPK_INDEX);
            int32_t weightBits = bufWeightsInt32.GetValue(static_cast<uint32_t>(topkIndex));
            metaInfoTensor_[copyIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        }

        LocalTensor<ActivationType> tokenScaleBuf = copyTmpTensors_[outIdx];
        LocalTensor<QuantScaleOutType> scaleBuf =
            tokenScaleBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScaleBuf,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], scaleBuf,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        int32_t nextIdx = copyIdx + BufferNum;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = metaInfoTensor_[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            // 等待本轮 MTE3 完成后再复用 outIdx 槽。
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            LoadTokenFromLocalRelay(srcServer, static_cast<uint32_t>(tokenIndex), outIdx, copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
            if constexpr (TopkWeightsPrefetch) {
                SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
            }
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
}

// ============================================================================
// CopyTokensFromRemoteRelay：远端完整 flag slice 经本地 GM 搬入 UB 后轮询
// ============================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyTokensFromRemoteRelay(
    int32_t rowDstOffsetInCore, uint32_t relayRank, uint32_t srcServer, int32_t copyNum, int64_t widthA,
    int64_t widthAScale)
{
    if (copyNum <= 0) {
        return;
    }

    SyncFuncStatic<AscendC::HardEvent::S_V, SYNC_EVENT_ID4>();
    Duplicate<int32_t>(relayReceivedTensor_, 0, static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256));
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

    uint64_t flagSnapshotBytes = static_cast<uint64_t>(m_) * sizeof(uint64_t);
    GM_ADDR scratchAddr =
        params_.workspaceInfo.dispatchL2CommPtr + static_cast<uint64_t>(blockIdx_) * dispatchL2ScratchBytes_;
    GlobalTensor<uint64_t> scratchFlagGlobalTensor;
    scratchFlagGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(scratchAddr));
    GM_ADDR remoteFlagAddr =
        GetRankWinAddrWithOffset(relayRank, dispatchFlagWinOffset_) + RelayFlagOffset(srcServer, 0);
    ChannelHandle channel = GetUrmaCommHandle(mc2Context_, relayRank, rankId_);
    DataCopyExtParams flagCopyParams{1U, static_cast<uint32_t>(flagSnapshotBytes), 0U, 0U, 0U};
    DataCopyPadExtParams<uint64_t> flagCopyPadParams{true, 0U, 0U, 0U};

    int32_t receivedCount = 0;
    while (receivedCount < copyNum) {
        hcomm_.ReadNbi<true>(channel, scratchAddr, remoteFlagAddr, flagSnapshotBytes);
        hcomm_.Drain(channel);
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID1>();
        // flag 在 win/L2 中按 bs * sizeof(uint64_t) 紧凑存放；搬入 UB 时由 DataCopyPad 处理非 32B 尾块。
        DataCopyPad(relayFlagTensor_, scratchFlagGlobalTensor, flagCopyParams, flagCopyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID1>();

        for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
            if (relayReceivedTensor_.GetValue(copyIdx) != int32_t(0)) {
                continue;
            }
            int32_t tokenIndex = metaInfoTensor_[copyIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            if (relayFlagTensor_.GetValue(tokenIndex) != uint64_t(1)) {
                continue;
            }

            uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, static_cast<uint32_t>(tokenIndex));
            GM_ADDR remoteRecordAddr = GetRankWinAddrWithOffset(relayRank, dispatchWinOffset_) + remoteCopyOffset;
            GM_ADDR tokenDstAddr =
                params_.workspaceInfo.dispatchRevDataPtr +
                static_cast<uint64_t>(rowDstOffsetInCore + copyIdx) * widthA * sizeof(ActivationType);
            GM_ADDR scaleDstAddr =
                params_.workspaceInfo.dispatchRevScalePtr +
                static_cast<uint64_t>(rowDstOffsetInCore + copyIdx) * widthAScale * sizeof(QuantScaleOutType);
            hcomm_.ReadNbi<true>(channel, tokenDstAddr, remoteRecordAddr, widthA * sizeof(ActivationType));
            hcomm_.ReadNbi<true>(channel, scaleDstAddr, remoteRecordAddr + mxQuantTokenAlignBytes_,
                                 widthAScale * sizeof(QuantScaleOutType));
            if constexpr (TopkWeightsPrefetch) {
                GM_ADDR weightDstAddr = params_.workspaceInfo.dispatchRevWeightsPtr +
                                        static_cast<uint64_t>(rowDstOffsetInCore + copyIdx) * weightAlignBytes_;
                hcomm_.ReadNbi<true>(channel, weightDstAddr,
                                     remoteRecordAddr + mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_,
                                     weightAlignBytes_);
            }
            relayReceivedTensor_.SetValue(copyIdx, int32_t(1));
            ++receivedCount;
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID1>();
    }
    hcomm_.Drain(channel);
}

// ============================================================================
// CopyGMToGMPerToken：组装 token 三元组并按 relay 位置选择搬运路径
// ----------------------------------------------------------------------------
//   Phase 1: 所有 token 的三元组 (rank, tokenIndex, topkIndex) 组装写入 tripleTensor_。
//   Phase 2: 本卡中转使用 UB 多 buffer；远端中转直接 GM->GM 到 GMM1 输入 workspace。
//   Phase 3: triple 三元组搬出。
// ============================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyGMToGMPerToken(int32_t rowDstOffsetInCore,
                                                                                          int32_t remoteRankIdx,
                                                                                          int32_t copyStartIdx,
                                                                                          int32_t copyNum)
{
    if (copyNum <= 0) {
        return;
    }
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                          MXFP_MULTI_BASE_SIZE; // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_;
    uint32_t srcServer = static_cast<uint32_t>(remoteRankIdx) / rankPerServer_;
    uint32_t srcRankInServer = static_cast<uint32_t>(remoteRankIdx) % rankPerServer_;
    uint32_t relayRank = serverId_ * rankPerServer_ + srcRankInServer;

    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / topK_;
        metaInfoTensor_[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        metaInfoTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        metaInfoTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    }

    if (relayRank == rankId_) {
        CopyTokensFromLocalRelay(rowDstOffsetInCore, srcServer, copyNum, widthA, widthAScale, copyInNum);
    } else {
        CopyTokensFromRemoteRelay(rowDstOffsetInCore, relayRank, srcServer, copyNum, widthA, widthAScale);
        if constexpr (TopkWeightsPrefetch) {
            for (int32_t i = 0; i < copyNum; ++i) {
                int32_t topkIndex = metaInfoTensor_[i * INT32_PER_256B].GetValue(TOPK_INDEX);
                __gm__ int32_t *weightGmI32 = reinterpret_cast<__gm__ int32_t *>(
                    params_.workspaceInfo.dispatchRevWeightsPtr +
                    static_cast<uint64_t>(rowDstOffsetInCore + i) * weightAlignBytes_);
                int32_t weightBits = ReadGmByPassDCache(weightGmI32 + static_cast<uint32_t>(topkIndex));
                metaInfoTensor_[i * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
            }
        }
    }

    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(metaInfoGlobalTensor_[rowDstOffsetInCore * INT32_PER_256B], metaInfoTensor_, copyNum * INT32_PER_256B);
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
}

// ====================================================================================================
// MetaInfoCalAndDispatch：专家接收token的三元组信息计算搬出 & token dispatch & 写Flag位
// ----------------------------------------------------------------------------------------------------
//   逐 rank 从 win 加载单个 [mask|count] 槽位到 gatherMaskTensor_，
//   再按 dispatchTotalRounds_ 分轮次做 GatherMask → CopyGMToGMPerToken；
//   Phase 2: dispatch->gmm1 flag 位 AtomicAdd，每个 expert 有 maxWavesPerExpert_ 个槽位；
// ====================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::MetaInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo,
                                                                                              int32_t localExpertId)
{
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 : cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);

    // A8W4 + prefetch 路径下 SwigluQuant 覆盖 V1 UB，topkIndexTensor_ 需重新初始化
    if constexpr (ENABLE_A8W4 && TopkWeightsPrefetch) {
        if (localExpertId != 0) {
            uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
                                                                static_cast<int64_t>(ALIGN_32));
            CreateVecIndex(topkIndexTensor_, 0, topkIndexTensorSize / sizeof(int32_t));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    constexpr int32_t MAX_META_INFO_ROWS_PER_CHUNK = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    for (uint32_t srcRankInServer = blockIdx_; srcRankInServer < rankPerServer_; srcRankInServer += blockNum_) {
        for (uint32_t srcServer = 0; srcServer < serverNum_; ++srcServer) {
            uint32_t dstRankIdx = srcServer * rankPerServer_ + srcRankInServer;
            if (dstRankIdx >= worldSize_) {
                continue;
            }
            int32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ?
                                            0 :
                                            cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));
            if (rowStartIdxInDst >= maxOutputSize_) {
                continue;
            }

            __gm__ uint8_t *rankMaskBasePtr = reinterpret_cast<__gm__ uint8_t *>(
                params_.peermemInfo.maskRecvPtr + static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ +
                static_cast<uint64_t>(dstRankIdx) * maskSlotSize_);

            __gm__ int32_t *rankCountPtr = reinterpret_cast<__gm__ int32_t *>(rankMaskBasePtr + maskAlignSize_);
            int32_t rankTotalCount = AscendC::ReadGmByPassDCache(rankCountPtr);
            if (rankTotalCount <= 0) {
                continue;
            }
            int32_t accumulatedRowSrcOffset = 0;

            for (uint32_t roundIdx = 0; roundIdx < dispatchTotalRounds_; ++roundIdx) {
                uint64_t roundStart =
                    static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(dispatchRoundSendTotalNum_);
                uint32_t roundLen = (roundIdx + 1 < dispatchTotalRounds_) ?
                                        dispatchRoundSendTotalNum_ :
                                        static_cast<uint32_t>(sendTotalNum_ - roundStart);
                uint32_t topkIndexTensorElemCount =
                    Ops::Base::CeilAlign(static_cast<int64_t>(roundLen * sizeof(int32_t)),
                                         static_cast<int64_t>(ALIGN_32)) /
                    sizeof(int32_t);

                // 逐轮加载本轮 mask 位（从完整 mask 的 roundStart/8 偏移处读取）
                uint32_t curRoundMaskBytes =
                    (roundLen == dispatchRoundSendTotalNum_) ?
                        dispatchRoundMaskAlignSize_ :
                        static_cast<uint32_t>(Ops::Base::CeilAlign(
                            static_cast<int64_t>(Ops::Base::CeilDiv(roundLen, 8U)), static_cast<int64_t>(ALIGN_32)));
                uint64_t roundMaskByteOffset = static_cast<uint64_t>(roundStart) / 8U;
                GlobalTensor<uint8_t> roundMaskSrc;
                roundMaskSrc.SetGlobalBuffer(rankMaskBasePtr + roundMaskByteOffset);
                SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
                DataCopy(gatherMaskTensor_, roundMaskSrc, curRoundMaskBytes);
                SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

                CreateVecIndex(topkIndexTensor_, static_cast<int32_t>(roundStart), topkIndexTensorElemCount);
                AscendC::PipeBarrier<PIPE_V>();

                uint64_t roundTokenCnt = 0;
                LocalTensor<uint32_t> rankMaskSlice = gatherMaskInt32Tensor_[0];
                GatherMask(validTopkIndexTensor_, topkIndexTensor_, rankMaskSlice, true, roundLen, {1, 1, 0, 0},
                           roundTokenCnt);
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                if (roundTokenCnt == 0) {
                    continue;
                }

                int32_t roundCopyCnt = static_cast<int32_t>(roundTokenCnt);
                if (rowStartIdxInDst + accumulatedRowSrcOffset + roundCopyCnt > maxOutputSize_) {
                    roundCopyCnt = maxOutputSize_ - rowStartIdxInDst - accumulatedRowSrcOffset;
                }
                if (roundCopyCnt <= 0) {
                    accumulatedRowSrcOffset += static_cast<int32_t>(roundTokenCnt);
                    continue;
                }

                for (int32_t chunkSrcIdx = 0; chunkSrcIdx < roundCopyCnt; chunkSrcIdx += MAX_META_INFO_ROWS_PER_CHUNK) {
                    int32_t chunkRows = roundCopyCnt - chunkSrcIdx;
                    if (chunkRows > MAX_META_INFO_ROWS_PER_CHUNK) {
                        chunkRows = MAX_META_INFO_ROWS_PER_CHUNK;
                    }
                    int32_t rowDstOffsetInCore = rowStartIdxInDst + accumulatedRowSrcOffset + chunkSrcIdx;
                    uint32_t metaInfoTensorAddr = ubBufferUsedAddr_;
                    uint32_t metaInfoTensorSize = chunkRows * ALIGN_32;
                    metaInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr,
                                                           metaInfoTensorSize / sizeof(int32_t));
                    CopyGMToGMPerToken(rowDstOffsetInCore, dstRankIdx, chunkSrcIdx, chunkRows);
                }
                int32_t roundRowStartLocal = rowStartIdxInDst + accumulatedRowSrcOffset - priorExpertCumsum;
                int32_t roundRowEndLocal = roundRowStartLocal + roundCopyCnt;
                int32_t waveLo = roundRowStartLocal / L1_TILE_M_I32;
                int32_t waveHi = (roundRowEndLocal - 1) / L1_TILE_M_I32;
                __gm__ int32_t *flagBase = gmmAddrInfo.dispatchToGmm1Flag;
                for (int32_t w = waveLo; w <= waveHi; ++w) {
                    int32_t waveStartLocal = w * L1_TILE_M_I32;
                    int32_t waveEndLocal = waveStartLocal + L1_TILE_M_I32;
                    int32_t lo = roundRowStartLocal > waveStartLocal ? roundRowStartLocal : waveStartLocal;
                    int32_t hi = roundRowEndLocal < waveEndLocal ? roundRowEndLocal : waveEndLocal;
                    AtomicAdd(flagBase + static_cast<int64_t>(w) * INT_CACHELINE, int32_t(hi - lo));
                }
                accumulatedRowSrcOffset += static_cast<int32_t>(roundTokenCnt);
            }
        }
    }
}

// =====================================================================================================
// UpdateGroupParams：更新当前expertIdx的problemShape，偏移掉本卡前侧专家收到的cnt数
// ----------------------------------------------------------------------------------------------------
//   Phase 1: 根据problemShape中的M(前一个专家收到的count数)，偏移计算baseOffset中gmm1与gmm2的左右矩阵偏移；
//   Phase 2: 更新当前专家id收到的count数;
// =====================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGroupParams(ExpertLoopState &state,
                                                                                         uint32_t expertIdx,
                                                                                         uint64_t sendCnt)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    // gmm1中当前专家收到的count数是由subBlockIdx_=1的aiv计算出并写入expertRevNumsGlobalTensor_，通知后续aic/aiv0读取该值
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        if (subBlockIdx_ == 0) { // aiv1进行SendCntCal计算完成后atomicAddFlag，aic/aiv0等到该flag位后读取cnt值
            __gm__ int32_t *sendCntFlag = (__gm__ int32_t *)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                                          static_cast<uint64_t>(expertIdx) * aicNum_ * INT_CACHELINE +
                                          static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
            while (AscendC::ReadGmByPassDCache(sendCntFlag) == 0) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {
                }
            }

            uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
            DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
                expertRevNumsGlobalTensor_[offsetInCnt]);
            Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
        } else {
            Get<M_VALUE>(state.problemShape) = sendCnt;
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            expertRevNumsGlobalTensor_[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = expertRevNumsGlobalTensor_.GetValue(offsetInCnt);
    }

    if (Get<M_VALUE>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

// =====================================================================================================
// UpdateSharedGroupParams：共享专家专用，M 恒为 m_，无 flag 等待与 DCache 操作。
// =====================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGroupParams(ExpertLoopState &state,
                                                                                               uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF / C_ELEMS_PER_BYTE;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF / B_ELEMS_PER_BYTE;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    Get<M_VALUE>(state.problemShape) = m_;
    return true;
}

// ==================================================================================
// UpdateGlobalBuffer：更新当前 expert 的 GMM 地址视图。
//                     GMM1 始终写 gmm1MmadResPtr；
//                     GMM2 始终写 gmm2MmadResPtr。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                          const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.dispatchRevDataPtr + Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
                                   Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        if constexpr (g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset),
                Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset),
                0L,
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        uint64_t expertSyncSlotOffset = static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) *
                                        params_.tilingData->combineSyncSlotCountPerExpert;
        gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr +
                                             expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
    }
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                   Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t *)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享专家专用，地址来自 shared* workspace，flags 为 nullptr。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGlobalBuffer(
    GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertInputDataPtr;
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertInputScalePtr;
        if constexpr (ENABLE_A8W4) {
            gmmAddrInfo.gmm1OutGlobal = params_.workspaceInfo.sharedExpertGmm1OutPtr +
                                        Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.bGlobal = params_.sharedBGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.sharedBScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.sharedExpertResultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertSwigluDataPtr +
                              Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertSwigluScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.sharedB2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.sharedB2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
}

// =============================================
// ResetGmm2CombineSyncCounters：重置 GMM2→Combine 同步计数器
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetGmm2CombineSyncCounters()
{
    if constexpr (g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(params_.tilingData->combineSyncSlotCountPerExpert *
                                                     moeExpertPerRank_ * static_cast<uint64_t>(INT_CACHELINE));
        int32_t coreLen, coreOffset;
        TilingByCore(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
        if (coreLen > 0) {
            Duplicate(resetTensor_, 0, coreLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
            DataCopy(gmm2CombineSyncCounterGm[coreOffset], resetTensor_, coreLen);
        }
    }
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitCombineBuffers()
{
    if constexpr (g_coreType == AIV) {
        LocalTensor<uint8_t> hcommTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
        hcomm_.Init(hcommTensor_, ALIGN_512);
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    uint32_t coreIdForGrouping = aivCoreIdx_;
    uint32_t totalCoresForGrouping = curUseAivNum;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        if (subBlockIdx_ != 1) {
            startTokenId = 0;
            endTokenId = 0;
            sendTokenNum = 0;
            return;
        }
        coreIdForGrouping = aivCoreIdx_ / 2;
        totalCoresForGrouping = curUseAivNum / 2;
    }
    sendTokenNum = curSendCnt / totalCoresForGrouping;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % totalCoresForGrouping; // 余数
    uint32_t newAivId = coreIdForGrouping;
    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

// =============================================
// BuildCombineRankInfo：Phase 1 — 从本地 URMA mask slot 读取各源 rank 的
//                      token count，构建 rank 信息表到 UB。
//                      返回 false 表示无 token 需处理。
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildCombineRankInfo(
    uint32_t expertIdx, uint32_t mExpert, uint32_t startRankId, uint32_t processRankNum, int64_t &offset,
    LocalTensor<int32_t> &rankInfoTensor, uint32_t &totalTokensToProcess, uint32_t &batchBaseOffset)
{
    // 定位 mask slot 基址：每个 expert 有 worldSize_ 个 slot
    uint64_t maskSlotOffset = maskWinOffset_ + static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_;
    __gm__ uint8_t *maskSlotBase = (__gm__ uint8_t *)(GetRankWinAddrWithOffset(rankId_, maskSlotOffset));

    // cumsum 本核范围之前的所有 rank
    uint32_t cumsumBeforeRange = 0;
    for (uint32_t index = 0; index < startRankId; index++) {
        __gm__ int32_t *cntAddr =
            reinterpret_cast<__gm__ int32_t *>(maskSlotBase + index * maskSlotSize_ + maskAlignSize_);
        cumsumBeforeRange += AscendC::ReadGmByPassDCache(cntAddr);
    }

    // 分配 rankInfoTensor : layout [startRow_0..N | tokenCnt_0..N | processedCnt_0..N]
    uint32_t rankInfoSize = Ops::Base::CeilAlign(static_cast<int64_t>(processRankNum * 3 * sizeof(int32_t)),
                                                 static_cast<int64_t>(ALIGN_32));
    rankInfoTensor = LocalTensor<int32_t>(TPosition::VECCALC, offset, rankInfoSize / sizeof(int32_t));
    offset += rankInfoSize;

    // 填充每个 rank 的初始状态
    uint32_t cumsumInRange = 0;
    totalTokensToProcess = 0;
    for (uint32_t i = 0; i < processRankNum; i++) {
        uint32_t r = startRankId + i;
        __gm__ int32_t *cntAddr = reinterpret_cast<__gm__ int32_t *>(maskSlotBase + r * maskSlotSize_ + maskAlignSize_);
        int32_t cnt = AscendC::ReadGmByPassDCache(cntAddr);
        rankInfoTensor.SetValue(i, cumsumBeforeRange + cumsumInRange); // startRow
        rankInfoTensor.SetValue(processRankNum + i, cnt);              // tokenCnt
        rankInfoTensor.SetValue(2 * processRankNum + i, 0);            // processedCnt
        cumsumInRange += cnt;
        totalTokensToProcess += cnt;
    }

    if (totalTokensToProcess == 0) {
        return false;
    }
    batchBaseOffset = static_cast<uint32_t>(offset);
    return true;
}

// =============================================
// ProcessCombineBatch：处理单个 rank 在单个 row-group 内的一批 token。
//                     加载 triple → DataCopy → CombineTokenGroup → 更新计数器。
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineBatch(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx, uint32_t rankIndex,
    LocalTensor<int32_t> &rankInfoTensor, uint32_t processRankNum, uint32_t batchBaseOffset, int64_t &offset,
    uint32_t &totalProcessed)
{
    constexpr bool IsQuant = CombineQuantMode != COMBINE_NO_QUANT;
    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));

    // 从 rankInfoTensor 读取当前 rank 的进度信息
    uint32_t startRow = rankInfoTensor.GetValue(rankIndex);
    uint32_t tokenCnt = rankInfoTensor.GetValue(processRankNum + rankIndex);
    uint32_t processedCnt = rankInfoTensor.GetValue(2 * processRankNum + rankIndex);
    uint32_t currentRow = startRow + processedCnt;
    uint32_t targetGroup = currentRow / L1_TILE_M_256;

    // 计算本批 token 数（不超过 group 边界和 rank 剩余量）
    uint32_t groupEndRow = (targetGroup + 1) * L1_TILE_M_256;
    if (groupEndRow > mExpert) {
        groupEndRow = mExpert;
    }
    uint32_t remainingTokens = tokenCnt - processedCnt;
    uint32_t batchCount = groupEndRow - currentRow;
    if (batchCount > remainingTokens) {
        batchCount = remainingTokens;
    }

    // 加载 triple info 到 UB（复用 VECIN 空间，每次 batch 覆盖）
    offset = batchBaseOffset;
    LocalTensor<int32_t> metaInfoTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, batchCount * META_INFO_SIZE);
    offset += batchCount * META_INFO_SIZE * sizeof(int32_t);

    AscendC::GlobalTensor<int32_t> tripleGm;
    tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.metaInfoPtr +
                                                                (gmm2State.expertBeforeCnt + currentRow) *
                                                                    META_INFO_SIZE * sizeof(int32_t)));
    SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID1>();
    AscendC::DataCopy(metaInfoTensor, tripleGm, batchCount * META_INFO_SIZE);

    // 执行 combine 并发送
    MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t, true, IsQuant>(
        currentRow, batchCount, k_, expertIdx, rankId_, gmmAddrInfo.gmm2OutGlobal, params_, metaInfoTensor,
        combineUbTensorSize_, offset, quantTokenSizeBytes);

    // 更新进度
    rankInfoTensor.SetValue(2 * processRankNum + rankIndex, processedCnt + batchCount);
    totalProcessed += batchCount;
}

// =============================================
// ProcessCombineGroups：Phase 1 + Phase 2 — 构建 rank 信息表后轮询各 rank 的
//                       row-group 同步计数器，发现完成的 group 后调用
//                       ProcessCombineBatch 批量处理。
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineGroups(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx, uint32_t &startRankId,
    uint32_t &endRankId)
{
    int64_t offset = ALIGN_512;
    uint32_t processRankNum = 0;
    SplitToCore(worldSize_, blockAivNum_, startRankId, endRankId, processRankNum);
    if (startRankId >= endRankId || processRankNum == 0) {
        return;
    }

    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    LocalTensor<int32_t> rankInfoTensor;
    uint32_t totalTokensToProcess = 0;
    uint32_t batchBaseOffset = 0;
    if (!BuildCombineRankInfo(expertIdx, mExpert, startRankId, processRankNum, offset, rankInfoTensor,
                              totalTokensToProcess, batchBaseOffset)) {
        return;
    }
    AscendC::SetCtrlSpr<60, 60>(0);

    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    // A8W4/A4W4 路径每两个 AIV 只有 subBlockIdx=1 参与 Combine，逻辑核数需与 producer 端一致。
    uint32_t logicalCoreCount = blockAivNum_;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        logicalCoreCount = blockAivNum_ / 2;
    }
    MegaMoeImpl::GroupSyncSlotLayout slotLayout = MegaMoeImpl::CalcGroupSyncSlotLayout(mExpert, logicalCoreCount);
    __gm__ int32_t *expertCounterBase = (__gm__ int32_t *)gmmAddrInfo.gmm2CombineSyncCounter;

    uint32_t rankIndex = 0;
    uint32_t totalProcessed = 0;
    while (totalProcessed < totalTokensToProcess) {
        while (rankInfoTensor.GetValue(2 * processRankNum + rankIndex) >=
               rankInfoTensor.GetValue(processRankNum + rankIndex)) {
            rankIndex = (rankIndex + 1) % processRankNum;
        }

        uint32_t startRow = rankInfoTensor.GetValue(rankIndex);
        uint32_t processedCnt = rankInfoTensor.GetValue(2 * processRankNum + rankIndex);
        uint32_t currentRow = startRow + processedCnt;
        uint32_t targetGroup = currentRow / L1_TILE_M_256;

        uint32_t firstSyncSlot = 0;
        uint32_t syncSlotCount = 0;
        MegaMoeImpl::GetGroupSyncSlotRange(targetGroup, slotLayout, firstSyncSlot, syncSlotCount);
        __gm__ int32_t *counterAddr = MegaMoeImpl::GetCombineSyncCounterAddress(expertCounterBase, firstSyncSlot);

        if (AscendC::ReadGmByPassDCache(counterAddr) >= static_cast<int32_t>(nTilesPerGroup)) {
            ProcessCombineBatch(gmmAddrInfo, gmm2State, expertIdx, rankIndex, rankInfoTensor, processRankNum,
                                batchBaseOffset, offset, totalProcessed);
        }
        rankIndex = (rankIndex + 1) % processRankNum;
    }
}

// =============================================
// DrainCombineChannels：Phase 3 — 冲刷本核负责的所有远端 rank 的 URMA 通道。
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DrainCombineChannels(uint32_t startRankId,
                                                                                            uint32_t endRankId)
{
    for (uint32_t dr = startRankId; dr < endRankId; ++dr) {
        if (dr == rankId_) {
            continue;
        }
        hcomm_.Drain(GetUrmaCommHandle(mc2Context_, dr, rankId_));
    }
}

// =============================================
// ProcessCombine：generic combine-quant 路径的 AIV 后处理。
//                 等待本 expert 的 row-group 计数满足后，读取 triple 和 GMM2 输出，
//                 再执行 row-group 级 CombineRowGroup。
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombine(const GMMAddrInfo &gmmAddrInfo,
                                                                                      const ExpertLoopState &gmm2State,
                                                                                      uint32_t expertIdx)
{
    uint32_t startRankId = 0;
    uint32_t endRankId = 0;
    ProcessCombineGroups(gmmAddrInfo, gmm2State, expertIdx, startRankId, endRankId);

    if (expertIdx == moeExpertPerRank_ - 1) {
        DrainCombineChannels(startRankId, endRankId);
    }
}

// =============================================
// UnpermuteBuffInit：Unpermute中使用的buffer申请
// =============================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)),
                                                    static_cast<uint32_t>(ALIGN_32));
    int32_t num =
        worldSize_ *
        Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * moeExpertPerRank_),
                             static_cast<uint32_t>(ALIGN_128)) *
        sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t fixedUbBeforeTopK = dataResBufAlign + dataResFp32BufAlign;
    uint32_t scaleUbCost = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilAlign(static_cast<uint32_t>(k_), static_cast<uint32_t>(ALIGN_32));
        scaleUbCost =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32),
                                 static_cast<uint32_t>(ALIGN_32)) +
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32),
                                 static_cast<uint32_t>(ALIGN_32));
    }
    uint32_t availUb = SEND_MASK_UB_LIMIT - fixedUbBeforeTopK - scaleUbCost;
    uint32_t perTokenBytes = topK_ * sizeof(float);
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        perTokenBytes += topK_ * sizeof(bfloat16_t);
    }
    topKWeightsChunkLen_ = (perTokenBytes > 0) ? availUb / perTokenBytes : m_;
    if (topKWeightsChunkLen_ == 0) {
        topKWeightsChunkLen_ = 1;
    }
    if (topKWeightsChunkLen_ > m_) {
        topKWeightsChunkLen_ = m_;
    }
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));

    // Tensor用处：Unpermute 函数用于存储mte2搬入token；
    // Tensor大小：大小为3 *
    // 单个token长度，2块是用于mte2搬运的doubleBuffer，1块是用于存储累加计算Cast完的输出结果，用于搬出；
    uint32_t dataResAddr = 0;
    uint32_t dataResSize = dataResBufAlign / sizeof(bfloat16_t);
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResSize);
    // Tensor用处：Unpermute 函数用于存储token Cast 目的Tensor；
    // Tensor大小：dataResTensor_开设大小乘以BF16_TO_FP32；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    uint32_t dataResFp32Size = dataResFp32BufAlign / sizeof(float);
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32Size);
    // Tensor用处：用于存储topKWeight；
    // Tensor大小：m_ * topK_ * sizeof(float) align到32字节对齐；
    uint32_t topKWeightsAddr = dataResFp32Addr + dataResFp32BufAlign;
    uint32_t topKWeightsSize = topKWeightsBufAlign / sizeof(float);
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, topKWeightsAddr, topKWeightsSize);
    uint32_t tempAddr = topKWeightsAddr + topKWeightsBufAlign;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(static_cast<uint32_t>(k_), static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * DEQUANT_SCALE_EXPAND，双缓冲 + scale 扩展
        uint32_t bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_SCALE_EXPAND),
                                 static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum * sizeof(float) * DOUBLE_BUFFER * DEQUANT_SCALE_EXPAND，双缓冲 + scale 扩展
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(
            static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_SCALE_EXPAND), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
    topKWeightsTempAddr_ = tempAddr;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteSharedExpert(int32_t tokenIdx)
{
    LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
    LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
    LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
    LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)params_.workspaceInfo.sharedExpertResultPtr);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        auto event = (sharedIdx % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto dataInBf16 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
        auto dataInFp32 = (sharedIdx % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
        WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        DataCopy(dataInBf16, sharedResult[(sharedIdx * m_ + tokenIdx) * k_], k_);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
        Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        PipeBarrier<PIPE_V>();
        Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE2>(event);
    }
}

// ===============================================================
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t chunkStart = coreOffset; chunkStart < coreLen + coreOffset;) {
        int32_t chunkEnd = chunkStart + static_cast<int32_t>(topKWeightsChunkLen_);
        if (chunkEnd > coreLen + coreOffset) {
            chunkEnd = coreLen + coreOffset;
        }
        int32_t chunkTokenCnt = chunkEnd - chunkStart;
        if constexpr (!TopkWeightsPrefetch) {
            if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
                GlobalTensor<float> topKWeightsGlobalTensor_;
                topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)params_.probsGmAddr);
                DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(float)),
                                                0U, 0U, 0U};
                DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[chunkStart * topK_], copyParams,
                            copyPadParams);
                SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
            }
            if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
                uint32_t tempBufAlign = Ops::Base::CeilAlign(
                    static_cast<uint32_t>(topKWeightsChunkLen_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
                LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, topKWeightsTempAddr_,
                                                  tempBufAlign / sizeof(bfloat16_t));
                GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
                topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)params_.probsGmAddr);
                DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(chunkTokenCnt * topK_ * sizeof(bfloat16_t)),
                                                0U, 0U, 0U};
                DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
                DataCopyPad(tempLocal, topkWeightsGlobalTensor[chunkStart * topK_], copyParams, copyPadParams);
                SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
                Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, chunkTokenCnt * topK_);
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
            }
        }
        for (int32_t tokenIdx = chunkStart; tokenIdx < chunkEnd; tokenIdx++) {
            int32_t localIdx = tokenIdx - chunkStart;
            SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
            LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
            LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
            LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
            LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
            for (int32_t expId = 0; expId < topK_; ++expId) {
                auto event = (expId % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
                auto dataInBf16 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Bf16 : dataIn1Bf16;
                auto dataInFp32 = (expId % DOUBLE_BUFFER == 0) ? dataIn0Fp32 : dataIn1Fp32;
                if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
                    WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                    DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * k_], k_);
                    SetFlag<AscendC::HardEvent::MTE2_V>(event);
                    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                    SetFlag<AscendC::HardEvent::S_V>(event);
                    WaitFlag<AscendC::HardEvent::S_V>(event);
                    Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
                } else {
                    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
                    uint32_t quantTokenSize = k_ + nScale;
                    uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
                    WaitFlag<AscendC::HardEvent::V_MTE2>(event);
                    DataCopy(dataInBf16, expandedX[(tokenIdx * topK_ + expId) * quantEleNum], quantEleNum);
                    SetFlag<AscendC::HardEvent::MTE2_V>(event);
                    WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                    using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t,
                                                              fp8_e5m2_t>::type;
                    SetFlag<AscendC::HardEvent::S_V>(event);
                    WaitFlag<AscendC::HardEvent::S_V>(event);
                    MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_,
                                                                          fp32ScaleTensor_, nScale, k_);
                }
                PipeBarrier<PIPE_V>();
                if constexpr (TopkWeightsPrefetch) {
                    if (expId == 0) {
                        DataCopy(dataResFp32Tensor_, dataInFp32, k_);
                    } else {
                        Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    float expScale = topKWeightsTensor_.GetValue(localIdx * topK_ + expId);
                    if (expId == 0) {
                        Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
                    } else {
                        Muls(dataInFp32, dataInFp32, expScale, k_);
                        PipeBarrier<PIPE_V>();
                        Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
                        PipeBarrier<PIPE_V>();
                    }
                }
                SetFlag<AscendC::HardEvent::V_MTE2>(event);
            }
            // 共享专家结果累加（直接加，不乘 topk_weight）
            if (sharedExpertNum_ > 0) {
                UnpermuteSharedExpert(tokenIdx);
            }
            // fp32 -> bf16
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
        }
        chunkStart = chunkEnd;
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rankSyncInWorldPtr前48K用于同步，后面区域用于记录当前syncCnt值
// ==============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = (__gm__ int32_t *)params_.peermemInfo.rankSyncInWorldPtr;
    __gm__ int32_t *syncCount =
        (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr + 48 * 1024 + aivCoreIdx_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    WriteGmByPassDCache(syncCount, count);
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        __gm__ int32_t *syncRemoteAddr = (__gm__ int32_t *)(winRankAddr_[rankIndex]) + rankId_ * 16;
        hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, rankIndex, rankId_), (GM_ADDR)syncRemoteAddr, (GM_ADDR)syncCount,
                        static_cast<int64_t>(sizeof(int32_t)));
        auto syncCheck = syncRank + rankIndex * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// SharedExpertCopyInput：从原始 bf16 输入量化后写入共享专家专用缓冲区
//   源: aGmAddr [bs × h] bf16（layered URMA 模式下 quantTokenScalePtr 未填充，需直接从原始输入量化）
//   目标: sharedExpertInputDataPtr [bs × h] fp8 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在 AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SharedExpertCopyInput()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t curentNum;
    int32_t curentOffset;
    TilingByCore(m_, curentNum, curentOffset, 1);
    uint32_t H = k_;

    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    uint32_t xInAddr = ALIGN_512;
    uint32_t xInSize = Ops::Base::CeilAlign(H, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> xInBuf0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAddr, xInSize / sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> xInBuf1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAddr + xInSize, xInSize / sizeof(bfloat16_t));
    uint32_t mxTempAddr = xInAddr + xInSize * 2;
    LocalTensor<uint16_t> mxTempBuf =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempAddr, MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));
    uint32_t xOutAddr = mxTempAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutSize =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    LocalTensor<ActivationType> xOutBuf0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutAddr, xOutSize / sizeof(ActivationType));
    LocalTensor<ActivationType> xOutBuf1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutAddr + xOutSize, xOutSize / sizeof(ActivationType));

    GlobalTensor<bfloat16_t> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleDstGlobalTensor;
    dataDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params_.workspaceInfo.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.sharedExpertInputScalePtr));

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        int32_t tokenIdx = curentOffset + index;
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInBuf = (index % DOUBLE_BUFFER == 0) ? xInBuf0 : xInBuf1;
        auto xOutBuf = (index % DOUBLE_BUFFER == 0) ? xOutBuf0 : xOutBuf1;

        srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(
            params_.aGmAddr + static_cast<uint64_t>(tokenIdx) * H * sizeof(bfloat16_t)));
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInBuf, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(event);
        WaitFlag<AscendC::HardEvent::MTE2_V>(event);

        __ubuf__ bfloat16_t *srcAddr = reinterpret_cast<__ubuf__ bfloat16_t *>(xInBuf.GetPhyAddr());
        __ubuf__ uint16_t *maxExpAddr = reinterpret_cast<__ubuf__ uint16_t *>(mxTempBuf.GetPhyAddr());
        __ubuf__ uint16_t *halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t *>(
            mxTempBuf[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))]
                .GetPhyAddr());
        __ubuf__ int8_t *outDataAddr = reinterpret_cast<__ubuf__ int8_t *>(xOutBuf.GetPhyAddr());
        __ubuf__ uint16_t *mxScaleAddr =
            reinterpret_cast<__ubuf__ uint16_t *>(xOutBuf[mxQuantTokenAlignBytes_].GetPhyAddr());

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, H);
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
        if constexpr (QuantMode == E2M1_QUANT) {
            Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        } else {
            Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                                  AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        LocalTensor<QuantScaleOutType> bufScale =
            xOutBuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(dataDstGlobalTensor[tokenIdx * widthA], xOutBuf,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleDstGlobalTensor[tokenIdx * widthAScale], bufScale,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

// ===============================================================
// GroupMatmulWithSwigluQuant：按实现路径分发到 A8W4 或 generic GMM1。
//                            A8W4 由 ENABLE_A8W4 控制；generic 路径的 subBlockIdx 判断已下沉到函数内部。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithSwigluQuant(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx, int32_t &vecSetSyncCom,
    int32_t &gmTileSequence)
{
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(state.baseOffset),
            Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
            Get<IDX_FLAG_OFFSET>(state.baseOffset),
            0L,
            0L,
            0L};
        if constexpr (IsShared) {
            sharedEpilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        } else {
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    }
    if constexpr (IsShared) {
        // 共享专家不参与权重前移，走原 UB ping-pong + 硬同步路径
        if constexpr (ENABLE_A8W4) {
            MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                                    QuantScaleOutType, GMM1_TILE_M, MegaMoeImpl::L1_TILE_M_256, false,
                                                    IsShared>(sharedEpilogueOp_, params_, state.problemShape,
                                                              gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                                                              state.expertBeforeCnt, expertIdx);
        } else {
            if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, true, GMM1_TILE_M,
                                                    MegaMoeImpl::L1_TILE_M_256, false, IsShared, false>(
                    sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            } else {
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, false, GMM1_TILE_M,
                                                    MegaMoeImpl::L1_TILE_M_256, false, IsShared, false>(
                    sharedEpilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            }
        }
    } else {
        // MoE 专家走 prefetch 路径
        if constexpr (ENABLE_A8W4) {
            MegaMoeImpl::GroupMatmulSwigluQuantA8W4<QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                                                    QuantScaleOutType, GMM1_TILE_M, EPILOGUE_TILE_M,
                                                    TopkWeightsPrefetch, IsShared>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                state.expertBeforeCnt, expertIdx);
        } else {
            if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
                params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
                // NZ format (A8W8_NZ / A4W4_NZ): isWeightNZ=true, EpilogueElementA 由 SwigluQuantOutType 自动处理类型提升
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, true, GMM1_TILE_M,
                                                    EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared, false>(
                    epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            } else {
                // Generic: fp8/fp4 activation × fp8/fp4 weight in ND format (includes A4W4 ND)
                MegaMoeImpl::GroupMatmulSwigluQuant<QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t,
                                                    QuantScaleOutType, QuantScaleOutType, false, GMM1_TILE_M,
                                                    EPILOGUE_TILE_M, TopkWeightsPrefetch, IsShared, false>(
                    epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
                    state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
            }
        }
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 swiglu flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx, int32_t &vecSetSyncCom,
    int32_t &gmTileSequence)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, Weight1Type, bfloat16_t,
                                             QuantScaleOutType, QuantScaleOutType, MegaMoeImpl::L1_TILE_M_256,
                                             TopkWeightsPrefetch, IsShared, true>(
            params_, state.problemShape, gmmAddrInfo, startBlockIdx_, gmTileSequence, state.expertBeforeCnt,
            gmm2PingPongIdx_);
    } else {
        // A8W8_NZ / Generic: both use the same GroupMatmul2 template, only LayoutB differs (ZN vs ND).
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                      QuantScaleOutType, true, true, MegaMoeImpl::L1_TILE_M_256, TopkWeightsPrefetch,
                                      IsShared>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom, state.expertBeforeCnt,
                gmm2PingPongIdx_);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                      QuantScaleOutType, false, true, MegaMoeImpl::L1_TILE_M_256, TopkWeightsPrefetch,
                                      IsShared>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom, state.expertBeforeCnt,
                gmm2PingPongIdx_);
        }
    }
    if constexpr (g_coreType == AIV && !IsShared) {
        ProcessCombine(gmmAddrInfo, state, expertIdx);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1(
    const TupleShape &initShape, const BlockOffset &initOffset, int32_t &gmTileSequence)
{
    sharedEpilogueOp_.Init({params_.workspaceInfo.sharedExpertSwigluDataPtr,
                            params_.workspaceInfo.sharedExpertSwigluScalePtr, nullptr, nullptr, nullptr, nullptr,
                            nullptr, params_.tilingData->clampLimit, static_cast<uint8_t>(ActMode::SWIGLU),
                            static_cast<uint8_t>(ActSubMode::DEFAULT), 1.0f, 1.0f});

    GMMAddrInfo sharedGmm1AddrInfo;
    ExpertLoopState sharedGmm1State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm1State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM1>(sharedGmm1AddrInfo, sharedGmm1State);
        GroupMatmulWithSwigluQuant<true>(sharedGmm1AddrInfo, sharedGmm1State, sharedIdx, vecSetSyncCom, gmTileSequence);
    }
    EndSync(vecSetSyncCom);
    startBlockIdx_ = 0; // 共享专家GMM1修改了startBlockIdx_，重置给GMM1使用
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2(
    const TupleShape &initShape, const BlockOffset &initOffset, int32_t &gmTileSequence)
{
    GMMAddrInfo sharedGmm2AddrInfo;
    ExpertLoopState sharedGmm2State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm2State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State, sharedIdx, vecSetSyncCom, gmTileSequence);
    }
    SyncAll<false>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();        // 源卡按所有全局专家算 mask 并推送到目标专家卡
    ResetFlagList();      // 清理workSpace空间上的flag位
    ResetDispatchState(); // cross-server URMA dispatch 队列与 relay ready flag 清零
    if (sharedExpertNum_ > 0) {
        SharedExpertCopyInput();
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // aic需要等待flag位reset清理完成

    // 共享专家 GMM1+SwiGLU (前移, 在 MoE 之前执行, 复用 MoE 函数)
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t gmTileSequence = 0; // Specialized A8W4/A4W4 AIC-AIV1 GM tile ready sequence.
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1(initShape, initOffset, gmTileSequence);
        SyncAll<false>();
    }

    CrossRankSyncInWorldSize();

    // 2.本卡专家接收数据dispatch & GroupMatmul1 & SwigluQuant
    DispatchBuffInit();
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    ExpertLoopState dispatchState{initShape, initOffset, 0};
    ExpertLoopState gmm1State{initShape, initOffset, 0};

    // Dispatch-prefetch count forwarding（无成员变量耦合）：
    //   SendCntCal 将 expert token 数写入 nextSendCnt；
    //   循环顶部 nextSendCnt → curSendCnt 显式转发；
    //   GMM1 consumer 始终读 curSendCnt。
    uint64_t curSendCnt = 0;  // 当前 expert 的 sendCnt（GMM1 consumer 使用）
    uint64_t nextSendCnt = 0; // 下一 expert 的 sendCnt（dispatch prefetch 算出）
    int32_t vecSetSyncCom = 0;

    // 预调度 expert 0。
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            DispatchTokenToRmtServer(0);
            SendCntCal(0, nextSendCnt);
            if (UpdateGroupParams<AddrUpdateMode::GMM1>(dispatchState, 0, nextSendCnt)) {
                UpdateGlobalBuffer<AddrUpdateMode::GMM1>(dispatchAddrInfo, dispatchState);
                MetaInfoCalAndDispatch(dispatchAddrInfo, 0);
            }
        }
    }

    for (int localExpertId = 0; localExpertId < moeExpertPerRank_; localExpertId++) {
        curSendCnt = nextSendCnt; // forward: dispatch(e) → GMM1(e)

        // Prefetch dispatch expert e+1，与当前 GMM1 consumer expert e 并发。
        if constexpr (g_coreType == AIV) {
            if (subBlockIdx_ == 1 && localExpertId + 1 < moeExpertPerRank_) {
                DispatchTokenToRmtServer(localExpertId + 1);
                SendCntCal(localExpertId + 1, nextSendCnt);
                if (UpdateGroupParams<AddrUpdateMode::GMM1>(dispatchState, localExpertId + 1, nextSendCnt)) {
                    UpdateGlobalBuffer<AddrUpdateMode::GMM1>(dispatchAddrInfo, dispatchState);
                    MetaInfoCalAndDispatch(dispatchAddrInfo, localExpertId + 1);
                }
            }
        }

        // GMM1 consumer 消费 expert e。
        if (!UpdateGroupParams<AddrUpdateMode::GMM1>(gmm1State, localExpertId, curSendCnt)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmm1AddrInfo, gmm1State);
        GroupMatmulWithSwigluQuant(gmm1AddrInfo, gmm1State, localExpertId, vecSetSyncCom, gmTileSequence);
    }
    if constexpr (TopkWeightsPrefetch) {
        if constexpr (g_coreType == AIV) {
            constexpr uint32_t epilogueSubIdx = ENABLE_A8W4 ? 1 : 0;
            if (subBlockIdx_ == epilogueSubIdx) {
                int32_t allDoneTag = static_cast<int32_t>(moeExpertPerRank_ + 1);
                __gm__ int32_t *allDoneAddr =
                    reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                    static_cast<int64_t>(moeExpertPerRank_) * params_.tilingData->maxTilesPerExpert * INT_CACHELINE;
                AscendC::WriteGmByPassDCache(allDoneAddr, allDoneTag);
            }
        } else { // AIC
            int32_t allDoneTag = static_cast<int32_t>(moeExpertPerRank_ + 1);
            __gm__ int32_t *allDoneAddr = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr) +
                                          static_cast<int64_t>(moeExpertPerRank_) *
                                              params_.tilingData->maxTilesPerExpert * INT_CACHELINE;
            while (AscendC::ReadGmByPassDCache(allDoneAddr) != allDoneTag) {
                int64_t st = AscendC::GetSystemCycle();
                while (AscendC::GetSystemCycle() - st < 100) {
                }
            }
        }
    } else {
        EndSync(vecSetSyncCom);
    }
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut(); // 本卡专家接受的tokenCnt总数搬出
        }
    }

    SyncAll<true>();
    // 3. 本卡专家接收数据GroupMatmul2 & Combine
    vecSetSyncCom = 0;
    GMMAddrInfo gmm2AddrInfo;
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    InitCombineBuffers();
    for (uint32_t expertIdx = 0; expertIdx < moeExpertPerRank_; expertIdx++) {
        if (!UpdateGroupParams<AddrUpdateMode::GMM2>(gmm2State, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmm2AddrInfo, gmm2State);
        GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx, vecSetSyncCom, gmTileSequence);
    }

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    // 3.5: 共享专家 GMM2 (MoE GMM2 之后, 复用 MoE 函数)
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2(initShape, initOffset, gmTileSequence);
    }

    // 4. 本卡数据Unpermute
    if constexpr (g_coreType == AIV) {
        UnpermuteBuffInit();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeLayeredTypeClass
#undef TemplateMegaMoeLayeredTypeFunc
#endif
