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
 * \file mega_moe_wave.h
 * \brief MegaMoe A8W8 wave 流水实现
 */

#ifndef MEGA_MOE_WAVE_H
#define MEGA_MOE_WAVE_H

#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
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
#include "mega_moe_utils.h"
#include "mega_moe_job_context.h"
#include "aiv_compute/mega_moe_quant_process.h"
#include "aiv_compute/mega_moe_shared_expert_prepare.h"
#include "aiv_compute/mega_moe_expert_token_count.h"
#include "aiv_comm/mega_moe_send_mask.h"
#include "aiv_comm/mega_moe_token_dispatch.h"
#include "mega_moe_workspace_reset.h"
#include "mega_moe_gmm1_swiglu.h"
#include "mega_moe_shared_expert_gmm1_swiglu.h"
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
enum class GmmWeightLayout : uint8_t {
    ND,
    NZ,
};
enum class GmmExpertMode : uint8_t {
    ROUTED,
    SHARED,
};

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeWaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch, bool IsGmm1Interleaved
#define TemplateMegaMoeWaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch, \
        IsGmm1Interleaved

template <TemplateMegaMoeWaveTypeClass>
class MegaMoeWave {
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
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    using ActivationType = QuantOutType;
    struct ExpertLoopState {
        TupleShape problemShape;
        BlockOffset baseOffset;
        // 每个游标独立记录当前专家之前的累计行数，用于解耦 dispatch 与 GMM 的预取状态。
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoeWave(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;
    using DispatchBufferConfig = MegaMoeDispatchBufferConfig;
    struct CombineBufferConfig {
        uint32_t rowBytes = 0;
        uint32_t rowStrideBytes = 0;
        uint32_t quantRowElements = 0;
        uint32_t quantRowStorageBytes = 0;
        uint32_t slotStrideBytes = 0;
        uint32_t quantTempElements = 0;
    };

    __aicore__ inline DispatchBufferConfig DispatchBuffInit();
    __aicore__ inline CombineBufferConfig CombineBuffInit();
    __aicore__ inline SendMaskBufferConfig SendAndQuantBuffInit();
    __aicore__ inline void ExpertTokenNumsBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void SendMaskCal(const SendMaskBufferConfig &bufferConfig);
    __aicore__ inline void SendCntCal(int32_t localExpertId, uint64_t &sendCnt);
    __aicore__ inline void DispatchExpert(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx,
                                          const DispatchBufferConfig &bufferConfig);
    __aicore__ inline void DispatchExpertsUntil(ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo,
                                                uint32_t &nextDispatchExpert, uint32_t dispatchEnd,
                                                const DispatchBufferConfig &bufferConfig);
    __aicore__ inline void MetaInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId,
                                                  const DispatchBufferConfig &bufferConfig);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt = 0);
    __aicore__ inline bool UpdateSharedGroupParams(ExpertLoopState &state, uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                              uint32_t expertIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                    uint32_t sharedExpertIdx);
    __aicore__ inline void Unpermute();
    __aicore__ inline UnpermuteBufferConfig UnpermuteBuffInit();
    __aicore__ inline void UnpermuteLoadWeights(int32_t coreOffset, int32_t batchTokenOffset, int32_t batchTokenCount,
                                                LocalTensor<bfloat16_t> &tempLocal);
    __aicore__ inline void UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
                                                 const GlobalTensor<bfloat16_t> &expandedX,
                                                 const UnpermuteBufferConfig &bufferConfig);
    __aicore__ inline void PublishGmm2Ready(uint32_t slotIdx);
    __aicore__ inline void WaitGmm2Ready(uint32_t slotIdx);
    __aicore__ inline uint64_t Gmm2ReadySlotStride() const;
    __aicore__ inline MegaMoeImpl::TokenRange GetCombineOwnedRange(uint32_t tokenCount);
    __aicore__ inline TEventID CombineRowEventId(uint32_t slot)
    {
        return static_cast<TEventID>(static_cast<int32_t>(EVENT_ID0) + static_cast<int32_t>(slot));
    }
    __aicore__ inline void DrainCombineRowRing(uint32_t issuedRowCount);
    __aicore__ inline void PreloadCombineMetaInfo(uint64_t metaInfoGmTokenOffset, uint32_t tokenCount,
                                                  uint32_t metaInfoUbTokenOffset);
    template <bool IsBufferReuse>
    __aicore__ inline void ProcessCombineToken(const CombineBufferConfig &bufferConfig,
                                               GlobalTensor<bfloat16_t> &gmm2OutGm, uint64_t gmRemoteBaseOffset,
                                               uint32_t tokenLocal, LocalTensor<int32_t> &tokenMetaInfo,
                                               uint32_t slot);
    __aicore__ inline void ProcessCombineGm(GM_ADDR gmm2OutGlobal, uint32_t tokenStart, uint32_t tokenCount,
                                            uint32_t metaInfoUbTokenOffset,
                                            const CombineBufferConfig &bufferConfig, uint32_t &rowSequence);
    __aicore__ inline void ProcessGmm1Wave(uint32_t batchBegin, uint32_t batchEnd,
                                           Gmm1ExpertLoopState &gmm1State, GMMAddrInfo &gmm1AddrInfo,
                                           Gmm1SwigluState &runtimeState);
    __aicore__ inline void ProcessGmm2Wave(uint32_t batchBegin, uint32_t batchEnd,
                                           ExpertLoopState &gmm2State, GMMAddrInfo &gmm2AddrInfo,
                                           int32_t &vecSetSyncCom);
    __aicore__ inline void ProcessCombineExperts(uint32_t batchBegin, uint32_t batchEnd,
                                                 ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
                                                 const CombineBufferConfig &bufferConfig);
    __aicore__ inline void ProcessRoutedExpertWaves(const DispatchBufferConfig &dispatchBufferConfig,
                                                    const CombineBufferConfig &combineBufferConfig,
                                                    int32_t &vecSetSyncCom);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline auto DispatchCopyTmpTensor(int32_t bufferIdx) -> LocalTensor<ActivationType>;
    template <bool IsBufferReuse>
    __aicore__ inline void FetchTokenNLoadMetaInfo(int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                   GlobalTensor<ActivationType> &remoteRankGlobalTensor,
                                                   uint32_t copyInNum);
    // 搬出一个 dispatch 槽：MTE3 写 token/scale/metaInfo 到 GM，并释放 buffer(MTE3_MTE2)与 metaInfo 槽(MTE3_S)
    __aicore__ inline void DispatchCopyMte3(int32_t bufferIdx, int32_t dstIdx,
                                            GlobalTensor<ActivationType> &tokenRevGlobalTensor,
                                            GlobalTensor<QuantScaleOutType> &scaleRevGlobalTensor,
                                            GlobalTensor<int32_t> &metaInfoGlobalTensor, int32_t copyStartIdx,
                                            int32_t copyIdx);
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx,
                                              int32_t copyNum, const DispatchBufferConfig &bufferConfig);
    __aicore__ inline void QuantProcessInRank();
    __aicore__ inline void SharedExpertCopyInput();
    __aicore__ inline void ProcessSharedExpertGmm1(const TupleShape &initShape, const BlockOffset &initOffset);
    __aicore__ inline void ProcessSharedExpertGmm2(const TupleShape &initShape, const BlockOffset &initOffset);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx, int32_t localIdx,
                                                 const UnpermuteBufferConfig &bufferConfig);
    __aicore__ inline void LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor, int32_t curentOffset,
                                               int32_t index, TEventID event);
    template <GmmExpertMode Mode = GmmExpertMode::ROUTED, uint32_t EpilogueTileM,
              bool EnableTopkWeightsPrefetch, bool IsInterleaved, bool IsWaveFlagGrained>
    __aicore__ inline void GroupMatmulWithSwigluQuant(
        BlockEpilogueSwigluMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EpilogueTileM, MegaMoeImpl::L1_TILE_N, EnableTopkWeightsPrefetch,
                                   IsInterleaved, IsWaveFlagGrained> &currentEpilogueOp,
        const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx,
        int32_t &vecSetSyncCom);
    template <GmmExpertMode Mode = GmmExpertMode::ROUTED>
    __aicore__ inline void GroupMatmulGmm2(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                           int32_t &vecSetSyncCom);
    template <GmmExpertMode Mode, GmmWeightLayout Layout>
    __aicore__ inline void GroupMatmulGmm2Impl(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                               int32_t &vecSetSyncCom);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    DispatchPrepareContext dispatchPrepareContext_;
    SendMaskArgs sendMaskArgs_;
    ResetWorkspaceArgs resetWorkspaceArgs_;
    QuantProcessArgs quantProcessArgs_;
    SharedExpertPrepareArgs sharedExpertPrepareArgs_;
    TokenDispatchContext tokenDispatchContext_;
    TokenDispatchArgs tokenDispatchArgs_;
    Gmm1SwigluContext gmm1Context_;
    Gmm1SwigluArgs gmm1Args_;
    SharedExpertGmm1SwigluContext sharedGmm1Context_;
    SharedExpertGmm1SwigluArgs sharedGmm1Args_;

    GlobalTensor<int32_t> expertTokenNumsOut_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertsPerBatch_ = 1;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint16_t gmm1PingPongIdx_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t blockNumPerRank_ = 2;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t swigluFlagSlotsPerExpert_ = 0;
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
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;   // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;  // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0; // quantTokenScalePtr 相对 win 基址的偏移
    int32_t compareCount_ = 0;
    // 共享专家相关成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    bool isPerExpertWeightTensor_ = false;
    uint32_t gmm2NTilesPerGroup_ = 0;
    // 大 BS 路由分批、环形缓冲区和清零分批相关成员
    int32_t sendRouteItemsPerBatch_ = 0; // SendMaskCal 每个 batch 处理的 route item 数
    int32_t sendRouteBatchCount_ = 0;    // SendMaskCal 的 batch 总数
    int32_t recvRouteItemsPerBatch_ = 0; // MetaInfoCalAndDispatch 每个 batch 处理的 route item 数
    int32_t recvRouteBatchCount_ = 0;    // MetaInfoCalAndDispatch 的 batch 总数
    int32_t resetBatchElementCount_ = 0; // 每个 reset batch 清零的 int32 元素数（封顶到 DISPATCH_RESET_BATCH）

    static constexpr bool GMM1_INTERLEAVED = IsGmm1Interleaved;
    static constexpr bool GMM1_UB_PINGPONG = GMM1_INTERLEAVED;

    /*
     * AIV1 上 Dispatch 与 Combine 分阶段复用 UB，进入 Combine 前 Dispatch 的动态 ring 已经排空：
     *   [0, 64 KiB)       Dispatch 的 cumsum 等跨 wave 常驻状态；路由流水结束后复用其中最多 36 KiB，
     *                     从 GM 恢复并压紧最多 1024 个专家的 token count；
     *   [64, 160 KiB)     非量化 Combine 的 6 个 BF16 row buffer（H 最大 8 KiB）；
     *                     量化 Combine 使用 2 个 [BF16 row | FP8 data + scale] 槽及共享量化 scratch；
     *   [160, 184 KiB)    空闲；
     *   [184, 187.5 KiB)  GMM2-ready 的 GM 搬入、ReduceSum scratch 和最终 sum；
     *   [187.5, 200 KiB)  空闲；
     *   [200, 248 KiB)    Combine 共用的 meta-info，共 1536 token * 8 int32；
     *   [248, 256 KiB)    硬件保留，不使用。
     */
    static constexpr uint32_t COMBINE_NO_QUANT_ROW_BUFFER_NUM = 6U;
    static constexpr uint32_t COMBINE_QUANT_ROW_BUFFER_NUM = 2U;
    static constexpr uint32_t COMBINE_ROW_BUFFER_NUM =
        CombineQuantMode == COMBINE_NO_QUANT ? COMBINE_NO_QUANT_ROW_BUFFER_NUM : COMBINE_QUANT_ROW_BUFFER_NUM;
    static constexpr uint32_t COMBINE_UB_BASE = 64U * 1024U;
    static constexpr uint32_t GMM2_READY_SCAN_UB_ADDR = 184U * 1024U;
    static constexpr uint32_t GMM2_READY_MAX_SCAN_BYTES =
        MAX_AICORE_NUM * INT_CACHELINE * sizeof(int32_t) > ALIGN_512 ?
            MAX_AICORE_NUM *INT_CACHELINE * sizeof(int32_t) :
            ALIGN_512;
    static constexpr uint32_t GMM2_READY_REDUCE_TMP_UB_ADDR =
        GMM2_READY_SCAN_UB_ADDR +
        ((GMM2_READY_MAX_SCAN_BYTES + ALIGN_512 - 1U) / ALIGN_512) * ALIGN_512;
    static constexpr uint32_t GMM2_READY_SUM_UB_ADDR = GMM2_READY_REDUCE_TMP_UB_ADDR + ALIGN_512;
    static constexpr uint32_t COMBINE_META_INFO_TOKEN_CAPACITY = 1536U;
    LocalTensor<int32_t> expertTokenNumsStridedTensor_;
    LocalTensor<int32_t> expertTokenNumsCompactTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> combineMetaInfoTensor_;
    LocalTensor<bfloat16_t> combineRowBufferTensor_;
    LocalTensor<float> combineQuantTempTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor_; // Unpermute bf16 weight 搬运中转
    QuantProcessScratch<ActivationType> quantProcessScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    Gmm1SwigluScratch gmm1Scratch_;

    static constexpr uint32_t GMM1_TILE_M = MegaMoeImpl::L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;

    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EPILOGUE_TILE_M, MegaMoeImpl::L1_TILE_N, TopkWeightsPrefetch, GMM1_INTERLEAVED,
                                   true>;
    using SharedBlockEpilogue =
        BlockEpilogueSwigluMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, false, GMM1_INTERLEAVED, true>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

// ========================
// Init：初始化成员并计算地址偏移
// ========================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1,
    GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = static_cast<uint64_t>(m_) * topK_;
    worldSize_ = tilingData->epWorldSize;
    moeExpertPerRank_ = tilingData->moeExpertPerRank;
    sharedExpertNum_ = tilingData->sharedExpertNum;
    isPerExpertWeightTensor_ = tilingData->isPerExpertWeightTensor;
    expertsPerBatch_ = tilingData->expertsPerBatch == 0U ? 1U : tilingData->expertsPerBatch;
    if (expertsPerBatch_ > moeExpertPerRank_) {
        expertsPerBatch_ = moeExpertPerRank_;
    }
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(GMM1_TILE_M)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    swigluFlagSlotsPerExpert_ = dispatchFlagSlotsPerExpert_;
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext *>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    moeWeightTensorListAddrs_ = {weight1, weightScales1, weight2, weightScales2};
    sharedWeightTensorListAddrs_ =
        {sharedWeight1, sharedWeightScales1, sharedWeight2, sharedWeightScales2};
    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, 1U);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t *)params_.expertTokenNumsOutGmAddr);
    tokenDispatchScratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
                      params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr,
                      params_.workspaceInfo.metaInfoPtr, tilingData->clampLimit, tilingData->actMode,
                      tilingData->actSubMode, tilingData->activationAlpha, tilingData->activationBeta});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr - params_.peermemInfo.rankSyncInWorldPtr);
    quantWinOffset_ =
        static_cast<uint64_t>(params_.peermemInfo.quantTokenScalePtr - params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ =
        Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_256)) /
        sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    mxQuantScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_ * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenScaleAlignBytes_ = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }

    dispatchPrepareContext_ = {
        {aivCoreIdx_, blockAivNum_}, {rankId_, worldSize_, moeExpertPerRank_}, {static_cast<int32_t>(m_), topK_, k_}};
    const SendMaskBufferConfig &sendMaskBufferConfig =
        aivCoreIdx_ < tilingData->sendMaskCoreCountWithExtraExpert ?
            tilingData->sendMaskConfigForCoreWithExtraExpert :
            tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendMaskArgs_ = {params_.expertIdxGmAddr, ::winRankAddr_, maskAlignSize_, maskSlotSize_, maskWinOffset_,
                     sendMaskBufferConfig};
    int32_t resetFlagNum =
        static_cast<int32_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    int32_t sharedExpertGmm2TileCounterNum =
        static_cast<int32_t>(Ops::Base::CeilDiv(m_, GMM1_TILE_M) * sharedExpertNum_ *
                             static_cast<uint64_t>(INT_CACHELINE));
    resetWorkspaceArgs_ = {params_.workspaceInfo.flagSwiGluToGmm2Ptr,
                           params_.workspaceInfo.gmm2CombineSyncCounterPtr,
                           params_.workspaceInfo.sharedExpertGmm2TileCounterPtr,
                           params_.workspaceInfo.gmm1TileStatusPtr,
                           resetFlagNum,
                           0,
                           sharedExpertGmm2TileCounterNum,
                           static_cast<int32_t>(moeExpertPerRank_),
                           static_cast<int32_t>(tilingData->maxTilesPerExpert),
                           0};
    quantProcessArgs_ = {params_.aGmAddr,
                         params_.probsGmAddr,
                         params_.peermemInfo.quantTokenScalePtr,
                         mxQuantTokenAlignBytes_,
                         mxQuantScaleAlignBytes_,
                         mxQuantTokenScaleAlignBytes_,
                         mxQuantScaleNumAlignPerToken_,
                         topK_};
    sharedExpertPrepareArgs_ = {params_.peermemInfo.quantTokenScalePtr,
                                params_.workspaceInfo.sharedExpertInputDataPtr,
                                params_.workspaceInfo.sharedExpertInputScalePtr,
                                mxQuantTokenAlignBytes_,
                                mxQuantScaleAlignBytes_,
                                mxQuantTokenScaleAlignBytes_,
                                1U};
    tokenDispatchContext_ = {{blockIdx_, blockNum_},
                             {blockIdx_, aicNum_},
                             worldSize_,
                             moeExpertPerRank_,
                             k_,
                             topK_,
                             blockNumPerRank_,
                             maxOutputSize_,
                             sendTotalNum_,
                             tilingData->dispatchBufferConfig,
                             maskAlignSize_,
                             maskSlotSize_,
                             quantWinOffset_,
                             mxQuantTokenAlignBytes_,
                             mxQuantScaleAlignBytes_,
                             mxQuantTokenScaleAlignBytes_,
                             dispatchFlagSlotsPerExpert_,
                             1U};
    tokenDispatchArgs_ = {::winRankAddr_,
                          params_.peermemInfo.maskRecvPtr,
                          params_.workspaceInfo.expertRevTokenNumsPtr,
                          params_.workspaceInfo.metaInfoPtr,
                          params_.workspaceInfo.cumsumInfoPtr,
                          params_.workspaceInfo.dispatchRevDataPtr,
                          params_.workspaceInfo.dispatchRevScalePtr,
                          params_.workspaceInfo.flagDispatchToGmm1Ptr,
                          params_.workspaceInfo.flagSendCntCalToUpdParamsPtr};
    gmm1Context_ = {{blockIdx_, blockNum_},
                    {blockIdx_, aicNum_},
                    moeExpertPerRank_,
                    moeExpertPerRank_,
                    k_,
                    static_cast<uint32_t>(hiddenDim_),
                    dispatchFlagSlotsPerExpert_,
                    swigluFlagSlotsPerExpert_,
                    tilingData->maxTilesPerExpert,
                    tilingData->groupedMatmulMode,
                    1U,
                    isPerExpertWeightTensor_};
    gmm1Args_ = {params_.workspaceInfo.expertRevTokenNumsPtr,
                 params_.workspaceInfo.dispatchRevDataPtr,
                 params_.workspaceInfo.dispatchRevScalePtr,
                 params_.workspaceInfo.gmm1MmadResPtr,
                 moeWeightTensorListAddrs_.weight1,
                 moeWeightTensorListAddrs_.weightScales1,
                 params_.workspaceInfo.flagDispatchToGmm1Ptr,
                 params_.workspaceInfo.flagSendCntCalToUpdParamsPtr,
                 params_.workspaceInfo.gmm1TileStatusPtr,
                 nullptr};
    sharedGmm1Context_ = {{blockIdx_, blockNum_},
                          sharedExpertNum_,
                          m_,
                          k_,
                          static_cast<uint32_t>(hiddenDim_),
                          tilingData->groupedMatmulMode,
                          isPerExpertWeightTensor_};
    sharedGmm1Args_ = {params_.workspaceInfo.sharedExpertInputDataPtr,
                       params_.workspaceInfo.sharedExpertInputScalePtr,
                       params_.workspaceInfo.sharedExpertGmm1OutPtr,
                       sharedWeightTensorListAddrs_.weight1,
                       sharedWeightTensorListAddrs_.weightScales1,
                       params_.workspaceInfo.sharedExpertSwigluDataPtr,
                       params_.workspaceInfo.sharedExpertSwigluScalePtr,
                       nullptr,
                       tilingData->clampLimit,
                       tilingData->actMode,
                       tilingData->actSubMode,
                       tilingData->activationAlpha,
                       tilingData->activationBeta};
    gmm1Scratch_.expertRevNumsGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.expertRevTokenNumsPtr));
}

// =================================================================================================
// DispatchBuffInit：申请 SendCntCal、MetaInfoCalAndDispatch 和 ExpertTokenNumCopyOut 使用的 buffer。
//   topkIndex/validTopkIndex 按 recvRouteItemsPerBatch_ 分配，metaInfoTensor_ 常驻 ring buffer。
// =================================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline typename MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchBufferConfig
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchBuffInit()
{
    DispatchBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    tokenDispatchScratch_.revTokenElemCnt = k_; // A8W8 输出 token 的元素数
    tokenDispatchScratch_.revScaleElemCnt =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE; // 输出 token-scale 元素数，紧密排列

    // 与 route batch 无关的固定占用
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // sendCntTensor_: 每 src rank 一个 burst(32B), 共 worldsize*32B（stride 只读 count 跳过 mask 区）
    uint32_t sendCntTensorSize = worldSize_ * static_cast<uint32_t>(ALIGN_32);
    // Dispatch 的 UB 布局与 AIV 分核无关；对应 host CalcDispatchBufferConfig 的唯一配置。
    bufferConfig = params_.tilingData->dispatchBufferConfig;
    recvRouteItemsPerBatch_ = bufferConfig.routeItemsPerBatch;
    recvRouteBatchCount_ = bufferConfig.routeBatchCount;

    // 按既定顺序落地址
    // Tensor 用途：在 SendCntCal 中记录本卡各专家收到的 token 总数；
    // Tensor 大小：仅记录 count 值且各专家之间复用，申请 32 字节；
    uint32_t expertTokenCntTensorAddr = 0;
    tokenDispatchScratch_.expertTokenCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor 用途：在 SendCntCal 中记录本卡专家收到 token count 的前缀和；
    // Tensor 大小：worldSize_ * moeExpertPerRank_ * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    tokenDispatchScratch_.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor 用途：SendCntCal 按 stride 跳过 mask 区读取 count 时，暂存各源 rank 的 count；
    // Tensor 大小：每个源 rank 占一个 32B burst，共 worldSize_ * 32B；
    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    tokenDispatchScratch_.sendCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr, sendCntTensorSize / sizeof(int32_t));
    // Tensor 用途：在 MetaInfoCalAndDispatch 中接收当前 batch 的 mask 切片；
    // Tensor 大小：recvRouteItemsPerBatch_ / 8 字节，每个 bit 对应一个路由项；
    uint32_t maskBatchAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchSize =
        static_cast<uint32_t>(recvRouteItemsPerBatch_ / 8) * static_cast<uint32_t>(sizeof(uint8_t));
    tokenDispatchScratch_.maskBatchTensor =
        LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint8_t));
    tokenDispatchScratch_.maskBatchU32Tensor =
        LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchAddr, maskBatchSize / sizeof(uint32_t));
    // Tensor 用途：MetaInfoCalAndDispatch 中 GatherMask 的目标 Tensor；
    // Tensor 大小：recvRouteItemsPerBatch_ * sizeof(int32_t)，向上对齐至 32 字节；
    uint32_t validTopkIndexTensorAddr = maskBatchAddr + maskBatchSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(recvRouteItemsPerBatch_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    tokenDispatchScratch_.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor 用途：MetaInfoCalAndDispatch 中 GatherMask 的源 Tensor，保存本 batch 的全局索引；
    // Tensor 大小：与 validTopkIndexTensor_ 一致，为 recvRouteItemsPerBatch_ * sizeof(int32_t)，
    // 向上对齐至 32 字节；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    tokenDispatchScratch_.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));
    // 路由批次 Tensor 后依次放置 copyTmp 环形缓冲区和 32B metaInfo 环形缓冲区。
    // Tensor 用途：MetaInfoCalAndDispatch 中的动态 dispatch 环形缓冲区，配合
    // EVENT_ID0..EVENT_ID(bufferCount-1) 形成软流水；
    // 只记基址：槽视图在热路径由 DispatchCopyTmpTensor(base + bufferIdx*mxQuantTokenScaleAlignBytes_) 现场构造，
    // Tensor 大小：bufferConfig.bufferCount 块（主线自适应 UB 预算给出的 2~6），
    // 每块 mxQuantTokenScaleAlignBytes_；
    // 该值即 Init() 算好的 Align256(token) + Align32(scale) + optional Align32(weight)，与 host
    // CalcDispatchBufferConfig 的 copyBufferBytes 恒相等，故连续 ring 中每个槽位均保持 32B 对齐。
    tokenDispatchScratch_.copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * mxQuantTokenScaleAlignBytes_;
    // Tensor 用途：CopyGMToGMPerToken 中的 metaInfo 环形缓冲区，逐 token 即时写入 GM；
    // Tensor 大小：bufferCount * 32B，与 copyTmp 槽位和事件编号一一对应。
    uint32_t metaInfoTensorAddr = tokenDispatchScratch_.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t metaInfoReserveSize =
        static_cast<uint32_t>(bufferConfig.bufferCount) * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    tokenDispatchScratch_.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoReserveSize / sizeof(int32_t));
    tokenDispatchScratch_.cumsumRevCntInRank = 0U;
    return bufferConfig;
}

// ======================================================================================
// CombineBuffInit：统一管理 wave Combine 使用的行环形缓冲区、量化临时空间和 metaInfo。
// Dispatch 与 Combine 分阶段复用 AIV1 UB；这里只建立 tensor 视图，不会覆盖 Dispatch 数据。
// ======================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline typename MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::CombineBufferConfig
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::CombineBuffInit()
{
    CombineBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }
    if (subBlockIdx_ != 1) {
        return bufferConfig;
    }

    bufferConfig.rowBytes = k_ * sizeof(bfloat16_t);
    bufferConfig.rowStrideBytes = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<uint64_t>(bufferConfig.rowBytes), static_cast<uint64_t>(ALIGN_32)));
    bufferConfig.slotStrideBytes = bufferConfig.rowStrideBytes;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t nScale = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        uint32_t tokenStorageBytes = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256));
        uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
        bufferConfig.quantRowStorageBytes = Ops::Base::CeilAlign(
            tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
        // Combine 的量化类型为 1B，元素数与完整通信记录字节数相同。
        bufferConfig.quantRowElements = bufferConfig.quantRowStorageBytes;
        bufferConfig.slotStrideBytes += bufferConfig.quantRowStorageBytes;
        bufferConfig.quantTempElements =
            Ops::Base::CeilAlign(storedScaleBytes, static_cast<uint32_t>(ALIGN_32)) + storedScaleBytes / 2U;
    }

    uint32_t rowRingBytes = COMBINE_ROW_BUFFER_NUM * bufferConfig.slotStrideBytes;
    combineRowBufferTensor_ = LocalTensor<bfloat16_t>(
        TPosition::VECIN, COMBINE_UB_BASE, rowRingBytes / sizeof(bfloat16_t));
    combineMetaInfoTensor_ = LocalTensor<int32_t>(
        TPosition::VECCALC, META_INFO_TENSOR_ADDR, COMBINE_META_INFO_TOKEN_CAPACITY * META_INFO_SIZE);
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        combineQuantTempTensor_ = LocalTensor<float>(
            TPosition::VECIN, COMBINE_UB_BASE + rowRingBytes, bufferConfig.quantTempElements);
    }
    return bufferConfig;
}

// ======================================================================================
// SendAndQuantBuffInit：申请 SendMaskCal、ResetFlagList 和 QuantProcessInRank 使用的 buffer。
//   topkIds/sendMask/sendGatherOut 按 sendRouteItemsPerBatch_ 分配，reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline typename MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::SendMaskBufferConfig
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 =
        static_cast<uint64_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetElementCountPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    // 单个 xOutTensor_ 槽位与 dispatch 的 token-scale-weight 通信记录使用相同布局。
    uint32_t xOutTensorSize = mxQuantTokenScaleAlignBytes_;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。SendMaskCal 按
    // expertId = aivCoreIdx_ + ownedIdx * blockAivNum_ 遍历，因此前 remainder 个 core 多处理一个 expert。
    bufferConfig = aivCoreIdx_ < params_.tilingData->sendMaskCoreCountWithExtraExpert ?
                       params_.tilingData->sendMaskConfigForCoreWithExtraExpert :
                       params_.tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendRouteItemsPerBatch_ = bufferConfig.routeItemsPerBatch;
    sendRouteBatchCount_ = bufferConfig.routeBatchCount;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize =
        static_cast<uint32_t>(sendRouteItemsPerBatch_) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantProcessScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantProcessScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantProcessScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantProcessScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantProcessScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    uint32_t sendGatherOutSize =
        static_cast<uint32_t>(sendRouteItemsPerBatch_) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    sharedExpertPrepareScratch_.copyBuffer0 = quantProcessScratch_.xOutTensor0;
    sharedExpertPrepareScratch_.copyBuffer1 = quantProcessScratch_.xOutTensor1;
    resetWorkspaceArgs_.resetBatchElementCount = resetBatchElementCount_;
    return bufferConfig;
}

// ===============================================================================================
// ResetFlagList：分批清零本卡 workspace 中的 flag（单批最多 DISPATCH_RESET_BATCH 个元素），
//   从 flagSwiGluToGmm2Ptr 起一次覆盖全部连续 flag，包括不同模式使用的 GMM2 计数器。
// ===============================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ResetFlagList()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    GlobalTensor<int32_t> swigluToGmm2FlagGm;
    swigluToGmm2FlagGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum =
        static_cast<int32_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();

    for (int32_t resetElementOffset = 0; resetElementOffset < coreLen; resetElementOffset += resetBatchElementCount_) {
        int32_t currentBatchElementCount = coreLen - resetElementOffset < resetBatchElementCount_ ?
                                               coreLen - resetElementOffset :
                                               resetBatchElementCount_;
        DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)), 0U,
                                             0U, 0U};
        DataCopyPad(swigluToGmm2FlagGm[coreOffset + resetElementOffset], resetTensor_, rankSyncCopyParams);
    }
    // 预取路径：清理 GMM1 tile 状态位区（含 allDone slot），避免上一轮残留导致软同步误判。
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<int32_t> statusGm;
        statusGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm1TileStatusPtr));
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
            DataCopyExtParams statusCopyParams{1U, static_cast<uint32_t>(currentBatchElementCount * sizeof(int32_t)),
                                               0U, 0U, 0U};
            DataCopyPad(statusGm[statusCoreOffset + resetElementOffset], resetTensor_, statusCopyParams);
        }
    }
}

// ======================================================================================
// ExpertTokenNumsBuffInit：最终输出阶段复用整片 UB。
//   strided Tensor 接收 expertRevNumsGlobalTensor_ 中每个专家的 32B count 槽；
//   compact Tensor 保存压紧后的连续 int32 count。
// ======================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ExpertTokenNumsBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1) {
        return;
    }

    uint32_t stridedTensorBytes = moeExpertPerRank_ * static_cast<uint32_t>(ALIGN_32);
    expertTokenNumsStridedTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, 0U, stridedTensorBytes / sizeof(int32_t));
    uint32_t compactTensorBytes = Ops::Base::CeilAlign(
        static_cast<uint64_t>(moeExpertPerRank_ * sizeof(int32_t)), static_cast<uint64_t>(ALIGN_32));
    expertTokenNumsCompactTensor_ =
        LocalTensor<int32_t>(TPosition::VECCALC, stridedTensorBytes, compactTensorBytes / sizeof(int32_t));
}

// ======================================================================================
// ExpertTokenNumCopyOut：从 GMM 实际消费的持久化 count 中恢复本卡路由专家 token 数。
//   每个专家在 expertRevNumsGlobalTensor_ 中为每个 AIC 保留一个 32B 槽；这里只取 block0 的槽，
//   一次搬入、在 UB 中压紧，再连续搬出，不依赖跨阶段保存的 cumsumInfoTensor_。
// ======================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ExpertTokenNumCopyOut()
{
    uint32_t expertCountStrideBytes =
        aicNum_ * static_cast<uint32_t>(INT32_PER_256B) * sizeof(int32_t);
    DataCopyExtParams loadParams{
        static_cast<uint16_t>(moeExpertPerRank_), static_cast<uint32_t>(sizeof(int32_t)),
        expertCountStrideBytes - static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPadParams{true, 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsStridedTensor_, tokenDispatchScratch_.expertRevNumsGlobalTensor, loadParams,
                loadPadParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();

    for (uint32_t expertIdx = 0U; expertIdx < moeExpertPerRank_; ++expertIdx) {
        int32_t tokenCount =
            expertTokenNumsStridedTensor_.GetValue(expertIdx * static_cast<uint32_t>(INT32_PER_256B));
        expertTokenNumsCompactTensor_.SetValue(expertIdx, tokenCount);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsCompactTensor_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// ======================================================================================================
// SendMaskCal：按通信域内所有专家 ID 计算本卡 topk 的 mask，并发送至目标专家卡。
//
//   阶段 1：本卡 topk 按路由批次分批搬入；
//   阶段 2：基于当前 topk 批次，逐专家生成 mask 并通过动态环形缓冲区推送。
//
// 流水（动态 2~6 buffer mask 推送）：
//   当前槽完成 mask 生成后即可发起 MTE3 推送；Vector 继续使用后续槽生成其他 expert 的 mask，
//   直到 ring 回绕时才等待对应槽的 MTE3 完成。跨 batch 时，下一批 topk 的 MTE2 加载也可与上一批
//   尚未完成的 MTE3 推送重叠。
//   EVENT_ID0~EVENT_ID(bufferCount-1) 控制各槽 MTE3 写入完成事件，保证环形缓冲区轮转使用不冲突。
//
// 关键细节：
//   - 非末 batch: pushBytes = sliceBytes（纯 mask 切片）
//   - 末 batch:   pushBytes = sliceBytes + 4B；末尾多写一个 int32 是该 expert 跨 batch 的累计 count
//                 （SendCntCal 通过 maskSlotSize 跳过 mask 区直接读 count，无需再翻 mask）
//   - sendCntAccTensor_[ownedIdx]：逐专家跨 batch 累加计数，末 batch 折叠进 mask 尾部
//   - 对端 window 地址：maskWinOffset_ + expert*srcRank*(mask+count slot) + batchStart/8 偏移
// ======================================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::SendMaskCal(const SendMaskBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 本 AIV 核负责的全局专家子集。
    int32_t totalExperts = static_cast<int32_t>(worldSize_ * moeExpertPerRank_);
    int32_t coreIdx = static_cast<int32_t>(aivCoreIdx_);
    int32_t ownedExpertNum =
        (coreIdx < totalExperts) ? Ops::Base::CeilDiv(totalExperts - coreIdx, static_cast<int32_t>(blockAivNum_)) : 0;
    if (ownedExpertNum <= 0) {
        return;
    }

    // 准备 GM 读写句柄
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t *)params_.expertIdxGmAddr);
    GlobalTensor<uint8_t> dstGlobalTensor;
    int32_t maskSliceBytesFull = sendRouteItemsPerBatch_ / 8;
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0};

    // 清零逐专家的跨 batch 累加器。
    Duplicate<int32_t>(sendMaskScratch_.sendCntAccTensor, 0, ownedExpertNum);
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

    // mask 推送环形缓冲区初始化：先将实际分配的全部槽位交给 Vector。
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }

    int32_t iter = 0;
    // 外层：路由批次循环。
    for (int32_t batchIdx = 0; batchIdx < sendRouteBatchCount_; ++batchIdx) {
        int32_t batchStart = batchIdx * sendRouteItemsPerBatch_;
        bool isLastBatch = (batchIdx == sendRouteBatchCount_ - 1);
        int32_t validLen = sendRouteItemsPerBatch_;
        int32_t sliceBytes = maskSliceBytesFull;
        int32_t pushBytes = sliceBytes;

        if (isLastBatch) {
            validLen = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(batchStart));
            if (batchStart / 8 + sliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                sliceBytes = static_cast<int32_t>(maskAlignSize_) - batchStart / 8;
            }
            pushBytes = sliceBytes + static_cast<int32_t>(sizeof(int32_t));
        }

        // 加载本 batch 的 topk
        SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
        DataCopyExtParams loadParams{1U, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(sendMaskScratch_.topkIdsTensor, srcGlobalTensor[batchStart], loadParams, loadPad);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

        // 内层：逐专家循环。
        for (int32_t ownedIdx = 0; ownedIdx < ownedExpertNum; ++ownedIdx, ++iter) {
            int32_t globalExpertId = coreIdx + ownedIdx * static_cast<int32_t>(blockAivNum_);
            int32_t dstRank = globalExpertId / static_cast<int32_t>(moeExpertPerRank_);
            int32_t localExpertId = globalExpertId % static_cast<int32_t>(moeExpertPerRank_);

            int32_t bufferIdx = iter % bufferConfig.bufferCount;
            TEventID bufEvent = static_cast<TEventID>(bufferIdx);
            LocalTensor<uint8_t> maskBuf =
                sendMaskScratch_.sendMaskTensor[bufferIdx * bufferConfig.bufferBytes];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();

            WaitFlag<AscendC::HardEvent::MTE3_V>(bufEvent);
            // DAV_3510 要求 CompareScalar 的 count * sizeof(int32_t) 按 256B 对齐，因此这里传入对齐后的
            // 批次长度，而不是 validLen。后续两处 GatherMask 均受 validLen 限制，会忽略补齐区域产生的 mask 位。
            CompareScalar(maskBuf, sendMaskScratch_.topkIdsTensor, globalExpertId, AscendC::CMPMODE::EQ,
                          sendRouteItemsPerBatch_);
            uint64_t batchMatchedRouteCount = 0;
            GatherMask(sendMaskScratch_.sendGatherOutTensor, sendMaskScratch_.topkIdsTensor, maskBufU32, true,
                       static_cast<uint32_t>(validLen), {1, 1, 0, 0}, batchMatchedRouteCount);

            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
            int32_t expertMatchedRouteCount =
                sendMaskScratch_.sendCntAccTensor.GetValue(ownedIdx) + static_cast<int32_t>(batchMatchedRouteCount);
            sendMaskScratch_.sendCntAccTensor.SetValue(ownedIdx, expertMatchedRouteCount);
            if (isLastBatch) {
                maskBuf.template ReinterpretCast<int32_t>().SetValue(sliceBytes / sizeof(int32_t),
                                                                     expertMatchedRouteCount);
            }
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();

            uint64_t dstOffset = maskWinOffset_ +
                                 static_cast<uint64_t>(localExpertId * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(rankId_)) *
                                     static_cast<uint64_t>(maskSlotSize_) +
                                 static_cast<uint64_t>(batchStart / 8);
            dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)GetRankWinAddrWithOffset(dstRank, dstOffset));
            DataCopyPad(dstGlobalTensor, maskBuf, {1U, static_cast<uint32_t>(pushBytes), 0U, 0U, 0U});
            SetFlag<AscendC::HardEvent::MTE3_V>(bufEvent);
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.bufferCount; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(static_cast<TEventID>(bufferIdx));
    }
}

// ======================================================================
// LoadTopkWeightsToUb：权重搬运到UB（TopkWeightsPrefetch=0 时仅做 MTE2_V 同步）
// ======================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::LoadTopkWeightsToUb(const LocalTensor<ActivationType> &xOutTensor,
                                                              int32_t curentOffset, int32_t index, TEventID event)
{
    uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        GlobalTensor<TopkWeightsType> weightGm;
        weightGm.SetGlobalBuffer(
            (__gm__ TopkWeightsType *)(params_.probsGmAddr + (curentOffset + index) * topK_ * sizeof(TopkWeightsType)));
        if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
            LocalTensor<TopkWeightsType> weightBf16Tmp =
                quantProcessScratch_.mxTempTensor.template ReinterpretCast<TopkWeightsType>();
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

// ===================================
// QuantProcessInRank：量化本卡 token
// ===================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::QuantProcessInRank()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 按 BS 在全部 AIV 核之间均分任务。
    int32_t currentNum;
    int32_t currentOffset;
    TilingByCore(m_, currentNum, currentOffset, 1);
    uint32_t H = k_;
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(H * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyExtParams xCopyOutParams = {1U, mxQuantTokenScaleAlignBytes_, 0U, 0U, 0U};
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        srcGlobalTensor.SetGlobalBuffer(
            (__gm__ bfloat16_t *)(params_.aGmAddr + static_cast<uint64_t>(currentOffset + index) *
                                                        static_cast<uint64_t>(H) * sizeof(bfloat16_t)));
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInTensor = (index % DOUBLE_BUFFER == 0) ? quantProcessScratch_.xInTensor0 :
                                                        quantProcessScratch_.xInTensor1;
        auto xOutTensor = (index % DOUBLE_BUFFER == 0) ? quantProcessScratch_.xOutTensor0 :
                                                         quantProcessScratch_.xOutTensor1;
        GlobalTensor<uint8_t> dstGlobalTensor;
        dstGlobalTensor.SetGlobalBuffer((__gm__ uint8_t *)(params_.peermemInfo.quantTokenScalePtr +
                                                           static_cast<uint64_t>(currentOffset + index) *
                                                               static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_)));
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopyPad(xInTensor, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
        LoadTopkWeightsToUb(xOutTensor, currentOffset, index, event);
        __ubuf__ bfloat16_t *srcAddr = (__ubuf__ bfloat16_t *)xInTensor.GetPhyAddr();
        __ubuf__ uint16_t *maxExpAddr = (__ubuf__ uint16_t *)quantProcessScratch_.mxTempTensor.GetPhyAddr();
        __ubuf__ uint16_t *halfScaleAddr =
            (__ubuf__ uint16_t *)
                quantProcessScratch_
                    .mxTempTensor[Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))]
                    .GetPhyAddr();
        __ubuf__ int8_t *outDataAddr = (__ubuf__ int8_t *)xOutTensor.GetPhyAddr();
        __ubuf__ uint16_t *mxScaleAddr = (__ubuf__ uint16_t *)xOutTensor[mxQuantTokenAlignBytes_].GetPhyAddr();

        Quant::ComputeMaxExp(srcAddr, maxExpAddr, H); // 计算最大指数
        Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr,
                                          mxQuantScaleNumAlignPerToken_); // 计算并写入量化 scale
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
                              AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, H);
        SetFlag<AscendC::HardEvent::V_MTE3>(event);
        WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        auto xOutBytesTensor = xOutTensor.template ReinterpretCast<uint8_t>();
        DataCopyPad(dstGlobalTensor, xOutBytesTensor, xCopyOutParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

// ==================================================================================================
// SendCntCal：按 stride 跳过 mask 区读取 count，得到当前专家收到的 token 总数。
//
//   阶段 1：按 stride 读取本 localExpert 的 worldSize 个 count；
//   阶段 2：逐 rank 读取 count 并计算前缀和；
//   阶段 3：写 expertRevNumsGlobalTensor_，再通过 AtomicAdd 通知 AIC。
// ==================================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::SendCntCal(int32_t localExpertId, uint64_t &sendCnt)
{
    sendCnt = 0;

    // 阶段 1：按 stride 读取本 localExpert 的 worldSize 个 count。
    GlobalTensor<int32_t> cntSrcGlobal;
    cntSrcGlobal.SetGlobalBuffer((__gm__ int32_t *)(params_.peermemInfo.maskRecvPtr +
                                                    static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_ +
                                                    maskAlignSize_));
    DataCopyExtParams cntCopyParams{static_cast<uint16_t>(worldSize_), static_cast<uint32_t>(sizeof(int32_t)),
                                    static_cast<uint32_t>(maskSlotSize_ - sizeof(int32_t)), 0U, 0U};
    DataCopyPadExtParams<int32_t> cntPad{true, 0U, 0U, 0U};
    DataCopyPad(tokenDispatchScratch_.sendCntTensor, cntSrcGlobal, cntCopyParams, cntPad);

    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>(); // count 读取(标量)就绪
    if constexpr (TopkWeightsPrefetch) {
        // 权重前移路径：进入 MetaInfoCalAndDispatch 前确保 MTE2 流水线干净，
        // 避免与 MetaInfoCalAndDispatch 内 mask 搬运的 MTE2_V(ID1) 产生跨函数 flag 干扰。
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
    }

    // 阶段 2：逐 rank 读取 count 并计算前缀和（4B count 按 32B burst 落位，下标为 rank * 8）。
    constexpr int32_t CNT_STRIDE_I32 = ALIGN_32 / sizeof(int32_t);
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t perRankCnt = tokenDispatchScratch_.sendCntTensor.GetValue(calRankId * CNT_STRIDE_I32);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        tokenDispatchScratch_.cumsumRevCntInRank += static_cast<uint64_t>(perRankCnt);
        tokenDispatchScratch_.cumsumInfoTensor.SetValue(
            localExpertId * worldSize_ + calRankId, static_cast<int32_t>(tokenDispatchScratch_.cumsumRevCntInRank));
    }

    // 阶段 3：写入 GM 并通知 AIC。
    tokenDispatchScratch_.expertTokenCntTensor.SetValue(0, sendCnt);
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopy<int32_t>(
        tokenDispatchScratch_
            .expertRevNumsGlobalTensor[localExpertId * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_],
        tokenDispatchScratch_.expertTokenCntTensor, INT32_PER_256B);
    PipeBarrier<PIPE_ALL>();

    __gm__ int32_t *sendCntFlag = (__gm__ int32_t *)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
                                  static_cast<uint64_t>(localExpertId) * aicNum_ * INT_CACHELINE +
                                  static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// AIV1 专职执行 dispatch。将状态推进集中在此函数中，保证滚动调度器对每个专家只推进一次 dispatch 游标。
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchExpert(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx, const DispatchBufferConfig &)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1) {
            return;
        }
        uint64_t sendCnt = 0;
        ComputeExpertTokenCountAndNotify<ActivationType, false, TopkWeightsPrefetch>(
            tokenDispatchContext_, tokenDispatchArgs_, tokenDispatchScratch_, expertIdx, sendCnt);
        if (UpdateGroupParams<AddrUpdateMode::GMM1>(state, expertIdx, sendCnt)) {
            UpdateGlobalBuffer<AddrUpdateMode::GMM1>(gmmAddrInfo, state, expertIdx);
            DispatchExpertTokens<ActivationType, QuantScaleOutType, false, GMM1_TILE_M, TopkWeightsPrefetch>(
                tokenDispatchContext_, tokenDispatchArgs_, tokenDispatchScratch_, expertIdx);
        }
    }
}

// 推进 AIV1 的 dispatch 流，确保区间 [nextDispatchExpert, dispatchEnd) 内每个专家只处理一次；
// 其他核角色不维护 dispatch 状态。
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchExpertsUntil(
    ExpertLoopState &state, GMMAddrInfo &gmmAddrInfo, uint32_t &nextDispatchExpert,
    uint32_t dispatchEnd, const DispatchBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ != 1) {
            return;
        }
        for (; nextDispatchExpert < dispatchEnd; ++nextDispatchExpert) {
            DispatchExpert(state, gmmAddrInfo, nextDispatchExpert, bufferConfig);
        }
    }
}

// ============================================================================
// DispatchCopyTmpTensor：由 UB 基址 + 槽偏移现场构造该槽的 buffer 视图。
//   热路径上取代 LocalTensor 数组索引，避免寄存器压力过大时溢出到 GM。
// ============================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline auto
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchCopyTmpTensor(int32_t bufferIdx) -> LocalTensor<ActivationType>
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        tokenDispatchScratch_.copyTmpBaseAddr +
            static_cast<uint32_t>(bufferIdx) * mxQuantTokenScaleAlignBytes_,
        mxQuantTokenScaleAlignBytes_ / sizeof(ActivationType));
}

// ============================================================================
// FetchTokenNLoadMetaInfo：取 token 并装载元信息——MTE2 从远程 win 取该 token，S 侧组装 metaInfo(rank/token/topk)，
//   分别 set MTE2_MTE3 / S_MTE3 供 MTE3 侧消费。
//   IsBufferReuse 为编译期常量：首窗填槽实例(<false>)不生成任何复用 WaitFlag；稳态实例(<true>)才在覆盖前等该槽
//   上一轮的 MTE3 释放(buffer 用 MTE3_MTE2，metaInfo 槽用 MTE3_S)。两个实例分开成环，每个 token 便无需运行时分支。
//   TopkWeightsPrefetch=1 时，weight 数据随 token 一起搬运(copyInNum 含 weightAlignBytes_)，
//   但 weight 提取延迟到 DispatchCopyMte3 中 MTE2 完成后进行，故此处 set MTE2_S 而非 S_MTE3。
// ============================================================================
template <TemplateMegaMoeWaveTypeClass>
template <bool IsBufferReuse>
__aicore__ inline void
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::FetchTokenNLoadMetaInfo(
    int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
    GlobalTensor<ActivationType> &remoteRankGlobalTensor, uint32_t copyInNum)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx); // buffer 0~5 直接对应 EVENT_ID0~EVENT_ID5
    LocalTensor<ActivationType> copyTmpTensor = DispatchCopyTmpTensor(bufferIdx);
    int32_t tokenIndex = topkIndex / topK_;
    uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(copyInNum);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId); // 等该槽上一轮 token/scale 搬完，方可覆盖
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], copyInNum);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId); // 等该 metaInfo 槽被 MTE3 读走，方可覆盖
    }
    tokenDispatchScratch_.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    tokenDispatchScratch_.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    tokenDispatchScratch_.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

// ============================================================================
// DispatchCopyMte3：搬出一个 dispatch 槽——token/scale/metaInfo 三段写 GM，收尾释放 buffer 与 metaInfo 槽。
//   TopkWeightsPrefetch=1 时，先 Wait<MTE2_S> 等 MTE2 搬运完成，从 copyTmp 中提取 weight 写入 WEIGHT_INDEX，
//   再 Set<S_MTE3>。
//   每 token 的元素数取自成员 revTokenElemCnt_/revScaleElemCnt_(DispatchBuffInit 算一次)，此处不再重算。
// ============================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DispatchCopyMte3(
    int32_t bufferIdx, int32_t dstIdx, GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleOutType> &scaleRevGlobalTensor, GlobalTensor<int32_t> &metaInfoGlobalTensor,
    int32_t copyStartIdx, int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx); // buffer 0~5 直接对应 EVENT_ID0~EVENT_ID5
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScalebuf = DispatchCopyTmpTensor(bufferIdx);
    LocalTensor<QuantScaleOutType> bufScale =
        tokenScalebuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();

    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
        LocalTensor<int32_t> bufWeightsInt32 = tokenScalebuf[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = tokenDispatchScratch_.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx);
        int32_t weightBits = bufWeightsInt32.GetValue(static_cast<uint32_t>(topkIndex % topK_));
        tokenDispatchScratch_.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }

    DataCopyPad(tokenRevGlobalTensor[dstIdx * tokenDispatchScratch_.revTokenElemCnt], tokenScalebuf,
                {1, static_cast<uint16_t>(tokenDispatchScratch_.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U,
                 0U});
    DataCopyPad(scaleRevGlobalTensor[dstIdx * tokenDispatchScratch_.revScaleElemCnt], bufScale,
                {1, static_cast<uint16_t>(tokenDispatchScratch_.revScaleElemCnt * sizeof(QuantScaleOutType)), 0U,
                 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId); // S 侧 metaInfo 组装完成后方可搬
    DataCopy(metaInfoGlobalTensor[dstIdx * INT32_PER_256B],
             tokenDispatchScratch_.metaInfoTensor[bufferIdx * INT32_PER_256B], INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId); // 释放 buffer
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);    // 释放 metaInfo 槽
}

// ============================================================================
// CopyGMToGMPerToken：使用动态 2~6 槽软流水，并通过 metaInfo 环形缓冲区即时写入 GM
// ----------------------------------------------------------------------------
//   阶段 1（启动）：先下发 token 0 的 MTE2，建立 MTE2 领先 MTE3 一个 token 的流水。
//   阶段 2a（首次填槽）：buffer 1~(bufferCount-1) 尚未使用，无需等待；每轮先下发 issueIdx，再搬出 issueIdx-1。
//   阶段 2b（稳态复用）：从 token bufferCount 起绕环复用 buffer，覆盖前等待对应 MTE3 释放，流水顺序保持不变。
//   阶段 3a（收尾搬出）：前两个循环都只搬到倒数第二个，这里补搬最后一个 token。
//   阶段 3b（收尾回收）：消费本次实际使用槽位中残留的 buffer-free event，避免影响下一次调用。
//
//   【为何 issue 必须排在 store 之前】若改成"搬完本轮再预取下一条"，预取前就得先等本轮 SetFlag<MTE3_MTE2>
//   被 MTE3 执行到，等价于把整条 MTE3 队列的完成时间压进每个 token 的关键路径，MTE3 深度被钉死为 1。
//   现在这种先 issue 后 store 的顺序下，阶段 2b 的 WaitFlag<MTE3_MTE2> 等待的是 bufferCount-1 轮前已释放的槽，
//   实际不阻塞，MTE3 得以自由流水。
//
//   将 IsBufferReuse 拆成 2a/2b 两个循环，使其成为编译期常量：首次填槽实例不生成 WaitFlag，稳态实例只在复用同一
//   槽位时等待，避免每个 token 做运行时分支。metaInfo 随 MTE2 下发现场组装到 ring buffer，并随 token/scale 即时
//   写 GM。事件编号由槽号直接转换，buffer 由 UB 基址 + 槽偏移构造，避免热路径数组索引溢出到 GM。
//
//   【入参约束】copyNum >= 1 由唯一调用方 MetaInfoCalAndDispatch 的
//   `if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin)` 保证，故此处不做 copyNum<=0 的入口判断
//   （阶段 3a 会访问 copyNum-1，依赖该前提；若将来新增调用方，必须自行保证或恢复该判断）。
//   buffer 数取自 host 侧自适应配置 bufferConfig.bufferCount(2~6)，替代原先固定的 DISPATCH_BUFFER_NUM。
// ============================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum,
    const DispatchBufferConfig &bufferConfig)
{
    // revTokenElemCnt_ / revScaleElemCnt_ 仅依赖 k_，已在 DispatchBuffInit 一次性算好(见成员)，此处不再逐调用重算。
    // copyInNum 直接复用 Init 算好的 Align256(token) + Align32(scale) + optional Align32(weight)；
    // wave 仅支持 A8W8，ActivationType 为 1B，元素数即字节数。
    // bufferCount 为 host 自适应 UB 预算给出的 ring 深度(2~6)，与 DispatchBuffInit 分配的 copyTmp/metaInfo 槽数一致。
    int32_t bufferCount = bufferConfig.bufferCount;
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_;
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(
        params_.workspaceInfo.dispatchRevDataPtr +
        rowDstOffsetInCore * tokenDispatchScratch_.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr +
        rowDstOffsetInCore * tokenDispatchScratch_.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(GetRankWinAddrWithOffset(remoteRankIdx, quantWinOffset_)));
    metaInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
        params_.workspaceInfo.metaInfoPtr + rowDstOffsetInCore * INT32_PER_256B * sizeof(int32_t)));

    // 无需 PipeBarrier<PIPE_ALL>：读取 validTopkIndexTensor_ 的 V→S 依赖已由调用方 MetaInfoCalAndDispatch
    // GatherMask 后的 SyncFuncStatic<V_S> 覆盖；跨调用复用 dispatch buffer / metaInfoTensor_ 的 MTE3 已由本函数
    // 末尾阶段 3b 排空；首次调用的跨阶段 UB 复用由 dispatch 阶段入口同步保证。

    // 阶段 1（启动）：先发 token 0 的 MTE2，下一步即可在发 token 1 后搬出 token 0。
    int32_t firstTopkIndex = tokenDispatchScratch_.validTopkIndexTensor.GetValue(copyStartIdx);
    FetchTokenNLoadMetaInfo<false>(0, firstTopkIndex, remoteRankIdx, remoteRankGlobalTensor, copyInNum);

    // 阶段 2a（首次填槽）：这些 buffer 没有上一轮 MTE3，用 <false> 在编译期删掉两个复用等待。
    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = tokenDispatchScratch_.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchTokenNLoadMetaInfo<false>(issueIdx, topkIndex, remoteRankIdx, remoteRankGlobalTensor, copyInNum);
        DispatchCopyMte3(copyIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
                         copyStartIdx, copyIdx);
    }

    // 阶段 2b（稳态复用）：从 token bufferCount 起绕环覆盖旧槽，用 <true> 等待该槽的 token/scale/metaInfo 均已搬出。
    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = tokenDispatchScratch_.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchTokenNLoadMetaInfo<true>(issueBufferIdx, topkIndex, remoteRankIdx, remoteRankGlobalTensor, copyInNum);
        DispatchCopyMte3(copyBufferIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
                         copyStartIdx, copyIdx);
    }

    // 阶段 3a（收尾搬出）：补搬最后一个 token（前两个循环都只搬到倒数第二个）。
    DispatchCopyMte3((copyNum - 1) % bufferCount, copyNum - 1, tokenRevGlobalTensor, scaleRevGlobalTensor,
                     metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

    // 阶段 3b（收尾回收）：消费最后一轮 MTE3 产生的 buffer-free event，防止残留影响下一次调用。
    // 收支平衡：SetFlag 共 copyNum 次（每次 DispatchCopyMte3 一对），阶段 2b 已消费
    // max(0, copyNum-bufferCount) 对，
    // 余下恰为 min(copyNum, bufferCount) = firstUseEnd 对，且残留槽号恰好覆盖 [0, firstUseEnd)。
    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx); // buffer i 对应 EVENT_IDi
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

// ====================================================================================================
// MetaInfoCalAndDispatch：按 source rank 扫描 route mask，将本 core 负责的命中项 dispatch 到目标行。
// 坐标系：match ordinal 是当前 expert/source rank 内的命中序号；dst row 是跨 expert 累加的 workspace 行号；
// expert row 是当前 expert 内的行号，用于更新 GMM1 wave flag。
// 数据流：route mask -> 压缩后的路由索引 -> 命中序号 -> 目标行 -> 专家内行号/GMM1 wave。
// ====================================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::MetaInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId,
                                                                 const DispatchBufferConfig &bufferConfig)
{
    constexpr int32_t GMM1_WAVE_ROW_COUNT = static_cast<int32_t>(GMM1_TILE_M);
    // cumsumInfo 按 [expert][source rank] 累加；前一个 expert 的末值就是当前 expert 的全局起始行。
    int32_t expertGlobalRowBegin = localExpertId == 0 ?
                                       0 :
                                       tokenDispatchScratch_.cumsumInfoTensor.GetValue(localExpertId * worldSize_ - 1);

    // 将 (source rank, rank 内 shard) 展平后分配给所有 block；同一 source rank 可由多个 core 并行 dispatch。
    for (uint32_t dispatchShardIdx = blockIdx_; dispatchShardIdx < worldSize_ * blockNumPerRank_;
         dispatchShardIdx += blockNum_) {
        uint32_t remoteRankIdx = dispatchShardIdx / blockNumPerRank_; // 当前扫描的 source rank
        uint32_t rankShardIdx = dispatchShardIdx % blockNumPerRank_;  // 当前 core 在该 rank 分片中的编号
        // 当前 expert 从该 source rank 接收的 token，在 dispatch workspace 中占据一个连续的 row segment。
        uint32_t rankSegmentDstRowBegin =
            ((remoteRankIdx == 0 && localExpertId == 0) ?
                 0 :
                 tokenDispatchScratch_.cumsumInfoTensor.GetValue(localExpertId * worldSize_ + remoteRankIdx - 1));
        // 当前 core 负责该 rank segment 中的命中序号区间 [coreMatchOrdinalBegin, coreMatchOrdinalEnd)。
        int32_t coreMatchOrdinalBegin = 0;
        int32_t coreMatchOrdinalEnd = 0;
        int32_t coreDstRowBegin = 0; // 上述区间首项在 dispatch workspace 中的全局目标行
        if (rankSegmentDstRowBegin < maxOutputSize_) {
            // rankTokenCount 是当前 source rank 发给当前 expert 的原始行数；rankDispatchRowCount 额外受
            // maxOutputSize_ 截断，是实际允许写入 workspace 的行数。
            int32_t rankTokenCount =
                tokenDispatchScratch_.cumsumInfoTensor.GetValue(localExpertId * worldSize_ + remoteRankIdx) -
                static_cast<int32_t>(rankSegmentDstRowBegin);
            int32_t rankDispatchRowCount = (rankSegmentDstRowBegin + rankTokenCount > maxOutputSize_) ?
                                               static_cast<int32_t>(maxOutputSize_ - rankSegmentDstRowBegin) :
                                               rankTokenCount;
            // 按行均分 rank segment；match ordinal 与该 segment 内的相对 row index 一一对应。
            int32_t rowsPerRankShard = Ops::Base::CeilDiv(rankDispatchRowCount, static_cast<int32_t>(blockNumPerRank_));
            int32_t rankShardRowBegin = rankShardIdx * rowsPerRankShard; // 当前 shard 在 rank segment 内的行偏移
            coreDstRowBegin = rankSegmentDstRowBegin + rankShardRowBegin;
            // 尾 shard 可能不足 rowsPerRankShard，需裁剪到 rank segment 的实际末尾。
            int32_t coreDispatchRowCount =
                (coreDstRowBegin + rowsPerRankShard > rankSegmentDstRowBegin + rankDispatchRowCount) ?
                    static_cast<int32_t>(rankSegmentDstRowBegin + rankDispatchRowCount - coreDstRowBegin) :
                    rowsPerRankShard;
            if (coreDispatchRowCount > 0) {
                coreMatchOrdinalBegin = rankShardRowBegin;
                coreMatchOrdinalEnd = rankShardRowBegin + coreDispatchRowCount;
            }
        }

        GlobalTensor<uint8_t> remoteRankMaskGlobal; // 当前 expert/source rank 对应的 route mask GM 视图
        int32_t matchedRouteCount = 0;              // 已扫描 batch 的累计命中数，即下一 batch 的首个 match ordinal
        int32_t dispatchedRowCount = 0;             // 当前 core 已实际 dispatch 的总行数
        for (int32_t batchIdx = 0; batchIdx < recvRouteBatchCount_ && matchedRouteCount < coreMatchOrdinalEnd;
             ++batchIdx) {
            int32_t batchRouteBegin = batchIdx * recvRouteItemsPerBatch_; // 当前 batch 在原始 route 数组中的起始下标
            bool isLastBatch = (batchIdx == recvRouteBatchCount_ - 1);
            int32_t validRouteCount = recvRouteItemsPerBatch_;    // 当前 batch 的有效 route item 数
            int32_t maskSliceBytes = recvRouteItemsPerBatch_ / 8; // 当前 batch 对应的 mask 搬运字节数
            if (isLastBatch) {
                validRouteCount = static_cast<int32_t>(sendTotalNum_ - static_cast<uint64_t>(batchRouteBegin));
                if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(maskAlignSize_)) {
                    maskSliceBytes = static_cast<int32_t>(maskAlignSize_) - batchRouteBegin / 8;
                }
            }
            remoteRankMaskGlobal.SetGlobalBuffer(
                (__gm__ uint8_t *)(params_.peermemInfo.maskRecvPtr +
                                   (static_cast<uint64_t>(localExpertId) * worldSize_ + remoteRankIdx) * maskSlotSize_ +
                                   static_cast<uint64_t>(batchRouteBegin / 8)));
            DataCopy(tokenDispatchScratch_.maskBatchTensor, remoteRankMaskGlobal,
                     static_cast<uint32_t>(maskSliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
            // GatherMask 根据 mask 压缩本 batch 的全局 route index，并返回当前 batch 的命中数量。
            CreateVecIndex(tokenDispatchScratch_.topkIndexTensor, batchRouteBegin, recvRouteItemsPerBatch_);
            uint64_t batchMatchedRouteCount = 0; // 当前 batch 中 mask=1 的 route item 数
            GatherMask(tokenDispatchScratch_.validTopkIndexTensor, tokenDispatchScratch_.topkIndexTensor,
                       tokenDispatchScratch_.maskBatchU32Tensor, true,
                       static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
            // 当前 batch 和本 core 各自在 match-ordinal 坐标系中的区间，二者交集即本次 dispatch 范围。
            int32_t batchMatchOrdinalBegin = matchedRouteCount; // 当前 batch 首个命中项的跨 batch 序号
            int32_t batchMatchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);
            int32_t dispatchMatchOrdinalBegin =
                batchMatchOrdinalBegin > coreMatchOrdinalBegin ? batchMatchOrdinalBegin : coreMatchOrdinalBegin;
            int32_t dispatchMatchOrdinalEnd =
                batchMatchOrdinalEnd < coreMatchOrdinalEnd ? batchMatchOrdinalEnd : coreMatchOrdinalEnd;
            if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
                // CopyGMToGMPerToken 的索引基于当前 batch 的压缩结果，目标行则使用跨 expert 的全局行号。
                int32_t batchLocalMatchBegin =
                    dispatchMatchOrdinalBegin - batchMatchOrdinalBegin; // 交集在 validTopkIndexTensor_ 中的起点
                int32_t batchDispatchRowCount =
                    dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin; // 本次从该 batch dispatch 的行数
                int32_t dispatchDstRowBegin = static_cast<int32_t>(rankSegmentDstRowBegin) + dispatchMatchOrdinalBegin;
                CopyGMToGMPerToken(dispatchDstRowBegin, remoteRankIdx, batchLocalMatchBegin, batchDispatchRowCount,
                                   bufferConfig);
                dispatchedRowCount += batchDispatchRowCount;
            }
            matchedRouteCount = batchMatchOrdinalEnd;
        }

        if (dispatchedRowCount > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            // GMM1 flag 按 expert 内的 wave 计数，因此先从全局 dst row 转换到 expert-local row。
            int32_t coreExpertRowBegin = coreDstRowBegin - expertGlobalRowBegin;
            int32_t coreExpertRowEnd = coreExpertRowBegin + dispatchedRowCount; // 当前 core 的 expert-local 半开区间
            int32_t firstWaveIdx = coreExpertRowBegin / GMM1_WAVE_ROW_COUNT;    // 该区间触达的首个 GMM1 wave
            int32_t lastWaveIdx = (coreExpertRowEnd - 1) / GMM1_WAVE_ROW_COUNT; // 该区间触达的末个 GMM1 wave
            __gm__ int32_t *flagBase = gmmAddrInfo.dispatchToGmm1Flag;
            for (int32_t waveIdx = firstWaveIdx; waveIdx <= lastWaveIdx; ++waveIdx) {
                int32_t waveExpertRowBegin = waveIdx * GMM1_WAVE_ROW_COUNT;
                int32_t waveExpertRowEnd = waveExpertRowBegin + GMM1_WAVE_ROW_COUNT;
                int32_t overlapRowBegin =
                    coreExpertRowBegin > waveExpertRowBegin ? coreExpertRowBegin : waveExpertRowBegin;
                int32_t overlapRowEnd = coreExpertRowEnd < waveExpertRowEnd ? coreExpertRowEnd : waveExpertRowEnd;
                // 每个 core 只累加自己与该 wave 的重叠行数；计数达到 wave 行数后 GMM1 才能消费。
                AtomicAdd(flagBase + static_cast<int64_t>(waveIdx) * INT_CACHELINE,
                          int32_t(overlapRowEnd - overlapRowBegin));
            }
        }
    }
}

// =====================================================================================================
// UpdateGroupParams：更新当前专家的 problemShape，并累加本卡前序专家对应的地址偏移
// ----------------------------------------------------------------------------------------------------
//   阶段 1：根据 problemShape 中的 M（前一个专家收到的 count 数），更新 baseOffset 中 GMM1/GMM2 的矩阵偏移；
//   阶段 2：更新当前专家收到的 count 数。
// =====================================================================================================
template <TemplateMegaMoeWaveTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UpdateGroupParams(
    ExpertLoopState &state, uint32_t expertIdx, uint64_t sendCnt)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_M_OFFSET>(state.baseOffset) += m;
        Get<IDX_GMM1_OFFSET>(state.baseOffset) += m * n;
        Get<IDX_GMM2_OFFSET>(state.baseOffset) += m * k;
    }

    // AIV1 计算当前专家收到的 count，写入 expertRevNumsGlobalTensor_ 后通知 AIC/AIV0 读取。
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        if (subBlockIdx_ == 0) { // AIC/AIV0 等待 AIV1 完成 SendCntCal 并更新 flag 后，再读取 count
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
                tokenDispatchScratch_.expertRevNumsGlobalTensor[offsetInCnt]);
            Get<M_VALUE>(state.problemShape) = tokenDispatchScratch_.expertRevNumsGlobalTensor.GetValue(offsetInCnt);
        } else {
            Get<M_VALUE>(state.problemShape) = sendCnt;
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        uint64_t offsetInCnt = expertIdx * 8 * aicNum_ + 8 * blockIdx_;
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
            tokenDispatchScratch_.expertRevNumsGlobalTensor[offsetInCnt]);
        Get<M_VALUE>(state.problemShape) = tokenDispatchScratch_.expertRevNumsGlobalTensor.GetValue(offsetInCnt);
    }

    if (Get<M_VALUE>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

// =====================================================================================================
// UpdateSharedGroupParams：共享专家专用，M 恒为 m_，无 flag 等待与 DCache 操作。
// =====================================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline bool MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UpdateSharedGroupParams(ExpertLoopState &state,
                                                                                         uint32_t expertIdx)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k;
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(state.baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(state.baseOffset) += n * scaleK;
        Get<IDX_C_OFFSET>(state.baseOffset) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(state.baseOffset) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(state.baseOffset) += 1;
        Get<IDX_B2_OFFSET>(state.baseOffset) += k * n / SWIGLU_N_HALF;
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
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                    const ExpertLoopState &state,
                                                                                    uint32_t expertIdx)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        if constexpr (TopkWeightsPrefetch) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.dispatchRevDataPtr + Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
                                   Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(moeWeightTensorListAddrs_.weight1,
                                                                  isPerExpertWeightTensor_, expertIdx,
                                                                  Get<IDX_B_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal =
            GetExpertWeightAddr<QuantScaleOutType>(moeWeightTensorListAddrs_.weightScales1,
                                                   isPerExpertWeightTensor_, expertIdx,
                                                   Get<IDX_B_SCALE_OFFSET>(state.baseOffset));

        if constexpr (g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset),
                Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset) * swigluFlagSlotsPerExpert_ / INT_CACHELINE,
                0L,
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<ActivationType>(moeWeightTensorListAddrs_.weight2,
                                                                  isPerExpertWeightTensor_, expertIdx,
                                                                  Get<IDX_B2_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal =
            GetExpertWeightAddr<QuantScaleOutType>(moeWeightTensorListAddrs_.weightScales2,
                                                   isPerExpertWeightTensor_, expertIdx,
                                                   Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
    }
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                   Get<IDX_FLAG_OFFSET>(state.baseOffset) * swigluFlagSlotsPerExpert_;
    // wave 流水中的每个 ready flag 独占一条 64B cache line，避免不同核轮询时发生伪共享。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t *)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享专家专用，地址来自 shared* workspace，flags 为 nullptr。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                          const ExpertLoopState &state,
                                                                                          uint32_t sharedExpertIdx)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertInputDataPtr;
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertInputScalePtr;
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<ActivationType>(sharedWeightTensorListAddrs_.weight1, isPerExpertWeightTensor_,
                                                sharedExpertIdx, Get<IDX_B_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal =
            GetExpertWeightAddr<QuantScaleOutType>(sharedWeightTensorListAddrs_.weightScales1,
                                                   isPerExpertWeightTensor_, sharedExpertIdx,
                                                   Get<IDX_B_SCALE_OFFSET>(state.baseOffset));
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        gmmAddrInfo.gmm2OutGlobal =
            params_.workspaceInfo.sharedExpertResultPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = params_.workspaceInfo.sharedExpertSwigluDataPtr +
                              Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.sharedExpertSwigluScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<ActivationType>(sharedWeightTensorListAddrs_.weight2, isPerExpertWeightTensor_,
                                                sharedExpertIdx, Get<IDX_B2_OFFSET>(state.baseOffset));
        gmmAddrInfo.bScaleGlobal =
            GetExpertWeightAddr<QuantScaleOutType>(sharedWeightTensorListAddrs_.weightScales2,
                                                   isPerExpertWeightTensor_, sharedExpertIdx,
                                                   Get<IDX_B2_SCALE_OFFSET>(state.baseOffset));
        uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, GMM1_TILE_M);
        uint32_t sharedIdx = static_cast<uint32_t>(state.expertBeforeCnt) / m_;
        gmmAddrInfo.sharedExpertGmm2TileCounter =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedIdx) * tokenGroupCount * INT_CACHELINE;
    }
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
}

template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline uint64_t MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::Gmm2ReadySlotStride() const
{
    return static_cast<uint64_t>(blockNum_) * static_cast<uint64_t>(INT_CACHELINE);
}

// ==================================================================================
// PublishGmm2Ready：每个 AIC（包括未分到 tile 的核）都向指定 slot 写入一次到达标记。
// FIX_S 保证该核此前所有 GMM2 直写 GM 的操作完成后，才发布到达标记。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::PublishGmm2Ready(uint32_t slotIdx)
{
    if constexpr (g_coreType == AIC) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(0);
        __gm__ int32_t *readyBase =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2ReadyPtr);
        uint64_t slotOffset = static_cast<uint64_t>(slotIdx) * Gmm2ReadySlotStride();
        AscendC::WriteGmByPassDCache(
            readyBase + slotOffset + static_cast<uint64_t>(blockIdx_) * INT_CACHELINE, int32_t(1));
    }
}

// ==================================================================================
// WaitGmm2Ready：每个物理 AIV1 独立搬入并归约所有 AIC 的到达 cache line，
// 不共享 Scalar 轮询地址，也不引入二级释放 flag。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::WaitGmm2Ready(uint32_t slotIdx)
{
    if constexpr (g_coreType == AIV) {
        // 静态 Tensor 编程预留 EVENT_ID6 和 EVENT_ID7。EVENT_ID0 仅在同方向事件完成配对等待后复用；
        // 不同 HardEvent 方向拥有独立的 set/wait 生命周期，也可安全复用同一编号。
        if (subBlockIdx_ != 1) {
            return;
        }
        uint32_t logicalAiv1Num = blockAivNum_ / 2U;
        if (logicalAiv1Num == 0U) {
            return;
        }

        // ready workspace 按 [slot][AIC][64B cache line] 排列。每个 AIC 只写
        // 自己 cache line 的首个 int32=1，其余位置由 ResetFlagList 保持为 0；
        // 因此整段求和就是已经完成 GMM2 写回的 AIC 数量。
        uint32_t readyElements = blockNum_ * static_cast<uint32_t>(INT_CACHELINE);

        __gm__ int32_t *readyBase =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.gmm2ReadyPtr);
        uint64_t readySlotOffset = static_cast<uint64_t>(slotIdx) * Gmm2ReadySlotStride();
        GlobalTensor<int32_t> readyGm;
        readyGm.SetGlobalBuffer(readyBase);
        // scanUb：一次轮询从 GM 搬入的完整 ready 区，包含每个 AIC 的 cache-line padding。
        LocalTensor<int32_t> scanUb(TPosition::VECCALC, GMM2_READY_SCAN_UB_ADDR, readyElements);
        // reduceTmpUb：显式提供固定 scratch，避免高阶 ReduceSum 依赖 TPipe 自动栈。
        LocalTensor<uint8_t> reduceTmpUb(TPosition::VECCALC, GMM2_READY_REDUCE_TMP_UB_ADDR, ALIGN_512);
        // sumUb：保存最终到达数，V->S 同步后由 Scalar 读取第 0 个元素。
        LocalTensor<int32_t> sumUb(TPosition::VECCALC, GMM2_READY_SUM_UB_ADDR, INT_CACHELINE);
        const uint32_t readyShape[] = {1U, readyElements};

        while (true) {
            DataCopy(scanUb, readyGm[readySlotOffset], readyElements);
            SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            // ReduceSum 直接得到整段 ready flag 的总和。允许复用 scanUb 是安全的：
            // 下一轮轮询会先由 DataCopy 完整覆盖它；每个 AIC 占 64B，末维天然满足 32B 对齐。
            ReduceSum<int32_t, AscendC::Pattern::Reduce::AR, true>(
                sumUb, scanUb, reduceTmpUb, readyShape, true);

            SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            if (sumUb.GetValue(0) >= static_cast<int32_t>(blockNum_)) {
                return;
            }
            int64_t waitStart = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - waitStart < 100) {
            }
        }
    }
}

template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline MegaMoeImpl::TokenRange
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::GetCombineOwnedRange(uint32_t tokenCount)
{
    uint32_t logicalAiv1Num = blockAivNum_ / 2U;
    if (logicalAiv1Num == 0U) {
        return {};
    }
    return MegaMoeImpl::GetBalancedTokenRange(tokenCount, aivCoreIdx_ / 2U, logicalAiv1Num);
}

template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::DrainCombineRowRing(uint32_t issuedRowCount)
{
    // 每个已使用槽最终保留一个 MTE3_MTE2 完成事件。只回收实际使用的槽，
    // 这样 token 少于 ring 深度时不会等待一个从未产生过的事件。
    uint32_t activeSlotCount =
        issuedRowCount < COMBINE_ROW_BUFFER_NUM ? issuedRowCount : COMBINE_ROW_BUFFER_NUM;
    if (activeSlotCount > 0U) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    if (activeSlotCount > 1U) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        if (activeSlotCount > 2U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
        if (activeSlotCount > 3U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID3);
        }
        if (activeSlotCount > 4U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        }
        if (activeSlotCount > 5U) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID5);
        }
    }
}

// 一次搬入当前 AIV1 所负责的连续专家 token 段对应的 metaInfo。
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::PreloadCombineMetaInfo(
    uint64_t metaInfoGmTokenOffset, uint32_t tokenCount, uint32_t metaInfoUbTokenOffset)
{
    if (tokenCount == 0U) {
        return;
    }
    LocalTensor<int32_t> metaInfoUb = combineMetaInfoTensor_[metaInfoUbTokenOffset * META_INFO_SIZE];
    GlobalTensor<int32_t> metaInfoGm;
    metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.metaInfoPtr));
    uint64_t metaInfoGmOffset = metaInfoGmTokenOffset * META_INFO_SIZE;
    DataCopy(metaInfoUb, metaInfoGm[metaInfoGmOffset], tokenCount * META_INFO_SIZE);
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

// 搬运、可选量化并发送一个 token。首轮填槽用 <false>，不生成无意义的
// MTE3_MTE2 wait；ring 回绕后用 <true> 等待旧发送完成再覆盖该槽。
template <TemplateMegaMoeWaveTypeClass>
template <bool IsBufferReuse>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessCombineToken(
    const CombineBufferConfig &bufferConfig, GlobalTensor<bfloat16_t> &gmm2OutGm, uint64_t gmRemoteBaseOffset,
    uint32_t tokenLocal, LocalTensor<int32_t> &tokenMetaInfo, uint32_t slot)
{
    TEventID eventId = CombineRowEventId(slot);
    if constexpr (IsBufferReuse) {
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

    uint32_t slotElementOffset = slot * bufferConfig.slotStrideBytes / sizeof(bfloat16_t);
    LocalTensor<bfloat16_t> rowUb = combineRowBufferTensor_[slotElementOffset];
    DataCopyExtParams gm2UbParams{1U, bufferConfig.rowBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<bfloat16_t> gm2UbPad{false, 0U, 0U, 0U};
    DataCopyPad(rowUb, gmm2OutGm[static_cast<uint64_t>(tokenLocal) * k_], gm2UbParams, gm2UbPad);

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        MegaMoeCombineImpl::SendCombineTokenRow<bfloat16_t>(
            k_, gmRemoteBaseOffset, tokenMetaInfo, rowUb, params_);
    } else {
        LocalTensor<bfloat16_t> quantUb =
            combineRowBufferTensor_[slotElementOffset + bufferConfig.rowStrideBytes / sizeof(bfloat16_t)];
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        MegaMoeCombineImpl::QuantMxFp8<CombineQuantMode, bfloat16_t>(
            quantUb, rowUb, combineQuantTempTensor_, k_);
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                                                  fp8_e4m3fn_t, fp8_e5m2_t>::type;
        LocalTensor<Fp8Type> quantSendUb = quantUb.template ReinterpretCast<Fp8Type>();
        MegaMoeCombineImpl::SendCombineTokenRow<Fp8Type>(
            bufferConfig.quantRowElements, gmRemoteBaseOffset, tokenMetaInfo, quantSendUb, params_);
    }
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
}

// AIV1 按 token 粒度执行 Combine。量化与非量化模式共用同一套专家/token 调度和 GM->UB 行环形缓冲区；
// 量化模式只在发送前增加一次 Vector 转换，其两个槽位均按 [BF16 row | FP8 data + scale] 布局。
// rowSequence 的生命周期覆盖整个 wave，因此跨专家切换时仍可保留尚未完成的发送槽。
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessCombineGm(
    GM_ADDR gmm2OutGlobal, uint32_t tokenStart, uint32_t tokenCount, uint32_t metaInfoUbTokenOffset,
    const CombineBufferConfig &bufferConfig, uint32_t &rowSequence)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1 || tokenCount == 0U) {
        return;
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        AscendC::SetCtrlSpr<60, 60>(0);
    }

    GlobalTensor<bfloat16_t> gmm2OutGm;
    gmm2OutGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(gmm2OutGlobal));
    uint64_t gmRemoteBaseOffset =
        params_.peermemInfo.combineSendPtr - params_.peermemInfo.rankSyncInWorldPtr;

    uint32_t tokenIdxInSlice = 0U;
    uint32_t firstUseCount = rowSequence < COMBINE_ROW_BUFFER_NUM ?
                                 COMBINE_ROW_BUFFER_NUM - rowSequence :
                                 0U;
    firstUseCount = firstUseCount < tokenCount ? firstUseCount : tokenCount;
    for (; tokenIdxInSlice < firstUseCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % COMBINE_ROW_BUFFER_NUM;
        LocalTensor<int32_t> tokenMetaInfo =
            combineMetaInfoTensor_[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        ProcessCombineToken<false>(bufferConfig, gmm2OutGm, gmRemoteBaseOffset,
                                   tokenStart + tokenIdxInSlice, tokenMetaInfo, slot);
    }
    for (; tokenIdxInSlice < tokenCount; ++tokenIdxInSlice, ++rowSequence) {
        uint32_t slot = rowSequence % COMBINE_ROW_BUFFER_NUM;
        LocalTensor<int32_t> tokenMetaInfo =
            combineMetaInfoTensor_[(metaInfoUbTokenOffset + tokenIdxInSlice) * META_INFO_SIZE];
        ProcessCombineToken<true>(bufferConfig, gmm2OutGm, gmRemoteBaseOffset,
                                  tokenStart + tokenIdxInSlice, tokenMetaInfo, slot);
    }
}

// ===============================================================
// UnpermuteLoadWeights：加载一个 token batch 的权重到 UB
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UnpermuteLoadWeights(
    int32_t coreOffset, int32_t batchTokenOffset, int32_t batchTokenCount, LocalTensor<bfloat16_t> &tempLocal)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float *)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_[(coreOffset + batchTokenOffset) * topK_], copyParams,
                    copyPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t *)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(batchTokenCount * topK_ * sizeof(bfloat16_t)), 0U, 0U,
                                        0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor[(coreOffset + batchTokenOffset) * topK_], copyParams,
                    copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, batchTokenCount * topK_);
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
    }
}

// ===============================================================
// UnpermuteProcessToken：单个 token 的逐专家累加
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UnpermuteProcessToken(int32_t tokenIdx, int32_t localIdx,
                                                                const GlobalTensor<bfloat16_t> &expandedX,
                                                                const UnpermuteBufferConfig &bufferConfig)
{
    for (int32_t expId = 0; expId < topK_; ++expId) {
        // 路由专家与共享专家结果在动态环形缓冲区中组成连续的累加输入序列。
        int32_t accumulationItemIdxInBatch = localIdx * (topK_ + static_cast<int32_t>(sharedExpertNum_)) + expId;
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = dataResTensor_[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 = dataResFp32Tensor_[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
        if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
        } else {
            uint32_t nScale = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
            uint32_t tokenStorageBytes = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256));
            uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
            uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(
                tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
            uint32_t quantEleNum = quantTokenSizeBytes / sizeof(bfloat16_t);
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16,
                     expandedX[(static_cast<uint64_t>(tokenIdx) * topK_ + expId) * quantEleNum],
                     quantEleNum);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type =
                typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                                          fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(
                dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_, nScale, k_);
        }
        // GetValue 在 Scalar 流水读取 expScale；两条路径汇合后统一等待，再由 Vector 流水消费。
        SetFlag<AscendC::HardEvent::S_V>(event);
        WaitFlag<AscendC::HardEvent::S_V>(event);
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
}

// ===============================================================
// UnpermuteBuffInit：分配 Unpermute 所需固定 buffer，返回本阶段的 buffer 配置
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline typename MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UnpermuteBufferConfig
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UnpermuteBuffInit()
{
    // 必须与 host SetAdaptiveBufferConfigs 对 TilingByCore(m_, ..., align=1) 的完整 chunk/tail chunk
    // 推导保持一致。coreLen 为 0 的非活跃 core 已在 Unpermute 中提前返回，不会读取 tail 配置。
    UnpermuteBufferConfig bufferConfig = aivCoreIdx_ < params_.tilingData->unpermuteFullTokenChunkCoreCount ?
                                             params_.tilingData->unpermuteConfigForFullTokenChunk :
                                             params_.tilingData->unpermuteConfigForTailTokenChunk;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign =
            Ops::Base::CeilAlign(scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) *
                                     static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign =
            Ops::Base::CeilAlign(scaleNum * static_cast<uint32_t>(sizeof(float)) *
                                     static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    int32_t tokensPerBatch = bufferConfig.tokensPerBatch;
    uint32_t topKWeightsBufAlign = bufferConfig.topKWeightsBufferBytes;
    uint32_t topKWeightsConversionBufferBytes = bufferConfig.topKWeightsConversionBufferBytes;

    uint32_t dataResBufAlign = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufAlign = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
    // Tensor用处：Unpermute 函数用于存储 mte2 搬入 token；
    // Tensor大小：(1 + bufferConfig.inputBufferCount) × 独立对齐后的 BF16 单槽大小；
    // 1 块用于累加/搬出，其余用于 MTE2 搬入。
    uint32_t dataResAddr = 0;
    dataResTensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, dataResAddr, dataResBufAlign / sizeof(bfloat16_t));
    // Tensor用处：Unpermute 函数用于存储 token Cast 目的 Tensor；
    // Tensor大小：(1 + bufferConfig.inputBufferCount) × 独立对齐后的 FP32 单槽大小；
    uint32_t dataResFp32Addr = dataResAddr + dataResBufAlign;
    dataResFp32Tensor_ = LocalTensor<float>(TPosition::VECCALC, dataResFp32Addr, dataResFp32BufAlign / sizeof(float));
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufAlign;

    // 权重缓冲区位于 scale 之前，与 master 顺序一致。
    // Tensor用处：用于存储 topKWeight；
    // Tensor大小：tokensPerBatch × topK_ × sizeof(float) align 到 32 字节对齐；
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, topKWeightsBufAlign / sizeof(float));
    tempAddr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        // Tensor用处：Unpermute 中 bf16 weight 搬运中转 buffer；
        // Tensor大小：tokensPerBatch × topK_ × sizeof(bfloat16_t) align 到 32 字节；
        topKWeightsBf16Tensor_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr,
                                                         topKWeightsConversionBufferBytes / sizeof(bfloat16_t));
        tempAddr += topKWeightsConversionBufferBytes;
    }

    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }

    return bufferConfig;
}

// ===============================================================
// UnpermuteSharedExpert：等待对应 GMM2 tile 完成后，将共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void
MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::UnpermuteSharedExpert(int32_t tokenIdx, int32_t localIdx,
                                                                const UnpermuteBufferConfig &bufferConfig)
{
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)params_.workspaceInfo.sharedExpertResultPtr);
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / GMM1_TILE_M;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, GMM1_TILE_M);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        __gm__ int32_t *counterAddr = MegaMoeImpl::GetCombineSyncCounterAddress(
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
                static_cast<uint64_t>(sharedIdx) * sharedExpertStride,
            tokenGroupIndex);
        while (AscendC::ReadGmByPassDCache(counterAddr) != static_cast<int32_t>(gmm2NTilesPerGroup_)) {
            int64_t waitStartCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
            }
        }

        int32_t accumulationItemIdxInBatch =
            localIdx * (topK_ + static_cast<int32_t>(sharedExpertNum_)) + topK_ + static_cast<int32_t>(sharedIdx);
        int32_t inputBufferIdx = accumulationItemIdxInBatch % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        LocalTensor<bfloat16_t> dataInBf16 = dataResTensor_[(inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount];
        LocalTensor<float> dataInFp32 = dataResFp32Tensor_[(inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount];
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
// Unpermute：主入口 — 初始化 buffer → 分批循环处理
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    if (coreLen == 0) {
        return;
    }
    UnpermuteBufferConfig bufferConfig = UnpermuteBuffInit();
    int32_t tokensPerBatch = bufferConfig.tokensPerBatch;

    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t *)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t *)params_.y2GmAddr);

    // 输出槽由 Vector 写入、MTE3 读出。先将槽位交给 Vector，最终回收最后一次 MTE3 完成信号。
    constexpr TEventID kOutputBufferEvent = EVENT_ID0;
    SetFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);

    // 外层：token batch
    for (int32_t batchTokenOffset = 0; batchTokenOffset < coreLen; batchTokenOffset += tokensPerBatch) {
        int32_t batchTokenCount =
            (batchTokenOffset + tokensPerBatch > coreLen) ? (coreLen - batchTokenOffset) : tokensPerBatch;

        if constexpr (!TopkWeightsPrefetch) {
            UnpermuteLoadWeights(coreOffset, batchTokenOffset, batchTokenCount, topKWeightsBf16Tensor_);
        }

        // 内层：token 循环
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
        for (int32_t localIdx = 0; localIdx < batchTokenCount; localIdx++) {
            int32_t tokenIdx = coreOffset + batchTokenOffset + localIdx;
            UnpermuteProcessToken(tokenIdx, localIdx, expandedX, bufferConfig);
            // 共享专家结果累加（直接加，不乘 topk_weight）
            if (sharedExpertNum_ > 0) {
                UnpermuteSharedExpert(tokenIdx, localIdx, bufferConfig);
            }
            // MTE2 使用独立输入槽，可与上一 token 的 MTE3 输出重叠；仅在覆盖输出槽前等待 MTE3 读完。
            WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
            Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
            DataCopy(output[static_cast<uint64_t>(tokenIdx) * k_], dataResTensor_, k_);
            SetFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
        }
        for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<TEventID>(bufferIdx));
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_V>(kOutputBufferEvent);
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rankSyncInWorldPtr前48K用于同步，后面区域用于记录当前syncCnt值
// ==============================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    __gm__ int32_t *syncRank = (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr);
    __gm__ int32_t *syncCount =
        (__gm__ int32_t *)(params_.peermemInfo.rankSyncInWorldPtr + 48 * 1024 + aivCoreIdx_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t *syncRemoteAddr = (__gm__ int32_t *)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    WriteGmByPassDCache(syncCount, count);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// SharedExpertCopyInput：将本卡量化后的交错 data+scale 拆分为连续布局
//   源: quantTokenScalePtr [token: data(256B aligned) | scale] 交错排列
//   目标: sharedExpertInputDataPtr [bs × h] 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在量化完成后、AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::SharedExpertCopyInput()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t curentNum;
    int32_t curentOffset;
    TilingByCore(m_, curentNum, curentOffset, 1);

    int64_t widthA = k_;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    // peermem 中每个 token 的 stride（prefetch 模式下含 weight，需用它计算偏移）
    uint32_t peermemTokenStride = mxQuantTokenScaleAlignBytes_;
    // 实际搬运量只需各自对齐后的 token 和 scale，不含 weight。
    uint32_t copyInNum = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;

    GlobalTensor<ActivationType> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleDstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(params_.peermemInfo.quantTokenScalePtr));
    dataDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params_.workspaceInfo.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleOutType *>(params_.workspaceInfo.sharedExpertInputScalePtr));
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < curentNum; index++) {
        int32_t tokenIdx = curentOffset + index;
        uint64_t remoteCopyOffset = static_cast<uint64_t>(tokenIdx) * static_cast<uint64_t>(peermemTokenStride);
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto copyTmpTensor = (index % DOUBLE_BUFFER == 0) ? sharedExpertPrepareScratch_.copyBuffer0 :
                                                            sharedExpertPrepareScratch_.copyBuffer1;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopy(copyTmpTensor, srcGlobalTensor[remoteCopyOffset], copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        LocalTensor<QuantScaleOutType> bufScale =
            copyTmpTensor[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(dataDstGlobalTensor[tokenIdx * widthA], copyTmpTensor,
                    {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleDstGlobalTensor[tokenIdx * widthAScale], bufScale,
                    {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeWaveTypeClass>
template <GmmExpertMode Mode, uint32_t EpilogueTileM, bool EnableTopkWeightsPrefetch,
          bool IsInterleaved, bool IsWaveFlagGrained>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::GroupMatmulWithSwigluQuant(
    BlockEpilogueSwigluMxQuant<ActivationType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                               EpilogueTileM, MegaMoeImpl::L1_TILE_N, EnableTopkWeightsPrefetch,
                               IsInterleaved, IsWaveFlagGrained> &currentEpilogueOp,
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx, int32_t &vecSetSyncCom)
{
    constexpr bool isShared = Mode == GmmExpertMode::SHARED;
    if constexpr (g_coreType == AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(state.baseOffset),
            Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
            Get<IDX_FLAG_OFFSET>(state.baseOffset) * swigluFlagSlotsPerExpert_ / INT_CACHELINE,
            0L,
            0L,
            0L};
        currentEpilogueOp.UpdateGlobalAddr(vecBaseOffset);
    }

    const bool useWeightNz =
        params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ;
    if (useWeightNz) {
        constexpr bool isWeightNz = true;
        MegaMoeImpl::GroupMatmulSwigluQuant<
            QuantOutType, ActivationType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
            isWeightNz, GMM1_TILE_M, EpilogueTileM, EnableTopkWeightsPrefetch, isShared, IsInterleaved,
            IsWaveFlagGrained>(
            currentEpilogueOp, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
            state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
    } else {
        constexpr bool isWeightNz = false;
        MegaMoeImpl::GroupMatmulSwigluQuant<
            QuantOutType, ActivationType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
            isWeightNz, GMM1_TILE_M, EpilogueTileM, EnableTopkWeightsPrefetch, isShared, IsInterleaved,
            IsWaveFlagGrained>(
            currentEpilogueOp, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom,
            state.expertBeforeCnt, expertIdx, gmm1PingPongIdx_);
    }
}

template <TemplateMegaMoeWaveTypeClass>
template <GmmExpertMode Mode, GmmWeightLayout Layout>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::GroupMatmulGmm2Impl(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, int32_t &vecSetSyncCom)
{
    constexpr bool isWeightNz = Layout == GmmWeightLayout::NZ;
    constexpr bool isShared = Mode == GmmExpertMode::SHARED;
    constexpr bool isLayered = false;
    constexpr bool isWaveFlagGrained = true;
    MegaMoeImpl::GroupMatmul2<
        COMBINE_NO_QUANT, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
        isWeightNz, isLayered, GMM1_TILE_M, TopkWeightsPrefetch, isShared, GMM1_INTERLEAVED,
        isWaveFlagGrained>(
        params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom, state.expertBeforeCnt,
        gmm2PingPongIdx_);
}

// ===============================================================
// GroupMatmulGmm2：按具名 ND/NZ layout 将 BF16 GMM2 结果直接写 GM。
// 路由专家的发送格式由后续 CombineQuantMode 分支决定。
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
template <GmmExpertMode Mode>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::GroupMatmulGmm2(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, int32_t &vecSetSyncCom)
{
    const bool isWeightNz =
        params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ;
    if (isWeightNz) {
        GroupMatmulGmm2Impl<Mode, GmmWeightLayout::NZ>(gmmAddrInfo, state, vecSetSyncCom);
    } else {
        GroupMatmulGmm2Impl<Mode, GmmWeightLayout::ND>(gmmAddrInfo, state, vecSetSyncCom);
    }
}

// ==================================================================================
// ProcessGmm1Wave：AIC/AIV0 执行 GMM1/SwiGLU 调度；AIV1 专职 dispatch/combine，
// 不参与 GMM 调度及其虚拟 tile 轮转。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessGmm1Wave(
    uint32_t batchBegin, uint32_t batchEnd, Gmm1ExpertLoopState &gmm1State,
    GMMAddrInfo &gmm1AddrInfo, Gmm1SwigluState &runtimeState)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            return;
        }
    }
    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
        if (!WaitAndUpdateGmm1GroupParams<false>(
                gmm1Context_, gmm1Args_, gmm1Scratch_, gmm1State, expertIdx, 0U)) {
            continue;
        }
        UpdateGmm1GlobalBuffer<ActivationType, Weight1Type, ActivationType, QuantScaleOutType, false,
                               TopkWeightsPrefetch>(
            gmm1Context_, gmm1Args_, epilogueOp_, gmm1AddrInfo, gmm1State, expertIdx);
        ControlGmm1SwigluPipelineByMode<QuantOutType, Weight1Type, ActivationType, QuantScaleOutType, false,
                                        GMM1_TILE_M, EPILOGUE_TILE_M, TopkWeightsPrefetch,
                                        GMM1_INTERLEAVED, true>(
            gmm1Context_, params_, epilogueOp_, gmm1AddrInfo, gmm1State, runtimeState, expertIdx);
    }
}

// ==================================================================================
// ProcessGmm2Wave：AIC 计算当前 wave 的全部专家；AIV0 只推进与 AIC 对齐的虚拟 GMM2 调度游标，
// 使下一轮 GMM1/SwiGLU 的核归属保持一致。AIV1 专职 dispatch/combine，不参与两级 GMM 调度。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessGmm2Wave(
    uint32_t batchBegin, uint32_t batchEnd, ExpertLoopState &gmm2State,
    GMMAddrInfo &gmm2AddrInfo, int32_t &vecSetSyncCom)
{
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            return;
        }
    }
    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
        if (!UpdateGroupParams<AddrUpdateMode::GMM2>(gmm2State, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmm2AddrInfo, gmm2State, expertIdx);
        GroupMatmulGmm2(gmm2AddrInfo, gmm2State, vecSetSyncCom);
        PublishGmm2Ready(expertIdx);
    }
}

// ==================================================================================
// ProcessCombineExperts：GMM2-ready 扇入完成后，AIV1 消费一个路由专家 wave。
// token 分核、metaInfo 分批和行环形缓冲区均封装在此，使主调度函数只描述阶段顺序。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessCombineExperts(
    uint32_t batchBegin, uint32_t batchEnd, ExpertLoopState &combineState, GMMAddrInfo &combineAddrInfo,
    const CombineBufferConfig &bufferConfig)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1) {
        return;
    }

    uint32_t rowSequence = 0U;
    for (uint32_t expertIdx = batchBegin; expertIdx < batchEnd; ++expertIdx) {
        if (!UpdateGroupParams<AddrUpdateMode::GMM2>(combineState, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::GMM2>(combineAddrInfo, combineState, expertIdx);
        uint32_t currentExpertTokenNum =
            static_cast<uint32_t>(Get<M_VALUE>(combineState.problemShape));
        MegaMoeImpl::TokenRange currentCoreTokenRange =
            GetCombineOwnedRange(currentExpertTokenNum);
        if (currentCoreTokenRange.count != 0U) {
            WaitGmm2Ready(expertIdx);
        }

        for (uint32_t processedTokenNum = 0U; processedTokenNum < currentCoreTokenRange.count;) {
            uint32_t remainingTokenNum = currentCoreTokenRange.count - processedTokenNum;
            uint32_t chunkTokenNum =
                remainingTokenNum < COMBINE_META_INFO_TOKEN_CAPACITY ?
                    remainingTokenNum :
                    COMBINE_META_INFO_TOKEN_CAPACITY;
            PreloadCombineMetaInfo(
                static_cast<uint64_t>(combineState.expertBeforeCnt) +
                    currentCoreTokenRange.start + processedTokenNum,
                chunkTokenNum, 0U);
            ProcessCombineGm(
                combineAddrInfo.gmm2OutGlobal,
                currentCoreTokenRange.start + processedTokenNum,
                chunkTokenNum, 0U, bufferConfig, rowSequence);
            processedTokenNum += chunkTokenNum;
        }
    }

    // 下一 wave 的 dispatch 会复用同一片 UB 和事件编号，离开前必须回收所有已使用槽位。
    DrainCombineRowRing(rowSequence);
}

// ==================================================================================
// 路由专家滚动流水：
//   AIC  ：GMM1(B) -> GMM2(B) 直写 GM -> GMM1(B+1)
//   AIV0 ：SwiGLU(B) -> SwiGLU(B+1)，不进入 Combine 或 GMM2-ready
//   AIV1 ：完整预取 Dispatch(B+1)，再按需量化并执行 Combine(B)
// 不设置批次级全局 barrier；每个 wave 仅依靠已有的 dispatch/SwiGLU 依赖和专家粒度 GMM2-ready 扇入推进。
// ==================================================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessRoutedExpertWaves(
    const DispatchBufferConfig &dispatchBufferConfig, const CombineBufferConfig &combineBufferConfig,
    int32_t &vecSetSyncCom)
{
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    GMMAddrInfo gmm2AddrInfo;
    GMMAddrInfo combineAddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState dispatchState{initShape, initOffset, 0};
    Gmm1ExpertLoopState gmm1State = CreateGmm1ExpertLoopState(gmm1Context_);
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    ExpertLoopState combineState{initShape, initOffset, 0};
    int32_t gmm1TileSequence = 0;
    Gmm1SwigluState gmm1RuntimeState{
        startBlockIdx_, vecSetSyncCom, gmm1TileSequence, gmm1PingPongIdx_};

    const uint32_t expertsPerWave = expertsPerBatch_;

    uint32_t nextDispatchExpert = 0U;
    uint32_t currentWaveExpertNum = 0U;
    for (uint32_t batchBegin = 0U; batchBegin < moeExpertPerRank_;
         batchBegin += currentWaveExpertNum) {
        currentWaveExpertNum =
            MegaMoeImpl::GetWaveExpertCount(batchBegin, moeExpertPerRank_, expertsPerWave);
        const uint32_t batchEnd = batchBegin + currentWaveExpertNum;
        const uint32_t waveBeginCursor = startBlockIdx_;

        // AIV1 在 Combine(B) 前连续完成当前 wave 与下一 wave 的 dispatch，因为 dispatch 与 combine
        // 复用同一片 UB 和事件资源。进入稳态后，游标会跳过已预取的当前 wave，只处理新出现的下一 wave。
        const uint32_t nextWaveExpertNum =
            MegaMoeImpl::GetWaveExpertCount(batchEnd, moeExpertPerRank_, expertsPerWave);
        const uint32_t nextBatchEnd = batchEnd + nextWaveExpertNum;
        DispatchExpertsUntil(
            dispatchState, dispatchAddrInfo, nextDispatchExpert, nextBatchEnd, dispatchBufferConfig);

        ProcessGmm1Wave(
            batchBegin, batchEnd, gmm1State, gmm1AddrInfo, gmm1RuntimeState);
        const uint32_t gmm1EndCursor = startBlockIdx_;

        ProcessGmm2Wave(batchBegin, batchEnd, gmm2State, gmm2AddrInfo, vecSetSyncCom);
        // AIC 与 AIV0 共同推进真实/虚拟 GMM tile 游标。若完整 wave 执行后游标回到入口位置，但 GMM1
        // 结束游标不同，说明 GMM2 恰好抵消了 GMM1 的核归属余数；下一 wave 的两级 GMM 会固定到同一组
        // 物理核。此时恢复到 GMM1 结束游标，可保留当前 wave 内的 GMM1->GMM2 映射，仅改变下一 wave
        // 的核归属相位。AIV1 跳过两级 GMM 调度，其本地游标不会触发该条件。
        const bool hasNextWave = batchEnd < moeExpertPerRank_;
        const bool fixedRoleResonance =
            startBlockIdx_ == waveBeginCursor && gmm1EndCursor != waveBeginCursor;
        if (hasNextWave && fixedRoleResonance) {
            startBlockIdx_ = gmm1EndCursor;
        }

        ProcessCombineExperts(
            batchBegin, batchEnd, combineState, combineAddrInfo, combineBufferConfig);
    }

    // 非预取路径仍有一次 AIC->AIV0 的 UB 消费确认需要回收；预取路径改用逐 tile 的 GM 状态位。
    if constexpr (!TopkWeightsPrefetch) {
        EndSync<GMM1_UB_PINGPONG>(vecSetSyncCom, gmm1PingPongIdx_);
    }
    vecSetSyncCom = 0;
    gmm1PingPongIdx_ = 0;
    if constexpr (g_coreType == AIV) {
        // 持久化 count 在各物理核间相同，仅由首个物理核的 AIV1 恢复并输出。
        if (aivCoreIdx_ == 1U) {
            ExpertTokenNumCopyOut();
        }
    }
}

// ===============================================================
// 可选共享专家流程：依次执行共享专家 GMM1 与 SwiGLU，并在结束后重置 GMM 调度状态。
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessSharedExpertGmm1(const TupleShape &,
                                                                                         const BlockOffset &)
{
    int32_t vecSetSyncCom = 0;
    int32_t gmTileSequence = 0;
    SharedExpertGmm1SwigluState runtimeState{
        startBlockIdx_, vecSetSyncCom, gmTileSequence, gmm1PingPongIdx_};
    ControlSharedExpertGmm1SwigluPipeline<QuantOutType, ActivationType, Weight1Type, ActivationType,
                                          QuantScaleOutType, false, GMM1_TILE_M, GMM1_INTERLEAVED, true>(
        sharedGmm1Context_, sharedGmm1Args_, params_, sharedEpilogueOp_, runtimeState);
    vecSetSyncCom = 0;
    gmm1PingPongIdx_ = 0;
    startBlockIdx_ = 0; // 共享专家 GMM1 修改了 startBlockIdx_，重置后供路由专家 GMM1 使用
}

// ===============================================================
// 可选共享专家流程：依次执行共享专家 GMM2，结果由 Unpermute 阶段按 tile-ready 状态消费。
// ===============================================================
template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::ProcessSharedExpertGmm2(const TupleShape &initShape,
                                                                                         const BlockOffset &initOffset)
{
    gmm2NTilesPerGroup_ = Ops::Base::CeilDiv(k_, L1_TILE_N);
    GMMAddrInfo sharedGmm2AddrInfo;
    ExpertLoopState sharedGmm2State{initShape, initOffset, 0};
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm2State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State, sharedIdx);
        GroupMatmulGmm2<GmmExpertMode::SHARED>(sharedGmm2AddrInfo, sharedGmm2State, vecSetSyncCom);
    }
}

template <TemplateMegaMoeWaveTypeClass>
__aicore__ inline void MegaMoeWave<TemplateMegaMoeWaveTypeFunc>::Process()
{
    // 保存入口时的溢出模式，并初始化输入准备阶段使用的 UB。
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // 阶段 1：AIV 完成本卡输入量化、路由 mask 推送和 flag 清零。
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareContext_, quantProcessArgs_, quantProcessScratch_);
    GatherAndSendExpertMasks(dispatchPrepareContext_, sendMaskArgs_, sendMaskScratch_);
    ResetDispatchWorkspace<false, TopkWeightsPrefetch>(
        dispatchPrepareContext_, resetWorkspaceArgs_, resetTensor_);
    if (sharedExpertNum_ > 0) {
        // 可选：为共享专家拆分连续布局的输入数据与 scale。
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType>(
            dispatchPrepareContext_, sharedExpertPrepareArgs_, sharedExpertPrepareScratch_);
    } else {
        if constexpr (g_coreType == AIV) {
            PipeBarrier<PIPE_ALL>();
        }
    }
    SyncAll<false>(); // AIC 等待 AIV 完成输入准备与 flag 清零后再进入计算

    // 可选：提前执行共享专家 GMM1 + SwiGLU，复用与路由专家相同的计算实现。
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm1(initShape, initOffset);
    }

    // 等待所有 rank 完成本轮输入准备，再读取远端 dispatch 数据。
    CrossRankSyncInWorldSize();

    // 阶段 2：按 wave 执行路由专家 Dispatch、GMM1、SwiGLU、GMM2 与 Combine。
    DispatchBufferConfig dispatchBufferConfig = DispatchBuffInit();
    CombineBufferConfig combineBufferConfig = CombineBuffInit();
    ExpertTokenNumsBuffInit();
    int32_t vecSetSyncCom = 0;

    ProcessRoutedExpertWaves(dispatchBufferConfig, combineBufferConfig, vecSetSyncCom);

    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
    }

    // 可选：路由专家流水结束后执行共享专家 GMM2。
    if (sharedExpertNum_ > 0) {
        ProcessSharedExpertGmm2(initShape, initOffset);
    }

    // 阶段 3：等待所有 rank 的 Combine 发送完成，再执行本卡 Unpermute。
    if constexpr (g_coreType == AIV) {
        CrossRankSyncInWorldSize();
        Unpermute();
    }
    // 恢复入口时保存的溢出模式。
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#undef TemplateMegaMoeWaveTypeClass
#undef TemplateMegaMoeWaveTypeFunc

} // namespace MegaMoeImpl
#endif
