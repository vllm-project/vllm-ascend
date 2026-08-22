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

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

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
#include "mega_moe_job_context.h"
#include "aiv_compute/mega_moe_quant_process.h"
#include "aiv_compute/mega_moe_shared_expert_prepare.h"
#include "aiv_comm/mega_moe_send_mask.h"
#include "mega_moe_workspace_reset.h"
#include "mega_moe_dispatch_gmm1_swiglu.h"
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

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t CombineQuantMode, bool TopkWeightsPrefetch
#define TemplateMegaMoeTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode, TopkWeightsPrefetch

template <TemplateMegaMoeTypeClass>
class MegaMoe {
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
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1,
                                GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
                                GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
                                GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    using UnpermuteBufferConfig = MegaMoeUnpermuteBufferConfig;
    using SendMaskBufferConfig = MegaMoeSendMaskBufferConfig;

    __aicore__ inline SendMaskBufferConfig SendAndQuantBuffInit();
    __aicore__ inline bool UpdateGmm2GroupParams(ExpertLoopState &state, uint32_t expertIdx);
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
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
                                          uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void ProcessSharedExpertGmm2(const TupleShape &initShape, const BlockOffset &initOffset,
                                                   int32_t &gmTileSequence);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx, int32_t localIdx,
                                                 const UnpermuteBufferConfig &bufferConfig);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
                                                  uint32_t expertIdx, uint32_t &startBlockIdx, int32_t &vecSetSyncCom,
                                                  int32_t &gmTileSequence);

    __gm__ Mc2MoeContext *mc2Context_{nullptr};
    __gm__ int32_t *gmmToEpilogueFlag_{nullptr};
    Params params_{};
    ExpertWeightTensorListAddrs moeWeightTensorListAddrs_{};
    ExpertWeightTensorListAddrs sharedWeightTensorListAddrs_{};
    DispatchPrepareContext dispatchPrepareContext_;
    DispatchGmm1SwigluContext gmm1PipelineContext_;
    SharedExpertGmm1SwigluContext sharedGmm1Context_;
    SendMaskArgs sendMaskArgs_;
    ResetWorkspaceArgs resetWorkspaceArgs_;
    QuantProcessArgs quantProcessArgs_;
    SharedExpertPrepareArgs sharedExpertPrepareArgs_;
    SharedExpertGmm1SwigluArgs sharedGmm1Args_;

    GlobalTensor<int32_t> expertTokenNumsOut_;
    // A8W4 路径下 GroupMatmulSwigluQuant 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。tokenDispatchScratch_ 中的 cumsumInfoGlobalTensor 作为 GM 持久备份：
    // ComputeExpertTokenCountAndNotify 中 Load → 计算 → Store；DispatchExpertTokens/ExpertTokenNumCopyOut 从 GM 恢复。

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
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
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;   // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;  // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0; // quantTokenScalePtr 相对 win 基址的偏移
    int32_t compareCount_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    // 主线 shared-expert 特性成员
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;
    bool isPerExpertWeightTensor_ = false;
    // ProcessCombine wave 流水参数：只依赖 k_(常量), InitCombineBuffers 算一次, 免每 expert 重算
    uint32_t gmm2NTilesPerGroup_ = 0; // CeilDiv(k_, L1_TILE_N)
    // Align32(Align256(k_) + Align2(CeilDiv(k_, MXFP_SCALE_GROUP_NUM)))
    uint32_t combineQuantTokenSizeBytes_ = 0;

    // 大 BS reset batch 成员
    int32_t resetBatchElementCount_ = 0; // 每个 reset batch 清零的 int32 元素数（封顶到 DISPATCH_RESET_BATCH）

    static constexpr uint32_t A_ELEMS_PER_BYTE = PackedElementTraits<QuantOutType>::ELEMENTS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = PackedElementTraits<Weight1Type>::ELEMENTS_PER_BYTE;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 =
        Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value && Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr uint32_t GMM1_TILE_M = MegaMoeImpl::L1_TILE_M_256;
    static constexpr uint32_t EPILOGUE_TILE_M =
        TopkWeightsPrefetch ? MegaMoeImpl::L1_TILE_M_128 : MegaMoeImpl::L1_TILE_M_256;
    QuantProcessScratch<ActivationType> quantScratch_;
    SharedExpertPrepareScratch<ActivationType> sharedExpertPrepareScratch_;
    SendMaskScratch sendMaskScratch_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<bfloat16_t> dataResTensor_;
    LocalTensor<float> dataResFp32Tensor_;
    LocalTensor<float> topKWeightsTensor_;
    LocalTensor<float> fp32ScaleTensor_;
    LocalTensor<bfloat16_t> bf16ScaleTensor_;
    LocalTensor<bfloat16_t> topKWeightsBf16Tensor_; // Unpermute bf16 weight 搬运中转

    // GMM2 走 A8W4 且 QuantMode 为 a4w4（E2M1）时，SwigluQuant 输出需提升为 fp8_e4m3fn_t。
    // 同时当 Weight2 非 fp4 但 QuantMode==E2M1 时（generic GMM2 路径），也需 promotion，
    // 否则会出现 A=QuantOutType(fp4) vs B=Weight1Type(fp8) 的类型不匹配。
    using SwigluQuantOutType = typename std::conditional<(QuantMode == E2M1_QUANT), fp8_e4m3fn_t, QuantOutType>::type;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<SwigluQuantOutType>::ELEMENTS_PER_BYTE;

    using BlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   EPILOGUE_TILE_M, MegaMoeImpl::L1_TILE_N, TopkWeightsPrefetch>;
    using SharedBlockEpilogue =
        BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true,
                                   MegaMoeImpl::L1_TILE_M_256, MegaMoeImpl::L1_TILE_N, false>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
    DispatchGmm1SwigluArgs gmm1PipelineArgs_;
    TokenDispatchScratch<ActivationType> tokenDispatchScratch_;
    Gmm1SwigluScratch gmm1SwigluScratch_;
};

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
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
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    gmm2PingPongIdx_ = 0;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(GMM1_TILE_M)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
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
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t *)params_.expertTokenNumsOutGmAddr);
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
    // 每个 win 槽位再追加 32B 存 count(源卡 GatherAndSendExpertMasks 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    mxQuantScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_ * static_cast<uint32_t>(sizeof(QuantScaleOutType)),
                             static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenScaleAlignBytes_ = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
    if constexpr (TopkWeightsPrefetch) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmToEpilogueFlag_ = reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.flagGmmToEpiloguePtr) +
                             static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    }

    dispatchPrepareContext_ = {
        {aivCoreIdx_, blockAivNum_}, {rankId_, worldSize_, moeExpertPerRank_}, {static_cast<int32_t>(m_), topK_, k_}};
    gmm1PipelineContext_ = {
        {{blockIdx_, blockNum_},
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
         A_ELEMS_PER_BYTE},
        {{blockIdx_, blockNum_},
         {blockIdx_, aicNum_},
         moeExpertPerRank_,
         moeExpertPerRank_,
         k_,
         static_cast<uint32_t>(hiddenDim_),
         dispatchFlagSlotsPerExpert_,
         INT_CACHELINE,
         tilingData->maxTilesPerExpert,
         tilingData->groupedMatmulMode,
         A_ELEMS_PER_BYTE,
         isPerExpertWeightTensor_}};
    gmm1PipelineArgs_ = {
        {::winRankAddr_,
         params_.peermemInfo.maskRecvPtr,
         params_.workspaceInfo.expertRevTokenNumsPtr,
         params_.workspaceInfo.metaInfoPtr,
         params_.workspaceInfo.cumsumInfoPtr,
         params_.workspaceInfo.dispatchRevDataPtr,
         params_.workspaceInfo.dispatchRevScalePtr,
         params_.workspaceInfo.flagDispatchToGmm1Ptr,
         params_.workspaceInfo.flagSendCntCalToUpdParamsPtr},
        {params_.workspaceInfo.expertRevTokenNumsPtr,
         params_.workspaceInfo.dispatchRevDataPtr,
         params_.workspaceInfo.dispatchRevScalePtr,
         params_.workspaceInfo.gmm1MmadResPtr,
         moeWeightTensorListAddrs_.weight1,
         moeWeightTensorListAddrs_.weightScales1,
         params_.workspaceInfo.flagDispatchToGmm1Ptr,
         params_.workspaceInfo.flagSendCntCalToUpdParamsPtr,
         params_.workspaceInfo.gmm1TileStatusPtr,
         reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_)}};
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
                       reinterpret_cast<GM_ADDR>(gmmToEpilogueFlag_),
                       tilingData->clampLimit,
                       tilingData->actMode,
                       tilingData->actSubMode,
                       tilingData->activationAlpha,
                       tilingData->activationBeta};
    const SendMaskBufferConfig &sendMaskBufferConfig = aivCoreIdx_ < tilingData->sendMaskCoreCountWithExtraExpert ?
                                                           tilingData->sendMaskConfigForCoreWithExtraExpert :
                                                           tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    sendMaskArgs_ = {params_.expertIdxGmAddr, ::winRankAddr_, maskAlignSize_,
                     maskSlotSize_, maskWinOffset_, sendMaskBufferConfig};
    int32_t resetFlagNum =
        static_cast<int32_t>(CalcMegaMoeFlagWorkspaceSize(params_.tilingData) / sizeof(int32_t));
    int32_t gmm2CombineSyncCounterNum = 0;
    int32_t sharedExpertGmm2TileCounterNum =
        static_cast<int32_t>(Ops::Base::CeilDiv(m_, GMM1_TILE_M) * sharedExpertNum_ *
                             static_cast<uint64_t>(INT_CACHELINE));
    resetWorkspaceArgs_ = {params_.workspaceInfo.flagSwiGluToGmm2Ptr,
                           params_.workspaceInfo.gmm2CombineSyncCounterPtr,
                           params_.workspaceInfo.sharedExpertGmm2TileCounterPtr,
                           params_.workspaceInfo.gmm1TileStatusPtr,
                           resetFlagNum,
                           gmm2CombineSyncCounterNum,
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
                                A_ELEMS_PER_BYTE};
}

// ======================================================================================
// SendAndQuantBuffInit：单核 mask/reset/quant/shared-prepare 模块使用的 buffer 申请。
//   shared prepare 复用 quant 输出双 buffer；reset 封顶 DISPATCH_RESET_BATCH。
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::SendMaskBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    SendMaskBufferConfig bufferConfig{};
    if constexpr (g_coreType == AIC) {
        return bufferConfig;
    }

    // 与 route batch 无关的固定占用
    uint64_t totalFlagInt32 = static_cast<uint64_t>(resetWorkspaceArgs_.flagNum);
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint64_t combineCounterNum = static_cast<uint64_t>(resetWorkspaceArgs_.gmm2CombineSyncCounterNum);
        totalFlagInt32 = totalFlagInt32 > combineCounterNum ? totalFlagInt32 : combineCounterNum;
    }
    uint64_t sharedCounterNum = static_cast<uint64_t>(resetWorkspaceArgs_.sharedExpertGmm2TileCounterNum);
    totalFlagInt32 = totalFlagInt32 > sharedCounterNum ? totalFlagInt32 : sharedCounterNum;
    if constexpr (TopkWeightsPrefetch) {
        uint64_t statusElementCount =
            (static_cast<uint64_t>(resetWorkspaceArgs_.gmm1StatusExpertCapacity) *
                 static_cast<uint64_t>(resetWorkspaceArgs_.maxTilesPerExpert) +
             1U) *
            INT_CACHELINE;
        totalFlagInt32 = totalFlagInt32 > statusElementCount ? totalFlagInt32 : statusElementCount;
    }
    uint32_t resetElementCountPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetElementCountPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetElementCountPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);

    uint32_t mxTempTensorSize = 2 * 1024;
    uint32_t xOutTensorSize = quantProcessArgs_.quantTokenScaleAlignBytes;
    uint32_t xInAlignSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t expertPerCoreMax = Ops::Base::CeilDiv(worldSize_ * moeExpertPerRank_, blockAivNum_);
    uint32_t sendCntAccSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(expertPerCoreMax * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));

    // 必须与 host SetAdaptiveBufferConfigs 的 quotient/remainder 分核保持一致。GatherAndSendExpertMasks 按
    // expertId = aivCoreIdx_ + ownedIdx * blockAivNum_ 遍历，因此前 remainder 个 core 多处理一个 expert。
    bufferConfig = sendMaskArgs_.bufferConfig;
    int32_t routeItemsPerBatch = bufferConfig.routeItemsPerBatch;

    // 按既定顺序落地址。routeItemsPerBatch 按 256 个 item 对齐，因此两个 int32 tensor 均天然满足 256B 对齐。
    uint32_t topkIdsTensorAddr = 0;
    uint32_t topkIdsTensorSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));
    sendMaskScratch_.topkIdsTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));

    uint32_t resetAddrActual = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetAddrActual, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    resetWorkspaceArgs_.resetBatchElementCount = resetBatchElementCount_;

    uint32_t mxTempTensorAddr = resetAddrActual + resetTensorSize;
    quantScratch_.mxTempTensor =
        LocalTensor<uint16_t>(TPosition::VECCALC, mxTempTensorAddr, mxTempTensorSize / sizeof(uint16_t));

    uint32_t xOutTensorAddr1 = mxTempTensorAddr + mxTempTensorSize;
    quantScratch_.xOutTensor0 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr1, xOutTensorSize / sizeof(ActivationType));
    sharedExpertPrepareScratch_.copyBuffer0 = quantScratch_.xOutTensor0;
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    quantScratch_.xOutTensor1 =
        LocalTensor<ActivationType>(TPosition::VECCALC, xOutTensorAddr2, xOutTensorSize / sizeof(ActivationType));
    sharedExpertPrepareScratch_.copyBuffer1 = quantScratch_.xOutTensor1;

    uint32_t xInAlignAddr1 = xOutTensorAddr2 + xOutTensorSize;
    quantScratch_.xInTensor0 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr1, xInAlignSize / sizeof(bfloat16_t));
    uint32_t xInAlignAddr2 = xInAlignAddr1 + xInAlignSize;
    quantScratch_.xInTensor1 =
        LocalTensor<bfloat16_t>(TPosition::VECCALC, xInAlignAddr2, xInAlignSize / sizeof(bfloat16_t));

    uint32_t sendMaskAddr = xInAlignAddr2 + xInAlignSize;
    uint32_t sendGatherOutSize = static_cast<uint32_t>(routeItemsPerBatch) * static_cast<uint32_t>(sizeof(int32_t));

    uint32_t sendMaskTotalBytes = static_cast<uint32_t>(bufferConfig.bufferCount) * bufferConfig.bufferBytes;
    sendMaskScratch_.sendMaskTensor = LocalTensor<uint8_t>(TPosition::VECCALC, sendMaskAddr, sendMaskTotalBytes);
    uint32_t sendGatherOutAddr = sendMaskAddr + sendMaskTotalBytes;
    sendMaskScratch_.sendGatherOutTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr, sendGatherOutSize / sizeof(int32_t));
    uint32_t sendCntAccAddr = sendGatherOutAddr + sendGatherOutSize;
    sendMaskScratch_.sendCntAccTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntAccAddr, sendCntAccSize / sizeof(int32_t));
    return bufferConfig;
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各路由专家收到的token总数输出（不包含共享专家）
// ==================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 恢复
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(tokenDispatchScratch_.cumsumInfoTensor,
                    tokenDispatchScratch_.cumsumInfoGlobalTensor,
                    {1U, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                    {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    tokenDispatchScratch_.expertTokenNumsOutTensor.SetValue(
        0, tokenDispatchScratch_.cumsumInfoTensor.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < static_cast<int32_t>(moeExpertPerRank_); expertIdx++) {
        int32_t cur =
            tokenDispatchScratch_.cumsumInfoTensor.GetValue(
                expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = tokenDispatchScratch_.cumsumInfoTensor.GetValue(
            (expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        tokenDispatchScratch_.expertTokenNumsOutTensor.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, tokenDispatchScratch_.expertTokenNumsOutTensor, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// =====================================================================================================
// UpdateGmm2GroupParams：更新当前expertIdx的problemShape，偏移掉本卡前侧专家收到的cnt数
// ----------------------------------------------------------------------------------------------------
//   Phase 1: 根据problemShape中的M(前一个专家收到的count数)，偏移计算baseOffset中gmm1与gmm2的左右矩阵偏移；
//   Phase 2: 更新当前专家id收到的count数;
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGmm2GroupParams(ExpertLoopState &state,
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

    uint64_t offsetInCnt = expertIdx * INT32_PER_256B * aicNum_ + INT32_PER_256B * blockIdx_;
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(
        gmm1SwigluScratch_.expertRevNumsGlobalTensor[offsetInCnt]);
    Get<M_VALUE>(state.problemShape) = gmm1SwigluScratch_.expertRevNumsGlobalTensor.GetValue(offsetInCnt);

    if (Get<M_VALUE>(state.problemShape) == 0) {
        return false;
    }
    return true;
}

// =====================================================================================================
// UpdateSharedGroupParams：共享专家专用，M 恒为 m_，无 flag 等待与 DCache 操作。
// =====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateSharedGroupParams(ExpertLoopState &state,
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
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                            const ExpertLoopState &state,
                                                                            uint32_t expertIdx)
{
    if constexpr (Mode == AddrUpdateMode::GMM1) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4) {
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
                Get<IDX_FLAG_OFFSET>(state.baseOffset),
                0L,
                0L,
                0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT) {
            gmmAddrInfo.gmm2OutGlobal =
                params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
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
        if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
            uint64_t expertSyncSlotOffset = static_cast<uint64_t>(Get<IDX_FLAG_OFFSET>(state.baseOffset)) *
                                            params_.tilingData->combineSyncSlotCountPerExpert;
            gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t *)params_.workspaceInfo.gmm2CombineSyncCounterPtr +
                                                 expertSyncSlotOffset * static_cast<uint64_t>(INT_CACHELINE);
        }
    }
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t *)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                   Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t *)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                     Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享专家 GMM2 地址来自 shared* workspace，flags 为 nullptr。
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateSharedGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
                                                                                  const ExpertLoopState &state,
                                                                                  uint32_t sharedExpertIdx)
{
    if constexpr (Mode == AddrUpdateMode::GMM2) {
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
        // tile counter: 每个 shared expert 独立一组 slot, 用 expertBeforeCnt/m_ 算出 sharedIdx
        uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, static_cast<uint32_t>(GMM1_TILE_M));
        uint32_t sharedIdx = static_cast<uint32_t>(state.expertBeforeCnt) / m_;
        gmmAddrInfo.sharedExpertGmm2TileCounter =
            reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.sharedExpertGmm2TileCounterPtr) +
            static_cast<uint64_t>(sharedIdx) * tokenGroupCount * static_cast<uint64_t>(INT_CACHELINE);
    }
    gmmAddrInfo.swigluToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        gmmAddrInfo.gmmToEpilogueFlag = gmmToEpilogueFlag_;
    }
}

// =============================================
// InitCombineBuffers：初始化 Combine 所需的 buffer 大小
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t tokenStorageBytes = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_256));
        uint32_t storedScaleBytes = Ops::Base::CeilAlign(nScale, 2U);
        // 下面两个只依赖 k_ 的量提成员, 供 ProcessCombine 每 expert 复用(原先每次调用重算)
        combineQuantTokenSizeBytes_ =
            Ops::Base::CeilAlign(tokenStorageBytes + storedScaleBytes, static_cast<uint32_t>(ALIGN_32));
        gmm2NTilesPerGroup_ = Ops::Base::CeilDiv(k_, L1_TILE_N);
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + combineQuantTokenSizeBytes_;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

/*
 * 主路径 combine-quant 的 AIV 后处理。
 * group 数不超过 logical core 数时，多核协作处理一个 group；group 数更多时，
 * logical core c 依次处理 c、c + logicalCoreCount、... 对应的 group。
 */
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessCombine(const GMMAddrInfo &gmmAddrInfo,
                                                                        const ExpertLoopState &gmm2State,
                                                                        uint32_t expertIdx)
{
    uint32_t expertTokenCount = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(expertTokenCount, COMBINE_TOKEN_GROUP_SIZE);

    // generic 路径的每个 AIV 都是 logical core；A8W4/A4W4 仅 subBlockIdx=1 参与并按物理核对映射。
    uint32_t logicalCoreId = aivCoreIdx_;
    uint32_t logicalCoreCount = blockAivNum_;
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        if (subBlockIdx_ != 1) {
            return; // 配对路径下仅 sub==1 的核参与 combine 后处理，sub==0 直接退出
        }
        logicalCoreId = aivCoreIdx_ / 2;
        logicalCoreCount = blockAivNum_ / 2;
    }

    uint32_t firstAssignedGroup = 0;
    uint32_t assignedGroupStride = 0;
    uint32_t coreIndexWithinGroup = 0;
    uint32_t coresAssignedToGroup = 0;
    MegaMoeImpl::ComputeCombineGroupsForCore(logicalCoreId, tokenGroupsThisExpert, logicalCoreCount, firstAssignedGroup,
                                             assignedGroupStride, coreIndexWithinGroup, coresAssignedToGroup);

    for (uint32_t groupIndex = firstAssignedGroup; groupIndex < tokenGroupsThisExpert;
         groupIndex += assignedGroupStride) {
        // 多核协作时每个 logical core 有独立 slot；一核处理多 group 时每个 group 有独立 slot。
        uint32_t syncSlotIndex = tokenGroupsThisExpert <= logicalCoreCount ? logicalCoreId : groupIndex;
        __gm__ int32_t *syncCounterAddress =
            MegaMoeImpl::GetCombineSyncCounterAddress(gmmAddrInfo.gmm2CombineSyncCounter, syncSlotIndex);
        while (AscendC::ReadGmByPassDCache(syncCounterAddress) != gmm2NTilesPerGroup_) {
            int64_t waitStartCycle = AscendC::GetSystemCycle();
            while (AscendC::GetSystemCycle() - waitStartCycle < 100) {
            }
        }

        uint32_t groupTokenStart = groupIndex * COMBINE_TOKEN_GROUP_SIZE;
        uint32_t groupTokenCount = COMBINE_TOKEN_GROUP_SIZE < expertTokenCount - groupTokenStart ?
                                       COMBINE_TOKEN_GROUP_SIZE :
                                       expertTokenCount - groupTokenStart;
        uint32_t tokensPerCore = Ops::Base::CeilDiv(groupTokenCount, coresAssignedToGroup);
        uint32_t tokenOffsetWithinGroup = coreIndexWithinGroup * tokensPerCore;
        // tail group 的 token 可能少于协作核数，部分核不分配 token。
        if (tokenOffsetWithinGroup >= groupTokenCount) {
            continue;
        }
        uint32_t tokenCountForCore = groupTokenCount - tokenOffsetWithinGroup;
        tokenCountForCore = tokenCountForCore < tokensPerCore ? tokenCountForCore : tokensPerCore;

        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> metaInfoTensor =
            LocalTensor<int32_t>(TPosition::VECIN, offset, tokenCountForCore * META_INFO_SIZE);
        offset += tokenCountForCore * META_INFO_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> metaInfoGm;
        metaInfoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            params_.workspaceInfo.metaInfoPtr +
            (gmm2State.expertBeforeCnt + groupTokenStart + tokenOffsetWithinGroup) * META_INFO_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(metaInfoTensor, metaInfoGm, tokenCountForCore * META_INFO_SIZE);
        PipeBarrier<PIPE_MTE2>();
        MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t>(
            groupTokenStart + tokenOffsetWithinGroup, tokenCountForCore, k_, expertIdx, rankId_,
            gmmAddrInfo.gmm2OutGlobal, params_, metaInfoTensor, combineUbTensorSize_, offset,
            combineQuantTokenSizeBytes_);
    }
}

// ===============================================================
// UnpermuteLoadWeights：加载一个 token batch 的权重到 UB
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteLoadWeights(int32_t coreOffset,
                                                                              int32_t batchTokenOffset,
                                                                              int32_t batchTokenCount,
                                                                              LocalTensor<bfloat16_t> &tempLocal)
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
// UnpermuteProcessToken：单个 token 的 per-expert 累加
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteProcessToken(
    int32_t tokenIdx, int32_t localIdx, const GlobalTensor<bfloat16_t> &expandedX,
    const UnpermuteBufferConfig &bufferConfig)
{
    for (int32_t expId = 0; expId < topK_; ++expId) {
        // Routed and shared expert results form one continuous accumulation-input sequence in the dynamic ring.
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
            uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
            // Scale starts after the 256B-aligned FP8 token region. The complete record is
            // 32B aligned and can therefore be moved through a BF16 tensor without truncation.
            uint32_t quantTokenElementCount = combineQuantTokenSizeBytes_ / sizeof(bfloat16_t);
            uint64_t routeIndex = static_cast<uint64_t>(tokenIdx) * topK_ + expId;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[routeIndex * quantTokenElementCount], quantTokenElementCount);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            using Fp8Type =
                typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
            MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_,
                                                                  fp32ScaleTensor_, nScale, k_);
        }
        // GetValue 在 Scalar 流水读取 expScale；两条反量化路径汇合后统一等待，再由 Vector 流水消费。
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
template <TemplateMegaMoeTypeClass>
__aicore__ inline typename MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBufferConfig
MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit()
{
    // 必须与 host SetAdaptiveBufferConfigs 对 TilingByCore(m_, ..., align=1) 的完整 chunk/tail chunk
    // 推导保持一致。coreLen 为 0 的非活跃 core 已在 Unpermute 中提前返回，不会读取 tail 配置。
    UnpermuteBufferConfig bufferConfig = aivCoreIdx_ < params_.tilingData->unpermuteFullTokenChunkCoreCount ?
                                             params_.tilingData->unpermuteConfigForFullTokenChunk :
                                             params_.tilingData->unpermuteConfigForTailTokenChunk;

    uint32_t bf16ScaleBufAlign = 0;
    uint32_t fp32ScaleBufAlign = 0;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(bfloat16_t) * DEQUANT_BF16_SCALE_EXPANSION),
                                 static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign =
            Ops::Base::CeilAlign(static_cast<uint32_t>(scaleNum * sizeof(float) * DEQUANT_FP32_SCALE_EXPANSION),
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

    // weight buffer（在 scale 之前，与 master 顺序一致）
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
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum × sizeof(bfloat16_t) × DEQUANT_BF16_SCALE_EXPANSION
        bf16ScaleTensor_ =
            LocalTensor<bfloat16_t>(TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum × sizeof(float) × DEQUANT_FP32_SCALE_EXPANSION
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }

    return bufferConfig;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UnpermuteSharedExpert(
    int32_t tokenIdx, int32_t localIdx, const UnpermuteBufferConfig &bufferConfig)
{
    GlobalTensor<bfloat16_t> sharedResult;
    sharedResult.SetGlobalBuffer((__gm__ bfloat16_t *)params_.workspaceInfo.sharedExpertResultPtr);
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / GMM1_TILE_M;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, GMM1_TILE_M);
    uint64_t sharedExpertStride =
        static_cast<uint64_t>(tokenGroupCount) * static_cast<uint64_t>(INT_CACHELINE);
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
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
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
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
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
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 swiglu flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo,
                                                                                const ExpertLoopState &state,
                                                                                uint32_t expertIdx,
                                                                                uint32_t &startBlockIdx,
                                                                                int32_t &vecSetSyncCom,
                                                                                int32_t &gmTileSequence)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        BlockJobContext blockJob{blockIdx_, blockNum_};
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode, SwigluQuantOutType, Weight1Type, bfloat16_t,
                                             QuantScaleOutType, QuantScaleOutType, GMM1_TILE_M,
                                             TopkWeightsPrefetch && !IsShared, IsShared>(
            params_, state.problemShape, gmmAddrInfo, startBlockIdx, gmTileSequence, blockJob, state.expertBeforeCnt,
            gmm2PingPongIdx_);
    } else {
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                      QuantScaleOutType, true, false, GMM1_TILE_M, TopkWeightsPrefetch && !IsShared,
                                      IsShared>(params_, state.problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom,
                                                state.expertBeforeCnt, gmm2PingPongIdx_);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType,
                                      QuantScaleOutType, false, false, GMM1_TILE_M, TopkWeightsPrefetch && !IsShared,
                                      IsShared>(params_, state.problemShape, gmmAddrInfo, startBlockIdx, vecSetSyncCom,
                                                state.expertBeforeCnt, gmm2PingPongIdx_);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV && !IsShared) {
        ProcessCombine(gmmAddrInfo, state, expertIdx);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ProcessSharedExpertGmm2(const TupleShape &initShape,
                                                                                 const BlockOffset &initOffset,
                                                                                 int32_t &gmTileSequence)
{
    gmm2NTilesPerGroup_ = Ops::Base::CeilDiv(k_, L1_TILE_N);
    GMMAddrInfo sharedGmm2AddrInfo;
    ExpertLoopState sharedGmm2State{initShape, initOffset, 0};
    uint32_t sharedStartBlockIdx = 0;
    int32_t vecSetSyncCom = 0;
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (!UpdateSharedGroupParams(sharedGmm2State, sharedIdx)) {
            continue;
        }
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State, sharedIdx);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State, sharedIdx, sharedStartBlockIdx,
                                     vecSetSyncCom, gmTileSequence);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();

    // Phase 1: 量化 + 共享专家输入拆分 + mask/reset (AIV)
    QuantizeLocalTokens<QuantMode, QuantOutType, ActivationType, TopkWeightsType, TopkWeightsPrefetch>(
        dispatchPrepareContext_, quantProcessArgs_, quantScratch_);
    GatherAndSendExpertMasks(dispatchPrepareContext_, sendMaskArgs_, sendMaskScratch_);
    ResetDispatchWorkspace<false, TopkWeightsPrefetch>(
        dispatchPrepareContext_, resetWorkspaceArgs_, resetTensor_);
    if (sharedExpertNum_ > 0) {
        PrepareSharedExpertInput<ActivationType, QuantScaleOutType>(
            dispatchPrepareContext_, sharedExpertPrepareArgs_, sharedExpertPrepareScratch_);
    }
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>(); // aic需要等待flag位reset清理完成

    // Phase 1.5: 共享专家单 block GMM1+SwiGLU（前移，在路由专家之前执行）
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t gmTileSequence = 0;
    if (sharedExpertNum_ > 0) {
        uint32_t sharedGmm1StartBlockIdx = 0U;
        int32_t sharedGmm1VecSetSyncCom = 0;
        uint16_t sharedGmm1PingPongIdx = 0U;
        SharedExpertGmm1SwigluState sharedGmm1State{
            sharedGmm1StartBlockIdx, sharedGmm1VecSetSyncCom, gmTileSequence, sharedGmm1PingPongIdx};
        ControlSharedExpertGmm1SwigluPipeline<QuantOutType, ActivationType, Weight1Type, SwigluQuantOutType,
                                              QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M>(
            sharedGmm1Context_, sharedGmm1Args_, params_, sharedEpilogueOp_, sharedGmm1State);
    }

    CrossRankSyncInWorldSize();

    // 2.本卡专家接收数据dispatch & GroupMatmul1 & SwigluQuant
    uint32_t gmm1StartBlockIdx = 0U;
    int32_t gmm1VecSetSyncCom = 0;
    uint16_t gmm1PingPongIdx = 0U;
    DispatchGmm1SwigluState gmm1PipelineState{
        gmm1StartBlockIdx, gmm1VecSetSyncCom, gmTileSequence, gmm1PingPongIdx};
    ControlDispatchGmm1SwigluPipeline<QuantOutType, ActivationType, Weight1Type, SwigluQuantOutType,
                                      QuantScaleOutType, ENABLE_A8W4, GMM1_TILE_M, EPILOGUE_TILE_M,
                                      TopkWeightsPrefetch>(
        gmm1PipelineContext_, gmm1PipelineArgs_, params_, epilogueOp_, tokenDispatchScratch_,
        gmm1SwigluScratch_, gmm1PipelineState);
    if constexpr (g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
    }
    // prefetch 路径AIV1 dispatch metaInfo 操作与后续AIV0使用需要保证同步
    if constexpr (TopkWeightsPrefetch) {
        SyncAll<false>();
    }

    // 3. 本卡专家接收数据GroupMatmul2 & Combine
    // MegaMoe currently runs GMM1 and GMM2 on the same full block set, so the
    // GMM1 scheduler state is already normalized for GMM2.
    uint32_t gmm2StartBlockIdx = gmm1PipelineState.startBlockIdx;
    int32_t vecSetSyncCom = 0;
    GMMAddrInfo gmm2AddrInfo;
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    InitCombineBuffers();

    for (uint32_t expertIdx = 0; expertIdx < moeExpertPerRank_; expertIdx++) {
        if (!UpdateGmm2GroupParams(gmm2State, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::GMM2>(gmm2AddrInfo, gmm2State, expertIdx);
        GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx, gmm2StartBlockIdx, vecSetSyncCom, gmTileSequence);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(vecSetSyncCom, gmm2PingPongIdx_);
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
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

} // namespace MegaMoeImpl
#undef TemplateMegaMoeTypeClass
#undef TemplateMegaMoeTypeFunc
#endif
