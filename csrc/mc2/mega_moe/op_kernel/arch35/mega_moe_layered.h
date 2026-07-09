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
#include "../../common_utils/mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"
#include "moe_distribute_dispatch_v2/quantize_functions.h"

using namespace AscendC;

namespace MegaMoeImpl {
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t>;

// 预留：XType OutputType TopkWeightsType Weight1Type
#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, \
                                  typename Weight1Type, int32_t QuantMode, int32_t CombineQuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, CombineQuantMode

template <TemplateMegaMoeTypeClass>
class MegaMoeLayered {
public:
    template <int32_t QM> struct QuantTraits { using OutType = fp8_e4m3fn_t; };
    template <> struct QuantTraits<E5M2_QUANT> { using OutType = fp8_e5m2_t; };
    template <> struct QuantTraits<E2M1_QUANT> { using OutType = fp4x2_e2m1_t; };
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType = typename std::conditional<
        Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value, uint8_t, QuantOutType>::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    struct ExpertLoopState {
        TupleShape problemShape;
        BlockOffset baseOffset;
        // Rows before the current expert, kept per cursor for dispatch/GMM prefetch state split.
        uint32_t expertBeforeCnt = 0;
    };
    __aicore__ inline MegaMoeLayered() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void SendAndQuantBuffInit();
    __aicore__ inline void UnpermuteBuffInit();
    __aicore__ inline void ResetFlagList();
    __aicore__ inline void ResetGmm2CombineSyncCounters();
    __aicore__ inline void SendMaskCal();
    __aicore__ inline void SendCntCal(int32_t localExpertId, uint64_t& sendCnt);
    __aicore__ inline void TripleInfoCalAndDispatch(GMMAddrInfo &gmmAddrInfo, int32_t localExpertId);
    template <AddrUpdateMode Mode>
    __aicore__ inline bool UpdateGroupParams(ExpertLoopState &state, uint32_t expertIdx,
        uint64_t sendCnt = 0);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State,
        uint32_t expertIdx);
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyGMToGMPerToken(int32_t rowDstOffsetInCore, int32_t remoteRankIdx,
        int32_t copyStartIdx, int32_t copyNum);
    __aicore__ inline void ResetDispatchState();
    __aicore__ inline void DispatchTokenToRmtServer(int32_t localExpertId);
    __aicore__ inline void DispatchComputeKernel(int32_t localExpertId, uint32_t realComputeCoreNum,
        int32_t roundTag);
    __aicore__ inline void DispatchComputeToken(GlobalTensor<int32_t> &topkIdsGlobal,
        int32_t localExpertId, uint32_t tokenIdx, uint32_t tokenStart, uint32_t dedupWordsPerServer);
    __aicore__ inline void DispatchSendKernel(uint32_t senderCoreStart, uint32_t senderCoreNum,
        uint32_t realComputeCoreNum, int32_t roundTag);
    __aicore__ inline void DispatchToken(uint32_t targetServerForSender, int32_t slot);
    __aicore__ inline int32_t UpdateDispatchTokenCnt(uint32_t targetServerForSender, int32_t cursor,
        int32_t endCount);
    __aicore__ inline bool DispatchSyncWithCompute(uint32_t realComputeCoreNum, int32_t roundTag);
    __aicore__ inline void ReceiveTokenFromRmtServer(uint32_t relayRank, uint64_t remoteCopyOffset,
        int32_t bufferIdx, uint32_t copyInNum);
    __aicore__ inline void QuantTokenToWorkspaceRecord(uint32_t tokenIdx, GM_ADDR recordAddr);
    __aicore__ inline uint64_t SendWorkspaceServerOffset(uint32_t targetServer);
    __aicore__ inline uint64_t RelayTokenOffset(uint32_t sourceServer, uint32_t tokenId);
    __aicore__ inline void GroupMatmulWithSwigluQuant(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state);
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state,
        uint32_t expertIdx);

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    Params params_{};

    GlobalTensor<int32_t> swigluToGmm2FlagGm_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    GlobalTensor<int32_t> tripleGlobalTensor_;
    GlobalTensor<int32_t> expertRevNumsGlobalTensor_;
    // A8W4 路径下 GroupMatmulSwigluQuant 会覆盖 V1 UB，导致 UB 上跨 expert 的状态
    // 无法保持。cumsumInfoGlobalTensor_ 作为 cumsum 数据的 GM 持久备份：
    // SendCntCal 中 Load → 计算 → Store；TripleInfoCalAndDispatch/ExpertTokenNumCopyOut 从 GM 恢复。
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
    uint32_t expertPerRank_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t vecSetSyncCom_ = 0;
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
    uint32_t ubBufferUsedAddr_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;
    uint64_t sendTotalNum_ = 0;
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;    // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint64_t maskWinOffset_ = 0;   // maskRecvPtr 相对 win 基址(rankSyncInWorldPtr)的偏移
    uint64_t quantWinOffset_ = 0;  // quantTokenScalePtr 相对 win 基址的偏移
    uint64_t dispatchWinOffset_ = 0;  // peermemInfo dispatchRecivePtr 相对 URMA win 基址的偏移
    uint32_t relayRecordBytes_ = 0;
    uint64_t sendWorkspaceRecordBytes_ = 0;
    uint64_t sendWorkspaceServerBytes_ = 0;
    uint64_t cumsumRevCntInRank_ = 0;
    int32_t compareCount_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）

    static constexpr uint32_t A_ELEMS_PER_BYTE = Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;
    static constexpr uint32_t B_ELEMS_PER_BYTE = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value ? 2U : 1U;
    // ENABLE_A8W4: A8W8 路径（fp8 act + fp4 w1），GMM1 使用 A8W4 prologue（W4→W8 + MMAD）。
    static constexpr bool ENABLE_A8W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp8_e4m3fn_t>::value;
    // ENABLE_A4W4: A4W4 路径（fp4 act + fp4 weight），GMM2 复用 A8W4 prologue。
    //             a4w4 场景下 GMM1 走 generic a4w4、GMM2 走 a8w4，避免两段都用 a4w4 导致精度损失过大。
    static constexpr bool ENABLE_A4W4 = Std::IsSame<Weight1Type, fp4x2_e2m1_t>::value &&
                                            Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value;
    static constexpr int32_t DISPATCH_BUFFER_NUM = 6;
    static constexpr uint32_t SEND_DEDUP_MASK_UB_BYTES = 8U * 1024U;
    static constexpr uint32_t SEND_DEDUP_MASK_BITS_PER_WORD = 32U;
    static constexpr uint32_t SEND_SCAN_WINDOW = 32U;
    static constexpr uint32_t MX_QUANT_TEMP_UB_BYTES = 2U * 1024U;
    static constexpr uint32_t MAX_CORENUM_USE_SEND = 8U;
    LocalTensor<int32_t> topkIndexTensor_;
    LocalTensor<uint8_t> gatherMaskTensor_;
    LocalTensor<uint32_t> gatherMaskInt32Tensor_;
    LocalTensor<int32_t> expertTokenCntTensor_;
    LocalTensor<int32_t> validTopkIndexTensor_;
    LocalTensor<int32_t> cumsumInfoTensor_;
    LocalTensor<ActivationType> copyTmpTensors_[DISPATCH_BUFFER_NUM]; // 6-buffer 软流水：占满 EVENT_ID0..EVENT_ID5。
    LocalTensor<int32_t> tripleTensor_;
    LocalTensor<bfloat16_t> xInTensor1_;
    LocalTensor<bfloat16_t> xInTensor2_;
    LocalTensor<ActivationType> xOutTensor1_;
    LocalTensor<ActivationType> xOutTensor2_;
    LocalTensor<uint16_t> mxTempTensor_;
    LocalTensor<int32_t> resetTensor_;
    LocalTensor<int32_t> topkIdsTensor_;
    LocalTensor<uint8_t> sendMaskTensor_[DOUBLE_BUFFER];  // SendMaskCal 源卡算 [mask|count] 的 ping-pong 缓冲
    LocalTensor<int32_t> sendGatherOutTensor_;            // SendMaskCal GatherMask 计 count 的废弃输出 scratch
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
    using SwigluQuantOutType = typename std::conditional<
        (QuantMode == E2M1_QUANT),
        fp8_e4m3fn_t, QuantOutType>::type;

    // SwigluQuant 输出的元素字节密度：fp4 时为 2elem/B，fp8 时为 1elem/B。
    static constexpr uint32_t C_ELEMS_PER_BYTE = Std::IsSame<SwigluQuantOutType, fp4x2_e2m1_t>::value ? 2U : 1U;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<SwigluQuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    aicNum_ = tilingData->aicNum;
    topK_ = tilingData->topK;
    sendTotalNum_ = m_ * topK_;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    blockNumPerRank_ = tilingData->blockNumPerEP;
    maxOutputSize_ = tilingData->maxOutputSize;
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(Ops::Base::CeilDiv(
        static_cast<int64_t>(maxOutputSize_), DISPATCH_WAVE_TILE_M));
    dispatchFlagSlotsPerExpert_ = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(maxWavesPerExpert_), static_cast<int64_t>(INT_CACHELINE)));
    hiddenDim_ = tilingData->hiddenDim;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
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
    params_.combineCommParams.rankId = rankId_;
    params_.combineCommParams.hcomm = &hcomm_;
    params_.combineCommParams.mc2Context = mc2Context_;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData, serverNum_);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData, A_ELEMS_PER_BYTE, serverNum_);
    params_.tilingData = tilingData;
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    expertRevNumsGlobalTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expertRevTokenNumsPtr);
    tripleGlobalTensor_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.tripleInfoPtr);
    // 每个 block 负责一个专家，cumsumInfo 中每个专家占 worldSize 个
    // int32_t 存 rank 维度的 cumsum 结果，blockIdx 决定了负责哪个专家。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumInfoGlobalTensor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(params_.workspaceInfo.cumsumInfoPtr + cumsumStride * blockIdx_));
    epilogueOp_.Init({params_.workspaceInfo.swigluQuantDataPtr, params_.workspaceInfo.swigluQuantScalePtr,
        params_.workspaceInfo.flagSwiGluToGmm2Ptr, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256,
        tilingData->clampLimit});
    // 各 win 区相对 win 基址(rankSyncInWorldPtr)的偏移; 所有卡 win 布局一致, 跨卡读写用同一偏移。
    maskWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.maskRecvPtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    dispatchWinOffset_ = static_cast<uint64_t>(params_.peermemInfo.dispatchRecivePtr -
        params_.peermemInfo.rankSyncInWorldPtr);
    // maskAlignSize_ 必与 PeermemInfo 中 maskAlignSize 公式数值一致。
    compareCount_ = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256)) / sizeof(int32_t);
    maskAlignSize_ = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_) / 8, static_cast<int64_t>(ALIGN_32));
    // 每个 win 槽位再追加 32B 存 count(源卡 SendMaskCal 同步算好), 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ = Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE),
        static_cast<uint32_t>(ALIGN_256)) * sizeof(ActivationType);
    mxQuantScaleAlignBytes_ = mxQuantScaleNumAlignPerToken_ * sizeof(uint8_t);
    mxQuantTokenScaleAlignBytes_ = Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_,
        static_cast<uint32_t>(ALIGN_32));
    relayRecordBytes_ = Ops::Base::CeilAlign(
        mxQuantTokenScaleAlignBytes_ + static_cast<uint64_t>(ALIGN_32), static_cast<uint64_t>(ALIGN_512));
    sendWorkspaceRecordBytes_ = relayRecordBytes_;
    sendWorkspaceServerBytes_ = static_cast<uint64_t>(ALIGN_32) + static_cast<uint64_t>(m_) * sendWorkspaceRecordBytes_;
}

// =================================================================================================
// DispatchBuffInit：SendCntCal & TripleInfoCalAndDispatch & ExpertTokenNumCopyOut 中使用的buffer申请
// =================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    LocalTensor<uint8_t> hcommTensor_=LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512);
    hcomm_.Init(hcommTensor_, ALIGN_512 / sizeof(uint8_t));
    // Tensor用处：SendCntCal 函数中记录本卡各专家收到的token总数；
    // Tensor大小：仅记录count值，且各专家之间复用，申请大小为32字节；
    uint32_t expertTokenCntTensorAddr = ALIGN_512;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    expertTokenCntTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr,
        expertTokenCntTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中记录本卡专家收到token count的cumsum累加值；
    // Tensor大小：worldSize_ * expertPerRank_ * sizeof(int32_t) align至32字节对齐；
    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr,
        cumsumInfoTensorSize / sizeof(int32_t));
    // Tensor用处：SendCntCal 函数中用来存储本卡上专家收到的mask+count位的buffer；
    // Tensor大小：maskSlotSize_ * worldSize_，每个maskSlotSize_中包含mask与count；
    uint32_t gatherMaskTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t gatherMaskTensorSize = maskSlotSize_ * worldSize_;
    gatherMaskTensor_ = LocalTensor<uint8_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint8_t));
    gatherMaskInt32Tensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, gatherMaskTensorAddr,
        gatherMaskTensorSize / sizeof(uint32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中GatherMask的dst Tensor；
    // Tensor大小：sendTotalNum_ * sizeof(int32_t) align至32字节对齐；
    uint32_t validTopkIndexTensorAddr = gatherMaskTensorAddr + gatherMaskTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr,
        validTopkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中GatherMask的src Tensor；
    // Tensor大小：sendTotalNum_ * sizeof(int32_t) align至32字节对齐；
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr,
        topkIndexTensorSize / sizeof(int32_t));
    // Tensor用处：TripleInfoCalAndDispatch 函数中的6个dispatch buffer, 配合EVENT_ID0..EVENT_ID5做软流水
    // Tensor大小：每块容纳token+scale，动态计算以节省UB空间
    uint32_t tokenScaleSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32));
    uint32_t COPY_TMP_BUFFER_SIZE = tokenScaleSize;
    uint32_t copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    for (int32_t index = 0; index < DISPATCH_BUFFER_NUM; ++index) {
        copyTmpTensors_[index] = LocalTensor<ActivationType>(TPosition::VECCALC,
            copyTmpBaseAddr + static_cast<uint32_t>(index) * COPY_TMP_BUFFER_SIZE,
            COPY_TMP_BUFFER_SIZE / sizeof(ActivationType));
    }
    // Tensor用处：Level1DispatchUrma 中按 (targetServer, localTokenIdx) 去重。
    uint32_t sendDedupMaskTensorAddr = copyTmpBaseAddr +
        static_cast<uint32_t>(DISPATCH_BUFFER_NUM) * COPY_TMP_BUFFER_SIZE;
    sendDedupMaskTensor_ = LocalTensor<uint32_t>(TPosition::VECCALC, sendDedupMaskTensorAddr,
        SEND_DEDUP_MASK_UB_BYTES / sizeof(uint32_t));
    Duplicate<uint32_t>(sendDedupMaskTensor_, 0, SEND_DEDUP_MASK_UB_BYTES / sizeof(uint32_t));
    // Tensor用处：QuantTokenToWorkspaceRecord 函数中用于量化计算中间区域。
    uint32_t urmaMxTempTensorAddr = sendDedupMaskTensorAddr + SEND_DEDUP_MASK_UB_BYTES;
    mxTempTensor_ = LocalTensor<uint16_t>(TPosition::VECCALC, urmaMxTempTensorAddr,
        MX_QUANT_TEMP_UB_BYTES / sizeof(uint16_t));
    // Tensor用处：QuantTokenToWorkspaceRecord 函数中用于存储量化输出和 32B meta。
    uint32_t urmaXOutTensorAddr = urmaMxTempTensorAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t urmaXOutTensorSize = mxQuantTokenScaleAlignBytes_ + static_cast<uint32_t>(ALIGN_32);
    xOutTensor1_ = LocalTensor<ActivationType>(TPosition::VECCALC, urmaXOutTensorAddr,
        urmaXOutTensorSize / sizeof(ActivationType));
    // Tensor用处：QuantTokenToWorkspaceRecord 函数中用于存储输入 token。
    uint32_t urmaXInTensorAddr = urmaXOutTensorAddr + urmaXOutTensorSize;
    uint32_t urmaXInTensorSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    xInTensor1_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, urmaXInTensorAddr,
        urmaXInTensorSize / sizeof(bfloat16_t));
    // Tensor用处：ExpertTokenNumCopyOut 函数中本卡各专家收到的tokenCnt数；
    // Tensor大小：expertPerRank_ * sizeof(int32_t) 对齐至32字节；
    uint32_t expertTokenNumsOutTensorAddr = urmaXInTensorAddr + urmaXInTensorSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(expertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    expertTokenNumsOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
        expertTokenNumsOutTensorSize / sizeof(int32_t));
    // 记录当前已被使用的ub地址，用于后续TripleInfoCalAndDispatch函数中分核后神申请tripleTensor_
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    CreateVecIndex(topkIndexTensor_, 0, (topkIndexTensorSize / sizeof(int32_t)));
    Duplicate<int32_t>(cumsumInfoTensor_, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    PipeBarrier<PIPE_ALL>();
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList localTensor申请
// ======================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    LocalTensor<uint8_t> hcommTensor_=LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
    hcomm_.Init(hcommTensor_, ALIGN_512);
    // Tensor用处：SendMaskCal 函数中搬运本卡的 topkIds；
    // Tensor大小：该Tensor在进行CompareScalar时，compareCount_要求256B对齐，申请大小256字节对齐；
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    topkIdsTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, topkIdsTensorAddr, topkIdsTensorSize / sizeof(int32_t));
    // Tensor用处：ResetFlagList中对于workSpace上的Flag位置区域进行清理；
    // Tensor大小：大小为清理区域大小均分到所有的blockAivNum_；
    // SwiGluToGmm2区 DispatchToGmm1区 SendCntCalToUpdParams区，三段在workSpace上连续；
    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    uint64_t totalFlagInt32 = static_cast<uint64_t>(expertPerRank_) *
        (static_cast<uint64_t>(INT_CACHELINE) + static_cast<uint64_t>(dispatchFlagSlotsPerExpert_) +
        static_cast<uint64_t>(INT_CACHELINE) * static_cast<uint64_t>(aicNum_));
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        int64_t tokenGroupResetSize = static_cast<int64_t>(expertPerRank_) * blockAivNum_ * INT_CACHELINE;
        totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize)
                         ? static_cast<int64_t>(totalFlagInt32) : tokenGroupResetSize;
    }
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    uint32_t resetTensorSize = Ops::Base::CeilAlign(static_cast<uint64_t>(resetNumPerCore),
        static_cast<uint64_t>(INT32_PER_256B)) * sizeof(int32_t);
    resetTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, resetTensorAddr, resetTensorSize / sizeof(int32_t));
    Duplicate<int32_t>(resetTensor_, 0, (resetTensorSize / sizeof(int32_t)));
    // Tensor用处：SendMaskCal函数中用于存储mask位；
    // Tensor大小：大小maskSlotSize_与保持一致，DOUBLE_BUFFER为2，开启双buffer；
    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] = LocalTensor<uint8_t>(TPosition::VECCALC,
            sendMaskAddr + static_cast<uint32_t>(index) * maskSlotSize_, maskSlotSize_);
    }
    // Tensor用处：SendMaskCal函数中用于GatherMask的dstTensor；
    // Tensor大小：大小为一次GatherMaksk长度compareCount_对齐到256；
    // SendMaskCal GatherMask 计 count 用的废弃输出 scratch(compareCount_ 个 int, 256B 对齐)。
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * maskSlotSize_;
    uint32_t sendGatherOutSize = Ops::Base::CeilAlign(static_cast<int64_t>(compareCount_ * sizeof(int32_t)),
        static_cast<int64_t>(ALIGN_256));
    sendGatherOutTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, sendGatherOutAddr,
        sendGatherOutSize / sizeof(int32_t));
}

// ===============================================================================================
// ResetFlagList：对本卡workSpace上对于Flag位的清理，包括flagSwiGluToGmm2Ptr & flagDispatchToGmm1Ptr
// ===============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ResetFlagList()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // workSpace Flag 清零
    // 总数 = SwiGluToGmm2(expertPerRank * INT_CACHELINE) + DispatchToGmm1(expertPerRank * dispatchFlagSlotsPerExpert_)
    //        + SendCntCalToUpdParams(expertPerRank * aicNum_ * INT_CACHELINE)
    swigluToGmm2FlagGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr);
    int32_t flagNum = static_cast<int32_t>(expertPerRank_) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_ +
        static_cast<int32_t>(INT_CACHELINE) * static_cast<int32_t>(aicNum_));
    int32_t coreLen, coreOffset;
    TilingByCore(flagNum, coreLen, coreOffset, 1);
    DataCopyExtParams rankSyncCopyParams{1U, static_cast<uint32_t>(coreLen * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID2>();
    if (coreLen != 0) {
        DataCopyPad(swigluToGmm2FlagGm_[coreOffset], resetTensor_, rankSyncCopyParams);
    }
    // combine量化模式下TokenGroupCompleteFlag清零
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        ResetGmm2CombineSyncCounters();
    }
}

// ==================================================
// ExpertTokenNumCopyOut：本卡各专家收到的token总数输出
// ==================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 恢复
    if constexpr (ENABLE_A8W4) {
        DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                {1U, static_cast<uint32_t>(worldSize_ * expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U},
                {true, 0U, 0U, 0U});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    }
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_.SetValue(0, cumsumInfoTensor_.GetValue(lastRankIdx));
    for (int32_t expertIdx = 1; expertIdx < expertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_.GetValue(expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx);
        int32_t prev = cumsumInfoTensor_.GetValue((expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx);
        expertTokenNumsOutTensor_.SetValue(expertIdx, cur - prev);
    }
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(expertPerRank_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOut_, expertTokenNumsOutTensor_, copyParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

// ======================================================================================================
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡
// ------------------------------------------------------------------------------------------------------
//   Phase 1: 本卡 topk 的搬入；
//   Phase 2: 根据专家id进行CompareScalar & GatherMask，生成mask位与count总数，doubleBuffer进行计算并进行发送；
// ======================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendMaskCal()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    // Phase 1: 加载本卡 topk 到 topkIdsTensor_ (compareCount_ 个 int, 尾部补 0)
    GlobalTensor<int32_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer((__gm__ int32_t*)params_.expertIdxGmAddr);
    Duplicate<int32_t>(topkIdsTensor_, 0, compareCount_);
    SyncFuncStatic<AscendC::HardEvent::V_MTE2, SYNC_EVENT_ID1>();
    DataCopyExtParams loadParams{1U, static_cast<uint32_t>(sendTotalNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> loadPad{false, 0U, 0U, 0U};
    DataCopyPad(topkIdsTensor_, srcGlobalTensor, loadParams, loadPad);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    // Phase 2: 逐个全局专家算 [mask|count] 推送 (ping-pong 双 buffer 软流水)
    // buffer内前 maskAlignSize_ 存mask, 末 32B 存 count
    constexpr TEventID kBufEvents[DOUBLE_BUFFER] = {EVENT_ID1, EVENT_ID2};
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        SetFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(int32_t);  // count 在槽内 int32 偏移
    DataCopyExtParams maskCopyParams{1U, static_cast<uint32_t>(maskSlotSize_), 0U, 0U, 0U};
    int32_t iter = 0;
    GlobalTensor<uint8_t> dstGlobalTensor;
    for (uint32_t curRankId = aivCoreIdx_; curRankId < totalExperts; curRankId += blockAivNum_) {
        for (uint32_t expertIdIndex = 0; expertIdIndex < expertPerRank_; ++expertIdIndex, ++iter) {
            int32_t curExpertId = curRankId * expertPerRank_ + expertIdIndex;
            TEventID eventId = kBufEvents[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint8_t> maskBuf = sendMaskTensor_[iter % static_cast<int32_t>(DOUBLE_BUFFER)];
            LocalTensor<uint32_t> maskBufU32 = maskBuf.template ReinterpretCast<uint32_t>();
            LocalTensor<int32_t> maskBufI32 = maskBuf.template ReinterpretCast<int32_t>();
            WaitFlag<AscendC::HardEvent::MTE3_V>(eventId); // 等本 buffer 上一轮 MTE3 推送完成
            CompareScalar(maskBuf, topkIdsTensor_, curExpertId, AscendC::CMPMODE::EQ, compareCount_);
            // 同步算 count: GatherMask 对 mask 取 set-bit 数(mask=sendTotalNum_ 跳过尾部 padding)。
            uint64_t sendCnt = 0;
            GatherMask(sendGatherOutTensor_, topkIdsTensor_, maskBufU32, true,
                static_cast<uint32_t>(sendTotalNum_), {1, 1, 0, 0}, sendCnt);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();   // count 标量就绪
            maskBufI32.SetValue(countWordIdx, static_cast<int32_t>(sendCnt));  // 写入槽内 count
            SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>(); // count 对 MTE3 可见(V 已被 V_S 排空)

            uint64_t srcOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                 static_cast<int32_t>(curRankId)) * static_cast<uint64_t>(maskSlotSize_);
            uint64_t dstOffset = maskWinOffset_ + static_cast<uint64_t>(expertIdIndex *
                static_cast<int32_t>(worldSize_) + static_cast<int32_t>(rankId_)) *
                static_cast<uint64_t>(maskSlotSize_);
            dstGlobalTensor.SetGlobalBuffer(
                (__gm__ uint8_t*)(params_.workspaceInfo.maskSlotPtr + srcOffset));
            if (curRankId == rankId_) {
                dstGlobalTensor.SetGlobalBuffer(
                    (__gm__ uint8_t*)(GetRankWinAddrWithOffset(rankId_, dstOffset)));
            }
            DataCopyPad(dstGlobalTensor, maskBuf, maskCopyParams);
            PipeBarrier<PIPE_ALL>();
            if (curRankId != rankId_) {
                GM_ADDR remoteDataAddr = GetRankWinAddrWithOffset(curRankId, dstOffset);
                GM_ADDR localGmAddr = params_.workspaceInfo.maskSlotPtr + srcOffset;
                hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, curRankId, rankId_), remoteDataAddr,
                    localGmAddr, maskSlotSize_);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(eventId);
        }
    }
    for (int32_t index = 0; index < static_cast<int32_t>(DOUBLE_BUFFER); ++index) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(kBufEvents[index]);
    }
    for (uint32_t curRankId = aivCoreIdx_; curRankId < totalExperts; curRankId += blockAivNum_) {
        if (curRankId == rankId_) {
            continue;
        }
        hcomm_.Drain(GetUrmaCommHandle(mc2Context_, curRankId, rankId_));
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendWorkspaceServerOffset(uint32_t targetServer)
{
    return static_cast<uint64_t>(targetServer) * static_cast<uint64_t>(sendWorkspaceServerBytes_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeTypeFunc>::RelayTokenOffset(uint32_t sourceServer,
    uint32_t tokenId)
{
    return (static_cast<uint64_t>(sourceServer) * static_cast<uint64_t>(m_) + tokenId) *
        static_cast<uint64_t>(relayRecordBytes_);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ResetDispatchState()
{
    if constexpr(g_coreType == AIC) {
        return;
    }

    for (uint32_t serverIdx = aivCoreIdx_; serverIdx < serverNum_; serverIdx += blockAivNum_) {
        __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(
            params_.workspaceInfo.dispatchL1CommPtr + SendWorkspaceServerOffset(serverIdx));
        WriteGmByPassDCache(countPtr, int32_t(0));
    }
    uint32_t sendRecordNum = serverNum_ * m_;
    for (uint32_t idx = aivCoreIdx_; idx < sendRecordNum; idx += blockAivNum_) {
        uint32_t serverIdx = idx / m_;
        uint32_t slotIdx = idx - serverIdx * m_;
        uint64_t recordOffset = SendWorkspaceServerOffset(serverIdx) + ALIGN_32 +
            static_cast<uint64_t>(slotIdx) * sendWorkspaceRecordBytes_;
        __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(
            params_.workspaceInfo.dispatchL1CommPtr + recordOffset + mxQuantTokenScaleAlignBytes_);
        WriteGmByPassDCache(metaPtr + 1, int32_t(0));
    }
    __gm__ int32_t* cursorPtr = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.dispatchCursorPtr);
    for (uint32_t serverIdx = aivCoreIdx_; serverIdx < serverNum_; serverIdx += blockAivNum_) {
        WriteGmByPassDCache(cursorPtr + serverIdx, int32_t(0));
    }
    __gm__ int32_t* donePtr = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.dispatchDonePtr);
    for (uint32_t idx = aivCoreIdx_; idx < blockNum_; idx += blockAivNum_) {
        WriteGmByPassDCache(donePtr + idx, int32_t(0));
    }
    uint32_t relayRecordNum = serverNum_ * m_;
    for (uint32_t idx = aivCoreIdx_; idx < relayRecordNum; idx += blockAivNum_) {
        uint64_t recordOffset = static_cast<uint64_t>(idx) * static_cast<uint64_t>(relayRecordBytes_);
        __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(
            params_.peermemInfo.dispatchRecivePtr + recordOffset + mxQuantTokenScaleAlignBytes_);
        WriteGmByPassDCache(metaPtr, int32_t(0));
        WriteGmByPassDCache(metaPtr + 1, int32_t(0));
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::QuantTokenToWorkspaceRecord(uint32_t tokenIdx,
    GM_ADDR recordAddr)
{
    LocalTensor<uint8_t> xOutBytesTensor = xOutTensor1_.template ReinterpretCast<uint8_t>();
    GlobalTensor<bfloat16_t> srcGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(
        params_.aGmAddr + static_cast<uint64_t>(tokenIdx) * k_ * sizeof(bfloat16_t)));
    GlobalTensor<uint8_t> workspaceDstGlobal;
    workspaceDstGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(recordAddr));

    DataCopyParams xCopyInParams = {1U, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), 0U, 0U};
    DataCopyPadParams xCopyInPadParams{true, 0, 0, 0};
    DataCopyPad(xInTensor1_, srcGlobalTensor, xCopyInParams, xCopyInPadParams);
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();

    __ubuf__ bfloat16_t* srcAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensor1_.GetPhyAddr());
    __ubuf__ uint16_t* maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxTempTensor_.GetPhyAddr());
    __ubuf__ uint16_t* halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(mxTempTensor_[Ops::Base::CeilAlign(
        mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32))].GetPhyAddr());
    __ubuf__ int8_t* outDataAddr = reinterpret_cast<__ubuf__ int8_t*>(xOutTensor1_.GetPhyAddr());
    __ubuf__ uint16_t* mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(
        xOutTensor1_[mxQuantTokenAlignBytes_].GetPhyAddr());

    Quant::ComputeMaxExp(srcAddr, maxExpAddr, k_);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
            AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, k_);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC,
            AscendC::RoundMode::CAST_RINT>(srcAddr, halfScaleAddr, outDataAddr, k_);
    }

    LocalTensor<int32_t> metaTensor = xOutBytesTensor[mxQuantTokenScaleAlignBytes_].template ReinterpretCast<int32_t>();
    metaTensor.SetValue(0, static_cast<int32_t>(tokenIdx));
    metaTensor.SetValue(1, int32_t(1));
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID1>();
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID2>();
    DataCopyPad(workspaceDstGlobal, xOutBytesTensor, {1U, mxQuantTokenScaleAlignBytes_, 0U, 0U, 0U});
    PipeBarrier<PIPE_MTE3>();
    DataCopyPad(workspaceDstGlobal[mxQuantTokenScaleAlignBytes_], xOutBytesTensor[mxQuantTokenScaleAlignBytes_],
        {1U, static_cast<uint32_t>(ALIGN_32), 0U, 0U, 0U});
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchComputeToken(
    GlobalTensor<int32_t> &topkIdsGlobal, int32_t localExpertId, uint32_t tokenIdx, uint32_t tokenStart,
    uint32_t dedupWordsPerServer)
{
    for (uint32_t topkIdx = 0; topkIdx < topK_; ++topkIdx) {
        int32_t globalExpertId = topkIdsGlobal.GetValue(tokenIdx * topK_ + topkIdx);
        if ((globalExpertId % static_cast<int32_t>(expertPerRank_)) != localExpertId) {
            continue;
        }
        uint32_t targetRank = static_cast<uint32_t>(globalExpertId) / expertPerRank_;
        uint32_t targetServer = targetRank / rankPerServer_;
        uint32_t localTokenIdx = tokenIdx - tokenStart;
        uint32_t dedupWordIdx = targetServer * dedupWordsPerServer +
            localTokenIdx / SEND_DEDUP_MASK_BITS_PER_WORD;
        uint32_t dedupBit = 1U << (localTokenIdx & (SEND_DEDUP_MASK_BITS_PER_WORD - 1U));
        uint32_t dedupWord = sendDedupMaskTensor_.GetValue(dedupWordIdx);
        if ((dedupWord & dedupBit) != 0U) {
            continue;
        }
        sendDedupMaskTensor_.SetValue(dedupWordIdx, dedupWord | dedupBit);
        uint64_t relayOffset = RelayTokenOffset(serverId_, static_cast<uint32_t>(tokenIdx));
        GM_ADDR recordAddr = GetRankWinAddrWithOffset(rankId_, dispatchWinOffset_) +
            relayOffset;
        if (targetServer != serverId_) {
            __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(
                params_.workspaceInfo.dispatchL1CommPtr + SendWorkspaceServerOffset(targetServer));
            int32_t slotIdx = AtomicAdd(countPtr, int32_t(1));
            uint64_t recordOffset = SendWorkspaceServerOffset(targetServer) + ALIGN_32 +
                static_cast<uint64_t>(slotIdx) * sendWorkspaceRecordBytes_;
            recordAddr = params_.workspaceInfo.dispatchL1CommPtr + recordOffset;
        }
        QuantTokenToWorkspaceRecord(tokenIdx, recordAddr);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchComputeKernel(
    int32_t localExpertId, uint32_t realComputeCoreNum, int32_t roundTag)
{
    uint32_t computeIdx = blockIdx_;
    uint32_t baseTokenNum = m_ / realComputeCoreNum;
    uint32_t tokenRemainder = m_ % realComputeCoreNum;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(computeIdx < tokenRemainder);
    uint32_t tokenStart = computeIdx * baseTokenNum +
        ((computeIdx < tokenRemainder) ? computeIdx : tokenRemainder);
    uint32_t tokenEnd = tokenStart + tokenNumInCore;
    uint32_t maxTokenNumPerComputeCore = Ops::Base::CeilDiv(m_, realComputeCoreNum);
    uint32_t dedupWordsPerServer = Ops::Base::CeilDiv(maxTokenNumPerComputeCore,
        SEND_DEDUP_MASK_BITS_PER_WORD);

    GlobalTensor<int32_t> topkIdsGlobal;
    topkIdsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.expertIdxGmAddr));
    for (uint32_t tokenIdx = tokenStart; tokenIdx < tokenEnd; ++tokenIdx) {
        DispatchComputeToken(topkIdsGlobal, localExpertId, tokenIdx, tokenStart, dedupWordsPerServer);
    }
    __gm__ int32_t* donePtr = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.dispatchDonePtr);
    WriteGmByPassDCache(donePtr + computeIdx, roundTag);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchToken(
    uint32_t targetServerForSender, int32_t slot)
{
    uint64_t recordOffset = SendWorkspaceServerOffset(targetServerForSender) + ALIGN_32 +
        static_cast<uint64_t>(slot) * sendWorkspaceRecordBytes_;
    GM_ADDR recordAddr = params_.workspaceInfo.dispatchL1CommPtr + recordOffset;
    __gm__ int32_t* srcMetaPtr = reinterpret_cast<__gm__ int32_t*>(
        recordAddr + mxQuantTokenScaleAlignBytes_);
    if (ReadGmByPassDCache(srcMetaPtr + 1) != int32_t(1)) {
        return;
    }
    int32_t tokenIdx = ReadGmByPassDCache(srcMetaPtr);
    uint64_t relayOffset = RelayTokenOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    uint32_t peerRank = targetServerForSender * rankPerServer_ + rankIdInServer_;
    GM_ADDR dstAddr = GetRankWinAddrWithOffset(peerRank, dispatchWinOffset_) + relayOffset;
    hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, peerRank, rankId_), dstAddr, recordAddr,
        mxQuantTokenScaleAlignBytes_);
    hcomm_.Drain(GetUrmaCommHandle(mc2Context_, peerRank, rankId_));
    hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, peerRank, rankId_), dstAddr + mxQuantTokenScaleAlignBytes_,
        recordAddr + mxQuantTokenScaleAlignBytes_, ALIGN_32);
    hcomm_.Drain(GetUrmaCommHandle(mc2Context_, peerRank, rankId_));
    SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID1>();
    WriteGmByPassDCache(srcMetaPtr + 1, int32_t(2));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline int32_t MegaMoeLayered<TemplateMegaMoeTypeFunc>::UpdateDispatchTokenCnt(
    uint32_t targetServerForSender, int32_t cursor, int32_t endCount)
{
    while (cursor < endCount) {
        uint64_t recordOffset = SendWorkspaceServerOffset(targetServerForSender) + ALIGN_32 +
            static_cast<uint64_t>(cursor) * sendWorkspaceRecordBytes_;
        GM_ADDR recordAddr = params_.workspaceInfo.dispatchL1CommPtr + recordOffset;
        __gm__ int32_t* srcMetaPtr = reinterpret_cast<__gm__ int32_t*>(
            recordAddr + mxQuantTokenScaleAlignBytes_);
        if (ReadGmByPassDCache(srcMetaPtr + 1) != int32_t(2)) {
            break;
        }
        ++cursor;
    }
    return cursor;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchSyncWithCompute(
    uint32_t realComputeCoreNum, int32_t roundTag)
{
    __gm__ int32_t* donePtr = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.dispatchDonePtr);
    for (uint32_t idx = 0; idx < realComputeCoreNum; ++idx) {
        if (ReadGmByPassDCache(donePtr + idx) < roundTag) {
            return false;
        }
    }
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchSendKernel(
    uint32_t senderCoreStart, uint32_t senderCoreNum, uint32_t realComputeCoreNum, int32_t roundTag)
{
    __gm__ int32_t* cursorPtr = reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.dispatchCursorPtr);
    uint32_t senderIdx = blockIdx_ - senderCoreStart;
    for (uint32_t targetServerForSender = senderIdx; targetServerForSender < serverNum_;
        targetServerForSender += senderCoreNum) {
        if (targetServerForSender == serverId_) {
            continue;
        }
        __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(
            params_.workspaceInfo.dispatchL1CommPtr + SendWorkspaceServerOffset(targetServerForSender));
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
                DispatchToken(targetServerForSender, slot);
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
            if (DispatchSyncWithCompute(realComputeCoreNum, roundTag) &&
                cursor == ReadGmByPassDCache(countPtr)) {
                break;
            }
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::DispatchTokenToRmtServer(int32_t localExpertId)
{
    uint32_t senderCoreLimit = (serverNum_ < MAX_CORENUM_USE_SEND) ? serverNum_ : MAX_CORENUM_USE_SEND;
    uint32_t senderCoreNum = (senderCoreLimit < blockNum_) ? senderCoreLimit : blockNum_;
    uint32_t senderCoreStart = blockNum_ - senderCoreNum;
    uint32_t realComputeCoreNum = blockNum_ - senderCoreNum;
    int32_t roundTag = localExpertId + 1;

    if (blockIdx_ < senderCoreStart) {
        DispatchComputeKernel(localExpertId, realComputeCoreNum, roundTag);
    } else {
        DispatchSendKernel(senderCoreStart, senderCoreNum, realComputeCoreNum, roundTag);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ReceiveTokenFromRmtServer(uint32_t relayRank,
    uint64_t remoteCopyOffset, int32_t bufferIdx, uint32_t copyInNum)
{
    GM_ADDR remoteRecordAddr = GetRankWinAddrWithOffset(relayRank, dispatchWinOffset_) +
        remoteCopyOffset;
    uint64_t flagOffset = static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_) + sizeof(int32_t);
    if (relayRank == rankId_) {
        __gm__ int32_t* readyFlag = reinterpret_cast<__gm__ int32_t*>(remoteRecordAddr + flagOffset);
        GmSignalWaitBarrier(readyFlag, int32_t(1));
        GlobalTensor<ActivationType> relayGlobalTensor;
        relayGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(remoteRecordAddr));
        DataCopy(copyTmpTensors_[bufferIdx], relayGlobalTensor, copyInNum);
        return;
    }
    GM_ADDR scratchAddr = params_.workspaceInfo.dispatchL2CommPtr +
        (static_cast<uint64_t>(blockIdx_) * DISPATCH_BUFFER_NUM + bufferIdx) * relayRecordBytes_;
    __gm__ int32_t* localReadyFlag = reinterpret_cast<__gm__ int32_t*>(scratchAddr + flagOffset);
    do {
        hcomm_.ReadNbi<true>(GetUrmaCommHandle(mc2Context_, relayRank, rankId_), scratchAddr + flagOffset,
            remoteRecordAddr + flagOffset, sizeof(int32_t));
        hcomm_.Drain(GetUrmaCommHandle(mc2Context_, relayRank, rankId_));
        PipeBarrier<PIPE_ALL>();
        int32_t readyValue = ReadGmByPassDCache(localReadyFlag);
        if (readyValue == int32_t(1)) {
            WriteGmByPassDCache(localReadyFlag, int32_t(0));
            break;
        }
    } while (true);
    hcomm_.ReadNbi<true>(GetUrmaCommHandle(mc2Context_, relayRank, rankId_), scratchAddr, remoteRecordAddr,
        copyInNum);
    hcomm_.Drain(GetUrmaCommHandle(mc2Context_, relayRank, rankId_));
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<ActivationType> scratchGlobalTensor;
    scratchGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(scratchAddr));
    DataCopy(copyTmpTensors_[bufferIdx], scratchGlobalTensor, copyInNum);
}

// ==================================================================================================
// SendCntCal：目标专家卡读 count 计数，得到当前专家Id收到的token总数
// --------------------------------------------------------------------------------------------------
//   Phase 1: 从本卡 win 将当前 localExpertId 的 worldSize_ 个 [mask|count] 槽位读进 gatherMaskTensor_;
//   Phase 2: 逐卡读取 count → 累加 sendCnt/cumsumRevCntInRank_, 写 cumsumInfoTensor_;
//   Phase 3: 写 expertRevNumsGlobalTensor_ + AtomicAdd 通知 AIC;
// ==================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::SendCntCal(int32_t localExpertId,
    uint64_t& sendCnt)
{
    sendCnt = 0;
    uint32_t slotWordStride = static_cast<uint32_t>(maskSlotSize_) / sizeof(uint32_t);
    uint32_t countWordIdx = static_cast<uint32_t>(maskAlignSize_) / sizeof(uint32_t);

    // Phase 1: 从本卡 win 读本 localExpert 的全部 worldSize_ 个 [mask|count] 槽位
    GlobalTensor<uint8_t> maskSrcGlobal;
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_));
    DataCopy(gatherMaskTensor_, maskSrcGlobal, worldSize_ * maskSlotSize_);

    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            // A8W4 路径下 cumsum 被 SwigluQuant 覆盖，从 GM 加载前序 expert 的 cumsum
            DataCopyPad(cumsumInfoTensor_, cumsumInfoGlobalTensor_,
                        {1U, static_cast<uint32_t>(worldSize_ * localExpertId * sizeof(int32_t)), 0U, 0U, 0U},
                        {true, 0U, 0U, 0U});
        }
    }

    SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();   // count 读取(标量)就绪
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();   // mask 供 Triple 的 GatherMask(V)就绪

    // Phase 2: 逐源卡直接读取槽内 count + cumsum
    for (int32_t calRankId = 0; calRankId < static_cast<int32_t>(worldSize_); ++calRankId) {
        int32_t perRankCnt = gatherMaskInt32Tensor_.GetValue(calRankId * slotWordStride + countWordIdx);
        sendCnt += static_cast<uint64_t>(perRankCnt);
        cumsumRevCntInRank_ += static_cast<uint64_t>(perRankCnt);
        cumsumInfoTensor_.SetValue(localExpertId * worldSize_ + calRankId, static_cast<int32_t>(cumsumRevCntInRank_));
    }
    
    // Phase 3: 写到 gm 上，并通知 AIC
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

    __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
        static_cast<uint64_t>(localExpertId) * aicNum_ * INT_CACHELINE +
        static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    AscendC::AtomicAdd(sendCntFlag, static_cast<int32_t>(1));
}

// ============================================================================
// CopyGMToGMPerToken：6-buffer 软流水，搬运对端rank数据至本专家卡
// ----------------------------------------------------------------------------
//   Phase 1: 所有 token 的三元组 (rank, tokenIndex, topkIndex) 组装写入tripleTensor_
//   Phase 2 prime: 连发 6 个 MTE2,中间不插 WaitFlag<MTE2_MTE3>, 让 MTE2 引擎同时持有 6 个跨卡读请求。
//   Phase 2 steady: 每轮做 (MTE3_out[i] + MTE2_in[i+6]), 槽位循环复用。
//   Phase 2 drain: 收尾不再发新 MTE2,只等 MTE3。
//   Phase 3: triple 三元组搬出。
// ============================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
    int32_t rowDstOffsetInCore, int32_t remoteRankIdx, int32_t copyStartIdx, int32_t copyNum)
{
    if (copyNum <= 0) {
        return;
    }
    constexpr int32_t BufferNum = 5;
    constexpr TEventID kBufEvents[BufferNum] = {EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;  // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = Ops::Base::CeilAlign(static_cast<int64_t>(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_),
        static_cast<int64_t>(ALIGN_32)); // 输入 token-scale 拼接,非紧密排列
    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleOutType> scaleRevGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(
        params_.workspaceInfo.dispatchRevDataPtr + rowDstOffsetInCore * widthA));
    scaleRevGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
        params_.workspaceInfo.dispatchRevScalePtr + rowDstOffsetInCore * widthAScale));
    uint32_t srcServer = static_cast<uint32_t>(remoteRankIdx) / rankPerServer_;
    uint32_t srcRankInServer = static_cast<uint32_t>(remoteRankIdx) % rankPerServer_;
    uint32_t relayRank = serverId_ * rankPerServer_ + srcRankInServer;

    // 预置 6 个 MTE3_MTE2 flag,Phase 2 的 prime/steady 的 WaitFlag 可立即通过。
    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    PipeBarrier<PIPE_ALL>();
    // Phase 1: token三元组信息组装(rank, tokenIndex, topkIndex)
    for (int32_t i = 0; i < copyNum; ++i) {
        int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
        int32_t tokenIndex = topkIndex / topK_;
        tripleTensor_[i * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
        tripleTensor_[i * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
        tripleTensor_[i * INT32_PER_256B].SetValue(TOPK_INDEX, topkIndex % topK_);
    }

    // 6-buffer SW 流水 MTE2 + MTE3
    // Phase 2 prime: 发出前 BufferNum 个 token 的 MTE2
    int32_t primeCount = (copyNum < BufferNum) ? copyNum : BufferNum;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = tripleTensor_[primeIdx * INT32_PER_256B].GetValue(TOKEN_ID);
        TEventID eventId = kBufEvents[primeIdx];
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, static_cast<uint32_t>(tokenIndex));
        ReceiveTokenFromRmtServer(relayRank, remoteCopyOffset, primeIdx, copyInNum);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    }
    // Phase 2 steady: MTE3[copyIdx] + issueMTE2[copyIdx + BufferNum]
    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % BufferNum;
        TEventID eventId = kBufEvents[outIdx];
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

        LocalTensor<ActivationType> tokenScalebuf = copyTmpTensors_[outIdx];
        LocalTensor<QuantScaleOutType> bufScale =
            tokenScalebuf[mxQuantTokenAlignBytes_].template ReinterpretCast<QuantScaleOutType>();
        DataCopyPad(tokenRevGlobalTensor[copyIdx * widthA], tokenScalebuf,
            {1, static_cast<uint16_t>(widthA * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleRevGlobalTensor[copyIdx * widthAScale], bufScale,
            {1, static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);

        // 发下一个槽的 MTE2 (copyIdx + BufferNum,复用 outIdx 槽)
        int32_t nextIdx = copyIdx + BufferNum;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = tripleTensor_[nextIdx * INT32_PER_256B].GetValue(TOKEN_ID);
            // WaitFlag 此处等待的是本轮刚发出的 SetFlag<MTE3_MTE2>(eventId),即等本槽 MTE3 完成。
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
            uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, static_cast<uint32_t>(tokenIndex));
            ReceiveTokenFromRmtServer(relayRank, remoteCopyOffset, outIdx, copyInNum);
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        }
    }
    // Phase 2 drain: 6 个 buffer-free flag
    for (int32_t bufferIdx = 0; bufferIdx < BufferNum; ++bufferIdx) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(kBufEvents[bufferIdx]);
    }
    // Phase 3: triple 三元组搬出
    SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
    DataCopy(tripleGlobalTensor_[rowDstOffsetInCore * INT32_PER_256B], tripleTensor_, copyNum * INT32_PER_256B);
}

// ====================================================================================================
// TripleInfoCalAndDispatch：专家接收token的三元组信息计算搬出 & token dispatch & 写Flag位
// ----------------------------------------------------------------------------------------------------
//   Phase 1: 按照blockNumPerRank_对aiv进行分组，一个rank归一个组aiv处理，组内根据该卡要发的token数量进行分核；
//   Phase 2: dispatch->gmm1 flag位AtomicAdd，每个 expert 有 maxWavesPerExpert_ 个槽位读写Flag；
// ====================================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::TripleInfoCalAndDispatch(
    GMMAddrInfo &gmmAddrInfo, int32_t localExpertId)
{
    // Phase 1: 分组分核进行三元组组装以及token dispatch
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M_256);
    int32_t priorExpertCumsum = (localExpertId == 0) ? 0 : // 前面所有 expert 在本卡的总 token 数
        cumsumInfoTensor_.GetValue(localExpertId * worldSize_ - 1);
    uint32_t topkIndexTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    // A8W4 路径下 SwigluQuant 覆盖 V1 UB，topkIndexTensor_ 需重新初始化
    if constexpr (ENABLE_A8W4) {
        if (localExpertId != 0) {
            CreateVecIndex(topkIndexTensor_, 0, topkIndexTensorSize / sizeof(int32_t));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    constexpr int32_t MAX_TRIPLE_ROWS_PER_CHUNK = static_cast<int32_t>(DISPATCH_WAVE_TILE_M);
    for (uint32_t srcRankInServer = blockIdx_; srcRankInServer < rankPerServer_; srcRankInServer += blockNum_) {
        for (uint32_t srcServer = 0; srcServer < serverNum_; ++srcServer) {
            uint32_t dstRankIdx = srcServer * rankPerServer_ + srcRankInServer;
            if (dstRankIdx >= worldSize_) {
                continue;
            }
            uint64_t sendToCurExpTokenCnt = 0;
            GatherMask(validTopkIndexTensor_, topkIndexTensor_,
                gatherMaskInt32Tensor_[dstRankIdx * (maskSlotSize_ / sizeof(uint32_t))],
                true, sendTotalNum_, {1, 1, 0, 0}, sendToCurExpTokenCnt);
            int32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 :
                cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));
            if (rowStartIdxInDst >= maxOutputSize_) {
                continue;
            }
            int32_t rowsCopyCnt = sendToCurExpTokenCnt;
            if (rowStartIdxInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartIdxInDst;
            }
            for (int32_t rowSrcIdxInCore = 0; rowSrcIdxInCore < rowsCopyCnt;
                rowSrcIdxInCore += MAX_TRIPLE_ROWS_PER_CHUNK) {
                int32_t rowsNumInCore = rowsCopyCnt - rowSrcIdxInCore;
                if (rowsNumInCore > MAX_TRIPLE_ROWS_PER_CHUNK) {
                    rowsNumInCore = MAX_TRIPLE_ROWS_PER_CHUNK;
                }
                int32_t rowDstOffsetInCore = rowStartIdxInDst + rowSrcIdxInCore;
                uint32_t tripleTensorAddr = ubBufferUsedAddr_;
                uint32_t tripleTensorSize = rowsNumInCore * ALIGN_32;
                tripleTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, tripleTensorAddr,
                    tripleTensorSize / sizeof(int32_t));
                SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();
                PipeBarrier<PIPE_ALL>();
                CopyGMToGMPerToken(rowDstOffsetInCore, dstRankIdx, rowSrcIdxInCore, rowsNumInCore);
                PipeBarrier<PIPE_ALL>();
                SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
                int32_t rowStartLocal = rowDstOffsetInCore - priorExpertCumsum;
                int32_t rowEndLocal = rowStartLocal + rowsNumInCore;
                int32_t waveLo = rowStartLocal / L1_TILE_M_I32;
                int32_t waveHi = (rowEndLocal - 1) / L1_TILE_M_I32;
                __gm__ int32_t* flagBase = gmmAddrInfo.dispatchToGmm1Flag;
                for (int32_t w = waveLo; w <= waveHi; ++w) {
                    int32_t waveStartLocal = w * L1_TILE_M_I32;
                    int32_t waveEndLocal = waveStartLocal + L1_TILE_M_I32;
                    int32_t lo = rowStartLocal > waveStartLocal ? rowStartLocal : waveStartLocal;
                    int32_t hi = rowEndLocal < waveEndLocal ? rowEndLocal : waveEndLocal;
                    AtomicAdd(flagBase + w, int32_t(hi - lo));
                }
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
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeTypeFunc>::UpdateGroupParams(ExpertLoopState &state,
    uint32_t expertIdx, uint64_t sendCnt)
{
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(state.problemShape);
        uint64_t n = Get<N_VALUE>(state.problemShape);
        uint64_t k = Get<K_VALUE>(state.problemShape);
        state.expertBeforeCnt += m;
        Get<IDX_A_OFFSET>(state.baseOffset) += m * k / A_ELEMS_PER_BYTE;
        Get<IDX_B_OFFSET>(state.baseOffset) += n * k / B_ELEMS_PER_BYTE;
        // only splitM
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
    if constexpr (Mode == AddrUpdateMode::kGmm1) {
        if (subBlockIdx_ == 0) { // aiv1进行SendCntCal计算完成后atomicAddFlag，aic/aiv0等到该flag位后读取cnt值
            __gm__ int32_t* sendCntFlag = (__gm__ int32_t*)params_.workspaceInfo.flagSendCntCalToUpdParamsPtr +
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
    } else if constexpr (Mode == AddrUpdateMode::kGmm2) {
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

// ==================================================================================
// UpdateGlobalBuffer：更新当前 expert 的 GMM 地址视图。
//                     GMM1 始终写 gmm1MmadResPtr；
//                     GMM2 始终写 gmm2MmadResPtr。
// ==================================================================================
template <TemplateMegaMoeTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo,
    const ExpertLoopState &state)
{
    if constexpr (Mode == AddrUpdateMode::kGmm1) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4) {
            gmmAddrInfo.gmm1OutGlobal =
                params_.workspaceInfo.gmm1MmadResPtr + Get<IDX_GMM1_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal = params_.workspaceInfo.dispatchRevDataPtr +
            Get<IDX_A_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.dispatchRevScalePtr +
            Get<IDX_A_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);

        gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(state.baseOffset) *
                                    sizeof(QuantScaleOutType);
        
        if constexpr(g_coreType == AIV) {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(state.baseOffset), Get<IDX_C_SCALE_OFFSET>(state.baseOffset),
                Get<IDX_FLAG_OFFSET>(state.baseOffset), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(Mode == AddrUpdateMode::kGmm2) {
        // guard 与 WorkspaceInfo 分配条件一致，由 TilingKey 保证同步。
        if constexpr (ENABLE_A8W4 || ENABLE_A4W4 || CombineQuantMode != COMBINE_NO_QUANT) {
            gmmAddrInfo.gmm2OutGlobal =
                params_.workspaceInfo.gmm2MmadResPtr + Get<IDX_GMM2_OFFSET>(state.baseOffset) * sizeof(bfloat16_t);
        }
        gmmAddrInfo.aGlobal =
            params_.workspaceInfo.swigluQuantDataPtr + Get<IDX_C_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.swigluQuantScalePtr +
                                   Get<IDX_C_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(state.baseOffset) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal =
            params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(state.baseOffset) * sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.swigluToGmm2Flag = (__gm__ int32_t*)params_.workspaceInfo.flagSwiGluToGmm2Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)params_.workspaceInfo.flagDispatchToGmm1Ptr +
                                Get<IDX_FLAG_OFFSET>(state.baseOffset) * dispatchFlagSlotsPerExpert_;
}

// =============================================
// ResetGmm2CombineSyncCounters：重置 GMM2→Combine 同步计数器
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ResetGmm2CombineSyncCounters()
{
    if constexpr(g_coreType == AIV) {
        int32_t totalCounters = static_cast<int32_t>(
            static_cast<int64_t>(expertPerRank_) * blockAivNum_ * INT_CACHELINE);
        int32_t coreLen, coreOffset;
        TilingByCore(totalCounters, coreLen, coreOffset);
        GlobalTensor<int32_t> gmm2CombineSyncCounterGm;
        gmm2CombineSyncCounterGm.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.gmm2CombineSyncCounterPtr);
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
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::InitCombineBuffers()
{
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

// =============================================
// ProcessCombine：generic combine-quant 路径的 AIV 后处理。
//                 等待本 expert 的 row-group 计数满足后，读取 triple 和 GMM2 输出，
//                 再执行 row-group 级 CombineRowGroup。
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::ProcessCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &gmm2State, uint32_t expertIdx)
{
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);

    GlobalTensor<int32_t> gmm2CombineSyncCounter;
    gmm2CombineSyncCounter.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.gmm2CombineSyncCounterPtr);

    uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
    uint32_t quantTokenSizeBytes = Ops::Base::CeilAlign(k_ + nScale, static_cast<uint32_t>(ALIGN_32));

    uint32_t m_expert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t tokenGroupsThisExpert = Ops::Base::CeilDiv(m_expert, L1_TILE_M_256);

    uint32_t coreIdForGrouping = aivCoreIdx_;
    uint32_t totalCoresForGrouping = blockAivNum_;
    if constexpr (ENABLE_A8W4) {
        if (subBlockIdx_ != 1) {
            return;
        }
        coreIdForGrouping = aivCoreIdx_ / 2;
        totalCoresForGrouping = blockAivNum_ / 2;
    }

    uint32_t myGroup, myIdxInGrp, myGrpSize;
    MegaMoeImpl::ComputeCoreGrouping(coreIdForGrouping, tokenGroupsThisExpert, totalCoresForGrouping,
        myGroup, myIdxInGrp, myGrpSize);

    if (myGroup >= tokenGroupsThisExpert) {
        return;
    }

    __gm__ int32_t* myCounterAddr = (__gm__ int32_t*)gmm2CombineSyncCounter.GetPhyAddr()
        + expertIdx * blockAivNum_ * INT_CACHELINE
        + aivCoreIdx_ * INT_CACHELINE;
    while (AscendC::ReadGmByPassDCache(myCounterAddr) != nTilesPerGroup)
    {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) {
        };
    }
    uint32_t tokenStart = myGroup * L1_TILE_M_256;
    uint32_t tokenCount = (L1_TILE_M_256 < m_expert - tokenStart) ? L1_TILE_M_256 : m_expert - tokenStart;
    uint32_t tokensPerCore = Ops::Base::CeilDiv(tokenCount, myGrpSize);
    int32_t myTokenOffset = myIdxInGrp * tokensPerCore;
    int32_t myTokenCount = 0;
    if (myTokenOffset < (int32_t)tokenCount) {
        myTokenCount = (tokensPerCore < tokenCount - myTokenOffset) ? tokensPerCore : tokenCount - myTokenOffset;
    }
    if (myTokenCount > 0) {
        AscendC::SetCtrlSpr<60, 60>(0);
        int64_t offset = 0;
        LocalTensor<int32_t> tripleTensor = LocalTensor<int32_t>(TPosition::VECIN, offset, myTokenCount * TRIPLE_SIZE);
        offset += myTokenCount * TRIPLE_SIZE * sizeof(int32_t);
        AscendC::GlobalTensor<int32_t> tripleGm;
        tripleGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.tripleInfoPtr
            + (gmm2State.expertBeforeCnt + tokenStart + myTokenOffset) * TRIPLE_SIZE * sizeof(int32_t)));
        AscendC::DataCopy(tripleTensor, tripleGm, myTokenCount * TRIPLE_SIZE);
        PipeBarrier<PIPE_MTE2>();
        MegaMoeCombineImpl::CombineTokenGroup<CombineQuantMode, bfloat16_t>(
            tokenStart + myTokenOffset, myTokenCount, k_, expertIdx, rankId_,
            gmmAddrInfo.gmm2OutGlobal, params_, tripleTensor, combineUbTensorSize_,
            offset, quantTokenSizeBytes);
    }
}

// =============================================
// UnpermuteBuffInit：Unpermute中使用的buffer申请
// =============================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::UnpermuteBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * HALF_TO_FP32;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));
    
    // Tensor用处：Unpermute 函数用于存储mte2搬入token；
    // Tensor大小：大小为3 * 单个token长度，2块是用于mte2搬运的doubleBuffer，1块是用于存储累加计算Cast完的输出结果，用于搬出；
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
        uint32_t scaleNum = Ops::Base::CeilAlign(static_cast<uint32_t>(k_), static_cast<uint32_t>(ALIGN_32));
        // Tensor用处：DeQuantMxFp8 中用于存储 bf16 格式的 scale（e8m0 转换后的中间结果）
        // Tensor大小：scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展
        uint32_t bf16ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(bfloat16_t) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        bf16ScaleTensor_ = LocalTensor<bfloat16_t>(
            TPosition::VECCALC, tempAddr, bf16ScaleBufAlign / sizeof(bfloat16_t));
        tempAddr += bf16ScaleBufAlign;
        // Tensor用处：DeQuantMxFp8 中用于存储 fp32 格式的 scale（广播后的最终 scale）
        // Tensor大小：scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32，双缓冲 + scale 扩展
        uint32_t fp32ScaleBufAlign = Ops::Base::CeilAlign(static_cast<uint32_t>
            (scaleNum * sizeof(float) * DOUBLE_BUFFER * HALF_TO_FP32), static_cast<uint32_t>(ALIGN_32));
        fp32ScaleTensor_ = LocalTensor<float>(TPosition::VECCALC, tempAddr, fp32ScaleBufAlign / sizeof(float));
        tempAddr += fp32ScaleBufAlign;
    }
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        uint32_t tempSize = tempBufAlign / sizeof(bfloat16_t);
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, tempAddr, tempSize);
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
    }
}

// ===============================================================
// Unpermute：对于各个专家还回来token的后处理，进行对应scale相乘与累加
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.combineSendPtr);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_EVENT_ID2>();
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResTensor_[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResTensor_[k_ * 2];
        LocalTensor<float> dataIn0Fp32 = dataResFp32Tensor_[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32Tensor_[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
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
                using Fp8Type = typename std::conditional<CombineQuantMode == MXFP8_E4M3_COMM_QUANT,
                    fp8_e4m3fn_t, fp8_e5m2_t>::type;
                MegaMoeCombineImpl::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32,
                    bf16ScaleTensor_, fp32ScaleTensor_, nScale, k_);
            }
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(dataResFp32Tensor_, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32Tensor_, dataResFp32Tensor_, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(dataResTensor_, dataResFp32Tensor_, AscendC::RoundMode::CAST_RINT, k_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_EVENT_ID3>();
        DataCopy(output[tokenIdx * k_], dataResTensor_, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

// ==============================================================================================
// CrossRankSyncInWorldSize：全卡同步，rankSyncInWorldPtr前48K用于同步，后面区域用于记录当前syncCnt值
// ==============================================================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    __gm__ int32_t* syncRank = (__gm__ int32_t*)params_.peermemInfo.rankSyncInWorldPtr;
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.rankSyncInWorldPtr +
        48 * 1024 + aivCoreIdx_ * 64);
    int count = ReadGmByPassDCache(syncCount) + 1;
    WriteGmByPassDCache(syncCount, count);
    for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
        if (rankIndex == rankId_) {
            continue;
        }
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[rankIndex]) + rankId_ * 16;
        hcomm_.WriteNbi(GetUrmaCommHandle(mc2Context_, rankIndex, rankId_), (GM_ADDR)syncRemoteAddr,
            (GM_ADDR)syncCount, static_cast<int64_t>(sizeof(int32_t)));
        auto syncCheck = syncRank + rankIndex * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
}

// ===============================================================
// GroupMatmulWithSwigluQuant：按实现路径分发到 A8W4 或 generic GMM1。
//                            A8W4 由 ENABLE_A8W4 控制；generic 路径的 subBlockIdx 判断已下沉到函数内部。
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::GroupMatmulWithSwigluQuant(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state)
{
    if constexpr (ENABLE_A8W4) {
        MegaMoeImpl::GroupMatmulSwigluQuantA8W4<
            QuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
    } else {
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ ||
            params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ) {
            // NZ format (A8W8_NZ / A4W4_NZ): isWeightNZ=true, EpilogueElementA 由 SwigluQuantOutType 自动处理类型提升
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType, true>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        } else {
            // Generic: fp8/fp4 activation × fp8/fp4 weight in ND format (includes A4W4 ND)
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, SwigluQuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
//                        A8W4/A4W4 走 A8W4 prologue（支持 combine-quant）；
//                        generic 路径同时承载非量化 combine 和 combine-quant 主线实现。
// ===============================================================
template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo &gmmAddrInfo, const ExpertLoopState &state, uint32_t expertIdx)
{
    if constexpr (ENABLE_A8W4 || ENABLE_A4W4) {
        MegaMoeImpl::GroupMatmul2CombineA8W4<CombineQuantMode,
            SwigluQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
            params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
            state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
    } else {
        // A8W8_NZ / Generic: both use the same GroupMatmul2 template, only LayoutB differs (ZN vs ND).
        if (params_.tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ) {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType, true, true>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
                state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
        } else {
            MegaMoeImpl::GroupMatmul2<CombineQuantMode, QuantOutType, QuantOutType, bfloat16_t,
                QuantScaleOutType, QuantScaleOutType, false, true>(
                params_, state.problemShape, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_,
                state.expertBeforeCnt, gmm2PingPongIdx_, expertIdx);
        }
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT && g_coreType == AIV) {
        ProcessCombine(gmmAddrInfo, state, expertIdx);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeTypeFunc>::Process()
{
    // 1.本卡数据处理
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    SendAndQuantBuffInit();
    SendMaskCal();               // 源卡按所有全局专家算 mask 并推送到目标专家卡
    ResetFlagList();             // 清理workSpace空间上的flag位
    ResetDispatchState();    // cross-server URMA dispatch 队列与 relay ready flag 清零

    if constexpr(g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();            // aic需要等待flag位reset清理完成
    CrossRankSyncInWorldSize();  // 全卡同步

    // 2.本卡专家接收数据dispatch & GroupMatmul1 & SwigluQuant
    DispatchBuffInit();
    GMMAddrInfo dispatchAddrInfo;
    GMMAddrInfo gmm1AddrInfo;
    TupleShape initShape;
    Get<N_VALUE>(initShape) = hiddenDim_;
    Get<K_VALUE>(initShape) = k_;
    BlockOffset initOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    ExpertLoopState dispatchState{initShape, initOffset, 0};
    ExpertLoopState gmm1State{initShape, initOffset, 0};

    // Dispatch-prefetch count forwarding（无成员变量耦合）：
    //   SendCntCal 将 expert token 数写入 nextSendCnt；
    //   循环顶部 nextSendCnt → curSendCnt 显式转发；
    //   GMM1 consumer 始终读 curSendCnt。
    uint64_t curSendCnt = 0;    // 当前 expert 的 sendCnt（GMM1 consumer 使用）
    uint64_t nextSendCnt = 0;   // 下一 expert 的 sendCnt（dispatch prefetch 算出）

    // 预调度 expert 0。
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            DispatchTokenToRmtServer(0);
            SendCntCal(0, nextSendCnt);
            if (UpdateGroupParams<AddrUpdateMode::kGmm1>(dispatchState, 0, nextSendCnt)) {
                UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(dispatchAddrInfo, dispatchState);
                TripleInfoCalAndDispatch(dispatchAddrInfo, 0);
            }
        }
    }

    for (int localExpertId = 0; localExpertId < expertPerRank_; localExpertId++) {
        curSendCnt = nextSendCnt;  // forward: dispatch(e) → GMM1(e)

        // Prefetch dispatch expert e+1，与当前 GMM1 consumer expert e 并发。
        if constexpr(g_coreType == AIV) {
            if (subBlockIdx_ == 1 && localExpertId + 1 < expertPerRank_) {
                DispatchTokenToRmtServer(localExpertId + 1);
                SendCntCal(localExpertId + 1, nextSendCnt);
                if (UpdateGroupParams<AddrUpdateMode::kGmm1>(dispatchState, localExpertId + 1, nextSendCnt)) {
                    UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(dispatchAddrInfo, dispatchState);
                    TripleInfoCalAndDispatch(dispatchAddrInfo, localExpertId + 1);
                }
            }
        }

        // GMM1 consumer 消费 expert e。
        if (!UpdateGroupParams<AddrUpdateMode::kGmm1>(gmm1State, localExpertId, curSendCnt)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm1>(gmm1AddrInfo, gmm1State);
        GroupMatmulWithSwigluQuant(gmm1AddrInfo, gmm1State);
    }
    EndSync(vecSetSyncCom_);
    if constexpr(g_coreType == AIV) {
        if (subBlockIdx_ == 1) {
            ExpertTokenNumCopyOut(); // 本卡专家接受的tokenCnt总数搬出
        }
    }

    SyncAll<true>();
    // 3. 本卡专家接收数据GroupMatmul2 & Combine
    vecSetSyncCom_ = 0;
    GMMAddrInfo gmm2AddrInfo;
    ExpertLoopState gmm2State{initShape, initOffset, 0};
    InitCombineBuffers();

    if constexpr (CombineQuantMode == COMBINE_NO_QUANT && g_coreType == AIV) {
        if (GetSubBlockIdx() == 1) {
            LocalTensor<uint8_t> hcommTensor_=LocalTensor<uint8_t>(TPosition::VECCALC, 0, ALIGN_512 / sizeof(uint8_t));
            hcomm_.Init(hcommTensor_, ALIGN_512);
            MegaMoeCombineImpl::CombineSendTokenToRemote<bfloat16_t>(params_);
        }
    }
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams<AddrUpdateMode::kGmm2>(gmm2State, expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer<AddrUpdateMode::kGmm2>(gmm2AddrInfo, gmm2State);
        GroupMatmulWithCombine(gmm2AddrInfo, gmm2State, expertIdx);
    }
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        EndGMM2Sync(vecSetSyncCom_, gmm2PingPongIdx_);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 4. 本卡数据Unpermute
    if constexpr(g_coreType == AIV) {
        UnpermuteBuffInit();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif
