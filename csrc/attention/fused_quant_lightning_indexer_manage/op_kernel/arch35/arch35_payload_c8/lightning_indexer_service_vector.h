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
 * \file quant_lightning_indexer_service_vector.h
 * \brief
 */
#ifndef VLLM_FAST_QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define VLLM_FAST_QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_common.h"
// 复用 BF16 版（lidu）的 vf：QK 为 fp32 累加（Mmad fp8×fp8→fp32），
// Fixpipe 原样搬运，乘权/归约/relu/bf16-sortable 全部与 lidu 走同一份实现
// （relu 与官方"Relu在cube随路做"数值等价），无 ×1/1024、无 fp16 往返；
// kScale 通过 6 参 MulWeightAndReduceSum（必填）在 Σ 之后乘（官方 QuantLI 语义）。
// build_c8.sh 只 flatten C8 目录：vf 以自包含拷贝方式放在本目录 vf/ 下，
// classify_vf.h 放在本算子 op_kernel 根，经 ../ 相对路径解析。
#include "vf/lightning_indexer_vector1.h"
#include "payload/hist_topk_index_update_a5_topk.h"
#include "../lim_pool_contract.h"
#include "../fused_quant_lightning_indexer_manage_workspace.h"
#include "payload/hist_topk_index_update_a5_evict_vf.h"
// classify_vf.h 位于本算子 op_kernel 根（与 arch35_payload_c8 同级），
// 用 ../ 相对路径解析（op_kernel 根不在 OPC 编译的 -I 路径上）。
#include "../fused_quant_lightning_indexer_manage_classify_vf.h"
#include "../fused_quant_lightning_indexer_manage_hit_sort_vf.h"
#include "../fused_quant_lightning_indexer_manage_sort.h"

namespace VllmFastQLIKernel {
using namespace VllmFastQLICommon;
// =====================================================================================
// Vector service 架构 — 重构后与 lidu (BF16 版) 对齐
// =====================================================================================
// 与 lidu 的 LightningIndexerServiceVector 同构，差异收敛为两处:
//   1. GetKeyScale() — key 反量化 scale 的 PA/非PA 搬运（lidu 无 scale）
//   2. ProcessVec1 前半段 — weight/qScale 预取 + weightFloat = float(weight)×qScale
// QK 为 fp32 累加（Mmad fp8×fp8→fp32），Fixpipe 原样搬到 UB；
// relu 在 vf 的 WeightedAccum 中对同一批 fp32 值完成（官方"Relu在cube
// 随路做"与其数值等价——元素级幂等，作用于相同操作数），无 ×1/1024、
// 无 fp16 往返。乘权/归约/bf16-sortable 全部复用 lidu 的 vf
// （VllmFastVector1::MulWeightAndReduceSum），kScale 经 6 参重载在 Σ 之后乘。
// ProcessTopK/FinalizePayloadUpdate/FindVictimsOnDemand 与 lidu 一致
// （payload 五件套与 classify vf 均为共享文件）。
// =====================================================================================

constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_6K = 6144;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    using W_T = typename QLIT::weightType;
    using SCALE_T = typename QLIT::scaleType;   // 反量化 scale 精度（fp32）
    using QK_T = typename QLIT::qkType;         // Cube 输出类型（int32，fp8 Mmad 累加）
    using SCORE_T = typename QLIT::scoreType;   // score 存储类型（uint16，bf16 sortable）
    static_assert(std::is_same_v<SCORE_T, uint16_t>, "A5FusedLiManageC8 score must be uint16_t");

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const VllmFastQLICommon::RunInfo &info);
    // The local candidate span may be any contiguous S2 interval assigned
    // to the current core; it is no longer a fixed 16K chunk.
    __aicore__ inline void ProcessTopK(const VllmFastQLICommon::RunInfo &info,
                                       bool allowFullSlotPrefetch,
                                       uint32_t chunkStart = 0U,
                                       uint32_t chunkLen = 0U,
                                       uint32_t chunkIdx = 0U,
                                       uint32_t maxChunks = 0U);
    __aicore__ inline void ReduceLocalTopK(
        const VllmFastQLICommon::RunInfo &info,
        uint32_t segmentCount, uint32_t candidateStride,
        uint32_t rowBegin, uint32_t rowEnd);
    __aicore__ inline void InitCandidateTensor(
        GlobalTensor<uint16_t> candidateScores,
        GlobalTensor<uint32_t> candidatePayloads,
        GlobalTensor<uint16_t> candidateTags);
    __aicore__ inline LocalTensor<uint16_t> TagBuffer(uint32_t index)
    {
        return candidatePayloadLocal_.template ReinterpretCast<uint16_t>()[
            (index & 1U) * topkCount_];
    }
    __aicore__ inline bool HasLongSourceTags() const
    {
        return constInfo_.kSeqSize > lim_payload_layout::PACKED_SOURCE_CAPACITY;
    }
    __aicore__ inline void GatherCurrentSourceTags(
        uint32_t loopIndex, uint32_t previousLen, uint32_t tokenBase)
    {
        if (!HasLongSourceTags()) return;
        topkOp_.GatherSourceTags(
            TagBuffer(loopIndex + 1U), TagBuffer(loopIndex),
            TagBuffer(loopIndex), previousLen, tokenBase, false);
        lastTagBufferIndex_ = (loopIndex + 1U) & 1U;
    }
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct VllmFastQLICommon::ConstInfo &constInfo,
                                      const VllmFastLIC8TilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                              GlobalTensor<SCALE_T> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<int32_t> cacheSlotsGm,
                                              GlobalTensor<int32_t> topkSlotsGm,
                                              GlobalTensor<int32_t> missCountGm);
#ifdef A5_MTP_CLASSIFY_ONLY
    __aicore__ inline void InitMtpTopkSourceTensor(
        GlobalTensor<int32_t> topkSourceIdsGm);
    __aicore__ inline void InitMtpThresholdTensor(
        GlobalTensor<uint16_t> thresholdGm);
#endif
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FinalizePayloadUpdate(const VllmFastQLICommon::RunInfo &info,
                                                 uint32_t outputRow,
                                                 uint64_t scoreRowOffset,
                                                 uint32_t validS2Len);
#ifdef A5_MTP_CLASSIFY_ONLY
    __aicore__ inline void BuildSortedMissPairs(
        const LocalTensor<int32_t> &classifiedIndex,
        uint32_t missCount);
    __aicore__ inline void ExtractSortedMissSourceIds(
        const LocalTensor<int32_t> &output, uint32_t missCount);
#endif
    __aicore__ inline uint32_t FindVictimsOnDemand(
        const VllmFastQLICommon::RunInfo &info, uint64_t scoreRowOffset,
        uint32_t validS2Len, uint32_t requiredCount, uint16_t kthValue,
        const LocalTensor<uint32_t>& compactPayloadLocal);

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> topkSlotsGm;
    GlobalTensor<int32_t> missCountGm;
    GlobalTensor<uint16_t> candidateScoresGm;
    GlobalTensor<uint32_t> candidatePayloadsGm;
    GlobalTensor<uint16_t> candidateTagsGm;
#ifdef A5_MTP_CLASSIFY_ONLY
    GlobalTensor<int32_t> topkSourceIdsGm;
    GlobalTensor<uint16_t> thresholdGm;
#endif
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT_KSCALE = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_KSCALE = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t VEC1_V_MTE2_EVENT_QSCALE = EVENT_ID6;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_QSCALE = EVENT_ID3;
    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;

private:
    __aicore__ inline void GetKeyScale(const VllmFastQLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    LocalTensor<float> weightFloatUB_;
    // tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<SCALE_T> kScaleUB_;
    // tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<SCALE_T> qScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    // One 2048-entry cache_slots DMA stage shared by survivor payload
    // construction, non-final-request prefix prefetch, and victim scanning.
    TBuf<TPosition::VECCALC> slotStageBuf_;
    LocalTensor<int32_t> slotStageLocal_;

    TBuf<TPosition::VECCALC> candidatePayloadBuf_;
    LocalTensor<uint32_t> candidatePayloadLocal_;
    uint32_t lastTagBufferIndex_ = 0U;

    LocalTensor<int32_t> outInvalidLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t weightStride_ = 0;
    uint32_t trunkLen_ = 0;
    struct VllmFastQLICommon::ConstInfo constInfo_;
    static constexpr uint32_t EVICT_CANDIDATE_CAP = 2048;
    vllm_fast_hist_topk_index_update_a5_payload::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));
    resMm1UB_ = resMm1Buf_.Get<QK_T>();
    // weight/qScale 预取在 decode 下每 (bN2, gS1) 只发生一次（qScale 无 pingpong），
    // 尺寸沿用原 2 * 布局以保持 UB 规划不变，实际只使用前一半。
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightFloatUB_ = weightFloatBuf_.Get<float>();
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    kScaleUB_ = kScaleBuf_.Get<SCALE_T>();
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<SCALE_T>();

    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>();

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

    pipe->InitBuffer(slotStageBuf_, topkCount_ * sizeof(int32_t));
    slotStageLocal_ = slotStageBuf_.Get<int32_t>();

    pipe->InitBuffer(candidatePayloadBuf_, EVICT_CANDIDATE_CAP * sizeof(uint32_t));
    candidatePayloadLocal_ = candidatePayloadBuf_.Get<uint32_t>();

    //刷-1
    Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * 16);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct VllmFastQLICommon::ConstInfo &constInfo,
                                                   const VllmFastLIC8TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    if constexpr (std::is_same_v<SCORE_T, uint32_t>) {
        trunkLen_ = TRUNK_LEN_6K;
    } else {
        trunkLen_ = TRUNK_LEN_16K;
    }
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, topkCount_, trunkLen_);
    topkCountAlign256_ = VllmFastQLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
    weightStride_ = tilingData->weightStride;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                                           GlobalTensor<SCALE_T> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm,
                                                           GlobalTensor<int32_t> cacheSlotsGm,
                                                           GlobalTensor<int32_t> topkSlotsGm,
                                                           GlobalTensor<int32_t> missCountGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->topkSlotsGm = topkSlotsGm;
    this->missCountGm = missCountGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitCandidateTensor(
    GlobalTensor<uint16_t> candidateScores,
    GlobalTensor<uint32_t> candidatePayloads,
    GlobalTensor<uint16_t> candidateTags)
{
    candidateScoresGm = candidateScores;
    candidatePayloadsGm = candidatePayloads;
    candidateTagsGm = candidateTags;
}

#ifdef A5_MTP_CLASSIFY_ONLY
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitMtpTopkSourceTensor(
    GlobalTensor<int32_t> topkSourceIdsGm)
{
    this->topkSourceIdsGm = topkSourceIdsGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitMtpThresholdTensor(
    GlobalTensor<uint16_t> thresholdGm)
{
    this->thresholdGm = thresholdGm;
}
#endif
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const VllmFastQLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<SCALE_T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(SCALE_T);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_], kScaleGm[blockId *
                                constInfo_.keyDequantScaleStride0 + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(SCALE_T);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(SCALE_T);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ +
                                               resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(SCALE_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                            kScaleGm[runInfo.tensorKeyScaleOffset],
                                            copyInParams, padParams);
    }
}

// =====================================================================================
// ProcessVec1 — 重构后: 直接复用 lidu vf（与 BF16 版的差异收敛为两处）
// =====================================================================================
// QK 为 fp32 累加（Mmad fp8×fp8→fp32），Fixpipe 原样搬运:
//   QK_fp32 --Fixpipe(直通)--> QK_fp32
// vector 侧:
//   1. weight 预取后: weightFloatUB_ = float(weight_bf16) × qScale      (fp32 域)
//   2. 调用 lidu 原版 VllmFastVector1::MulWeightAndReduceSum:
//      Relu(QK)×weight 归约 + kScale(6 参重载, Σ 之后乘) + bf16 sortable
//      （relu 与官方"Relu在cube随路做"数值等价）
//   kScale 预取与事件同步保持原 QuantLI 结构（16 块窗口 + pingpong）。
// =====================================================================================

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const VllmFastQLICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto kScalepingpong = (info.kScaleLoop % 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;
    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(VllmFastQLICommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }

    if (info.isFirstS2InnerLoop) {
        // weight / qScale 预取（decode 下每 (bN2, gS1) 仅一次，无 qScale pingpong）
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
        //weightsGm --> weightUB_
        int64_t weightGmOffset =
            info.tensorWeightsOffset + curAivS1Idx * weightStride_;
        DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams wDataCopyExtParams;
        wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
        constexpr uint32_t weightRowStride =
            UB_BANK_DEPTH_STRIDE / sizeof(W_T);
        const uint32_t weightGapBytes =
            (weightStride_ - gSize_) * sizeof(W_T);
        if (weightGapBytes % 32U == 0U) {
            // Fast multi-row transfer. GM srcStride is in bytes while the UB
            // destination stride is in 32-byte blocks.
            wDataCopyExtParams.blockCount = curAivS1ProcNum;
            wDataCopyExtParams.srcStride = weightGapBytes;
            wDataCopyExtParams.dstStride =
                (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
            DataCopyPad(weightUB_, weightsGm[weightGmOffset],
                        wDataCopyExtParams, padWeightsParams);
        } else {
            // A legal caller-owned projection view may have a row gap that is
            // not representable by the multi-block MTE stride. At most two
            // rows are owned by one AIV, so issue aligned single-row copies
            // while keeping the valid requests in the shared LI path.
            wDataCopyExtParams.blockCount = 1;
            wDataCopyExtParams.srcStride = 0;
            wDataCopyExtParams.dstStride = 0;
            for (uint32_t row = 0U;
                 row < static_cast<uint32_t>(curAivS1ProcNum); ++row) {
                DataCopyPad(
                    weightUB_[row * weightRowStride],
                    weightsGm[weightGmOffset + row * weightStride_],
                    wDataCopyExtParams, padWeightsParams);
            }
        }

        //qScaleGm  -->  qScaleUB_
        DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
        DataCopyExtParams qDataCopyExtParams;
        qDataCopyExtParams.blockCount = curAivS1ProcNum;
        qDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
        qDataCopyExtParams.srcStride = 0;
        qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
        const int64_t queryScaleGmOffset =
            info.tensorQueryScaleOffset + curAivS1Idx * gSize_;
        DataCopyPad(qScaleUB_, qScaleGm[queryScaleGmOffset],
                    qDataCopyExtParams, padQScaleParams);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + 0);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + 0);

        // Rows are separated by UB_BANK_DEPTH_STRIDE. The old decode path
        // only handled one row per AIV; s1Base=4 assigns two rows and must
        // convert/scale both without treating the bank padding as data.
        constexpr uint32_t floatRowStride =
            UB_BANK_DEPTH_STRIDE / sizeof(float);
        Cast(weightFloatUB_, weightUB_, RoundMode::CAST_NONE, gSize_);
        Mul(weightFloatUB_, weightFloatUB_, qScaleUB_, gSize_);
        if (curAivS1ProcNum == 2) {
            Cast(weightFloatUB_[floatRowStride],
                 weightUB_[weightRowStride],
                 RoundMode::CAST_NONE, gSize_);
            Mul(weightFloatUB_[floatRowStride],
                weightFloatUB_[floatRowStride],
                qScaleUB_[floatRowStride], gSize_);
        }
    }

    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        // kScale 预取: 一次 16 个 s2 块, pingpong 覆盖大 source 长度的连续窗口
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : 16 * s2BaseSize_;
        //kScaleGm  -->  kScaleUB_（含 PA blkTable 跳转）
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(VllmFastQLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;
    // QK 已是 fp32（Mmad fp8×fp8→fp32 累加 + Fixpipe 原样搬运），
    // 此处无 ×1/1024、无 fp16 往返；relu 由 vf 的 WeightedAccum 完成
    // （与官方"Relu在cube随路做"数值等价）。

    LocalTensor<SCORE_T> outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    auto weightFloatBase = weightFloatUB_;  // float(weight) * qScale（行 0）
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];

    // Weighted reduction, post-sum kScale, then BF16 sortable conversion.
    // A four-query tile assigns two rows to each AIV; fuse those rows in one
    // VF just like official QuantLI, while retaining the verified one-row C8
    // path for partial tiles.
    auto out = (__ubuf__ SCORE_T *)outBase.GetPhyAddr();
    auto qk = (__ubuf__ float *)qkBase.GetPhyAddr();
    auto weightFloat = (__ubuf__ float *)weightFloatBase.GetPhyAddr();
    auto kScale = (__ubuf__ SCALE_T *)kScaleBase.GetPhyAddr();
    if (curAivS1ProcNum == 2) {
        constexpr uint32_t outRowStride =
            UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T);
        constexpr uint32_t floatRowStride =
            UB_BANK_DEPTH_STRIDE / sizeof(float);
        const uint32_t qkRowStride =
            gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float);
        VllmFastVector1::MulWeightAndReduceSum2C8(
            out, out + outRowStride,
            qk, qk + qkRowStride,
            qkVLstride,
            weightFloat, weightFloat + floatRowStride,
            gSize_, kScale);
    } else {
        VllmFastVector1::MulWeightAndReduceSum(
            out, qk, qkVLstride, weightFloat, gSize_, kScale);
    }
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    }
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    //outUB_ --->  scoreGm
    // Keep active route score rows contiguous in the request's workspace
    // slice. For a partial tile (for example R=2 with tile=4), the odd
    // AIV starts at row 1 rather than the full-tile row 2.
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            CeilDiv(curS1ProcNum, 2) *
                                VllmFastQLICommon::Align(
                                    (uint64_t)constInfo_.kSeqSize,
                                    (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - copyOutParams.blockLen) / 32;
    copyOutParams.dstStride = (VllmFastQLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<VllmFastQLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(VllmFastQLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

constexpr uint32_t VICTIM_SCAN_CHUNK = 2048;

template <typename QLIT>
__aicore__ inline uint32_t QLIVector<QLIT>::FindVictimsOnDemand(
    const VllmFastQLICommon::RunInfo &info, uint64_t scoreRowOffset,
    uint32_t validS2Len, uint32_t requiredCount, uint16_t kthValue,
    const LocalTensor<uint32_t>& compactPayloadLocal)
{
    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize;
    LocalTensor<uint16_t> scanScoresLocal = mrgValueLocal_;
    LocalTensor<int16_t> scanSlotsLocal =
        mrgValueLocal_[VICTIM_SCAN_CHUNK].template ReinterpretCast<int16_t>();
    uint32_t candidateCount = 0;

    AscendC::DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint16_t> scorePadParams{true, 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> slotPadParams{true, 0, 0, 0};

    for (uint32_t chunkBase = 0;
         chunkBase < validS2Len && candidateCount < requiredCount;
         chunkBase += VICTIM_SCAN_CHUNK) {
        uint32_t chunkLen =
            chunkBase + VICTIM_SCAN_CHUNK > validS2Len
                ? validS2Len - chunkBase
                : VICTIM_SCAN_CHUNK;
        uint32_t alignedLen = VllmFastQLICommon::Align(chunkLen, static_cast<uint32_t>(64));

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        copyParams.blockLen = chunkLen * sizeof(uint16_t);
        AscendC::DataCopyPad(
            scanScoresLocal, scoreGm[scoreRowOffset + chunkBase],
            copyParams, scorePadParams);
        copyParams.blockLen = chunkLen * sizeof(int32_t);
        AscendC::DataCopyPad(
            slotStageLocal_, cacheSlotsGm[cacheBase + chunkBase],
            copyParams, slotPadParams);
        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

        asc_vf_call<lim_normalize_pool_slots_vf::Normalize>(
            (__ubuf__ uint32_t *)slotStageLocal_.GetPhyAddr(),
            (chunkLen + 63U) / 64U);
        PipeBarrier<PIPE_V>();
        Cast(scanSlotsLocal, slotStageLocal_, RoundMode::CAST_NONE, chunkLen);
        if (alignedLen > chunkLen) {
            Duplicate(scanSlotsLocal[chunkLen], static_cast<int16_t>(-1),
                      alignedLen - chunkLen);
        }
        PipeBarrier<PIPE_V>();

        VllmFastLightningIndexerPayloadEvictVF::CompactEligiblePayloads(
            (__ubuf__ uint32_t *)compactPayloadLocal.GetPhyAddr(),
            (__ubuf__ uint16_t *)scanScoresLocal.GetPhyAddr(),
            (__ubuf__ uint16_t *)scanSlotsLocal.GetPhyAddr(),
            kthValue, chunkBase, alignedLen / 64);
        uint32_t compactCount = static_cast<uint32_t>(
            AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() /
            sizeof(uint32_t));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

        uint32_t keepCount = compactCount;
        uint32_t remaining = requiredCount - candidateCount;
        if (keepCount > remaining) {
            keepCount = remaining;
        }
        for (uint32_t i = 0; i < keepCount; ++i) {
            candidatePayloadLocal_.SetValue(
                candidateCount + i, compactPayloadLocal.GetValue(i));
        }
        candidateCount += keepCount;
    }

    if (candidateCount < requiredCount) {
        // Rare tie fallback. Build an exact open-addressed set of the final
        // TopK token IDs, then select any cached token outside that set. This
        // keeps the normal path bitmap-free and guarantees C - T semantics.
        constexpr uint32_t HASH_CAPACITY = 4096;
        constexpr uint32_t HASH_MASK = HASH_CAPACITY - 1;
        constexpr uint32_t TOKEN_MASK =
            vllm_fast_hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
        LocalTensor<uint32_t> topkHash =
            mrgValueLocal_.template ReinterpretCast<uint32_t>();
        Duplicate(topkHash.template ReinterpretCast<int32_t>(),
                  static_cast<int32_t>(-1), HASH_CAPACITY);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
        WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

        for (uint32_t i = 0; i < topkCount_; ++i) {
            uint32_t token = indicesOutLocal_.GetValue(i) & TOKEN_MASK;
            uint32_t hashPos = (token * 2654435761U) & HASH_MASK;
            while (topkHash.GetValue(hashPos) != 0xffffffffU &&
                   topkHash.GetValue(hashPos) != token) {
                hashPos = (hashPos + 1) & HASH_MASK;
            }
            topkHash.SetValue(hashPos, token);
        }

        candidateCount = 0;
        for (uint32_t chunkBase = 0;
             chunkBase < validS2Len && candidateCount < requiredCount;
             chunkBase += VICTIM_SCAN_CHUNK) {
            uint32_t chunkLen =
                chunkBase + VICTIM_SCAN_CHUNK > validS2Len
                    ? validS2Len - chunkBase
                    : VICTIM_SCAN_CHUNK;
            copyParams.blockLen = chunkLen * sizeof(int32_t);
            AscendC::DataCopyPad(
                slotStageLocal_, cacheSlotsGm[cacheBase + chunkBase],
                copyParams, slotPadParams);
            SetFlag<HardEvent::MTE2_S>(V_MTE2_EVENT1);
            WaitFlag<HardEvent::MTE2_S>(V_MTE2_EVENT1);

            for (uint32_t i = 0;
                 i < chunkLen && candidateCount < requiredCount; ++i) {
                int32_t slot = slotStageLocal_.GetValue(i);
                if (slot < 0) {
                    continue;
                }
                uint32_t token = chunkBase + i;
                uint32_t hashPos = (token * 2654435761U) & HASH_MASK;
                while (topkHash.GetValue(hashPos) != 0xffffffffU &&
                       topkHash.GetValue(hashPos) != token) {
                    hashPos = (hashPos + 1) & HASH_MASK;
                }
                if (topkHash.GetValue(hashPos) == token) {
                    continue;
                }
                candidatePayloadLocal_.SetValue(
                    candidateCount,
                    (static_cast<uint32_t>(slot) <<
                     vllm_fast_hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) |
                    token);
                ++candidateCount;
            }
        }
    }
    return candidateCount;
}

#ifdef A5_MTP_CLASSIFY_ONLY
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::BuildSortedMissPairs(
    const LocalTensor<int32_t> &classifiedIndex, uint32_t missCount)
{
    if (missCount == 0U) {
        return;
    }

    // SortAll emits interleaved (key, source) pairs. This is the exact
    // private union input, so keep that
    // representation instead of extracting source IDs and packing them
    // again after the global barrier.
    constexpr uint32_t SORT_RUN = 32U;
    constexpr uint32_t SORT_KEY_BASE_BITS = 0x40000000U;
    const uint32_t SORT_TOKEN_MASK = HasLongSourceTags()
        ? lim_payload_layout::LONG_SOURCE_MASK
        : lim_payload_layout::SOURCE_MASK;
    const uint32_t alignedCount =
        (missCount + SORT_RUN - 1U) / SORT_RUN * SORT_RUN;
    const uint32_t sortLength =
        fused_quant_li_manage_sort::SortLength(missCount);

    // UB reuse after histogram TopK has completed:
    //   mrgValue:     [keys, payloads], then published interleaved miss pairs
    //   topkSharedTmp: SortAll scratch
    // Sort directly into the publication buffer to avoid a full-row copy.
    // The smallest configured trunk still leaves enough room for both rows.
    LocalTensor<float> keyLocal =
        mrgValueLocal_.template ReinterpretCast<float>();
    LocalTensor<uint32_t> payloadLocal =
        keyLocal[sortLength].template ReinterpretCast<uint32_t>();
    LocalTensor<float> sortScratchLocal =
        topkSharedTmpLocal_.template ReinterpretCast<float>();

    Duplicate(keyLocal.template ReinterpretCast<uint32_t>(), 0U, sortLength);
    Duplicate(payloadLocal, 0U, sortLength);
    PipeBarrier<PIPE_V>();
    DataCopy(payloadLocal,
             classifiedIndex.template ReinterpretCast<uint32_t>(),
             alignedCount);
    PipeBarrier<PIPE_V>();
    Muls(keyLocal.template ReinterpretCast<int32_t>(),
         payloadLocal.template ReinterpretCast<int32_t>(), -1,
         alignedCount);
    PipeBarrier<PIPE_V>();
    Adds(keyLocal.template ReinterpretCast<int32_t>(),
         keyLocal.template ReinterpretCast<int32_t>(),
         static_cast<int32_t>(SORT_KEY_BASE_BITS + SORT_TOKEN_MASK),
         alignedCount);
    PipeBarrier<PIPE_V>();

    // Padded lanes contain scratch data. Give them the smallest key so they
    // sort behind every valid miss. This scalar tail is
    // bounded by 31 elements, replacing the old O(miss_count^2) scalar sort.
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    LocalTensor<uint32_t> keyBits =
        keyLocal.template ReinterpretCast<uint32_t>();
    for (uint32_t index = missCount; index < alignedCount; ++index) {
        keyBits.SetValue(index, 0U);
    }
    SetFlag<HardEvent::S_V>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::S_V>(V_MTE2_EVENT3);

    // Sort is descending. key=base+mask-source therefore produces source IDs
    // in ascending order while retaining the pair layout MrgSort consumes.
    // Use A3's fixed-bucket SortAll rather than a variable-length full sort.
    // The zero padding is finite and sorts after every valid source key.
    fused_quant_li_manage_sort::SortAll(keyLocal, sortScratchLocal, sortLength);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ExtractSortedMissSourceIds(
    const LocalTensor<int32_t> &output, uint32_t missCount)
{
    if (missCount == 0U) {
        return;
    }
    GatherMaskParams params;
    params.repeatTimes =
        (missCount * 2U * sizeof(uint32_t) + 255U) / 256U;
    params.src0BlockStride = 1;
    params.src0RepeatStride = 8;
    params.src1RepeatStride = 0;
    uint64_t reserved = 0U;
    // Sort emits interleaved (key, source-id) pairs. Pattern 2 extracts the
    // odd words, preserving the ascending source-ID order.
    GatherMask(
        output.template ReinterpretCast<uint32_t>(),
        mrgValueLocal_.template ReinterpretCast<uint32_t>(),
        static_cast<uint8_t>(2), false, static_cast<uint32_t>(0),
        params, reserved);
    PipeBarrier<PIPE_V>();
}
#endif

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FinalizePayloadUpdate(
    const VllmFastQLICommon::RunInfo &info, uint32_t outputRow,
    uint64_t scoreRowOffset, uint32_t validS2Len)
{
    constexpr uint32_t CLASSIFY_CHUNK = VllmFastTopkIndexerClassifyVF::CHUNK_SIZE;
    LocalTensor<int32_t> classifiedIndex = topkSharedTmpLocal_.template ReinterpretCast<int32_t>();
    LocalTensor<uint32_t> compactPayloadLocal =
        topkSharedTmpLocal_[topkCountAlign256_];
    const uint32_t publicOutputBaseRow =
        static_cast<uint32_t>(info.indiceOutOffset / topkCount_);
    const uint32_t fixedWorkspaceRow =
        info.workspaceRow + outputRow - publicOutputBaseRow;

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    if (info.denseDirectPublish) {
        // DENSE owns the complete HBM sequence.  The streaming TopK payload
        // is therefore already the final logical/physical token ID; never
        // dereference req_pool_entries or cache_slots for this mode.
        LocalTensor<int32_t> denseTokenIds =
            indicesOutLocal_.template ReinterpretCast<int32_t>();
        if (validS2Len >= topkCount_) {
            // RunTokenPayloadOnly retains the packed {slot15, token17}
            // representation used by the shared streaming TopK. DENSE has no
            // cache slot, so strip the miss-slot prefix before publishing the
            // plain logical token IDs expected by the ABI.
            SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
            WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
            if (HasLongSourceTags()) {
                asc_vf_call<fused_quant_li_manage_hit_sort_vf::DecodeDenseLong>(
                    (__ubuf__ uint32_t *)classifiedIndex.GetPhyAddr(),
                    (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
                    (__ubuf__ uint16_t *)TagBuffer(lastTagBufferIndex_).GetPhyAddr(),
                    topkCount_ / CLASSIFY_CHUNK);
            } else {
                VllmFastTopkIndexerClassifyVF::SqueezeIndexerMissTokenIds(
                    (__ubuf__ uint32_t *)classifiedIndex.GetPhyAddr(),
                    (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
                    topkCount_ / CLASSIFY_CHUNK);
            }
            PipeBarrier<PIPE_V>();
            denseTokenIds = classifiedIndex;
        }
        // Dense -3/-2 rows do not feed the sparse ordered union. Their public
        // count tensors are published by the lifecycle-specific manager, so
        // do not write sparse-only count/threshold workspace metadata here.
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyParams outputCopy{
            1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
#ifdef A5_MTP_SLOT_OUTPUT_STRIDE
        const uint64_t outputOffset =
            static_cast<uint64_t>(outputRow) * A5_MTP_SLOT_OUTPUT_STRIDE;
#else
        const uint64_t outputOffset =
            static_cast<uint64_t>(outputRow) * topkCount_;
#endif
#ifdef A5_MTP_CLASSIFY_ONLY
        DataCopyPad(
            topkSourceIdsGm[outputOffset],
            denseTokenIds,
            outputCopy);
#else
        DataCopyPad(
            indiceOutGm[outputOffset],
            denseTokenIds,
            outputCopy);
#endif
        DataCopyPad(
            topkSlotsGm[outputOffset],
            denseTokenIds,
            outputCopy);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        return;
    }
    if (info.cacheTokenCount == 0) {
        Duplicate(classifiedIndex, static_cast<int32_t>(-1), topkCount_);
        Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
        LocalTensor<int32_t> missCountLocal =
            candidatePayloadLocal_.template ReinterpretCast<int32_t>();
        missCountLocal.SetValue(0, 0);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
        AscendC::DataCopyParams scalarCopy{
            1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
#ifdef A5_MTP_CLASSIFY_ONLY
        DataCopyPad(
            missCountGm[static_cast<uint64_t>(fixedWorkspaceRow) *
                fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE],
            missCountLocal, scalarCopy);
#else
        DataCopyPad(missCountGm[info.bIdx], missCountLocal, scalarCopy);
#endif
        SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyParams outputCopy{
            1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
#ifdef A5_MTP_SLOT_OUTPUT_STRIDE
        const uint64_t sourceOutOffset =
            static_cast<uint64_t>(outputRow) * A5_MTP_SLOT_OUTPUT_STRIDE;
        const uint64_t slotOutOffset =
            static_cast<uint64_t>(outputRow) *
            A5_MTP_SLOT_OUTPUT_STRIDE;
#else
        const uint64_t sourceOutOffset =
            static_cast<uint64_t>(outputRow) * topkCount_;
        const uint64_t slotOutOffset = sourceOutOffset;
#endif
#ifndef A5_MTP_CLASSIFY_ONLY
        DataCopyPad(
            indiceOutGm[sourceOutOffset], classifiedIndex, outputCopy);
#else
        DataCopyPad(
            topkSourceIdsGm[sourceOutOffset], classifiedIndex, outputCopy);
#endif
        DataCopyPad(
            topkSlotsGm[slotOutOffset], slotStageLocal_, outputCopy);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        return;
    }
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    uint16_t kthValue = topkOp_.GetLastKthValue();
    VllmFastTopkIndexerClassifyVF::SqueezeIndexerMissTokenIds(
        (__ubuf__ uint32_t *)classifiedIndex.GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(), topkCount_ / CLASSIFY_CHUNK);
    uint32_t currentMissCount = static_cast<uint32_t>(
        AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
#ifdef A5_MTP_CLASSIFY_ONLY
    // Match the A3 classified-row order in Stage 1: one hardware sort places
    // misses first, then orders both miss and hit source IDs. Keep the packed
    // slot payload so no post-publication GM round trip is needed for hits.
    LocalTensor<uint32_t> orderKeys = topkSharedTmpLocal_;
    LocalTensor<uint32_t> orderPayloads = topkSharedTmpLocal_[topkCount_];
    LocalTensor<float> orderScratch =
        topkSharedTmpLocal_[topkCount_ * 2U].template ReinterpretCast<float>();
    if (HasLongSourceTags()) {
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::PackClassifiedLong>(
            (__ubuf__ uint32_t *)orderKeys.GetPhyAddr(),
            (__ubuf__ uint32_t *)orderPayloads.GetPhyAddr(),
            (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
            (__ubuf__ uint16_t *)TagBuffer(lastTagBufferIndex_).GetPhyAddr(),
            topkCount_ / fused_quant_li_manage_hit_sort_vf::LANES);
    } else {
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::PackClassified>(
            (__ubuf__ uint32_t *)orderKeys.GetPhyAddr(),
            (__ubuf__ uint32_t *)orderPayloads.GetPhyAddr(),
            (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
            topkCount_ / fused_quant_li_manage_hit_sort_vf::LANES);
    }
    PipeBarrier<PIPE_V>();
    AscendC::Sort<float, true>(
        mrgValueLocal_.template ReinterpretCast<float>(),
        orderKeys.template ReinterpretCast<float>(), orderPayloads,
        orderScratch, topkCount_ / 32U);
    PipeBarrier<PIPE_V>();
    if (HasLongSourceTags()) {
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::UnpackClassifiedLong>(
            (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
            (__ubuf__ uint32_t *)slotStageLocal_.GetPhyAddr(),
            (__ubuf__ uint32_t *)mrgValueLocal_.GetPhyAddr(),
            topkCount_ / fused_quant_li_manage_hit_sort_vf::LANES);
    } else {
        asc_vf_call<fused_quant_li_manage_hit_sort_vf::UnpackClassified>(
            (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
            (__ubuf__ uint32_t *)slotStageLocal_.GetPhyAddr(),
            (__ubuf__ uint32_t *)mrgValueLocal_.GetPhyAddr(),
            topkCount_ / fused_quant_li_manage_hit_sort_vf::LANES);
    }
    PipeBarrier<PIPE_V>();
    // Stage 3's private union consumes (key, source-id) miss pairs. Rebuild
    // that representation after decoding the ordered public source row.
    BuildSortedMissPairs(
        indicesOutLocal_.template ReinterpretCast<int32_t>(),
        currentMissCount);
    LocalTensor<int32_t> routeMissCountLocal =
        candidatePayloadLocal_.template ReinterpretCast<int32_t>();
    routeMissCountLocal.SetValue(0, static_cast<int32_t>(currentMissCount));
    // Stage 3 consumes one threshold per route after the kernel-wide
    // producer-MTE3 -> SyncAll -> consumer-MTE2 hand-off. Publish a full
    // 32-byte row so the consumer never needs an unaligned scalar GM load.
    constexpr uint32_t MTP_THRESHOLD_STRIDE = 16U;
    Duplicate(scoreOutLocal_, kthValue, MTP_THRESHOLD_STRIDE);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::DataCopyParams routeScalarCopy{
        1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    AscendC::DataCopyParams slotOutputCopy{
        1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
#ifdef A5_MTP_SLOT_OUTPUT_STRIDE
    const uint64_t slotOutOffset =
        static_cast<uint64_t>(outputRow) * A5_MTP_SLOT_OUTPUT_STRIDE;
#else
    const uint64_t slotOutOffset =
        static_cast<uint64_t>(outputRow) * topkCount_;
#endif
    // All sources are disjoint and stay live until TOPK_MTE3_V_EVENT is
    // consumed before the next route (or by FreeEventID at phase end).
    DataCopyPad(
        missCountGm[static_cast<uint64_t>(fixedWorkspaceRow) *
            fused_quant_li_manage_workspace::ROUTE_COUNT_STRIDE],
        routeMissCountLocal, routeScalarCopy);
    if (currentMissCount > 0U) {
        AscendC::DataCopyParams pairOutputCopy{
            1,
            static_cast<uint16_t>(currentMissCount * 2U * sizeof(int32_t)),
            0, 0};
        const uint64_t pairOutOffset =
            static_cast<uint64_t>(fixedWorkspaceRow) * topkCount_ * 2U;
        DataCopyPad(
            indiceOutGm[pairOutOffset],
            mrgValueLocal_.template ReinterpretCast<int32_t>(),
            pairOutputCopy);
    }
    DataCopy(
        thresholdGm[static_cast<uint64_t>(fixedWorkspaceRow) *
                    MTP_THRESHOLD_STRIDE],
        scoreOutLocal_, MTP_THRESHOLD_STRIDE);
    DataCopyPad(topkSourceIdsGm[slotOutOffset],
        indicesOutLocal_.template ReinterpretCast<int32_t>(), slotOutputCopy);
    DataCopyPad(topkSlotsGm[slotOutOffset], slotStageLocal_, slotOutputCopy);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    return;
#endif

    VllmFastTopkIndexerClassifyVF::SqueezeIndexerHitTokenIds(
        (__ubuf__ uint32_t *)classifiedIndex[currentMissCount].GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(),
        topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    uint32_t candidateCount = 0;
#if LI_UPDATE_ABLATION_MODE >= 4
    if (currentMissCount > 0) {
        candidateCount = FindVictimsOnDemand(
            info, scoreRowOffset, validS2Len, currentMissCount,
            kthValue, compactPayloadLocal);
    }
#endif

    Duplicate(slotStageLocal_, static_cast<int32_t>(-1), topkCount_);
    PipeBarrier<PIPE_V>();
    VllmFastTopkIndexerClassifyVF::SqueezeIndexerHitSlots(
        (__ubuf__ uint32_t *)slotStageLocal_[currentMissCount].GetPhyAddr(),
        (__ubuf__ uint32_t *)indicesOutLocal_.GetPhyAddr(), topkCount_ / CLASSIFY_CHUNK);
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_S>(V_MTE2_EVENT3);
    WaitFlag<HardEvent::V_S>(V_MTE2_EVENT3);

    const uint64_t cacheBase =
        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize;
    if (currentMissCount > 0 && candidateCount >= currentMissCount) {
        for (uint32_t i = 0; i < currentMissCount; ++i) {
            uint32_t payload = candidatePayloadLocal_.GetValue(i);
            uint32_t slot =
                (payload >> vllm_fast_hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_SHIFT) &
                vllm_fast_hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_SLOT_MASK;
            uint32_t evictToken = payload &
                vllm_fast_hist_topk_index_update_a5_payload::INDEXER_PAYLOAD_TOKEN_MASK;
            uint32_t missToken =
                static_cast<uint32_t>(classifiedIndex.GetValue(i));
            slotStageLocal_.SetValue(i, static_cast<int32_t>(slot));
#if LI_UPDATE_ABLATION_MODE >= 5
            cacheSlotsGm.SetValue(cacheBase + evictToken,
                lim_pool_contract::INVALID_POOL_SLOT);
            cacheSlotsGm.SetValue(cacheBase + missToken,
                                  static_cast<int32_t>(slot));
#else
            (void)evictToken;
            (void)missToken;
#endif
        }
#if LI_UPDATE_ABLATION_MODE >= 5
        PipeBarrier<PIPE_ALL>();
#endif
    }

    LocalTensor<int32_t> missCountLocal =
        candidatePayloadLocal_.template ReinterpretCast<int32_t>();
    missCountLocal.SetValue(0, static_cast<int32_t>(currentMissCount));
    SetFlag<HardEvent::S_MTE3>(EVENT_ID1);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID1);
    AscendC::DataCopyParams scalarCopy{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
    DataCopyPad(missCountGm[info.bIdx], missCountLocal, scalarCopy);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID1);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    AscendC::DataCopyParams outputCopy{
        1, static_cast<uint16_t>(topkCount_ * sizeof(int32_t)), 0, 0};
    uint64_t outOffset = static_cast<uint64_t>(outputRow) * topkCount_;
    DataCopyPad(indiceOutGm[outOffset], classifiedIndex, outputCopy);
    DataCopyPad(topkSlotsGm[outOffset], slotStageLocal_, outputCopy);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(
    const VllmFastQLICommon::RunInfo &info, bool allowFullSlotPrefetch,
    uint32_t chunkStart, uint32_t chunkLen,
    uint32_t chunkIdx, uint32_t maxChunks)
{
    const bool localCandidates = chunkLen != 0U;
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    LocalTensor<int32_t> slotPrefetchLocal = slotStageLocal_;
    uint32_t slotPrefetchCap = topkCount_;
    if (allowFullSlotPrefetch) {
        slotPrefetchLocal = resMm1UB_.template ReinterpretCast<int32_t>();
        // 预取 cap 必须钳到 resMm1Buf_ 的 int32 容量以内。
        // QuantLI 的 mBaseSize = 2*gSize（tSize<=64），缓冲只有 BF16 LI（4*gSize）
        // 的一半（32768B）；若按 trunkLen_=16384 个 int32（65536B）预取会溢出，
        // 覆盖后续 UB 缓冲（kScale/qScale/out/mrgValue），把分数流前段污染成
        // slot 数据（-1 -> 0xFFFF 最大键），导致 topk 选择错误集合。
        // 超出部分由 RunSidecarPayload 的 LoadCurrentSlotsCompactRange 分块补载，
        // 语义不变。
        slotPrefetchCap = 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_;
    }
    int32_t cuRealAcSeq = info.actS2Size;
    if (info.causalClip) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }
    if (localCandidates) {
        cuRealAcSeq -= static_cast<int32_t>(chunkStart);
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset =
            blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;

        uint16_t zero = 0;
        int32_t neg = -1;
        if (info.causalClip) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        if (localCandidates && validS2Len > static_cast<int32_t>(chunkLen)) {
            validS2Len = static_cast<int32_t>(chunkLen);
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                                          indicesOutLocal_.ReinterpretCast<int32_t>(),
                                                          copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = VllmFastQLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(uint16_t); // byte
                AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * VllmFastQLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + chunkStart],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                if (info.denseDirectPublish) {
                    topkOp_.RunTokenPayloadOnly(
                        mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                        chunkStart, validS2LenAlign, 0U, 1U);
                } else {
                    topkOp_.RunAblationStage(
                        mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                        slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                        static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                        chunkStart, topkCount_, validS2LenAlign,
                        static_cast<uint32_t>(validS2Len), 0, 1,
                        true, slotPrefetchCap, info.offloadedLen, info.cacheTokenCount);
                }
                GatherCurrentSourceTags(0U, 0U, chunkStart);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(uint16_t); // byte
                        AscendC::DataCopyPad(
                            mrgValueLocal_,
                            scoreGm[vecOffset * VllmFastQLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + chunkStart],
                            copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                        if (info.denseDirectPublish) {
                            topkOp_.RunTokenPayloadOnly(
                                mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                                chunkStart, trunkLen_, loopIdx, s2LoopNum);
                        } else {
                            topkOp_.RunAblationStage(
                                mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                                slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                                static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                                chunkStart, topkCount_, trunkLen_, trunkLen_,
                                loopIdx, s2LoopNum, true, slotPrefetchCap, info.offloadedLen, info.cacheTokenCount);
                        }
                        GatherCurrentSourceTags(loopIdx, 0U, chunkStart);
                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len
                                                                               ? validS2Len % trunkLen_
                                                                               :trunkLen_;
                    uint32_t offset = vecOffset *
                                 VllmFastQLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 chunkStart + loopIdx * trunkLen_;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                             topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(uint16_t); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256],
                                        zero, VllmFastQLICommon::Align(validTrunkLen,
                                        (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    const uint32_t mergedValidLen =
                        VllmFastQLICommon::Align(
                            topkCountAlign256_ + validTrunkLen,
                            static_cast<uint32_t>(256));
                    if (info.denseDirectPublish) {
                        topkOp_.RunTokenPayloadOnly(
                            mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            chunkStart + loopIdx * trunkLen_, mergedValidLen,
                            loopIdx, s2LoopNum);
                    } else {
                        topkOp_.RunAblationStage(
                            mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            slotPrefetchLocal, slotStageLocal_, cacheSlotsGm,
                            static_cast<uint64_t>(info.cacheRowIdx) * constInfo_.cacheSlotsSize,
                            chunkStart + loopIdx * trunkLen_, topkCount_, mergedValidLen,
                            validTrunkLen, loopIdx, s2LoopNum,
                            true, slotPrefetchCap, info.offloadedLen, info.cacheTokenCount);
                    }
                    GatherCurrentSourceTags(
                        loopIdx, topkCount_, chunkStart + loopIdx * trunkLen_);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
        }

        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }

        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64],
                                        neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        uint32_t outputRow = static_cast<uint32_t>(info.indiceOutOffset / topkCount_) +
                             curS1Idx + rowIdx;
        uint64_t scoreRowOffset = vecOffset *
            VllmFastQLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);
        if (localCandidates) {
            // Preserve the packed {slot, token} payload. The reducer needs
            // both arrays in the same survivor order; public rows are written
            // only after the global merge.
            const uint64_t candidateOffset =
                (static_cast<uint64_t>(info.workspaceRow + curS1Idx + rowIdx) *
                     maxChunks + chunkIdx) * topkCount_;
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyParams scoreCopy{
                1, static_cast<uint16_t>(topkCount_ * sizeof(uint16_t)), 0, 0};
            DataCopyPad(candidateScoresGm[candidateOffset], scoreOutLocal_,
                        scoreCopy);
            DataCopyPad(candidatePayloadsGm[candidateOffset],
                        indicesOutLocal_, copyOutParams);
            if (HasLongSourceTags()) {
                DataCopyPad(candidateTagsGm[candidateOffset],
                            TagBuffer(lastTagBufferIndex_), scoreCopy);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            continue;
        }
#if LI_UPDATE_ABLATION_MODE < 2
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(
            indiceOutGm[static_cast<uint64_t>(outputRow) * topkCount_],
            indicesOutLocal_.template ReinterpretCast<int32_t>(),
            copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
#elif LI_UPDATE_ABLATION_MODE == 2
        // P2 deliberately exports the packed survivor payload. Performance
        // tests ignore the diagnostic output; avoiding an extra unpack keeps
        // P2-P1 focused on payload construction and chunk matching.
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(
            indiceOutGm[static_cast<uint64_t>(outputRow) * topkCount_],
            indicesOutLocal_.template ReinterpretCast<int32_t>(),
            copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
#else
        FinalizePayloadUpdate(info, outputRow, scoreRowOffset,
                              static_cast<uint32_t>(validS2Len));
#endif
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ReduceLocalTopK(
    const VllmFastQLICommon::RunInfo &info,
    uint32_t segmentCount, uint32_t candidateStride,
    uint32_t rowBegin, uint32_t rowEnd)
{
    const int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    const int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size
        ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    AscendC::DataCopyExtParams scoreCopy{
        1, static_cast<uint32_t>(topkCount_ * sizeof(uint16_t)), 0, 0, 0};
    AscendC::DataCopyExtParams payloadCopy{
        1, static_cast<uint32_t>(topkCount_ * sizeof(uint32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint16_t> scorePad{false, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> payloadPad{false, 0, 0, 0};
    // Merge the score and packed {slot, token} payload together. The
    // reference C8 LD admits eight 2048-element score groups per 16K
    // window. A 128-row MM tile also has 64 KiB of reusable payload space,
    // enough for eight groups; the 64-row tile fits four groups.
    const uint32_t ldSegmentsPerRound =
        constInfo_.mBaseSize >= 128U
            ? (HasLongSourceTags() ? 4U : 8U)
            : (HasLongSourceTags() ? 2U : 4U);
    LocalTensor<uint32_t> roundPayloadLocal =
        resMm1UB_.template ReinterpretCast<uint32_t>();
    LocalTensor<uint16_t> roundTagsLocal =
        roundPayloadLocal[ldSegmentsPerRound * topkCount_]
            .template ReinterpretCast<uint16_t>();
    const uint32_t roundCount =
        (segmentCount + ldSegmentsPerRound - 1U) /
        ldSegmentsPerRound;

    for (uint32_t rowIdx = rowBegin;
         rowIdx < rowEnd && rowIdx < static_cast<uint32_t>(curS1ProcNum);
         ++rowIdx) {
        const uint32_t vecOffset = rowIdx;
        const uint64_t candidateRow =
            static_cast<uint64_t>(info.workspaceRow + curS1Idx + rowIdx) *
            candidateStride * topkCount_;
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t round = 0U; round < roundCount; ++round) {
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
            const uint32_t previousLen = round == 0U ? 0U : topkCount_;
            if (previousLen != 0U) {
                DataCopy(mrgValueLocal_, scoreOutLocal_, topkCount_);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
            }
            const uint32_t firstSegment =
                round * ldSegmentsPerRound;
            const uint32_t segmentsInRound =
                segmentCount - firstSegment < ldSegmentsPerRound
                    ? segmentCount - firstSegment : ldSegmentsPerRound;
            for (uint32_t j = 0U; j < segmentsInRound; ++j) {
                const uint64_t candidateOffset = candidateRow +
                    static_cast<uint64_t>(firstSegment + j) * topkCount_;
                DataCopyPad(
                    mrgValueLocal_[previousLen + j * topkCount_],
                    candidateScoresGm[candidateOffset], scoreCopy, scorePad);
                DataCopyPad(
                    roundPayloadLocal[j * topkCount_],
                    candidatePayloadsGm[candidateOffset],
                    payloadCopy, payloadPad);
                if (HasLongSourceTags()) {
                    DataCopyPad(
                        roundTagsLocal[j * topkCount_],
                        candidateTagsGm[candidateOffset], scoreCopy, scorePad);
                }
            }
            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            topkOp_.RunExistingPayload(
                mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                roundPayloadLocal,
                previousLen + segmentsInRound * topkCount_,
                previousLen, round, roundCount);
            if (HasLongSourceTags()) {
                topkOp_.GatherSourceTags(
                    TagBuffer(round + 1U), TagBuffer(round),
                    roundTagsLocal, previousLen, 0U, true);
                lastTagBufferIndex_ = (round + 1U) & 1U;
            }
            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
        }
        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        const uint32_t outputRow =
            static_cast<uint32_t>(info.indiceOutOffset / topkCount_) +
            curS1Idx + rowIdx;
        const uint64_t scoreRowOffset = static_cast<uint64_t>(vecOffset) *
            VllmFastQLICommon::Align(
                static_cast<uint64_t>(constInfo_.kSeqSize),
                static_cast<uint64_t>(s2BaseSize_));
        const uint32_t visibleLen = info.causalClip
            ? static_cast<uint32_t>(
                info.actS2SizeOrig - info.actS1Size + curS1Idx + rowIdx + 1)
            : info.actS2Size;
        FinalizePayloadUpdate(info, outputRow, scoreRowOffset, visibleLen);
    }
}

}  // namespace VllmFastQLIKernel
#endif
