/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_sparse_attention_overlap_service_vector_mla.h
 * \brief
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_VECTOR_MLA_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_VECTOR_MLA_H

// 优化探针：批量 topk 索引预取（已回退，设为 0）。
// 开启后 MergeKv 主循环不再逐 token 标量读 topkGm_，改为进入循环前一次 DataCopy 进 UB。
// 实测会引入精度问题（疑似 UB 缓冲区与 plan/valid size 数据冲突），暂时关闭，后续排查。
#define FSAO_PROBE_BULK_TOPK 0

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_sparse_attention_overlap_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FusedSparseAttentionOverlapTraits> class FusedSparseAttentionOverlapVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename FusedSparseAttentionOverlapTraits::kvType;
    using OUT_T = typename FusedSparseAttentionOverlapTraits::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline FusedSparseAttentionOverlapVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const FusedSparseAttentionOverlapTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<KV_T> &kvMergeGm,
                                                const GlobalTensor<KV_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitSelectionUpdateGlobalTensor(
        const GlobalTensor<KV_T> &selectionKRopeGm, const GlobalTensor<KV_T> &selectionKvCacheGm,
        const GlobalTensor<int32_t> &selectionKvBlockTableGm,
        const GlobalTensor<int32_t> &selectionKvBlockStatusGm,
        const GlobalTensor<int16_t> &selectionMembershipMapGm,
        const GlobalTensor<int32_t> &selectionKvActualSeqGm, int64_t selectionKvBlockSize,
        int64_t selectionMaxBlockNum, int64_t selectionTopkBlockSize,
        int64_t selectionStatusStride, int64_t selectionMembershipStride,
        bool enableSelectionUpdate);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline void MergeKv(const RunInfo &runInfo);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline bool CanUsePairedKvCopy(
        int64_t realS2Idx1, int64_t realS2Idx2,
        int64_t keyOffset1, int64_t keyOffset2, int64_t s2IdLimit,
        const RunInfo &runInfo, int64_t &keySrcStride, int64_t &keyRopeSrcStride);
    __aicore__ inline bool CanUsePairedSelectionCopy(
        int64_t selectionTokenOffset1, int64_t selectionTokenOffset2,
        int64_t &kvSrcStride, int64_t &ropeSrcStride);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline int64_t GetSelectionRow(const RunInfo &runInfo);
    __aicore__ inline int64_t GetSelectionTokenOffset(int64_t selectionRow, int64_t topkPos,
                                                       int32_t topkValue, int32_t currentStatus,
                                                       int64_t &cachedBlockTableIdx, int32_t &cachedBlockNum);
    __aicore__ inline int64_t GetSelectionSlotTokenOffset(
        int64_t selectionRow, int32_t selectionSlot,
        int64_t &cachedBlockTableIdx, int32_t &cachedBlockNum);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, int64_t selectionTokenOffset1,
                                    int64_t selectionTokenOffset2, int64_t s2IdLimit,
                                    const RunInfo &runInfo,
                                    int64_t &ubTokenIdx1, int64_t &ubTokenIdx2);
    __aicore__ inline void FlushDirectSelectionUpdate(
        const int64_t *updateDstSlot, const int64_t *updateUbIdx, int64_t &updateCount,
        bool &collectEnabled, int64_t mte3Size, int64_t mte2Size, int64_t mergeMte3Idx,
        int64_t selectionRow, bool batchCollected);
    __aicore__ inline void CopyInSelectionKvRun(int64_t &mte2Size, int64_t mte3Size,
                                                int64_t mergeMte3Idx, int64_t selectionTokenOffset,
                                                int64_t tokenCount);
    __aicore__ inline void CopyInSelectionKvPair(
        int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
        int64_t selectionTokenOffset, int64_t kvSrcStride, int64_t ropeSrcStride);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyOutSelectionUpdate(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                                  int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyOutSelectionUpdateFromKvMerge(const RunInfo &runInfo);
    __aicore__ inline void CopyOutSparseSelectionUpdateFromKvMerge(const RunInfo &runInfo);
    __aicore__ inline bool IsSelectionUpdateEnabled() const;
    __aicore__ inline bool UseAllCoreSelectionUpdate() const;
    __aicore__ inline bool UsePipelineSelectionUpdate() const;
    __aicore__ inline void SnapshotSelectionSourceReadyAtEntry();
    __aicore__ inline void RunAllCoreSelectionUpdate();
    __aicore__ inline bool UseSetResidentSelection() const;
    __aicore__ inline void ProcessSetResidentSelectionRow(
        int64_t selectionRow, uint32_t batchIdx, int64_t s2IdLimit);
    __aicore__ inline bool IsPositionResidentSelectionHit(
        LocalTensor<int32_t> currentTopkLocal,
        LocalTensor<int32_t> residentStatusLocal,
        LocalTensor<int32_t> scratchLocal,
        int32_t previousValidCount);
    __aicore__ inline bool IsTokenSetResidentSelectionHit(
        LocalTensor<int32_t> currentTopkLocal,
        LocalTensor<int16_t> membershipStorageLocal,
        LocalTensor<uint32_t> membershipByteOffsetLocal,
        LocalTensor<int16_t> gatheredMembershipLocal,
        LocalTensor<int32_t> gatheredMembershipInt32Local,
        int64_t membershipBase, int32_t previousValidCount, int64_t s2IdLimit,
        bool &membershipSlotMapLoaded);
    __aicore__ inline bool IsSelectionMembershipMapReady(
        int64_t membershipBase, LocalTensor<int16_t> controlLocal);
    __aicore__ inline void ClearSelectionUpdatePlanMarker(int64_t membershipBase);
    __aicore__ inline void WriteSelectionUpdatePlan(
        int64_t membershipBase, int32_t planCount, int32_t selectionHitCount,
        LocalTensor<int16_t> membershipStorageLocal, int64_t planOffset,
        bool preserveMembershipMap);
    __aicore__ inline void WriteSparseSelectionUpdatePlan(
        int64_t membershipBase, int32_t updateCount,
        LocalTensor<int16_t> membershipStorageLocal, int64_t planOffset);
    __aicore__ inline void PublishTokenSetResidentSelectionMap(
        LocalTensor<int32_t> residentStatusLocal,
        LocalTensor<int16_t> membershipStorageLocal,
        int64_t membershipBase, int32_t validTopkNum, int64_t s2IdLimit);
    __aicore__ inline int32_t BuildDenseResidentSelectionPlan(
        LocalTensor<int32_t> gatheredSlotLocal,
        LocalTensor<int32_t> insertStatusLocal,
        LocalTensor<int32_t> hitSourceLocal);
    __aicore__ inline int16_t EncodeSelectionPlanValue(int32_t planValue) const;
    __aicore__ inline bool IsSelectionPlanHit(int16_t planValue) const;
    __aicore__ inline bool IsSelectionPlanUpdate(int16_t planValue) const;
    __aicore__ inline int32_t DecodeSelectionPlanSlot(int16_t planValue) const;
    __aicore__ inline int32_t BuildSetResidentSelectionPlan(
        int64_t selectionRow, int64_t selectionGroupBaseRow, int64_t s2IdLimit,
        LocalTensor<int32_t> currentTopkLocal,
        LocalTensor<int32_t> residentStatusLocal,
        LocalTensor<int32_t> sourceStatusLocal,
        LocalTensor<uint32_t> indexLocal,
        LocalTensor<int32_t> insertStatusLocal,
        LocalTensor<int32_t> hitSourceLocal,
        LocalTensor<int32_t> sortBufferLocal);
    __aicore__ inline void GatherValidSelectionTopk(
        LocalTensor<int32_t> currentTopkLocal,
        LocalTensor<int32_t> scratch0Local,
        LocalTensor<uint32_t> scratch1Local,
        LocalTensor<uint32_t> scratch2Local,
        int32_t maxValidTokenId, int32_t &validTopkNum);
    __aicore__ inline void SortSelectionTopk(
        LocalTensor<int32_t> sourceLocal, LocalTensor<uint32_t> indexLocal,
        LocalTensor<float> tempLocal, LocalTensor<float> sortedLocal,
        LocalTensor<int32_t> sortedTopkLocal,
        LocalTensor<uint32_t> sortedTopkIndexLocal, int32_t validNum);
    __aicore__ inline void FindSelectionTopkHit(
        LocalTensor<int32_t> sortedTopkLocal,
        LocalTensor<uint32_t> sortedTopkIndexLocal,
        LocalTensor<int32_t> sortedStatusLocal,
        LocalTensor<uint32_t> sortedStatusIndexLocal,
        LocalTensor<int32_t> insertStatusLocal,
        LocalTensor<int32_t> hitSourceLocal,
        int32_t validTopkNum, bool sameRow, int64_t sourceRow,
        int32_t &maxSameRowHitSlot);
    __aicore__ inline void CopySelectionUpdateTokenFromSelectionCache(
        int64_t selectionRow, int64_t destinationSlot, int64_t encodedSourceSlot);
    __aicore__ inline void MergeKvFromSelection(const RunInfo &runInfo);
    __aicore__ inline void MergeKvFromSelectionWithSparseUpdates(const RunInfo &runInfo);
    __aicore__ inline uint64_t GetActualQSeqLenForSelectionUpdate(uint32_t batchIdx);
    __aicore__ inline uint64_t GetActualKVSeqLenForSelectionUpdate(uint32_t batchIdx);
    __aicore__ inline void CopySelectionUpdateTokenFromFullCache(
        int64_t selectionRow, uint32_t batchIdx, int64_t topkPos, int32_t topkValue, int64_t s2IdLimit);
    __aicore__ inline void UpdateSelectionRange(
        int64_t selectionRow, uint32_t batchIdx, int64_t tokenStart, int64_t tokenEnd, int64_t s2IdLimit,
        bool appendActualSeq = false);
    __aicore__ inline void RefreshSelectionStatusRange(
        int64_t topkBase, int64_t statusBase, int64_t tokenStart, int64_t tokenEnd);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t selectionTokenOffset,
                                          int64_t s2IdLimit, const RunInfo &runInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void AmlaVecCompute(const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb,
                                          LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ProcessAmlaNupdate(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void ReduceSplitKV(uint32_t slotBegin, uint32_t partNum, uint32_t mCount,
                                         uint64_t attenOutOffset);
    // ====== helper：空闲核的 V0/V1 帮搬 MergeKv ======
    __aicore__ inline void InitHelperSync(const GlobalTensor<int32_t> &doneGm,
                                          const GlobalTensor<int32_t> &creditGm,
                                          uint32_t pairs);
    __aicore__ inline void InitHelperV0(const GlobalTensor<KV_T> &targetKvMergeGm, uint32_t pairIdx);
    __aicore__ inline void SetHelperParts(uint32_t partIdx, uint32_t partCount,
                                          const GlobalTensor<int32_t> &helperValidSizeGm,
                                          uint32_t pairIdx);
    __aicore__ inline bool IsHelperV0() const { return isHelperV0_; }
    __aicore__ inline void PublishHelperCredit(uint32_t loop);
    __aicore__ inline void WaitHelperCredit(uint32_t loop);
    __aicore__ inline void SignalMergeDone(uint32_t loop);
    __aicore__ inline void WaitMergeDone(uint32_t loop);
    __aicore__ inline void FixValidSize(const RunInfo &info);
    __aicore__ inline void ZeroHelperSyncArea();
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素�?
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

    // ====== helper：空闲核帮搬 MergeKv 的核间握手参数 ======
    // 环形槽 64 个，每槽 32B（8 个 int32）：DataCopyPad 和 SetAtomicAdd 都要求 32B 粒度。
    // ring 远大于 kvMergeGm_ 的 4 槽流水深度，helper 又被 credit 通道卡住领先幅度，
    // 不会绕回冲突。事件 id 用 2：本文件现有代码只用了 0/1/4/5。
    static constexpr uint32_t HELPER_SYNC_RING = 64;
    static constexpr uint32_t HELPER_SYNC_SLOT_INT32 = 8;
    static constexpr uint32_t HELPER_SIGNALS_PER_BLOCK = 2;   // helper 的 V0、V1 子核各发一次
    static constexpr uint32_t HELPER_SYNC_EVENT_ID = 2;
    static constexpr uint32_t HELPER_POLL_BACKOFF = 128;      // 轮询退避拍数（≈0.3μs）

private:
    static constexpr bool PAGE_ATTENTION = FusedSparseAttentionOverlapTraits::pageAttention;
    static constexpr int TEMPLATE_MODE = FusedSparseAttentionOverlapTraits::templateMode;
    static constexpr bool FLASH_DECODE = FusedSparseAttentionOverlapTraits::flashDecode;
    static constexpr FusedSparseAttentionOverlapLayout LAYOUT_T = FusedSparseAttentionOverlapTraits::layout;
    static constexpr FusedSparseAttentionOverlapLayout KV_LAYOUT_T = FusedSparseAttentionOverlapTraits::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr int64_t SELECTION_STATUS_UB_OFFSET = 512;
    static constexpr int32_t SELECTION_MAX_TOPK = 2048;
    static constexpr int32_t SELECTION_SORT_UNIT = 32;
    static constexpr int32_t SELECTION_COMPARE_SCALAR_NUM = 256 / sizeof(int32_t);
    static constexpr int32_t SELECTION_COMPARE_MASK_UNIT = 16;
    static constexpr int32_t SELECTION_SORT_BUFFER_COUNT = 8;
    static constexpr int32_t SELECTION_POSITION_PROBE_COUNT = 4;
    static constexpr int32_t SELECTION_MEMBERSHIP_MAX_TOKEN = 16376;
    static constexpr int32_t SELECTION_MEMBERSHIP_MAP_INT16_COUNT =
        SELECTION_MEMBERSHIP_MAX_TOKEN;
    static constexpr int16_t SELECTION_MEMBERSHIP_READY_MARKER = 0x5A4D;
    static constexpr int16_t SELECTION_PLAN_READY_MARKER = 0x5A50;
    static constexpr int16_t SELECTION_SPARSE_PLAN_READY_MARKER = 0x5A53;
    static constexpr int16_t SELECTION_EXTERNAL_PLAN_READY_MARKER = 0x5A45;
    static constexpr int16_t SELECTION_DIRECT_LAYOUT_MARKER = 0x5A44;
    static constexpr int16_t SELECTION_PAIRED_COPY_MARKER = 0x5A56;
    static constexpr int32_t SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT = 8;
    static constexpr int32_t SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT =
        BYTE_BLOCK / sizeof(int16_t);
    static constexpr int32_t SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT =
        ((SELECTION_MEMBERSHIP_MAP_INT16_COUNT +
          SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1) /
         SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT) *
        SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT;
    static constexpr int32_t SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT =
        ((SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT +
          SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT +
          SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT - 1) /
         SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT) *
        SELECTION_MEMBERSHIP_ALIGNMENT_INT16_COUNT;
    static constexpr int16_t SELECTION_COMPACT_PLAN_INVALID = 0;
    static constexpr int32_t SELECTION_PLAN_UPDATE_FLAG = 1 << 30;
    static constexpr int32_t SELECTION_PLAN_HIT_FLAG = 1 << 29;
    static constexpr int32_t SELECTION_PLAN_DESTINATION_BITS = 11;
    static constexpr int32_t SELECTION_PLAN_SLOT_MASK =
        (1 << SELECTION_PLAN_DESTINATION_BITS) - 1;
    static constexpr int32_t SELECTION_PLAN_HIT_SLOT_MASK = SELECTION_PLAN_HIT_FLAG - 1;
    static constexpr int32_t SELECTION_SYNC_COPY_CAPACITY = 64;
    static constexpr int32_t SELECTION_SYNC_COPY_THRESHOLD = 64;
    static constexpr int32_t SELECTION_SPARSE_PLAN_VALUE_COUNT =
        SELECTION_SYNC_COPY_CAPACITY * 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;

    const FusedSparseAttentionOverlapTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;
    GlobalTensor<KV_T> selectionKRopeGm_;
    GlobalTensor<KV_T> selectionKvCacheGm_;
    GlobalTensor<int32_t> selectionKvBlockTableGm_;
    GlobalTensor<int32_t> selectionKvBlockStatusGm_;
    GlobalTensor<int16_t> selectionMembershipMapGm_;
    GlobalTensor<int32_t> selectionKvActualSeqGm_;
    int64_t selectionKvBlockSize_ = 0;
    int64_t selectionMaxBlockNum_ = 0;
    int64_t selectionTopkBlockSize_ = 0;
    int64_t selectionStatusStride_ = 0;
    int64_t selectionMembershipStride_ = 0;
    bool enableSelectionUpdate_ = false;
    bool useAllCoreSelectionUpdate_ = false;
    bool selectionSourceReadyAtEntry_ = false;
    bool selectionUpdatePlanActive_ = false;
    bool selectionSparseUpdatePlanActive_ = false;
    int64_t selectionUpdatePlanOffset_ = 0;
    int32_t selectionUpdatePlanCount_ = 0;
    int64_t selectionDataRow_ = -1;
    int64_t selectionDirectRowStride_ = 0;
    bool selectionPairedCopyActive_ = false;
    // 第 k 位 = 第 k 个 32-token 窗口的 update 条目已经在 MergeKv 里趁数据还在 UB
    // 时写完了，CopyOutSelectionUpdateFromKvMerge 据此跳过。窗口编号两边都用
    // 「相对 s2GmStartOffset 的偏移 >> 5」。
    // ⚠ 收集用的两个 32 元素数组必须是**局部**变量（放在 MergeKv 里、按指针传给
    // FlushDirectSelectionUpdate）。放成成员时实测全命中点标量 +30.6 μs、
    // 搬运 +25.3 μs —— 那一批 update 条目为零、收集器一次都没写，涨的纯粹是
    // 「这两个数组存在」的代价。回写路的 updateDstSlot[32] 也是局部的。
    uint32_t directWrittenWindowMask_ = 0;

    // ================================Local Buffer�?===================================
    TBuf<> inputBuff1;            // 64K
    TBuf<> inputBuff2;            // 16K
    TBuf<> outputBuff1;           // 32K
    TBuf<> outputBuff2;           // 4K

    TBuf<> tmpBuff1;              // 32K
    TBuf<> v0ValidSizeBuff;       // 8K
    TBuf<> helperSyncBuff;        // 2K，helper 握手 scratch（与所有计算 buffer 无重叠）

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;

    // ====== helper：空闲核帮搬 MergeKv ======
    bool isHelperV0_ = false;          // 本核是 helper（V0/V1 子核同值，名字沿用最初设计）
    uint32_t helperPartIdx_ = 0;       // 半区内第几份：target=0，helper=1
    uint32_t helperPartCount_ = 1;     // 半区分几份：1=无 helper（原行为），2=target+helper 各半
    bool mergeSplitActive_ = false;    // 本轮 MergeKv 是否真的切了（selection 路径不切）
    GlobalTensor<int32_t> mergeDoneGm_;       // done 计数环形槽：helper 原子加，target 轮询
    GlobalTensor<int32_t> mergeCreditGm_;     // credit 槽：target 发布，helper 轮询
    GlobalTensor<int32_t> helperValidSizeGm_; // target 用：helper 核的 kvValidSizeGm_（回收部分计数）
    int32_t helperCreditSeen_ = 0;            // helper 已读到的信用水位（缓存，覆盖本轮就免轮询）
    uint32_t helperSyncPairs_ = 1;            // 握手 ring 的 pair 数（ring 按 pair 隔离）
    uint32_t mergeSyncPairOffset_ = 0;        // 本 pair 的 ring 起点（int32）= pairIdx*RING*SLOT
};

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K * 2);  // 2:pingpong
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // helper 握手 scratch，只有 V 模板会用到；C 模板不分配，不占它的 UB 预算
        pipe->InitBuffer(helperSyncBuff, ConstInfo::BUFFER_SIZE_BYTE_2K);
    }

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();
    ropeMergUb_ = inputBuff2.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitParams(const struct ConstInfo &constInfo,
                                                 const FusedSparseAttentionOverlapTilingDataMla *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitVec0GlobalTensor(
    const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<KV_T> &kvMergeGm,
    const GlobalTensor<KV_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitSelectionUpdateGlobalTensor(
    const GlobalTensor<KV_T> &selectionKRopeGm, const GlobalTensor<KV_T> &selectionKvCacheGm,
    const GlobalTensor<int32_t> &selectionKvBlockTableGm,
    const GlobalTensor<int32_t> &selectionKvBlockStatusGm,
    const GlobalTensor<int16_t> &selectionMembershipMapGm,
    const GlobalTensor<int32_t> &selectionKvActualSeqGm, int64_t selectionKvBlockSize,
    int64_t selectionMaxBlockNum, int64_t selectionTopkBlockSize,
    int64_t selectionStatusStride, int64_t selectionMembershipStride,
    bool enableSelectionUpdate)
{
    this->selectionKRopeGm_ = selectionKRopeGm;
    this->selectionKvCacheGm_ = selectionKvCacheGm;
    this->selectionKvBlockTableGm_ = selectionKvBlockTableGm;
    this->selectionKvBlockStatusGm_ = selectionKvBlockStatusGm;
    this->selectionMembershipMapGm_ = selectionMembershipMapGm;
    this->selectionKvActualSeqGm_ = selectionKvActualSeqGm;
    this->selectionKvBlockSize_ = selectionKvBlockSize;
    this->selectionMaxBlockNum_ = selectionMaxBlockNum;
    this->selectionTopkBlockSize_ = selectionTopkBlockSize;
    int64_t minimumStatusStride = static_cast<int64_t>(constInfo.sparseBlockCount) + 1;
    this->selectionStatusStride_ = selectionStatusStride >= minimumStatusStride ?
        selectionStatusStride : minimumStatusStride;
    this->selectionMembershipStride_ =
        selectionMembershipStride >= SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT ?
        selectionMembershipStride : SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT;
    this->enableSelectionUpdate_ = enableSelectionUpdate;
    this->useAllCoreSelectionUpdate_ = enableSelectionUpdate;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                    GlobalTensor<UPDATE_T> vec2ResGm,
                                                                    GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ElewiseCompute(const RunInfo &info,
                                                                            const LocalTensor<T> &mmResUb,
                                                                            uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判�?
        uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t s2ProcessSize = info.actualSingleProcessSInnerSize;
        int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
        int64_t s2Mid = CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        if (s2Mid > s2ProcessSize) {
            s2Mid = s2ProcessSize;
        }
        if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
            int64_t s2StartCeilAlign = CeilAlign(s2ValidSizeFirstPart, 8);
            int64_t s2MidFloorAlign = s2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景�?s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导�?s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, dealRowCount, columnCount, s2ValidSizeFirstPart,
                        s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
        }
        if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导�?s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景�?s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导�?s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t s2StartCeilAlign = CeilAlign(s2Mid + s2ValidSizeSecondPart, 8);
            int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, s2Mid + s2ValidSizeSecondPart,
                        s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2EndFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign, s2ProcessSize);
        }
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                                         uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分�?inf, endId、startId为endId一个blk内部的下�?
    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1 << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId += 8) {
        Duplicate(mmResUb[rowId * columnCount + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, CeilDiv(columnCount, 8), 0);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                                       uint32_t dealRowCount, uint32_t columnCount,
                                                                       uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分�?inf, startId、endId�?2B对齐的下�?
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId++) {
        Duplicate(mmResUb[rowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, FUSED_SPARSE_ATTENTION_OVERLAP_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = FusedSparseAttentionOverlapAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        pipe_barrier(PIPE_V);
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::AmlaVecCompute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = dealRowCount;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    pipe_barrier(PIPE_V);
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    pipe_barrier(PIPE_V);

    uint32_t prOutIdx = (info.loop - 1) % (constInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (info.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, ConstInfo::FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    pipe_barrier(PIPE_V);
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, ConstInfo::FLOAT_ZERO, calCount);
    pipe_barrier(PIPE_V);

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    pipe_barrier(PIPE_V);
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    pipe_barrier(PIPE_V);
    Muls(nTmp, nTmp, LN2, calCount);
    pipe_barrier(PIPE_V);
    Exp(nTmp, nTmp, calCount);
    pipe_barrier(PIPE_V);
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount);       // fp32->fp16/bf16
    pipe_barrier(PIPE_V);
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    pipe_barrier(PIPE_V);
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }
    if (actualColumnCount == 0) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        return;
    }
    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
    pipe_barrier(PIPE_V);
    RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (info.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount);       // cof_0=1
        pipe_barrier(PIPE_V);
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount);    // 1 / cof(i)
    } else {
        pipe_barrier(PIPE_V);
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    pipe_barrier(PIPE_V);

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, ConstInfo::FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount); // cof(i - 1) / cof(i) - 1
    pipe_barrier(PIPE_V);
    Muls(epsUb, epsUb, (T)1.5, calCount);    // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    pipe_barrier(PIPE_V);
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    pipe_barrier(PIPE_V);
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    pipe_barrier(PIPE_V);
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    pipe_barrier(PIPE_V);

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow]; // 缓存nUpdate

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
    pipe_barrier(PIPE_V);

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t loopId)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);

    pipe_barrier(PIPE_V);
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    pipe_barrier(PIPE_V);
    AmlaVecCompute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                    info.actualSingleProcessSInnerSize);

    pipe_barrier(PIPE_V);
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessAmlaNupdate(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (info.isFirstSInnerLoop) {
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize = 64U;     // tmpQue size 32KB，一次只能处�?4个N，最大保存的数据大小�?4*128*sizeof(int32)
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B

    uint32_t subMSize = FusedSparseAttentionOverlapAlign(mSplitInfo.vecDealM, 16U);
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t);      // 单个datablock的元素数，int32_t类型的为32/4=8
    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize; // 尾块

    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }
        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // (m,1)单次brcb扩充�?m,8), 重复16�? 扩充�?m,128)
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock],
                 nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(dGroupSize / elementPerBlock), // 单次迭代内，目的操作数不同datablock间地址步长,单位为datablock
                  static_cast<uint16_t>(dGroupSize)});                 // 相邻迭代间，目的操作数相同datablock地址步长
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                              (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + loop * mSplitSize) * constInfo.headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE; // 每个block�?28个元素，单位�?2B
        dataCopyParams.srcStride = 0;                                            // 前面一个数据块的尾与后面数据块的头的间�?
        dataCopyParams.dstStride = static_cast<uint16_t>((constInfo.headDim - dGroupSize) *
                                                         sizeof(int32_t) / ONE_BLOCK_SIZE); // 单位�?2B
        for (uint32_t i = 0; i < constInfo.headDim / dGroupSize; i++) {          // 4=512/128
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize] ,tmpQue, dataCopyParams);
        }
        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最�?12, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(loopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进�?
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign, i);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                            int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int64_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetSelectionRow(const RunInfo &runInfo)
{
    int64_t qTokenOffset = 0;
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        uint64_t actualSeqQPrefixSum =
            (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(runInfo.bIdx - 1);
        qTokenOffset = static_cast<int64_t>(actualSeqQPrefixSum) +
                       static_cast<int64_t>(runInfo.gS1Idx / constInfo.gSize);
    } else {
        qTokenOffset = static_cast<int64_t>(runInfo.bIdx) * static_cast<int64_t>(constInfo.qSeqSize) +
                       static_cast<int64_t>(runInfo.gS1Idx / constInfo.gSize);
    }
    return qTokenOffset * static_cast<int64_t>(constInfo.kvHeadNum) +
           static_cast<int64_t>(runInfo.n2Idx);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int64_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetSelectionTokenOffset(
    int64_t selectionRow, int64_t topkPos, int32_t topkValue, int32_t currentStatus,
    int64_t &cachedBlockTableIdx, int32_t &cachedBlockNum)
{
    if (topkValue < 0 || currentStatus != topkValue) {
        return -1;
    }

    return GetSelectionSlotTokenOffset(
        selectionRow, static_cast<int32_t>(topkPos), cachedBlockTableIdx, cachedBlockNum);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int64_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetSelectionSlotTokenOffset(
    int64_t selectionRow, int32_t selectionSlot,
    int64_t &cachedBlockTableIdx, int32_t &cachedBlockNum)
{
    if (selectionSlot < 0) {
        return -1;
    }
    if (selectionDirectRowStride_ > 0) {
        if (selectionSlot >= selectionDirectRowStride_) {
            return -1;
        }
        return selectionRow * selectionDirectRowStride_ + selectionSlot;
    }
    if (selectionSlot >= static_cast<int32_t>(constInfo.sparseBlockCount)) {
        return -1;
    }

    int64_t selectionBlockTableIdx = selectionSlot / selectionKvBlockSize_;
    if (selectionBlockTableIdx >= selectionMaxBlockNum_) {
        return -1;
    }
    if (selectionBlockTableIdx != cachedBlockTableIdx) {
        cachedBlockTableIdx = selectionBlockTableIdx;
        cachedBlockNum = selectionKvBlockTableGm_.GetValue(
            selectionRow * selectionMaxBlockNum_ + selectionBlockTableIdx);
    }
    if (cachedBlockNum < 0) {
        return -1;
    }
    return static_cast<int64_t>(cachedBlockNum) * selectionKvBlockSize_ +
           selectionSlot % selectionKvBlockSize_;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int64_t FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetKeyGmOffset(int64_t realS2Idx,
                                                                 const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        realKeyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.kvCacheBlockSize) *
                                static_cast<int64_t>(constInfo.kvHeadNum) +
                                blkTableOffset;
    } else {
        realKeyGmOffset = (runInfo.tensorBOffset +
                           realS2Idx * constInfo.kvHeadNum * constInfo.headDim) /
                           constInfo.headDim;
    }
    return realKeyGmOffset;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int64_t FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetKeyRopeGmOffset(int64_t realS2Idx,
                                                                 const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyRopeGmOffset = 0;
    realKeyRopeGmOffset = (runInfo.tensorBRopeOffset +
                           realS2Idx * constInfo.kvHeadNum * constInfo.headDimRope) /
                           constInfo.headDimRope;
    return realKeyRopeGmOffset;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CanUsePairedKvCopy(
    int64_t realS2Idx1, int64_t realS2Idx2,
    int64_t keyOffset1, int64_t keyOffset2, int64_t s2IdLimit,
    const RunInfo &runInfo, int64_t &keySrcStride, int64_t &keyRopeSrcStride)
{
    keySrcStride = 0;
    keyRopeSrcStride = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blockTableSrcStride =
            ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
              (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize);
        keySrcStride = blockTableSrcStride * constInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride = blockTableSrcStride * constInfo.headDimRope * sizeof(KV_T);
    } else {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, runInfo, s2IdLimit);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, runInfo, s2IdLimit);
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                         (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) *
                       constInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride = ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) :
                             (keyRopeOffset2 - keyRopeOffset1)) - constInfo.sparseBlockSize) *
                           constInfo.headDimRope * sizeof(KV_T);
    }

    return keySrcStride < INT32_MAX && keySrcStride >= 0 &&
        (PAGE_ATTENTION || (keyRopeSrcStride < INT32_MAX && keyRopeSrcStride >= 0)) &&
        realS2Idx1 + constInfo.sparseBlockSize < s2IdLimit &&
        realS2Idx2 + constInfo.sparseBlockSize < s2IdLimit;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CanUsePairedSelectionCopy(
    int64_t selectionTokenOffset1, int64_t selectionTokenOffset2,
    int64_t &kvSrcStride, int64_t &ropeSrcStride)
{
    if (selectionDirectRowStride_ <= 0 ||
        selectionTokenOffset1 < 0 || selectionTokenOffset2 <= selectionTokenOffset1) {
        return false;
    }
    int64_t skippedTokens = selectionTokenOffset2 - selectionTokenOffset1 - 1;
    kvSrcStride = skippedTokens * constInfo.headDim * sizeof(KV_T);
    ropeSrcStride = skippedTokens * constInfo.headDimRope * sizeof(KV_T);
    return kvSrcStride < INT32_MAX && ropeSrcStride < INT32_MAX;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                       int64_t keyBNBOffset, int64_t selectionTokenOffset,
                                       int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (selectionTokenOffset < 0 && keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    if (selectionTokenOffset >= 0) {
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * constInfo.headDim],
                    selectionKvCacheGm_[selectionTokenOffset * constInfo.headDim], intriParams, padParams);
    } else {
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * constInfo.headDim],
                    keyGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
    }
    intriParams.blockLen = validS2Count * constInfo.headDimRope * sizeof(KV_T);

    if (selectionTokenOffset >= 0) {
        DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * constInfo.headDimRope],
                    selectionKRopeGm_[selectionTokenOffset * constInfo.headDimRope], intriParams, padParams);
    } else {
        DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * constInfo.headDimRope],
                    keyRopeGm_[keyBNBOffset * constInfo.headDimRope], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyInSelectionKvRun(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t selectionTokenOffset,
    int64_t tokenCount)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = static_cast<uint32_t>(tokenCount);
    copyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * constInfo.headDim],
                selectionKvCacheGm_[selectionTokenOffset * constInfo.headDim], copyParams, padParams);

    copyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * constInfo.headDimRope],
                selectionKRopeGm_[selectionTokenOffset * constInfo.headDimRope], copyParams, padParams);
    mte2Size += tokenCount;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyInSelectionKvPair(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t selectionTokenOffset, int64_t kvSrcStride, int64_t ropeSrcStride)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 2;
    copyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    copyParams.srcStride = kvSrcStride;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(
        kvMergUb_[mergeMte3Idx % 2 * 32 * 512 +
                  (mte2Size - mte3Size) * constInfo.headDim],
        selectionKvCacheGm_[selectionTokenOffset * constInfo.headDim],
        copyParams, padParams);

    copyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    copyParams.srcStride = ropeSrcStride;
    DataCopyPad(
        ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 +
                    (mte2Size - mte3Size) * constInfo.headDimRope],
        selectionKRopeGm_[selectionTokenOffset * constInfo.headDimRope],
        copyParams, padParams);
    mte2Size += 2;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                        int64_t realS2Idx1, int64_t realS2Idx2,
                                                        int64_t selectionTokenOffset1,
                                                        int64_t selectionTokenOffset2, int64_t s2IdLimit,
                                                        const RunInfo &runInfo,
                                                        int64_t &ubTokenIdx1, int64_t &ubTokenIdx2)
{
    // ubTokenIdxN = 这个 token 落在 UB 半区里的第几格，−1 = 没写入或不需要。
    // 只有**未命中**的 token 才可能是 update 条目（IsSelectionPlanHit 与
    // IsSelectionPlanUpdate 互斥），命中的一律留 −1。
    //
    // 换位在这里是现成的：配对搬运按 keyOffset 小的那个起头，谁在前当场就知道。
    // 回写路 CopyOutSelectionUpdateFromKvMerge 是事后从 GM 重读 topk 和块表反推
    // 同一件事（优化 02 把那四次读包进了条件，仍要读）——从这里直接带出去，
    // 那段倒推可以整个不做。
    ubTokenIdx1 = -1;
    ubTokenIdx2 = -1;
    bool validS2Idx1 = realS2Idx1 >= 0 && realS2Idx1 < s2IdLimit;
    bool validS2Idx2 = realS2Idx2 >= 0 && realS2Idx2 < s2IdLimit;
    bool selectionHit1 = selectionTokenOffset1 >= 0 && validS2Idx1;
    bool selectionHit2 = selectionTokenOffset2 >= 0 && validS2Idx2;
    if (selectionHit1 || selectionHit2) {
        if (selectionHit1 && selectionHit2 && selectionTokenOffset2 == selectionTokenOffset1 + 1) {
            CopyInSelectionKvRun(mte2Size, mte3Size, mergeMte3Idx, selectionTokenOffset1, 2);
            return;
        }
        int64_t kvSrcStride = 0;
        int64_t ropeSrcStride = 0;
        // CanUsePairedSelectionCopy 要求后一个槽位号更大（stride 不能为负）。
        // 槽位打散之后两者的先后是随机的，约一半的对配不上，退成两次
        // CopyInSingleKv = 四条搬运指令，而配上只要两条。
        //
        // 但这两项都是**命中**（selectionTokenOffsetN >= 0 只可能来自
        // IsSelectionPlanHit 为真），而它们在合并缓冲里的先后是可以换的：
        // attention 对 key 的次序不敏感，回写路 CopyOutSelectionUpdateFromKvMerge
        // 只处理未命中条目、跟这两项不相交。所以按槽位号排一下就能让 stride
        // 恒为正，配对总能成 —— 搬运指令从平均 1.5 条/token 降到 1.0。
        int64_t pairLo = selectionTokenOffset1;
        int64_t pairHi = selectionTokenOffset2;
        if (pairHi < pairLo) {
            pairLo = selectionTokenOffset2;
            pairHi = selectionTokenOffset1;
        }
        if (selectionPairedCopyActive_ && selectionHit1 && selectionHit2 &&
            CanUsePairedSelectionCopy(
                pairLo, pairHi,
                kvSrcStride, ropeSrcStride)) {
            CopyInSelectionKvPair(
                mte2Size, mte3Size, mergeMte3Idx, pairLo,
                kvSrcStride, ropeSrcStride);
            return;
        }

        int64_t keyOffset1 = selectionHit1 ? -1 : GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
        int64_t keyOffset2 = selectionHit2 ? -1 : GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
        int64_t singleSlot1 = mte2Size - mte3Size;
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1,
                       selectionHit1 ? selectionTokenOffset1 : -1, s2IdLimit, runInfo);
        int64_t singleSlot2 = mte2Size - mte3Size;
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2,
                       selectionHit2 ? selectionTokenOffset2 : -1, s2IdLimit, runInfo);
        // 两个来源都无效时 CopyInSingleKv 不写、mte2Size 不动，那一格归下一个 token
        if (!selectionHit1 && keyOffset1 >= 0) {
            ubTokenIdx1 = singleSlot1;
        }
        if (!selectionHit2 && keyOffset2 >= 0) {
            ubTokenIdx2 = singleSlot2;
        }
        return;
    }

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    bool usePairedCopy = CanUsePairedKvCopy(
        realS2Idx1, realS2Idx2, keyOffset1, keyOffset2, s2IdLimit,
        runInfo, keySrcStride, keyRopeSrcStride);
    int64_t pairBaseSlot = mte2Size - mte3Size;
    if (unlikely(!usePairedCopy)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原�?条搬运指�?
        int64_t fallbackSlot1 = mte2Size - mte3Size;
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, -1, s2IdLimit, runInfo);
        int64_t fallbackSlot2 = mte2Size - mte3Size;
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, -1, s2IdLimit, runInfo);
        if (keyOffset1 >= 0) {
            ubTokenIdx1 = fallbackSlot1;
        }
        if (keyOffset2 >= 0) {
            ubTokenIdx2 = fallbackSlot2;
        }
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * constInfo.headDim],
                    keyGm_[startGmOffset * constInfo.headDim], intriParams, padParams);

        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDimRope * sizeof(KV_T);
        intriParams.dstStride = 0;
        intriParams.srcStride = keyRopeSrcStride;
        DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * constInfo.headDimRope],
                    keyRopeGm_[startGmOffset * constInfo.headDimRope], intriParams, padParams);
        // keyOffset 小的那个排在前面 —— 这正是回写路事后要反推的那个换位。
        bool pairTransposed = keyOffset1 > -1 && keyOffset2 > -1 && keyOffset2 < keyOffset1;
        if (keyOffset1 > -1) {
            ubTokenIdx1 = pairTransposed ? pairBaseSlot + constInfo.sparseBlockSize : pairBaseSlot;
        }
        if (keyOffset2 > -1) {
            ubTokenIdx2 = (keyOffset1 > -1 && !pairTransposed) ?
                pairBaseSlot + constInfo.sparseBlockSize : pairBaseSlot;
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                 int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                 const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[runInfo.loop % 4 * 512 * 576 + (s2GmStartOffset + mte3Size)*constInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * 32 * 512], dataCopyParams);

    dataCopyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    DataCopyPad(kvMergeGm_[runInfo.loop % 4 * 512 * 576 + 512 * 512 + (s2GmStartOffset + mte3Size) *
                constInfo.headDimRope], ropeMergUb_[mergeMte3Idx % 2 * 32 * 64], dataCopyParams);

    (void)runInfo;
}

// 未命中的 token 取回来时已经在 kvMergUb_ 里待过一趟。CopyOutMrgeResult 把它写进
// kvMergeGm_（计算流要读），而回写路 CopyOutSelectionUpdateFromKvMerge 之后又从
// kvMergeGm_ 读回 UB 才写进 selection 缓存 —— 同一份数据在 UB 里出现两次，中间
// 白跑一个 GM 来回。这里趁数据还在 UB 就把 update 条目直接发出去。
//
// 只接条目密的窗口。条目稀时优化 03 的按需回读只读回真要写的那几个，本来就不亏，
// 留给它走老路 —— 那也是这一刀的零对照臂（命中率 ≥75% 的点结构上一行不变）。
template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::FlushDirectSelectionUpdate(
    const int64_t *updateDstSlot, const int64_t *updateUbIdx, int64_t &updateCount,
    bool &collectEnabled, int64_t mte3Size, int64_t mte2Size, int64_t mergeMte3Idx,
    int64_t selectionRow, bool batchCollected)
{
    int64_t count = updateCount;
    updateCount = 0;
    // 判据与回写路的 sparseWriteback 互为反面，两边各自从同一段 GM 范围数出
    // 同一个 updateCount，不需要通信。
    //
    // ⚠ 收集常开（移植自 yangqiang2018/yq/opt-merge-0723 的 c089d20d）：摘掉
    // 密度预测（原 count*4 > batchTokenCount 才继续收）。原判据在 strided 布局
    // miss=512（hit 75%）恰卡数学边界（每窗 8 条、8*4=32 不大于 32），05 直写
    // 整档失效、退回老路逐窗全扫；常开后 hit 0.75~1.0 段保留 UB 直写快路。
    // 风险（原作者记录）：05 三版迭代期高命中端曾 +2.7~+4.8% 回归（记账成本+
    // 寄存器压力）；窗口化后 flush 点已降到每窗一次，需 A/B 复核，回归复现
    // 则恢复下行密度预测。
    collectEnabled = true;
    // 窗口编号 = 批起点 >> 5，与回写路的 (localOffset − s2GmStartOffset) >> 5 同一套。
    // mte3Size 按定义就是这一批在 GM 里的起点偏移（CopyOutMrgeResult 写到
    // s2GmStartOffset + mte3Size）。
    //
    // ⚠ 起点对齐还不够，批大小也必须恰好 32：只攒了 30 个就落盘的批如果被认下，
    // 回写路会整个跳过 32 个 token 的窗口，多出的那两个条目就漏写。不满 32 的
    // 批（尾批、或有 token 被跳过）一律不认、退回老路，代价只是一个窗口。
    if ((mte3Size & 31) != 0 || mte2Size - mte3Size != 32) {
        return;
    }
    int64_t windowIdx = mte3Size >> 5;
    if (windowIdx >= 32) {
        return;
    }
    if (count <= 0) {
        // 满批零条目也置掩码（移植自 22c2af8f）：批恰 32 蕴含本窗全部 token
        // 有效且落了 UB（无效 token 会 mergeStopped、批必然不满 32）。但
        // count=0 等价于"本窗确无 update 条目"有两个前提，缺一个都是灾难：
        //   1) 本批 plan 真解码过（useSelectionPlanSource）。不成立时 Phase A
        //      根本不读 plan，slotBuf 恒 0，收集是瞎子、count 恒 0——
        //      dense 计划的 control(2) 记的是 **hitCount**（见
        //      WriteSelectionUpdatePlan），全 miss 的冷跑 hitCount=0 正是这种
        //      配置，此时回读路是唯一写入方，任何掩码都是漏写（71cfade2 只挡
        //      了下一条，没挡这条：冷跑只写出每 part 首窗 32 行）。
        //   2) 本批收集真开着。helper 核 directWritebackActive=false，首轮
        //      收集关闭，收集常开（c089）要到本函数第一次执行才重开——那首
        //      轮的 count=0 是"没查"而非"查过没有"。
        // 两个前提合起来就是调用点留底的 batchCollected。count<0 是超 32 条
        // 整批作废的防御分支，同样不置掩码、退回老路兜底。
        if (count == 0 && batchCollected) {
            directWrittenWindowMask_ |= (1U << static_cast<uint32_t>(windowIdx));
        }
        return;
    }

    // MTE2→MTE3 的等待 CopyOutMrgeResult 已经做过（count > 0 蕴含 mte2Size > mte3Size，
    // 那条路上必然执行到那次 SetFlag/WaitFlag），这里不重复。
    int64_t ubBase = mergeMte3Idx % 2 * 32 * 512;
    int64_t ropeUbBase = mergeMte3Idx % 2 * 32 * 64;
    DataCopyExtParams kvParams;
    kvParams.blockLen = constInfo.headDim * sizeof(KV_T);
    kvParams.srcStride = 0;
    kvParams.dstStride = 0;
    DataCopyExtParams ropeParams;
    ropeParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    ropeParams.srcStride = 0;
    ropeParams.dstStride = 0;

    int64_t idx = 0;
    while (idx < count) {
        int64_t destinationSlot = updateDstSlot[idx];
        int64_t destinationBlockTableIdx = destinationSlot / selectionKvBlockSize_;
        int64_t destinationBlockOffset = destinationSlot % selectionKvBlockSize_;
        int32_t destinationBlockNum = selectionKvBlockTableGm_.GetValue(
            selectionRow * selectionMaxBlockNum_ + destinationBlockTableIdx);
        int64_t srcStart = updateUbIdx[idx];
        int64_t runLen = 1;
        while (idx + runLen < count &&
               updateDstSlot[idx + runLen] == destinationSlot + runLen &&
               updateUbIdx[idx + runLen] == srcStart + runLen &&
               destinationBlockOffset + runLen < selectionKvBlockSize_) {
            runLen++;
        }
        if (destinationBlockNum >= 0) {
            kvParams.blockCount = runLen;
            int64_t dstKvAddr =
                (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ +
                 destinationBlockOffset) * static_cast<int64_t>(constInfo.headDim);
            DataCopyPad(selectionKvCacheGm_[dstKvAddr],
                kvMergUb_[ubBase + srcStart * static_cast<int64_t>(constInfo.headDim)],
                kvParams);

            if (constInfo.headDimRope > 0) {
                ropeParams.blockCount = runLen;
                int64_t dstRopeAddr =
                    (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ +
                     destinationBlockOffset) * static_cast<int64_t>(constInfo.headDimRope);
                DataCopyPad(selectionKRopeGm_[dstRopeAddr],
                    ropeMergUb_[ropeUbBase + srcStart * static_cast<int64_t>(constInfo.headDimRope)],
                    ropeParams);
            }
        }
        idx += runLen;
    }
    directWrittenWindowMask_ |= (1U << static_cast<uint32_t>(windowIdx));
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyOutSelectionUpdate(
    int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset, int64_t mergeMte3Idx, const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size || selectionKvBlockSize_ <= 0 || selectionMaxBlockNum_ <= 0 ||
        selectionTopkBlockSize_ != 1 || constInfo.sparseBlockSize != 1) {
        return;
    }

    int64_t tokenCount = mte2Size - mte3Size;
    int64_t logicalSparseOffset =
        static_cast<int64_t>(runInfo.s2Idx) * static_cast<int64_t>(constInfo.s2BaseSize) +
        s2StartGmOffset + mte3Size;
    int64_t selectionRow = GetSelectionRow(runInfo);

    int64_t copiedCount = 0;
    while (copiedCount < tokenCount) {
        int64_t topkPos = logicalSparseOffset + copiedCount;
        int64_t selBlockTableIdx = topkPos / selectionKvBlockSize_;
        int64_t selBlockOffset = topkPos % selectionKvBlockSize_;
        int64_t writeCount = tokenCount - copiedCount;
        int64_t remainingInBlock = selectionKvBlockSize_ - selBlockOffset;
        if (writeCount > remainingInBlock) {
            writeCount = remainingInBlock;
        }

        int32_t selBlockNum = selectionKvBlockTableGm_.GetValue(
            selectionRow * selectionMaxBlockNum_ + selBlockTableIdx);
        if (selBlockNum >= 0) {
            DataCopyExtParams kvParams;
            kvParams.blockCount = writeCount;
            kvParams.blockLen = constInfo.headDim * sizeof(KV_T);
            kvParams.srcStride = 0;
            kvParams.dstStride = 0;

            int64_t dstKvAddr = static_cast<int64_t>(selBlockNum) * selectionKvBlockSize_ *
                                    static_cast<int64_t>(constInfo.headDim) +
                                selBlockOffset * static_cast<int64_t>(constInfo.headDim);
            LocalTensor<KV_T> srcKv =
                kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + copiedCount * static_cast<int64_t>(constInfo.headDim)];
            DataCopyPad(selectionKvCacheGm_[dstKvAddr], srcKv, kvParams);

            DataCopyExtParams ropeParams;
            ropeParams.blockCount = writeCount;
            ropeParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
            ropeParams.srcStride = 0;
            ropeParams.dstStride = 0;

            int64_t dstRopeAddr = static_cast<int64_t>(selBlockNum) * selectionKvBlockSize_ *
                                      static_cast<int64_t>(constInfo.headDimRope) +
                                  selBlockOffset * static_cast<int64_t>(constInfo.headDimRope);
            LocalTensor<KV_T> srcRope =
                ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + copiedCount * static_cast<int64_t>(constInfo.headDimRope)];
            DataCopyPad(selectionKRopeGm_[dstRopeAddr], srcRope, ropeParams);
        }

        int64_t statusAddr = selectionRow * selectionStatusStride_ + topkPos;
        for (int64_t idx = 0; idx < writeCount; idx++) {
            int32_t topkValue = topkGm_.GetValue(runInfo.topKBaseOffset + topkPos + idx);
            selectionKvBlockStatusGm_.SetValue(statusAddr + idx, topkValue);
        }
        copiedCount += writeCount;
    }

    if (logicalSparseOffset == 0 && GetSubBlockIdx() == 0) {
        int32_t actualSeq = static_cast<int32_t>(constInfo.sparseBlockCount);
        selectionKvActualSeqGm_.SetValue(selectionRow, actualSeq);
        selectionKvBlockStatusGm_.SetValue(
            selectionRow * selectionStatusStride_ +
            static_cast<int64_t>(constInfo.sparseBlockCount),
            actualSeq);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::IsSelectionUpdateEnabled() const
{
    return enableSelectionUpdate_;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::UseAllCoreSelectionUpdate() const
{
    return useAllCoreSelectionUpdate_;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::UsePipelineSelectionUpdate() const
{
    return false;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::UseSetResidentSelection() const
{
    return enableSelectionUpdate_ && useAllCoreSelectionUpdate_ && PAGE_ATTENTION &&
        TEMPLATE_MODE == V_TEMPLATE && selectionKvBlockSize_ > 0 && selectionMaxBlockNum_ > 0 &&
        selectionTopkBlockSize_ == 1 && constInfo.sparseBlockSize == 1 &&
        constInfo.sparseBlockCount > 0 && constInfo.sparseBlockCount <= SELECTION_MAX_TOPK &&
        selectionStatusStride_ >= static_cast<int64_t>(constInfo.sparseBlockCount) + 1 &&
        selectionMembershipStride_ >= SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SnapshotSelectionSourceReadyAtEntry()
{
    constexpr int64_t statusElementsPerDataBlock = BYTE_BLOCK / sizeof(int32_t);
    int64_t readyOffset = static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t readyCopyStart = readyOffset / statusElementsPerDataBlock * statusElementsPerDataBlock;
    DataCopyExtParams readyParams;
    readyParams.blockCount = 1;
    readyParams.blockLen = BYTE_BLOCK;
    readyParams.srcStride = 0;
    readyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> readyPadParams{false, 0, 0, 0};
    DataCopyPad(v0ValidSizeUb_, selectionKvBlockStatusGm_[readyCopyStart], readyParams, readyPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(0);
    WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    selectionSourceReadyAtEntry_ = enableSelectionUpdate_ &&
        v0ValidSizeUb_.GetValue(readyOffset - readyCopyStart) ==
            static_cast<int32_t>(constInfo.sparseBlockCount);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline uint64_t FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetActualQSeqLenForSelectionUpdate(uint32_t batchIdx)
{
    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        if (batchIdx > 0) {
            return actualSeqLengthsQGm.GetValue(batchIdx) - actualSeqLengthsQGm.GetValue(batchIdx - 1);
        }
        return actualSeqLengthsQGm.GetValue(0);
    } else {
        if (constInfo.actualLenDimsQ == 0) {
            return constInfo.qSeqSize;
        }
        if (constInfo.actualLenDimsQ == 1) {
            return actualSeqLengthsQGm.GetValue(0);
        }
        return actualSeqLengthsQGm.GetValue(batchIdx);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline uint64_t FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetActualKVSeqLenForSelectionUpdate(uint32_t batchIdx)
{
    if constexpr (KV_LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        if (batchIdx > 0) {
            return actualSeqLengthsKVGm.GetValue(batchIdx) - actualSeqLengthsKVGm.GetValue(batchIdx - 1);
        }
        return actualSeqLengthsKVGm.GetValue(0);
    } else {
        if (constInfo.actualLenDimsKV == 0) {
            return constInfo.kvSeqSize;
        }
        if (constInfo.actualLenDimsKV == 1) {
            return actualSeqLengthsKVGm.GetValue(0);
        }
        return actualSeqLengthsKVGm.GetValue(batchIdx);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopySelectionUpdateTokenFromFullCache(
    int64_t selectionRow, uint32_t batchIdx, int64_t topkPos, int32_t topkValue, int64_t s2IdLimit)
{
    if constexpr (!PAGE_ATTENTION) {
        return;
    }
    if (topkValue < 0 || static_cast<int64_t>(topkValue) >= s2IdLimit) {
        return;
    }

    int64_t selBlockTableIdx = topkPos / selectionKvBlockSize_;
    int32_t selBlockNum = selectionKvBlockTableGm_.GetValue(
        selectionRow * selectionMaxBlockNum_ + selBlockTableIdx);
    if (selBlockNum < 0) {
        return;
    }

    int64_t fullBlockTableIdx = static_cast<int64_t>(topkValue) / constInfo.kvCacheBlockSize;
    int32_t fullBlockNum = blkTableGm_.GetValue(
        static_cast<int64_t>(batchIdx) * constInfo.maxBlockNumPerBatch + fullBlockTableIdx);
    if (fullBlockNum < 0) {
        return;
    }

    int64_t selBlockOffset = topkPos % selectionKvBlockSize_;
    int64_t fullBlockOffset = static_cast<int64_t>(topkValue) % constInfo.kvCacheBlockSize;
    int64_t n2Idx = selectionRow % static_cast<int64_t>(constInfo.kvHeadNum);
    int64_t srcTokenOffset =
        (static_cast<int64_t>(fullBlockNum) * constInfo.kvCacheBlockSize + fullBlockOffset) *
            static_cast<int64_t>(constInfo.kvHeadNum) +
        n2Idx;
    int64_t dstKvAddr = static_cast<int64_t>(selBlockNum) * selectionKvBlockSize_ *
                            static_cast<int64_t>(constInfo.headDim) +
                        selBlockOffset * static_cast<int64_t>(constInfo.headDim);

    DataCopyExtParams kvParams;
    kvParams.blockCount = 1;
    kvParams.blockLen = constInfo.headDim * sizeof(KV_T);
    kvParams.srcStride = 0;
    kvParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_, keyGm_[srcTokenOffset * static_cast<int64_t>(constInfo.headDim)], kvParams, padParams);

    if (constInfo.headDimRope == 0) {
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        DataCopyPad(selectionKvCacheGm_[dstKvAddr], kvMergUb_, kvParams);
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        return;
    }
    int64_t dstRopeAddr = static_cast<int64_t>(selBlockNum) * selectionKvBlockSize_ *
                               static_cast<int64_t>(constInfo.headDimRope) +
                           selBlockOffset * static_cast<int64_t>(constInfo.headDimRope);
    DataCopyExtParams ropeParams;
    ropeParams.blockCount = 1;
    ropeParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    ropeParams.srcStride = 0;
    ropeParams.dstStride = 0;
    DataCopyPad(ropeMergUb_, keyRopeGm_[srcTokenOffset * static_cast<int64_t>(constInfo.headDimRope)],
        ropeParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

    DataCopyPad(selectionKvCacheGm_[dstKvAddr], kvMergUb_, kvParams);
    DataCopyPad(selectionKRopeGm_[dstRopeAddr], ropeMergUb_, ropeParams);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::
CopySelectionUpdateTokenFromSelectionCache(
    int64_t selectionRow, int64_t destinationSlot, int64_t encodedSourceSlot)
{
    int64_t topkCount = static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t sourceRow = encodedSourceSlot / topkCount;
    int64_t sourceSlot = encodedSourceSlot - sourceRow * topkCount;
    if (sourceRow < 0 || sourceSlot < 0 || sourceSlot >= topkCount ||
        (sourceRow == selectionRow && sourceSlot == destinationSlot)) {
        return;
    }

    int64_t sourceBlockTableIdx = sourceSlot / selectionKvBlockSize_;
    int64_t destinationBlockTableIdx = destinationSlot / selectionKvBlockSize_;
    int32_t sourceBlockNum = selectionKvBlockTableGm_.GetValue(
        sourceRow * selectionMaxBlockNum_ + sourceBlockTableIdx);
    int32_t destinationBlockNum = selectionKvBlockTableGm_.GetValue(
        selectionRow * selectionMaxBlockNum_ + destinationBlockTableIdx);
    if (sourceBlockNum < 0 || destinationBlockNum < 0) {
        return;
    }

    int64_t sourceBlockOffset = sourceSlot % selectionKvBlockSize_;
    int64_t destinationBlockOffset = destinationSlot % selectionKvBlockSize_;
    int64_t sourceKvOffset =
        (static_cast<int64_t>(sourceBlockNum) * selectionKvBlockSize_ + sourceBlockOffset) *
        static_cast<int64_t>(constInfo.headDim);
    int64_t destinationKvOffset =
        (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ + destinationBlockOffset) *
        static_cast<int64_t>(constInfo.headDim);

    DataCopyExtParams kvParams;
    kvParams.blockCount = 1;
    kvParams.blockLen = constInfo.headDim * sizeof(KV_T);
    kvParams.srcStride = 0;
    kvParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_, selectionKvCacheGm_[sourceKvOffset], kvParams, padParams);

    DataCopyExtParams ropeParams;
    ropeParams.blockCount = 1;
    ropeParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    ropeParams.srcStride = 0;
    ropeParams.dstStride = 0;
    if (constInfo.headDimRope > 0) {
        int64_t sourceRopeOffset =
            (static_cast<int64_t>(sourceBlockNum) * selectionKvBlockSize_ + sourceBlockOffset) *
            static_cast<int64_t>(constInfo.headDimRope);
        DataCopyPad(ropeMergUb_, selectionKRopeGm_[sourceRopeOffset], ropeParams, padParams);
    }

    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    DataCopyPad(selectionKvCacheGm_[destinationKvOffset], kvMergUb_, kvParams);
    if (constInfo.headDimRope > 0) {
        int64_t destinationRopeOffset =
            (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ + destinationBlockOffset) *
            static_cast<int64_t>(constInfo.headDimRope);
        DataCopyPad(selectionKRopeGm_[destinationRopeOffset], ropeMergUb_, ropeParams);
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::UpdateSelectionRange(
    int64_t selectionRow, uint32_t batchIdx, int64_t tokenStart, int64_t tokenEnd, int64_t s2IdLimit,
    bool appendActualSeq)
{
    int64_t topkBase = selectionRow * static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t statusBase = selectionRow * selectionStatusStride_;
    constexpr int64_t statusElementsPerDataBlock = BYTE_BLOCK / sizeof(int32_t);
    int64_t tokenCount = tokenEnd - tokenStart;
    int64_t statusStart = statusBase + tokenStart;
    LocalTensor<int32_t> statusLocal = v0ValidSizeUb_;
    int64_t statusLocalOffset = 0;
    int64_t statusCopyStart = statusStart;
    int64_t statusCopyCount = tokenCount + static_cast<int64_t>(appendActualSeq);
    bool useAlignedCopy = statusStart % statusElementsPerDataBlock == 0;
    if (!useAlignedCopy) {
        statusCopyStart = statusStart / statusElementsPerDataBlock * statusElementsPerDataBlock;
        statusLocalOffset = statusStart - statusCopyStart;
        statusCopyCount = statusElementsPerDataBlock;
        DataCopyExtParams loadStatusParams;
        loadStatusParams.blockCount = 1;
        loadStatusParams.blockLen = BYTE_BLOCK;
        loadStatusParams.srcStride = 0;
        loadStatusParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(statusLocal, selectionKvBlockStatusGm_[statusCopyStart], loadStatusParams, padParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(1);
        WaitFlag<AscendC::HardEvent::MTE2_S>(1);
    }
    for (int64_t topkPos = tokenStart; topkPos < tokenEnd; topkPos++) {
        int32_t topkValue = topkGm_.GetValue(topkBase + topkPos);
        int64_t localStatusIdx = statusLocalOffset + topkPos - tokenStart;
        int32_t currentStatus = useAlignedCopy ?
            selectionKvBlockStatusGm_.GetValue(statusBase + topkPos) : statusLocal.GetValue(localStatusIdx);
        if (currentStatus != topkValue) {
            CopySelectionUpdateTokenFromFullCache(selectionRow, batchIdx, topkPos, topkValue, s2IdLimit);
        }
        statusLocal.SetValue(localStatusIdx, topkValue);
    }
    if (appendActualSeq) {
        statusLocal.SetValue(statusLocalOffset + tokenCount, static_cast<int32_t>(constInfo.sparseBlockCount));
    }
    if (statusCopyCount > 0) {
        SetFlag<AscendC::HardEvent::S_MTE3>(1);
        WaitFlag<AscendC::HardEvent::S_MTE3>(1);
        DataCopyExtParams statusParams;
        statusParams.blockCount = 1;
        statusParams.blockLen = static_cast<uint32_t>(statusCopyCount * sizeof(int32_t));
        statusParams.srcStride = 0;
        statusParams.dstStride = 0;
        DataCopyPad(selectionKvBlockStatusGm_[statusCopyStart], statusLocal, statusParams);
        SetFlag<AscendC::HardEvent::MTE3_S>(1);
        WaitFlag<AscendC::HardEvent::MTE3_S>(1);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::RefreshSelectionStatusRange(
    int64_t topkBase, int64_t statusBase, int64_t tokenStart, int64_t tokenEnd)
{
    for (int64_t topkPos = tokenStart; topkPos < tokenEnd; topkPos++) {
        selectionKvBlockStatusGm_.SetValue(
            statusBase + topkPos, topkGm_.GetValue(topkBase + topkPos));
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GatherValidSelectionTopk(
    LocalTensor<int32_t> currentTopkLocal, LocalTensor<int32_t> scratch0Local,
    LocalTensor<uint32_t> scratch1Local, LocalTensor<uint32_t> scratch2Local,
    int32_t maxValidTokenId, int32_t &validTopkNum)
{
    int64_t topkCount = static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t topkBlockAlign = CeilDiv(topkCount, static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t))) *
                             static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t));
    int64_t compareNum = CeilDiv(topkBlockAlign, static_cast<int64_t>(SELECTION_COMPARE_SCALAR_NUM)) *
                         static_cast<int64_t>(SELECTION_COMPARE_SCALAR_NUM);

    LocalTensor<float> topkFloatLocal = scratch0Local.ReinterpretCast<float>();
    Cast(topkFloatLocal, currentTopkLocal, RoundMode::CAST_ROUND, topkBlockAlign);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> lowerMaskLocal = scratch1Local.ReinterpretCast<uint8_t>();
    CompareScalar(lowerMaskLocal, topkFloatLocal, -1.0f, CMPMODE::GT, compareNum);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint8_t> upperMaskLocal = scratch2Local.ReinterpretCast<uint8_t>();
    CompareScalar(upperMaskLocal, topkFloatLocal, static_cast<float>(maxValidTokenId), CMPMODE::LE, compareNum);
    PipeBarrier<PIPE_V>();

    LocalTensor<uint16_t> lowerMaskU16 = lowerMaskLocal.ReinterpretCast<uint16_t>();
    LocalTensor<uint16_t> upperMaskU16 = upperMaskLocal.ReinterpretCast<uint16_t>();
    And(lowerMaskU16, lowerMaskU16, upperMaskU16, compareNum / SELECTION_COMPARE_MASK_UNIT);
    PipeBarrier<PIPE_V>();

    uint64_t reservedCount = 0;
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = 1;
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = 8;
    gatherMaskParams.src1RepeatStride = 0;
    LocalTensor<float> compactTopkFloat = currentTopkLocal.ReinterpretCast<float>();
    LocalTensor<uint32_t> maskLocal = lowerMaskLocal.ReinterpretCast<uint32_t>();
    GatherMask(compactTopkFloat, topkFloatLocal, maskLocal, true, topkBlockAlign,
               gatherMaskParams, reservedCount);
    PipeBarrier<PIPE_V>();
    Cast(currentTopkLocal, compactTopkFloat, RoundMode::CAST_ROUND, topkBlockAlign);
    PipeBarrier<PIPE_V>();
    validTopkNum = static_cast<int32_t>(reservedCount);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SortSelectionTopk(
    LocalTensor<int32_t> sourceLocal, LocalTensor<uint32_t> indexLocal,
    LocalTensor<float> tempLocal, LocalTensor<float> sortedLocal,
    LocalTensor<int32_t> sortedTopkLocal, LocalTensor<uint32_t> sortedTopkIndexLocal,
    int32_t validNum)
{
    int64_t sortAlign = CeilDiv(static_cast<int64_t>(constInfo.sparseBlockCount),
                                static_cast<int64_t>(SELECTION_SORT_UNIT)) * SELECTION_SORT_UNIT;
    LocalTensor<float> sourceFloatLocal = sourceLocal.ReinterpretCast<float>();
    if (validNum > 0) {
        Cast(sourceFloatLocal, sourceLocal, RoundMode::CAST_ROUND, validNum);
        PipeBarrier<PIPE_V>();
    }

    int64_t duplicateNum = validNum % SELECTION_SORT_UNIT;
    if (duplicateNum > 0) {
        int64_t duplicateIndex = validNum - duplicateNum;
        uint64_t mask0 = UINT64_MAX;
        mask0 = mask0 << duplicateNum;
        mask0 = mask0 & (UINT64_MAX >> SELECTION_SORT_UNIT);
        uint64_t mask[2] = {mask0, 0};
        Duplicate(sourceFloatLocal[duplicateIndex], -1.0f, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    int64_t duplicateStart = CeilDiv(static_cast<int64_t>(validNum),
                                     static_cast<int64_t>(SELECTION_SORT_UNIT)) * SELECTION_SORT_UNIT;
    if (duplicateStart < sortAlign) {
        Duplicate(sourceFloatLocal[duplicateStart], -1.0f, sortAlign - duplicateStart);
        PipeBarrier<PIPE_V>();
    }

    Concat(sourceFloatLocal, sourceFloatLocal, tempLocal, sortAlign / SELECTION_SORT_UNIT);
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sortedLocal, sourceFloatLocal, indexLocal, tempLocal,
                      sortAlign / SELECTION_SORT_UNIT);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sortedTopkFloatLocal = sortedTopkLocal.ReinterpretCast<float>();
    Extract(sortedTopkFloatLocal, sortedTopkIndexLocal, sortedLocal,
            sortAlign / SELECTION_SORT_UNIT);
    PipeBarrier<PIPE_V>();
    Cast(sortedTopkLocal, sortedTopkFloatLocal, RoundMode::CAST_ROUND, sortAlign);
    PipeBarrier<PIPE_V>();
    Cast(sourceLocal, sourceFloatLocal, RoundMode::CAST_ROUND, sortAlign);
    PipeBarrier<PIPE_V>();
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::FindSelectionTopkHit(
    LocalTensor<int32_t> sortedTopkLocal, LocalTensor<uint32_t> sortedTopkIndexLocal,
    LocalTensor<int32_t> sortedStatusLocal, LocalTensor<uint32_t> sortedStatusIndexLocal,
    LocalTensor<int32_t> insertStatusLocal, LocalTensor<int32_t> hitSourceLocal,
    int32_t validTopkNum, bool sameRow, int64_t sourceRow, int32_t &maxSameRowHitSlot)
{
    int32_t currentIdx = 0;
    int32_t statusIdx = 0;
    int32_t topkCount = static_cast<int32_t>(constInfo.sparseBlockCount);
    while (currentIdx < validTopkNum && statusIdx < topkCount) {
        int32_t currentToken = sortedTopkLocal.GetValue(currentIdx);
        int32_t statusToken = sortedStatusLocal.GetValue(statusIdx);
        if (currentToken < 0 || statusToken < 0) {
            break;
        }
        if (currentToken == statusToken) {
            int32_t currentPosition = static_cast<int32_t>(sortedTopkIndexLocal.GetValue(currentIdx));
            int32_t statusSlot = static_cast<int32_t>(sortedStatusIndexLocal.GetValue(statusIdx));
            if (sameRow) {
                insertStatusLocal.SetValue(statusSlot, currentPosition);
                hitSourceLocal.SetValue(
                    currentPosition, SELECTION_PLAN_HIT_FLAG | statusSlot);
                if (statusSlot > maxSameRowHitSlot) {
                    maxSameRowHitSlot = statusSlot;
                }
            } else if (hitSourceLocal.GetValue(currentPosition) == -1) {
                hitSourceLocal.SetValue(
                    currentPosition, static_cast<int32_t>(sourceRow * topkCount + statusSlot));
            }
            currentIdx++;
            statusIdx++;
        } else if (currentToken > statusToken) {
            currentIdx++;
        } else {
            statusIdx++;
        }
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int32_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::BuildSetResidentSelectionPlan(
    int64_t selectionRow, int64_t selectionGroupBaseRow, int64_t s2IdLimit,
    LocalTensor<int32_t> currentTopkLocal, LocalTensor<int32_t> residentStatusLocal,
    LocalTensor<int32_t> sourceStatusLocal, LocalTensor<uint32_t> indexLocal,
    LocalTensor<int32_t> insertStatusLocal, LocalTensor<int32_t> hitSourceLocal,
    LocalTensor<int32_t> sortBufferLocal)
{
    int64_t topkCount = static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t sortAlign = CeilDiv(topkCount, static_cast<int64_t>(SELECTION_SORT_UNIT)) *
                        static_cast<int64_t>(SELECTION_SORT_UNIT);
    ArithProgression<int32_t>(indexLocal.ReinterpretCast<int32_t>(), 0, 1, sortAlign);
    Duplicate(insertStatusLocal, -1, sortAlign);
    Duplicate(hitSourceLocal, -1, sortAlign);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> sortedTopkLocal = sortBufferLocal;
    LocalTensor<uint32_t> sortedTopkIndexLocal =
        sortBufferLocal[sortAlign].ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> sortedStatusLocal = sortBufferLocal[sortAlign * 2];
    LocalTensor<uint32_t> sortedStatusIndexLocal =
        sortBufferLocal[sortAlign * 3].ReinterpretCast<uint32_t>();

    int32_t validTopkNum = static_cast<int32_t>(topkCount);
    if (s2IdLimit < topkCount) {
        GatherValidSelectionTopk(
            currentTopkLocal, sortedTopkLocal, sortedTopkIndexLocal, sortedStatusIndexLocal,
            static_cast<int32_t>(s2IdLimit - 1), validTopkNum);
    }

    LocalTensor<float> tempLocal = sortedTopkLocal.ReinterpretCast<float>();
    LocalTensor<float> sortedLocal = sortBufferLocal[sortAlign * 4].ReinterpretCast<float>();
    SortSelectionTopk(currentTopkLocal, indexLocal, tempLocal, sortedLocal,
                      sortedTopkLocal, sortedTopkIndexLocal, validTopkNum);

    tempLocal = sortedStatusLocal.ReinterpretCast<float>();
    SortSelectionTopk(residentStatusLocal, indexLocal, tempLocal, sortedLocal,
                      sortedStatusLocal, sortedStatusIndexLocal,
                      static_cast<int32_t>(topkCount));
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);

    int32_t maxSameRowHitSlot = -1;
    FindSelectionTopkHit(sortedTopkLocal, sortedTopkIndexLocal,
                         sortedStatusLocal, sortedStatusIndexLocal,
                         insertStatusLocal, hitSourceLocal, validTopkNum,
                         true, selectionRow, maxSameRowHitSlot);

    int64_t topkBlockAlign = CeilDiv(topkCount,
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t));
    for (uint32_t sourceHead = 0; sourceHead < constInfo.kvHeadNum; sourceHead++) {
        int64_t sourceRow = selectionGroupBaseRow + static_cast<int64_t>(sourceHead);
        if (sourceRow == selectionRow) {
            continue;
        }
        DataCopyExtParams statusParams;
        statusParams.blockCount = 1;
        statusParams.blockLen = static_cast<uint32_t>(topkCount * sizeof(int32_t));
        statusParams.srcStride = 0;
        statusParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> statusPadParams{
            true, 0, static_cast<uint8_t>(topkBlockAlign - topkCount), -1};
        DataCopyPad(sourceStatusLocal,
            selectionKvBlockStatusGm_[sourceRow * selectionStatusStride_],
            statusParams, statusPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(0);
        WaitFlag<AscendC::HardEvent::MTE2_S>(0);

        tempLocal = sortedStatusLocal.ReinterpretCast<float>();
        SortSelectionTopk(sourceStatusLocal, indexLocal, tempLocal, sortedLocal,
                          sortedStatusLocal, sortedStatusIndexLocal,
                          static_cast<int32_t>(topkCount));
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
        FindSelectionTopkHit(sortedTopkLocal, sortedTopkIndexLocal,
                             sortedStatusLocal, sortedStatusIndexLocal,
                             insertStatusLocal, hitSourceLocal, validTopkNum,
                             false, sourceRow, maxSameRowHitSlot);
    }

    int32_t slotsToRelease = maxSameRowHitSlot + 1 - validTopkNum;
    if (slotsToRelease > 0) {
        int32_t scannedSlots = 0;
        for (int32_t slot = maxSameRowHitSlot; slot >= 0 && scannedSlots < slotsToRelease;
             slot--, scannedSlots++) {
            int32_t currentPosition = insertStatusLocal.GetValue(slot);
            if (currentPosition >= 0) {
                hitSourceLocal.SetValue(
                    currentPosition, static_cast<int32_t>(selectionRow * topkCount + slot));
                insertStatusLocal.SetValue(slot, -1);
            }
        }
    }
    return validTopkNum;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::IsPositionResidentSelectionHit(
    LocalTensor<int32_t> currentTopkLocal,
    LocalTensor<int32_t> residentStatusLocal,
    LocalTensor<int32_t> scratchLocal,
    int32_t previousValidCount)
{
    int32_t topkCount = static_cast<int32_t>(constInfo.sparseBlockCount);
    if (previousValidCount != topkCount) {
        return false;
    }

    int32_t probeCount = topkCount < SELECTION_POSITION_PROBE_COUNT ?
        topkCount : SELECTION_POSITION_PROBE_COUNT;
    for (int32_t position = 0; position < probeCount; position++) {
        if (currentTopkLocal.GetValue(position) != residentStatusLocal.GetValue(position)) {
            return false;
        }
    }

    int64_t topkBlockAlign = CeilDiv(static_cast<int64_t>(topkCount),
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t));
    LocalTensor<float> currentTopkFloat = scratchLocal.ReinterpretCast<float>();
    LocalTensor<float> residentStatusFloat = scratchLocal[topkBlockAlign].ReinterpretCast<float>();
    LocalTensor<float> differenceFloat = scratchLocal[topkBlockAlign * 2].ReinterpretCast<float>();
    LocalTensor<float> reduceWorkLocal = scratchLocal[topkBlockAlign * 3].ReinterpretCast<float>();

    Cast(currentTopkFloat, currentTopkLocal, RoundMode::CAST_ROUND, topkCount);
    Cast(residentStatusFloat, residentStatusLocal, RoundMode::CAST_ROUND, topkCount);
    PipeBarrier<PIPE_V>();
    Sub(differenceFloat, currentTopkFloat, residentStatusFloat, topkCount);
    PipeBarrier<PIPE_V>();
    Abs(differenceFloat, differenceFloat, topkCount);
    PipeBarrier<PIPE_V>();
    ReduceMax(currentTopkFloat, differenceFloat, reduceWorkLocal, topkCount);
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);
    return currentTopkFloat.GetValue(0) == 0.0f;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::IsSelectionMembershipMapReady(
    int64_t membershipBase, LocalTensor<int16_t> controlLocal)
{
    if (selectionMembershipStride_ < SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT) {
        return false;
    }

    DataCopyExtParams controlParams;
    controlParams.blockCount = 1;
    controlParams.blockLen = SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT * sizeof(int16_t);
    controlParams.srcStride = 0;
    controlParams.dstStride = 0;
    DataCopyPadExtParams<int16_t> controlPadParams{false, 0, 0, 0};
    DataCopyPad(controlLocal,
        selectionMembershipMapGm_[membershipBase +
            SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT],
        controlParams, controlPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(0);
    WaitFlag<AscendC::HardEvent::MTE2_S>(0);
    return controlLocal.GetValue(0) == SELECTION_MEMBERSHIP_READY_MARKER;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ClearSelectionUpdatePlanMarker(
    int64_t membershipBase)
{
    selectionMembershipMapGm_.SetValue(
        membershipBase + SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT + 1,
        -1);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::IsTokenSetResidentSelectionHit(
    LocalTensor<int32_t> currentTopkLocal,
    LocalTensor<int16_t> membershipStorageLocal,
    LocalTensor<uint32_t> membershipByteOffsetLocal,
    LocalTensor<int16_t> gatheredMembershipLocal,
    LocalTensor<int32_t> gatheredMembershipInt32Local,
    int64_t membershipBase, int32_t previousValidCount, int64_t s2IdLimit,
    bool &membershipSlotMapLoaded)
{
    membershipSlotMapLoaded = false;
    int32_t topkCount = static_cast<int32_t>(constInfo.sparseBlockCount);
    if (previousValidCount != topkCount || s2IdLimit < topkCount ||
        s2IdLimit > SELECTION_MEMBERSHIP_MAX_TOKEN ||
        selectionMembershipStride_ < SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT) {
        return false;
    }
    if (!IsSelectionMembershipMapReady(
            membershipBase,
            membershipStorageLocal[SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT])) {
        return false;
    }

    int64_t mapCopyIntCount = CeilDiv(
        s2IdLimit, static_cast<int64_t>(BYTE_BLOCK / sizeof(int16_t))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int16_t));
    DataCopyExtParams metadataParams;
    metadataParams.blockCount = 1;
    metadataParams.blockLen = static_cast<uint32_t>(mapCopyIntCount * sizeof(int16_t));
    metadataParams.srcStride = 0;
    metadataParams.dstStride = 0;
    DataCopyPadExtParams<int16_t> metadataPadParams{false, 0, 0, 0};
    DataCopyPad(membershipStorageLocal,
        selectionMembershipMapGm_[membershipBase],
        metadataParams, metadataPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(0);
    WaitFlag<AscendC::HardEvent::MTE2_S>(0);

    Muls(membershipByteOffsetLocal.ReinterpretCast<int32_t>(), currentTopkLocal,
        static_cast<int32_t>(sizeof(int16_t)), topkCount);
    PipeBarrier<PIPE_V>();
    Gather(gatheredMembershipLocal, membershipStorageLocal,
        membershipByteOffsetLocal, 0, topkCount);
    PipeBarrier<PIPE_V>();
    membershipSlotMapLoaded = true;

    int64_t topkBlockAlign = CeilDiv(
        static_cast<int64_t>(topkCount),
        static_cast<int64_t>(BYTE_BLOCK / sizeof(half))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(half));
    int64_t scratchOffset = mapCopyIntCount;
    if (scratchOffset + topkBlockAlign * 2 +
            static_cast<int64_t>(BYTE_BLOCK / sizeof(int16_t)) >
        static_cast<int64_t>(ConstInfo::BUFFER_SIZE_BYTE_32K * 2 / sizeof(int16_t))) {
        return false;
    }
    LocalTensor<half> gatheredHalfLocal =
        membershipStorageLocal[scratchOffset].ReinterpretCast<half>();
    LocalTensor<half> reduceResultLocal = gatheredHalfLocal[topkBlockAlign];
    LocalTensor<half> reduceWorkLocal =
        reduceResultLocal[BYTE_BLOCK / sizeof(half)];
    Cast(gatheredHalfLocal, gatheredMembershipLocal, RoundMode::CAST_NONE, topkCount);
    PipeBarrier<PIPE_V>();
    ReduceMin(reduceResultLocal, gatheredHalfLocal, reduceWorkLocal, topkCount);
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);
    bool allHit =
        (reduceResultLocal.ReinterpretCast<uint16_t>().GetValue(0) & 0x8000U) == 0;
    if (!allHit) {
        Cast(gatheredMembershipInt32Local, gatheredHalfLocal,
             RoundMode::CAST_RINT, topkCount);
        PipeBarrier<PIPE_V>();
    }
    return allHit;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int16_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::EncodeSelectionPlanValue(
    int32_t planValue) const
{
    if (planValue >= 0 && (planValue & SELECTION_PLAN_HIT_FLAG) != 0) {
        return static_cast<int16_t>((planValue & SELECTION_PLAN_HIT_SLOT_MASK) + 1);
    }
    if (planValue >= 0 && (planValue & SELECTION_PLAN_UPDATE_FLAG) != 0) {
        return static_cast<int16_t>(-((planValue & SELECTION_PLAN_SLOT_MASK) + 1));
    }
    return SELECTION_COMPACT_PLAN_INVALID;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::IsSelectionPlanHit(
    int16_t planValue) const
{
    return planValue > 0;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline bool
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::IsSelectionPlanUpdate(
    int16_t planValue) const
{
    return planValue < 0;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int32_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::DecodeSelectionPlanSlot(
    int16_t planValue) const
{
    return planValue > 0 ? static_cast<int32_t>(planValue) - 1 :
        -static_cast<int32_t>(planValue) - 1;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline int32_t
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::BuildDenseResidentSelectionPlan(
    LocalTensor<int32_t> gatheredSlotLocal,
    LocalTensor<int32_t> insertStatusLocal,
    LocalTensor<int32_t> hitSourceLocal)
{
    int32_t topkCount = static_cast<int32_t>(constInfo.sparseBlockCount);
    int64_t sortAlign = CeilDiv(
        static_cast<int64_t>(topkCount), static_cast<int64_t>(SELECTION_SORT_UNIT)) *
        static_cast<int64_t>(SELECTION_SORT_UNIT);
    Duplicate(insertStatusLocal, -1, sortAlign);
    Duplicate(hitSourceLocal, -1, sortAlign);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);

    for (int32_t topkPosition = 0; topkPosition < topkCount; topkPosition++) {
        int32_t slotPlusOne = gatheredSlotLocal.GetValue(topkPosition);
        if (slotPlusOne <= 0 || slotPlusOne > topkCount) {
            continue;
        }
        int32_t sourceSlot = slotPlusOne - 1;
        insertStatusLocal.SetValue(sourceSlot, topkPosition);
        hitSourceLocal.SetValue(
            topkPosition, SELECTION_PLAN_HIT_FLAG | sourceSlot);
    }
    return topkCount;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::PublishTokenSetResidentSelectionMap(
    LocalTensor<int32_t> residentStatusLocal,
    LocalTensor<int16_t> membershipStorageLocal,
    int64_t membershipBase, int32_t validTopkNum, int64_t s2IdLimit)
{
    if (selectionMembershipStride_ < SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT) {
        return;
    }

    if (s2IdLimit > SELECTION_MEMBERSHIP_MAX_TOKEN) {
        LocalTensor<int16_t> compactControlLocal =
            membershipStorageLocal[SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT];
        Duplicate(compactControlLocal, static_cast<int16_t>(-1),
                  SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
        compactControlLocal.SetValue(0, SELECTION_MEMBERSHIP_READY_MARKER);
        SetFlag<AscendC::HardEvent::S_MTE3>(0);
        WaitFlag<AscendC::HardEvent::S_MTE3>(0);

        DataCopyExtParams controlParams;
        controlParams.blockCount = 1;
        controlParams.blockLen =
            SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT * sizeof(int16_t);
        controlParams.srcStride = 0;
        controlParams.dstStride = 0;
        DataCopyPad(selectionMembershipMapGm_[membershipBase +
                        SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT],
                    compactControlLocal, controlParams);
        SetFlag<AscendC::HardEvent::MTE3_S>(0);
        WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        return;
    }

    Duplicate(membershipStorageLocal, static_cast<int16_t>(-1),
              SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);
    for (int32_t selectionSlot = 0; selectionSlot < validTopkNum; selectionSlot++) {
        int32_t tokenId = residentStatusLocal.GetValue(selectionSlot);
        if (tokenId >= 0 && tokenId < SELECTION_MEMBERSHIP_MAX_TOKEN) {
            membershipStorageLocal.SetValue(tokenId, selectionSlot + 1);
        }
    }
    membershipStorageLocal.SetValue(
        SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT,
        SELECTION_MEMBERSHIP_READY_MARKER);
    SetFlag<AscendC::HardEvent::S_MTE3>(0);
    WaitFlag<AscendC::HardEvent::S_MTE3>(0);

    DataCopyExtParams metadataParams;
    metadataParams.blockCount = 1;
    metadataParams.blockLen =
        SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT * sizeof(int16_t);
    metadataParams.srcStride = 0;
    metadataParams.dstStride = 0;
    DataCopyPad(selectionMembershipMapGm_[membershipBase],
                membershipStorageLocal, metadataParams);
    SetFlag<AscendC::HardEvent::MTE3_S>(0);
    WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::WriteSelectionUpdatePlan(
    int64_t membershipBase, int32_t planCount, int32_t selectionHitCount,
    LocalTensor<int16_t> membershipStorageLocal, int64_t planOffset,
    bool preserveMembershipMap)
{
    if (planCount <= 0 ||
        selectionMembershipStride_ < SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT) {
        return;
    }
    LocalTensor<int16_t> controlLocal =
        membershipStorageLocal[SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT];
    Duplicate(controlLocal, static_cast<int16_t>(-1),
              SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);
    controlLocal.SetValue(
        0, preserveMembershipMap ? SELECTION_MEMBERSHIP_READY_MARKER : -1);
    controlLocal.SetValue(1, SELECTION_PLAN_READY_MARKER);
    controlLocal.SetValue(2, selectionHitCount);
    controlLocal.SetValue(3, static_cast<int32_t>(planOffset));
    controlLocal.SetValue(4, preserveMembershipMap ? 1 : 0);
    SetFlag<AscendC::HardEvent::S_MTE3>(0);
    WaitFlag<AscendC::HardEvent::S_MTE3>(0);

    DataCopyExtParams planParams;
    planParams.blockCount = 1;
    planParams.blockLen = static_cast<uint32_t>(
        (planOffset + planCount) * sizeof(int16_t));
    planParams.srcStride = 0;
    planParams.dstStride = 0;
    DataCopyPad(selectionMembershipMapGm_[membershipBase],
                membershipStorageLocal, planParams);

    DataCopyExtParams controlParams;
    controlParams.blockCount = 1;
    controlParams.blockLen =
        SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT * sizeof(int16_t);
    controlParams.srcStride = 0;
    controlParams.dstStride = 0;
    DataCopyPad(selectionMembershipMapGm_[membershipBase +
                    SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT],
                controlLocal, controlParams);
    SetFlag<AscendC::HardEvent::MTE3_S>(0);
    WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::WriteSparseSelectionUpdatePlan(
    int64_t membershipBase, int32_t updateCount,
    LocalTensor<int16_t> membershipStorageLocal, int64_t planOffset)
{
    if (updateCount <= 0 || updateCount > SELECTION_SYNC_COPY_CAPACITY ||
        planOffset < 0 ||
        planOffset + SELECTION_SPARSE_PLAN_VALUE_COUNT >
            SELECTION_MEMBERSHIP_MAP_INT16_COUNT) {
        return;
    }

    LocalTensor<int16_t> controlLocal =
        membershipStorageLocal[SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT];
    Duplicate(controlLocal, static_cast<int16_t>(-1),
              SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_S>(0);
    WaitFlag<AscendC::HardEvent::V_S>(0);
    controlLocal.SetValue(0, SELECTION_MEMBERSHIP_READY_MARKER);
    controlLocal.SetValue(1, SELECTION_SPARSE_PLAN_READY_MARKER);
    controlLocal.SetValue(2, updateCount);
    controlLocal.SetValue(3, static_cast<int32_t>(planOffset));
    controlLocal.SetValue(4, 1);

    SetFlag<AscendC::HardEvent::S_MTE3>(0);
    WaitFlag<AscendC::HardEvent::S_MTE3>(0);

    DataCopyExtParams planParams;
    planParams.blockCount = 1;
    planParams.blockLen =
        SELECTION_SPARSE_PLAN_VALUE_COUNT * sizeof(int16_t);
    planParams.srcStride = 0;
    planParams.dstStride = 0;
    DataCopyPad(selectionMembershipMapGm_[membershipBase + planOffset],
                membershipStorageLocal[planOffset], planParams);

    DataCopyExtParams controlParams;
    controlParams.blockCount = 1;
    controlParams.blockLen =
        SELECTION_MEMBERSHIP_CONTROL_INT16_COUNT * sizeof(int16_t);
    controlParams.srcStride = 0;
    controlParams.dstStride = 0;
    DataCopyPad(selectionMembershipMapGm_[membershipBase +
                    SELECTION_MEMBERSHIP_CONTROL_OFFSET_INT16_COUNT],
                controlLocal, controlParams);
    SetFlag<AscendC::HardEvent::MTE3_S>(0);
    WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessSetResidentSelectionRow(
    int64_t selectionRow, uint32_t batchIdx, int64_t s2IdLimit)
{
    int64_t topkCount = static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t sortAlign = CeilDiv(topkCount, static_cast<int64_t>(SELECTION_SORT_UNIT)) *
                        static_cast<int64_t>(SELECTION_SORT_UNIT);
    int64_t topkBlockAlign = CeilDiv(topkCount,
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t));
    int64_t statusBlockAlign = CeilDiv(topkCount + 1,
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int32_t));
    int64_t statusLocalStride = sortAlign > statusBlockAlign ? sortAlign : statusBlockAlign;

    LocalTensor<int32_t> residentStatusLocal = tmpBuff1.Get<int32_t>();
    LocalTensor<int32_t> sourceStatusLocal = residentStatusLocal[statusLocalStride];
    LocalTensor<uint32_t> indexLocal = outputBuff1.Get<uint32_t>();
    LocalTensor<int32_t> insertStatusLocal = outputBuff1.Get<int32_t>()[sortAlign];
    LocalTensor<int32_t> hitSourceLocal = outputBuff1.Get<int32_t>()[sortAlign * 2];
    LocalTensor<int32_t> currentTopkLocal = outputBuff1.Get<int32_t>()[sortAlign * 3];
    LocalTensor<int32_t> sortBufferLocal = inputBuff1.Get<int32_t>();
    LocalTensor<int16_t> membershipStorageLocal = inputBuff1.Get<int16_t>();
    LocalTensor<int16_t> gatheredMembershipLocal =
        tmpBuff1.Get<int16_t>()[statusLocalStride * 2 *
                                sizeof(int32_t) / sizeof(int16_t)];
    LocalTensor<int32_t> gatheredMembershipInt32Local =
        tmpBuff1.Get<int32_t>()[statusLocalStride * 2 +
            topkBlockAlign * sizeof(int16_t) / sizeof(int32_t)];
    int64_t membershipBase = selectionRow * selectionMembershipStride_;

    IsSelectionMembershipMapReady(membershipBase, gatheredMembershipLocal);
    if (gatheredMembershipLocal.GetValue(1) ==
        SELECTION_EXTERNAL_PLAN_READY_MARKER) {
        selectionKvActualSeqGm_.SetValue(selectionRow, topkCount);
        return;
    }

    DataCopyExtParams topkParams;
    topkParams.blockCount = 1;
    topkParams.blockLen = static_cast<uint32_t>(topkCount * sizeof(int32_t));
    topkParams.srcStride = 0;
    topkParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> topkPadParams{
        true, 0, static_cast<uint8_t>(topkBlockAlign - topkCount), -1};
    DataCopyPad(currentTopkLocal, topkGm_[selectionRow * topkCount],
                topkParams, topkPadParams);

    DataCopyExtParams statusParams;
    statusParams.blockCount = 1;
    statusParams.blockLen = static_cast<uint32_t>((topkCount + 1) * sizeof(int32_t));
    statusParams.srcStride = 0;
    statusParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> statusPadParams{
        true, 0, static_cast<uint8_t>(statusBlockAlign - topkCount - 1), -1};
    int64_t statusBase = selectionRow * selectionStatusStride_;
    DataCopyPad(residentStatusLocal, selectionKvBlockStatusGm_[statusBase],
                statusParams, statusPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(0);
    WaitFlag<AscendC::HardEvent::MTE2_S>(0);

    int32_t previousValidCount = residentStatusLocal.GetValue(topkCount);
    if (IsPositionResidentSelectionHit(
            currentTopkLocal, residentStatusLocal, sortBufferLocal, previousValidCount)) {
        if (!IsSelectionMembershipMapReady(membershipBase, gatheredMembershipLocal)) {
            PublishTokenSetResidentSelectionMap(
                residentStatusLocal, membershipStorageLocal, membershipBase,
                previousValidCount, s2IdLimit);
        } else {
            ClearSelectionUpdatePlanMarker(membershipBase);
        }
        selectionKvActualSeqGm_.SetValue(selectionRow, previousValidCount);
        return;
    }
    bool membershipSlotMapLoaded = false;
    bool tokenSetResidentHit = IsTokenSetResidentSelectionHit(
        currentTopkLocal, membershipStorageLocal, indexLocal,
        gatheredMembershipLocal, gatheredMembershipInt32Local, membershipBase,
        previousValidCount, s2IdLimit, membershipSlotMapLoaded);
    if (tokenSetResidentHit) {
        ClearSelectionUpdatePlanMarker(membershipBase);
        selectionKvActualSeqGm_.SetValue(selectionRow, previousValidCount);
        return;
    }
    int64_t activeMapCount = CeilDiv(
        s2IdLimit, static_cast<int64_t>(BYTE_BLOCK / sizeof(int16_t))) *
        static_cast<int64_t>(BYTE_BLOCK / sizeof(int16_t));
    bool membershipMapLocalPreserved = membershipSlotMapLoaded &&
        activeMapCount * static_cast<int64_t>(sizeof(int16_t)) +
            topkBlockAlign * 4 * static_cast<int64_t>(sizeof(int32_t)) <=
            static_cast<int64_t>(ConstInfo::BUFFER_SIZE_BYTE_32K * 2);
    membershipMapLocalPreserved = membershipMapLocalPreserved &&
        activeMapCount + topkCount <= SELECTION_MEMBERSHIP_MAP_INT16_COUNT;
    int64_t selectionPlanLocalOffset =
        membershipMapLocalPreserved ? activeMapCount : 0;
    int64_t selectionGroupBaseRow =
        selectionRow / static_cast<int64_t>(constInfo.kvHeadNum) *
        static_cast<int64_t>(constInfo.kvHeadNum);
    int32_t validTopkNum = membershipSlotMapLoaded ?
        BuildDenseResidentSelectionPlan(
            gatheredMembershipInt32Local, insertStatusLocal, hitSourceLocal) :
        BuildSetResidentSelectionPlan(
            selectionRow, selectionGroupBaseRow, s2IdLimit,
            currentTopkLocal, residentStatusLocal, sourceStatusLocal,
            indexLocal, insertStatusLocal, hitSourceLocal, sortBufferLocal);
    LocalTensor<int16_t> compactPlanLocal =
        membershipStorageLocal[selectionPlanLocalOffset];
    bool compactPlanStarted = false;

    bool statusDirty = previousValidCount != validTopkNum;
    int32_t updateCount = 0;
    int32_t selectionHitCount = 0;
    int32_t smallUpdateTopkPosition[SELECTION_SYNC_COPY_CAPACITY];
    int32_t smallUpdateDestination[SELECTION_SYNC_COPY_CAPACITY];
    int32_t smallUpdateStaleToken[SELECTION_SYNC_COPY_CAPACITY];
    int32_t nextInsertSlot = 0;
    for (int32_t topkPosition = 0; topkPosition < validTopkNum; topkPosition++) {
        int32_t planValue = hitSourceLocal.GetValue(topkPosition);
        if (planValue >= 0 && (planValue & SELECTION_PLAN_HIT_FLAG) != 0) {
            if (compactPlanStarted) {
                compactPlanLocal.SetValue(
                    topkPosition,
                    static_cast<int16_t>((planValue & SELECTION_PLAN_HIT_SLOT_MASK) + 1));
            }
            selectionHitCount++;
            continue;
        }

        int32_t destinationSlot = -1;
        for (int32_t slot = nextInsertSlot; slot < static_cast<int32_t>(topkCount); slot++) {
            if (insertStatusLocal.GetValue(slot) < 0) {
                destinationSlot = slot;
                break;
            }
        }
        if (destinationSlot < 0) {
            break;
        }
        nextInsertSlot = destinationSlot + 1;
        if (updateCount < SELECTION_SYNC_COPY_CAPACITY) {
            smallUpdateTopkPosition[updateCount] = topkPosition;
            smallUpdateDestination[updateCount] = destinationSlot;
            smallUpdateStaleToken[updateCount] = residentStatusLocal.GetValue(destinationSlot);
        }
        updateCount++;

        int32_t tokenId = currentTopkLocal.GetValue(topkPosition);
        int32_t staleToken = residentStatusLocal.GetValue(destinationSlot);
        hitSourceLocal.SetValue(
            topkPosition,
            SELECTION_PLAN_UPDATE_FLAG | destinationSlot);
        if (!compactPlanStarted && updateCount > SELECTION_SYNC_COPY_THRESHOLD) {
            Duplicate(compactPlanLocal, SELECTION_COMPACT_PLAN_INVALID, topkCount);
            PipeBarrier<PIPE_V>();
            SetFlag<AscendC::HardEvent::V_S>(0);
            WaitFlag<AscendC::HardEvent::V_S>(0);
            for (int32_t planIdx = 0; planIdx <= topkPosition; planIdx++) {
                compactPlanLocal.SetValue(
                    planIdx,
                    EncodeSelectionPlanValue(hitSourceLocal.GetValue(planIdx)));
            }
            compactPlanStarted = true;
        } else if (compactPlanStarted) {
            compactPlanLocal.SetValue(
                topkPosition, static_cast<int16_t>(-(destinationSlot + 1)));
        }
        if (membershipMapLocalPreserved) {
            if (staleToken >= 0 && staleToken < SELECTION_MEMBERSHIP_MAX_TOKEN) {
                membershipStorageLocal.SetValue(staleToken, -1);
            }
            if (tokenId >= 0 && tokenId < SELECTION_MEMBERSHIP_MAX_TOKEN) {
                membershipStorageLocal.SetValue(tokenId, destinationSlot + 1);
            }
        }
        if (staleToken != tokenId) {
            statusDirty = true;
        }
        residentStatusLocal.SetValue(destinationSlot, tokenId);
    }

    for (int32_t slot = validTopkNum; slot < static_cast<int32_t>(topkCount); slot++) {
        if (residentStatusLocal.GetValue(slot) != -1) {
            residentStatusLocal.SetValue(slot, -1);
            statusDirty = true;
        }
    }
    residentStatusLocal.SetValue(topkCount, validTopkNum);

    bool useSparseSelectionUpdatePlan =
        membershipMapLocalPreserved && previousValidCount == validTopkNum &&
        validTopkNum == static_cast<int32_t>(topkCount) && updateCount > 0 &&
        updateCount <= SELECTION_SYNC_COPY_CAPACITY;
    if (useSparseSelectionUpdatePlan) {
        for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
            int32_t topkPosition = smallUpdateTopkPosition[updateIdx];
            int32_t tokenId = currentTopkLocal.GetValue(topkPosition);
            if (tokenId < 0 || tokenId >= SELECTION_MEMBERSHIP_MAX_TOKEN) {
                useSparseSelectionUpdatePlan = false;
                break;
            }
        }
    }
    bool deferStatusWriteDrain =
        statusDirty && (useSparseSelectionUpdatePlan ||
                        updateCount > SELECTION_SYNC_COPY_THRESHOLD);
    if (statusDirty) {
        SetFlag<AscendC::HardEvent::S_MTE3>(0);
        WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        DataCopyExtParams statusOutParams;
        statusOutParams.blockCount = 1;
        statusOutParams.blockLen = static_cast<uint32_t>((topkCount + 1) * sizeof(int32_t));
        statusOutParams.srcStride = 0;
        statusOutParams.dstStride = 0;
        DataCopyPad(selectionKvBlockStatusGm_[statusBase], residentStatusLocal, statusOutParams);
        if (!deferStatusWriteDrain) {
            SetFlag<AscendC::HardEvent::MTE3_S>(0);
            WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        }
    }
    if (useSparseSelectionUpdatePlan) {
        Duplicate(compactPlanLocal, static_cast<int16_t>(0),
                  SELECTION_SPARSE_PLAN_VALUE_COUNT);
        PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_S>(0);
        WaitFlag<AscendC::HardEvent::V_S>(0);
        for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
            int32_t topkPosition = smallUpdateTopkPosition[updateIdx];
            int32_t destinationSlot = smallUpdateDestination[updateIdx];
            int32_t tokenId = currentTopkLocal.GetValue(topkPosition);
            compactPlanLocal.SetValue(updateIdx * 2,
                                      static_cast<int16_t>(destinationSlot + 1));
            compactPlanLocal.SetValue(updateIdx * 2 + 1,
                                      static_cast<int16_t>(tokenId + 1));
        }
        for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
            int32_t staleToken = smallUpdateStaleToken[updateIdx];
            if (staleToken >= 0 && staleToken < SELECTION_MEMBERSHIP_MAX_TOKEN) {
                selectionMembershipMapGm_.SetValue(membershipBase + staleToken, -1);
            }
        }
        for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
            int32_t topkPosition = smallUpdateTopkPosition[updateIdx];
            int32_t tokenId = currentTopkLocal.GetValue(topkPosition);
            int32_t destinationSlot = smallUpdateDestination[updateIdx];
            selectionMembershipMapGm_.SetValue(
                membershipBase + tokenId, destinationSlot + 1);
        }
        WriteSparseSelectionUpdatePlan(
            membershipBase, updateCount, membershipStorageLocal,
            selectionPlanLocalOffset);
    } else if (updateCount <= SELECTION_SYNC_COPY_THRESHOLD) {
        for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
            int32_t topkPosition = smallUpdateTopkPosition[updateIdx];
            int32_t destinationSlot = smallUpdateDestination[updateIdx];
            int32_t tokenId = currentTopkLocal.GetValue(topkPosition);
            CopySelectionUpdateTokenFromFullCache(
                selectionRow, batchIdx, destinationSlot, tokenId, s2IdLimit);
        }
        if (updateCount > 0) {
            SetFlag<AscendC::HardEvent::MTE3_V>(0);
            WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        if (membershipMapLocalPreserved) {
            for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
                int32_t staleToken = smallUpdateStaleToken[updateIdx];
                if (staleToken >= 0 && staleToken < SELECTION_MEMBERSHIP_MAX_TOKEN) {
                    selectionMembershipMapGm_.SetValue(membershipBase + staleToken, -1);
                }
            }
            for (int32_t updateIdx = 0; updateIdx < updateCount; updateIdx++) {
                int32_t topkPosition = smallUpdateTopkPosition[updateIdx];
                int32_t tokenId = currentTopkLocal.GetValue(topkPosition);
                int32_t destinationSlot = smallUpdateDestination[updateIdx];
                if (tokenId >= 0 && tokenId < SELECTION_MEMBERSHIP_MAX_TOKEN) {
                    selectionMembershipMapGm_.SetValue(
                        membershipBase + tokenId, destinationSlot + 1);
                }
            }
            ClearSelectionUpdatePlanMarker(membershipBase);
        } else {
            PublishTokenSetResidentSelectionMap(
                residentStatusLocal, membershipStorageLocal, membershipBase,
                validTopkNum, s2IdLimit);
        }
    } else {
        WriteSelectionUpdatePlan(
            membershipBase, static_cast<int32_t>(topkCount), selectionHitCount,
            membershipStorageLocal, selectionPlanLocalOffset,
            membershipMapLocalPreserved);
    }
    selectionKvActualSeqGm_.SetValue(selectionRow, validTopkNum);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::RunAllCoreSelectionUpdate()
{
    if (!UseSetResidentSelection()) {
        return;
    }

    // AIV 侧 GetBlockNum() 返回 AIC 核数（c:v = 1:2），乘 2 覆盖全部 AIV 子核参与 P0 行分摊，
    // 否则后半数子核（workerIdx >= 24）直接退出，bs > 24 时行回绕使首个核承担双倍工作量。
    uint32_t workerCount = GetBlockNum() * 2;
    uint32_t workerIdx = (GetBlockIdx() / 2) * 2 + GetSubBlockIdx();
    if (workerCount == 0 || workerIdx >= workerCount) {
        return;
    }

    int64_t qTokenPrefix = 0;
    int64_t selectionGroupIdx = 0;
    for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
        uint64_t actualQSeqLen = GetActualQSeqLenForSelectionUpdate(batchIdx);
        uint64_t actualKVSeqLen = GetActualKVSeqLenForSelectionUpdate(batchIdx);
        for (uint64_t qIdx = 0; qIdx < actualQSeqLen; qIdx++) {
            if (selectionGroupIdx % static_cast<int64_t>(workerCount) != workerIdx) {
                selectionGroupIdx++;
                continue;
            }
            int64_t s2IdLimit = static_cast<int64_t>(actualKVSeqLen);
            if (constInfo.sparseMode == 3) {
                s2IdLimit = static_cast<int64_t>(actualKVSeqLen) - static_cast<int64_t>(actualQSeqLen) +
                            static_cast<int64_t>(qIdx) + 1;
            }
            if (s2IdLimit < 0) {
                s2IdLimit = 0;
            }
            int64_t qTokenIdx = 0;
            if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
                qTokenIdx = qTokenPrefix + static_cast<int64_t>(qIdx);
            } else {
                qTokenIdx = static_cast<int64_t>(batchIdx) * constInfo.qSeqSize +
                            static_cast<int64_t>(qIdx);
            }
            int64_t selectionGroupBaseRow =
                qTokenIdx * static_cast<int64_t>(constInfo.kvHeadNum);
            for (uint32_t n2Idx = 0; n2Idx < constInfo.kvHeadNum; n2Idx++) {
                ProcessSetResidentSelectionRow(
                    selectionGroupBaseRow + static_cast<int64_t>(n2Idx), batchIdx, s2IdLimit);
            }
            selectionGroupIdx++;
        }
        qTokenPrefix += static_cast<int64_t>(actualQSeqLen);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::
CopyOutSparseSelectionUpdateFromKvMerge(const RunInfo &runInfo)
{
    if (!selectionSparseUpdatePlanActive_ || selectionUpdatePlanCount_ <= 0 ||
        selectionUpdatePlanCount_ > SELECTION_SYNC_COPY_CAPACITY) {
        return;
    }

    int64_t selectionRow = GetSelectionRow(runInfo);
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 :
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ?
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    if (s2GmStartOffset >= s2GmLimit) {
        return;
    }

    int64_t physicalSlotBase =
        static_cast<int64_t>(runInfo.s2Idx) * static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t rangeStart = physicalSlotBase + s2GmStartOffset;
    int64_t rangeLimit = physicalSlotBase + s2GmLimit;
    LocalTensor<int16_t> sparsePlanLocal =
        v0ValidSizeBuff.Get<int16_t>()[SELECTION_STATUS_UB_OFFSET *
            sizeof(int32_t) / sizeof(int16_t)];

    int32_t updateIdx = 0;
    while (updateIdx < selectionUpdatePlanCount_) {
        int32_t destinationSlot =
            static_cast<int32_t>(sparsePlanLocal.GetValue(updateIdx * 2)) - 1;
        if (destinationSlot < rangeStart) {
            updateIdx++;
            continue;
        }
        if (destinationSlot >= rangeLimit) {
            break;
        }

        int64_t destinationBlockTableIdx =
            static_cast<int64_t>(destinationSlot) / selectionKvBlockSize_;
        int64_t destinationBlockOffset =
            static_cast<int64_t>(destinationSlot) % selectionKvBlockSize_;
        int64_t runLen = 1;
        while (updateIdx + runLen < selectionUpdatePlanCount_ &&
               runLen < SELECTION_SYNC_COPY_CAPACITY &&
               destinationBlockOffset + runLen < selectionKvBlockSize_) {
            int32_t nextDestinationSlot = static_cast<int32_t>(
                sparsePlanLocal.GetValue((updateIdx + runLen) * 2)) - 1;
            if (nextDestinationSlot != destinationSlot + runLen ||
                nextDestinationSlot >= rangeLimit) {
                break;
            }
            runLen++;
        }

        int32_t destinationBlockNum = selectionKvBlockTableGm_.GetValue(
            selectionRow * selectionMaxBlockNum_ + destinationBlockTableIdx);
        if (destinationBlockNum >= 0) {
            int64_t sourceOffset =
                static_cast<int64_t>(destinationSlot) - physicalSlotBase;
            DataCopyExtParams kvParams;
            kvParams.blockCount = static_cast<uint32_t>(runLen);
            kvParams.blockLen = constInfo.headDim * sizeof(KV_T);
            kvParams.srcStride = 0;
            kvParams.dstStride = 0;
            DataCopyPadExtParams<KV_T> padParams;
            DataCopyPad(kvMergUb_,
                kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           sourceOffset * static_cast<int64_t>(constInfo.headDim)],
                kvParams, padParams);

            DataCopyExtParams ropeParams;
            ropeParams.blockCount = static_cast<uint32_t>(runLen);
            ropeParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
            ropeParams.srcStride = 0;
            ropeParams.dstStride = 0;
            if (constInfo.headDimRope > 0) {
                DataCopyPad(ropeMergUb_,
                    kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                               512 * static_cast<int64_t>(constInfo.headDim) +
                               sourceOffset * static_cast<int64_t>(constInfo.headDimRope)],
                    ropeParams, padParams);
            }
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

            int64_t destinationKvOffset =
                (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ +
                 destinationBlockOffset) * static_cast<int64_t>(constInfo.headDim);
            DataCopyPad(selectionKvCacheGm_[destinationKvOffset], kvMergUb_, kvParams);
            if (constInfo.headDimRope > 0) {
                int64_t destinationRopeOffset =
                    (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ +
                     destinationBlockOffset) *
                    static_cast<int64_t>(constInfo.headDimRope);
                DataCopyPad(selectionKRopeGm_[destinationRopeOffset],
                            ropeMergUb_, ropeParams);
            }
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        updateIdx += static_cast<int32_t>(runLen);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CopyOutSelectionUpdateFromKvMerge(const RunInfo &runInfo)
{
    if (!selectionUpdatePlanActive_ || !UseSetResidentSelection()) {
        return;
    }
    if (selectionSparseUpdatePlanActive_) {
        CopyOutSparseSelectionUpdateFromKvMerge(runInfo);
        return;
    }

    int64_t logicalSelectionRow = GetSelectionRow(runInfo);
    int64_t selectionRow = selectionDataRow_ >= 0 ?
        selectionDataRow_ : logicalSelectionRow;
    int64_t membershipBase = logicalSelectionRow * selectionMembershipStride_;
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t subBlockStart = GetSubBlockIdx() == 0 ? 0 :
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t subBlockLimit = GetSubBlockIdx() == 0 ?
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
    if (subBlockLimit > s2ProcessSize) {
        subBlockLimit = s2ProcessSize;
    }
    // helper 也分担回写：mergeSplitActive_ 时按 MergeKv 同一 partIdx 再切，
    // target/helper 各处理自己搬的那部分 token 的 update 槽位（纯数据搬运）。
    // （对应 helper-only 仓 d06cb9f0 + c517aa70 修复版：V1 起点带 2 倍因子）
    int64_t s2GmStartOffset = subBlockStart;
    int64_t s2GmLimit = subBlockLimit;
    if (mergeSplitActive_) {
        int64_t alignUnit = 2L * constInfo.sparseBlockSize;
        int64_t subBlockSize = subBlockLimit - subBlockStart;
        if (subBlockSize < 0) {
            subBlockSize = 0;
        }
        int64_t partSize = CeilDiv(subBlockSize, (int64_t)helperPartCount_ * alignUnit) * alignUnit;
        s2GmStartOffset = subBlockStart + helperPartIdx_ * partSize;
        s2GmLimit = subBlockStart + (helperPartIdx_ + 1) * partSize;
        if (s2GmLimit > subBlockLimit) {
            s2GmLimit = subBlockLimit;
        }
        if (s2GmStartOffset > s2GmLimit) {
            s2GmStartOffset = s2GmLimit;
        }
    }
    if (s2GmStartOffset >= s2GmLimit) {
        return;
    }

    int64_t logicalPlanStart =
        static_cast<int64_t>(runInfo.s2Idx) * static_cast<int64_t>(constInfo.s2BaseSize) +
        s2GmStartOffset;
    int64_t planCount = s2GmLimit - s2GmStartOffset;
    LocalTensor<int16_t> planLocal =
        v0ValidSizeBuff.Get<int16_t>()[SELECTION_STATUS_UB_OFFSET * sizeof(int32_t) / sizeof(int16_t)];
    LocalTensor<uint32_t> packedPlanLocal = planLocal.ReinterpretCast<uint32_t>();
    DataCopyExtParams planParams;
    planParams.blockCount = 1;
    planParams.blockLen = static_cast<uint32_t>(planCount * sizeof(int16_t));
    planParams.srcStride = 0;
    planParams.dstStride = 0;
    DataCopyPadExtParams<int16_t> planPadParams{false, 0, 0, 0};
    DataCopyPad(planLocal,
        selectionMembershipMapGm_[membershipBase + selectionUpdatePlanOffset_ + logicalPlanStart],
        planParams, planPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(1);
    WaitFlag<AscendC::HardEvent::MTE2_S>(1);

    int64_t localOffset = s2GmStartOffset;
    while (localOffset < s2GmLimit) {
        int64_t writeCount = s2GmLimit - localOffset;
        if (writeCount > 32) {
            writeCount = 32;
        }
        // 这个窗口的 update 条目已经在 MergeKv 里趁数据还在 UB 时写完了
        // （FlushDirectSelectionUpdate：密窗趁 UB 直写完毕，满批零窗已证确无条目）。扫描、回读、写回
        // 三样一样都不用做。
        int64_t directWindowIdx = (localOffset - s2GmStartOffset) >> 5;
        if (directWindowIdx < 32 &&
            ((directWrittenWindowMask_ >> static_cast<uint32_t>(directWindowIdx)) & 1U) != 0) {
            localOffset += writeCount;
            continue;
        }
        int64_t updateDstSlot[32];
        int64_t updateSrcIdx[32];
        int64_t updateCount = 0;
        int64_t s2IdLimit = runInfo.curActualSeqLenOri;
        if (constInfo.sparseMode == 3) {
            s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size +
                        runInfo.gS1Idx / constInfo.gSize + 1;
        }
        // 换位判定只依赖 pairOffset。一对里两个 token 都是 update 条目时下面那段会
        // 走两遍，而 pairOffset 随 idx 单调不减，缓存一格就够。
        int64_t cachedPairOffset = -1;
        bool cachedPairTransposed = false;
        // 原来是逐 token 走，每次判一次「这个半字所在的 int32 跟上次是不是同一个」。
        // 那个判断**每两次真一次、假一次**，是最难预测的那种分支；每 token 还要一次
        // 三元选择取半字。整段扫描消融实测 15.15 μs / 1024 token ≈ 27 周期/token，
        // 而每 token 的实活只有半次 UB 读加一次比较。
        //
        // 改成一次取一对：int32 读一次、两个半字当场拆开，那个分支整个消失，
        // 外层循环次数也减半。读的次数一次没变（原来也是每 int32 读一次）。
        //
        // 对齐前提：localOffset − s2GmStartOffset 是各次 writeCount 之和，而
        // writeCount 恒为 32（planCount 是 32 的整数倍），所以 basePlanIdx 恒为偶数，
        // 低半字就是 base 这个 token。
        for (int64_t base = 0; base < writeCount; base += 2) {
            int64_t basePlanIdx = localOffset - s2GmStartOffset + base;
            int64_t cachedPackedPlanIdx = basePlanIdx >> 1;
            uint32_t cachedPackedPlanValue =
                packedPlanLocal.GetValue(cachedPackedPlanIdx);
            int64_t pairCount = (base + 1 < writeCount) ? 2 : 1;
            for (int64_t half = 0; half < pairCount; half++) {
            int64_t idx = base + half;
            int64_t planIdx = basePlanIdx + half;
            int16_t planValue = half == 0 ?
                static_cast<int16_t>(cachedPackedPlanValue & 0xFFFFU) :
                static_cast<int16_t>(cachedPackedPlanValue >> 16);
            if (!IsSelectionPlanUpdate(planValue)) {
                continue;
            }
            int32_t destinationSlot = DecodeSelectionPlanSlot(planValue);
            int64_t srcIdx = idx;
            int64_t absoluteOffset = localOffset + idx;
            int64_t pairSize = 2L * static_cast<int64_t>(constInfo.sparseBlockSize);
            int64_t pairOffset = (absoluteOffset / pairSize) * pairSize;
            if (constInfo.sparseBlockSize == 1 && pairOffset + 1 < s2ProcessSize) {
                bool pairTransposed;
                if (pairOffset == cachedPairOffset) {
                    pairTransposed = cachedPairTransposed;
                } else {
                bool pairHasSelectionHit = false;
                int64_t pairPlanIdx = pairOffset - s2GmStartOffset;
                if (pairPlanIdx >= 0 && pairPlanIdx + 1 < planCount) {
                    int64_t pairPackedPlanIdx = pairPlanIdx / 2;
                    uint32_t pairPackedPlanValue =
                        pairPackedPlanIdx == cachedPackedPlanIdx ?
                            cachedPackedPlanValue :
                            packedPlanLocal.GetValue(pairPackedPlanIdx);
                    int16_t pairPlan0 =
                        static_cast<int16_t>(pairPackedPlanValue & 0xFFFFU);
                    int16_t pairPlan1 =
                        static_cast<int16_t>(pairPackedPlanValue >> 16);
                    pairHasSelectionHit = IsSelectionPlanHit(pairPlan0) ||
                        IsSelectionPlanHit(pairPlan1);
                }
                // pairHasSelectionHit 为真时，下面这四次 GM 标量读全是白做的：
                // usedPairedCopy 的第一个合取项就是 !pairHasSelectionHit，整个表达式
                // 短路成 false，realS2Idx0/1 与 keyOffset0/1 之后再没人读。
                // 而 GetRealS2Idx 读一次 topkGm_、GetKeyGmOffset 读一次 blkTableGm_，
                // 四次都随 update 条目数走 —— 命中率越高，白做的比例越大。
                pairTransposed = false;
                if (!pairHasSelectionHit) {
                int64_t realS2Idx0 = -1;
                int64_t realS2Idx1 = -1;
                GetRealS2Idx(pairOffset, realS2Idx0, runInfo.topKBaseOffset, runInfo);
                GetRealS2Idx(pairOffset + 1, realS2Idx1, runInfo.topKBaseOffset, runInfo);
                int64_t keyOffset0 = GetKeyGmOffset(realS2Idx0, runInfo, s2IdLimit);
                int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
                int64_t keySrcStride = 0;
                int64_t keyRopeSrcStride = 0;
                bool usedPairedCopy = keyOffset0 >= 0 && keyOffset1 >= 0 &&
                    CanUsePairedKvCopy(
                        realS2Idx0, realS2Idx1, keyOffset0, keyOffset1, s2IdLimit,
                        runInfo, keySrcStride, keyRopeSrcStride);
                pairTransposed = usedPairedCopy && keyOffset1 < keyOffset0;
                }
                cachedPairOffset = pairOffset;
                cachedPairTransposed = pairTransposed;
                }
                if (pairTransposed) {
                    int64_t srcAbsoluteOffset = (absoluteOffset == pairOffset) ? pairOffset + 1 : pairOffset;
                    if (srcAbsoluteOffset >= localOffset && srcAbsoluteOffset < localOffset + writeCount) {
                        srcIdx = srcAbsoluteOffset - localOffset;
                    }
                }
            }
            updateDstSlot[updateCount] = destinationSlot;
            updateSrcIdx[updateCount] = srcIdx;
            updateCount++;
            }
        }

        if (updateCount > 0) {
            DataCopyExtParams kvParams;
            kvParams.blockLen = constInfo.headDim * sizeof(KV_T);
            kvParams.srcStride = 0;
            kvParams.dstStride = 0;
            DataCopyPadExtParams<KV_T> padParams;

            DataCopyExtParams ropeParams;
            ropeParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
            ropeParams.srcStride = 0;
            ropeParams.dstStride = 0;

            // 整段把 writeCount 个 token 读回 UB，是为了下面那个循环能从 UB 取。
            // 但真要写回 selection 缓存的只有 updateCount 条 —— 90% 命中下
            // 32 个里约 3 个，其余 29 个读回来没人碰。整趟约 1.18 MB 的 MTE2。
            //
            // 条目密的时候按条读会把一条 DataCopyPad 拆成 updateCount 条，
            // 拿指令数换字节数不划算，所以只在稀疏时走按需读；密的时候逐字走
            // 老路（那也是这一刀的零对照臂 —— 低命中端结构上一行都没变）。
            const bool sparseWriteback = (updateCount * 4 <= writeCount);
            if (sparseWriteback) {
                kvParams.blockCount = 1;
                ropeParams.blockCount = 1;
                for (int64_t i = 0; i < updateCount; i++) {
                    int64_t srcSlotIdx = updateSrcIdx[i];
                    DataCopyPad(
                        kvMergUb_[srcSlotIdx * static_cast<int64_t>(constInfo.headDim)],
                        kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                   (localOffset + srcSlotIdx) *
                                       static_cast<int64_t>(constInfo.headDim)],
                        kvParams, padParams);
                    if (constInfo.headDimRope > 0) {
                        DataCopyPad(
                            ropeMergUb_[srcSlotIdx *
                                        static_cast<int64_t>(constInfo.headDimRope)],
                            kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       512 * static_cast<int64_t>(constInfo.headDim) +
                                       (localOffset + srcSlotIdx) *
                                           static_cast<int64_t>(constInfo.headDimRope)],
                            ropeParams, padParams);
                    }
                }
            } else {
                kvParams.blockCount = writeCount;
                DataCopyPad(kvMergUb_,
                    kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                localOffset * static_cast<int64_t>(constInfo.headDim)],
                    kvParams, padParams);
                ropeParams.blockCount = writeCount;
                DataCopyPad(ropeMergUb_,
                    kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                               512 * static_cast<int64_t>(constInfo.headDim) +
                               localOffset * static_cast<int64_t>(constInfo.headDimRope)],
                    ropeParams, padParams);
            }
            SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
            int64_t updateIdx = 0;
            while (updateIdx < updateCount) {
                int64_t destinationSlot = updateDstSlot[updateIdx];
                int64_t destinationBlockTableIdx = destinationSlot / selectionKvBlockSize_;
                int64_t destinationBlockOffset = destinationSlot % selectionKvBlockSize_;
                int32_t destinationBlockNum = selectionKvBlockTableGm_.GetValue(
                    selectionRow * selectionMaxBlockNum_ + destinationBlockTableIdx);
                int64_t srcStart = updateSrcIdx[updateIdx];
                int64_t runLen = 1;
                while (updateIdx + runLen < updateCount &&
                       updateDstSlot[updateIdx + runLen] == destinationSlot + runLen &&
                       updateSrcIdx[updateIdx + runLen] == srcStart + runLen &&
                       destinationBlockOffset + runLen < selectionKvBlockSize_) {
                    runLen++;
                }
                if (destinationBlockNum >= 0) {
                    DataCopyExtParams updateKvParams = kvParams;
                    updateKvParams.blockCount = runLen;
                    int64_t dstKvAddr =
                        (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ +
                         destinationBlockOffset) * static_cast<int64_t>(constInfo.headDim);
                    DataCopyPad(selectionKvCacheGm_[dstKvAddr],
                        kvMergUb_[srcStart * static_cast<int64_t>(constInfo.headDim)],
                        updateKvParams);

                    if (constInfo.headDimRope > 0) {
                        DataCopyExtParams updateRopeParams = ropeParams;
                        updateRopeParams.blockCount = runLen;
                        int64_t dstRopeAddr =
                            (static_cast<int64_t>(destinationBlockNum) * selectionKvBlockSize_ +
                             destinationBlockOffset) *
                            static_cast<int64_t>(constInfo.headDimRope);
                        DataCopyPad(selectionKRopeGm_[dstRopeAddr],
                            ropeMergUb_[srcStart * static_cast<int64_t>(constInfo.headDimRope)],
                            updateRopeParams);
                    }
                }
                updateIdx += runLen;
            }
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        }
        localOffset += writeCount;
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::MergeKvFromSelection(
    const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);

    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ?
        0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ?
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }

    int64_t selectionRow = GetSelectionRow(runInfo);
    int64_t topkCount = static_cast<int64_t>(constInfo.sparseBlockCount);
    int64_t statusBase = selectionRow * selectionStatusStride_;
    int64_t physicalSlotBase =
        static_cast<int64_t>(runInfo.s2Idx) * static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size +
                    runInfo.gS1Idx / constInfo.gSize + 1;
    }

    int64_t localOffset = s2GmStartOffset;
    while (localOffset < s2GmLimit) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }

        int64_t physicalSlot = physicalSlotBase + localOffset;
        int64_t selectionBlockTableIdx = physicalSlot / selectionKvBlockSize_;
        if (selectionBlockTableIdx >= selectionMaxBlockNum_) {
            break;
        }
        int32_t selectionBlockNum = selectionKvBlockTableGm_.GetValue(
            selectionRow * selectionMaxBlockNum_ + selectionBlockTableIdx);
        if (selectionBlockNum < 0) {
            break;
        }

        int64_t selectionBlockOffset = physicalSlot % selectionKvBlockSize_;
        int64_t bufferedTokenCount = mte2Size - mte3Size;
        int64_t runCount = 32 - bufferedTokenCount;
        int64_t remainingTokenCount = s2GmLimit - localOffset;
        if (runCount > remainingTokenCount) {
            runCount = remainingTokenCount;
        }
        int64_t remainingSelectionBlock = selectionKvBlockSize_ - selectionBlockOffset;
        if (runCount > remainingSelectionBlock) {
            runCount = remainingSelectionBlock;
        }

        int64_t validRunCount = 0;
        for (; validRunCount < runCount; validRunCount++) {
            int32_t tokenId = selectionKvBlockStatusGm_.GetValue(
                statusBase + physicalSlot + validRunCount);
            if (tokenId < 0 || static_cast<int64_t>(tokenId) >= s2IdLimit) {
                break;
            }
        }
        if (validRunCount == 0) {
            break;
        }

        int64_t selectionTokenOffset =
            static_cast<int64_t>(selectionBlockNum) * selectionKvBlockSize_ +
            selectionBlockOffset;
        CopyInSelectionKvRun(
            mte2Size, mte3Size, mergeMte3Idx, selectionTokenOffset, validRunCount);
        localOffset += validRunCount;

        bool flushCurrentBuffer = mte2Size - mte3Size >= 32 ||
                                  localOffset >= s2GmLimit || validRunCount < runCount;
        if (flushCurrentBuffer) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
        if (validRunCount < runCount) {
            break;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams zeroParams;
        zeroParams.blockCount = 1;
        zeroParams.blockLen = constInfo.headDim * sizeof(KV_T);
        zeroParams.srcStride = 0;
        zeroParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size;
             s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(
                kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           s2GmOffset * constInfo.headDim],
                kvMergUb_, zeroParams);
        }
        zeroParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size;
             s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(
                kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           512 * constInfo.headDim + s2GmOffset * constInfo.headDimRope],
                kvMergUb_, zeroParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams validSizeParams;
    validSizeParams.blockCount = 1;
    validSizeParams.blockLen = 128 * sizeof(int32_t);
    validSizeParams.srcStride = 0;
    validSizeParams.dstStride = 0;
    DataCopyPad(
        kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) +
                       GetSubBlockIdx() * 128],
        v0ValidSizeUb_, validSizeParams);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::
MergeKvFromSelectionWithSparseUpdates(const RunInfo &runInfo)
{
    int64_t selectionRow = GetSelectionRow(runInfo);
    int64_t membershipBase = selectionRow * selectionMembershipStride_;
    LocalTensor<int16_t> sparsePlanLocal =
        v0ValidSizeBuff.Get<int16_t>()[SELECTION_STATUS_UB_OFFSET *
            sizeof(int32_t) / sizeof(int16_t)];
    DataCopyExtParams sparsePlanParams;
    sparsePlanParams.blockCount = 1;
    sparsePlanParams.blockLen =
        SELECTION_SPARSE_PLAN_VALUE_COUNT * sizeof(int16_t);
    sparsePlanParams.srcStride = 0;
    sparsePlanParams.dstStride = 0;
    DataCopyPadExtParams<int16_t> sparsePlanPadParams{false, 0, 0, 0};
    DataCopyPad(sparsePlanLocal,
        selectionMembershipMapGm_[membershipBase + selectionUpdatePlanOffset_],
        sparsePlanParams, sparsePlanPadParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(1);
    WaitFlag<AscendC::HardEvent::MTE2_S>(1);

    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    bool needWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);

    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 :
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ?
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }

    int64_t physicalSlotBase =
        static_cast<int64_t>(runInfo.s2Idx) * static_cast<int64_t>(constInfo.s2BaseSize);
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size +
                    runInfo.gS1Idx / constInfo.gSize + 1;
    }

    int32_t updateIdx = 0;
    int64_t rangeStart = physicalSlotBase + s2GmStartOffset;
    while (updateIdx < selectionUpdatePlanCount_ &&
           static_cast<int32_t>(sparsePlanLocal.GetValue(updateIdx * 2)) - 1 <
               rangeStart) {
        updateIdx++;
    }

    int64_t localOffset = s2GmStartOffset;
    while (localOffset < s2GmLimit) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }

        int64_t physicalSlot = physicalSlotBase + localOffset;
        while (updateIdx < selectionUpdatePlanCount_ &&
               static_cast<int32_t>(sparsePlanLocal.GetValue(updateIdx * 2)) - 1 <
                   physicalSlot) {
            updateIdx++;
        }
        int64_t nextDestinationSlot = static_cast<int64_t>(constInfo.sparseBlockCount);
        if (updateIdx < selectionUpdatePlanCount_) {
            nextDestinationSlot = static_cast<int32_t>(
                sparsePlanLocal.GetValue(updateIdx * 2)) - 1;
        }

        if (nextDestinationSlot == physicalSlot) {
            int32_t tokenId = static_cast<int32_t>(
                sparsePlanLocal.GetValue(updateIdx * 2 + 1)) - 1;
            int64_t keyOffset = GetKeyGmOffset(tokenId, runInfo, s2IdLimit);
            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, tokenId,
                           keyOffset, -1, s2IdLimit, runInfo);
            localOffset++;
            updateIdx++;
        } else {
            int64_t selectionBlockTableIdx = physicalSlot / selectionKvBlockSize_;
            if (selectionBlockTableIdx >= selectionMaxBlockNum_) {
                break;
            }
            int32_t selectionBlockNum = selectionKvBlockTableGm_.GetValue(
                selectionRow * selectionMaxBlockNum_ + selectionBlockTableIdx);
            if (selectionBlockNum < 0) {
                break;
            }

            int64_t selectionBlockOffset = physicalSlot % selectionKvBlockSize_;
            int64_t runCount = 32 - (mte2Size - mte3Size);
            int64_t remainingTokenCount = s2GmLimit - localOffset;
            if (runCount > remainingTokenCount) {
                runCount = remainingTokenCount;
            }
            int64_t remainingSelectionBlock =
                selectionKvBlockSize_ - selectionBlockOffset;
            if (runCount > remainingSelectionBlock) {
                runCount = remainingSelectionBlock;
            }
            if (nextDestinationSlot > physicalSlot &&
                runCount > nextDestinationSlot - physicalSlot) {
                runCount = nextDestinationSlot - physicalSlot;
            }
            if (runCount <= 0) {
                break;
            }

            int64_t selectionTokenOffset =
                static_cast<int64_t>(selectionBlockNum) * selectionKvBlockSize_ +
                selectionBlockOffset;
            CopyInSelectionKvRun(mte2Size, mte3Size, mergeMte3Idx,
                                 selectionTokenOffset, runCount);
            localOffset += runCount;
        }

        if (mte2Size - mte3Size >= 32 || localOffset >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset,
                              mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams zeroParams;
        zeroParams.blockCount = 1;
        zeroParams.blockLen = constInfo.headDim * sizeof(KV_T);
        zeroParams.srcStride = 0;
        zeroParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size;
             s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(
                kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           s2GmOffset * constInfo.headDim],
                kvMergUb_, zeroParams);
        }
        zeroParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size;
             s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(
                kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           512 * constInfo.headDim +
                           s2GmOffset * constInfo.headDimRope],
                kvMergUb_, zeroParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams validSizeParams;
    validSizeParams.blockCount = 1;
    validSizeParams.blockLen = 128 * sizeof(int32_t);
    validSizeParams.srcStride = 0;
    validSizeParams.dstStride = 0;
    DataCopyPad(
        kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) +
                       GetSubBlockIdx() * 128],
        v0ValidSizeUb_, validSizeParams);
}

// b s1 k
template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::MergeKv(const RunInfo &runInfo)
{
    selectionUpdatePlanActive_ = false;
    selectionSparseUpdatePlanActive_ = false;
    selectionUpdatePlanOffset_ = 0;
    selectionUpdatePlanCount_ = 0;
    selectionDirectRowStride_ = 0;
    selectionPairedCopyActive_ = false;
    directWrittenWindowMask_ = 0;
    mergeSplitActive_ = false;   // 默认不切；只有纯主路径且 helperPartCount_>1 才置位
    bool useSelectionPlanSource = false;
    bool useExternalPlanSource = false;
    int64_t selectionRow = -1;
    int64_t selectionMembershipBase = -1;
    if (UseSetResidentSelection()) {
        selectionRow = GetSelectionRow(runInfo);
        selectionDataRow_ = selectionRow;
        selectionMembershipBase = selectionRow * selectionMembershipStride_;
        LocalTensor<int16_t> controlLocal =
            v0ValidSizeBuff.Get<int16_t>()[SELECTION_STATUS_UB_OFFSET * sizeof(int32_t) / sizeof(int16_t)];
        bool membershipMapReady =
            IsSelectionMembershipMapReady(selectionMembershipBase, controlLocal);
        int16_t planMarker = controlLocal.GetValue(1);
        selectionSparseUpdatePlanActive_ =
            planMarker == SELECTION_SPARSE_PLAN_READY_MARKER;
        selectionUpdatePlanActive_ = selectionSparseUpdatePlanActive_ ||
            planMarker == SELECTION_PLAN_READY_MARKER ||
            planMarker == SELECTION_EXTERNAL_PLAN_READY_MARKER;
        useExternalPlanSource =
            planMarker == SELECTION_EXTERNAL_PLAN_READY_MARKER;
        int64_t externalPhysicalRow =
            static_cast<int64_t>(controlLocal.GetValue(4));
        if (useExternalPlanSource && externalPhysicalRow >= 0) {
            selectionDataRow_ = externalPhysicalRow;
            int64_t externalDirectRowStride =
                static_cast<int64_t>(controlLocal.GetValue(6));
            if (controlLocal.GetValue(5) == SELECTION_DIRECT_LAYOUT_MARKER &&
                externalDirectRowStride > 0) {
                selectionDirectRowStride_ = externalDirectRowStride;
            }
            selectionPairedCopyActive_ =
                controlLocal.GetValue(7) == SELECTION_PAIRED_COPY_MARKER &&
                selectionDirectRowStride_ > 0;
        }
        if (!selectionUpdatePlanActive_ && membershipMapReady) {
            // helper 不走 selection 变体：token 范围没切，由 target 全量搬（信号照发）
            if (!isHelperV0_) {
                MergeKvFromSelection(runInfo);
            }
            return;
        }
        selectionUpdatePlanOffset_ = selectionUpdatePlanActive_ ?
            static_cast<int64_t>(controlLocal.GetValue(3)) : 0;
        selectionUpdatePlanCount_ = selectionSparseUpdatePlanActive_ ?
            static_cast<int32_t>(controlLocal.GetValue(2)) : 0;
        if (selectionSparseUpdatePlanActive_ && selectionUpdatePlanCount_ > 0 &&
            selectionUpdatePlanCount_ <= SELECTION_SYNC_COPY_CAPACITY) {
            // helper 不走 selection 变体：同上，target 全量搬
            if (!isHelperV0_) {
                MergeKvFromSelectionWithSparseUpdates(runInfo);
            }
            return;
        }
        useSelectionPlanSource = !selectionSparseUpdatePlanActive_ &&
            selectionUpdatePlanActive_ &&
            controlLocal.GetValue(2) > 0;
        // plan source（LRU）路径 helper 也参与搬运（helper-extend 合并线放开，对应
        // helper-only 仓 db33ab90）：helper 只读 plan（读到本核自己的 UB，无副作用）+
        // 搬数据进 target 的 kvMergeGm_（InitHelperV0 已指向 target 核）；跨核副作用
        // （plan 标记清理、UB 直写、status 更新）全部由 target 侧做——helper 的
        // directWritebackActive 下方强制 false，helper 那半的 update 由回读路
        // （CopyOutSelectionUpdateFromKvMerge）兜底。
    }

    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == FusedSparseAttentionOverlapLayout::TND) {
        uint64_t actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(runInfo.bIdx - 1);
        topkGmBaseOffset += (actualSeqQPrefixSum + runInfo.gS1Idx / constInfo.gSize) * constInfo.kvHeadNum *
                            constInfo.sparseBlockCount + runInfo.n2Idx * constInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += runInfo.bIdx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            runInfo.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    // 从 UB 直发回写要三样：能读到计划值（才知道哪些是 update 条目、写去哪个槽位）、
    // 一个 token 一格（sparseBlockSize == 1，与回写路的换位逻辑同一前提）、
    // 块表可用。sparse 计划那条路在上面已经分流走了。
    bool directWritebackActive = useSelectionPlanSource && IsSelectionUpdateEnabled() &&
        constInfo.sparseBlockSize == 1 && selectionKvBlockSize_ > 0 && selectionMaxBlockNum_ > 0 &&
        !isHelperV0_;   // helper 不 UB 直写：它那半的 update 由回读路兜底（target 侧回读
                        // 覆盖全范围；mask 只标 target 自己直写过的窗口，helper 的不标）
    int64_t directSelectionRow = selectionDataRow_ >= 0 ? selectionDataRow_ : selectionRow;
    int64_t directUpdateDstSlot[32];
    int64_t directUpdateUbIdx[32];
    int64_t directUpdateCount = 0;
    // 第一批照收，之后由 FlushDirectSelectionUpdate 按上一批的密度决定还收不收
    bool directCollectEnabled = directWritebackActive;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    // helper：把本子核的半区再对半切，target 搬前半、helper 搬后半（两个子核
    // 同理，合起来一块 4 路并行）。plan source 路径同样切（helper-extend 合并线
    // 放开，对应 helper-only 仓 db33ab90）。
    // partSize 必须按 2*sparseBlockSize 对齐 —— 主循环按 token 对步进。
    mergeSplitActive_ = helperPartCount_ > 1;
    int64_t s2GmStartOffset;
    int64_t s2GmLimit;
    if (mergeSplitActive_) {
        int64_t subBlockStart = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        int64_t subBlockLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
        if (subBlockLimit > s2ProcessSize) {
            subBlockLimit = s2ProcessSize;
        }
        int64_t alignUnit = 2L * constInfo.sparseBlockSize;
        int64_t subBlockSize = subBlockLimit - subBlockStart;
        if (subBlockSize < 0) {
            subBlockSize = 0;
        }
        int64_t partSize = CeilDiv(subBlockSize, (int64_t)helperPartCount_ * alignUnit) * alignUnit;
        s2GmStartOffset = subBlockStart + helperPartIdx_ * partSize;
        s2GmLimit = subBlockStart + (helperPartIdx_ + 1) * partSize;
        if (s2GmLimit > subBlockLimit) {
            s2GmLimit = subBlockLimit;
        }
        if (s2GmStartOffset > s2GmLimit) {
            s2GmStartOffset = s2GmLimit;
        }
    } else {
        s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        s2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: s2ProcessSize;
    }
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }

    int64_t cachedSelectionBlockTableIdx = -1;
    int32_t cachedSelectionBlockNum = -1;
    LocalTensor<int16_t> selectionPlanLocal =
        v0ValidSizeBuff.Get<int16_t>()[SELECTION_STATUS_UB_OFFSET * sizeof(int32_t) / sizeof(int16_t)];
    LocalTensor<uint32_t> packedSelectionPlanLocal =
        selectionPlanLocal.ReinterpretCast<uint32_t>();
    if (useSelectionPlanSource && s2GmStartOffset < s2GmLimit) {
        int64_t logicalPlanStart =
            static_cast<int64_t>(runInfo.s2Idx) *
                static_cast<int64_t>(constInfo.s2BaseSize) +
            s2GmStartOffset;
        int64_t planCount = s2GmLimit - s2GmStartOffset;
        DataCopyExtParams planParams;
        planParams.blockCount = 1;
        planParams.blockLen = static_cast<uint32_t>(planCount * sizeof(int16_t));
        planParams.srcStride = 0;
        planParams.dstStride = 0;
        DataCopyPadExtParams<int16_t> planPadParams{false, 0, 0, 0};
        DataCopyPad(selectionPlanLocal,
            selectionMembershipMapGm_[selectionMembershipBase +
                                      selectionUpdatePlanOffset_ + logicalPlanStart],
            planParams, planPadParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(1);
        WaitFlag<AscendC::HardEvent::MTE2_S>(1);
    }
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size + runInfo.gS1Idx / constInfo.gSize + 1;
    }
#if FSAO_PROBE_BULK_TOPK
    // 批量预取 topk 索引到 UB，后面循环中不再逐 token 标量读 GM。
    // 只对 sparseBlockSize == 1 生效（生产路径），>1 时回退到逐 token 读。
    // 放在 v0ValidSizeUb_ 的高区 [1024, 1536)，与 plan 数据 [512, 768) 和
    // valid size [0, 128) 均不重叠。
    LocalTensor<int32_t> topkBatchUb = v0ValidSizeUb_[1024];
    bool useBulkTopk = (constInfo.sparseBlockSize == 1) && (s2GmStartOffset < s2GmLimit);
    if (useBulkTopk) {
        int64_t batchTopkStart = topkGmBaseOffset +
            static_cast<int64_t>(runInfo.s2Idx) * static_cast<int64_t>(constInfo.s2BaseSize) +
            s2GmStartOffset;
        int64_t batchTopkCount = s2GmLimit - s2GmStartOffset;
        DataCopy(topkBatchUb, topkGm_[batchTopkStart],
                 static_cast<uint32_t>(batchTopkCount));
    }
#endif
    if (constInfo.sparseBlockSize == 1) {
    // ===== 窗口化两段式主循环（无分支批量段）=====
    // 移植自 helper-only 仓 d756c697，适配主线两处差异：
    //   1) CopyInKv 主线带 selectionTokenOffset/ubTokenIdx 出参（hit 走 selection 路
    //      不进这里，miss 传 -1；-1,-1）；
    //   2) miss 分支顺带做 UB 直写收集（slotBuf 的负值即本窗 update 目标槽 -值-1），
    //      Phase C 末尾 FlushDirectSelectionUpdate——窗口 mask 语义与逐对版一致，
    //      回读路的跳过判据不受影响。
    // 旧版按 token 对迭代（128 次/半区），每对一串数据依赖分支：plan 解码、
    // run 检测 while（scattered 下几乎每对都白转一轮、还多读一轮 plan）、
    // 奇偶回退、逐对 flush 判断 —— 这是 aiv_scalar 固定部分的主体。
    // 新版按 32-token 窗口迭代，窗内两步：
    //   Phase A 批量解码本窗 plan 到 slotBuf（-1=非 hit）：plan 的 GetValue 次数
    //           减半（删掉 run while 的重读），无任何分支；
    //   Phase B 窗内游标顺序执行：hit 走 selection（连续 slot 段用 Run 整段搬、
    //           相邻单 hit 用 Pair 合一笔），miss 走 topk+块表（相邻双 miss 交给
    //           CopyInKv 配对）。所有 CopyIn* 函数原样复用；搬运目的地由 token
    //           逻辑位置决定、与处理顺序无关 —— 数据与逐对版逐字节等价。
    //   无效 token（topk 越界 / >= s2IdLimit）只在序列尾部出现：遇到即停，
    //   剩余由循环后的补零段统一填零（与逐对版同一模型）。
    const int32_t s2Start32 = (int32_t)s2GmStartOffset;
    const int32_t s2Limit32 = (int32_t)s2GmLimit;
    const int32_t sparseBlockSize32 = (int32_t)constInfo.sparseBlockSize;
    int32_t slotBuf[32];   // 原始紧凑 plan 值：>0 = hit（槽=值-1）、<0 = update（槽=-值-1）、0 = 纯 miss
    bool mergeStopped = false;
    for (int32_t winStart = s2Start32; winStart < s2Limit32 && !mergeStopped; winStart += 32) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        int32_t winCount = s2Limit32 - winStart;
        if (winCount > 32) {
            winCount = 32;
        }
        // ---- Phase A：批量解码本窗 plan（plan 已在 UB；无 plan 时全 0 = 全 miss）----
        // 存原始紧凑 plan 值而不是解码后的槽位：符号就是类型 —— >0 = hit（槽位 =
        // 值-1）、<0 = update 条目（目标槽位 = -值-1）、0 = 普通 miss。UB 直写收集
        //（优化05）要的 update 位因此零成本保留，Phase B 用同一套比较完成分流。
        //（单数组方案移植自 yq/opt-merge-0723：每 token 一次写、无分支；替代旧的
        // slotBuf+updateSlotBuf 双数组解码。）
        int32_t localBase = winStart - s2Start32;
        for (int32_t j = 0; j < winCount; j += 2) {
            int32_t plan0 = 0;
            int32_t plan1 = 0;
            if (useSelectionPlanSource) {
                uint32_t packedPlanValue =
                    packedSelectionPlanLocal.GetValue((localBase + j) / 2);
                plan0 = static_cast<int16_t>(packedPlanValue & 0xFFFFU);
                plan1 = static_cast<int16_t>(packedPlanValue >> 16);
            }
            slotBuf[j] = plan0;
            if (j + 1 < winCount) {
                slotBuf[j + 1] = plan1;
            }
        }
        // ---- Phase B：窗内顺序执行 ----
        int32_t cur = 0;
        while (cur < winCount && !mergeStopped) {
            int32_t tokenPos = winStart + cur;
            if (slotBuf[cur] > 0) {
                // hit：换算 selection 偏移（direct 布局纯算术；非 direct 走块表缓存）
                int64_t selectionTokenOffset = GetSelectionSlotTokenOffset(
                    selectionDataRow_, slotBuf[cur] - 1, cachedSelectionBlockTableIdx,
                    cachedSelectionBlockNum);
                if (selectionTokenOffset < 0) {
                    mergeStopped = true;
                    break;
                }
                // 非 external 计划时逐对版会用 topk 校验有效性，保持一致
                if (!useExternalPlanSource) {
                    GetRealS2Idx(tokenPos, s2IdxArray0, topkGmBaseOffset, runInfo);
                    if (s2IdxArray0 < 0 || s2IdxArray0 >= s2IdLimit) {
                        mergeStopped = true;
                        break;
                    }
                }
                // 连续 slot 段：整段搬（run 识别在本地数组上做，无 plan 重读）
                int32_t runLen = 1;
                while (cur + runLen < winCount &&
                       slotBuf[cur + runLen] == slotBuf[cur] + runLen) {
                    runLen++;
                }
                if (selectionDirectRowStride_ <= 0 && selectionKvBlockSize_ > 0) {
                    // 非 direct（PA 块表）布局：run 不跨 selection block 边界
                    int32_t inBlock = (int32_t)(selectionKvBlockSize_ -
                        (slotBuf[cur] - 1) % selectionKvBlockSize_);
                    if (runLen > inBlock) {
                        runLen = inBlock;
                    }
                }
                if (runLen >= 2) {
                    CopyInSelectionKvRun(
                        mte2Size, mte3Size, mergeMte3Idx, selectionTokenOffset, runLen);
                    cur += runLen;
                    continue;
                }
                // 单 hit：相邻也是单 hit（slot 不连续）时尝试 Pair 合一笔。
                // 先按偏移排序再试（移植自 yq/opt-merge-0723 的优化01）——
                // CanUsePairedSelectionCopy 要求后一个偏移更大（stride 恒正），
                // 不排的话降序对（scattered 下约一半）全部配不上、退化成两次
                // CopyInSingleKv（1.5 条搬运/token → 1.0）。两个都是命中，在
                // 合并缓冲里的先后可以换：attention 对 key 次序不敏感（与
                // CopyInKv 内部排序同一论证）；命中不是 update 条目，回写不受影响。
                if (cur + 1 < winCount && slotBuf[cur + 1] > 0 &&
                    slotBuf[cur + 1] != slotBuf[cur] + 1 && selectionPairedCopyActive_) {
                    int64_t selectionTokenOffset2 = GetSelectionSlotTokenOffset(
                        selectionDataRow_, slotBuf[cur + 1] - 1, cachedSelectionBlockTableIdx,
                        cachedSelectionBlockNum);
                    if (selectionTokenOffset2 >= 0) {
                        int64_t pairLo = selectionTokenOffset;
                        int64_t pairHi = selectionTokenOffset2;
                        if (pairHi < pairLo) {
                            pairLo = selectionTokenOffset2;
                            pairHi = selectionTokenOffset;
                        }
                        int64_t kvSrcStride = 0;
                        int64_t ropeSrcStride = 0;
                        if (CanUsePairedSelectionCopy(pairLo, pairHi,
                                kvSrcStride, ropeSrcStride)) {
                            CopyInSelectionKvPair(mte2Size, mte3Size, mergeMte3Idx,
                                pairLo, kvSrcStride, ropeSrcStride);
                            cur += 2;
                            continue;
                        }
                    }
                }
                CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, 0, -1,
                               selectionTokenOffset, s2IdLimit, runInfo);
                cur += 1;
            } else {
                // miss：读 topk + 块表（55ns×miss 的 GM 读 stall，本重构不碰）
                GetRealS2Idx(tokenPos, s2IdxArray0, topkGmBaseOffset, runInfo);
                if (unlikely(s2IdxArray0 < 0 || s2IdxArray0 >= s2IdLimit)) {
                    mergeStopped = true;
                    break;
                }
                int64_t ubTokenIdx0 = -1;
                int64_t ubTokenIdx1 = -1;
                if (cur + 1 < winCount && slotBuf[cur + 1] <= 0) {
                    // 相邻双 miss：交给 CopyInKv 配对（内部 CanUsePairedKvCopy 兜底；
                    // 第二个无效时 blockCount=1 只搬第一个）
                    GetRealS2Idx(tokenPos + sparseBlockSize32, s2IdxArray1,
                                 topkGmBaseOffset, runInfo);
                    CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1,
                             -1, -1, s2IdLimit, runInfo, ubTokenIdx0, ubTokenIdx1);
                    // UB 直写收集：本对里属于 update 条目的（slotBuf < 0）且
                    // 真落进 UB 的（ubTokenIdxN >= 0）记下来，与逐对版同一模型；
                    // 超 32 条整批作废退回回读路（同逐对版）。
                    if (directCollectEnabled && directUpdateCount >= 0) {
                        bool takeUpdate0 = slotBuf[cur] < 0 && ubTokenIdx0 >= 0;
                        bool takeUpdate1 = slotBuf[cur + 1] < 0 && ubTokenIdx1 >= 0;
                        if (directUpdateCount + (int64_t)takeUpdate0 + (int64_t)takeUpdate1 > 32) {
                            directUpdateCount = -1;
                        } else {
                            if (takeUpdate0) {
                                directUpdateUbIdx[directUpdateCount] = ubTokenIdx0;
                                directUpdateDstSlot[directUpdateCount] = -slotBuf[cur] - 1;
                                directUpdateCount++;
                            }
                            if (takeUpdate1) {
                                directUpdateUbIdx[directUpdateCount] = ubTokenIdx1;
                                directUpdateDstSlot[directUpdateCount] = -slotBuf[cur + 1] - 1;
                                directUpdateCount++;
                            }
                        }
                    }
                    if (s2IdxArray1 < 0 || s2IdxArray1 >= s2IdLimit) {
                        mergeStopped = true;
                    }
                    cur += 2;
                } else {
                    CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, -1,
                             -1, -1, s2IdLimit, runInfo, ubTokenIdx0, ubTokenIdx1);
                    if (directCollectEnabled && directUpdateCount >= 0) {
                        bool takeUpdate0 = slotBuf[cur] < 0 && ubTokenIdx0 >= 0;
                        if (directUpdateCount + (int64_t)takeUpdate0 > 32) {
                            directUpdateCount = -1;
                        } else if (takeUpdate0) {
                            directUpdateUbIdx[directUpdateCount] = ubTokenIdx0;
                            directUpdateDstSlot[directUpdateCount] = -slotBuf[cur] - 1;
                            directUpdateCount++;
                        }
                    }
                    cur += 1;
                }
            }
        }
        // ---- Phase C：倒出本窗 + flush 直写 + 推进（flush 判断从逐对降到每窗一次）----
        CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
        // count=0 可信的两个前提（plan 解码过 + 收集开着）必须调用前留底：
        // Flush 内收集常开（c089）会把 collectEnabled 重新打开；helper 核首轮
        // 没收过、无 plan-source 时整批没收过，那些 count=0 都是"没查"而非
        // "查过没有"，不能据以置掩码（见 FlushDirectSelectionUpdate）。
        bool batchCollected = directCollectEnabled && useSelectionPlanSource;
        FlushDirectSelectionUpdate(directUpdateDstSlot, directUpdateUbIdx,
            directUpdateCount, directCollectEnabled, mte3Size, mte2Size,
            mergeMte3Idx, directSelectionRow, batchCollected);
        mte3Size = mte2Size;
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
        mergeMte3Idx++;
        needWaitMte3ToMte2 = true;
    }
    } else {
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        int16_t currentPlanValue0 = SELECTION_COMPACT_PLAN_INVALID;
        int16_t currentPlanValue1 = SELECTION_COMPACT_PLAN_INVALID;
        if (useSelectionPlanSource) {
            int64_t bufferedTokenCount = mte2Size - mte3Size;
            int64_t selectionRunLimit = 32 - bufferedTokenCount;
            int64_t remainingTokenCount = s2GmLimit - s2GmOffsetArray;
            if (selectionRunLimit > remainingTokenCount) {
                selectionRunLimit = remainingTokenCount;
            }
            int64_t localPlanStart = s2GmOffsetArray - s2GmStartOffset;
            uint32_t packedPlanValue =
                packedSelectionPlanLocal.GetValue(localPlanStart / 2);
            currentPlanValue0 = static_cast<int16_t>(packedPlanValue & 0xFFFFU);
            currentPlanValue1 = static_cast<int16_t>(packedPlanValue >> 16);
            int32_t firstSourceSlot = IsSelectionPlanHit(currentPlanValue0) ?
                DecodeSelectionPlanSlot(currentPlanValue0) : -1;
            int64_t selectionRunStart = GetSelectionSlotTokenOffset(
                selectionDataRow_, firstSourceSlot, cachedSelectionBlockTableIdx,
                cachedSelectionBlockNum);
            if (firstSourceSlot >= 0) {
                int64_t remainingInSelectionBlock = selectionDirectRowStride_ > 0 ?
                    selectionDirectRowStride_ - firstSourceSlot :
                    selectionKvBlockSize_ - firstSourceSlot % selectionKvBlockSize_;
                if (selectionRunLimit > remainingInSelectionBlock) {
                    selectionRunLimit = remainingInSelectionBlock;
                }
            }
            int64_t selectionRunCount = selectionRunStart >= 0 ? 1 : 0;
            if (selectionRunCount == 1 && selectionRunLimit > 1 &&
                IsSelectionPlanHit(currentPlanValue1) &&
                DecodeSelectionPlanSlot(currentPlanValue1) == firstSourceSlot + 1) {
                selectionRunCount = 2;
            }
            while (selectionRunCount >= 2 && selectionRunCount < selectionRunLimit) {
                int64_t nextPlanStart = localPlanStart + selectionRunCount;
                uint32_t nextPackedPlanValue =
                    packedSelectionPlanLocal.GetValue(nextPlanStart / 2);
                int16_t nextPlanValue0 =
                    static_cast<int16_t>(nextPackedPlanValue & 0xFFFFU);
                if (!IsSelectionPlanHit(nextPlanValue0) ||
                    DecodeSelectionPlanSlot(nextPlanValue0) !=
                        firstSourceSlot + selectionRunCount) {
                    break;
                }
                selectionRunCount++;
                if (selectionRunCount >= selectionRunLimit) {
                    break;
                }
                int16_t nextPlanValue1 =
                    static_cast<int16_t>(nextPackedPlanValue >> 16);
                if (!IsSelectionPlanHit(nextPlanValue1) ||
                    DecodeSelectionPlanSlot(nextPlanValue1) !=
                        firstSourceSlot + selectionRunCount) {
                    break;
                }
                selectionRunCount++;
            }
            if (selectionRunCount > 2 && (selectionRunCount & 1) != 0 &&
                s2GmOffsetArray + selectionRunCount < s2GmLimit) {
                selectionRunCount--;
            }
            if (selectionRunCount >= 2) {
                CopyInSelectionKvRun(
                    mte2Size, mte3Size, mergeMte3Idx, selectionRunStart, selectionRunCount);
                if (mte2Size - mte3Size >= 32 ||
                    s2GmOffsetArray + selectionRunCount >= s2GmLimit) {
                    CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
                    // 成段快路上全是命中，不产生新的 update 条目，但这一批 UB 里
                    // 混着之前配对路攒下的，一并发掉。
                    bool batchCollected = directCollectEnabled && useSelectionPlanSource;  // 留底含义见窗口化主循环 Phase C
                    FlushDirectSelectionUpdate(directUpdateDstSlot, directUpdateUbIdx,
                        directUpdateCount, directCollectEnabled, mte3Size, mte2Size,
                        mergeMte3Idx, directSelectionRow, batchCollected);
                    mte3Size = mte2Size;
                    SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
                    mergeMte3Idx++;
                    needWaitMte3ToMte2 = true;
                }
                s2GmOffsetArray += selectionRunCount - 2 * constInfo.sparseBlockSize;
                continue;
            }
        }
        int64_t selectionTokenOffset0 = -1;
        int64_t selectionTokenOffset1 = -1;
        if (useSelectionPlanSource) {
            int32_t sourceSlot0 = IsSelectionPlanHit(currentPlanValue0) ?
                DecodeSelectionPlanSlot(currentPlanValue0) : -1;
            selectionTokenOffset0 = GetSelectionSlotTokenOffset(
                selectionDataRow_, sourceSlot0, cachedSelectionBlockTableIdx,
                cachedSelectionBlockNum);
            if (s2GmOffsetArray + constInfo.sparseBlockSize < s2GmLimit) {
                int32_t sourceSlot1 = IsSelectionPlanHit(currentPlanValue1) ?
                    DecodeSelectionPlanSlot(currentPlanValue1) : -1;
                selectionTokenOffset1 = GetSelectionSlotTokenOffset(
                    selectionDataRow_, sourceSlot1, cachedSelectionBlockTableIdx,
                    cachedSelectionBlockNum);
            }
        }
        bool externalHit0 = useExternalPlanSource &&
            constInfo.sparseBlockSize == 1 &&
            selectionTokenOffset0 >= 0;
        bool externalHit1 = useExternalPlanSource &&
            constInfo.sparseBlockSize == 1 &&
            selectionTokenOffset1 >= 0;
        if (externalHit0) {
            s2IdxArray0 = 0;
        }
#if FSAO_PROBE_BULK_TOPK
        else if (useBulkTopk) {
            s2IdxArray0 = topkBatchUb.GetValue(s2GmOffsetArray - s2GmStartOffset);
        }
#endif
        else {
            GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        }
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            // 这条路不更新 mte3Size，但 CopyOutMrgeResult 写出去的就是 [mte3Size, mte2Size)，
            // flush 用同一段范围。
            bool batchCollected = directCollectEnabled && useSelectionPlanSource;  // 留底含义见窗口化主循环 Phase C
            FlushDirectSelectionUpdate(directUpdateDstSlot, directUpdateUbIdx,
                directUpdateCount, directCollectEnabled, mte3Size, mte2Size,
                mergeMte3Idx, directSelectionRow, batchCollected);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            break;
        }
        if (externalHit1) {
            s2IdxArray1 = 0;
        }
#if FSAO_PROBE_BULK_TOPK
        else if (useBulkTopk) {
            s2IdxArray1 = topkBatchUb.GetValue(
                s2GmOffsetArray + constInfo.sparseBlockSize - s2GmStartOffset);
        }
#endif
        else {
            GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize,
                         s2IdxArray1, topkGmBaseOffset, runInfo);
        }
        int64_t ubTokenIdx0 = -1;
        int64_t ubTokenIdx1 = -1;
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1,
                 selectionTokenOffset0, selectionTokenOffset1, s2IdLimit, runInfo,
                 ubTokenIdx0, ubTokenIdx1);
        // 未命中的 token 刚落进 UB，趁它还在就记下要写去哪个槽位。
        // ubTokenIdxN ≥ 0 已经蕴含「这个 token 真的写进了 UB 且不是命中」。
        if (directCollectEnabled && directUpdateCount >= 0) {
            bool takeUpdate0 = ubTokenIdx0 >= 0 && IsSelectionPlanUpdate(currentPlanValue0);
            bool takeUpdate1 = s2GmOffsetArray + constInfo.sparseBlockSize < s2GmLimit &&
                ubTokenIdx1 >= 0 && IsSelectionPlanUpdate(currentPlanValue1);
            if (directUpdateCount + takeUpdate0 + takeUpdate1 > 32) {
                // 一批 32 个 token 最多 16 对、每对最多 2 条，装满恰好 32 条，
                // 走到这里说明前提被打破了。少记一条会让 flush 仍按「密」认下窗口、
                // 回写路整个跳过，漏写是静默的 —— 宁可整批作废退回老路。
                directUpdateCount = -1;
            } else {
                if (takeUpdate0) {
                    directUpdateUbIdx[directUpdateCount] = ubTokenIdx0;
                    directUpdateDstSlot[directUpdateCount] =
                        DecodeSelectionPlanSlot(currentPlanValue0);
                    directUpdateCount++;
                }
                if (takeUpdate1) {
                    directUpdateUbIdx[directUpdateCount] = ubTokenIdx1;
                    directUpdateDstSlot[directUpdateCount] =
                        DecodeSelectionPlanSlot(currentPlanValue1);
                    directUpdateCount++;
                }
            }
        }
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            bool batchCollected = directCollectEnabled && useSelectionPlanSource;  // 留底含义见窗口化主循环 Phase C
            FlushDirectSelectionUpdate(directUpdateDstSlot, directUpdateUbIdx,
                directUpdateCount, directCollectEnabled, mte3Size, mte2Size,
                mergeMte3Idx, directSelectionRow, batchCollected);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * constInfo.headDim],
                        kvMergUb_, dataCopyParams);
        }
        dataCopyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * constInfo.headDim +
                                   s2GmOffset * constInfo.headDimRope],
                        kvMergUb_, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        CrossCoreWaitFlag(constInfo.syncC2V1);
        // add nUpdate to mm2ResGm
        if (info.actualSingleProcessSInnerSize != 0) {
            ProcessAmlaNupdate(info, mSplitInfo);
            CrossCoreSetFlag<ConstInfo::FUSED_SPARSE_ATTENTION_OVERLAP_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1NupdateC2);
        }
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1) {
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
                    auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
                    auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline uint64_t FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    // 保持 0：splitKV 把「第几份」整个交给 tndCoreStartKVSplitPos，而那个值就是核号。
    // 任务号 = 核号 / 段数、段号 = 核号 % 段数，两下合起来还原成核号本身，
    // 于是每个核写自己那一格，天生不会撞车。见 kernel_mla.h 的 InitCalcParamsEach。
    return 0;
}

// splitKV 归约：把同一个请求的 partNum 份局部结果并成一份，写进 attentionOutGm。
//
// 每份存着那一段 KV 自己算出来的 attention 结果，外加那一段的 softmax 分母 sum
// 和最大值 max。合并就是按分母加权平均：
//     M    = max(各段的 max)
//     d[s] = sum[s] * exp(max[s] - M)      把各段的分母折算到同一个基准上
//     out  = Σ(结果[s] * d[s]) / Σ d[s]
// ⚠ 这是教科书 FlashDecode，与 AMLA 那套 2 的幂缩放无关 —— accumOutGm 里存的
//    已经是 DealBmm2ResBaseBlock 除干净之后的局部输出，缩放因子早消掉了。
template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ReduceSplitKV(
    uint32_t slotBegin, uint32_t partNum, uint32_t mCount, uint64_t attenOutOffset)
{
    // ⚠ 整个函数体包在 if constexpr 里，FD=0 时它一行都不进二进制。
    //    按 C++ 规则这本来就该如此（唯一的调用点在 if constexpr(false) 分支里，
    //    不算 odr-use），加这层是因为不确定这条编译器守不守 —— 而 bs=16
    //    实测退化 4.4 μs 只出现在全命中端，形状正像多出一段代码占了 I-cache。
    //    为了让 diff 只有两行，里面没有跟着缩进。
    if constexpr (FLASH_DECODE) {
    if (partNum <= 1 || mCount == 0) {
        return;
    }
    // 一个 aiCore 上有两个 vector，让偶数那个把这个请求的行全干完。
    // ⚠ 没照抄主计算那边的分半公式 —— 那边是先按 nBufferMBaseSize 分块、块内再分半，
    //    两层。归约这点活（k 份加权和）不值得为省一半时间去跟那套耦合，
    //    耦合上了以后那边一改分块方式，这边就静默算错。
    if ((GetBlockIdx() % 2) != 0) {
        return;
    }
    uint32_t mStart = 0;
    uint32_t mNum = mCount;

    const uint32_t hd = static_cast<uint32_t>(constInfo.headDim);
    const uint32_t rowsPerSlot = static_cast<uint32_t>(constInfo.kvHeadNum) * constInfo.mBaseSize;
    // 累加区放 tmpBuff1(32K)，一批最多装得下这么多行
    uint32_t mBatch = static_cast<uint32_t>(ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T)) / hd;
    if (mBatch == 0) {
        mBatch = 1;
    }

    LocalTensor<T> accUb = tmpBuff1.Get<T>();      // [rows, hd] 加权累加
    LocalTensor<T> partUb = inputBuff1.Get<T>();   // [rows, hd] 当前这一份
    LocalTensor<OUT_T> outUb = inputBuff2.Get<OUT_T>();
    LocalTensor<T> lseBuf = outputBuff1.Get<T>();  // 五张 [rows, 8] 的小表

    for (uint32_t m0 = 0; m0 < mNum; m0 += mBatch) {
        uint32_t rows = ((mNum - m0) < mBatch) ? (mNum - m0) : mBatch;
        uint32_t lseLen = rows * FP32_BLOCK_ELEMENT_NUM;
        LocalTensor<T> curMax = lseBuf;
        LocalTensor<T> curSum = lseBuf[lseLen];
        LocalTensor<T> gMax = lseBuf[lseLen * 2];
        LocalTensor<T> dSum = lseBuf[lseLen * 3];
        LocalTensor<T> wTmp = lseBuf[lseLen * 4];
        uint64_t rowBase = static_cast<uint64_t>(slotBegin) * rowsPerSlot + mStart + m0;
        uint64_t lseBase = rowBase * FP32_BLOCK_ELEMENT_NUM;
        uint64_t outBase = rowBase * hd;
        uint64_t lseStride = static_cast<uint64_t>(rowsPerSlot) * FP32_BLOCK_ELEMENT_NUM;
        uint64_t outStride = static_cast<uint64_t>(rowsPerSlot) * hd;

        // 一遍：找出各段最大值里的最大值
        for (uint32_t s = 0; s < partNum; s++) {
            DataCopy(curMax, lseMaxFdGm[lseBase + s * lseStride], lseLen);
            pipe_barrier(PIPE_ALL);
            if (s == 0) {
                Adds(gMax, curMax, ConstInfo::FLOAT_ZERO, lseLen);
            } else {
                Max(gMax, gMax, curMax, lseLen);
            }
            // ⚠ 要 ALL 不是 V：下一轮的 DataCopy 会盖掉 curMax，
            //    得先等这一轮读完，而 PIPE_V 拦不住搬运提前动手。
            pipe_barrier(PIPE_ALL);
        }

        // 二遍：算各段的权重，加权累加分子与分母
        Duplicate(accUb, ConstInfo::FLOAT_ZERO, rows * hd);
        Duplicate(dSum, ConstInfo::FLOAT_ZERO, lseLen);
        pipe_barrier(PIPE_V);
        for (uint32_t s = 0; s < partNum; s++) {
            DataCopy(curMax, lseMaxFdGm[lseBase + s * lseStride], lseLen);
            DataCopy(curSum, lseSumFdGm[lseBase + s * lseStride], lseLen);
            DataCopy(partUb, accumOutGm[outBase + s * outStride], rows * hd);
            pipe_barrier(PIPE_ALL);
            Sub(wTmp, curMax, gMax, lseLen);
            pipe_barrier(PIPE_V);
            Exp(wTmp, wTmp, lseLen);
            pipe_barrier(PIPE_V);
            Mul(curSum, curSum, wTmp, lseLen); // d[s]：这一段折算后的分母
            pipe_barrier(PIPE_V);
            Add(dSum, dSum, curSum, lseLen);
            RowMuls(partUb, partUb, curSum, rows, hd, hd);
            pipe_barrier(PIPE_V);
            Add(accUb, accUb, partUb, rows * hd);
            pipe_barrier(PIPE_ALL);   // 同上：下一轮要把 partUb 重新搬一遍
        }

        // ⚠ 不原地除：RowDivs 的现有调用点 dst 都与 src 分开，别在这里开先例
        RowDivs(partUb, accUb, dSum, rows, hd, hd);
        pipe_barrier(PIPE_V);
        if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双
            Cast(outUb, partUb, AscendC::RoundMode::CAST_RINT, rows * hd);
        } else {
            Cast(outUb, partUb, AscendC::RoundMode::CAST_ROUND, rows * hd);
        }
        pipe_barrier(PIPE_ALL);
        DataCopyExtParams copyParams;
        copyParams.blockCount = rows;
        copyParams.blockLen = hd * sizeof(OUT_T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(attentionOutGm[attenOutOffset + (mStart + m0) * hd], outUb, copyParams);
        pipe_barrier(PIPE_ALL);
    }
    }  // if constexpr (FLASH_DECODE)
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (info.s2Idx + 1 != info.curSInnerLoopTimes) {
        return;
    }
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename FusedSparseAttentionOverlapTraits> __aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ProcessVec2Inner(const RunInfo &info,
                                                                              const MSplitInfo &mSplitInfo,
                                                                              uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}


template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                        uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
                      info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // 份数offset
                      wsMStart * actualColumnCount;                                                                 // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    // ⚠ 这个判断原来是反的：`== 0`（这一段没有数据）时把 tmp 写出去，
    // `!= 0`（有数据）时反而清零。旁边的 ComputeLogSumExpAndCopyToGm 同一个
    // 判断是对的 —— 有数据写数据、没数据填中性值，两相对照可确认。
    // FLASH_DECODE=1 从没编译过，所以没人发现。
    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                         uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模�?
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                            mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    // 将绝对值大�?e10的数置为0
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = bmm2ResUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    pipe_barrier(PIPE_V);
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    pipe_barrier(PIPE_V);
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO,
           SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    pipe_barrier(PIPE_V);
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = info.loop % (constInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384]; // sumUb用临时内�?16 * 32B  = 512B
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    pipe_barrier(PIPE_V);
    RowDivs(bmm2ResUb, tmpBmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，�?个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，�?个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处�?56B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数�?
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256�?此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，�?个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，�?个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部�?
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat�?56B数据，正�?个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount�?
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾�?
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

// ====== helper：空闲核帮搬 MergeKv 的核间握手 ======
// 协议（每个 S2 块一轮，以流水线序号 loop 做 ring 索引）：
//   target V0 拿到 flag3 信用（= AIC 已读完该槽）→ PublishHelperCredit 发布 loop+1
//   helper V0/V1 轮询 credit 到位 → 各搬 1/4 范围 → SignalMergeDone 原子加 1
//   target V0/V1 轮询 done 到 2（按圈数折算期望）→ 并回 helper 的有效计数 → set syncV0C1
// credit 是回压通道：helper 最多领先 target 一个信用窗口，不会套圈覆写 AIC 正读的槽。
// GM 读写全部走 DataCopyPad/DataCopy（MTE2/MTE3，过 L2，跨核可见），不用 scalar 轮询。
template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitHelperSync(
    const GlobalTensor<int32_t> &doneGm, const GlobalTensor<int32_t> &creditGm, uint32_t pairs)
{
    mergeDoneGm_ = doneGm;
    mergeCreditGm_ = creditGm;
    helperSyncPairs_ = pairs > 0 ? pairs : 1;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::InitHelperV0(
    const GlobalTensor<KV_T> &targetKvMergeGm, uint32_t pairIdx)
{
    isHelperV0_ = true;
    helperPartIdx_ = 1;
    helperPartCount_ = 2;
    kvMergeGm_ = targetKvMergeGm; // helper 直接写 target 核的 merge 缓冲
    mergeSyncPairOffset_ = pairIdx * HELPER_SYNC_RING * HELPER_SYNC_SLOT_INT32;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SetHelperParts(
    uint32_t partIdx, uint32_t partCount, const GlobalTensor<int32_t> &helperValidSizeGm,
    uint32_t pairIdx)
{
    helperPartIdx_ = partIdx;
    helperPartCount_ = partCount;
    helperValidSizeGm_ = helperValidSizeGm;
    mergeSyncPairOffset_ = pairIdx * HELPER_SYNC_RING * HELPER_SYNC_SLOT_INT32;
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::PublishHelperCredit(uint32_t loop)
{
    if (GetSubBlockIdx() != 0) {
        return; // credit 只由 target 的 V0 发一次，helper 两个子核都读它
    }
    // scratch 按 loop 轮换 4 槽：同槽复用隔着 4 个流水线迭代（μs 级），
    // 32B 的 MTE3 拷贝早已落地，省掉等拷贝的 MTE3_S 同步，发布即发即弃
    LocalTensor<int32_t> scratch = helperSyncBuff.Get<int32_t>()[(loop & 3) * HELPER_SYNC_SLOT_INT32];
    scratch.SetValue(0, static_cast<int32_t>(loop + 1));
    SetFlag<AscendC::HardEvent::S_MTE3>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::S_MTE3>(HELPER_SYNC_EVENT_ID);
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = HELPER_SYNC_SLOT_INT32 * sizeof(int32_t);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPad(mergeCreditGm_[mergeSyncPairOffset_ + (loop % HELPER_SYNC_RING) * HELPER_SYNC_SLOT_INT32], scratch, params);
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::WaitHelperCredit(uint32_t loop)
{
    // target 发布的是 loop+1，单调递增：ring 绕圈后旧值必然小于期望值，不会误判
    int32_t expected = static_cast<int32_t>(loop + 1);
    if (helperCreditSeen_ >= expected) {
        return; // 已读到的信用水位覆盖本轮（初始 4 信用爆发期），零 GM 流量
    }
    LocalTensor<int32_t> scratch = helperSyncBuff.Get<int32_t>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = HELPER_SYNC_SLOT_INT32 * sizeof(int32_t);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    uint32_t spinsink = 0;
    while (true) {
        DataCopyPad(scratch, mergeCreditGm_[mergeSyncPairOffset_ + (loop % HELPER_SYNC_RING) * HELPER_SYNC_SLOT_INT32],
                    params, padParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(HELPER_SYNC_EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_S>(HELPER_SYNC_EVENT_ID);
        helperCreditSeen_ = scratch.GetValue(0);
        if (helperCreditSeen_ >= expected) {
            break;
        }
        // 轮询降频：两次 GM 读之间垫一段标量空转，把 MTE2 从事务风暴里卸下来。
        // LCG 乘法链 + UB 写，编译器折不掉；scratch[1] 没人读，值无所谓
        for (uint32_t i = 0; i < HELPER_POLL_BACKOFF; ++i) {
            spinsink = spinsink * 1664525 + 1013904223;
        }
        scratch.SetValue(1, spinsink);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::SignalMergeDone(uint32_t loop)
{
    // MergeKv 的 kvMerge 数据和 validSize 都是 MTE3 写，全部落地后才能计数
    pipe_barrier(PIPE_ALL);
    LocalTensor<int32_t> scratch = helperSyncBuff.Get<int32_t>()[HELPER_SYNC_SLOT_INT32];
    scratch.SetValue(0, 1);
    SetFlag<AscendC::HardEvent::S_MTE3>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::S_MTE3>(HELPER_SYNC_EVENT_ID);
    // 照抄 sparse_flash_attention 的原子加写法：DataCopyParams 的 blockLen 以 32B 为单位
    DataCopyParams params;
    params.blockCount = 1;
    params.blockLen = HELPER_SYNC_SLOT_INT32 * sizeof(int32_t) / 32;
    params.srcStride = 0;
    params.dstStride = 0;
    SetAtomicAdd<int32_t>();
    DataCopy(mergeDoneGm_[mergeSyncPairOffset_ + (loop % HELPER_SYNC_RING) * HELPER_SYNC_SLOT_INT32], scratch, params);
    SetAtomicNone();
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::WaitMergeDone(uint32_t loop)
{
    // 同一槽每绕 ring 一圈被 helper 两个子核各 +1，期望值按圈数折算
    //（scratch 在 [32]，避开 credit 发布的 4 槽轮换区 [0,32) 这段）
    uint32_t slot = loop % HELPER_SYNC_RING;
    int32_t expected = static_cast<int32_t>((loop / HELPER_SYNC_RING + 1) * HELPER_SIGNALS_PER_BLOCK);
    LocalTensor<int32_t> scratch = helperSyncBuff.Get<int32_t>()[4 * HELPER_SYNC_SLOT_INT32];
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = HELPER_SYNC_SLOT_INT32 * sizeof(int32_t);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    uint32_t spinsink = 0;
    while (true) {
        DataCopyPad(scratch, mergeDoneGm_[mergeSyncPairOffset_ + slot * HELPER_SYNC_SLOT_INT32], params, padParams);
        SetFlag<AscendC::HardEvent::MTE2_S>(HELPER_SYNC_EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_S>(HELPER_SYNC_EVENT_ID);
        if (scratch.GetValue(0) >= expected) {
            break;
        }
        // 轮询降频：两次 GM 读之间垫一段标量空转（LCG 乘法链 + UB 写防优化），
        // 把 MTE2 从事务风暴里卸下来；scratch[1] 没人读，值无所谓
        for (uint32_t i = 0; i < HELPER_POLL_BACKOFF; ++i) {
            spinsink = spinsink * 1664525 + 1013904223;
        }
        scratch.SetValue(1, spinsink);
    }
}

template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::FixValidSize(
    const RunInfo &info)
{
    if (!mergeSplitActive_) {
        return; // selection 路径 helper 没搬，valid 计数本来就是完整半区的
    }
    // 把 helper 那 1/4 范围的有效计数并回来。helper 的计数在它自己核的
    // kvValidSizeGm_[mergeSlot*256 + subBlock*128 + mergeSlot]；自己的部分计数在
    // v0ValidSizeUb_[mergeSlot] 还留着（MergeKv 尾部写的），不用回读 GM。
    // 两个计数都在各自 128 块的首 32B 内（mergeSlot < 4），按 32B 块读写即可。
    // 本函数在 setV0C1 之前调：valid 计数在下一轮迭代被 Vec1L 读，给它覆盖顺序的
    // PIPE_MTE3 旗标只有 syncV0C1 一个，fix 挪到旗标之后就和消费者失去序了。
    uint32_t mergeSlot = info.loop % MERGE_CACHE_GM_BUF_NUM;
    uint32_t blockBase = mergeSlot * (128 * 2) + GetSubBlockIdx() * 128;
    LocalTensor<int32_t> helperChunk = helperSyncBuff.Get<int32_t>()[5 * HELPER_SYNC_SLOT_INT32];
    LocalTensor<int32_t> writeChunk = helperSyncBuff.Get<int32_t>()[6 * HELPER_SYNC_SLOT_INT32];
    DataCopyExtParams fixParams;
    fixParams.blockCount = 1;
    fixParams.blockLen = HELPER_SYNC_SLOT_INT32 * sizeof(int32_t);
    fixParams.srcStride = 0;
    fixParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    // 自己 MergeKv 尾部的 validSize 块写（MTE3）先落地，后面的 32B 写回不能冲到它前面
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(HELPER_SYNC_EVENT_ID);
    DataCopyPad(helperChunk, helperValidSizeGm_[blockBase], fixParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::MTE2_S>(HELPER_SYNC_EVENT_ID);
    int32_t merged = v0ValidSizeUb_.GetValue(mergeSlot) + helperChunk.GetValue(mergeSlot);
    writeChunk.SetValue(mergeSlot, merged); // 其余 7 格随便，消费端只读 [mergeSlot]
    SetFlag<AscendC::HardEvent::S_MTE3>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::S_MTE3>(HELPER_SYNC_EVENT_ID);
    DataCopyPad(kvValidSizeGm_[blockBase], writeChunk, fixParams);
}

// 清零握手区（done 2KB + credit 2KB）。仅 core0 的 V0 在 Process 的 SyncAll 前调一次。
// 必须走 MTE3（DataCopyPad）：本算子的跨核可见性先例（splitKV 归约）就是
// MTE3 写 + SyncAll。标量 store（S 管）与 SyncAll 旗标/原子加之间没有跨管序
// 保证 —— 曾经 1024 次标量写清零在背靠背调用下被 helper 的首次原子加反超，
// done 被抹回 0，target 死等（scattered 路 helper 信号极快才踩中，plan=off 没事）。
template <typename FusedSparseAttentionOverlapTraits>
__aicore__ inline void
FusedSparseAttentionOverlapVectorService<FusedSparseAttentionOverlapTraits>::ZeroHelperSyncArea()
{
    // helperSyncBuff 2KB 恰好装满 512 个 int32 零；ring 按 pair 隔离，逐 pair 清 done+credit。
    // ⚠ 不要改成 blockCount=pairs 单发：DataCopyPad 的 srcStride=0 是"连续推进源地址"，
    // 会从 2KB 的 zeroLocal 越界读出 pairs×2KB（bs≥2 时 UB 越界）。
    LocalTensor<int32_t> zeroLocal = helperSyncBuff.Get<int32_t>();
    Duplicate(zeroLocal, static_cast<int32_t>(0), HELPER_SYNC_RING * HELPER_SYNC_SLOT_INT32);
    SetFlag<AscendC::HardEvent::V_MTE3>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::V_MTE3>(HELPER_SYNC_EVENT_ID);
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = HELPER_SYNC_RING * HELPER_SYNC_SLOT_INT32 * sizeof(int32_t);
    params.srcStride = 0;
    params.dstStride = 0;
    for (uint32_t pair = 0; pair < helperSyncPairs_; pair++) {
        uint32_t pairOffset = pair * HELPER_SYNC_RING * HELPER_SYNC_SLOT_INT32;
        DataCopyPad(mergeDoneGm_[pairOffset], zeroLocal, params);
        DataCopyPad(mergeCreditGm_[pairOffset], zeroLocal, params);
    }
    // 拷贝落地再返回，下一次调用复用 zeroLocal 才安全
    SetFlag<AscendC::HardEvent::MTE3_V>(HELPER_SYNC_EVENT_ID);
    WaitFlag<AscendC::HardEvent::MTE3_V>(HELPER_SYNC_EVENT_ID);
}

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_SERVICE_VECTOR_MLA_H
