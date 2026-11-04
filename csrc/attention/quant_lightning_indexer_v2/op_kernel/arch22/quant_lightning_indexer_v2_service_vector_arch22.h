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
 * \file quant_lightning_indexer_v2_service_vector_arch22.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"
#include "quant_lightning_indexer_v2_vector.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
using namespace QLIV2ServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t ELE_NUM_32 = 32;
constexpr uint32_t ELE_NUM_128 = 128;
constexpr uint32_t ELE_NUM_512 = 512;

template <typename QLIV2T>
class QLIV2Vector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;
    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline QLIV2Vector(){};
    __aicore__ inline void ProcessVec0(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessVec1(const QLIV2Common::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                      const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                      const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void ProcessLD();
    // mode=1 + LD：块级 topk 分段结果的归并与输出（与位置级归并链完全同构，宽度换成块级口径）
    __aicore__ inline void ReduceBlockTopkRow(uint32_t rowInBlock);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm,
                                                  GlobalTensor<float> blockVec1ResGm);
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                  GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                  GlobalTensor<int32_t> outputIdxOffsetGm,
                                                  bool outputIdxOffsetValid);
    // D3a: 候选 gather —— 每个 AIV 搬 16 个候选块（8 token × 128B）到 UB，再写 GM staging ring。
    // V1 阶段 cube 尚未消费（探针模式），仅验证地址/吞吐；D3b 接入 ring 握手。
    __aicore__ inline void InitVecCandidateGatherTensor(GlobalTensor<typename QLIV2T::keyType> keyGm,
                                                        GlobalTensor<int8_t> stagingKeyGm,
                                                        GlobalTensor<half> stagingScaleGm,
                                                        GlobalTensor<int32_t> candLenGm,
                                                        GlobalTensor<float> candScoreGm,
                                                        GlobalTensor<half> candAuxScaleGm, bool enabled);
    // v4 候选任务驱动（对齐 DSL 软件流水）：按 token 交错执行 ——
    //   先 gather token t 的全部候选 tile（生产），再消费 token t-1 的分数（消费滞后一拍）；
    // taskSeq 决定分数区奇偶槽；任务内最后一个 token 的消费延迟到下一任务（或核尾
    // ConsumePendingCandScore），保证每次分数读之前 AIC 已继续推进过（GM 写已落地）。
    __aicore__ inline void RunCandTask(uint32_t bIdx, uint32_t n2Idx, uint32_t rowBase, uint32_t taskTokens,
                                       uint32_t maxTiles, uint32_t taskSeq, uint32_t s1RowBase, uint32_t actS1Size,
                                       uint32_t actS2Size, uint32_t actS2SizeOrig, uint32_t indiceOutOffset,
                                       uint32_t outputIdxOffsetBase);
    // 消费队首的一条 pending 记录（队列满时由 RunCandTask 调用）
    __aicore__ inline void ConsumePendingCandScore();
    // 核尾排空：消费队列中所有 pending 记录
    // （调用前须等 AIC 主循环结束令牌；由 kernel 在 ProcessMain 末尾触发）
    __aicore__ inline void DrainPendingCandScore();
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline int32_t AlignS2(int32_t cuS2Len);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    // mode=1 + LD：块级 topk 分段结果备份区（布局与 vec1ResGm 相同，行槽位宽 = 2 × candidateTopkBlocks）
    GlobalTensor<float> blockVec1ResGm_;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateTopkIndexInGm;
    GlobalTensor<int32_t> candidateTopkIndexOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // D3a 候选 gather：K 源、GM staging（K/scale）与使能标志
    GlobalTensor<typename QLIV2T::keyType> keyGmCand_;
    GlobalTensor<int8_t> candStagingKeyGm_;
    GlobalTensor<half> candStagingScaleGm_;
    GlobalTensor<int32_t> candLenGm_;       // B5: 每行有效候选块数
    GlobalTensor<float> candScoreGm_;       // B5: AIC 写出的候选分数（每 (token,tile) 256 float）
    GlobalTensor<half> candAuxScaleGm_;     // v4: 每 tile 32 块 × 16 half 的 kscale（任务奇偶双缓冲）
    bool candGatherEnabled_ = false;
    // v4: 候选 tile 生产序号（每个 tile 自增一次，槽号 = seq % CAND_RING_SLOTS；
    // 与 AIC 侧消费序号按同一顺序推进，独立计数即可对齐）
    uint32_t candProdTileSeq_ = 0U;
    // v4: 跨任务延迟消费的"任务末尾 token"记录队列（深度 8，正常保留 ≤7 条）。
    // 每任务压 2 条（最后两个 token）⇒ 队首约延迟 3 个任务再消费，对 1-tile 任务也保证
    // ≥3 个后续 tile 的 AIC 计算量，覆盖 GM 写落地延迟；核尾由 DrainPendingCandScore 排空。
    static constexpr uint32_t CAND_PENDING_SCORE_SLOTS = 8;
    struct PendingCandScore {
        bool valid = false;
        uint32_t bIdx = 0;
        uint32_t rowBase = 0;
        uint32_t s1RowBase = 0;
        uint32_t n2Idx = 0;
        uint32_t tokenIdx = 0;
        uint32_t maxTiles = 0;
        uint32_t actS1Size = 0;
        uint32_t actS2Size = 0;
        uint32_t actS2SizeOrig = 0;
        uint32_t indiceOutOffset = 0;
        uint32_t outputIdxOffsetBase = 0;
        uint64_t scoreSlotBase = 0;
    };
    PendingCandScore pendingScore_[CAND_PENDING_SCORE_SLOTS];
    uint32_t pendingHead_ = 0U;
    uint32_t pendingCount_ = 0U;
    // =================================常量区=================================

private:
    __aicore__ inline void GetKeyScale(const QLIV2Common::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // candidate (two-level topk)
    __aicore__ inline int32_t CountGE(const LocalTensor<float> &sortedDesc, int32_t n, float x);
    __aicore__ inline void BuildCandidateMask(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                              int32_t cuBaseS2Idx, int32_t innerS1Idx);
    __aicore__ inline void ProcessCandBlockTopk(const QLIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t cuS2Len,
                                                int32_t cuS2LenVecAlign, int32_t cuRealAcSeq, int32_t innerS1Idx);
    __aicore__ inline void CopyOutCandTopkIndex(const QLIV2Common::RunInfo &info, int32_t cuS1Idx, int32_t innerS1Idx);
    // mode=1 + LD：把本行块级累积器溢写到 GM，供跨核归约侧（ProcessLD）读取
    __aicore__ inline void SpillBlockTopkToGm(const QLIV2Common::RunInfo &info, int32_t blockRowIdx,
                                              int32_t innerS1Idx);
    // v4 候选驱动：单 token 的 gather / 分数消费（由 RunCandTask 按 token 交错调用）
    __aicore__ inline void GatherTokenCandidates(uint32_t bIdx, uint32_t n2Idx, uint32_t rowBase, uint32_t tokenIdx,
                                                 uint32_t maxTiles, uint64_t scoreSlotBase);
    __aicore__ inline void ConsumeCandScoreToken(uint32_t bIdx, uint32_t rowBase, uint32_t s1RowBase, uint32_t n2Idx,
                                                 uint32_t tokenIdx, uint32_t maxTiles, uint32_t actS1Size,
                                                 uint32_t actS2Size, uint32_t actS2SizeOrig, uint32_t indiceOutOffset,
                                                 uint32_t outputIdxOffsetBase, uint64_t scoreSlotBase);
    // 记录任务最后一个 token（压入 pending 队列）
    __aicore__ inline void RecordPendingCandScore(uint32_t bIdx, uint32_t rowBase, uint32_t s1RowBase, uint32_t n2Idx,
                                                  uint32_t tokenIdx, uint32_t maxTiles, uint32_t actS1Size,
                                                  uint32_t actS2Size, uint32_t actS2SizeOrig, uint32_t indiceOutOffset,
                                                  uint32_t outputIdxOffsetBase, uint64_t scoreSlotBase);
    // candidate 新增向量 op 的 64 元素分块封装 (本版本 Level-2 count > 64 的 vmax 等指令会触发 aicore 异常)
    __aicore__ inline void VecMaxPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecMinPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src, int32_t n);
    __aicore__ inline void VecMinsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecMaxsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecSubPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    __aicore__ inline void VecAddsScalar(const LocalTensor<float> &dst, const LocalTensor<float> &src, float v,
                                         int32_t n);
    __aicore__ inline void VecMulPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                      const LocalTensor<float> &src1, int32_t n);
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    // tmp buff for candidate (two-level topk)
    TBuf<TPosition::VECCALC> blockSortOutBuf_; // mode=1: 块级 topk 累加器 [CeilDiv(s1BaseSize,2), 2048, 2]
    TBuf<TPosition::VECCALC> candBuf_;         // mode=2: 排序后的候选块索引 [2048, 2] (value+idx)
    TBuf<TPosition::VECCALC> candConstBuf_;    // mode=1/2: negHuge 常量 [2048]
    TBuf<TPosition::VECCALC> candGatherScaleBuf_; // D3a: gather 的 scale 暂存 (16 块 × 8 token × 2B)

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> globalBlockTopkUb_;
    LocalTensor<float> candIsOut_;       // mode=2: position 级 0/1 (1=候选外), 仅用于分数降级 (R11 leak)
    LocalTensor<float> candNegHuge_;     // -1e30 常量
    GlobalTensor<int32_t> outputIdxOffsetGm_;        // A15: 每行输出索引偏移 (仅 sparse_indices)
    bool isOutputIdxOffsetValid_ = false;            // A15: offset 是否传入 (经 InitVecCandidateTensor 传递)

    int32_t blockId_ = -1;
    // P3: topk 自适应宽度 — 累加器/抽取/LD 归并按 sparseCount 收缩 (64 对齐),
    // 动态 topk (topk<2048) 时消除固定 2048 宽度的无效排序合并与搬运
    uint32_t topkWidth_ = BASE_TOPK;        // Align(sparseCount, 64)
    uint32_t topkPairWidth_ = 4096;         // 2 * topkWidth_ (value+index 对)
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

    // para for LD
    uint32_t mrgListNum_ = 4;

    struct QLIV2Common::ConstInfo constInfo_;
    struct QLIV2Common::LdSplitCoreInfo ldInfo_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::GetKeyScale(const QLIV2Common::RunInfo &runInfo,
                                                        const LocalTensor<half> &resUb, int64_t batchId,
                                                        int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        // A11: k_scale 0 轴非连续 — 块基址 = 表值 x keyDequantScaleStride0 (真实 stride);
        // 0 时兜底原紧凑公式 (kCacheBlockSize_), 现网行为不变 (R8)
        // Paged hybrid caches can span more than 2^31 scale elements. Promote
        // before multiplication so high physical block IDs do not wrap.
        int64_t kScaleBlkStride = constInfo_.keyDequantScaleStride0 != 0 ?
                                      static_cast<int64_t>(constInfo_.keyDequantScaleStride0) : kCacheBlockSize_;
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
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb,
                                 kScaleGm[blockId * kScaleBlkStride + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * kScaleBlkStride], copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2);                                    // 32KB
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));                                         // 8 KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                        // 8 KB
    pipe->InitBuffer(tmpBuf_, 64 * 1024);                                                              // 64KB
    // P3: 累加器按 topk 宽度分配 (topk<2048 时省 UB; 2048 时与原一致)
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_ * sizeof(float));

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // candidate (two-level topk) 按模式分配, mode=3 不占用额外 UB
    // UB 预算 (192KB): 基础 144KB + mode=1 blockSortOutBuf 32KB = 176KB / mode=2 candBuf 16KB + candConstBuf 8KB = 168KB
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
        pipe->InitBuffer(blockSortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float)); // 32KB
        globalBlockTopkUb_ = blockSortOutBuf_.Get<float>();
        InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
    } else if (constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER) {
        // R6 修复: candBuf 按行分区 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行, 行间 tile0 重排序
        // 会互相覆盖) — 2 行 x 2048 对 x 8B = 32KB; mode=2 UB 预算 144+32+8=184KB <= 192KB
        pipe->InitBuffer(candBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));
        pipe->InitBuffer(candConstBuf_, BASE_TOPK * sizeof(float)); // 8KB: negHuge
        candNegHuge_ = candConstBuf_.Get<float>();
        Duplicate(candNegHuge_.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_HUGE_F32, BASE_TOPK);
        PipeBarrier<PIPE_V>();
    }
    // D3a: gather 的 scale 暂存（16 块 × 8 token × 2B = 256B，按 512B 分配）
    if (candGatherEnabled_) {
        pipe->InitBuffer(candGatherScaleBuf_, 512);
    }

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    // P3: LD 归并缓冲按 topk 实际宽度收缩 (mrgListNum_ 列表 × 每列表 value+idx 对)
    // mode=1（candidate source，候选块来源）下同一组缓冲在位置级归并之后还要复用给块级归并：
    // 块级每行对宽 = 2 × candidateTopkBlocks，故按两者较大值分配，不再单开一组块级缓冲
    // （UB 预算 192KB：位置级 16KB×2 + 块级 64KB×2 若各开一组会吃掉 160KB，复用后只占 128KB）。
    uint32_t mrgPairWidth = topkPairWidth_;
    uint32_t mrgOutWidth = topkWidth_;
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
        uint32_t candPairWidth = 2U * constInfo_.candidateTopkBlocks;
        mrgPairWidth = (candPairWidth > mrgPairWidth) ? candPairWidth : mrgPairWidth;
        mrgOutWidth = (constInfo_.candidateTopkBlocks > mrgOutWidth) ? constInfo_.candidateTopkBlocks : mrgOutWidth;
    }
    pipe->InitBuffer(ldToBeMrgBuf_, mrgPairWidth * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, mrgPairWidth * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, mrgOutWidth * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, mrgOutWidth * sizeof(int32_t));
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                                       const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                                       const QLIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 2048
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    // P3: 按 sparseCount 计算实际 topk 工作宽度 (对齐 64; MergeSort/ExtractIndex 元素粒度需求)
    topkWidth_ = QLIV2Common::Align(static_cast<uint32_t>(constInfo.sparseCount),
                                    static_cast<uint32_t>(64));
    topkPairWidth_ = 2 * topkWidth_;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecInputTensor(GlobalTensor<half> weightsGm,
                                                               GlobalTensor<half> qScaleGm, GlobalTensor<half> kScaleGm,
                                                               GlobalTensor<int32_t> indiceOutGm,
                                                               GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                                   GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                   GlobalTensor<float> vec1ResGm,
                                                                   GlobalTensor<float> blockVec1ResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec0OutGm = vec0OutGm;
    this->blockVec1ResGm_ = blockVec1ResGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecCandidateTensor(GlobalTensor<int32_t> candidateTopkIndexInGm,
                                                                   GlobalTensor<int32_t> candidateTopkIndexOutGm,
                                                                   GlobalTensor<int32_t> outputIdxOffsetGm,
                                                                   bool outputIdxOffsetValid)
{
    this->candidateTopkIndexInGm = candidateTopkIndexInGm;
    this->candidateTopkIndexOutGm = candidateTopkIndexOutGm;
    this->outputIdxOffsetGm_ = outputIdxOffsetGm;  // A15
    this->isOutputIdxOffsetValid_ = outputIdxOffsetValid;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::AllocEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::FreeEventID()
{}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    QLIV2ServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
    // mode=1: candidate_topk_index 同步填 -1 (行偏移 = sparse 偏移 / sparseCount * candidateTopkBlocks)
    if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
        uint64_t candOffset = static_cast<uint64_t>(invalidS1offset) / constInfo_.sparseCount *
                              constInfo_.candidateTopkBlocks;
        LocalTensor<float> candULocal = outQueue_.AllocTensor<float>();
        LocalTensor<int32_t> candIdxLocal = candULocal.template ReinterpretCast<int32_t>();
        Duplicate(candIdxLocal, constInfo_.INVALID_IDX, constInfo_.candidateTopkBlocks);
        outQueue_.EnQue<float>(candULocal);
        candULocal = outQueue_.DeQue<float>();
        QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[candOffset], candIdxLocal, constInfo_.candidateTopkBlocks);
        outQueue_.FreeTensor(candULocal);
    }
}


// Level-0 vector repeats avoid large Level-2 counts and per-repeat dispatch overhead.
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Max(dst[off], src0[off], src1[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Min(dst[off], src0[off], src1[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAbs(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                                   int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Abs(dst[off], src[off], mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMinsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Mins(dst[off], src[off], v, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMaxsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Maxs(dst[off], src[off], v, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecSubPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Sub(dst[off], src0[off], src1[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecAddsScalar(const LocalTensor<float> &dst,
                                                          const LocalTensor<float> &src, float v, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Adds(dst[off], src[off], v, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::VecMulPair(const LocalTensor<float> &dst, const LocalTensor<float> &src0,
                                                       const LocalTensor<float> &src1, int32_t n)
{
    {
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Mul(dst[off], src0[off], src1[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
}

// 统计降序序列 (偶数 lane 为 value) 中 >= x 的元素个数, 二分实现
template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::CountGE(const LocalTensor<float> &sortedDesc, int32_t n, float x)
{
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (sortedDesc.GetValue(2 * mid) >= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// mode=2 (use_candidate): 加载/排序候选行 (每行首个S2分片), 判定 tile 内块归属,
// 产出 position 级 isOut (fp32 0/1, 1=候选外) 与其 int32 形式
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::BuildCandidateMask(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                               int32_t cuBaseS2Idx, int32_t innerS1Idx)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    // R6 修复: 同核多行 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行) 共享 candBuf 时,
    // 后一行的 tile0 重排序会覆盖前行候选 (s2 内层循环按 gS1 块整体推进) — candBuf 按行分区
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> candPairs = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    if (info.isFirstS2InnerLoop) {
        // 加载候选行到 tmp 尾部, 转 fp32 后与有序索引组成 [values | idx] 对并降序排序
        LocalTensor<int32_t> candInt = tmp[12288].template ReinterpretCast<int32_t>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = candBlocks * sizeof(int32_t);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(candInt,
                             candidateTopkIndexInGm[info.candidateOutOffset +
                                                    cuS1Idx * constInfo_.candidateTopkBlocks],
                             copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        Cast(candPairs, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        DataCopy(candPairs[candBlocks].template ReinterpretCast<int32_t>(), globalTopkIndice_, candBlocks);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> candSortTmp = tmp[4096];
        QLIV2ServiceVec::SortAll(candPairs, candSortTmp, candBlocks);
        PipeBarrier<PIPE_V>();
        // V->S 围栏: 排序结果对后续标量 GetValue (CountGE 二分) 可见
        // (PipeBarrier<PIPE_V> 不足以保证标量读到 V 写数据; 参照 CANN topk_v200 的 SetFlag/WaitFlag<V_S> 模式)
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    // 候选与 tile 块号 [tileBlockBase, tileBlockBase+tileBlkNum) 的最小距离, 0 即命中
    // (candBuf 按行分区, R6: 同核行间覆盖已修复)
    LocalTensor<float> candPairs2 = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    int32_t lo = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase));
    int32_t hi = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase + tileBlkNum));
    // 降序排序下: 值 >= base+tileBlkNum 占 [0, hi), 落在 tile 内的候选占 [hi, lo), < base (含 -1 pad) 占 [lo, ...)
    // 注：原来这里先用 Cast+Adds 造 tile 块号向量、再逐候选 |块号-候选| 取 min（每候选 3 条向量指令），
    // 现改为下面的块级命中位图，块号向量与整条距离链删除。
    // 块级命中位图：先用一次 Duplicate 全置 1（候选外），再把落在本 tile 内的候选块号
    // （排序后落在 [hi, lo) 区间，块号必在 [tileBlockBase, tileBlockBase+tileBlkNum)）标量置 0。
    // 替代原来"逐候选 Adds→VecAbs→VecMinPair"的距离链：那里每个候选要 3 条向量指令 + 3 次流水 drain，
    // 本 shape 每 tile 约 32 个候选 → 约 96 次 drain；改为 1 次 Duplicate + N 次标量写 + 1 次 S_V 围栏。
    // （v220 的 ReduceMin 是整张量归约到 1 个最小值，不能做"每 8 个候选取 min"，故不走 2D 归约路线。）
    // 用 float 存 0.0f/1.0f（不能存 int32 的 0/1 再 ReinterpretCast —— 1 会变成 1.4e-45 的 denormal）
    LocalTensor<float> blkHit = tmp[8192];   // [tileBlkNum]：0.0f=候选内, 1.0f=候选外
    Duplicate(blkHit, 1.0f, tileBlkNum);
    PipeBarrier<PIPE_V>();
    // 标量写不受 PipeBarrier<PIPE_V> 围栏（设计文档 P2 铁律），必须显式 V→S / S→V
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    for (int32_t j = hi; j < lo; j++) {
        int32_t candBlk = static_cast<int32_t>(candPairs2.GetValue(2 * j));
        blkHit.SetValue(candBlk - tileBlockBase, 0.0f);
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    // 位图展开到位置级（每块 candidate_block_size 个位置），语义与原来的"距离 clamp 到 1"一致
    LocalTensor<float> posDist = tmp[12288]; // candInt 已释放, 可复用
    Brcb(posDist, blkHit, tileBlkNum / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    VecMinsScalar(posDist, posDist, 1.0f, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    candIsOut_ = posDist;
    // R11 (leak 语义, §1.6): isOut 仅用于分数降级 (pen 链), 索引保留真实位置号 —
    // 候选外可达位置作为 topk 填充泄漏成有效索引 (对齐模型 where(idxs < compress_lens) 语义),
    // 不再把候选外索引改 -1 (原 isOutI32/CAST_RINT 链已删)。
}

// mode=1 (is_candidate_source): 块内 amax (log2(blockSize) 轮 Max 树) + pin 尾块 + 块级排序/归并
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessCandBlockTopk(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                                 int32_t cuS2Len, int32_t cuS2LenVecAlign,
                                                                 int32_t cuRealAcSeq, int32_t innerS1Idx)
{
            int32_t blkLen = cuS2LenVecAlign;
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t blockNum = blkLen / blockSize;                         // 含尾块 (pad -inf 后按对齐长度计)
    int32_t realBlockNum = (cuS2Len + blockSize - 1) / blockSize; // 含有效位置的块数
    // 所有向量 count 按 64 对齐 (非对齐 count 的向量指令在本架构触发 aicore 异常);
    // 且块数必须过 AlignS2 (32*(4^n)*m, m<=3): SortAll 的 MrgSort 循环按 mrgGroups/4 缩减,
    // 非 4 幂组数 (如 192 块=6 组) 会整组丢失 (实测 s2=1536 丢块 128..191 段)
    int32_t blockNumPad = (blockNum < 64) ? 64 : AlignS2(blockNum);
    int32_t tileBlockBase = info.s2Idx * s2BaseSize_ / blockSize;
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    // 块内归约: vcgmax 每 32B 块 (8 fp32) 出 1 个 max, 紧凑输出 [blockNum]
    LocalTensor<float> blkScore = tmp[6144];
    // R10: blkLen=96 是 AlignS2 输出中唯一非 64 倍数的值 (≤128 段对齐到 32 的倍数),
    // 96/64 整除截断为 1 只归约 [0,64) — 块 8..11 残留上一 tile/行的 stale 分数
    // (实测 big128k_b2_varlen b1 行 73 tile 56: 块 14344 拿到 tile 55 块 14088 的
    // 3.5568, 虚高挤掉 2048 名边界块 2760; 小 shape 总块数≤2048 集合不变故未暴露)。
    // 改 CeilDiv 覆盖全部块; 多归约的 [96,128) stale 只落在 pad 块槽位, 被 -inf
    // 位型链位精确覆盖, 无害。
    int32_t brmRepeat = CeilDiv(blkLen, 64);
    BlockReduceMax(blkScore, tmp[0], brmRepeat, 64, 1, 1, 8);
    PipeBarrier<PIPE_V>();
    int32_t lastBlk = (cuRealAcSeq - 1) / blockSize;
    int32_t pinLocal = lastBlk - tileBlockBase;
    // 块级 [scores | idx] 对: idx = tileBlockBase + j; j >= realBlockNum 的 pad 块置 -1
    // (禁止标量 SetValue: 标量写不受 PipeBarrier<PIPE_V> 围栏, 与后续 V 管道读存在确定性竞态
    //  (实测 innerS1Idx=1 行的填充被 SortAll 抢跑覆盖); 禁止 s32->f32 Cast (v220 无此组合)。
    //  改为纯 int32 向量算术: idx' = idx - (idx+1)*isPad, isPad = clamp(idx - thr, 0, 1),
    //  thr = base + realBlockNum - 1; 所有指令覆盖 blockNumPad (64 对齐), 用 Level-0 repeatTimes 批量下发。
    //  scratch 用 mode=1 独占区 [14336,15360) (mode=2 的 isOutI32 已随 R11 leak 改造删除),
    //  避开主路径 tmpSortBuf [4096,12288), 消除跨迭代残留读的隐患)
    LocalTensor<int32_t> idxScr = tmp[14336].template ReinterpretCast<int32_t>();      // 等差源
    LocalTensor<int32_t> thrI = tmp[14592].template ReinterpretCast<int32_t>();        // 阈值
    LocalTensor<int32_t> offI = tmp[14848].template ReinterpretCast<int32_t>();        // idx-thr -> isPad
    LocalTensor<int32_t> tI = tmp[15104].template ReinterpretCast<int32_t>();          // (idx+1)*isPad
    LocalTensor<int32_t> blkIdx = tmp[6144 + blockNumPad].template ReinterpretCast<int32_t>(); // 最终 idx
    ArithProgression<int32_t>(idxScr, static_cast<int32_t>(tileBlockBase), 1, blockNumPad);
    Duplicate(thrI, tileBlockBase + realBlockNum - 1, blockNumPad);
    PipeBarrier<PIPE_V>();
    {
        int32_t off = 0;
        while (off < blockNumPad) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
            Sub(offI[off], idxScr[off], thrI[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
    {
        int32_t off = 0;
        while (off < blockNumPad) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
            Maxs(offI[off], offI[off], 0, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
    {
        int32_t off = 0;
        while (off < blockNumPad) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
            Mins(offI[off], offI[off], 1, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
    {
        int32_t off = 0;
        while (off < blockNumPad) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
            Adds(tI[off], idxScr[off], 1, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
    {
        int32_t off = 0;
        while (off < blockNumPad) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
            Mul(tI[off], tI[off], offI[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
    {
        int32_t off = 0;
        while (off < blockNumPad) {
            uint64_t mask = 0;
            uint8_t repeatTimes = 0;
            QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
            Sub(blkIdx[off], idxScr[off], tI[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
            off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
        }
    }
    PipeBarrier<PIPE_V>();
    // pad 块 score 归一为 -inf 位型: 覆盖 [realBlockNum, blockNumPad), 含 stale 区 [blockNum, blockNumPad)
    // (尾 tile cuS2Len 非整块时 BlockReduceMax 只写 [0, blockNum), 其后是脏数据 —
    //  脏 score 的 (-inf,-1) 不同构对会挤进 topk 挤掉真实块, 实测 128K 场景 -1 槽超标)
    // score' = score + (score - NINF_bits)*(-isPad), isPad 于 pad 位为 1; 全 int32 向量
    if (blockNumPad > realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // -isPad
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // (score-NINF)*(-isPad)
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Duplicate(pRaw[off], -1, mask, repeatTimes, 1, B32_VEC_REPEAT_STRIDE);
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Mul(pRaw[off], pRaw[off], offI[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams()); // -isPad
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Adds(pNeg[off], scoreI[off], 8388608, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams()); // score - NINF_bits (NINF=0xFF800000=-8388608)
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Mul(pNeg[off], pNeg[off], pRaw[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams());
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Add(scoreI[off], scoreI[off], pNeg[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams()); // pad 块 -> -inf 位型
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
    }
    // pin 最新 token 所在块 (+inf 位型注入该块 score): 纯 int32 向量实现, 在 isPad 链之后
    // (复用其 scratch; 标量 SetValue 写不受 PipeBarrier<PIPE_V> fence, 与 SortAll 多轮读存在竞态, 已弃用)。
    // flag = -1 于 pin 位, 0 于其余: score' = score + (score - PINF_bits) * flag,
    // pin 位得 PINF_bits (0x7F800000 = +inf), 其余不变 (flag=0 使 pad 位 -inf 的中间溢出无害)。
    if (pinLocal >= 0 && pinLocal < realBlockNum) {
        LocalTensor<int32_t> pRaw = tmp[15360].template ReinterpretCast<int32_t>(); // [15360,15616)
        LocalTensor<int32_t> pNeg = tmp[15616].template ReinterpretCast<int32_t>(); // [15616,15872)
        LocalTensor<int32_t> pA = thrI;   // 复用 isPad 链已释放 scratch
        LocalTensor<int32_t> pB = offI;
        LocalTensor<int32_t> pFlag = tI;
        LocalTensor<int32_t> scoreI = blkScore.template ReinterpretCast<int32_t>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                // idxScr 为全局块号, pin 也必须用全局 id (tileBlockBase + pinLocal) 比较
                Adds(pRaw[off], idxScr[off], -(tileBlockBase + pinLocal), mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams()); // 0 于 pin 块
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        Duplicate(pNeg, 0, blockNumPad);
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Sub(pNeg[off], pNeg[off], pRaw[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams()); // -(idx-pinLocal)
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Maxs(pA[off], pRaw[off], 0, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Mins(pA[off], pA[off], 1, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams()); // 1 若 idx > pin
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Maxs(pB[off], pNeg[off], 0, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams());
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Mins(pB[off], pB[off], 1, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams()); // 1 若 idx < pin
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Add(pFlag[off], pA[off], pB[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams()); // isNotPin (0/1, 互斥)
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Adds(pFlag[off], pFlag[off], -1, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams()); // -1 于 pin, 0 其余
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Adds(pRaw[off], scoreI[off], -2139095040, mask, repeatTimes, QLIV2ServiceVec::B32RepeatParams()); // score - 0x7F800000 (复用 pRaw)
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Mul(pRaw[off], pRaw[off], pFlag[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams()); // (score-PINF)*flag
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
        {
            int32_t off = 0;
            while (off < blockNumPad) {
                uint64_t mask = 0;
                uint8_t repeatTimes = 0;
                QLIV2ServiceVec::VecRepeatPlan(blockNumPad - off, mask, repeatTimes);
                Add(scoreI[off], scoreI[off], pRaw[off], mask, repeatTimes, QLIV2ServiceVec::B32BinaryRepeatParams()); // pin -> +inf 位型
                off += static_cast<int32_t>(repeatTimes) * B32_VEC_ELM_NUM;
            }
        }
        PipeBarrier<PIPE_V>();
    }
    // 块级排序 + 归并到块级累加器
    LocalTensor<float> blkPairs = tmp[6144];   // [scores blockNumPad | idx blockNumPad]
    // MrgSort tmp 需容纳 mrgDst+mrgSrc = 2048*2 + blockNumPad*2 <= 4608 floats;
    // 11776 + 4608 = 16384 恰为 tmpBuf_(64KB) 末尾, 12288 起会越界 2KB (aicore 异常)
    LocalTensor<float> blkSortTmp = tmp[11776]; // 与 blkPairs 不重叠 (MrgSort src/tmp 禁止重叠)
    QLIV2ServiceVec::SortAll(blkPairs, blkSortTmp, blockNumPad);
    PipeBarrier<PIPE_V>();
    // 候选专用合并: 纯 V 回拷 (MergeSort 尾部 UB->UB DataCopy 不受 PipeBarrier<PIPE_V> fence,
    // 多 tile 下一次 MrgSort 读 acc 存在调度敏感竞态, 实测 tile1 读到 stale 数据)
    QLIV2ServiceVec::MergeSortVecCopy(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                                      constInfo_.candidateTopkBlocks, blkPairs, blockNumPad, blkSortTmp);
    PipeBarrier<PIPE_V>();
}

// mode=1: 行末直出 candidate_topk_index 并复位块级累加器
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CopyOutCandTopkIndex(const QLIV2Common::RunInfo &info, int32_t cuS1Idx,
                                                                 int32_t innerS1Idx)
{
    LocalTensor<uint32_t> candIdxULocal = outQueue_.AllocTensor<uint32_t>();
    ExtractIndex(candIdxULocal,
                 globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE].template ReinterpretCast<uint32_t>(),
                 constInfo_.candidateTopkBlocks);
    PipeBarrier<PIPE_V>();
    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
    outQueue_.EnQue<uint32_t>(candIdxULocal);
    candIdxULocal = outQueue_.DeQue<uint32_t>();
    QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[info.candidateOutOffset +
                                                     cuS1Idx * constInfo_.candidateTopkBlocks],
                             candIdxULocal.template ReinterpretCast<int32_t>(), constInfo_.candidateTopkBlocks);
    outQueue_.FreeTensor(candIdxULocal);
}

// mode=1 + LD：块级 topk 累积器溢写到 GM（与位置级 globalTopkUb_ → vec1ResGm 完全对称）。
// GM 布局 [核][S2 段][S1 行][value+index 对]，行槽位宽 = 2 × candidateTopkBlocks；
// 段内行号用 **gS1 块内相对行号** blockRowIdx，与 ProcessLD 侧 ldInfo_.mStart + j 同一坐标。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::SpillBlockTopkToGm(const QLIV2Common::RunInfo &info, int32_t blockRowIdx,
                                                               int32_t innerS1Idx)
{
    uint32_t candPairWidth = 2U * constInfo_.candidateTopkBlocks;
    uint64_t blockWsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * candPairWidth +
                             static_cast<uint64_t>(blockRowIdx) * candPairWidth;
    AscendC::DataCopyExtParams blockWsParams;
    blockWsParams.blockLen = candPairWidth * sizeof(float);
    blockWsParams.srcStride = 0;
    blockWsParams.dstStride = 0;
    blockWsParams.blockCount = 1;
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    AscendC::DataCopyPad(blockVec1ResGm_[blockWsOffset], globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE],
                         blockWsParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    // 溢写后复位本行块级累积器（下一 S2 段/下一行重新累积），与位置级同语义
    InitSortOutBuf(globalBlockTopkUb_[innerS1Idx * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec0(const QLIV2Common::RunInfo &info)
{
    // v4: 候选任务先领 W 槽回收令牌 —— 两个 AIV 都 wait，保持计数守恒
    // （AIC 每完成一个候选任务回补 1 个；预置 2 个 ⇒ AIV 最多领先 2 个任务，
    //   写入的槽是 AIC 已读完的槽，不会覆盖正在读的 W）
    if (info.candTask) {
        AscendC::CrossCoreWaitFlag(QLIV2Common::CAND_WFREE_FLAG);
    }
    // 只需要一个v核做
    if (blockId_ % 2 != 0) {
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    // 计算输出w基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*64 + aic_offset
    int64_t vec0OutGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_ * BLOCK_CUBE));
    // 计算输入weight的地址偏移，qScale的地址偏移与weight相同
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * qHeadNum_;
    // 当前需要计算的S1行数，处理尾块场景
    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuProcRealNum = cuS1ProcNum * gSize_;
    int32_t cuProcEleNum = QLIV2Common::Align(cuProcRealNum, 32); // 32: UB对齐, 参照v1

    LocalTensor<half> inWeightsUb = inQueue_.AllocTensor<half>();
    LocalTensor<half> inQScaleUb = inWeightsUb[cuProcEleNum];
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = cuProcRealNum * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPad(inWeightsUb, weightsGm[weightGmOffset], copyInParams, padParams);
    AscendC::DataCopyPad(inQScaleUb, qScaleGm[weightGmOffset], copyInParams, padParams);

    inQueue_.EnQue<half>(inWeightsUb);
    inWeightsUb = inQueue_.DeQue<half>();
    AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb, cuProcEleNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> resUb = outQueue_.AllocTensor<half>();
    AscendC::Brcb(resUb, inWeightsUb, static_cast<uint8_t>(cuProcEleNum / 8), {1, 8});
    inQueue_.FreeTensor(inWeightsUb);

    outQueue_.EnQue<half>(resUb);
    resUb = outQueue_.DeQue<half>();
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = cuProcRealNum * BLOCK_CUBE * sizeof(half);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    AscendC::DataCopyPad(vec0OutGm[vec0OutGmOffset], resUb, copyOutParams);
    outQueue_.FreeTensor(resUb);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecCandidateGatherTensor(
    GlobalTensor<typename QLIV2T::keyType> keyGm, GlobalTensor<int8_t> stagingKeyGm,
    GlobalTensor<half> stagingScaleGm, GlobalTensor<int32_t> candLenGm,
    GlobalTensor<float> candScoreGm, GlobalTensor<half> candAuxScaleGm, bool enabled)
{
    this->keyGmCand_ = keyGm;
    this->candStagingKeyGm_ = stagingKeyGm;
    this->candStagingScaleGm_ = stagingScaleGm;
    this->candLenGm_ = candLenGm;
    this->candScoreGm_ = candScoreGm;
    this->candAuxScaleGm_ = candAuxScaleGm;
    this->candGatherEnabled_ = enabled;
}

// v4 候选任务驱动（对齐 DSL 软件流水）：按 token 交错执行 ——
//   先 gather token t 的全部候选 tile（生产），再消费 token t-1 的分数（消费滞后一拍）。
// 任务内**最后一个 token** 不在此任务内消费：记入 pendingScore_，延迟到下一任务 gather 了
// 1~2 个 token 之后（或核尾 ConsumePendingCandScore）再读 —— 这样最后一次分数读之前，
// AIC 一定已经推进过至少一个 tile 的计算（规避 910B 上 GM"写后立即读"不可见的时序问题）。
// taskSeq 决定分数区奇偶槽，跨任务消费上一任务分数时不会被本任务的写覆盖。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::RunCandTask(
    uint32_t bIdx, uint32_t n2Idx, uint32_t rowBase, uint32_t taskTokens, uint32_t maxTiles, uint32_t taskSeq,
    uint32_t s1RowBase, uint32_t actS1Size, uint32_t actS2Size, uint32_t actS2SizeOrig, uint32_t indiceOutOffset,
    uint32_t outputIdxOffsetBase)
{
    if (!candGatherEnabled_ || taskTokens == 0U) {
        return;
    }
    const uint64_t scoreSlotBase = static_cast<uint64_t>(taskSeq & 1U) * QLIV2Common::CAND_SCORE_TILES_PER_TASK *
                                   QLIV2Common::CAND_TILE_TOKENS;
    if (maxTiles == 0U) {
        // 空候选任务：不产生 ring 流量与分数信号（AIC 侧同样提前返回），只做整行 -1 输出。
        // pending 队列不在此处消费（延迟策略由队列统一管理，核尾排空）。
        for (uint32_t t = 0U; t < taskTokens; t++) {
            ConsumeCandScoreToken(bIdx, rowBase, s1RowBase, n2Idx, t, maxTiles, actS1Size, actS2Size,
                                  actS2SizeOrig, indiceOutOffset, outputIdxOffsetBase, scoreSlotBase);
        }
        return;
    }
    // 任务内 token 交错：gather token t 后消费 token t-2（两 token 领先）。
    // 退两拍的原因：AIC 在算完 token 的**最后一个 tile** 后立刻发 SCORE_READY，
    // 退一拍时该 tile 的分数写还没有落地余量（随机数据下会读到 0/旧值）；
    // 退两拍后，读 token t-2 时 AIC 必然已继续推进过 token t-1 的计算。
    // 末尾两个 token 不入循环，记入 pending 队列，至少再延迟约 3 个任务消费。
    for (uint32_t t = 0U; t < taskTokens; t++) {
        GatherTokenCandidates(bIdx, n2Idx, rowBase, t, maxTiles, scoreSlotBase);
        if (t >= 2U) {
            ConsumeCandScoreToken(bIdx, rowBase, s1RowBase, n2Idx, t - 2U, maxTiles, actS1Size, actS2Size,
                                  actS2SizeOrig, indiceOutOffset, outputIdxOffsetBase, scoreSlotBase);
        }
    }
    // drain：AIC 每任务 wait 的令牌数 = tile 数 + 1，最后一个 tile 也能错拍消费
    AscendC::CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
        QLIV2Common::CAND_READY_FLAG);
    // 末尾两个 token 入队；队列满（>7 条）时消费队首（约 3 个任务之前的那条）
    if (taskTokens >= 2U) {
        RecordPendingCandScore(bIdx, rowBase, s1RowBase, n2Idx, taskTokens - 2U, maxTiles, actS1Size, actS2Size,
                               actS2SizeOrig, indiceOutOffset, outputIdxOffsetBase, scoreSlotBase);
    }
    RecordPendingCandScore(bIdx, rowBase, s1RowBase, n2Idx, taskTokens - 1U, maxTiles, actS1Size, actS2Size,
                           actS2SizeOrig, indiceOutOffset, outputIdxOffsetBase, scoreSlotBase);
    while (pendingCount_ > CAND_PENDING_SCORE_SLOTS - 1U) {
        ConsumePendingCandScore();
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::RecordPendingCandScore(
    uint32_t bIdx, uint32_t rowBase, uint32_t s1RowBase, uint32_t n2Idx, uint32_t tokenIdx, uint32_t maxTiles,
    uint32_t actS1Size,
    uint32_t actS2Size, uint32_t actS2SizeOrig, uint32_t indiceOutOffset, uint32_t outputIdxOffsetBase,
    uint64_t scoreSlotBase)
{
    const uint32_t slot = (pendingHead_ + pendingCount_) % CAND_PENDING_SCORE_SLOTS;
    PendingCandScore &rec = pendingScore_[slot];
    rec.valid = true;
    rec.bIdx = bIdx;
    rec.rowBase = rowBase;
    rec.s1RowBase = s1RowBase;
    rec.n2Idx = n2Idx;
    rec.tokenIdx = tokenIdx;
    rec.maxTiles = maxTiles;
    rec.actS1Size = actS1Size;
    rec.actS2Size = actS2Size;
    rec.actS2SizeOrig = actS2SizeOrig;
    rec.indiceOutOffset = indiceOutOffset;
    rec.outputIdxOffsetBase = outputIdxOffsetBase;
    rec.scoreSlotBase = scoreSlotBase;
    ++pendingCount_;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ConsumePendingCandScore()
{
    if (pendingCount_ == 0U) {
        return;
    }
    PendingCandScore &rec = pendingScore_[pendingHead_];
    if (rec.valid) {
        rec.valid = false;
        ConsumeCandScoreToken(rec.bIdx, rec.rowBase, rec.s1RowBase, rec.n2Idx, rec.tokenIdx, rec.maxTiles,
                              rec.actS1Size, rec.actS2Size, rec.actS2SizeOrig, rec.indiceOutOffset,
                              rec.outputIdxOffsetBase, rec.scoreSlotBase);
    }
    pendingHead_ = (pendingHead_ + 1U) % CAND_PENDING_SCORE_SLOTS;
    --pendingCount_;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::DrainPendingCandScore()
{
    while (pendingCount_ > 0U) {
        ConsumePendingCandScore();
    }
}

// 单 token gather：每个 AIV 搬本候选 tile 的一半（16 个候选块），块内 8 token × 128B 连续读。
// 候选块号来自 candidate_topk_index 原始顺序（与后续位置映射同一份表）；
// 超出有效前缀（c<0）的槽位不搬，内容由 vector 掩码丢弃。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::GatherTokenCandidates(uint32_t bIdx, uint32_t n2Idx, uint32_t rowBase,
                                                                  uint32_t tokenIdx, uint32_t maxTiles,
                                                                  uint64_t scoreSlotBase)
{
    if (!candGatherEnabled_ || maxTiles == 0U) {
        return;
    }
    using K_T = typename QLIV2T::keyType;
    constexpr uint32_t CAND_BLOCKS_PER_TILE = QLIV2Common::CAND_TILE_BLOCKS; // 32 块 = 256 token
    constexpr uint32_t CAND_BLOCKS_PER_AIV = 16U;
    constexpr uint32_t CAND_TOKENS_PER_TILE = QLIV2Common::CAND_TILE_TOKENS;
    constexpr uint32_t CAND_BLOCK_BYTES = 8U * 128U; // 8 token × D=128（int8）
    const uint32_t aivHalf = static_cast<uint32_t>(blockId_ % 2);
    const uint32_t halfBlkBase = aivHalf * CAND_BLOCKS_PER_AIV;
    const uint32_t pageBlkNum = static_cast<uint32_t>(kCacheBlockSize_) / 8U;
    if (pageBlkNum == 0U) {
        return;
    }
    // 与 Cube KeyNd2NzForPA 同一套地址口径：块基址 = 页表值 × stride(0 时兜底紧凑 stride)
    const uint64_t blkStride = (constInfo_.keyStride0 != 0) ?
        static_cast<uint64_t>(constInfo_.keyStride0) :
        static_cast<uint64_t>(kCacheBlockSize_) * static_cast<uint64_t>(kHeadNum_) * 128ULL;
    const uint64_t scaleStride = (constInfo_.keyDequantScaleStride0 != 0) ?
        static_cast<uint64_t>(constInfo_.keyDequantScaleStride0) : static_cast<uint64_t>(kCacheBlockSize_);
    const uint64_t tableBase = static_cast<uint64_t>(bIdx) * static_cast<uint64_t>(maxBlockNumPerBatch_);
    LocalTensor<half> scaleUb = candGatherScaleBuf_.Get<half>();
    const uint32_t row = rowBase + tokenIdx;
    const uint64_t rowOff = (static_cast<uint64_t>(row) * static_cast<uint64_t>(kHeadNum_) + n2Idx) *
                            constInfo_.candidateTopkBlocks;
    for (uint32_t tile = 0U; tile < maxTiles; tile++) {
        LocalTensor<K_T> kUb = inQueue_.AllocTensor<K_T>();
        for (uint32_t i = 0U; i < CAND_BLOCKS_PER_AIV; i++) {
            int32_t c = candidateTopkIndexInGm.GetValue(
                rowOff + static_cast<uint64_t>(tile) * CAND_BLOCKS_PER_TILE + halfBlkBase + i);
            if (c < 0) {
                continue;
            }
            uint64_t page = static_cast<uint64_t>(c) / pageBlkNum;
            uint64_t within = static_cast<uint64_t>(c) % pageBlkNum;
            int32_t phys = blockTableGm.GetValue(tableBase + page);
            uint64_t kOff = static_cast<uint64_t>(phys) * blkStride + within * 8ULL * 128ULL;
            uint64_t sOff = static_cast<uint64_t>(phys) * scaleStride + within * 8ULL;
            AscendC::DataCopyPadExtParams<K_T> kPad{false, 0, 0, 0};
            AscendC::DataCopyExtParams kParams;
            kParams.blockCount = 1;
            kParams.blockLen = CAND_BLOCK_BYTES * sizeof(K_T);
            kParams.srcStride = 0;
            kParams.dstStride = 0;
            kParams.rsv = 0;
            AscendC::DataCopyPad(kUb[i * CAND_BLOCK_BYTES], keyGmCand_[kOff], kParams, kPad);
            AscendC::DataCopyPadExtParams<half> sPad{false, 0, 0, 0};
            AscendC::DataCopyExtParams sParams;
            sParams.blockCount = 1;
            sParams.blockLen = 8 * sizeof(half);
            sParams.srcStride = 0;
            sParams.dstStride = 0;
            sParams.rsv = 0;
            // UB 目的地址按 32B 对齐：每块 scale 占 32B（16B 实数据 + 16B 填充）
            AscendC::DataCopyPad(scaleUb[i * 16], kScaleGm[sOff], sParams, sPad);
        }
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        // v4 ring：槽号 = 全局生产序号 % RING_SLOTS；等 AIC 回补一个空槽令牌再覆盖写
        const uint32_t ringSlot = candProdTileSeq_ % QLIV2Common::CAND_RING_SLOTS;
        ++candProdTileSeq_;
        AscendC::CrossCoreWaitFlag(QLIV2Common::CAND_FREE_FLAG);
        const uint64_t slotKeyBase =
            static_cast<uint64_t>(ringSlot) * CAND_TOKENS_PER_TILE * 128ULL +
            static_cast<uint64_t>(aivHalf) * CAND_BLOCKS_PER_AIV * CAND_BLOCK_BYTES;
        AscendC::DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = CAND_BLOCKS_PER_AIV * CAND_BLOCK_BYTES;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        AscendC::DataCopyPad(candStagingKeyGm_[slotKeyBase], kUb, outParams);
        const uint64_t slotScaleBase =
            static_cast<uint64_t>(ringSlot) * (CAND_BLOCKS_PER_TILE * 32ULL) +
            static_cast<uint64_t>(aivHalf) * CAND_BLOCKS_PER_AIV * 32ULL; // 每块 32B（含 16B 填充）
        AscendC::DataCopyExtParams outSParams;
        outSParams.blockCount = 1;
        outSParams.blockLen = CAND_BLOCKS_PER_AIV * 32U;
        outSParams.srcStride = 0;
        outSParams.dstStride = 0;
        outSParams.rsv = 0;
        AscendC::DataCopyPad(candStagingScaleGm_[slotScaleBase], scaleUb, outSParams);
        // v4: kscale 按 tile 落辅助区（本 AIV 半区 16 块 × 16 half = 512B，一次 DMA）；
        // 消费侧按 (token,tile) 一次 1KB 读回，取代逐块 16B DMA + block_table 标量读。
        const uint64_t auxTileBase =
            2ULL * (scoreSlotBase + static_cast<uint64_t>(tokenIdx * maxTiles + tile) *
                                        QLIV2Common::CAND_TILE_TOKENS) +
            static_cast<uint64_t>(aivHalf) * (CAND_BLOCKS_PER_AIV * 16ULL);
        AscendC::DataCopyExtParams auxParams;
        auxParams.blockCount = 1;
        auxParams.blockLen = CAND_BLOCKS_PER_AIV * 32U; // 16 块 × 32B(16B 实数据 + 16B 填充)
        auxParams.srcStride = 0;
        auxParams.dstStride = 0;
        auxParams.rsv = 0;
        AscendC::DataCopyPad(candAuxScaleGm_[auxTileBase], scaleUb, auxParams);
        // 本 AIV 的一半已写完 → 通知 AIC（两个 AIV 的 set 合并成 1 个令牌/tile）
        AscendC::CrossCoreSetFlag<QLIV2Common::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
            QLIV2Common::CAND_READY_FLAG);
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        inQueue_.FreeTensor(kUb);
    }
}

// 单 token 分数消费：掩码 A（列表序号 < L）/ 掩码 B（逻辑 token = c×8 + 块内偏移 < cuRealAcSeq）。
// 无效位置分数置 -inf、索引 -1；有效位置直接写逻辑 token 号（TopK 之后无需再映射）。
// 空行/零候选行走 CleanInvalidOutput（全 -1）。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ConsumeCandScoreToken(
    uint32_t bIdx, uint32_t rowBase, uint32_t s1RowBase, uint32_t n2Idx, uint32_t tokenIdx, uint32_t maxTiles,
    uint32_t actS1Size, uint32_t actS2Size, uint32_t actS2SizeOrig, uint32_t indiceOutOffset,
    uint32_t outputIdxOffsetBase, uint64_t scoreSlotBase)
{
    constexpr uint32_t CAND_TILE_TOKENS = QLIV2Common::CAND_TILE_TOKENS;
    constexpr uint32_t CAND_BLOCKS_PER_TILE = QLIV2Common::CAND_TILE_BLOCKS;
    constexpr uint32_t BLOCK_TOKENS = 8U;
    const int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    const uint32_t aivParity = static_cast<uint32_t>(blockId_ % 2);
    LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
    LocalTensor<float> sortTmp = tmpBuf_.Get<float>()[2 * CAND_TILE_TOKENS];
    LocalTensor<int32_t> idxUb = sortBuff[CAND_TILE_TOKENS].template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> scoreBits = sortBuff.template ReinterpretCast<int32_t>();
    // AIC 每 token set 一次；两个 AIV 各 wait 一次（广播语义）。未分到的 token 也要 wait，
    // 保持计数守恒。maxTiles==0 时 AIC 不产生任何 token（ComputeMm1Candidate 提前返回），
    // 这里也不能 wait，否则死锁。
    if (maxTiles > 0U) {
        AscendC::CrossCoreWaitFlag(QLIV2Common::CAND_SCORE_READY_FLAG);
    }
    if ((tokenIdx % 2U) != aivParity) {
        return;
    }
    const uint32_t row = rowBase + tokenIdx;
    int32_t L = candLenGm_.GetValue(row);
    L = (L < 0) ? 0 : ((L > candBlocks) ? candBlocks : L);
    int32_t V = static_cast<int32_t>(actS2Size);
    if (constInfo_.attenMaskFlag) {
        V = (static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size) +
             static_cast<int32_t>(s1RowBase + tokenIdx) + 1) /
            static_cast<int32_t>(constInfo_.cmpRatio);
    }
    const uint64_t outRow = static_cast<uint64_t>(indiceOutOffset) +
                            static_cast<uint64_t>(tokenIdx) * constInfo_.sparseCount;
    if (L <= 0 || V <= 0 || maxTiles == 0U) {
        CleanInvalidOutput(static_cast<int64_t>(outRow));
        return;
    }
    const uint32_t inner = tokenIdx / 2U;
    InitSortOutBuf(globalTopkUb_[inner * topkPairWidth_], topkPairWidth_);
    PipeBarrier<PIPE_V>();
    const uint64_t candRowOff =
        (static_cast<uint64_t>(row) * static_cast<uint64_t>(kHeadNum_) + n2Idx) *
        constInfo_.candidateTopkBlocks;
    for (uint32_t tile = 0U; tile < maxTiles; tile++) {
        if (static_cast<int32_t>(tile * CAND_BLOCKS_PER_TILE) >= L) {
            break;
        }
        const uint64_t scoreBase =
            scoreSlotBase + (static_cast<uint64_t>(tokenIdx) * maxTiles + tile) * CAND_TILE_TOKENS;
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = CAND_TILE_TOKENS * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        // 上一 tile 的 V 读（SortAll/MergeSort）完成后才能覆盖 sortBuff
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(sortBuff, candScoreGm_[scoreBase], copyParams, padParams);
        // v4: 本 tile 的 kscale 从辅助区一次读回（32 块 × 16 half = 1KB，块内前 8 half 有效）。
        // gather 时两个 AIV 各写本半；消费至少延后一拍（任务内退两拍 / 跨任务队列），
        // 跨 AIV 的写已可见。
        LocalTensor<half> scaleHalf = sortTmp.template ReinterpretCast<half>();
        AscendC::DataCopyPadExtParams<half> sPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams sParams;
        sParams.blockCount = 1;
        sParams.blockLen = QLIV2Common::CAND_AUX_SCALE_HALVES_PER_TILE * sizeof(half); // 1KB
        sParams.srcStride = 0;
        sParams.dstStride = 0;
        sParams.rsv = 0;
        AscendC::DataCopyPad(scaleHalf, candAuxScaleGm_[2ULL * scoreBase], sParams, sPad);
        // 标量读/写 UB 不受 PipeBarrier 围栏：DMA→标量必须显式 MTE2_S（P2 铁律），
        // 且上一 tile 的 V 读也要先完成（V→S）
        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (uint32_t b = 0U; b < CAND_BLOCKS_PER_TILE; b++) {
            const uint32_t ci = tile * CAND_BLOCKS_PER_TILE + b;
            int32_t c = -1;
            if (static_cast<int32_t>(ci) < L) {
                c = candidateTopkIndexInGm.GetValue(candRowOff + ci);
            }
            for (uint32_t j = 0U; j < BLOCK_TOKENS; j++) {
                const uint32_t p = b * BLOCK_TOKENS + j;
                const int32_t tok =
                    (c >= 0) ? (c * static_cast<int32_t>(BLOCK_TOKENS) + static_cast<int32_t>(j)) : -1;
                if (c >= 0 && tok < V) {
                    // key_dequant_scale：与全量扫 ProcessVec1 的 Cast+Mul 等价（fp32 相乘）。
                    // 必须走 sortBuff（float 视图）读写：scoreBits 是 int32 位型视图，
                    // 用 SetValue(int32) 写 float 结果会截断/丢精度。
                    sortBuff.SetValue(p, sortBuff.GetValue(p) * static_cast<float>(scaleHalf.GetValue(b * 16U + j)));
                    idxUb.SetValue(p, tok);
                } else {
                    scoreBits.SetValue(p, QLIV2ServiceVec::NEG_INF);
                    idxUb.SetValue(p, -1);
                }
            }
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        QLIV2ServiceVec::SortAll(sortBuff, sortTmp, CAND_TILE_TOKENS);
        PipeBarrier<PIPE_V>();
        QLIV2ServiceVec::MergeSort(globalTopkUb_[inner * topkPairWidth_], topkWidth_, sortBuff,
                                   CAND_TILE_TOKENS, sortTmp);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
    QLIV2ServiceVec::ExtractIndex(idxULocal,
                                  globalTopkUb_[inner * topkPairWidth_].template ReinterpretCast<uint32_t>(),
                                  topkWidth_);
    PipeBarrier<PIPE_V>();
    if (isOutputIdxOffsetValid_) {
        int32_t rowOff = outputIdxOffsetGm_.GetValue(outputIdxOffsetBase +
                                                     static_cast<uint64_t>(tokenIdx) * kHeadNum_);
        if (rowOff != 0) {
            LocalTensor<int32_t> idxI32 = idxULocal.template ReinterpretCast<int32_t>();
            for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                Adds(idxI32[off], idxI32[off], rowOff, 64);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    outQueue_.EnQue<uint32_t>(idxULocal);
    idxULocal = outQueue_.DeQue<uint32_t>();
    QLIV2ServiceVec::CopyOut(indiceOutGm[outRow], idxULocal.template ReinterpretCast<int32_t>(),
                             constInfo_.sparseCount);
    outQueue_.FreeTensor(idxULocal);
    InitSortOutBuf(globalTopkUb_[inner * topkPairWidth_], topkPairWidth_);
    PipeBarrier<PIPE_V>();
}

template <typename QLIV2T>
__aicore__ inline int32_t QLIV2Vector<QLIV2T>::AlignS2(int32_t cuS2Len){
    // 限制：当前cuS2Len最大为2048，暂不考虑更长
    // 该函数目的是将cuS2Len对齐到形如 32*(4^n)*m 的形式 (m ∈ [1, 3])，方便后续sort/merge
    if (cuS2Len <= ELE_NUM_128) {
        return Align(cuS2Len, ELE_NUM_32);
    } else if (cuS2Len <= ELE_NUM_512) {
        return Align(cuS2Len, ELE_NUM_128);
    } else {
        return Align(cuS2Len, ELE_NUM_512);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec1(const QLIV2Common::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*2048 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (s1BaseSize_ * s2BaseSize_);

    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    // 基本块基地址偏移奇数核加一个S1地址偏移
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * s2BaseSize_;
    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_);
        blockS2StartIdx_ = 0;
        if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
            InitSortOutBuf(globalBlockTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
        }
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    int32_t cuRealAcSeqIni = cuRealAcSeq;

    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        // 当前 vec1ResGm 对应 S1 的位置。
        // LD 布局：每分段一份 s1Base 行预留区（起点 = saveWorkSpaceIdx × s1Base × pairW），
        // 行内槽位用 **gS1 块内相对行号** —— 必须与 ProcessLD 侧 ldInfo_.mStart + j 同一坐标
        // （metadata 的 fdMStart/fdMNum 也是块内相对行）。原实现用 batch 全局行号 cuS1Idx，
        // gS1Idx>0 时与归约侧错位，归约会读到相邻行的部分结果（910B 开 LD 后表现为相邻行交换）。
        int32_t blockRowIdx = cuS1Idx - cuBaseS1Idx;
        uint64_t wsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * topkPairWidth_ +
                            static_cast<uint64_t>(blockRowIdx) * topkPairWidth_;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = AlignS2(cuS2Len);
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = mmInUb[cuS2LenVecAlign];
            LocalTensor<half> kScaleTUb = kScaleUb.template ReinterpretCast<half>()[cuS2LenVecAlign];
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> padTParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = cuS2Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            AscendC::DataCopyPad(mmInUb, mm1ResGm[mmGmOffset + innerS1Idx * s2BaseSize_], copyInParams, padParams);
            GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, cuS2Len);
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, cuS2Len);
            PipeBarrier<PIPE_V>();
            AscendC::Mul(mmInUb, mmInUb, kScaleUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            // mode=2 (use_candidate, R11 leak): 候选块外 score 降级 NEG_HUGE (S4a, 纯算术无 NaN);
            // 仅降级排序资格、不取消入选资格 — 不足 topk 时候选外可达位置作为填充泄漏 (§1.6)
            if (constInfo_.candidateMode == CANDIDATE_MODE_CONSUMER) {
                BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> pen = tmpBuf_.Get<float>()[4096]; // blkIdxF/acc/d 已释放, 复用
                Sub(pen, candNegHuge_, mmInUb, cuS2Len);
                PipeBarrier<PIPE_V>();
                Mul(pen, pen, candIsOut_, cuS2Len);
                PipeBarrier<PIPE_V>();
                Add(mmInUb, mmInUb, pen, cuS2Len);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), QLIV2ServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, mmInUb, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(mmInUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            // R11 (leak 语义, §1.6): 候选块外索引不再置 -1 —
            // 分数已在 S4a 降级 NEG_HUGE (仍高于 -inf: 不足 topk 时作为填充入选, 排候选内之后),
            // 索引保留真实位置号, 与模型 where(idxs < compress_lens, idxs + offset, -1) 等价:
            // 仅不可达位置 (score -inf 沉底 + 尾部 -1 填充) 输出 -1。
            // mode=1 (is_candidate_source): 块化 amax + pin + 块级排序归并 (S5a/S6a)
            if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                ProcessCandBlockTopk(info, cuS1Idx, cuS2Len, cuS2LenVecAlign, cuRealAcSeq, innerS1Idx);
            }
            LocalTensor<float> tmpSortBuf = sortBuff[2 * cuS2LenVecAlign];
            QLIV2ServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            QLIV2ServiceVec::MergeSort(globalTopkUb_[innerS1Idx * topkPairWidth_], topkWidth_, sortBuff,
                                       cuS2LenVecAlign, tmpSortBuf); // P3
            PipeBarrier<PIPE_V>();
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            // 中间结果保存
            if (needCopyOutGm && !info.isNeedLD && !info.suppressOut) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                // P3: 只抽取前 topkWidth_ 对 (降序排序下 2048 名之后永不入选, 截断语义等价)
                ExtractIndex(idxULocal,
                             globalTopkUb_[innerS1Idx * topkPairWidth_].template ReinterpretCast<uint32_t>(),
                             topkWidth_);
                PipeBarrier<PIPE_V>();
                // A15: output_idx_offset — 输出拷 GM 前逐元素加行偏移 (对齐 arch35 IndicesAddOffset;
                // 零偏移零开销; GM 标量读无 V_S 竞态; int32 整数域 Adds 位精确, -1 槽 +0 不变。
                // candidate_topk_index 不加 (相对块号契约, §11.6))
                if (isOutputIdxOffsetValid_) {
                    int32_t rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset +
                                                                 cuS1Idx * kHeadNum_);
                    if (rowOff != 0) {
                        LocalTensor<int32_t> idxI32 = idxULocal.template ReinterpretCast<int32_t>();
                        for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                            Adds(idxI32[off], idxI32[off], rowOff, 64);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                }
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                QLIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                         idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
                // mode=1: 块级 topk 直出 candidate_topk_index
                if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                    CopyOutCandTopkIndex(info, cuS1Idx, innerS1Idx);
                }
            }
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                AscendC::DataCopyExtParams copyWsParams;
                copyWsParams.blockLen = topkPairWidth_ * sizeof(float); // P3: 按实际 topk 宽度
                copyWsParams.srcStride = 0;
                copyWsParams.dstStride = 0;
                copyWsParams.blockCount = 1;
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                AscendC::DataCopyPad(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                     copyWsParams);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                PipeBarrier<PIPE_V>();
                // mode=1：块级累积器也必须同样溢写。原实现只在非 LD 分支里直出
                // candidate_topk_index，跨核切分（LD）后块级累积器没有溢写也没有归约，
                // 归约侧拿到的是空/旧数据，块级输出整行错误（实测 0/24）。
                if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                    SpillBlockTopkToGm(info, blockRowIdx, innerS1Idx);
                }
            }
        } else if (cuRealAcSeq <= 0) {
            // 无效长度处理
            if (info.isNeedLD && info.isLastS2InnerLoop) {
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                         topkPairWidth_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                    SpillBlockTopkToGm(info, blockRowIdx, innerS1Idx); // 与位置级成对溢写
                }
            } else if (!info.suppressOut) {
                CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
            }
        } else if (cuS2Len <= 0) {
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QLIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                         topkPairWidth_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                PipeBarrier<PIPE_V>();
                if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
                    SpillBlockTopkToGm(info, blockRowIdx, innerS1Idx); // 与位置级成对溢写
                }
            }
        }
    }

    // BNSD场景无效S1 输出-1
    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2SizeOrig;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

// mode=1 + LD：块级 topk 的分段归并。
// 与位置级（ProcessLD）的 MrgSort 链完全同构，只有三个口径不同：
//   ① 元素宽度：每段 candidateTopkBlocks 对（value+index），即 2×candidateTopkBlocks 个 float；
//   ② 槽位步长：块级 GM 区每行 slot = 2×candidateTopkBlocks 个 float；
//   ③ 输出宽度：candidateTopkIndexOutGm 每行 candidateTopkBlocks 个块号。
// 归并缓冲与位置级**共用**（InitLDBuffers 已按两者较大宽度分配），故必须在位置级
// Extract/CopyOut 之后再执行（本函数在 ProcessLD 每行的位置级输出之后调用）。
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ReduceBlockTopkRow(uint32_t rowInBlock)
{
    uint32_t candBlocks = constInfo_.candidateTopkBlocks;
    uint32_t candPairWidth = 2U * candBlocks;
    LocalTensor<float> curCand = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpCand = ldTmpBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxCand = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyPadExtParams<float> candPadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams candParams;
    candParams.blockLen = candPairWidth * sizeof(float);
    candParams.srcStride = 0;
    candParams.dstStride = 0;
    candParams.blockCount = 1;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4 段归约任务做一次 MrgSort
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾段数

    // 本行在 GM 中的起点：与写侧 SpillBlockTopkToGm 同一公式
    // （workspaceIdx × s1Base × pairW + 段内相对行号 × pairW）
    uint64_t rowBase = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * candPairWidth +
                       static_cast<uint64_t>(rowInBlock) * candPairWidth;
    uint64_t segGap = static_cast<uint64_t>(s1BaseSize_) * candPairWidth; // 相邻 S2 段起点间隔

    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    AscendC::DataCopyPad(curCand, blockVec1ResGm_[rowBase], candParams, candPadParams);
    uint32_t nextOffset = candPairWidth;
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    AscendC::MrgSort4Info prm;
    prm.elementLengths[0] = candBlocks;
    prm.elementLengths[1] = candBlocks;
    prm.elementLengths[2] = candBlocks;
    prm.elementLengths[3] = candBlocks;
    prm.ifExhaustedSuspension = true;
    prm.validBit = 0b1111;
    prm.repeatTimes = 1;

    for (uint32_t i = 0; i < ldProcessNum; i++) {
        uint64_t segBase = rowBase + static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * segGap;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        for (uint32_t k = 0; k < ldProcessLen - 1U; k++) {
            AscendC::DataCopyPad(curCand[nextOffset + k * candPairWidth], blockVec1ResGm_[segBase + k * segGap],
                                 candParams, candPadParams);
        }
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = curCand[0];
        srcList.src2 = curCand[candPairWidth];
        srcList.src3 = curCand[2 * candPairWidth];
        srcList.src4 = curCand[3 * candPairWidth];
        MrgSort(tmpCand, srcList, prm);
        PipeBarrier<PIPE_V>();
        DataCopy(curCand, tmpCand, candPairWidth);
        PipeBarrier<PIPE_V>();
    }

    if (ldTailLen != 0) {
        uint64_t tailBase = rowBase + static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * segGap;
        AscendC::DataCopyExtParams tailParams = candParams;
        tailParams.blockCount = ldTailLen;
        // srcStride 单位是**字节**，填"段间空隙"（段起点间隔 − 一段长度），与位置级同一修法
        tailParams.srcStride = (s1BaseSize_ * candPairWidth - candPairWidth) * sizeof(float);
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(curCand[nextOffset], blockVec1ResGm_[tailBase], tailParams, candPadParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        AscendC::MrgSort4Info tailPrm;
        tailPrm.elementLengths[0] = candBlocks;
        tailPrm.elementLengths[1] = candBlocks;
        tailPrm.elementLengths[2] = candBlocks;
        tailPrm.elementLengths[3] = candBlocks;
        tailPrm.ifExhaustedSuspension = true;
        tailPrm.validBit = (ldTailLen == 1) ? 0b0011 : ((ldTailLen == 2) ? 0b0111 : 0b1111);
        tailPrm.repeatTimes = 1;
        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = curCand[0];
        srcList.src2 = curCand[candPairWidth];
        srcList.src3 = curCand[2 * candPairWidth];
        srcList.src4 = curCand[3 * candPairWidth];
        MrgSort(tmpCand, srcList, tailPrm);
        PipeBarrier<PIPE_V>();
        DataCopy(curCand, tmpCand, candPairWidth);
        PipeBarrier<PIPE_V>();
    }

    // 抽出块号并写到 candidate_topk_index（行偏移 = 行内相对行号 × kHeadNum × candBlocks）
    ExtractIndex(outIdxCand, curCand.template ReinterpretCast<uint32_t>(), candBlocks);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    uint64_t outOffset = ldInfo_.candidateOutCoreOffset +
                         static_cast<uint64_t>(rowInBlock) * constInfo_.kHeadNum * candBlocks;
    QLIV2ServiceVec::CopyOut(candidateTopkIndexOutGm[outOffset], outIdxCand.template ReinterpretCast<int32_t>(),
                             static_cast<int64_t>(candBlocks));
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessLD()
{
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<float> indexValuePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams indexValueParams;
    // P3: 每列表搬运/间距按实际 topk 对宽 (vec1ResGm 内相邻列表间距 = 3 对宽)
    indexValueParams.blockLen = topkPairWidth_ * sizeof(int32_t); // bytes
    // P10: 分段槽按核分布 (核 i 的区起点 = i*s1Base 槽)。
    // P11 更正：`DataCopyExtParams.srcStride` 的单位是**字节**、语义是"块与块之间的空隙"
    // （不是起点间距、更不是 32B 块数）—— 原写法 `s1Base*pairW*4/32` 只跳 512B 而不是一个槽位，
    // 段与段几乎重叠，正是 507015 的根因。归并（3 段）走的是逐段拷贝、blockCount=1 时该字段被忽略；
    // 只有尾段（blockCount=ldTailLen）真正用它，故在尾段分支里按 (s1Base×pairW − pairW)×4B 赋值。
    indexValueParams.srcStride = 0;
    indexValueParams.dstStride = 0;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4块归约任务做一次Merge
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾块长度

    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        // 拷贝第一块归约任务
        indexValueParams.blockCount = 1;
        uint64_t wsOffsetIni = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                               static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
        AscendC::DataCopyPad(curValueIdxUb, vec1ResGm[wsOffsetIni], indexValueParams, indexValuePadParams);
        uint64_t valueOffset = topkPairWidth_;
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 处理等于4的部分
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // LD处理偏移
            uint64_t wsOffset =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
            indexValueParams.blockCount = 1; // P10: 逐段拷贝 (见上)

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            // P10: 绕开 srcStride 语义不确定性 — 3 个分段槽逐个显式拷贝
            for (uint32_t k = 0; k < ldProcessLen - 1U; k++) {
                uint64_t segSlot = wsOffset + static_cast<uint64_t>(k) * s1BaseSize_ * topkPairWidth_;
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * topkPairWidth_],
                                     vec1ResGm[segSlot], indexValueParams, indexValuePadParams);
            }
            // merge参数 (P3: elementLengths 按实际 topk 宽度)
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            params.validBit = 0b1111;
            params.repeatTimes = 1;

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[topkPairWidth_];
            srcList.src3 = curValueIdxUb[2 * topkPairWidth_];
            srcList.src4 = curValueIdxUb[3 * topkPairWidth_];
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, topkPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 处理不等于4的部分
        if (ldTailLen != 0) {
            // 搬运尾块
            uint64_t wsOffsetTail =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
            // 尾段一次性跨段搬运（blockCount = ldTailLen）。原实现在这里崩 507015 的原因是
            // srcStride 单位写错：DataCopyExtParams.srcStride 的单位是**字节**，而原代码按"32B 块"
            // 除了 32（s1Base*pairW*4/32 = 512），实际只跳 512B 而不是一个槽位 16KB，
            // 段与段几乎重叠；该路径只在"一行切成 ≥3 段"（ldTailLen≥2）时才走到，故 2 段用例从未暴露。
            indexValueParams.blockCount = ldTailLen;
            indexValueParams.srcStride = (s1BaseSize_ * topkPairWidth_ - topkPairWidth_) * sizeof(int32_t);  // 字节：段间"空隙"
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            AscendC::DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffsetTail], indexValueParams,
                                 indexValuePadParams);
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = 0b0011;
            } else if (ldTailLen == 2) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[topkPairWidth_];
            srcList.src3 = curValueIdxUb[2 * topkPairWidth_];
            srcList.src4 = curValueIdxUb[3 * topkPairWidth_];
            PipeBarrier<PIPE_V>();
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, topkPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 搬出 (P3: Extract repeat = topkWidth_/32, 每 repeat 消费 32 对)
        Extract(outValueUb, outIdxUb, curValueIdxUb, (topkWidth_ / 32));
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(curValueIdxUb, topkPairWidth_);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // A15: LD(decode) 路径同样加 output_idx_offset (行前缀 = indiceOutCoreOffset/sparseCount, 即
        // batch 前缀 x kHeadNum + n2; 与 ProcessVec1 消费点同语义)
        if (isOutputIdxOffsetValid_) {
            uint32_t rowGlobal = ldInfo_.mStart + j;
            int32_t rowOff = outputIdxOffsetGm_.GetValue(
                ldInfo_.indiceOutCoreOffset / constInfo_.sparseCount + rowGlobal * constInfo_.kHeadNum);
            if (rowOff != 0) {
                for (int32_t off = 0; off < constInfo_.sparseCount; off += 64) {
                    Adds(idxULocal1[off], idxULocal1[off], rowOff, 64);
                }
                PipeBarrier<PIPE_V>();
            }
        }
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * constInfo_.sparseCount;
        AscendC::DataCopyPad(indiceOutGm[outOffset], idxULocal1, copyOutParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        // mode=1：同一行再做一遍块级归约，输出 candidate_topk_index（块级输出无 LD 时由
        // CopyOutCandTopkIndex 直出；有 LD 时每段只有部分结果，必须在这里归并）
        if (constInfo_.candidateMode == CANDIDATE_MODE_SOURCE) {
            ReduceBlockTopkRow(ldInfo_.mStart + j);
        }
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
