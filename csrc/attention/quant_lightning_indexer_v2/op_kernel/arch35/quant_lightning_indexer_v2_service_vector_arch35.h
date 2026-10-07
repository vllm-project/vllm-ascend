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
 * \file quant_lightning_indexer_v2_service_vector_arch35.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch35.h"
#include "../arch35/vf/quant_lightning_indexer_v2_vector1.h"
#include "../arch35/vf/quant_lightning_indexer_v2_topk.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
constexpr uint32_t TRUNK_LEN_16K = 16384;
// LD阶段UB目标占用（248KB = 256KB UB - 8KB 系统保留）
constexpr uint32_t LD_UB_TARGET_BYTES = 248 * 1024;
constexpr uint32_t TRUNK_LEN_12K = 12288;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TRUNK_LEN_4K = 4096;
constexpr uint32_t TRUNK_LEN_2K = 2048;
constexpr uint32_t TOPK_2K = 2048;
constexpr uint32_t TOPK_3K = 3072;
constexpr uint32_t TOPK_4K = 4096;
constexpr uint32_t TOPK_5K = 5120;
constexpr uint32_t TOPK_6K = 6144;
constexpr uint32_t DUPSIZE = 256; // duplicate所需额外地址空间

template <typename QLIV2T>
class QLIV2Vector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIV2T::pageAttention;

    using QK_T = typename QLIV2T::queryKeyType;
    using SCORE_T = typename QLIV2T::scoreType;
    using WEIGHT_T = typename QLIV2T::weightType;
    static constexpr bool IS_MX = QLIV2T::isMx;
    static constexpr bool IS_MXFP4 = QLIV2T::isMxFp4;
    using SCALE_T = typename QLIV2T::scaleType;
    static constexpr bool IS_WEIGHT_FP16 = QLIV2T::isWeightFP16;
    __aicore__ inline QLIV2Vector(){};
    __aicore__ inline void ProcessVec1(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLIV2Common::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLIV2Common::ConstInfo &constInfo,
                                      const struct QLIV2Common::LdSplitCoreInfo &ldInfo,
                                      const QLIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm, GlobalTensor<SCORE_T> ldScoreGm,
                                                  GlobalTensor<int32_t> ldIndexGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<WEIGHT_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm, GlobalTensor<bfloat16_t> valueOutGm,
                                              GlobalTensor<int32_t> outputIdxOffsetGm,
                                              GlobalTensor<SCALE_T> qScaleGm = {}, GlobalTensor<SCALE_T> kScaleGm = {});
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe, const struct QLIV2Common::LdSplitCoreInfo &ldInfo);
    __aicore__ inline void DoTndPadding(const QLIV2Common::RunInfo &runInfo);

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<WEIGHT_T> weightsGm;
    GlobalTensor<SCORE_T> ldScoreGm;
    GlobalTensor<int32_t> ldIndexGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<bfloat16_t> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;
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
    static constexpr uint32_t MTE3_V_EVENT = EVENT_ID6;

    static constexpr uint64_t BLOCK_CUBE = 16;

private:
    __aicore__ inline void GetKeyScale(const QLIV2Common::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<WEIGHT_T> weightUB_;
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightTempBuf_;
    LocalTensor<float> weightTempUB_;
    // tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<SCALE_T> kScaleUB_;
    // tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<SCALE_T> qScaleUB_;
    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<bfloat16_t> valueOutLocal_;
    // tmp buff for LD
    TBuf<TPosition::VECCALC> ldValueBuf_;
    LocalTensor<uint32_t> ldValueLocal_; // SCORE_T
    TBuf<TPosition::VECCALC> ldIndexBuf_;
    LocalTensor<uint32_t> ldIndexLocal_;

    TBuf<TPosition::VECCALC> topkIndexBuf_;
    LocalTensor<uint32_t> topkIndexLocal_;

    TBuf<TPosition::VECCALC> topkValueBuf_;
    LocalTensor<uint32_t> topkValueLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

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
    uint32_t topkCountAlign16_ = 0;  // LD读取到UB，需要满足32B对齐
    float globalQScale_ = 1.0f;      // quantMode=4时全局query scale
    float globalKScale_ = 1.0f;      // quantMode=4时全局key scale
    uint32_t trunkLen_ = 0;          // ProcessTopK（非LD路径）每次处理的s2长度
    uint32_t trunkLenLd_ = 0;        // ProcessLD路径每次处理的s2长度，根据sparseCount动态计算以填满UB
    bool returnValueFlag = false;

    struct QLIV2Common::ConstInfo constInfo_;
    struct QLIV2Common::LdSplitCoreInfo ldInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitBuffers(TPipe *pipe)
{
    // 大小：2(开dB) 2 * 64 * 128 * 4 = 128KB
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSizeMax, 2) * s2BaseSize_ * sizeof(QK_T));
    resMm1UB_ = resMm1Buf_.Get<QK_T>(); // qk
    // weight buffer按WEIGHT_T访问；UB_BANK_DEPTH_STRIDE的单位已经是字节，无需再乘数据类型大小。
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<WEIGHT_T>(); // weight
    pipe->InitBuffer(weightTempBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightTempUB_ = weightTempBuf_.Get<float>();
    // 大小：2(开dB) * 128 * 4 = 1KB
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    kScaleUB_ = kScaleBuf_.Get<SCALE_T>(); // kScale
    // 大小：2(开dB) * 2 * 64 * 4 = 1KB
    pipe->InitBuffer(qScaleBuf_,
                     2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<SCALE_T>(); // qScale
    // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>(); // out
    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    valueOutLocal_ = mrgValueBuf_.Get<bfloat16_t>();

    pipe->InitBuffer(indicesOutBuf_, topkCountAlign256_ * sizeof(uint32_t) + DUPSIZE);
    // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T) + DUPSIZE);
    // (topkCountAlign256_ + 64) * sizeof(SCORE_T) 64:duplicate刷-1额外空间
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);

    // MX场景不走vector侧kScale路径
    if constexpr (!IS_MX) {
        Duplicate(kScaleUB_, (SCALE_T)(0), 2 * s2BaseSize_ * 16);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitLDBuffers(TPipe *pipe,
                                                          const struct QLIV2Common::LdSplitCoreInfo &ldInfo)
{
    pipe->Reset();

    // 根据sparseCount动态计算LD的trunk长度，使LD阶段UB占用尽量接近248KB且不超限
    //
    // UB总占用 = Σ各buffer，按 A=topkCountAlign256_、L=trunkLenLd_ 归并后：
    //   Total = 20·A + 8·L + fixedBytes
    //   A系数20 = ldIndex(4)+indicesOut(4)+mrgValue(2)+scoreOut(2)+valueOut(2)+topkTmp(6)
    //   L系数8  = ldIndex(4)+mrgValue(2)+topkTmp(2)
    //   fixedBytes = topk内部固定区 + 三输出buffer的+64余量(见下方命名常量)

    // topk内部固定区: histograms(256)+idxHigh(256)+idxLow(256)+nkValue(64)，均为uint32
    constexpr uint32_t TOPK_FIXED_BYTES = (3 * 256 + 64) * sizeof(uint32_t);
    // 三输出buffer的+64余量(用于duplicate刷-1/刷0): indicesOut(uint32)+scoreOut(SCORE_T)+valueOut(bf16)
    constexpr uint32_t OUTPUT_PAD_BYTES = 64 * (sizeof(uint32_t) + sizeof(SCORE_T) + sizeof(bfloat16_t));
    const uint32_t fixedBytes = TOPK_FIXED_BYTES + OUTPUT_PAD_BYTES;

    uint32_t availForTrunk = (LD_UB_TARGET_BYTES > 20 * topkCountAlign256_ + fixedBytes) ?
                                 (LD_UB_TARGET_BYTES - 20 * topkCountAlign256_ - fixedBytes) :
                                 0;
    trunkLenLd_ = availForTrunk / 8;
    // trunkLenLd_向下对齐到256，确保LD阶段UB不超限且满足topk直方图对齐要求
    trunkLenLd_ = (trunkLenLd_ / 256) * 256;

    // 更新topkOp_的trunkLen，使GetSharedTmpBufferSize返回正确尺寸
    topkOp_.Init(topkCount_, trunkLenLd_);

    pipe->InitBuffer(ldIndexBuf_, (topkCountAlign256_ + trunkLenLd_) * sizeof(uint32_t));
    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLenLd_) * sizeof(SCORE_T));
    pipe->InitBuffer(scoreOutBuf_, (topkCountAlign256_ + 64) * sizeof(SCORE_T));
    pipe->InitBuffer(valueOutBuf_, (topkCountAlign256_ + 64) * sizeof(bfloat16_t));
    // 补齐topk共享tmp buffer分配：pipe->Reset()后原topkSharedTmpBuf_已释放，
    // 需重新分配并重新绑定topkOp_内部指针，避免悬垂指针
    // Re-allocate topk shared tmp buffer: pipe->Reset() freed the original
    // topkSharedTmpBuf_; re-allocate and rebind topkOp_ internal pointers
    pipe->InitBuffer(topkSharedTmpBuf_, topkOp_.GetSharedTmpBufferSize());

    ldIndexLocal_ = ldIndexBuf_.Get<uint32_t>();
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();
    valueOutLocal_ = valueOutBuf_.Get<bfloat16_t>();
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);
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
    s2BaseSize_ = constInfo.s2BaseSize; // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    returnValueFlag = constInfo.returnValue;
    blockId_ = GetBlockIdx();
    trunkLen_ = constInfo.sparseCount <= TOPK_2K ? TRUNK_LEN_16K :
                constInfo.sparseCount <= TOPK_3K ? TRUNK_LEN_12K :
                constInfo.sparseCount <= TOPK_4K ? TRUNK_LEN_8K :
                constInfo.sparseCount <= TOPK_5K ? TRUNK_LEN_4K :
                constInfo.sparseCount <= TOPK_6K ? TRUNK_LEN_2K :
                                                   TRUNK_LEN_12K;
    topkCount_ = constInfo.sparseCount;
    topkCountAlign256_ = QLIV2Common::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
    topkCountAlign16_ = QLIV2Common::Align(constInfo.sparseCount, (uint64_t)16);   // topkCount对齐到16
    topkOp_.Init(topkCount_, trunkLen_);

    if constexpr (!IS_MX) {
        if (constInfo_.quantMode == 4) {
            globalQScale_ = qScaleGm.GetValue(0);
            globalKScale_ = kScaleGm.GetValue(0);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecInputTensor(
    GlobalTensor<WEIGHT_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<int32_t> blockTableGm,
    GlobalTensor<bfloat16_t> valueOutGm, GlobalTensor<int32_t> outputIdxOffsetGm, GlobalTensor<SCALE_T> qScaleGm,
    GlobalTensor<SCALE_T> kScaleGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
    this->valueOutGm = valueOutGm;
    this->outputIdxOffsetGm = outputIdxOffsetGm;
    if constexpr (!IS_MX) {
        this->qScaleGm = qScaleGm;
        this->kScaleGm = kScaleGm;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                                   GlobalTensor<SCORE_T> ldScoreGm,
                                                                   GlobalTensor<int32_t> ldIndexGm)
{
    this->scoreGm = scoreGm; // resucesum*k
    this->ldScoreGm = ldScoreGm;
    this->ldIndexGm = ldIndexGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::AllocEventID()
{
    if constexpr (!IS_MX) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    }
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::FreeEventID()
{
    if constexpr (!IS_MX) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    uint64_t dealSize = constInfo_.sparseCount;
    GlobalTensor<int32_t> indexOutput = indiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(indexOutput, dealSize, constInfo_.INVALID_IDX);

    if (returnValueFlag) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), constInfo_.NEG_INF_BFLOAT,
                  constInfo_.sparseCount);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = constInfo_.sparseCount * sizeof(bfloat16_t);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(valueOutGm[invalidS1Offset], valueOutLocal_, copyOutValueParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::DoTndPadding(const QLIV2Common::RunInfo &runInfo)
{
    uint32_t paddingLen = runInfo.curCuSeqlensQ - runInfo.curSequsedQ;
    uint64_t paddingOffset =
        runInfo.indiceOutOffset + runInfo.curSequsedQ * constInfo_.kHeadNum * constInfo_.sparseCount;
    uint64_t dealSize = paddingLen * constInfo_.kHeadNum * constInfo_.sparseCount;
    GlobalTensor<int32_t> indiceOutPaddingStart = indiceOutGm[paddingOffset];
    AscendC::InitGlobalMemory(indiceOutPaddingStart, dealSize, constInfo_.INVALID_IDX);

    if (constInfo_.returnValue) {
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);

        GlobalTensor<uint16_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
        GlobalTensor<uint16_t> valueOut = valueOutGmTmp[paddingOffset];

        AscendC::InitGlobalMemory(valueOut, dealSize, constInfo_.NEG_INF_BFLOAT);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    }
}
template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::GetKeyScale(const QLIV2Common::RunInfo &runInfo,
                                                        LocalTensor<SCALE_T> &kScaleUB,
                                                        int64_t batchId, int64_t startS2,
                                                        int64_t getLen)
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
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + startBlockTableOffset],
                                 copyInParams, padParams);
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
            AscendC::DataCopyPad(
                kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ + resUbBaseOffset + i * kCacheBlockSize_],
                kScaleGm[blockId * constInfo_.keyDequantScaleStride0], copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(SCALE_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                             kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessVec1(const QLIV2Common::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto qScalepingpong = (info.qScaleLoop % 2);
    auto kScalepingpong = (info.kScaleLoop % 2);
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_V>(
            QLIV2Common::ConstInfo::CROSS_CV_EVENT + pingpong); // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_V>(
            QLIV2Common::ConstInfo::CROSS_VC_EVENT + pingpong); // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }

    if (info.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
        // weightsGm --> weightUB_
        int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
        DataCopyPadExtParams<WEIGHT_T> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams qwDataCopyExtParams;
        qwDataCopyExtParams.blockCount = curAivS1ProcNum;
        qwDataCopyExtParams.blockLen = gSize_ * sizeof(WEIGHT_T);
        qwDataCopyExtParams.srcStride = 0;
        qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qwDataCopyExtParams.blockLen) / 32;
        DataCopyPad(weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(WEIGHT_T))], weightsGm[weightGmOffset],
                    qwDataCopyExtParams, padWeightsParams);

        if constexpr (!IS_MX) {
            if (constInfo_.quantMode != 4) {
                // qScaleGm  -->  qScaleUB_
                DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
                DataCopyExtParams qScaleDataCopyExtParams;
                qScaleDataCopyExtParams.blockCount = curAivS1ProcNum;
                qScaleDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
                qScaleDataCopyExtParams.srcStride = 0;
                qScaleDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qScaleDataCopyExtParams.blockLen) / 32;
                DataCopyPad(qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))], qScaleGm[weightGmOffset],
                            qScaleDataCopyExtParams, padQScaleParams);
            }
        }

        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
    }

    if constexpr (!IS_MX) {
        if (((info.s2Idx - info.s2Start) % 16 == 0) && (constInfo_.quantMode != 4)) {
            WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
            uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_) ?
                                  info.validS2Len - info.s2Idx * s2BaseSize_ :
                                  16 * s2BaseSize_;
            // kScaleGm  -->  kScaleUB_
            GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
            SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
            WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        }
    }

    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    // CV同步
    CrossCoreWaitFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_V>(
        QLIV2Common::ConstInfo::CROSS_CV_EVENT + info.loop % 2); // V核等C核计算完mm1，mm1Res已搬运到UB

    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    auto outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(SCORE_T))];
    auto weightBase = weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(WEIGHT_T))];
    auto weightTempBase = weightTempUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];

    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSizeMax;
    if constexpr (IS_MXFP4) {
        vector1::BatchMulWeightAndReduceSumMXFP4(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T), qkBase, qkVLstride,
                                                 (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)), weightBase,
                                                 UB_BANK_DEPTH_STRIDE / sizeof(WEIGHT_T), gSize_, curAivS1ProcNum);
    } else if constexpr (IS_MX) {
        vector1::BatchMulWeightAndReduceSumMX(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T), qkBase, qkVLstride,
                                              (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)), weightBase,
                                              UB_BANK_DEPTH_STRIDE / sizeof(WEIGHT_T), weightTempBase, gSize_,
                                              curAivS1ProcNum);
    } else if constexpr (IS_WEIGHT_FP16) {
        auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(WEIGHT_T))];
        auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ +
                                    ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
        vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                            qkBase, qkVLstride,
                                            (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)),
                                            weightBase, UB_BANK_DEPTH_STRIDE / sizeof(WEIGHT_T), weightTempBase,
                                            kScaleBase, (uint32_t)0,
                                            qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T),
                                            gSize_, curAivS1ProcNum);
    } else if (constInfo_.quantMode == 4) { // 4: per_tensor量化
        // quantMode为4时不适用sacle的UB
        float kScaleValue = globalKScale_;
        float qScaleValue = globalQScale_;
        vector1::BatchMulWeightAndReduceSumPerTensor(
            outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T), qkBase, qkVLstride,
            (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)), weightBase,
            UB_BANK_DEPTH_STRIDE / sizeof(WEIGHT_T), weightTempBase, kScaleValue, qScaleValue, gSize_, curAivS1ProcNum);
    } else if (constInfo_.quantMode == 1) {
        auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
        auto kScaleBase =
            kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
        vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T), qkBase, qkVLstride,
                                            (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)), weightBase,
                                            UB_BANK_DEPTH_STRIDE / sizeof(WEIGHT_T), weightTempBase, kScaleBase,
                                            (uint32_t)0, qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float), gSize_,
                                            curAivS1ProcNum);
    }
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }
    if constexpr (!IS_MX) {
        if (((info.s2Idx - info.s2Start) % 16 == 0) && (constInfo_.quantMode != 4)) {
            SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        }
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    int64_t vec1OutGmOffset =
        blockId_ % 2 == 0 ?
            curS2Idx :
            s1BaseSizePerAIV * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - copyOutParams.blockLen) / 32;
    copyOutParams.dstStride =
        (QLIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLIV2Common::ConstInfo::QLIV2_SYNC_MODE4, PIPE_V>(
        QLIV2Common::ConstInfo::CROSS_VC_EVENT + pingpong); // V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessTopK(const QLIV2Common::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2StartIdx = info.s2Start * s2BaseSize_;
    int64_t curS2EndIdx = (info.s2LoopEnd + 1) * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    // LD 需要搬运到的起始位置 32字节
    int64_t ldDstOffset = (info.isNeedLD) ? curS2StartIdx : 0;
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

    AscendC::DataCopyParams ldCopyOutParams; // ldIndicesCopyOutParams
    ldCopyOutParams.blockCount = 1;
    ldCopyOutParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
    ldCopyOutParams.srcStride = 0;
    ldCopyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyScoreOutParams;
    ldCopyScoreOutParams.blockCount = 1;
    ldCopyScoreOutParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
    ldCopyScoreOutParams.srcStride = 0;
    ldCopyScoreOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validAllS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        if (i > 0) {
            SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
            WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
        }
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;
        int64_t outputIdxOffset = 0;
        if (info.isOutputIdxOffsetValid) {
            outputIdxOffset = outputIdxOffsetGm.GetValue(info.outputIdxCoreOffset + rowIdx * kHeadNum_);
        }

        SCORE_T zero = 0;
        int32_t neg = -1;
        uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
        if (constInfo_.attenMaskFlag) {
            validAllS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t validS2Len = validAllS2Len;
        if (info.isNeedLD) {
            // 当前核处理的s2长度validS2Len
            validS2Len = Min((info.s2LoopEnd + 1) * s2BaseSize_, validAllS2Len) - curS2StartIdx;
        }

        uint64_t offset = info.saveWorkSpaceIdx * s1BaseSize_ * topkCountAlign16_ + rowIdx * topkCountAlign16_;
        if (validS2Len <= 0 && !info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValueFlag) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>(), constInfo_.NEG_INF_BFLOAT, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(bfloat16_t);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                     copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        } else if (validS2Len <= 0 && info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCountAlign16_);
            Duplicate(scoreOutLocal_, zero, topkCountAlign16_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[offset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[offset], indicesOutLocal_.ReinterpretCast<int32_t>(), ldCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            bool useSingleLoop =
                (s2LoopNum == 1) || ((topkCount_ > trunkLen_) && (validS2Len <= (uint32_t)topkCountAlign256_));
            if (useSingleLoop) {
                uint32_t validS2LenAlign = QLIV2Common::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                            ldDstOffset],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, info.isNeedLD,
                             returnValueFlag, outputIdxOffset);
            } else {
                uint32_t outputIdxOffsetTmp = 0;
                uint32_t actS2LoopNum = 0;
                if (topkCount_ > trunkLen_) {
                    actS2LoopNum = 1 + (validS2Len - topkCountAlign256_ + trunkLen_ - 1) / trunkLen_;
                } else {
                    actS2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
                }
                for (uint32_t loopIdx = 0; loopIdx < actS2LoopNum; loopIdx++) {
                    if (loopIdx == actS2LoopNum - 1) {
                        outputIdxOffsetTmp = outputIdxOffset;
                    }
                    if (loopIdx == 0) {
                        if (topkCount_ > trunkLen_) {
                            copyInParams.blockLen = topkCountAlign256_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(scoreOutLocal_,
                                                 scoreGm[vecOffset * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize,
                                                                                        (uint64_t)s2BaseSize_) +
                                                         ldDstOffset],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                            AscendC::CreateVecIndex(topkSharedTmpLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                        } else {
                            copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(mrgValueLocal_,
                                                 scoreGm[vecOffset * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize,
                                                                                        (uint64_t)s2BaseSize_) +
                                                         ldDstOffset],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx,
                                         actS2LoopNum, info.isNeedLD, returnValueFlag, outputIdxOffsetTmp);
                        }

                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = 0;
                    uint32_t offset = 0;
                    if (topkCount_ > trunkLen_) {
                        validTrunkLen = (topkCountAlign256_ + (loopIdx - 1) * trunkLen_ + trunkLen_) > validS2Len ?
                                            (validS2Len - topkCountAlign256_) % trunkLen_ :
                                            trunkLen_;
                        offset = vecOffset * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 topkCountAlign256_ + (loopIdx - 1) * trunkLen_ + ldDstOffset;
                    } else {
                        validTrunkLen =
                            (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                        offset = vecOffset * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 loopIdx * trunkLen_ + ldDstOffset;
                    }

                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    // 如果topk > trunkLen，第一轮调用topk是直接拷贝的，不需要刷0
                    bool isZeroPadding = (topkCount_ > trunkLen_) ? (loopIdx > 1) : true;
                    if (topkCountAlign256_ != topkCount_ && isZeroPadding) {
                        uint64_t mask[2] = {0, 0}; // 0: mask初始化，代表对应位置不需要刷0
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
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero,
                                  QLIV2Common::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_.TopK(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                                 QLIV2Common::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx,
                                 actS2LoopNum, info.isNeedLD, returnValueFlag, outputIdxOffsetTmp);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            if (outputIdxOffset != 0) {
                topkb16gather::IndicesAddOffset(indicesOutLocal_, outputIdxOffset, constInfo_.sparseCount);
            }
            // LD或returnValue场景需要将score搬入scoreOutLocal_
            if (info.isNeedLD || returnValueFlag) {
                copyInParams.blockLen = QLIV2Common::Align(validS2Len, (int32_t)32) * sizeof(SCORE_T);
                AscendC::DataCopyPad(
                    scoreOutLocal_,
                    scoreGm[vecOffset * QLIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                            ldDstOffset],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }

        if (!info.isNeedLD) {
            if (validS2Len < topkCount_) {
                uint64_t mask[2] = {0, 0};
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 8); // 将`mask[0]`左移`validS2Len % 8`位 对齐
                PipeBarrier<PIPE_V>();
                // repeatTime  每次读取连续的8个datablock（每个block32Bytes，共256Bytes）
                // dstBlockStride 单次迭代内，矢量目的操作数不同datablock间地址步长。
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
            }

            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg,
                          topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            if (returnValueFlag) {
                WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                vector1::UIntToFloatReturnValue(valueOutLocal_, scoreOutLocal_, topkCountAlign256_);
                // 无效值刷-inf
                if (validS2Len < topkCount_) {
                    // mask[1]=0：bit模式高64位不参与，第一段只刷低64个bf16元素，剩余由下方count版Duplicate补刷
                    uint64_t mask[2] = {0, 0};
                    mask[0] = ~0;
                    mask[0] = mask[0] << (validS2Len % 16);
                    PipeBarrier<PIPE_V>();
                    Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>()[validS2Len / 16 * 16],
                              constInfo_.NEG_INF_BFLOAT, mask, 1, 1, 0);
                }

                if (validS2Len / 16 * 16 + 64 < topkCount_) {
                    PipeBarrier<PIPE_V>();
                    Duplicate(valueOutLocal_.template ReinterpretCast<uint16_t>()[validS2Len / 16 * 16 + 64],
                              constInfo_.NEG_INF_BFLOAT, topkCount_ - (validS2Len / 16 * 16 + 64));
                }
                SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            }

            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            if (returnValueFlag) {
                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(bfloat16_t); // bytes
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                     copyOutValueParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        } else {
            PipeBarrier<PIPE_V>();
            AscendC::Adds(indicesOutLocal_.ReinterpretCast<int32_t>(), indicesOutLocal_.ReinterpretCast<int32_t>(),
                          static_cast<int32_t>(curS2StartIdx), topkCountAlign16_);
            if (validS2Len < topkCount_) {
                uint64_t mask[2] = {0, 0};
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskI[2] = {0, 0};
                maskI[0] = ~0;
                maskI[0] = maskI[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, maskI, 1, 1, 0);
            }
            if (validS2Len / scoreAlign * scoreAlign + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign + 64], zero,
                          topkCount_ - (validS2Len / scoreAlign * scoreAlign + 64));
            }
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg,
                          topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            if (topkCountAlign16_ != topkCount_) {
                uint64_t mask[2] = {0, 0};
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskIndices[2] = {0, 0};
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 8 * 8], neg, maskIndices, 1, 1, 0);
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[offset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[offset], indicesOutLocal_.ReinterpretCast<int32_t>(), ldCopyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Vector<QLIV2T>::ProcessLD()
{
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams copyOutValueParams;
    copyOutValueParams.blockCount = 1;
    copyOutValueParams.blockLen = topkCount_ * sizeof(bfloat16_t);
    copyOutValueParams.srcStride = 0;
    copyOutValueParams.dstStride = 0;

    uint32_t copyBytes = topkCountAlign256_; // 32B对齐 topkCountAlign256_

    AscendC::DataCopyPadExtParams<SCORE_T> scorePadParams{true, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> indexPadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams ldScoreParams;
    ldScoreParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T);                      // bytes
    ldScoreParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(SCORE_T); // 两个基本块之间的距离
    ldScoreParams.dstStride = 0;
    AscendC::DataCopyExtParams ldIndexParams;
    ldIndexParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
    ldIndexParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(uint32_t);
    ldIndexParams.dstStride = 0;

    uint64_t ldProcessLen = ldInfo_.workspaceNum * topkCountAlign16_;
    if (ldProcessLen <= 0) {
        return;
    }
    uint64_t mrgValueLen = trunkLenLd_ + topkCountAlign256_;
    uint32_t ldWorkspaceNum = (ldProcessLen > mrgValueLen) ? (trunkLenLd_ / topkCountAlign16_) : ldInfo_.workspaceNum;
    // 搬运次数，一次搬运ldworkspaceNum块
    uint32_t ldProcessNum = CeilDiv(ldInfo_.workspaceNum, ldWorkspaceNum);
    uint32_t ldProcessOffset = 0;

    SCORE_T zero = 0;
    int32_t neg = -1;
    uint32_t zero32 = 0;
    uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // 读取数据段过长
            if (ldProcessNum > 1) {
                ldProWorkspaceNum = (i == ldProcessNum - 1) ? ldInfo_.workspaceNum - i * ldWorkspaceNum :
                                                              ldWorkspaceNum; // 当前搬运块数
            }
            ldProcessOffset = (i != 0) ? topkCountAlign256_ : 0;
            PipeBarrier<PIPE_V>();
            // 索引全部刷-1 value全部刷0
            Duplicate(mrgValueLocal_[ldProcessOffset], zero, topkCountAlign256_ + trunkLenLd_ - ldProcessOffset);
            Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[ldProcessOffset], neg,
                      topkCountAlign256_ + trunkLenLd_ - ldProcessOffset);

            ldScoreParams.blockCount = ldProWorkspaceNum;
            ldIndexParams.blockCount = ldProWorkspaceNum;

            int32_t s2Len = topkCountAlign16_ * ldProWorkspaceNum + ldProcessOffset;
            uint32_t s2LenAlign = QLIV2Common::Align(s2Len, (int32_t)256); // 寄存器需要256对齐
            uint64_t LDGmOffset = ldInfo_.workspaceIdx * s1BaseSize_ * topkCountAlign16_ +
                                  topkCountAlign16_ * (ldInfo_.mStart + j) +
                                  i * ldWorkspaceNum * s1BaseSize_ * topkCountAlign16_; // 加入LD处理偏移
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(mrgValueLocal_[ldProcessOffset], ldScoreGm[LDGmOffset], ldScoreParams, scorePadParams);
            AscendC::DataCopyPad(ldIndexLocal_[ldProcessOffset], ldIndexGm[LDGmOffset].ReinterpretCast<uint32_t>(),
                                 ldIndexParams, indexPadParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            // 对非对齐索引和值刷0 -1
            if (s2LenAlign != s2Len) {
                uint64_t mask[2] = {0, 0};
                mask[0] = ~0;
                mask[0] = mask[0] << (s2Len % 64);
                Duplicate(mrgValueLocal_[s2Len / 64 * 64], zero, mask, 1, 1, 0);
                // 把s2Len对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64], neg, mask, 1, 1, 0);
            }
            if (s2Len / 64 * 64 + 64 < s2LenAlign) {
                PipeBarrier<PIPE_V>();
                Duplicate(mrgValueLocal_[s2Len / 64 * 64 + 64], zero, s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64 + 64], neg,
                          s2LenAlign - (s2Len / 64 * 64 + 64));
            }

            PipeBarrier<PIPE_V>();
            topkOp_.LdTopK(mrgValueLocal_, ldIndexLocal_, indicesOutLocal_, scoreOutLocal_, s2LenAlign, j,
                           ldProcessNum);
            if (topkCountAlign256_ != topkCount_) {
                uint64_t mask[2] = {0, 0};
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                uint64_t maskIndices[2] = {0, 0};
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64], neg, maskIndices, 1, 1, 0);
            }
            if (topkCount_ / 64 * 64 + 64 < topkCountAlign256_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64 + 64], zero,
                          topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64 + 64], neg,
                          topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
            }

            PipeBarrier<PIPE_V>();
            AscendC::DataCopy(ldIndexLocal_, indicesOutLocal_, copyBytes);
            AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, copyBytes);
        }
        if (returnValueFlag) {
            PipeBarrier<PIPE_V>();
            // 可排序键还原为 bf16，并将无效位（score==0，对应 index==-1）刷为 -inf
            // Convert sortable key to bf16 and flush invalid (score==0, index==-1) positions to -inf
            vector1::UIntToFloatReturnValueWithInfMask(valueOutLocal_, scoreOutLocal_, topkCountAlign256_,
                                                       constInfo_.NEG_INF_BFLOAT);
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        uint64_t indiceOutGmOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * topkCount_;
        AscendC::DataCopyPad(indiceOutGm[indiceOutGmOffset], indicesOutLocal_.ReinterpretCast<int32_t>(),
                             copyOutParams);
        if (returnValueFlag) {
            AscendC::DataCopyPad(valueOutGm[indiceOutGmOffset], valueOutLocal_, copyOutValueParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}
} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
