/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH35_H
#define POOL_KEY_INDEXER_SERVICE_VECTOR_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../pool_key_indexer_common.h"
#include "../arch35/vf/pool_key_indexer_vector1.h"
#include "../arch35/vf/pool_key_indexer_topk.h"
#include "../arch35/vf/pool_key_indexer_vf_expand.h"

namespace PkiKernel {
using namespace PkiCommon;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TRUNK_LEN_4K = 4096;
constexpr uint32_t TRUNK_LEN_2K = 2048;
constexpr uint32_t TOPK_LEN_7K = 7168;
constexpr uint32_t TOPK_LEN_5K = 5120;

template <typename Q_T, typename W_T = void>
struct PkiTypeTraits {
    using weightsType = Q_T; // 默认：weightsType绑定Q_T
};

template <typename Q_T>
struct PkiTypeTraits<Q_T, float> {
    using weightsType = float; // W_T=float时，强制weightsType为float
};

// FP8模式: weights 为半精度(FP16/BF16, entry 按 ORIG_DTYPE_WEIGHTS 分发), W_T 必须绑定
// 半精度(fp8 标量语义编译器不支持); 量化场景 weightsType 由 PkiType 显式携带
template <typename W_HALF_T>
struct PkiQuantWeightsTraits {
    using weightsType = W_HALF_T;
};
template <>
struct PkiTypeTraits<fp8_e4m3fn_t> {
    using weightsType = half;
};
template <typename LIT>
class PoolKeyIndexerServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr PkiLayout LAYOUT_T = LIT::layout;
    static constexpr PkiLayout K_LAYOUT_T = LIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using SCORE_T = uint32_t;
    // 量化场景: PkiType 显式携带 weightsType(half/bfloat16_t), 优先于推导;
    // 非量化(WEIGHTS_T=void)沿用 PkiTypeTraits 推导
    using W_DERIVED_T =
        typename PkiTypeTraits<Q_T, typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;
    using W_EXPLICIT_T = typename LIT::weightsType;
    using W_T = typename std::conditional<std::is_same<W_EXPLICIT_T, void>::value, W_DERIVED_T, W_EXPLICIT_T>::type;

    __aicore__ inline PoolKeyIndexerServiceVector(){};
    __aicore__ inline void ProcessVec1(const PkiCommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const PkiCommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct PkiCommon::ConstInfo &constInfo,
                                      const PoolKeyIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<float> valueOutGm, GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<float> qScaleGm = {}, GlobalTensor<float> kScaleGm = {});
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

private:
    __aicore__ inline void ExpandAndAppendIndices(LocalTensor<int32_t> poolIndices, LocalTensor<int32_t> &tokenIndices,
                                                  LocalTensor<int32_t> &workBuf, uint32_t sparseCount,
                                                  uint32_t poolSize, uint32_t validS2Len, int32_t poolTailK,
                                                  int32_t L_orig, uint32_t curS1Idx, uint32_t curS1Size);
    // mode=0: k_descale GM->UB 搬运(非 PA 连续寻址 / PA 按 block_table 查物理块)。
    // dstOffset 为乒乓区内本 s2 块的段偏移, 与读取位置一致
    __aicore__ inline void GetKeyScale(LocalTensor<float> kScaleUB, uint64_t keyScaleGmOffset, int64_t batchId,
                                       int64_t startS2, int64_t getLen, uint32_t dstOffset = 0);

    // arch35 标量(S pipe)与向量(V pipe)/MTE3 间必须显式硬同步,
    // PipeBarrier<PIPE_V> 不保证 S 侧读写顺序
    __aicore__ inline void VToSSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    __aicore__ inline void SToVSync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventID);
        WaitFlag<HardEvent::S_MTE3>(eventID);
    }

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // mode=0 (FP8 per-token-head) scale GM(q/k_descale, float)
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    // mode=0 k_descale 搬运: block_table 标量读(S pipe)后 MTE2 搬运的同步
    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;
    // mode=0: weight(fp32 cast) × qScale 预乘结果的 fp32 UB(向量 SIMD 预乘)
    TBuf<TPosition::VECCALC> weightScaleFp32Buf_;
    LocalTensor<float> weightScaleFp32UB_;

    // mode=0 (FP8 per-token-head) scale UB(乒乓)
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<float> qScaleUB_;
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<float> kScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<float> valueOutLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> expandOutBuf_;
    LocalTensor<int32_t> expandOutLocal_;
    TBuf<TPosition::VECCALC> workBuf_;
    LocalTensor<int32_t> workLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;
    int32_t blockId_ = -1;
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
    uint32_t topkCountAlign256_ = 0;
    uint32_t trunkLen_ = 0;
    uint32_t poolSize_ = 1;
    uint32_t outputLen_ = 0;
    uint32_t kScaleLoop_ = 0; // mode=0: k_descale 16 轮乒乓搬运计数
    bool returnValue = false;

    struct PkiCommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(float));
    resMm1UB_ = resMm1Buf_.Get<float>();

    // weightBuf_: 行距 Align(gSize,16); 量化路径 LoadAlign 每行固定读 64 个 W_T,
    // gSizeAlign16<64 时末行读取越过缓冲末端
    pipe->InitBuffer(weightBuf_,
                     2 * CeilDiv(s1BaseSize_, 2) * PkiCommon::Align((uint64_t)gSize_, (uint64_t)16) * sizeof(W_T));
    weightUB_ = weightBuf_.Get<W_T>();
    if constexpr (LIT::isFp8PerToken) {
        // mode=0: weight×qScale SIMD 预乘输出缓冲(fp32, 行距 128 元素 bank 对齐),
        // 向量预乘替代逐 g 标量循环
        pipe->InitBuffer(weightScaleFp32Buf_, 2 * CeilDiv(s1BaseSize_, 2) * 128 * sizeof(float));
        weightScaleFp32UB_ = weightScaleFp32Buf_.Get<float>();
        // mode=0 scale UB: qScale 行距固定 128 元素(512B bank 对齐, 满足
        // LoadAlign 256B 对齐要求); kScale 为 s2 维度 16 轮乒乓
        pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * 128 * sizeof(float));
        qScaleUB_ = qScaleBuf_.Get<float>();
        pipe->InitBuffer(kScaleBuf_, 2 * 16 * s2BaseSize_ * sizeof(float));
        kScaleUB_ = kScaleBuf_.Get<float>();
        // kScale 铺 0: 尾部未搬运区为垃圾值, 乘入会污染 TopK(0 与无效池语义一致)
        Duplicate(kScaleUB_, 0.0f, 2 * 16 * s2BaseSize_);
        SetFlag<HardEvent::V_MTE2>(KSCALE_S_MTE2_EVENT);
        WaitFlag<HardEvent::V_MTE2>(KSCALE_S_MTE2_EVENT);
    }
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>(); // out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    // returnvalue
    if (topkCount_ <= 2048) {
        pipe->InitBuffer(valueOutBuf_, topkCountAlign256_ * sizeof(float));
        valueOutLocal_ = valueOutBuf_.Get<float>();
    } else {                                        // sparseCount > 2k时，复用return value相关UB
        valueOutLocal_ = mrgValueBuf_.Get<float>(); // returnValue float
    }

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    if (poolSize_ > 1) {
        uint32_t expandOutLen = PkiCommon::Align(static_cast<uint64_t>(outputLen_ + 64), (uint64_t)8);
        pipe->InitBuffer(expandOutBuf_, expandOutLen * sizeof(uint32_t));
        expandOutLocal_ = expandOutBuf_.Get<int32_t>();
        // workLocal_ 布局(段起始均 256B 对齐): [0,64)=r/ps 与 [64,128)=r%ps 为
        // pow2 ps gather 模板, [128,128+ps) 为非 pow2 ps 的 offsetTpl 回退模板
        uint32_t workSize = 128 + PkiCommon::Align(static_cast<uint64_t>(poolSize_ + 64), (uint64_t)8);
        pipe->InitBuffer(workBuf_, workSize * sizeof(uint32_t));
        workLocal_ = workBuf_.Get<int32_t>();
        // 展开模板一次性构建并常驻(替代每行标量重建 + SToVSync 的逐行开销)
        AscendC::CreateVecIndex(workLocal_[128], static_cast<int32_t>(0), poolSize_);
        // pow2 ps 的 gather 模板为 S pipe 标量写, 后续 V pipe LoadAlign 读前需 S->V 硬同步
        const uint32_t tplLoop = 64;
        for (uint32_t r = 0; r < tplLoop; r++) {
            workLocal_.SetValue(r, static_cast<int32_t>(r / poolSize_));
            workLocal_.SetValue(tplLoop + r, static_cast<int32_t>(r % poolSize_));
        }
        SToVSync();
    }

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitParams(
    const struct PkiCommon::ConstInfo &constInfo, const PoolKeyIndexerTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
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
    returnValue = constInfo.returnValue;
    blockId_ = GetBlockIdx();
    trunkLen_ = constInfo.sparseCount > TOPK_LEN_5K ?
                    (constInfo.sparseCount > TOPK_LEN_7K ? TRUNK_LEN_2K : TRUNK_LEN_4K) :
                    TRUNK_LEN_8K;
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = PkiCommon::Align(constInfo.sparseCount, (uint64_t)256);
    poolSize_ = static_cast<uint32_t>(constInfo.poolSize);
    outputLen_ = constInfo.sparseCount * poolSize_ + poolSize_ - 1;
    kScaleLoop_ = 0;
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVecInputTensor(
    GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm, GlobalTensor<float> valueOutGm,
    GlobalTensor<int32_t> blockTableGm, GlobalTensor<float> qScaleGm, GlobalTensor<float> kScaleGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
    // mode=0 (FP8 per-token-head): vector 侧 scale 融合
    if constexpr (LIT::isFp8PerToken) {
        this->qScaleGm = qScaleGm;
        this->kScaleGm = kScaleGm;
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

// mode=0: k_descale GM->UB(每池 1 个 float scale); 非 PA 连续寻址,
// PA 按 block_table 逐物理块搬运(块基址 × keyDequantScaleStride0)
template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::GetKeyScale(LocalTensor<float> kScaleUB,
                                                                     uint64_t keyScaleGmOffset, int64_t batchId,
                                                                     int64_t startS2, int64_t getLen,
                                                                     uint32_t dstOffset)
{
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    if constexpr (PAGE_ATTENTION) {
        // 按池号连续写入 UB 段, GM 源按物理块跳转; 段内偏移用相对本段
        // 起点的池号, 与块边界解耦
        int64_t written = 0; // 段内已写池数
        while (written < getLen) {
            int64_t absPool = startS2 + written; // batch 内绝对池号
            int64_t s2BlkId = absPool / kCacheBlockSize_;
            int64_t s2BlkOffset = absPool % kCacheBlockSize_;
            int64_t blockId = blockTableGm.GetValue(batchId * maxBlockNumPerBatch_ + s2BlkId);
            int64_t partLen = Min((int64_t)kCacheBlockSize_ - s2BlkOffset, getLen - written);
            copyInParams.blockLen = partLen * sizeof(float);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[dstOffset + static_cast<uint32_t>(written)],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + s2BlkOffset], copyInParams,
                                 padParams);
            written += partLen;
        }
    } else {
        copyInParams.blockLen = getLen * sizeof(float);
        AscendC::DataCopyPad(kScaleUB[dstOffset], kScaleGm[keyScaleGmOffset], copyInParams, padParams);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // 无效行清理(与 ProcessVec1 无效行路径同构): Duplicate 铺 UB → V_MTE3 硬同步
    // → DataCopyPad 出 GM(行宽非 32B 对齐, 不可用普通 DataCopy/InitGlobalMemory)
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = (poolSize_ > 1 ? outputLen_ : topkCount_) * sizeof(uint32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    if (poolSize_ > 1) {
        Duplicate(expandOutLocal_, constInfo_.INVALID_IDX, PkiCommon::Align(outputLen_, (uint32_t)8));
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], expandOutLocal_, copyOutParams);
    } else {
        Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), constInfo_.INVALID_IDX, topkCount_);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
    }
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    if (returnValue) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>(), constInfo_.INVALID_VAL, topkCount_);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = topkCount_ * sizeof(float);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        // invalidS1Offset 是 indices 行偏移(行宽 outputLen_/sparseCount);
        // value 行宽为 sparseCount, 需换算行号后重算偏移, 否则越界写且本行 value 漏写
        uint64_t idxStride = (poolSize_ > 1) ? outputLen_ : constInfo_.sparseCount;
        uint64_t valueOffset = (static_cast<uint64_t>(invalidS1Offset) / idxStride) * constInfo_.sparseCount;
        AscendC::DataCopyPad(valueOutGm[valueOffset], valueOutLocal_, copyOutValueParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ExpandAndAppendIndices(
    LocalTensor<int32_t> poolIndices, LocalTensor<int32_t> &tokenIndices, LocalTensor<int32_t> &workBuf,
    uint32_t sparseCount, uint32_t poolSize, uint32_t validS2Len, int32_t poolTailK, int32_t L_orig, uint32_t curS1Idx,
    uint32_t curS1Size)
{
    uint32_t topk = sparseCount * poolSize;
    uint32_t totalOut = topk + poolSize - 1;
    uint32_t alignedTotalOut = PkiCommon::Align(totalOut, (uint32_t)8);

    // 尾块可见 token 数(两条路径共用)
    int32_t visibleTailK = 0;
    if (poolTailK > 0) {
        if (constInfo_.maskMode == 0) {
            visibleTailK = poolTailK;
        } else {
            int32_t globalPosQ = L_orig - static_cast<int32_t>(curS1Size) + static_cast<int32_t>(curS1Idx);
            visibleTailK = PkiCommon::Max(0, PkiCommon::Min(poolTailK, globalPosQ - static_cast<int32_t>(topk) + 1));
        }
    }

    // pow2 pool_size(2/4/.../64)走 gather 向量展开(见 vf/pool_key_indexer_vf_expand.h):
    // out[m] = ps*idx[m/ps] + m%ps, 消除逐 pool 标量循环; ps=128 走下方 Duplicate+Add 路径
    if ((poolSize & (poolSize - 1)) == 0 && poolSize <= 64) {
        Duplicate<int32_t>(tokenIndices, -1, alignedTotalOut);
        PipeBarrier<PIPE_V>();
        if (validS2Len > 0) {
            // 展开轮数受 TopK 选池数截断, 多余可见池不展开(防越写 outputLen 污染尾区)
            uint32_t expandRounds = PkiCommon::Min(validS2Len, sparseCount);
            uint32_t effExpand = expandRounds * poolSize;
            // gather 表为 TopK 向量写产物, 调用方已 PipeBarrier<PIPE_V>, V pipe 读可见
            pkiexpand::ExpandPow2PoolIndices(tokenIndices, poolIndices.ReinterpretCast<uint32_t>(), workBuf, effExpand,
                                             poolSize);
        }
        uint32_t validExpand = validS2Len * poolSize;
        if (validExpand < topk) {
            PipeBarrier<PIPE_V>();
            // [validExpand, topk) 补 -1 覆盖 gather 尾块垃圾; 起始可能非 32B 对齐, 走 mask Duplicate
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validExpand % 8);
            Duplicate<int32_t>(tokenIndices[validExpand / 8 * 8], -1, mask, 1, 1, 0);
            if (validExpand / 8 * 8 + 64 < topk) {
                PipeBarrier<PIPE_V>();
                Duplicate<int32_t>(tokenIndices[validExpand / 8 * 8 + 64], -1, topk - (validExpand / 8 * 8 + 64));
            }
            PipeBarrier<PIPE_V>();
        }
        // 尾区 [topk, totalOut) 恒做 -1 清空(gather 尾块 lane 可能越 topk 写垃圾, 同走 mask Duplicate)
        uint32_t tailPos = topk;
        uint64_t mask[1];
        mask[0] = ~0;
        mask[0] = mask[0] << (tailPos % 8);
        Duplicate<int32_t>(tokenIndices[tailPos / 8 * 8], -1, mask, 1, 1, 0);
        if (tailPos / 8 * 8 + 64 < totalOut) {
            PipeBarrier<PIPE_V>();
            Duplicate<int32_t>(tokenIndices[tailPos / 8 * 8 + 64], -1, totalOut - (tailPos / 8 * 8 + 64));
        }
        if (visibleTailK > 0) {
            // V→S: -1 向量填充先落地, 再标量精确写尾 token(不可对齐写, 防越出尾区容量)
            VToSSync();
            for (int32_t t = 0; t < visibleTailK; t++) {
                tokenIndices.SetValue(tailPos + t, L_orig - poolTailK + t);
            }
            // SCALAR(SetValue) 写 UB 后由 MTE3(DataCopyPad) 读出, 需硬同步
            SToMTE3Sync();
        }
        return;
    }

    // poolSize 非 8 的倍数: 向量写偏移不满足 32B 对齐会触发 aicore exception,
    // 退化为先向量铺 -1 再逐 token 标量精确写(精确到 pool 边界, 无对齐要求)。
    if (poolSize % 8 != 0) {
        Duplicate<int32_t>(tokenIndices, -1, alignedTotalOut);
        // V→S: -1 向量填充先落地, 同时保证 poolIndices 向量写对标量 GetValue 可见
        VToSSync();
        if (validS2Len > 0) {
            uint32_t expandRounds = PkiCommon::Min(validS2Len, sparseCount);
            for (uint32_t k = 0; k < expandRounds; k++) {
                int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
                uint32_t off = k * poolSize;
                for (uint32_t p = 0; p < poolSize; p++) {
                    tokenIndices.SetValue(off + p, base + static_cast<int32_t>(p));
                }
            }
        }
        for (int32_t t = 0; t < visibleTailK; t++) {
            tokenIndices.SetValue(topk + static_cast<uint32_t>(t), L_orig - poolTailK + t);
        }
        // SCALAR(SetValue) 写 UB 后由 MTE3(DataCopyPad) 读出, 需硬同步
        SToMTE3Sync();
        return;
    }

    uint32_t alignedPoolSize = PkiCommon::Align(poolSize, (uint32_t)8);

    Duplicate<int32_t>(tokenIndices, -1, alignedTotalOut);
    PipeBarrier<PIPE_V>();

    if (validS2Len > 0) {
        // V→S: poolIndices 最近写入方为向量操作, 标量 GetValue 前须等其可见
        VToSSync();

        // 展开模板 offsetTpl(0..ps-1) 已在 InitBuffers 构建并常驻 workLocal_[128]
        LocalTensor<int32_t> offsetTpl = workBuf[128];

        uint32_t outOff = 0;
        // 展开轮数受 TopK 选池数截断, 多余可见池不展开(防越写污染尾区/相邻行)
        uint32_t expandRounds = PkiCommon::Min(validS2Len, sparseCount);
        for (uint32_t k = 0; k < expandRounds; k++) {
            int32_t base = poolIndices.GetValue(k) * static_cast<int32_t>(poolSize);
            Duplicate<int32_t>(tokenIndices[outOff], base, alignedPoolSize);
            PipeBarrier<PIPE_V>();
            Add<int32_t>(tokenIndices[outOff], tokenIndices[outOff], offsetTpl, alignedPoolSize);
            outOff += poolSize;
        }
    }

    uint32_t validExpand = validS2Len * poolSize;
    if (validExpand < topk) {
        PipeBarrier<PIPE_V>();
        Duplicate<int32_t>(tokenIndices[validExpand], -1, PkiCommon::Align(topk - validExpand, (uint32_t)8));
        PipeBarrier<PIPE_V>();
    }

    if (visibleTailK > 0) {
        uint32_t tailPos = topk;
        // 先按 8 对齐清空整个尾区, 再标量精确写尾 token(不可对齐写, 防越出尾区容量)
        Duplicate<int32_t>(tokenIndices[tailPos], -1, PkiCommon::Align(totalOut - tailPos, (uint32_t)8));
        // V→S: -1 向量填充必须先于标量 SetValue 落地
        VToSSync();
        for (int32_t t = 0; t < visibleTailK; t++) {
            tokenIndices.SetValue(tailPos + t, L_orig - poolTailK + t);
        }
        // SCALAR(SetValue) 写 UB 后由 MTE3(DataCopyPad) 读出, 需硬同步
        SToMTE3Sync();
    }
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessVec1(const PkiCommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    // CV同步, V核等C核计算完mm1，mm1Res已搬运到UB
    CrossCoreWaitFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_CV_EVENT + pingpong);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        // V核处理完，通知C核可以把mm1Res搬运到UB
        CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + pingpong);
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // weightsGm --> weightUB_
    uint64_t gSizeAlign16 = PkiCommon::Align((uint64_t)gSize_, (uint64_t)16);
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{true, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (gSizeAlign16 - gSize_) * sizeof(W_T) / 32;
    DataCopyPad(weightUB_[pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16], weightsGm[weightGmOffset],
                qwDataCopyExtParams, padWeightsParams);

    if constexpr (LIT::isFp8PerToken) {
        // mode=0: q_descale 与 weights 同 pattern 同 GM 偏移(shape 同构),
        // dst 行距 128 元素(512B bank 对齐)
        DataCopyPadExtParams<float> padQScaleParams{true, 0, 0, 0};
        DataCopyExtParams qScaleDataCopyExtParams;
        qScaleDataCopyExtParams.blockCount = curAivS1ProcNum;
        qScaleDataCopyExtParams.blockLen = gSize_ * sizeof(float);
        qScaleDataCopyExtParams.srcStride = 0;
        qScaleDataCopyExtParams.dstStride = (128 - gSize_) * sizeof(float) / 32;
        DataCopyPad(qScaleUB_[pingpong * CeilDiv(s1BaseSize_, 2) * 128], qScaleGm[weightGmOffset],
                    qScaleDataCopyExtParams, padQScaleParams);
        // k_descale: 每 16 轮 s2BaseSize 乒乓搬运一次((s2Idx-s2Start)%16==0 时刷新),
        // 写入起点 = 乒乓区头 + 本块段偏移, 与读取位置一致
        if ((info.s2Idx - info.s2Start) % 16 == 0) {
            uint32_t kScalePingpong = (kScaleLoop_ % 2);
            uint32_t kScaleBlkOff = ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_;
            uint32_t getLen = 16 * s2BaseSize_ > (info.actS2Size - info.s2Idx * s2BaseSize_) ?
                                  (info.actS2Size - info.s2Idx * s2BaseSize_) :
                                  16 * s2BaseSize_;
            // 钳制到本乒乓区剩余容量(刷新点在段头, 通常不触达)
            uint32_t capacity = 16 * s2BaseSize_ - kScaleBlkOff;
            if (getLen > capacity) {
                getLen = capacity;
            }
            GetKeyScale(kScaleUB_[kScalePingpong * 16 * s2BaseSize_], info.tensorKeyScaleOffset, info.bIdx, curS2Idx,
                        getLen, kScaleBlkOff);
            kScaleLoop_++;
        }
    }

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    if constexpr (LIT::isFp8PerToken) {
        // mode=0: weight 半精度 -> fp32 cast × qScale 的 SIMD 向量预乘落 UB,
        // 替代逐 g 标量循环; 行内 [gSize,64) 补 0(0 与无效 g 语义一致)
        for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
            uint64_t dstOff = pingpong * CeilDiv(s1BaseSize_, 2) * 128 + s1IdxTmp * 128;
            Duplicate<float>(weightScaleFp32UB_[dstOff], 0.0f, 128);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
            uint64_t srcOff = pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16 + s1IdxTmp * gSizeAlign16;
            uint64_t dstOff = pingpong * CeilDiv(s1BaseSize_, 2) * 128 + s1IdxTmp * 128;
            // 独立 __simd_vf__ helper: bf16 Cast 需硬件展开,
            // 内联到普通 __aicore__ 函数不被编译器支持
            vector1::MulWeightQScaleSIMD((__local_mem__ W_T *)weightUB_[srcOff].GetPhyAddr(),
                                         (__local_mem__ float *)qScaleUB_[dstOff].GetPhyAddr(),
                                         (__local_mem__ float *)weightScaleFp32UB_[dstOff].GetPhyAddr());
        }
        // V 写落地后, WithScale 的向量 LoadAlign 才能读(同 pipe 有序性足够,
        // 但跨 ForEachS1 循环保险起见显式 barrier)
        PipeBarrier<PIPE_V>();
    }

    for (int64_t s1IdxTmp = 0; s1IdxTmp < curAivS1ProcNum; s1IdxTmp++) {
        if constexpr (LIT::isFp8PerToken) {
            // mode=0: kScale 取本 s2 基本块对应的 float scale(逐 s2 元素;
            // 乒乓区起点 + 块内偏移 (s2Idx-s2Start)%16 * s2BaseSize)
            uint32_t kScalePingpong = ((kScaleLoop_ - 1) % 2);
            uint32_t kScaleBlkOff = ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_;
            vector1::MulWeightAndReduceSumWithScale(
                vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + s1IdxTmp * s2BaseSize_],
                resMm1UB_[pingpong * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ + s1IdxTmp * gSize_ * s2BaseSize_],
                weightScaleFp32UB_[pingpong * CeilDiv(s1BaseSize_, 2) * 128 + s1IdxTmp * 128],
                kScaleUB_[kScalePingpong * 16 * s2BaseSize_ + kScaleBlkOff], gSize_, constInfo_.qkScale);
        } else {
            vector1::MulWeightAndReduceSum(
                vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + s1IdxTmp * s2BaseSize_],
                resMm1UB_[pingpong * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ + s1IdxTmp * gSize_ * s2BaseSize_],
                weightUB_[pingpong * CeilDiv(s1BaseSize_, 2) * gSizeAlign16 + s1IdxTmp * gSizeAlign16], gSize_,
                constInfo_.qkScale);
        }
    }
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    uint64_t kSeqSizeAlign = PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx : CeilDiv(s1BaseSize_, 2) * kSeqSizeAlign + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = (kSeqSizeAlign - s2BaseSize_) * sizeof(SCORE_T);

    DataCopyPad(scoreGm[vec1OutGmOffset], vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_], copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    // V核处理完，通知C核可以把mm1Res搬运到UB
    CrossCoreSetFlag<PkiCommon::ConstInfo::PKI_SYNC_MODE4, PIPE_V>(PkiCommon::ConstInfo::CROSS_VC_EVENT + pingpong);
}

template <typename LIT>
__aicore__ inline void PoolKeyIndexerServiceVector<LIT>::ProcessTopK(const PkiCommon::RunInfo &info)
{
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
    copyOutParams.blockLen = (poolSize_ > 1 ? outputLen_ : topkCount_) * sizeof(uint32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.poolSize);
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (poolSize_ > 1) {
                Duplicate(expandOutLocal_, neg, PkiCommon::Align(outputLen_, (uint32_t)8));
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * outputLen_],
                                     expandOutLocal_, copyOutParams);
            } else {
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                     indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValue) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>(), constInfo_.INVALID_VAL, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(float);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                     copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
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
                uint32_t validS2LenAlign = PkiCommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(
                    mrgValueLocal_,
                    scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1, returnValue);
            } else {
                uint32_t actS2LoopNum = 0;
                if (topkCount_ > trunkLen_) {
                    actS2LoopNum = 1 + (validS2Len - topkCountAlign256_ + trunkLen_ - 1) / trunkLen_;
                } else {
                    actS2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
                }
                for (uint32_t loopIdx = 0; loopIdx < actS2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        if (topkCount_ > trunkLen_) {
                            copyInParams.blockLen = topkCountAlign256_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(scoreOutLocal_,
                                                 scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                      (uint64_t)s2BaseSize_)],
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
                                                 scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize,
                                                                                      (uint64_t)s2BaseSize_)],
                                                 copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, actS2LoopNum,
                                    returnValue);
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
                        offset = vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 topkCountAlign256_ + (loopIdx - 1) * trunkLen_;
                    } else {
                        validTrunkLen =
                            (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                        offset = vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                 loopIdx * trunkLen_;
                    }
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    // 如果是tok > trunklen, 第一轮每调用topk，是直接拷贝的，所以不需要刷零
                    bool isZeroPadding = (topkCount_ > trunkLen_) ? (loopIdx > 1) : true;
                    if (topkCountAlign256_ != topkCount_ && isZeroPadding) {
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
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero,
                                  PkiCommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            PkiCommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, actS2LoopNum,
                            returnValue);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
            if (returnValue) {
                copyInParams.blockLen = PkiCommon::Align(validS2Len, (int32_t)32) * sizeof(SCORE_T);
                AscendC::DataCopyPad(
                    scoreOutLocal_,
                    scoreGm[vecOffset * PkiCommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)],
                    copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
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
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg,
                      topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        if (poolSize_ > 1) {
            PipeBarrier<PIPE_V>();
            ExpandAndAppendIndices(indicesOutLocal_.ReinterpretCast<int32_t>(), expandOutLocal_, workLocal_, topkCount_,
                                   poolSize_, static_cast<uint32_t>(validS2Len), info.poolTailK,
                                   static_cast<int32_t>(info.actS2SizeOrig), static_cast<uint32_t>(curAivS1Idx + i),
                                   info.actS1Size);
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        if (poolSize_ > 1) {
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * outputLen_], expandOutLocal_,
                                 copyOutParams);
        } else {
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        }

        // // 是否返回Value值
        if (returnValue) {
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            // uint32_t -> float
            vector1::UIntToFloatReturnValue(valueOutLocal_, scoreOutLocal_, topkCountAlign256_);

            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 16);
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 16 * 16],
                          constInfo_.INVALID_VAL, mask, 1, 1, 0);
            }
            if (validS2Len / 16 * 16 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 16 * 16 + 64],
                          constInfo_.INVALID_VAL, topkCount_ - (validS2Len / 16 * 16 + 64));
            }
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyParams copyOutValueParams;
            copyOutValueParams.blockCount = 1;
            copyOutValueParams.blockLen = topkCount_ * sizeof(float); // bytes
            copyOutValueParams.srcStride = 0;
            copyOutValueParams.dstStride = 0;
            // 搬运到GM
            AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                 copyOutValueParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
}
} // namespace PkiKernel
#endif
