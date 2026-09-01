/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_sparse_flash_attention_service_vector_mla.h
 * \brief
 */
#ifndef TURBOQUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
#define TURBOQUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H

// [TQ4 scale 传递] 复用 kvValidSizeGm_ 的上半区（见 kernel_mla.h 的 SetGlobalBuffer 注释）。
// 该区按 half 视图寻址：base = TQ4_SCALE_HALF_BASE，布局 slot(loop % 4) × TQ4_SCALE_SLOT_STRIDE。
static constexpr uint32_t TQ4_SCALE_HALF_BASE = 2048U;  // int32 [1024, 2048) 对应 half [2048, 4096)
static constexpr uint32_t TQ4_SCALE_SLOT_STRIDE = 512U; // = s2BaseSize 上界
static_assert(TQ4_SCALE_SLOT_STRIDE * sizeof(uint16_t) <= 1024U,
              "TQ4 half staging exceeds the first 1K of tq4ScaleBuf_");
// fp32 展开区（float 视图，= half 视图的 2560），供按列 Mul 使用
static constexpr uint32_t TQ4_SCALE_UB_F32 = 256U; // 新 buffer 内 byte 1024
static_assert(TQ4_SCALE_UB_F32 * sizeof(float) >= 1024U, "TQ4 fp32 area must start after the half staging area");
static_assert((TQ4_SCALE_UB_F32 + TQ4_SCALE_SLOT_STRIDE) * sizeof(float) <= 4096U,
              "TQ4 fp32 scale exceeds tq4ScaleBuf_ (4K)");

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "turbo_quant_sparse_flash_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename QSFAT>
class QSFAVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename QSFAT::kvType;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using OUT_T = typename QSFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline QSFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const TurboQuantSparseFlashAttentionTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
                                                GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm);
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
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // [TQ4] codebook dequant of dealRow combined slots -> antiKvTensorAsB16 [dealRow,512] bf16 (Phase B)
    __aicore__ inline void Tq4DequantRows(LocalTensor<KV_T> &srcTensor, LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // 有效 KV 长度为 0 时补齐 LSE 输出（该场景不会进入 CopyFALseToGm）
    __aicore__ inline void InitLseForZeroSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
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

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    // [TQ4] scale 的 MTE2->V 同步，用未被本文件其他 MTE2_V 用例占用的 ID
    static constexpr uint64_t TQ4_SCALE_SYNC_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                               // 32/4=8
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    // [TQ4] 反量化批大小：bf16 快路径 16（scratch：compact 4K + half 8K +
    // idx 16K = 28K ≤ 32K inputBuff2；byte-LUT 每字节一次 Gather，half/idx 区按
    // 字节而非 nibble 计）；非 bf16 慢路径保留原 4（float work 8K + half 4K +
    // idx 8K = 20K）。
    static constexpr int32_t TQ4_DEQUANT_CHUNK = IsSameType<K_ROPE_T, bfloat16_t>::value ? 16 : 4;

    const TurboQuantSparseFlashAttentionTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<K_ROPE_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;
    // [TQ4] kvValidSizeGm_ 上半区的 uint16 视图。GlobalTensor 没有 ReinterpretCast，
    // 按仓内 mm2ResInt32Gm 的做法，从同一物理地址另建一个视图。
    GlobalTensor<uint16_t> tq4ScaleGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K * 2
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;        // 32K
    TBuf<> tmpBuff2;        // 8K
    TBuf<> v0ValidSizeBuff; // 8K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxMaxDefaultBuff; // 1K
    TBuf<> softmaxSumDefaultBuff; // 1K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;

    // [TQ4] persistent centSigned codebook (setup once in InitBuffers); int4b_t HW Cast unpack
    // needs no nibble masks / reorder idx.
    TBuf<> tq4CentBuf_; // 16 float (centSigned[k] = _CENT[(k+8)%16])
    // [TQ4] 256 项 byte-LUT：byteLut[b] 一次携带一个字节的两个质心 bf16 位型
    //（高 16 位 = 高 nibble、低 16 位 = 低 nibble；小端 + low-nibble-first 打包，
    // 低半字恰好落偶数维）。bf16 反量化每字节一次 Gather（见 Tq4DequantRows）。
    TBuf<> tq4ByteLutBuf_; // 256 uint32 = 1KB
    // [TQ4] scale 批量导出用的索引表：sTIdx[i] = i*32（字节偏移），一次性初始化
    TBuf<> tq4STIdxBuf_; // 512B = 128 个 uint32
    // [TQ4] per-column scale 专用暂存。曾经挤在 v0ValidSizeUb_ 内（手工偏移 4096/5120），
    // 实测 M 分块数>1 时 Cast/Duplicate 必崩 507015，故改为独立 buffer。
    //   half 视图 [0, 512)   : vec1 从 GM 读回的 512 个 s_j
    //   float 视图 [256, 768): 展开成 fp32 供按列 Mul（起始 byte 1024）
    TBuf<> tq4ScaleBuf_; // 4K
    LocalTensor<float> tq4Cent_;
};

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    // [TQ4] one-time setup: centSigned codebook (gather index = int4b signed nibble + 8). Done ONCE.
    // [TQ4] 索引表一次性初始化：每 32B 取头 2B
    pipe->InitBuffer(tq4STIdxBuf_, ConstInfo::BUFFER_SIZE_BYTE_512B);
    {
        LocalTensor<uint32_t> qsfaSTIdxInit = tq4STIdxBuf_.Get<uint32_t>();
        for (uint32_t i = 0; i < 128U; ++i) {
            qsfaSTIdxInit.SetValue(i, i * 32U);
        }
    }
    pipe->InitBuffer(tq4ScaleBuf_, ConstInfo::BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(tq4CentBuf_, ConstInfo::BUFFER_SIZE_BYTE_256B);
    pipe->InitBuffer(tq4ByteLutBuf_, ConstInfo::BUFFER_SIZE_BYTE_1K);
    tq4Cent_ = tq4CentBuf_.Get<float>();
    tq4Cent_.SetValue(0, 0.00547294f);
    tq4Cent_.SetValue(1, 0.01680406f); // centSigned[k]=_CENT[(k+8)%16]
    tq4Cent_.SetValue(2, 0.02857605f);
    tq4Cent_.SetValue(3, 0.04108622f);
    tq4Cent_.SetValue(4, 0.05492980f);
    tq4Cent_.SetValue(5, 0.07101817f);
    tq4Cent_.SetValue(6, 0.09115373f);
    tq4Cent_.SetValue(7, 0.12037795f);
    tq4Cent_.SetValue(8, -0.12091285f);
    tq4Cent_.SetValue(9, -0.09111122f);
    tq4Cent_.SetValue(10, -0.07112455f);
    tq4Cent_.SetValue(11, -0.05513602f);
    tq4Cent_.SetValue(12, -0.04132067f);
    tq4Cent_.SetValue(13, -0.02874970f);
    tq4Cent_.SetValue(14, -0.01700489f);
    tq4Cent_.SetValue(15, -0.00568677f);
    // [TQ4] bf16 反量化 byte-LUT（纯标量侧生成，无跨流水序问题）。标量域没有
    // bf16 寄存器（SetValue<bfloat16_t> 后端不支持），按 RNE 手工舍出 bf16 位型，
    // 与热路径 Cast(CAST_RINT) 的 f32->bf16 舍入一致，查表直取位型。
    if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) {
        // byteLut[b] = (质心(b>>4) 的 bf16 位型 << 16) | 质心(b&0xf)。
        // nibble n 的质心 = _CENT[n] = tq4Cent_[n^8]（^8 恰等于 (k+8)%16，即码本
        // 注释里的旋转），RNE 舍出公式与 Cast(CAST_RINT) 一致。
        LocalTensor<uint32_t> qsfaByteLut = tq4ByteLutBuf_.Get<uint32_t>();
        for (uint32_t hi = 0; hi < 16U; ++hi) {
            union {
                float f;
                uint32_t u;
            } cvtHi;
            cvtHi.f = tq4Cent_.GetValue(hi ^ 8U);
            uint32_t qsfaHiBits = (cvtHi.u + 0x7FFFU + ((cvtHi.u >> 16) & 1U)) >> 16;
            for (uint32_t lo = 0; lo < 16U; ++lo) {
                union {
                    float f;
                    uint32_t u;
                } cvtLo;
                cvtLo.f = tq4Cent_.GetValue(lo ^ 8U);
                uint32_t qsfaLoBits = (cvtLo.u + 0x7FFFU + ((cvtLo.u >> 16) & 1U)) >> 16;
                qsfaByteLut.SetValue(hi * 16U + lo, (qsfaHiBits << 16) | qsfaLoBits);
            }
        }
    }
    // [TQ4] fp32 scale 区一次性初始化为 1.0。目的是保证该区域永不含未初始化数据
    // （否则 NaN 会在掩码之前被乘进 score 并穿透 softmax）。有了它，热路径就不必
    // 再从 qsfaScaleF[validCol] 起补尾部 —— 那是个非 32B 对齐的向量写，会抛 507015。
    Duplicate(tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32], static_cast<T>(1.0), TQ4_SCALE_SLOT_STRIDE);
    PipeBarrier<PIPE_ALL>();
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitParams(
    const struct ConstInfo &constInfo, const TurboQuantSparseFlashAttentionTilingDataMla *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                                      const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                                      const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                                      const GlobalTensor<KV_T> &keyGm,
                                                                      const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
    this->tq4ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(kvValidSizeGm.GetPhyAddr(0)));
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm, GlobalTensor<T> lseSumFdGm,
    GlobalTensor<int32_t> topKGm, GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<T> vec2ResGm,
                                                                      GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                             const MSplitInfo &mSplitInfo,
                                                                             LocalTensor<T> &softmaxSumUb,
                                                                             LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t qsfaBaseOffset = mSplitInfo.nBufferStartM / 2;
    size_t qsfaSize = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t qsfaAccumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t qsfaOffset = (qsfaAccumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize +          // taskoffset
                           info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize + // 份数offset
                           mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                          FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxSumUb[qsfaBaseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxMaxUb[qsfaBaseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[qsfaOffset], qsfaSize, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[qsfaOffset], qsfaSize, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
// 有效 KV 长度为 0 时 s2 循环次数为 0，CopyFALseToGm 不会被调用；而 softmax_max /
// softmax_sum 在调用侧由 at::empty 分配，不写就会把未初始化内存返回给调用方。
// 此处按该 query 行的精确长度（gSize 个元素）落值：sum = 0、max = SOFTMAX_MIN_NUM，
// 与 CopyFALseToGm 的取值口径一致。
// 不用 matmul::InitOutput：实测其在 32 字节这种小粒度下会写出远超请求范围的数据，
// 越界覆盖相邻输出缓冲。
__aicore__ inline void QSFAVectorService<QSFAT>::InitLseForZeroSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx)
{
    if (!constInfo.returnSoftmaxLse) {
        return;
    }
    size_t size = constInfo.gSize;
    int64_t offset = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQTotal = (bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(constInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        offset = n2Idx * actualSeqQTotal * constInfo.gSize + (actualSeqQPrefixSum + s1Idx) * constInfo.gSize;
    } else {
        offset = bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.gSize +
                 n2Idx * constInfo.qSeqSize * constInfo.gSize + s1Idx * constInfo.gSize;
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = sizeof(T) * size;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);

    LocalTensor<T> tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Duplicate(tmp, static_cast<T>(ConstInfo::FLOAT_ZERO), alignedSize);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopyPad(softmaxSumGm[offset], tmp, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

    tmp = outputBuff2.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Duplicate(tmp, SOFTMAX_MIN_NUM, alignedSize);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                               LocalTensor<T> &softmaxSumUb,
                                                               LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM;

    int64_t offset = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQTotal = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(constInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
        offset += info.n2Idx * actualSeqQTotal * constInfo.gSize +
                  (actualSeqQPrefixSum + info.gS1Idx / constInfo.gSize) * constInfo.gSize + mSplitInfo.nBufferStartM +
                  mSplitInfo.vecStartM;
    } else {
        offset += info.bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.gSize +
                  info.n2Idx * constInfo.qSeqSize * constInfo.gSize + info.gS1Idx / constInfo.gSize * constInfo.gSize +
                  mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    }

    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(T) * size;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxMaxUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxSumUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxSumGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb,
                                                                uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    // [TQ4] score 侧按列施加 s_j：score[i][j] = s_j·(Q_i·ŷ_j)，等价于原先在 dequant 里
    // 逐行缩放 K。放在这里（掩码之前、softmax 之前）有两个必要性：
    //   1) 掩码随后会把无效列覆盖成 -inf，故 scale 尾部取值不影响结果；
    //   2) softmax 看到的是完整的 true_score，导出的 LSE 与原实现一致 —— DCP 的
    //      跨 rank merge 依赖它，若放到 softmax 之后就会错。
    if (constInfo.keyQuantMode == QUANT_MODE::TQ4) {
        LocalTensor<T> qsfaColScale = tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32];
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < dealRowCount; ++r) {
            Mul(mmResUb[r * columnCount], mmResUb[r * columnCount], qsfaColScale, columnCount);
        }
        PipeBarrier<PIPE_V>();
    }
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判断
        uint64_t qsfaS2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t qsfaS2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t qsfaS2ProcessSize = info.actualSingleProcessSInnerSize;
        int64_t qsfaS2Pair = CeilDiv(qsfaS2ProcessSize, 2L * constInfo.sparseBlockSize);
        int64_t qsfaS2Mid = CeilDiv(qsfaS2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        if (qsfaS2Mid > qsfaS2ProcessSize) {
            qsfaS2Mid = qsfaS2ProcessSize;
        }
        if (unlikely(qsfaS2ValidSizeFirstPart < qsfaS2Mid)) {
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2ValidSizeFirstPart, 8);
            int64_t qsfaS2MidFloorAlign = qsfaS2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, dealRowCount, columnCount, qsfaS2ValidSizeFirstPart,
                        qsfaS2StartCeilAlign >= qsfaS2Mid ? qsfaS2Mid : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, qsfaS2StartCeilAlign, qsfaS2MidFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        qsfaS2StartCeilAlign <= qsfaS2MidFloorAlign ? qsfaS2MidFloorAlign : qsfaS2StartCeilAlign,
                        qsfaS2Mid);
        }
        if (unlikely(qsfaS2ValidSizeSecondPart < qsfaS2ProcessSize - qsfaS2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2Mid + qsfaS2ValidSizeSecondPart, 8);
            int64_t qsfaS2EndFloorAlign = qsfaS2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, qsfaS2Mid + qsfaS2ValidSizeSecondPart,
                        qsfaS2StartCeilAlign >= qsfaS2ProcessSize ? qsfaS2ProcessSize : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, qsfaS2StartCeilAlign, qsfaS2EndFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        qsfaS2StartCeilAlign <= qsfaS2EndFloorAlign ? qsfaS2EndFloorAlign : qsfaS2StartCeilAlign,
                        qsfaS2ProcessSize);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                                             uint32_t columnCount, uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t qsfaStartFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t qsfaNotComputePreMaskOneBlk = (1 << (startId - qsfaStartFloorAlignSize)) - 1;
    uint64_t qsfaNotComputePostMaskOneBlk = ~((1 << (endId - qsfaStartFloorAlignSize)) - 1);
    uint64_t qsfaNotComputeMaskOneBlk = qsfaNotComputePreMaskOneBlk ^ qsfaNotComputePostMaskOneBlk;

    uint64_t qsfaMaskOneBlk = ~qsfaNotComputeMaskOneBlk;
    uint64_t mask[1] = {qsfaMaskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (qsfaMaskOneBlk << (i * 8));
    }
    for (uint64_t qsfaRowId = 0; qsfaRowId < dealRowCount; qsfaRowId += 8) {
        Duplicate(mmResUb[qsfaRowId * columnCount + qsfaStartFloorAlignSize], SOFTMAX_MIN_NUM, mask, 1,
                  CeilDiv(columnCount, 8), 0);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t qsfaRowId = 0; qsfaRowId < dealRowCount; qsfaRowId++) {
        Duplicate(mmResUb[qsfaRowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (actualColumnCount != 0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, QSFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCount);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCount);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t loopId)
{
    uint32_t qsfaComputeSize = dealRowCount * columnCount;
    uint64_t qsfaInOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                                 (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> qsfaMmResUb = inputBuff1.Get<MM1_OUT_T>();
    qsfaMmResUb = qsfaMmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(qsfaMmResUb, mm1ResGm[qsfaInOutGmOffset], qsfaComputeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, qsfaMmResUb, dealRowCount, columnCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> qsfaTmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> qsfaSoftmaxTmpUb = qsfaTmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, qsfaMmResUb, qsfaSoftmaxTmpUb, startRow, dealRowCount, columnCount,
                          info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    // [TQ4] V 侧：MLA 中 K=V，score 侧已按列乘过 s_j，输出侧还需再乘一次 ——
    //   out[i] = Σ_j P[i][j]·(s_j·ŷ_j) = Σ_j (P[i][j]·s_j)·ŷ_j
    // 即“缩放 V 的行”等价于“缩放 P 的列”。必须在 softmax 之后，
    // 否则会改变 softmax 的输入分布（LSE 也就跟着错，DCP 的 merge 会崩）。
    if (constInfo.keyQuantMode == QUANT_MODE::TQ4) {
        LocalTensor<T> qsfaScaleF = tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32];
        for (uint32_t r = 0; r < dealRowCount; ++r) {
            Mul(qsfaMmResUb[r * columnCount], qsfaMmResUb[r * columnCount], qsfaScaleF, columnCount);
        }
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<K_ROPE_T> tmpMMResCastTensor = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, qsfaMmResUb, AscendC::RoundMode::CAST_ROUND, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[qsfaInOutGmOffset], tmpMMResCastTensor, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t qsfaMSplitSize = info.actualSingleProcessSInnerSize == 0 ?
                                  16 :
                                  (BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign);
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    qsfaMSplitSize = qsfaMSplitSize >> 3U << 3U;

    if (qsfaMSplitSize > mSplitInfo.vecDealM) {
        qsfaMSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t qsfaLoopCount = (mSplitInfo.vecDealM + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mSplitInfo.vecDealM - (qsfaLoopCount - 1) * qsfaMSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)], dataCopyParams,
                    padParams);
        // [TQ4 scale 传递] 与 validSize 同处读回，沿用同一 MTE2_S 同步点。
        // 一次取整槽（两个 AIV 各写的一半都在内），与 vec0 的分区写严格配对。
        if (constInfo.keyQuantMode == QUANT_MODE::TQ4) {
            DataCopyExtParams qsfaScaleInParams;
            qsfaScaleInParams.blockCount = 1;
            qsfaScaleInParams.blockLen = TQ4_SCALE_SLOT_STRIDE * sizeof(uint16_t);
            qsfaScaleInParams.srcStride = 0;
            qsfaScaleInParams.dstStride = 0;
            DataCopyPadExtParams<uint16_t> qsfaScalePad{false, 0, 0, 0};
            DataCopyPad(tq4ScaleBuf_.Get<uint16_t>(),
                        tq4ScaleGm_[TQ4_SCALE_HALF_BASE + info.loop % MERGE_CACHE_GM_BUF_NUM * TQ4_SCALE_SLOT_STRIDE],
                        qsfaScaleInParams, qsfaScalePad);
            // [TQ4] scale 是 MTE2 刚搬进来的，而下面的 Cast 是向量读 —— 原代码只建立了
            // MTE2->S（它只用标量读 validSize），向量侧没有任何同步。缺这一条在 910B4 上
            // 不是静默算错而是直接抛 507015，实测 nq>=2（环形槽开始轮转）必崩。
            SetFlag<HardEvent::MTE2_V>(TQ4_SCALE_SYNC_FLAG);
            WaitFlag<HardEvent::MTE2_V>(TQ4_SCALE_SYNC_FLAG);
            // 展开成 fp32 供 score 与 P 两处按列施加。scale 在整个 buffer 内不变，
            // 故每 buffer 只做一次（原先放在 DealBmm1ResBaseBlock 里，每个 M 分块重复一次）。
            // 尾部（对齐补出来的列）的 1.0 已在 InitBuffers 一次性填好，此处不再触碰。
            {
                LocalTensor<half> qsfaScaleH = tq4ScaleBuf_.Get<half>();
                LocalTensor<T> qsfaScaleF = tq4ScaleBuf_.Get<T>()[TQ4_SCALE_UB_F32];
                uint32_t qsfaValidCol = info.actualSingleProcessSInnerSize;
                // 只覆盖有效前缀，起点恒为 0；尾部保持 InitBuffers 里填好的 1.0。
                // 长度为 0 时整段跳过：向量指令元素数为 0 非法。
                if (qsfaValidCol > 0) {
                    Cast(qsfaScaleF, qsfaScaleH, AscendC::RoundMode::CAST_NONE, qsfaValidCol);
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(qsfaLoopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进行
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, qsfaI * qsfaMSplitSize, dealSize,
                             info.actualSingleProcessSInnerSizeAlign, qsfaI);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t qsfaTopkGmIdx = (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(qsfaTopkGmIdx >= constInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + qsfaTopkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

template <typename QSFAT>
__aicore__ inline int64_t QSFAVectorService<QSFAT>::GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo,
                                                                    int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        realKeyBNBOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                               static_cast<int64_t>(constInfo.kvCacheBlockSize) *
                               static_cast<int64_t>(constInfo.kvHeadNum) +
                           blkTableOffset;
    } else {
        realKeyBNBOffset = (runInfo.tensorBOffset + realS2Idx * constInfo.kvHeadNum * constInfo.combineHeadDim) /
                           constInfo.combineHeadDim;
    }
    return realKeyBNBOffset;
}

// [TQ4] Phase B: codebook dequant of `dealRow` combined slots (int4 nope + rope + 2B scale)
// -> dstB16 [dealRow,headDim] bf16 (Hadamard-space K=V). bf16 path: 32B-burst compact
// the packed bytes then one byte-LUT Gather per byte (uint32 carries 2 centroids);
// other dtypes: per-nibble int4b_t Cast/index/Gather/output Cast in 4-row chunks.
// The 2B slot scale is expected to be vecNorm/sqrt(Sum c^2) on the fused-SFA write path.
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Tq4DequantRows(LocalTensor<KV_T> &srcTensor,
                                                                LocalTensor<K_ROPE_T> &dstB16, int32_t dealRow)
{
    uint32_t HD = constInfo.headDim;
    uint32_t ROW_BYTES =
        QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput), static_cast<uint32_t>(BYTE_BLOCK));
    constexpr int32_t CHUNK = TQ4_DEQUANT_CHUNK;
    constexpr uint32_t CHUNK_ELEMS = CHUNK * 512; // 慢路径按 nibble 计（HD=512）
    // [TQ4] bf16 快路径 scratch：compact(CHUNK×256B) + half(CHUNK×256×2B) +
    // idx(CHUNK×256×4B)，无 float work 区（byte-LUT Gather 直写 dst 的 uint32
    // 视图，每字节一个表项）；慢路径保留原布局（float work + half + idx，按
    // nibble 计），故 chunk 上限不同（见 TQ4_DEQUANT_CHUNK）。
    constexpr bool TQ4_FAST_BF16 = IsSameType<K_ROPE_T, bfloat16_t>::value;
    constexpr uint32_t CHUNK_BYTES = CHUNK * 256U; // 每行打包字节数（HD/2=256）
    constexpr uint32_t HALF_ELEMS = TQ4_FAST_BF16 ? CHUNK_BYTES : CHUNK_ELEMS;
    constexpr uint32_t IDX_ELEMS = TQ4_FAST_BF16 ? CHUNK_BYTES : CHUNK_ELEMS;
    constexpr uint32_t COMPACT_BYTES = TQ4_FAST_BF16 ? CHUNK_BYTES : 0U;
    constexpr uint32_t SHALF_BYTE_OFF = TQ4_FAST_BF16 ? COMPACT_BYTES : CHUNK_ELEMS * sizeof(float);
    constexpr uint32_t IDX_BYTE_OFF = SHALF_BYTE_OFF + HALF_ELEMS * sizeof(half);
    static_assert(IDX_BYTE_OFF + IDX_ELEMS * sizeof(int32_t) <= ConstInfo::BUFFER_SIZE_BYTE_32K,
                  "TQ4 dequant scratch exceeds inputBuff2");

    LocalTensor<half> sHalfBase = inputBuff2.Get<half>()[SHALF_BYTE_OFF / sizeof(half)];
    LocalTensor<int32_t> idxI = inputBuff2.Get<int32_t>()[IDX_BYTE_OFF / sizeof(int32_t)];
    LocalTensor<uint32_t> idxU = inputBuff2.Get<uint32_t>()[IDX_BYTE_OFF / sizeof(uint32_t)];
    LocalTensor<float> centBuf = tq4Cent_;
    LocalTensor<int4b_t> srcI4 = srcTensor.template ReinterpretCast<int4b_t>();

    PipeBarrier<PIPE_ALL>();
    if (unlikely(dealRow <= 0)) {
        return;
    }

    // per-row 的 s_t 不在此处施加：MLA 里 K=V，缩放 K 的行等价于缩放 score/P 的列，
    // 故改到 attention 侧按列做一次（见 ElewiseCompute 与 DealBmm1ResBaseBlock）。
    // 省掉的是 dealRow 条 512 宽的 Muls，单算子实测约 3%。
    if constexpr (TQ4_FAST_BF16) {
        // [TQ4] 每块四条向量指令（摊到每 16 行四条）：
        //   1) V 侧 32B-burst 拷贝把 CHUNK 行的 256B nibble 区按 416B 行距抽紧成
        //      连续字节（uint8 视图寻址）；
        //   2) 两条 Cast：u8->half（值域 0..255 在 half 精确）再 ->int32 —— Gather
        //      的偏移张量形参只认 uint32（dav_m200/kernel_operator_vec_gather_impl.h），
        //      且无 u8->i32 直通组合（vconv 支持表），u8 必须经 half 中转；
        //   3) ShiftLeft <<2：字节下标 -> uint32 表项字节地址 [0, 1020]（×4 不能
        //      烙进标量 baseOffset，它是全体元素共用的常数）；
        //   4) Gather 直查 256 项 byte-LUT，uint32 直写 dst —— 一次携带相邻两维
        //      的 bf16 质心位型，每字节一次 Gather。
        // 向量指令 0.25/行，barrier 5/16行。
        LocalTensor<uint8_t> compactU8 = inputBuff2.Get<uint8_t>();
        LocalTensor<uint32_t> byteLut = tq4ByteLutBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> dstU32 = dstB16.template ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> srcU8 = srcTensor.template ReinterpretCast<uint8_t>();
        // 行距 416B=13×32B、nibble 区 256B=8×32B 均为 32B 整数倍（QSFAAlign 保证）
        uint16_t nibbleBlk = static_cast<uint16_t>((HD / 2) / BYTE_BLOCK);
        uint16_t rowGapBlk = static_cast<uint16_t>(ROW_BYTES / BYTE_BLOCK - nibbleBlk);
        for (int32_t base = 0; base < dealRow; base += CHUNK) {
            int32_t cur = (base + CHUNK <= dealRow) ? CHUNK : (dealRow - base);
            uint32_t cnt = static_cast<uint32_t>(cur) * (HD / 2U); // 每字节一个 uint32 输出
            DataCopyParams compactParams;
            compactParams.blockCount = static_cast<uint16_t>(cur);
            compactParams.blockLen = nibbleBlk;
            compactParams.srcStride = rowGapBlk;
            compactParams.dstStride = 0;
            DataCopy(compactU8, srcU8[base * ROW_BYTES], compactParams);
            PipeBarrier<PIPE_V>();
            Cast(sHalfBase, compactU8, RoundMode::CAST_NONE, cnt); // byte value -> half, 0..255
            PipeBarrier<PIPE_V>();
            Cast(idxI, sHalfBase, RoundMode::CAST_ROUND, cnt);
            PipeBarrier<PIPE_V>();
            ShiftLeft(idxI, idxI, static_cast<int32_t>(2), cnt); // -> uint32 表字节地址 [0, 1020]
            PipeBarrier<PIPE_V>();
            Gather(dstU32[base * (HD / 2U)], byteLut, idxU, 0, cnt); // non-negative offsets only (base 0)
            PipeBarrier<PIPE_V>();
        }
    } else {
        LocalTensor<float> workBase = inputBuff2.Get<float>();
        for (int32_t base = 0; base < dealRow; base += CHUNK) {
            int32_t cur = (base + CHUNK <= dealRow) ? CHUNK : (dealRow - base);
            uint32_t cnt = static_cast<uint32_t>(cur) * HD;

            for (int32_t rr = 0; rr < cur; ++rr) {
                int32_t r = base + rr;
                Cast(sHalfBase[rr * HD], srcI4[r * ROW_BYTES * 2], RoundMode::CAST_NONE, HD); // int4b -> half, -8..7
            }
            PipeBarrier<PIPE_V>();
            Adds(sHalfBase, sHalfBase, static_cast<half>(8.0f), cnt); // signed nibble +8 -> [0, 15]
            PipeBarrier<PIPE_V>();
            Muls(sHalfBase, sHalfBase, static_cast<half>(4.0f), cnt); // *4 -> byte offset [0, 60]
            PipeBarrier<PIPE_V>();
            Cast(idxI, sHalfBase, RoundMode::CAST_ROUND, cnt);
            PipeBarrier<PIPE_V>();
            Gather(workBase, centBuf, idxU, 0, cnt); // non-negative offsets only (base 0)
            PipeBarrier<PIPE_V>();
            Cast(dstB16[base * HD], workBase, RoundMode::CAST_ROUND, cnt);
            PipeBarrier<PIPE_V>();
        }
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size,
                                                                int64_t mergeMte3Idx, int64_t realS2Idx,
                                                                int64_t keyBNBOffset, int64_t s2IdLimit,
                                                                const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        ((realS2Idx + constInfo.sparseBlockSize > s2IdLimit) ? (s2IdLimit - realS2Idx) : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;

    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        // [TQ4] slot = headDim/2 int4 nope + headDimRope*sizeof(K_ROPE_T) + 2B vecNorm(fp16)
        uint32_t combineBytes = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T) + sizeof(half);
        intriParams.blockLen = combineBytes;
        uint32_t combineDim = combineBytes / sizeof(KV_T);
        uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign - combineDim;
        padParams.paddingValue = 0;
        DataCopyPad(
            kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * combineDimAlign],
            keyGm_[keyBNBOffset * combineDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit = runInfo.curActualSeqLenOri - runInfo.actS1Size + runInfo.gS1Idx / constInfo.gSize + 1;
    }

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) : (keyBNBOffset2 - keyBNBOffset1)) -
         constInfo.sparseBlockSize);
    // [TQ4] slot = headDim/2 int4 nope + headDimRope*sizeof(K_ROPE_T) + 2B vecNorm(fp16)
    uint32_t combineBytes = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T) + sizeof(half);
    int64_t keySrcStride = sparseBlockSrcStride * combineBytes;
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 || realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit) ||
        constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = constInfo.sparseBlockSize * combineBytes;
            uint32_t combineDim = combineBytes / sizeof(KV_T);
            uint32_t combineDimAlign = CeilAlign(combineBytes, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign - combineDim;
            padParams.paddingValue = 0;
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) +
                                  (mte2Size - mte3Size) * combineDimAlign],
                        keyGm_[startGmOffset * combineDim], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    LocalTensor<K_ROPE_T> antiKvTensorAsB16 = tmpBuff1.Get<K_ROPE_T>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    uint32_t qsfaRopeByteOff = constInfo.headDim * sizeof(KV_T);
    uint32_t slotBytes = static_cast<uint32_t>(tilingData->baseParams.dSizeVInput);
    uint8_t qsfaRopeRowStrideBlk =
        static_cast<uint8_t>(QSFAAlign(slotBytes, static_cast<uint32_t>(BYTE_BLOCK)) / BYTE_BLOCK);
    uint64_t mergeGmStride = 512 * constInfo.combineHeadDim;
    if (constInfo.keyQuantMode == QUANT_MODE::TQ4) {
        // tmpBuff1 / tmpBuff2 是跨批次共享的临时 UB（非 ping-pong），而本函数在
        // MergeKv 的循环里被多次调用。V 流水写入它们之前必须等上一批的 MTE3 读完，
        // 否则下一批 V 会覆盖仍被 MTE3 读取的数据；编译已关闭自动同步
        // （--cce-auto-sync=off），且 PipeBarrier 只解决同流水依赖，跨流水依赖必须
        // 用 SetFlag/WaitFlag。MTE3_V(SYNC_OUTPUT_BUF1_FLAG) 在 Init 处置位、在
        // 结束处等待，此处 Wait/Set 成对出现，总数保持平衡。
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        // [TQ4] codebook dequant -> antiKvTensorAsB16 [dealRow,headDim] bf16
        Tq4DequantRows(srcTensor, antiKvTensorAsB16, dealRow);
        qsfaRopeByteOff = constInfo.headDim / 2;

        // [TQ4 scale 传递] 把本批 dealRow 行的 2B scale 收集后写入 GM 的 scale 区。
        // 列号 = s2GmStartOffset + mte3Size + rr，与 kvMergeGm_ 的行号严格对齐。
        // 位置与 kvMergeGm_ 的写出同处一函数、同一 MTE3 队列，沿用其同步；
        // 不像 O8 另开通道，故不存在“两条流水各自推进”的窗口。
        {
            uint32_t qsfaScaleByteOff = constInfo.headDim / 2 + constInfo.headDimRope * sizeof(K_ROPE_T);
            uint32_t qsfaSlotRowBlk = QSFAAlign(static_cast<uint32_t>(tilingData->baseParams.dSizeVInput),
                                                static_cast<uint32_t>(BYTE_BLOCK)) /
                                      BYTE_BLOCK;
            // [TQ4] 向量化导出：每 token 读 1 个 32B block、跳过 (rowBlk-1) 个 block，
            // 得到 [dealRow, 32B]；再用预置索引 (i*32) Gather 出每行头 2B，凑成连续向量。
            // 两条向量指令替掉 dealRow 次标量读写。
            LocalTensor<half> qsfaSTUb = tmpBuff2.Get<half>();
            LocalTensor<half> qsfaSTUb32 = tmpBuff2.Get<half>()[512];
            LocalTensor<half> qsfaSrcHalf = srcTensor.template ReinterpretCast<half>()[qsfaScaleByteOff / 2];
            DataCopyParams qsfaSTGathParams;
            qsfaSTGathParams.blockCount = static_cast<uint16_t>(dealRow);
            qsfaSTGathParams.blockLen = 1;
            qsfaSTGathParams.srcStride = static_cast<uint16_t>(qsfaSlotRowBlk - 1);
            qsfaSTGathParams.dstStride = 0;
            DataCopy(qsfaSTUb32, qsfaSrcHalf, qsfaSTGathParams);
            PipeBarrier<PIPE_V>();
            LocalTensor<uint32_t> qsfaSTIdx = tq4STIdxBuf_.Get<uint32_t>();
            Gather(qsfaSTUb, qsfaSTUb32, qsfaSTIdx, 0, static_cast<uint32_t>(dealRow));
            SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
            LocalTensor<uint16_t> qsfaScaleSrc = qsfaSTUb.template ReinterpretCast<uint16_t>();
            DataCopyExtParams qsfaScaleOutParams;
            qsfaScaleOutParams.blockCount = 1;
            qsfaScaleOutParams.blockLen = static_cast<uint32_t>(dealRow) * sizeof(uint16_t);
            qsfaScaleOutParams.srcStride = 0;
            qsfaScaleOutParams.dstStride = 0;
            DataCopyPad(
                tq4ScaleGm_[TQ4_SCALE_HALF_BASE + runInfo.loop % MERGE_CACHE_GM_BUF_NUM * TQ4_SCALE_SLOT_STRIDE +
                            (s2GmStartOffset + mte3Size)],
                qsfaScaleSrc, qsfaScaleOutParams);
        }

        DataCopyExtParams tq4DataCopyParams;
        tq4DataCopyParams.blockCount = static_cast<uint16_t>(dealRow);
        tq4DataCopyParams.blockLen = constInfo.headDim * sizeof(K_ROPE_T);
        tq4DataCopyParams.srcStride = 0;
        tq4DataCopyParams.dstStride = (constInfo.combineHeadDim - constInfo.headDim) * sizeof(K_ROPE_T);
        uint64_t tq4GmBase = runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride +
                             (s2GmStartOffset + mte3Size) * constInfo.combineHeadDim;
        DataCopyPad(kvMergeGm_[tq4GmBase], antiKvTensorAsB16, tq4DataCopyParams);

        LocalTensor<K_ROPE_T> tq4KRopeUb = srcTensor[qsfaRopeByteOff].template ReinterpretCast<K_ROPE_T>();
        tq4DataCopyParams.blockLen = constInfo.headDimRope * sizeof(K_ROPE_T);
        // DataCopyPad UB-side srcStride is in 32B datablocks, not bytes (cf. sparse_flash_attention
        // CopyOutMrgeResult). The rope row pitch is qsfaRopeRowStrideBlk blocks; subtract blockLen's blocks.
        tq4DataCopyParams.srcStride =
            static_cast<uint32_t>(qsfaRopeRowStrideBlk) - constInfo.headDimRope * sizeof(K_ROPE_T) / BYTE_BLOCK;
        tq4DataCopyParams.dstStride = (constInfo.combineHeadDim - constInfo.headDimRope) * sizeof(K_ROPE_T);
        DataCopyPad(kvMergeGm_[tq4GmBase + constInfo.headDim], tq4KRopeUb, tq4DataCopyParams);
        // 本批三次 DataCopyPad 已发出，释放 tmpBuff1 / tmpBuff2 供下一批 V 复用。
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        return;
    }
}

// b s1 k
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::MergeKv(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t qsfaActualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(runInfo.bIdx - 1);
        topkGmBaseOffset += (qsfaActualSeqQPrefixSum + runInfo.gS1Idx / constInfo.gSize) * constInfo.kvHeadNum *
                                constInfo.sparseBlockCount +
                            runInfo.n2Idx * constInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += runInfo.bIdx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            runInfo.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
    }
    int64_t qsfaMergeMte3Idx = 0;
    int64_t qsfaMte2Size = 0;
    int64_t qsfaMte3Size = 0;
    int64_t qsfaS2IdxArray0 = -1;
    int64_t qsfaS2IdxArray1 = -1;
    bool qsfaNeedWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    int64_t qsfaS2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t qsfaS2GmLimit = GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
    if (qsfaS2GmLimit > s2ProcessSize) {
        qsfaS2GmLimit = s2ProcessSize;
    }
    for (int64_t s2GmOffsetArray = qsfaS2GmStartOffset; s2GmOffsetArray < qsfaS2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (qsfaNeedWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaNeedWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, qsfaS2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(qsfaS2IdxArray0 < 0)) {
            CopyOutMrgeResult(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaMergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, qsfaS2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(qsfaMte2Size, qsfaMte3Size, qsfaMergeMte3Idx, qsfaS2IdxArray0, qsfaS2IdxArray1, runInfo);
        if ((qsfaMte2Size - qsfaMte3Size + 2 * constInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= qsfaS2GmLimit) {
            CopyOutMrgeResult(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx, runInfo);
            qsfaMte3Size = qsfaMte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % 2);
            qsfaMergeMte3Idx++;
            qsfaNeedWaitMte3ToMte2 = true;
        }
    }

    if (unlikely(qsfaS2GmStartOffset + qsfaMte2Size < qsfaS2GmLimit)) {
        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx & 1);
        LocalTensor<K_ROPE_T> mergeUb = kvMergUb_.template ReinterpretCast<K_ROPE_T>();
        Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        uint64_t mergeGmStride = 512 * constInfo.combineHeadDim;
        if (constInfo.keyQuantMode == QUANT_MODE::TQ4) {
            // [TQ4] MergeKv lays a row out at row * combineHeadDim with nope and rope contiguous.
            // The c8 fill in the else branch addresses a different layout and would zero live rows.
            Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), constInfo.combineHeadDim);
            SetFlag<AscendC::HardEvent::V_MTE3>(0);
            WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            dataCopyParams.blockCount = 1;
            dataCopyParams.blockLen = constInfo.combineHeadDim * sizeof(K_ROPE_T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride +
                                       s2GmOffset * constInfo.combineHeadDim],
                            mergeUb, dataCopyParams);
            }
        } else {
            dataCopyParams.blockCount = constInfo.headDim / blockElementNum;
            dataCopyParams.blockLen = blockElementNum * sizeof(K_ROPE_T);
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = (constInfo.s2BaseSize - 1) * blockElementNum * sizeof(K_ROPE_T);
            for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
                DataCopyPad(
                    kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride + s2GmOffset * blockElementNum],
                    mergeUb, dataCopyParams);
            }
            dataCopyParams.blockCount = constInfo.headDimRope / blockElementNum;
            for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * mergeGmStride + 512 * constInfo.headDim +
                                       s2GmOffset * blockElementNum],
                            mergeUb, dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx & 1);
        qsfaMergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, qsfaMte2Size);
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

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t qsfaNBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = info.actMBaseSize - (qsfaNBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = qsfaI;
        mSplitInfo.nBufferStartM = qsfaI * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (qsfaI + 1 != qsfaNBufferLoopTimes) ? constInfo.nBufferMBaseSize : qsfaNBufferTail;

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
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1 && (constInfo.returnSoftmaxLse || info.tndIsS2SplitCore)) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            if (constInfo.returnSoftmaxLse) {
                CopyFALseToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename QSFAT>
__aicore__ inline uint64_t QSFAVectorService<QSFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (gPreSplitSize > mSplitInfo.vecDealM) {
        gPreSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * gPreSplitSize, dealSize, constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (mSplitInfo.vecStartM + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        uint64_t vecPre2ResGmOffset =
            ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<T> softmaxExpBrcb = tmpBuff2.Get<T>();
        Brcb(softmaxExpBrcb, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
             (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpBrcb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        LocalTensor<T> softmaxSumBrcb = tmpBuff2.Get<T>();
        Brcb(softmaxSumBrcb, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
             (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumBrcb, dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mSplitInfo.vecStartM + startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t vecPre2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t qsfaNBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = info.actMBaseSize - (qsfaNBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = qsfaI;
        mSplitInfo.nBufferDealM = (qsfaI + 1 != qsfaNBufferLoopTimes) ? constInfo.nBufferMBaseSize : qsfaNBufferTail;
        mSplitInfo.nBufferStartM = qsfaI * constInfo.nBufferMBaseSize;

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

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                  uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t qsfaMSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (qsfaMSplitSize > mDealSize) {
        qsfaMSplitSize = mDealSize;
    }

    uint32_t qsfaLoopCount = (mDealSize + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mDealSize - (qsfaLoopCount - 1) * qsfaMSplitSize;
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, qsfaI * qsfaMSplitSize + mStartRow, dealSize, constInfo.headDim,
                             constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                                   uint32_t wsMStart, uint32_t dealRowCount,
                                                                   uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset =
        accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
        info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // 份数offset
        wsMStart * actualColumnCount;                                                                 // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2DataCopyOutTrans(const RunInfo &info,
                                                                      LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                                      uint32_t dealRowCount, uint32_t columnCount,
                                                                      uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                                    uint32_t wsMStart, uint32_t dealRowCount,
                                                                    uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> qsfaTmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(qsfaTmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(qsfaTmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, qsfaTmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                                uint32_t wsMStart, uint32_t dealRowCount,
                                                                uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (!FLASH_DECODE) {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    } else {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub,
                                                         LocalTensor<float> src1Ub, uint32_t dealRowCount,
                                                         uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t qsfaDtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaDLoop = actualColumnCount / qsfaDtypeMask;
    uint32_t qsfaDRemain = actualColumnCount % qsfaDtypeMask;

    BinaryRepeatParams qsfaRepeatParamsDiv;
    qsfaRepeatParamsDiv.src0BlkStride = 1;
    qsfaRepeatParamsDiv.src1BlkStride = 0;
    qsfaRepeatParamsDiv.dstBlkStride = 1;
    qsfaRepeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    qsfaRepeatParamsDiv.src1RepStride = 1;
    qsfaRepeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t qsfaColumnRepeatCount = qsfaDLoop;
    if (qsfaColumnRepeatCount <= dealRowCount) {
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaDtypeMask, dealRowCount, qsfaRepeatParamsDiv);
            qsfaOffset += qsfaDtypeMask;
        }
    } else {
        BinaryRepeatParams qsfaColumnRepeatParams;
        qsfaColumnRepeatParams.src0BlkStride = 1;
        qsfaColumnRepeatParams.src1BlkStride = 0;
        qsfaColumnRepeatParams.dstBlkStride = 1;
        qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        qsfaColumnRepeatParams.src1RepStride = 0;
        qsfaColumnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub[qsfaI * FP32_BLOCK_ELEMENT_NUM], qsfaDtypeMask,
                qsfaColumnRepeatCount, qsfaColumnRepeatParams);
            qsfaOffset += columnCount;
        }
    }
    if (qsfaDRemain > 0) {
        Div(dstUb[qsfaDLoop * qsfaDtypeMask], src0Ub[qsfaDLoop * qsfaDtypeMask], src1Ub, qsfaDRemain, dealRowCount,
            qsfaRepeatParamsDiv);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub,
                                                         LocalTensor<T> src1Ub, uint32_t dealRowCount,
                                                         uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t qsfaDLoop = actualColumnCount / qsfaRepeatElementNum;
    uint32_t qsfaDRemain = actualColumnCount % qsfaRepeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * qsfaBlockElementNum) {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstBlkStride = 1;
        qsfaRepeatParams.src0RepStride = columnCount / qsfaBlockElementNum;
        qsfaRepeatParams.src1RepStride = 1;
        qsfaRepeatParams.dstRepStride = columnCount / qsfaBlockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (qsfaDLoop <= dealRowCount) {
            uint32_t qsfaOffset = 0;
            for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
                Mul(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaRepeatElementNum, dealRowCount,
                    qsfaRepeatParams);
                qsfaOffset += qsfaRepeatElementNum;
            }
        } else {
            BinaryRepeatParams qsfaColumnRepeatParams;
            qsfaColumnRepeatParams.src0BlkStride = 1;
            qsfaColumnRepeatParams.src1BlkStride = 0;
            qsfaColumnRepeatParams.dstBlkStride = 1;
            qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            qsfaColumnRepeatParams.src1RepStride = 0;
            qsfaColumnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
                Mul(dstUb[qsfaI * columnCount], src0Ub[qsfaI * columnCount], src1Ub[qsfaI * qsfaBlockElementNum],
                    qsfaRepeatElementNum, qsfaDLoop, qsfaColumnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (qsfaDRemain > 0) {
            Mul(dstUb[qsfaDLoop * qsfaRepeatElementNum], src0Ub[qsfaDLoop * qsfaRepeatElementNum], src1Ub, qsfaDRemain,
                dealRowCount, qsfaRepeatParams);
        }
    } else {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1RepStride = 0;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstRepStride = 8;
        qsfaRepeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[qsfaI * columnCount], src0Ub[qsfaI * columnCount], src1Ub[qsfaI * qsfaBlockElementNum],
                qsfaRepeatElementNum, qsfaDLoop, qsfaRepeatParams);
            //  计算一行中的尾块
            if (qsfaDRemain > 0) {
                Mul(dstUb[qsfaI * columnCount + qsfaDLoop * qsfaRepeatElementNum],
                    src0Ub[qsfaI * columnCount + qsfaDLoop * qsfaRepeatElementNum], src1Ub[qsfaI * qsfaBlockElementNum],
                    qsfaDRemain, 1, qsfaRepeatParams);
            }
        }
    }
}

#endif // TURBOQUANT_SPARSE_FLASH_ATTENTION_SERVICE_VECTOR_MLA_H
