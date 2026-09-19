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
 * \file flash_attn_block_vec_dn.h
 * \brief FANoQuantGqaBlockVecDn —— Dn 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTN_BLOCK_VEC_DN_H_
#define FLASH_ATTN_BLOCK_VEC_DN_H_

#include <limits>
#include "../../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../a5_mla_common/op_kernel/const_def.h"
#include "../../../a5_mla_common/op_kernel/vector_common.h"
#include "../../../a5_mla_common/op_kernel/init_output.h"
#include "memory_copy_arch35.h"
#include "../utils/attn_sink_gs1.h"
#include "../../../a5_mla_common/op_kernel/arch_info.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

template <typename FA_T>
class FANoQuantGqaBlockVecDn {
public:
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;

    using T = float;
    static constexpr uint32_t dTemplateAlign64 = BaseApi::Align64Func((uint16_t)FA_T::dVBaseSize);

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_MM_0 = 0U;
    static constexpr uint32_t CC_MM_1 = 1U;
    static constexpr uint32_t CC_MM_2 = 2U;
    static constexpr uint32_t CC_MM_3 = 3U;
    static constexpr uint32_t CC_L1P_0 = 5U;
    static constexpr uint32_t CC_L1P_1 = 6U;
    static constexpr uint32_t CC_L1P_2 = 7U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;

    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> l1PBuffers_;

    // UB
    static constexpr uint32_t UB_MM_RES_BUFCNT = (dBaseSize > 128) ? 2U : 4U;
    static constexpr uint32_t UB_MM_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * (s2BaseSize > dVBaseSize ? s2BaseSize : dVBaseSize) * sizeof(T);
    // bmm1/bmm2(mmRes) 区域总字节数：FD block 以此作为 FD 业务区的 UB 起始偏移
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES = UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES;
    // FD 业务区在 UB 中的起始字节偏移，供 kernel 传给 FD block 的 InitBuffers
    static __aicore__ inline constexpr uint32_t GetFdBaseOffset()
    {
        return UB_MM_RES_TOTAL_BYTES;
    }
    LocalTensor<uint8_t> ubMmResBuffers_;
    uint32_t mmResBufId_ = 0;

    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    LocalTensor<T> ubVec2Res_; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (mBaseSize / CV_RATIO + 1U) * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    uint32_t vec1ResUbBufId_ = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxSumBuf_;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxMaxBuf_;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxExpBuf_;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * 8 * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<uint8_t> ubLseOutBuffers_;
    uint32_t lseOutUbBufId_ = 0;

    const ConstInfo_t &constInfo_;

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool_;

    // GM
    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor_;
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<LAYOUT_T>()> copyAttenOutUbToGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    GlobalTensor<float> sinkGm_;

    T negativeFloatScalar_;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVecDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                             SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool)
        : constInfo_(constInfo),
          qSeqLensTool_(qSeqLensTool),
          kvSeqLensTool_(kvSeqLensTool){};

    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSizeV,
                           outGmTensor_, attentionOut);

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if (constInfo_.enableFlashDecode) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
        if (constInfo_.learnableSinkFlag) {
            sinkGm_.SetGlobalBuffer((__gm__ float *)learnableSink);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMmResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                               UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES); // CV通信BUF
        addrUb = UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES;
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_VEC2_RES_BUF_BYTES)
                         .template ReinterpretCast<T>(); // 输出BUF: attn_out拷出
        addrUb += UB_VEC2_RES_BUF_BYTES;
        ubVec1ResBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES); // 2 * 32.25K = 64.5K, 输出BUF: softmax结果拷贝至L1
        addrUb += UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES;

        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES;
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES;
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES;

        ubLseOutBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_LSE_OUT_BUFCNT *
                UB_LSE_OUT_BUF_BYTES); // 输出BUF:
                                       // FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出；按actVecMSize*8(fp32块对齐)定容
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx, const RunInfo &runInfo)
    {
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        if (constInfo_.learnableSinkFlag && runInfo.isFirstFdBlock) {
            Duplicate<T>(sumUb, static_cast<T>(1), softmaxBufElementCount);

            uint32_t gs1Start = runInfo.gS1Idx + runInfo.vecMbaseIdx;

            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();
                if constexpr (LAYOUT_T == FA_LAYOUT::BSND || LAYOUT_T == FA_LAYOUT::TND) {
                    AttentionCommon::SinkCopyInS1G(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                                   runInfo.n2Idx, constInfo_.gSize);
                } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    AttentionCommon::SinkCopyInGS1(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                                   runInfo.n2Idx, constInfo_.gSize);
                }
            }
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();

                if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
                    AttentionCommon::SinkExpandMaxVf<T, float, true>(maxUb, sinkTmpUb, gs1Start, runInfo.actVecMSize,
                                                                     runInfo.actS1Size, constInfo_.gSize);
                } else {
                    AttentionCommon::SinkExpandMaxVf<T, float, false>(maxUb, sinkTmpUb, gs1Start, runInfo.actVecMSize,
                                                                      runInfo.actS1Size, constInfo_.gSize);
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        } else {
            Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
            Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_0);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_1);
        if constexpr (dBaseSize <= 128) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_2);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM_3);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void ProcessVec1(RunInfo runInfo)
    {
        uint32_t mmResUbBufId = mmResBufId_;
        mmResBufId_ = (mmResBufId_ + 1) % UB_MM_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMmResBuffers_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT, runInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        ProcessVec1Dn(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知BMM2: Vec1已读完mmRes, 可覆写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
        Vec1PostProcess(runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfo_.needInitOutput) {
            uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum;
            uint64_t tSize = constInfo_.bSize * constInfo_.s1Size;
            if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                tSize = qSeqLensTool_.cuSeqLensParser.GetTSize();
            }
            uint64_t attenOutTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize * constInfo_.dSizeV;

            static constexpr OUTPUT_T ATTEN_OUT_INIT_VAL = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(OUTPUT_T);
            AttentionCommon::InitOutput<OUTPUT_T, EVENT_ID0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE,
                                        true>(attentionOutGm_, attenOutTotalSize, vecCoreNum, ATTEN_OUT_INIT_VAL);

            if (constInfo_.isSoftmaxLseEnable) {
                uint64_t lseTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize;

                static constexpr float LSE_INIT_VAL = 3e+99;
                static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_32K;
                static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
                    softmaxLseGm_, lseTotalSize, vecCoreNum, LSE_INIT_VAL);
            }

            SyncAll();
        }
    }

    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut &outGmTensor, __gm__ uint8_t *gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T *)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool_.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfo runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (constInfo_.enableFlashDecode) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if (constInfo_.enableFlashDecode) {
            if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
            uint32_t prefixBS1 = qSeqLensTool_.cuSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = runInfo.n2Idx * constInfo_.gSize * constInfo_.t1Size + prefixBS1;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfo_t>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx,
                                                            runInfo.actVecMSize, constInfo_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BSND) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfo_t>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                         constInfo_);
        } else if constexpr (LAYOUT_T == FA_LAYOUT::BNSD) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                                 runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size;
            uint64_t qActSeqLens = qSeqLensTool_.seqUsedParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfo_t>(softmaxLseGm_, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                         constInfo_, qActSeqLens);
        }
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         RunInfo runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize / CV_RATIO;
        static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        static constexpr uint32_t vec1S2CopyCountDn = (mBaseSize >> 4) / CV_RATIO;
        static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
        static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;

        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        float descaleQK = 1.0;

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb, runInfo.actMSizeAlign32 >> 1,
            runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size, static_cast<T>(constInfo_.scaleValue),
            descaleQK, negativeFloatScalar_, 0.0F, false);

        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;

        if (runInfo.actSingleLoopS2Size > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign +
                                  vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - runInfo.actSingleLoopS2SizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.actSingleLoopS2SizeAlign + 1), 0});
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        vec1ResUbBufId_ = (vec1ResUbBufId_ + 1U) % UB_VEC1_RES_BUFCNT;
    }

    __aicore__ inline void Vec1PostProcess(RunInfo runInfo)
    {
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.n2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo_.dSizeV};
        copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfo &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        LocalTensor<float> sumBrdcstBuf =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(sumBrdcstBuf, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDSumGm_[gmOffset], sumBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1U) % UB_LSE_OUT_BUFCNT;

        LocalTensor<float> maxBrdcstBuf =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(maxBrdcstBuf, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1U) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfo &runInfo, LocalTensor<T> &ubVec2Res, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm_[gmOffset], ubVec2Res, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(RunInfo runInfo)
    {
        uint32_t mmResUbBufId = mmResBufId_;
        mmResBufId_ = (mmResBufId_ + 1) % UB_MM_RES_BUFCNT;
        uint32_t mmSyncIdx = CC_MM_0 + mmResUbBufId;
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
            return;
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        {
            Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            LocalTensor<T> mm2ResUbTensor =
                ubMmResBuffers_[mmResUbBufId * UB_MM_RES_BUF_BYTES].template ReinterpretCast<T>();
            if (unlikely(runInfo.isFirstS2Loop)) {
                uint32_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
                DataCopy(ubVec2Res_, mm2ResUbTensor, vec2CalcSize);
            } else {
                LocalTensor<T> expUb =
                    softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                LocalTensor<T> pScaleUb;

                float deSCalePreVValue = 1.0f;
                if (!runInfo.isLastS2Loop) {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2Res_, mm2ResUbTensor, ubVec2Res_, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        1.0, 1.0);
                } else {
                    LocalTensor<float> sumUb = softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) *
                                                              (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        ubVec2Res_, mm2ResUbTensor, ubVec2Res_, expUb, pScaleUb, sumUb, runInfo.actVecMSize,
                        dTemplateAlign64, 1.0, 1.0);
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        }
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知下个BMM1: Vec2已读完mmRes, slot空闲

        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                LocalTensor<float> sumUb =
                    softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    ubVec2Res_, ubVec2Res_, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, 0.0F);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            }
            uint32_t mStartVec = 0;
            uint32_t mDealSize = runInfo.actVecMSize;
            if (constInfo_.enableFlashDecode && runInfo.isS2SplitCore) {
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2ResForFDCopyOut(runInfo, ubVec2Res_, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                LocalTensor<OUTPUT_T> attenOut;
                int64_t dSizeAligned64 = (int64_t)dVBaseSize;

                attenOut.SetAddr(ubVec2Res_.address_);
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Cast(attenOut, ubVec2Res_, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);

                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            }
        }
    }
};

// AIC/AIV 分编译占位（Mix kernel 在 AIC 侧重编译时使用）
template <typename FA_T>
class FANoQuantGqaBlockVecDummyDn {
public:
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    using SEQLEN_T = uint32_t;

    __aicore__ inline FANoQuantGqaBlockVecDummyDn(ConstInfo_t &constInfo, SeqLensTool<LAYOUT_T, SEQLEN_T> &qSeqLensTool,
                                                  SeqLensTool<LAYOUT_KV, SEQLEN_T> &kvSeqLensTool){};
};

} // namespace FlashAttnKernel
#endif // FLASH_ATTN_BLOCK_VEC_DN_H_
