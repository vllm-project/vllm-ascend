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
 * \file fia_block_vec_fullquant_gqa.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_FULLQUANT_GQA_H_
#define FIA_BLOCK_VEC_FULLQUANT_GQA_H_

#include "kernel_operator.h"

#include "flash_attention_score_common_regbase.h"
#include "adv_api/activation/softmax.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "vf/vf_flashupdate_new.h"
#include "vf/vf_div_cast.h"
#include "vf/vf_flash_decode.h"
#include "fia_public_define_arch35.h"
#include "../vector_common.h"
#include "../memory_copy_arch35.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

/* ============确定bmm2ResBuffer的类型============= */
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type =
        std::conditional_t<(useDn && isFp8), BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                           BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};

namespace BaseApi {

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantGqaBlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ScmBlock = mBaseSize * 8;
    static constexpr uint32_t vec1ScmBlockFp32 = mBaseSize * 4;
    static constexpr uint32_t vec1ScmBlockFp8 = mBaseSize * 16;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse，刷无效行会用到剩余ub，需要加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT = (layout == LayOutTypeEnum::LAYOUT_BSH ||
        layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_SBH) ? MaskFormat::SG : MaskFormat::GS;

    static constexpr bool USE_DN = useDn;
    static constexpr uint8_t KV_LAYOUT = 4; // 4: K与K_Scale在同一物理内存中交叉排列
    static constexpr bool IS_PER_TOEKN_HEAD = true;
    static constexpr bool PAGE_ATTENTION = (KvLayoutType > 0);

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN, IS_PER_TOEKN_HEAD>();
    static constexpr GmFormat K_SCALE_FORMAT = GetKeyScaleGmFormat<layout, KV_LAYOUT, PAGE_ATTENTION>();
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using attenMaskGmType = typename std::conditional<HAS_MASK, GlobalTensor<uint8_t>, int8_t>::type;
    using postQuantGmType = typename std::conditional<POST_QUANT, GlobalTensor<float>, int8_t>::type;
    using postQuantBf16GmType = typename std::conditional<POST_QUANT, GlobalTensor<bfloat16_t>, int8_t>::type;

    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<half> attentionOutInitGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGmKv;
    GlobalTensor<int32_t> blockTableGm;
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;

    attenMaskGmType attenMaskGmInt;

    postQuantGmType postQuantScaleGm;
    postQuantGmType postQuantOffsetGm;
    postQuantBf16GmType postQuantScaleBf16Gm;
    postQuantBf16GmType postQuantOffsetBf16Gm;
    quantGmType pScaleGm;
    FaGmTensor<float, Q_SCALE_FORMAT> queryScaleGm;
    FaGmTensor<float, K_SCALE_FORMAT> keyScaleGm;
    quantGmType deScaleVGm;

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    CopyQueryScaleGmToUb<float, Q_SCALE_FORMAT> copyQueryScaleGmToUb;
    CopyKeyScaleGmToUb<float, K_SCALE_FORMAT> copyKeyScaleGmToUb;
    // ub
    TBuf<> commonTBuf; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue[2];
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    TBuf<> softmaxMaxBuf[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf[PRELOAD_N + 1];
    TBuf<> vselrIndexesBuf[4];
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;

    TBuf<> lseTmpBuff;
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    TQue<QuePosition::VECOUT, 1> FDResOutputQue;
    TQue<QuePosition::VECIN, 1> accumOutInputQue;
    TQue<QuePosition::VECIN, 1> postQuantScaleQue;  // postQuant
    TQue<QuePosition::VECIN, 1> postQuantOffsetQue; // postQuant

    TQue<QuePosition::VECIN, 1> queryAntiqScaleInputQue;
    TQue<QuePosition::VECIN, 1> keyAntiqScaleInputQue;
    const ConstInfoX &constInfo;
    T negativeFloatScalar;
    float pScaleValue { 1.0f };
    bool isSkipMask { false };

    // ==================== Functions ======================
    __aicore__ inline FAFullQuantGqaBlockVec(ConstInfoX &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *pScale,
                                        __gm__ uint8_t *blockTable, __gm__ uint8_t *deqScaleQ,
                                        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe = pipe;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, blockTable, deqScaleQ,
                     deqScaleK, deqScaleV, attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *blockTable,
                                        __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
                                        __gm__ uint8_t *deqScaleV, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);

        if constexpr (HAS_MASK) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }
        if constexpr (isFp8) {
            if (pScale != nullptr) {
                pScaleGm.SetGlobalBuffer((__gm__ float *)pScale);
                pScaleValue = this->pScaleGm.GetValue(0);
            }
            if (deqScaleV != nullptr) {
                deScaleVGm.SetGlobalBuffer((__gm__ float *)deqScaleV);
            }
            if (constInfo.actualSeqLenKVSize != 0) {
                actualSeqLengthsGmKv.SetGlobalBuffer(
                    (__gm__ uint64_t *)actualSeqKvlenAddr, constInfo.actualSeqLenKVSize);
            }
            if constexpr (PAGE_ATTENTION) {
                blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
            }
        }

        InitQScaleBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size,
                    1, actualSeqLengthsGmQ, constInfo.actualSeqLenSize,
                    queryScaleGm, deqScaleQ);
        InitKScaleBuffer(constInfo.bSize, constInfo.s2Size, actualSeqLengthsGmKv, constInfo.actualSeqLenKVSize,
                    constInfo.n2Size, constInfo.blockSize, keyScaleGm,
                    deqScaleK, constInfo.kScaleStrides.bnStride, constInfo.kScaleStrides.n2Stride);

        if constexpr (FLASH_DECODE) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, GlobalTensor<uint64_t> actualSeqLenGmQ,
                                            uint32_t actualLenQDims,
                                            FaGmTensor<float, Q_SCALE_FORMAT> &qScaleGmTensor, __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_NT) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, actualSeqLenGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void InitKScaleBuffer(uint32_t batchSize, uint32_t kvSeqSize,
                                            GlobalTensor<uint64_t> actualSeqLenGmKv, uint32_t actualLenDims,
                                            uint32_t n2Size, uint32_t kvCacheBlockSize,
                                            FaGmTensor<float, K_SCALE_FORMAT> &kScaleGmTensor, __gm__ uint8_t *gm,
                                            uint64_t bnStrides, uint64_t n2Strides)
    {
        kScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        if constexpr (GmLayoutParams<K_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_BnNBs_KS) {
            kScaleGmTensor.offsetCalculator.Init(n2Size, kvCacheBlockSize, blockTableGm,
                                                 constInfo.maxBlockNumPerBatch, bnStrides, n2Strides);
        }
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX runInfo)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm1ResBuf.SetCrossCore();
            outputBuf.SetCrossCore();
            return;
        }

        if constexpr (USE_DN) {
            ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo);
        } else {
            // ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo);
        }
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo.isExistRowInvalid;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.realN2Size * constInfo.realGSize * constInfo.dSizeV;
        if constexpr (POST_QUANT) {
            totalOutputSize /= 2;
        }
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            } else {
                matmul::InitOutput<OUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.realN2Size * constInfo.realGSize;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo)
    {
        FaUbTensor<float> ubTensor {
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoord gmCoord {
             .bIdx = runInfo.bIdx,
             .n2Idx = runInfo.realN2Idx,
             .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
             .dIdx = dOffset,
             .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb(ubTensor, queryScaleGm, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo);
    }


    __aicore__ inline void CopyKeyScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset,
                                             uint32_t dRealSize, RunInfoX &runInfo)
    {
        FaUbTensor<float> ubTensor {
            .tensor = dstTensor,
            .rowCount = runInfo.actSingleLoopS2Size,
            .colCount = 1
        };

        AntiqGmCoord antiqGmCoord {
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.n2Idx,
            .s2Idx = runInfo.s2Idx,
            .s2DealSize = runInfo.actSingleLoopS2Size
        };
        copyKeyScaleGmToUb(ubTensor, keyScaleGm, antiqGmCoord);
    }

    __aicore__ inline void CopyKeyScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo)
    {
        CopyKeyScaleSlice(dstTensor, 0, 1, runInfo);
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        LocalTensor<uint8_t> attenMaskUb;
        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInQue[0].template AllocTensor<uint8_t>();
            AttenMaskCopyInDn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];

        auto expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[0];
        int64_t stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;

        // 加载qScale/kScale
        LocalTensor<float> qScaleUbTensor;
        LocalTensor<float> kScaleUbTensor = keyAntiqScaleInputQue.template AllocTensor<float>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            qScaleUbTensor = queryAntiqScaleInputQue.template AllocTensor<float>();
            CopyQueryScaleTile(qScaleUbTensor, runInfo);
            queryAntiqScaleInputQue.template EnQue(qScaleUbTensor);
            qScaleUbTensor = queryAntiqScaleInputQue.DeQue<float>();
        }
        CopyKeyScaleTile(kScaleUbTensor, runInfo);
        keyAntiqScaleInputQue.template EnQue(kScaleUbTensor);
        kScaleUbTensor = keyAntiqScaleInputQue.DeQue<float>();

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (unlikely(runInfo.isLastS2Loop && !runInfo.isFirstS2Loop)) {
            if (!isSkipMask) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue), descaleQK,
                    pScaleValue, negativeFloatScalar, 0.0F, true);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue), descaleQK,
                    pScaleValue, negativeFloatScalar, 0.0F, true);
            }
        } else if (unlikely(runInfo.isFirstS2Loop)) {
            if (!isSkipMask) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, hasAtten, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue), descaleQK,
                    pScaleValue, negativeFloatScalar, 0.0F, true);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, false, false, s2BaseSize, true>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue), descaleQK,
                    pScaleValue, negativeFloatScalar, 0.0F, true);
            }
        } else {
            if (!isSkipMask) {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, hasAtten, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue), descaleQK,
                    pScaleValue, negativeFloatScalar, 0.0F, true);
            } else {
                FaVectorApi::ProcessVec1VfDnPerTokenHead<T, INPUT_T, true, false, s2BaseSize>(
                    stage1CastTensor, sumUb, maxUb, mmRes, expUb,
                    this->vselrIndexesBuf, attenMaskUb, qScaleUbTensor, kScaleUbTensor,
                    ((runInfo.actMSizeAlign32 >> 1) + 63) >> 6 << 6, runInfo.actSingleLoopS2SizeAlign,
                    runInfo.actSingleLoopS2Size, static_cast<T>(constInfo.scaleValue), descaleQK,
                    pScaleValue, negativeFloatScalar, 0.0F, true);
            }
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            queryAntiqScaleInputQue.FreeTensor(qScaleUbTensor);
        }
        keyAntiqScaleInputQue.FreeTensor(kScaleUbTensor);

        bmm1ResBuf.SetCrossCore();
        if constexpr (HAS_MASK) {
            this->attenMaskInQue[0].template FreeTensor(attenMaskUb);
        }

        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        // 按照64对齐搬运
        uint32_t singleProcessSOuterSize = mBaseSize >> 1;
        uint32_t s2RealSizeAlign = (((runInfo.actSingleLoopS2Size + 63) >> 6) << 6);
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * singleProcessSOuterSize * s2RealSizeAlign],
            stage1CastTensor, {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * singleProcessSOuterSize * s2RealSizeAlign +
            s2RealSizeAlign * 32], stage1CastTensor[65 << 5],
            {static_cast<uint16_t>((runInfo.actSingleLoopS2Size + 63) >> 6), 64, 66, 0});

        //-----------------------------------------------------------------
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
        return;
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if constexpr (FLASH_DECODE) {
            if (!runInfo.isS2SplitCore && constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfoX &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();

        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize);
        softmaxLseQueue.template EnQue(lseUb);
        softmaxLseQueue.DeQue<float>();

        if constexpr (outLayout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo.realN2Size * constInfo.realGSize +
                                 runInfo.realN2Idx * constInfo.realGSize;
            DataCopySoftmaxLseTNDArch35NoGS1Merge<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                                 runInfo.actVecMSize, constInfo);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint32_t s1Size = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo.n2Size * constInfo.gSize + runInfo.n2Idx * constInfo.gSize;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                       constInfo, s1Size);
        } else if constexpr (outLayout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo, s1LeftPaddingSize);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t s1LeftPaddingSize = 0;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, runInfo.actVecMSize,
                                                        constInfo, qActSeqLens, s1LeftPaddingSize);
        }

        softmaxLseQueue.FreeTensor(lseUb);
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        int32_t deScaleVOffset = runInfo.n2Idx;
        float deScaleVValue = this->deScaleVGm.GetValue(deScaleVOffset);

        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            if (!runInfo.isLastS2Loop) {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) { // s2方向第二轮循环更新第一轮循环的deScaleV
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            } else {
                if (unlikely(runInfo.s2Idx == s2BaseSize)) { // s2方向第二轮循环更新第一轮循环的deScaleV
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                } else {
                    LocalTensor<float> sumUb =
                        this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                    FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                        vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize, dTemplateAlign64,
                        deScaleVValue, deScaleVValue);
                }
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64, deScaleVValue);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
    }

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;

        attenOut.SetAddr(vec2ResUb.address_);

        if constexpr (!POST_QUANT) {
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
            Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        } else {
            PostQuant(runInfo, attenOut, vec2ResUb, mStartVec, mDealSize, dSizeAligned64);
            RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        }
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void PostQuant(RunInfoX &runInfo, LocalTensor<OUTPUT_T> &attenOut,
                                     LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                     int64_t dSizeAligned64)
    {
        return;
    }


    __aicore__ inline void CopyOutAttentionOut(RunInfoX runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                Bmm2ResForFDCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            } else {
                Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
            }
        } else {
            Bmm2ResCastAndCopyOut(runInfo, vec2ResUb, mStartVec, mDealSize);
        }
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx;
        int32_t s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH || layout == LayOutTypeEnum::LAYOUT_SBH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.realGSize;
            s1EndTdx = vecMEndIdx / constInfo.realGSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx == gEndIdx) {
                // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            } else {
                // 跨多个G
                ret = (s1StartTdx < s1FirstValidToken);
                ret = ret || (s1EndTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken = AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0),
                                                             runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(AttentionCommon::Max(runInfo.preTokensLeftUp +
                                                            runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
                                       ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                        hasValidRow); // sparse = 0 or 3 or 4，preToekens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }

            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;

            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                if constexpr (!POST_QUANT) {
                    RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                                              static_cast<uint32_t>(dSizeAligned64));
                } else {
                    uint32_t dStride =
                        CeilDiv(static_cast<uint32_t>(static_cast<uint32_t>(dSizeAligned64)), sizeof(float));
                    uint16_t dSize = CeilDiv(constInfo.dSizeV, sizeof(float)); // w8后量化后的处理长度
                    RowInvalidUpdateVF<float>(*((LocalTensor<float> *)&vec2ResUb), maxTensor, mDealSize, dSize,
                                              dStride);
                }
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)};
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.realN2Idx,
                        .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                              constInfo.s1Size, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize, false, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                              constInfo.s1Size, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize, false, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV,
                                              actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV,
                                              actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        // Copy sum to gm
        LocalTensor<float> sumOutTensor = sumBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(sumOutTensor, sumUb, runInfo.actVecMSize);
        sumBrdcst.template EnQue(sumOutTensor);
        sumBrdcst.template DeQue<float>();
        DataCopy(softmaxFDSumGm[gmOffset], sumOutTensor, calculateSize);
        sumBrdcst.template FreeTensor(sumOutTensor);

        // Copy max to gm
        LocalTensor<float> maxOutTensor = maxBrdcst.template AllocTensor<float>();
        FaVectorApi::BroadcastMaxSum(maxOutTensor, maxUb, runInfo.actVecMSize);
        maxBrdcst.template EnQue(maxOutTensor);
        maxBrdcst.template DeQue<float>();
        DataCopy(softmaxFDMaxGm[gmOffset], maxOutTensor, calculateSize);
        maxBrdcst.template FreeTensor(maxOutTensor);
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfoX &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        // 是否要改成halfMRealSize
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // flashDecodeS2Idx?nBufferStartM?
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop % DB]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[runInfo.loop % DB]);
        uint64_t gmOffset =
            runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) / (FA_BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            ProcessVec2OnUb(bmm2ResBuf, runInfo);
        }
        return;
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        tPipe->InitBuffer(softmaxSumBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxSumBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxSumBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(softmaxMaxBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxMaxBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxMaxBuf[2], 256); // [64, 1]

        tPipe->InitBuffer(softmaxExpBuf[0], 256); // [64, 1]
        tPipe->InitBuffer(softmaxExpBuf[1], 256); // [64, 1]
        tPipe->InitBuffer(softmaxExpBuf[2], 256); // [64, 1]

        if constexpr (FLASH_DECODE) {
            tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
            tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
        }
    }

    __aicore__ inline void InitBuffers()
    {
        if constexpr (!bmm2Write2Ub) {
            tPipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
        }
        SoftmaxInitBuffer();
        tPipe->InitBuffer(queryAntiqScaleInputQue, 1, (mBaseSize >> 1U) * sizeof(float));
        tPipe->InitBuffer(keyAntiqScaleInputQue, 1, s2BaseSize * sizeof(float));
        tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
        tPipe->InitBuffer(stage1OutQue[0], 1, 16640); // (32 + 1) * (256 / 32) * 64
        tPipe->InitBuffer(stage1OutQue[1], 1, 16640);
        tPipe->InitBuffer(commonTBuf, 512);
        if constexpr (HAS_MASK) {
            tPipe->InitBuffer(attenMaskInQue[0], 1, 16384); // 256 * 64
        }

        if (constInfo.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->tPipe->InitBuffer(softmaxLseQueue, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }
        if constexpr (isFp8) {
            if constexpr (USE_DN) {
                tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)], 256);

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < (256 >> 2); j++) {
                        vselrIndexesTensor.SetValue(i * (256 >> 2) + j, i + (j << 2));
                    }
                }
            } else {
                tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)],
                                  128); // s2realsize (64, 128]
                tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)],
                                  64); // s2realsize (0, 64]

                LocalTensor<uint8_t> vselrIndexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
                vselrIndexesTensor.SetValue(0, 0x7f);
                for (int i = 0; i < 128; i++) {
                    vselrIndexesTensor.SetValue(i, i * 2);
                }
                vselrIndexesTensor =
                    vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
                for (int i = 0; i < 64; i++) {
                    vselrIndexesTensor.SetValue(i, i * 4);
                }
            }
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb,
                                           uint32_t vecMIdx, uint32_t mDealSize, RunInfoX &runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.realGSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSize - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        bool IsSkipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        if (IsSkipMask && IsSkipMaskForPre) {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
            return;
        }

        if (!IsSkipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSize);
        }

        if (!IsSkipMaskForPre) {
            LocalTensor<uint8_t> attenMaskUbPre = this->attenMaskInQue[0].template AllocTensor<uint8_t>();
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt,
                maskInfo, true);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            this->attenMaskInQue[0].template FreeTensor(attenMaskUbPre);
        }
    }

    __aicore__ inline void AttenMaskCopyInDn(LocalTensor<uint8_t> attenMaskUb,
                                             uint32_t vecMIdx, uint32_t mDealSize, RunInfoX &runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mBaseSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.realGSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = 0U;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        isSkipMask = IsSkipAttentionmask(maskInfo);
        if (!isSkipMask) {
            AttentionmaskCopyInDn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt, maskInfo);
        }
    }
};

template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantGqaBlockVecDummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    __aicore__ inline FAFullQuantGqaBlockVecDummy(ConstInfoX &constInfo) {};
};
} // namespace BaseApi
#endif // FIA_BLOCK_VEC_FULLQUANT_GQA_H_