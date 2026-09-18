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
 * \file flash_mla_c8_vec.h
 * \brief MLA 全量化重构模板 Vec Block
 */
#ifndef FIA_BLOCK_VEC_FULLQUANT_MLA_H_
#define FIA_BLOCK_VEC_FULLQUANT_MLA_H_

#include "kernel_operator.h"
#include <limits>
#include "../../../a5_mla_common/op_kernel/arch35/attenmask_gs1_arch35.h"
#include "adv_api/activation/softmax.h"
#include "../../../a5_mla_common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../a5_mla_common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"

#include "flash_mla_c8_public.h"
#include "../../../a5_mla_common/op_kernel/vector_common.h"

#include "flash_mla_c8_memory.h"
#include "flash_mla_c8_dcp_output.h"
#include "flash_mla_c8_vec128_packed_p.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <
    typename INPUT_T, typename T, typename OUTPUT_T, FlashMlaC8Layout layout = FlashMlaC8Layout::None,
    FlashMlaC8Layout outLayout = FlashMlaC8Layout::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned576,
    DTemplateType dVTemplateType = DTemplateType::Aligned512, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMlaBlockVec {
public:
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2;

    static constexpr uint32_t initOutputEventId = 0U;

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT =
        (layout == FlashMlaC8Layout::LAYOUT_BSH || layout == FlashMlaC8Layout::LAYOUT_TND ||
         layout == FlashMlaC8Layout::LAYOUT_SBH) ?
            MaskFormat::SG :
            MaskFormat::GS;

    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    static constexpr bool IS_PER_TOKEN_HEAD = true;
    static constexpr GmFormat Q_SCALE_FORMAT = GetQueryScaleGmFormat<layout, USE_DN, IS_PER_TOKEN_HEAD, true>();

    static constexpr bool POST_QUANT = !IsSameType<OUTPUT_T, half>::value && !IsSameType<OUTPUT_T, bfloat16_t>::value &&
                                       !IsSameType<OUTPUT_T, float>::value;
    using pseShiftType = half;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using MM1_OUT_T = T;
    using MM2_OUT_T = T;
    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> attentionWireGm;
    GlobalTensor<float> softmaxLseGm;
    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    ActualSeqLensParser<Q_MODE, uint32_t, true> qActSeqLensParser;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<float> deScaleQGm;
    GlobalTensor<float> deScaleKGm;
    GlobalTensor<float> deScaleVGm;
    FaGmTensor<float, Q_SCALE_FORMAT, uint32_t, true> queryScaleGm;
    CopyQueryScaleGmToUb<float, Q_SCALE_FORMAT> copyQueryScaleGmToUb;
    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

    // ub
    TBuf<> commonTBuf;
    TQue<QuePosition::VECOUT, 1> stage1OutQue[2];
    TQue<QuePosition::VECIN, 1> attenMaskInQue;
    TBuf<> stage2OutBuf;
    TEventID mte3ToVId[2];
    TEventID vToMte3Id[2];
    TEventID mte2ToVId[2];
    TEventID vToMte2Id[2];
    TBuf<> softmaxMaxBuf[PRELOAD_N + 1];
    TBuf<> softmaxSumBuf[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf[PRELOAD_N + 1];
    TBuf<> preLoopMaxBuf;
    TBuf<> preLoopSumBuf;
    TBuf<> firstLoopSumBuf;
    TBuf<> vselrIndexesBuf[4];
    TBuf<> lseTmpBuff;
    TBuf<> queryAntiqScaleInputQue[2];
    TBuf<> pScaleBuf[3];
    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域, FLASH_DECODE 写 softmaxFDSumGm/softmaxFDMaxGm */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;

    const ConstInfoX &constInfo;
    T negativeFloatScalar;
    float deScaleKValue{1.0f};
    float deScaleVValue{1.0f};
    bool isSkipMask{false};
    bool isFullMask{false};
    uint32_t minValue{NEGATIVE_MIN_VALUE_FP32};

    __aicore__ inline FAFullQuantMlaBlockVec(ConstInfoX &constInfo)
        : constInfo(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *dequantScaleQuery,
                                        __gm__ uint8_t *dequantScaleKey, __gm__ uint8_t *dequantScaleValue,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe = pipe;
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar = *((T *)&tmp1);

        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, dequantScaleQuery, dequantScaleKey, dequantScaleValue,
                     attenMask, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
                                        __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                        __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        attentionWireGm.SetGlobalBuffer((__gm__ float *)attentionOut);
        if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_NTD_DCP) {
            static_assert(layout == FlashMlaC8Layout::LAYOUT_TND);
            static_assert(IsSameType<OUTPUT_T, bfloat16_t>::value);
        }
        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);

        if constexpr (HAS_MASK) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        // MLA 全量化 dequantScale: Q per-token, KV per-tensor
        if (dequantScaleQuery != nullptr) {
            deScaleQGm.SetGlobalBuffer((__gm__ float *)dequantScaleQuery);
            InitQScaleBuffer(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize, constInfo.s1Size, 1,
                             actualSeqLengthsGmQ, constInfo.actualSeqLenSize, queryScaleGm, dequantScaleQuery);
        }
        if (dequantScaleKey != nullptr) {
            deScaleKGm.SetGlobalBuffer((__gm__ float *)dequantScaleKey);
            deScaleKValue = this->deScaleKGm.GetValue(0);
        }
        if (dequantScaleValue != nullptr) {
            deScaleVGm.SetGlobalBuffer((__gm__ float *)dequantScaleValue);
            deScaleVValue = this->deScaleVGm.GetValue(0);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
        }
    }

    __aicore__ inline void InitQScaleBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                            uint32_t headDim, GlobalTensor<uint32_t> actualSeqLenGmQ,
                                            uint32_t actualLenQDims, FaGmTensor<float, Q_SCALE_FORMAT, uint32_t, true> &qScaleGmTensor,
                                            __gm__ uint8_t *gm)
    {
        qScaleGmTensor.gmTensor.SetGlobalBuffer((__gm__ float *)gm);
        if constexpr (GmLayoutParams<Q_SCALE_FORMAT>::CATEGORY == FormatCategory::GM_ANTIQ_TN) {
            qScaleGmTensor.offsetCalculator.Init(n2Size, gSize, actualSeqLenGmQ, actualLenQDims);
        }
    }

    __aicore__ inline void SetQueryLengths(__gm__ uint8_t *cu, __gm__ uint8_t *used)
    {

        qActSeqLensParser.Init(cu, constInfo.actualSeqLenSize, used, used ? constInfo.bSize : 0);
        queryScaleGm.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, qActSeqLensParser);
    }

    __aicore__ inline void CopyQueryScaleSlice(const LocalTensor<float> &dstTensor, uint32_t dOffset,
                                               uint32_t dRealSize, RunInfoX &runInfo, bool isPreload)
    {
        FaUbTensor<float> ubTensor{
            .tensor = dstTensor,
            .rowCount = runInfo.actVecMSize,
            .colCount = dRealSize,
        };

        GmCoordGs1Merge gmCoord{
            .bIdx = runInfo.bIdx,
            .n2Idx = runInfo.realN2Idx,
            .gS1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
            .dIdx = dOffset,
            .gS1DealSize = runInfo.actVecMSize,
        };
        copyQueryScaleGmToUb(ubTensor, queryScaleGm, gmCoord);
    }

    __aicore__ inline void CopyQueryScaleTile(const LocalTensor<float> &dstTensor, RunInfoX &runInfo, bool isPreload)
    {
        CopyQueryScaleSlice(dstTensor, 0, 1, runInfo, isPreload);
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

        // MLA 全量化走 ND 路径
        ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo);
    }

    __aicore__ inline void ClearOutput()
    {
        if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_NTD_DCP) {
            // Row-aligned ownership prevents another core's zero DMA from
            // overwriting a wire LSE. Physical T includes graph padding.
            InitDcpOutput<OUTPUT_T, initOutputEventId>(attentionOutGm, attentionWireGm, softmaxLseGm, constInfo);
            SyncAll();
            return;
        }
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
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
        return constInfo.needInit || constInfo.isExistRowInvalid;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == FlashMlaC8Layout::LAYOUT_TND || layout == FlashMlaC8Layout::LAYOUT_NTD ||
                      layout == FlashMlaC8Layout::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.realN2Size * constInfo.realGSize * constInfo.dSizeV;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum);
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUTPUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == FlashMlaC8Layout::LAYOUT_TND || layout == FlashMlaC8Layout::LAYOUT_NTD ||
                      layout == FlashMlaC8Layout::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.realN2Size * constInfo.realGSize;
        int64_t singleCoreSize = (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum);
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, -std::numeric_limits<float>::infinity());
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
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
        SoftmaxInitBuffer();
        tPipe->InitBuffer(preLoopMaxBuf, 256);
        tPipe->InitBuffer(preLoopSumBuf, 256);
        tPipe->InitBuffer(firstLoopSumBuf, 256);
        tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
        tPipe->InitBuffer(stage1OutQue[0], 1, 4224); // 4224: (s1BaseSize / CV_RATIO + 1) * s2BaseSize * sizeof(INPUT_T)
        tPipe->InitBuffer(stage1OutQue[1], 1, 4224);
        tPipe->InitBuffer(commonTBuf, 512);

        if constexpr (hasAtten) {
            // GS1方向需要循环处理，一个vector计算softmax的数据量最大为32*128，对应mask(bool/int8/uint8)的数据量为4096Bytes
            tPipe->InitBuffer(attenMaskInQue, 1, 4096); // 32 rows x 128 columns.
        }

        // MLA 全量化: queryAntiqScaleInputQue (per-token Q scale) 和 pScaleBuf
        constexpr uint32_t softmaxRowmaxBufSize = 256; // s1 baseSize * 4b(fp32)
        tPipe->InitBuffer(queryAntiqScaleInputQue[0], (mBaseSize / CV_RATIO) * sizeof(float));
        tPipe->InitBuffer(queryAntiqScaleInputQue[1], (mBaseSize / CV_RATIO) * sizeof(float));
        tPipe->InitBuffer(pScaleBuf[0], softmaxRowmaxBufSize);
        tPipe->InitBuffer(pScaleBuf[1], softmaxRowmaxBufSize);
        tPipe->InitBuffer(pScaleBuf[2], softmaxRowmaxBufSize);

        if (constInfo.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->tPipe->InitBuffer(softmaxLseQueue, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
        }
        // ND
        tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)],
                          128); // s2realsize (64, 128]
        tPipe->InitBuffer(vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)],
                          64); // s2realsize (0, 64]

        LocalTensor<uint8_t> vselrIndexesTensor =
            vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)].template Get<uint8_t>();
        vselrIndexesTensor.SetValue(0, 0x7f);
        for (int i = 0; i < 128; i++) {
            vselrIndexesTensor.SetValue(i, i << 1);
        }
        vselrIndexesTensor =
            vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_0_AND_LTE_64_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 64; i++) {
            vselrIndexesTensor.SetValue(i, i << 2);
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        mte2ToVId[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        mte2ToVId[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        vToMte2Id[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        vToMte2Id[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(vToMte2Id[0]);
        SetFlag<HardEvent::V_MTE2>(vToMte2Id[1]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId[1]);
        WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Id[0]);
        WaitFlag<AscendC::HardEvent::V_MTE2>(vToMte2Id[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVId[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(mte2ToVId[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Id[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(vToMte2Id[1]);
    }

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
                                           RunInfoX &runInfo)
    {
        uint32_t s2RealSize = runInfo.actSingleLoopS2Size;
        constexpr uint32_t s2BaseSizeCur = s2BaseSize;

        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.realGSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = s2RealSize;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSizeCur;
        maskInfo.preToken = constInfo.preTokens;
        maskInfo.nextToken = constInfo.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo.sparseMode);
        maskInfo.batchIdx = (constInfo.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo.attenMaskS1Size * constInfo.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (s2BaseSizeCur - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar;
        maskInfo.s1LeftPaddingSize = runInfo.qPaddingBeginOffset;
        maskInfo.s2LeftPaddingSize = runInfo.kvPaddingBeginOffset;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL; // compatible with int8/uint8

        bool IsSkipMask = IsSkipAttentionmask(maskInfo);
        if (unlikely(!IsSkipMask)) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSizeCur>(attenMaskUb, attenMaskGmInt, maskInfo);
        } else {
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * s2BaseSizeCur);
        }
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        uint32_t actVecMSizeAlign16 = runInfo.actMSizeAlign32 >> 1;

        if constexpr (HAS_MASK) {
            attenMaskUb = this->attenMaskInQue.template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

        int64_t stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0f;

        // Reuse the per-M-tile scales on every KV tile; only the first tile
        // fills this UB buffer from GM.
        LocalTensor<float> queryScaleUb = queryAntiqScaleInputQue[runInfo.mloop % DB].template Get<float>();
        LocalTensor<T> pScaleUb;
        // MLA 全量化: 首个 S2 loop 加载 per-token Q scale, 取 pScale buffer
        if (unlikely(runInfo.isFirstS2Loop)) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2Id[runInfo.mloop % DB]);
            CopyQueryScaleTile(queryScaleUb, runInfo, false);
            SetFlag<HardEvent::MTE2_V>(mte2ToVId[runInfo.mloop % DB]);
            WaitFlag<HardEvent::MTE2_V>(mte2ToVId[runInfo.mloop % DB]);
        }
        pScaleUb = pScaleBuf[runInfo.loop % 3].template Get<T>();

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();

        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        // 按 S2 实际长度选择 oriNRange（编译期模板参数，需静态分支实例化），对齐老模板
        // ProcessVec1Vf 模板参数: <T, T2, pseShiftType, isUpdate, s1BaseSize, s2BaseSize, oriNRange,
        //                          hasAtten, pseMode, hasDrop, isMlaSgd, isMlaFullQuant, useNz, hasSink>
        // MLA s2BaseSize=128 固定，仅需 EQ_128 / GT_0_AND_LTE_64 / GT_64_AND_LTE_128 三段
        // isMlaSgd: 保留老模板 gSize>=32 判定（layout!=BNSD 且 gSize>=32 时走 isMlaSgd=true 实例化）
        constexpr bool isMlaSgdTrue = true;
        constexpr bool isMlaSgdFalse = false;
        if (unlikely(runInfo.isFirstS2Loop)) {
            // 首 loop: isUpdate=false
            if (likely(s2CalcSize == s2BaseSize)) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                                           pseMode, hasDrop, isMlaSgdFalse, true, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F, queryScaleUb,
                    deScaleKValue);
            } else if (s2CalcSize <= 64) {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                                           HAS_MASK, pseMode, hasDrop, isMlaSgdFalse, true, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F, queryScaleUb,
                    deScaleKValue);
            } else {
                FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                                           HAS_MASK, pseMode, hasDrop, isMlaSgdFalse, true, false, false>(
                    stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F, queryScaleUb,
                    deScaleKValue);
            }
        } else {
            // 非首 loop: isUpdate=true
            bool isMlaSgdRuntime = (constInfo.gSize >= 32) && (constInfo.gSize % 32 == 0);
            if (likely(s2CalcSize == s2BaseSize)) {
                if constexpr (!HAS_MASK && !hasDrop && pseMode == PseTypeEnum::PSE_NONE_TYPE &&
                              mBaseSize == 64 && s2BaseSize == 128 && IsSameType<INPUT_T, fp8_e4m3fn_t>::value) {
                    auto indexes = this->vselrIndexesBuf[static_cast<int>(VselrIndexEnum::GT_64_AND_LTE_128_INDEX)]
                                       .template Get<uint8_t>();
                    FaVectorApi::C8ProcessVec1Update128PackedP(stage1CastTensor, indexes, mmRes, maxUb,
                        apiTmpBuffer, pScaleUb, runInfo.actVecMSize, static_cast<T>(constInfo.scaleValue),
                        queryScaleUb, deScaleKValue);
                } else if (isMlaSgdRuntime) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                                               pseMode, hasDrop, isMlaSgdTrue, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                        posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F,
                        queryScaleUb, deScaleKValue);
                } else {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, EQ_128, HAS_MASK,
                                               pseMode, hasDrop, isMlaSgdFalse, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                        posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F,
                        queryScaleUb, deScaleKValue);
                }
            } else if (s2CalcSize <= 64) {
                if (isMlaSgdRuntime) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                                               HAS_MASK, pseMode, hasDrop, isMlaSgdTrue, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                        posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F,
                        queryScaleUb, deScaleKValue);
                } else {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64,
                                               HAS_MASK, pseMode, hasDrop, isMlaSgdFalse, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                        posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F,
                        queryScaleUb, deScaleKValue);
                }
            } else {
                if (isMlaSgdRuntime) {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                                               HAS_MASK, pseMode, hasDrop, isMlaSgdTrue, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                        posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F,
                        queryScaleUb, deScaleKValue);
                } else {
                    FaVectorApi::ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128,
                                               HAS_MASK, pseMode, hasDrop, isMlaSgdFalse, true, false, false>(
                        stage1CastTensor, this->vselrIndexesBuf, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                        pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, s2CalcSize, pseStride, slopes,
                        posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 1.0F,
                        queryScaleUb, deScaleKValue);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();

        if constexpr (HAS_MASK) {
            this->attenMaskInQue.template FreeTensor(attenMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        if (likely(runInfo.actVecMSize != 0)) {
            int64_t dstOffset = constInfo.subBlockIdx * (mBaseSize * 16);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor,
                     {s2BaseSize / 32, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                      (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();
        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }
        if (unlikely(runInfo.isLastS2Loop)) {
            SetFlag<HardEvent::V_MTE2>(vToMte2Id[runInfo.mloop % DB]);
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfoX runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if constexpr (FLASH_DECODE) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
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

        uint32_t vecMSize = runInfo.actVecMSize;
        uint32_t gmDealRowCount = runInfo.actVecMSize;

        uint32_t vecMIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        LocalTensor<float> lseUb = this->softmaxLseQueue.template AllocTensor<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, vecMSize, minValue);
        softmaxLseQueue.template EnQue(lseUb);
        softmaxLseQueue.DeQue<float>();

        if constexpr (layout == FlashMlaC8Layout::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset =
                runInfo.realN2Idx * constInfo.realGSize * constInfo.t1Size;
            DataCopySoftmaxLseTNDtoNTArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                                                                 gmDealRowCount, prefixBS1, constInfo);
        } else if constexpr (layout == FlashMlaC8Layout::LAYOUT_NTD) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint32_t s1Size = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo.n2Size * constInfo.gSize + runInfo.n2Idx * constInfo.gSize;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                       constInfo, s1Size);
        } else if constexpr (layout == FlashMlaC8Layout::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo, 0);
        } else {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoX>(softmaxLseGm, lseUb, bN2Offset, vecMIdx, gmDealRowCount,
                                                        constInfo, qActSeqLensParser.GetActualSeqLength(runInfo.bIdx),
                                                        0);
        }
        if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_NTD_DCP) {
            CopyDcpLse(attentionWireGm, lseUb, runInfo.realN2Idx, vecMIdx, gmDealRowCount,
                       qActSeqLensParser.GetTBase(runInfo.bIdx), constInfo);
        }
        softmaxLseQueue.FreeTensor(lseUb);
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
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        // MLA 全量化 vec2
        bmm2ResBuf.WaitCrossCore();
        if constexpr (bmm2Write2Ub) {
            ProcessVec2OnUb(bmm2ResBuf, runInfo);
        }
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }
        uint32_t vecMSize = runInfo.actVecMSize;
        int64_t vec2CalcSize = vecMSize * dTemplateAlign64;

        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;
            pScaleUb = pScaleBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            // Static per-tensor V scale is shared by every KV tile. Keep the
            // running sum in quantized-V units and apply its descale once at
            // the final tile, instead of multiplying all 512 columns per tile.
            if (likely(!runInfo.isLastS2Loop)) {
                FlashUpdateNew<T, T, OUTPUT_T, dTemplateAlign64, false, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, vecMSize,
                    dTemplateAlign64, 1.0f, 1.0f);
            } else {
                LocalTensor<float> sumUb =
                    this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true, true>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, vecMSize,
                    dTemplateAlign64, deScaleVValue, deScaleVValue);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (unlikely(runInfo.isLastS2Loop)) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, true>(vec2ResUb, vec2ResUb, sumUb, vecMSize,
                                                                         (uint16_t)dTemplateAlign64, deScaleVValue);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, vecMSize);
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
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

    __aicore__ inline void Bmm2ResCastAndCopyOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                                 uint32_t mDealSize)
    {
        LocalTensor<OUTPUT_T> attenOut;
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }

        attenOut.SetAddr(vec2ResUb.address_);

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);

        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec, int64_t mDealSize,
                                      RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (HAS_MASK) {
            int64_t s1FirstValidToken =
                AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(
                AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid =
                constInfo.isRowInvalidOpen || ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) && hasValidRow);
            if (!batchNeedRowInvalid) {
                return;
            }
        }
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

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)};
        GmCoordGs1Merge gmCoord{.bIdx = info.bIdx,
                                .n2Idx = info.realN2Idx,
                                .gS1Idx = info.gS1Idx + info.vecMbaseIdx + vecMIdx,
                                .dIdx = 0,
                                .gS1DealSize = dealRowCount,
                                .dDealSize = (uint32_t)constInfo.dSizeV};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoordGs1Merge &gmCoord)
    {
        if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_NTD_DCP) {
            CopyDcpAttentionOut(attentionOutGm, ubTensor, gmCoord,
                                qActSeqLensParser.GetTBase(gmCoord.bIdx), constInfo);
        } else if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                              constInfo.s1Size, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize, false, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.realN2Size, constInfo.realGSize,
                                              constInfo.s1Size, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize, false, 0);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV,
                                              actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if constexpr (outLayout == FlashMlaC8Layout::LAYOUT_NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT, uint32_t, true> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.realN2Size, constInfo.realGSize, constInfo.dSizeV,
                                              actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        }
    }
};

// Dummy 类
template <
    typename INPUT_T, typename T, typename OUTPUT_T, FlashMlaC8Layout layout = FlashMlaC8Layout::None,
    FlashMlaC8Layout outLayout = FlashMlaC8Layout::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned64,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned576,
    DTemplateType dVTemplateType = DTemplateType::Aligned512, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FAFullQuantMlaBlockVecDummy {
public:
    __aicore__ inline void SetQueryLengths(__gm__ uint8_t *, __gm__ uint8_t *) {}

    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using ConstInfoX = ConstInfo_t<FiaKernelType::FULL_QUANT>;
    using OUT_T = OUTPUT_T;
    __aicore__ inline FAFullQuantMlaBlockVecDummy(ConstInfoX &constInfo)
        : constInfo(constInfo){};
    __aicore__ inline void InitVecBlock(TPipe *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,
                                        __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *,
                                        __gm__ uint8_t *)
    {}
    __aicore__ inline void InitBuffers() {}
    __aicore__ inline void AllocEventID() {}
    __aicore__ inline void FreeEventID() {}
    __aicore__ inline void ClearOutput() {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &, RunInfoX)
    {}
    __aicore__ inline void ProcessVec2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &, RunInfoX) {}
    __aicore__ inline void ProcessVec2(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> &, RunInfoX) {}
    const ConstInfoX &constInfo;
};

} // namespace BaseApi
#endif // FIA_BLOCK_VEC_FULLQUANT_MLA_H_
