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
 * \file fia_block_vec_noquant_gqa.h
 * \brief
 */
#ifndef FIA_BLOCK_VEC_NOQUANT_GQA_H_
#define FIA_BLOCK_VEC_NOQUANT_GQA_H_

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

namespace BaseApi {
template <
    typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
    LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
    S2TemplateType s2TemplateType = S2TemplateType::Aligned128, DTemplateType dTemplateType = DTemplateType::Aligned128,
    DTemplateType dVTemplateType = DTemplateType::Aligned128, PseTypeEnum pseMode = PseTypeEnum::PSE_NONE_TYPE,
    bool hasAtten = false, bool hasDrop = false, bool hasRope = false, uint8_t KvLayoutType = 0, bool isFd = false,
    bool enableKVPrefix = false, bool useDn = false, bool bmm2Write2Ub = true, bool splitD = false>
class FANoQuantGqaBlockVec {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize >> 1;
    static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
    static constexpr uint32_t vec1S2CopyCountDn = mBaseSize >> 5;
    static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
    static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;
    static constexpr uint32_t vec1Srcstride = (mBaseSize >> 1) + 1;  // 解bank冲突，需要加1行
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);

    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2; // C1 C1 C2
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;

    static constexpr uint32_t initOutputEventId = 0U;  // attenOut和lse，刷无效行会用到剩余ub，需要加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();
    static constexpr MaskFormat MASK_LAYOUT = (layout == LayOutTypeEnum::LAYOUT_BSH ||
        layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_SBH) ? MaskFormat::SG : MaskFormat::GS;

    using pseShiftType = INPUT_T;

    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0; // 用于mask为bool类型
    uint32_t negativeIntScalar = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
                                                Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    using attenMaskGmType = typename std::conditional<hasAtten, GlobalTensor<uint8_t>, int8_t>::type;
    using vec2ResGmType = typename std::conditional<splitD, GlobalTensor<float>, int8_t>::type;
    using flashdecodeGmType = typename std::conditional<FLASH_DECODE, GlobalTensor<float>, int8_t>::type;

    using ConstInfoNoQuant = ConstInfo_t<FiaKernelType::NO_QUANT>;

    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<float> softmaxLseGm;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    ActualSeqLensParser<Q_MODE> qActSeqLensParser;

    attenMaskGmType attenMaskGmInt;

    vec2ResGmType vec2ResGm[PRELOAD_N + 1];

    flashdecodeGmType accumOutGm;
    flashdecodeGmType softmaxFDSumGm;
    flashdecodeGmType softmaxFDMaxGm;

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
    TBuf<> mm2InBuf;
    /* 用来做Broadcast[S1,1]->[S2,8]的临时UB区域 */
    TQue<QuePosition::VECOUT, 1> maxBrdcst;
    TQue<QuePosition::VECOUT, 1> sumBrdcst;

    TQue<QuePosition::VECOUT, 1> softmaxLseQueue;

    const ConstInfoNoQuant &constInfo;
    T negativeFloatScalar = *((const T *)&NEGATIVE_MIN_VALUE_FP32);
    int64_t bmm2SubBlockOffset = 0;
    int64_t vec2SubBlockOffset = 0;

    // ==================== Functions ======================
    __aicore__ inline FANoQuantGqaBlockVec(ConstInfoNoQuant &constInfo) : constInfo(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe,
                                        __gm__ uint8_t *pse, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *postQuantScale,
                                        __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe = pipe;
        InitVecInput(pse, actualSeqQlenAddr, actualSeqKvlenAddr, postQuantScale, postQuantOffset,
            attenMask, learnableSink, softmaxLse, attentionOut, workspace);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *pse, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *postQuantScale,
                                        __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *attenMask,
                                        __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);

        if (constInfo.isSoftmaxLseEnable) {
            softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr, constInfo.actualSeqLenSize);
        qActSeqLensParser.Init(actualSeqLengthsGmQ, constInfo.actualSeqLenSize, constInfo.s1Size);

        if constexpr (hasAtten) {
            attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if constexpr (splitD) {
            int64_t vec2ResultSize = mBaseSize * constInfo.dBasicBlock;
            vec2SubBlockOffset = constInfo.subBlockIdx * vec2ResultSize >> 1;
            int64_t prevCoretotalOffset = constInfo.aicIdx * 3 * vec2ResultSize; // 3为preload次数
            vec2ResGm[0].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset);
            vec2ResGm[1].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + vec2ResultSize);
            vec2ResGm[2].SetGlobalBuffer((__gm__ T *)workspace + prevCoretotalOffset + 2 * vec2ResultSize);
            workspace = workspace + constInfo.coreNum * 3 * vec2ResultSize * sizeof(T);
        }

        if constexpr (FLASH_DECODE) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize);
            softmaxFDMaxGm.SetGlobalBuffer((__gm__ float *)workspace + constInfo.accumOutSize +
                                           constInfo.logSumExpSize);
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

        if constexpr (useDn) {
            ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo);
        } else {
            ProcessVec1Nd(outputBuf, bmm1ResBuf, runInfo);
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        // TND、NTD场景且不存在无效行,不需要初始化
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            /*
             * tiling中提前算好了是否可能出现无效行, 正常从tiling中提取这个标记位(constInfo.isExistRowInvalid),
             * 对于FD场景, 有可能整体是没有无效行的,
             * 但当前FD处理的这部分s2是无效的。为规避潜在的风险，只要带mask(constInfo.isExistRowInvalid)
             * 就认为可能存在无效行
             */
            bool isExistRowInvalid = FLASH_DECODE ? HAS_MASK : constInfo.isExistRowInvalid;
            if (!isExistRowInvalid) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);     // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        int64_t tSize = constInfo.bSize * constInfo.s1Size;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD ||
                      layout == LayOutTypeEnum::LAYOUT_NTD_TND) {
            tSize = qActSeqLensParser.GetTSize();
        }
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeV;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<OUT_T>(attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize, 0);
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
        int64_t totalOutputSize = tSize * constInfo.n2Size * constInfo.gSize;
        int64_t singleCoreSize =
            (totalOutputSize + (2 * constInfo.coreNum) - 1) / (2 * constInfo.coreNum); // 2 means c:v = 1:2
        int64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm[constInfo.aivIdx * singleCoreSize],
                                      singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        LocalTensor<uint8_t> attenMaskUb;

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[0];

        auto expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>()[0];
        int64_t stage1Offset = runInfo.loop % DB;

        float descaleQK = 1.0;

        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (unlikely(runInfo.isFirstS2Loop)) {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, false, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, nullptr, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, false);
        } else {
            FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
                stage1CastTensor, sumUb, maxUb, mmRes, expUb, nullptr, attenMaskUb,
                runInfo.actMSizeAlign32 >> 1, runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size,
                static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar, 0.0F, false);
        }
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        if (runInfo.actSingleLoopS2Size > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign +
                                  vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - runInfo.actSingleLoopS2SizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.actSingleLoopS2SizeAlign + 1), 0});
        }
        outputBuf.SetCrossCore();
        //-----------------------------------------------------------------
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);
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

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp,
                                             LocalTensor<float> &softmaxMaxTmp,
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

        if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint32_t s1Size = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo.n2Size * constInfo.gSize + runInfo.n2Idx * constInfo.gSize;
            DataCopySoftmaxLseNTDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                runInfo.actVecMSize, constInfo, s1Size);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
            uint32_t prefixBS1 = qActSeqLensParser.GetTBase(runInfo.bIdx);
            uint64_t bN2Offset = prefixBS1 * constInfo.n2Size * constInfo.gSize + runInfo.n2Idx * constInfo.gSize;
            DataCopySoftmaxLseTNDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                runInfo.actVecMSize, constInfo);
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBSNDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                runInfo.actVecMSize, constInfo, 0);
        } else { // BNSD
            uint64_t bN2Offset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * constInfo.s1Size +
                                 runInfo.n2Idx * constInfo.gSize * constInfo.s1Size;
            uint64_t qActSeqLens = qActSeqLensParser.GetActualSeqLength(runInfo.bIdx);
            DataCopySoftmaxLseBNSDArch35<T, ConstInfoNoQuant>(softmaxLseGm, lseUb, bN2Offset, vecMIdx,
                runInfo.actVecMSize, constInfo, qActSeqLens, 0);
        }

        softmaxLseQueue.FreeTensor(lseUb);
    }

    __aicore__ inline void ProcessVec1Nd(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo)
    {
        LocalTensor<pseShiftType> pseUb;
        LocalTensor<uint8_t> dropMaskUb;
        float slopes = 0.0f;
        float posShift = 0.0f;
        uint32_t pseStride = 0;
        float descaleQK = 1.0;
        float deSCaleKValue = 1.0;
        int64_t stage1Offset = runInfo.loop % DB;

        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<uint8_t> attenMaskUbPre;
        if constexpr (hasAtten == true) {
            attenMaskUb = this->attenMaskInQue[runInfo.loop % DB].template AllocTensor<uint8_t>();
            AttenMaskCopyIn(attenMaskUb, 0, runInfo.actVecMSize, runInfo); // 全量拷贝
        }

        LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> maxUb = this->softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
        LocalTensor<float> expUb = this->softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
        LocalTensor<T> pScaleUb;
        LocalTensor<T> queryScaleUb;
        LocalTensor<uint8_t> apiTmpBuffer;

        apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
        LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();
        auto stage1CastTensor = this->stage1OutQue[stage1Offset].template AllocTensor<INPUT_T>();
        if (runInfo.isFirstS2Loop) {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, EQ_128, hasAtten, pseMode,
                              hasDrop, false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes,
                                                     expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer,
                                                     pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                                                     pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue),
                                                     descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    ProcessVec1Vf<T, INPUT_T, pseShiftType, false, mBaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten,
                                  pseMode, hasDrop>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb,
                                                    sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb,
                                                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                                                    negativeFloatScalar, 0.0F);
                }
            }
        } else {
            if (likely(runInfo.actSingleLoopS2Size == 128)) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, EQ_128, hasAtten, pseMode, hasDrop,
                              false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb,
                                            maxUb, attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, pScaleUb,
                                            runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                            posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                                            negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size <= 64) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_0_AND_LTE_64, hasAtten, pseMode,
                              hasDrop, false, false>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes,
                                                     expUb, sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer,
                                                     pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                                                     pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue),
                                                     descaleQK, negativeFloatScalar, 0.0F, queryScaleUb, deSCaleKValue);
            } else if (runInfo.actSingleLoopS2Size < 128) {
                ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_64_AND_LTE_128, hasAtten,
                              pseMode, hasDrop, false, false>(
                    stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb, sumUb, maxUb, attenMaskUb,
                    pseUb, dropMaskUb, apiTmpBuffer, pScaleUb, runInfo.actVecMSize, runInfo.actSingleLoopS2Size,
                    pseStride, slopes, posShift, static_cast<T>(constInfo.scaleValue), descaleQK, negativeFloatScalar,
                    0.0F, queryScaleUb, deSCaleKValue);
            } else {
                if constexpr (s2BaseSize == 256) {
                    ProcessVec1Vf<T, INPUT_T, pseShiftType, true, mBaseSize, s2BaseSize, GT_128_AND_LTE_256, hasAtten,
                                  pseMode, hasDrop>(stage1CastTensor, nullptr, sumUb, maxUb, mmRes, expUb,
                                                    sumUb, maxUb, attenMaskUb, pseUb, dropMaskUb, apiTmpBuffer, expUb,
                                                    runInfo.actVecMSize, runInfo.actSingleLoopS2Size, pseStride, slopes,
                                                    posShift, static_cast<T>(constInfo.scaleValue), descaleQK,
                                                    negativeFloatScalar, 0.0F);
                }
            }
        }
        bmm1ResBuf.SetCrossCore();
        if constexpr (hasAtten) {
            this->attenMaskInQue[runInfo.loop % DB].template FreeTensor(attenMaskUb);
        }

        // ===================DataCopy to L1 ====================
        this->stage1OutQue[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue[stage1Offset].template DeQue<INPUT_T>();
        LocalTensor<INPUT_T> mm2AL1Tensor;
        mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();

        if (likely(runInfo.actVecMSize != 0)) {
            DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (blockBytes / sizeof(INPUT_T)) *
                (runInfo.actMSize - runInfo.actVecMSize)], stage1CastTensor,
                {s2BaseSize / 16, (uint16_t)runInfo.actVecMSize, (uint16_t)(vec1Srcstride - runInfo.actVecMSize),
                (uint16_t)(mBaseSize - runInfo.actVecMSize)});
        }
        this->stage1OutQue[stage1Offset].template FreeTensor(stage1CastTensor);

        outputBuf.SetCrossCore();
        // ======================================================

        if (!runInfo.isFirstS2Loop) {
            UpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, runInfo.actVecMSize);
        }

        if (unlikely(runInfo.isLastS2Loop)) {
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
        }
    }

    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        ProcessVec2OnUb(bmm2ResBuf, runInfo);
        return;
    }

    __aicore__ inline void ProcessVec2OnUb(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf,
                                           RunInfoX runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }

        int64_t vec2CalcSize = runInfo.actVecMSize * dTemplateAlign64;
        LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId[0]);
        if (unlikely(runInfo.isFirstS2Loop)) {
            DataCopy(vec2ResUb, mmRes, vec2CalcSize);
        } else {
            LocalTensor<T> expUb = softmaxExpBuf[runInfo.loop % (PRELOAD_N + 1)].template Get<T>();
            LocalTensor<T> pScaleUb;

            if (!runInfo.isLastS2Loop) {
                FlashUpdateNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, runInfo.actVecMSize, dTemplateAlign64, 1.0, 1.0);
            } else {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                FlashUpdateLastNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false, false>(
                    vec2ResUb, mmRes, vec2ResUb, expUb, pScaleUb, sumUb, runInfo.actVecMSize,
                    dTemplateAlign64, 1.0, 1.0);
            }
        }
        bmm2ResBuf.SetCrossCore();
        if (runInfo.isLastS2Loop) {
            if (unlikely(runInfo.isFirstS2Loop)) {
                LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>();
                LastDivNew<T, INPUT_T, OUTPUT_T, dTemplateAlign64, false>(
                    vec2ResUb, vec2ResUb, sumUb, runInfo.actVecMSize, (uint16_t)dTemplateAlign64,
                    0.0F);
            }
            CopyOutAttentionOut(runInfo, vec2ResUb, 0, runInfo.actVecMSize);
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
        attenOut.SetAddr(vec2ResUb.address_);

        int64_t dSizeAligned64 = static_cast<int64_t>(dVTemplateType);
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }

        RowInvalid(vec2ResUb, mStartVec, mDealSize, runInfo, dSizeAligned64);
        Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, mDealSize * dSizeAligned64);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id[0]);
        Bmm2DataCopyOutTrans(runInfo, attenOut, mStartVec, mDealSize);
    }

    __aicore__ inline bool CalcBlockNeedRowInvalid(RunInfoX &runInfo, int64_t s1FirstValidToken,
                                                   int64_t s1LastValidToken)
    {
        int32_t vecMStartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        int32_t vecMEndIdx = vecMStartIdx + runInfo.actVecMSize - 1;
        int32_t s1StartTdx, s1EndTdx;
        bool ret = false;
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH ||
                      layout == LayOutTypeEnum::LAYOUT_TND) {
            // S1G layout
            s1StartTdx = vecMStartIdx / constInfo.gSize;
            s1EndTdx = vecMEndIdx / constInfo.gSize;
            ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
        } else {
            // GS1 layout
            s1StartTdx = vecMStartIdx % runInfo.actS1Size;
            s1EndTdx = vecMEndIdx % runInfo.actS1Size;
            int32_t gStartIdx = vecMStartIdx / runInfo.actS1Size;
            int32_t gEndIdx = vecMEndIdx / runInfo.actS1Size;
            if (gStartIdx != gEndIdx) {  // 跨多个G
                ret = (s1FirstValidToken > 0) || (s1LastValidToken < (runInfo.actS1Size - 1));
            } else {                     // 只跨1个G
                ret = (s1StartTdx < s1FirstValidToken) || (s1EndTdx > s1LastValidToken);
            }
        }
        return ret;
    }

    template <typename VEC2_RES_T>
    __aicore__ inline void RowInvalid(LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t mStartVec,
                                      int64_t mDealSize, RunInfoX &runInfo, int64_t dSizeAligned64)
    {
        if constexpr (hasAtten) {
            int64_t s1FirstValidToken = AttentionCommon::Min(AttentionCommon::Max(-runInfo.nextTokensLeftUp, 0), runInfo.actS1Size);
            int64_t s1LastValidToken = AttentionCommon::Min(AttentionCommon::Max(runInfo.preTokensLeftUp + runInfo.actS2Size, 0), runInfo.actS1Size);
            s1LastValidToken = AttentionCommon::Max(s1LastValidToken - 1, 0);
            bool hasValidRow = (s1FirstValidToken > 0) || (s1LastValidToken < runInfo.actS1Size);
            bool batchNeedRowInvalid = constInfo.isRowInvalidOpen || // 手动开启行无效
                                       ((constInfo.sparseMode != SparseMode::LEFT_UP_CAUSAL) &&
                                       hasValidRow); // sparse = 0 or 3 or 4，preTokens or nextTokens负数
            if (!batchNeedRowInvalid) {
                return;
            }
            bool blockNeedRowInvalid = CalcBlockNeedRowInvalid(runInfo, s1FirstValidToken, s1LastValidToken);
            blockNeedRowInvalid = blockNeedRowInvalid || constInfo.isRowInvalidOpen;
            if (blockNeedRowInvalid) {
                LocalTensor<float> maxTensor =
                    softmaxMaxBuf[runInfo.mloop % (PRELOAD_N + 1)].template Get<float>()[mStartVec];
                RowInvalidUpdateVF<float>(vec2ResUb, maxTensor, mDealSize, constInfo.dSizeV,
                    static_cast<uint32_t>(dSizeAligned64));
            }
        }
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfoX &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        GmCoord gmCoord{.bIdx = info.bIdx,
                        .n2Idx = info.n2Idx,
                        .gS1Idx = (info.gS1Idx + info.vecMbaseIdx + vecMIdx),
                        .dIdx = 0,
                        .gS1DealSize = dealRowCount,
                        .dDealSize = (uint32_t)constInfo.dSizeV};
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb,
                                      .rowCount = dealRowCount,
                                      .colCount = (uint32_t)(splitD ? constInfo.dBasicBlock : dTemplateAlign64)};
        CopyAttentionOut(ubTensor, gmCoord);
    }

    __aicore__ inline void CopyAttentionOut(FaUbTensor<OUTPUT_T> &ubTensor, GmCoord &gmCoord)
    {
        if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BSNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                              constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
            constexpr GmFormat OUT_FORMAT = GmFormat::TNGD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
            CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, GetOutUbFormat<layout>()> copyAttenOutUbToGm;
            copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
            constexpr GmFormat OUT_FORMAT = GmFormat::NGTD;
            FaGmTensor<OUTPUT_T, OUT_FORMAT> outGmTensor;
            outGmTensor.gmTensor = attentionOutGm;
            outGmTensor.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                                              constInfo.actualSeqLenSize);
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
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVTemplateType;
        if constexpr (splitD) {
            dSizeAligned64 = constInfo.dBasicBlock;
        }
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

        tPipe->InitBuffer(maxBrdcst, 1, 2048); // [64, 8]
        tPipe->InitBuffer(sumBrdcst, 1, 2048); // [64, 8]
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t mm1ResultSize = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
        uint32_t mm2ResultSize = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
        if constexpr (!bmm2Write2Ub) {
            tPipe->InitBuffer(mm2InBuf, 32768); // bmm2结果在Gm，vector2开启多层循环，每次处理32KB
        }
        if constexpr (s2BaseSize == 256) {    // mBaseSize = 128
            if constexpr (mBaseSize == 128) { // mBaseSize = 128 s2BaseSize = 256
                tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
                SoftmaxInitBuffer();
                tPipe->InitBuffer(stage1OutQue[0], 1, 33024);
                tPipe->InitBuffer(stage1OutQue[1], 1, 33024);
            } else { // mBaseSize = 64 s2BaseSize = 256
                SoftmaxInitBuffer();
                tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
                tPipe->InitBuffer(stage2OutBuf, 32 * dTemplateAlign64 * sizeof(T));
                tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
                tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
                if constexpr (hasAtten) {
                    tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
                    tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
                }
            }
        } else {
            SoftmaxInitBuffer();
            if constexpr (!useDn) {
                if constexpr (hasAtten) {
                    tPipe->InitBuffer(attenMaskInQue[0], 1, 8192);
                    tPipe->InitBuffer(attenMaskInQue[1], 1, 8192);
                }

                tPipe->InitBuffer(commonTBuf, 512); // 实际上只需要512Bytes
            }
            if constexpr (bmm2Write2Ub) {
                // 小于128Bmm2结果和Vec2结果都在UB
                tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            } else if constexpr (dTemplateAlign64 <= 256) {
                // bmm2结果在Gm，Vector2结果在UB，开启多层循环，每次处理32KB
                tPipe->InitBuffer(stage2OutBuf, 64 * dTemplateAlign64 * sizeof(T));
            } else {
                // bmm2结果在Gm，Vector2结果也在Gm，开启多层循环，每次处理32KB
                tPipe->InitBuffer(stage2OutBuf, 32768);
            }

            tPipe->InitBuffer(stage1OutQue[0], 1, 16896);
            tPipe->InitBuffer(stage1OutQue[1], 1, 16896);
        }

        if (constInfo.isSoftmaxLseEnable) {
            // 8: 适配TND，每行的结果存为8个重复lse元素（32B对齐）
            this->tPipe->InitBuffer(softmaxLseQueue, 1, (mBaseSize >> 1U) * sizeof(float) * 8);
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

    __aicore__ inline void AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, uint32_t vecMIdx, uint32_t mDealSize,
        RunInfoX &runInfo)
    {
        MaskInfo maskInfo;
        maskInfo.gs1StartIdx = runInfo.gS1Idx + runInfo.vecMbaseIdx + vecMIdx;
        maskInfo.gs1dealNum = mDealSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
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
            LocalTensor<uint8_t> attenMaskUbPre = this->attenMaskInQue[1 - runInfo.loop % DB].template AllocTensor<uint8_t>();
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt, maskInfo, true);
            MergeMask(attenMaskUb, attenMaskUbPre, maskInfo.gs1dealNum, s2BaseSize);
            this->attenMaskInQue[1 - runInfo.loop % DB].template FreeTensor(attenMaskUbPre);
        }
    }

    __aicore__ inline void DealZeroActSeqLen(uint32_t bN2Cur)
    {
        uint32_t n2Idx = bN2Cur % constInfo.n2Size;
        uint32_t bIdx = bN2Cur / constInfo.n2Size;
        // 对整个batch的结果置0
        if (constInfo.outputLayout == FIA_LAYOUT::BSH) {
            OffsetCalculator<GmFormat::BSNGD> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            DealActSeqLenIsZero<GmFormat::BSNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::BNSD) {
            OffsetCalculator<GmFormat::BNGSD> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            DealActSeqLenIsZero<GmFormat::BNGSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NBSD) {
            OffsetCalculator<GmFormat::NGBSD> offsetCalculator;
            offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                constInfo.dSizeV, actualSeqLengthsGmQ, constInfo.actualSeqLenSize);
            DealActSeqLenIsZero<GmFormat::NGBSD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::TND) {
            OffsetCalculator<GmFormat::TNGD> offsetCalculator;
            offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                constInfo.actualSeqLenSize);
            DealActSeqLenIsZero<GmFormat::TNGD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
        } else if (constInfo.outputLayout == FIA_LAYOUT::NTD) {
            OffsetCalculator<GmFormat::NGTD> offsetCalculator;
            offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSizeV, actualSeqLengthsGmQ,
                constInfo.actualSeqLenSize);
            DealActSeqLenIsZero<GmFormat::NGTD, OUTPUT_T>(bIdx, n2Idx, offsetCalculator, attentionOutGm);
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
class FANoQuantGqaBlockVecDummy {
public:
    static constexpr bool HAS_MASK = hasAtten;
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoNoQuant = ConstInfo_t<FiaKernelType::NO_QUANT>;
    __aicore__ inline FANoQuantGqaBlockVecDummy(ConstInfoNoQuant &constInfo) {};
};

} // namespace BaseApi

#endif // FIA_BLOCK_VEC_NOQUANT_GQA_H_
