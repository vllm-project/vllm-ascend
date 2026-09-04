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
 * \file flash_attention_score_block_cube_mla_fullquant.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_CUBE_MLA_FULLQUANT_H_
#define FLASH_ATTENTION_SCORE_BLOCK_CUBE_MLA_FULLQUANT_H_
#include "util_regbase.h"
#include "../offset_calculator.h"
#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../CopyInL1.h"

#include "infer_flash_attention_comm.h"
#include "kernel_operator_list_tensor_intf.h"
#include "flash_attention_score_common_regbase.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;

namespace BaseApi {
namespace BlockCubeMlaFullquant {
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNSD;
    } else {
        return GmFormat::NTD;
    }
}

/* ============确定Query的L1类型============= */
template <typename INPUT_T, uint32_t dBaseSize>
struct QL1BuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float> ||
        (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, int8_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>) && dBaseSize > 256),
        BuffersPolicySingleBuffer<BufferType::L1>,
        BuffersPolicyDB<BufferType::L1>>;
};

/* ============确定Key的L1类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSel {
    constexpr static bool isFP8DType =  
            std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>;
    constexpr static bool isINT8DType = std::is_same_v<INPUT_T, int8_t>;
    using Type = std::conditional_t<
            ((isFP8DType || isINT8DType) && s2BaseSize == 128 && dBaseSize == 576),
            BuffersPolicy4buff<BufferType::L1>,
            std::conditional_t<
                (!(isFP8DType || isINT8DType) && s2BaseSize == 256 && dBaseSize > 128),
                BuffersPolicySingleBuffer<BufferType::L1>, BuffersPolicyDB<BufferType::L1>
            >
        >;
};

/* ============确定L0A的类型============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>,
        BuffersPolicySingleBuffer<BufferType::L0A>,
        BuffersPolicyDB<BufferType::L0A>>;
};

/* ============确定L0B的类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float> || (s2BaseSize == 256 && dBaseSize > 128 && 
        !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> || std::is_same_v<INPUT_T, fp8_e5m2_t> ||
        std::is_same_v<INPUT_T, int8_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
        BuffersPolicySingleBuffer<BufferType::L0B>,
        BuffersPolicyDB<BufferType::L0B>>;
};

/* ============确定L0C的类型============= */
template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSel {
    using Type = std::conditional_t<
        (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 &&
        s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
        BuffersPolicy4buff<BufferType::L0C>,
        BuffersPolicyDB<BufferType::L0C>>;
};
}

TEMPLATES_DEF
class FABlockCubeMlaFullquant {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr uint32_t s2SplitSize = (uint32_t)256;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool isMlaFullQuant = (isFp8 || isInt8) && hasRope;
    static constexpr bool useDn = false;
    static constexpr bool useNz = false;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    using ROPE_T = std::conditional_t<isMlaFullQuant, bfloat16_t, INPUT_T>;
    using MLA_FULLQUANT_MM2_T = std::conditional_t<isMlaFullQuant && isInt8, int32_t, T>;
    static constexpr TPosition bmm2OutPos = GetC2Position(dVTemplateType,
                                                          UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop, hasRope,
                                                                                s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64), isMlaFullQuant);
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub};
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;

    __aicore__ inline FABlockCubeMlaFullquant() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable, 
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
        CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
    __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *deqScaleQ,
        __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV) {};

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(CVSharedParams<isInfer, isPa> *sharedParams,
        __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr);
    __aicore__ inline void CalcS1Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CalcS2Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetKvByTensorList(RunInfo<isInfer> &runInfo, const ConstInfo<isInfer, hasRope> &constInfo,
        GlobalTensor<INPUT_T> &keyValueGm, GlobalTensor<INPUT_T> &tempKeyValueGm);
    __aicore__ inline GlobalTensor<INPUT_T> GetKeyGm(RunInfo<isInfer> &runInfo,
                                                     ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline GlobalTensor<INPUT_T> GetValueGm(RunInfo<isInfer> &runInfo,
                                                       ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm1MLAFullQuant(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void MatmulFromValues(const float avalue, const float bvalue,
        const LocalTensor<T> &cL0Tensor, const MMParam &param, RunInfo<isInfer> &runInfo);
    __aicore__ inline void IterateBmm1MLAFullQuantInt8(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2MLAFullQuant(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline bool IsGS1Merge(ConstInfo<isInfer, hasRope> &constInfo);

    TPipe *tPipe;

    /* =====================GM变量==================== */
    __gm__ uint8_t *currentKey; // pageattention需要
    __gm__ uint8_t *currentValue; // pageattention需要
    __gm__ uint8_t *blocktablePtr; // pageattention需要
    GlobalTensor<int32_t> blockTableGm; // pageattention需要
    static constexpr GmFormat Q_FORMAT = BlockCubeMlaFullquant::GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = BlockCubeMlaFullquant::GetKVGmFormat<layout>();
    FaGmTensor<INPUT_T, Q_FORMAT> queryGm;
    FaGmTensor<INPUT_T, KV_FORMAT> keyGm;
    FaGmTensor<INPUT_T, KV_FORMAT> valueGm;
    FaGmTensor<INPUT_T, KV_FORMAT> keySharedPrefixGm;
    FaGmTensor<INPUT_T, KV_FORMAT> valueSharedPrefixGm;
    FaGmTensor<ROPE_T, Q_FORMAT> queryRopeGm;
    FaGmTensor<ROPE_T, KV_FORMAT> keyRopeGm;
    GlobalTensor<float> deScaleQGm;
    GlobalTensor<float> deScaleKGm;
    GlobalTensor<float> deScaleVGm;

    uint32_t kvCacheBlockSize = 0; // pageattention需要
    uint32_t maxBlockNumPerBatch = 0; // pageattention需要
    KVLAYOUT kvLayout; // pageattention需要

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;
    // 新增BufferManager来管理的bias内存分配
    BufferManager<BufferType::C2> l0biasBufferManager;

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用, GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    typename BlockCubeMlaFullquant::QL1BuffSel<INPUT_T, dBaseSize>::Type l1QBuffers;
    // mm1右矩阵K
    typename BlockCubeMlaFullquant::KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1KBuffers;

    // mm2右矩阵V
    typename BlockCubeMlaFullquant::KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1VBuffers;
    // 存放偏执矩阵L1中的位置
    BuffersPolicySingleBuffer<BufferType::L1> biasBufL1;
    // 存放偏执矩阵C2的物理位置
    BuffersPolicySingleBuffer<BufferType::C2> tmpBufBias;
    // L0A
    using L0AType = typename BlockCubeMlaFullquant::L0ABuffSel<INPUT_T>::Type;
    L0AType mmL0ABuffers;
    // L0B
    using L0BType = typename BlockCubeMlaFullquant::L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;
    L0BType mmL0BBuffers;
    // L0C
    using L0CType = typename BlockCubeMlaFullquant::L0CBuffSel<INPUT_T, s1BaseSize, s2BaseSize, dVBaseSize>::Type;
    L0CType mmL0CBuffers;
    // 新增计算标志位
    bool isFirstIter = true;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::InitCubeBlock(TPipe *pipe,
    BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable,  __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)query);
        if constexpr (!isInfer) {
            this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
            this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
        }
        if constexpr (isPa) {
            blocktablePtr = blockTable;
        }
        if constexpr (hasRope) {
            this->queryRopeGm.gmTensor.SetGlobalBuffer((__gm__ ROPE_T *)queryRope);
            this->keyRopeGm.gmTensor.SetGlobalBuffer((__gm__ ROPE_T *)keyRope);
        }
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *key,
    __gm__ uint8_t *value, CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
    __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen)
{
    if ASCEND_IS_AIC {
        if constexpr (isInfer) {
            if (sharedParams->fromFused) {
                ListTensorDesc keyListTensorDescInit((__gm__ void *)key);
                ListTensorDesc valueListTensorDescInit((__gm__ void *)value);
                currentValue = (__gm__ uint8_t *)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
                currentKey = (__gm__ uint8_t *)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
                if (sharedParams->isKvContinuous == 1) {
                    this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
                    this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
                } else {
                    this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
                    this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
                }
            } else {
                this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
                this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
            }
            if constexpr (enableKVPrefix) {
                this->keySharedPrefixGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)keySharedPrefix);
                this->valueSharedPrefixGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)valueSharedPrefix);
            }
            attenMaskInfo->compressMode = sharedParams->compressMode;
            attenMaskInfo->attenMaskS1Size = sharedParams->attenMaskS1Size;
            attenMaskInfo->attenMaskS2Size = sharedParams->attenMaskS2Size;
            attenMaskInfo->preTokens = sharedParams->preTokens;
            attenMaskInfo->nextTokens = sharedParams->nextTokens;
            if constexpr (isPa) {
                this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blocktablePtr);
                this->kvCacheBlockSize = sharedParams->blockSize;
                this->maxBlockNumPerBatch = sharedParams->blockTableDim2;
                if (sharedParams->paLayoutType == 2) { // NZ下paLayoutType == 2
                    kvLayout = KVLAYOUT::NZ;
                } else {
                    kvLayout = sharedParams->paLayoutType == 1 ?
                               KVLAYOUT::BBH : KVLAYOUT::BNBD;
                }
            }
        }
        InitGmTensor(sharedParams, actualSeqQlenAddr, actualSeqKvlenAddr);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::InitLocalBuffer()
{
    if constexpr (isMlaFullQuant) {
        constexpr uint32_t dRopeBaseSize = dBaseSize - dVBaseSize;
        constexpr uint32_t mm1QSize = s1BaseSize * dVBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1QRopeSize = s1BaseSize * dRopeBaseSize * sizeof(bfloat16_t);
        constexpr uint32_t mm1KSize = dVBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1KRopeSize =  dRopeBaseSize * s2BaseSize * sizeof(bfloat16_t);
        constexpr uint32_t mm1PSize =  s1BaseSize * s2BaseSize * sizeof(INPUT_T);
        // 新增
        constexpr uint32_t biasSize =  BT_PP_SIZE * KB_TO_BYTES; // 4k
        constexpr uint32_t avalSize =  BT_AVAL_SIZE * KB_TO_BYTES; // 16k
        constexpr uint32_t bvalSize =  BT_BVAL_SIZE * KB_TO_BYTES; // 16k

        l1QBuffers.Init((*l1BufferManagerPtr), mm1QSize + mm1QRopeSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1KSize + mm1KRopeSize);

        // 在L0上初始化buffer
        l0aBufferManager.Init(tPipe, MLA_L0A_SIZE * KB_TO_BYTES); //MLA_L0A_SIZE =64
        l0bBufferManager.Init(tPipe, MLA_L0B_SIZE * KB_TO_BYTES); //MLA_L0B_SIZE =64
        l0cBufferManager.Init(tPipe, L0C_SIZE *KB_TO_BYTES);
        l0biasBufferManager.Init(tPipe, BT_PP_SIZE * KB_TO_BYTES);
        mmL0ABuffers.Init(l0aBufferManager, (MLA_L0A_SIZE / NUM_2) * KB_TO_BYTES);
        mmL0BBuffers.Init(l0bBufferManager, (MLA_L0B_SIZE / NUM_2) * KB_TO_BYTES);
        mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
        if constexpr (isInt8 && isMlaFullQuant) {
            // 新增初始化bias的buffer
            biasBufL1.Init((*l1BufferManagerPtr), biasSize + avalSize + bvalSize);
            // 新增偏执矩阵C2的buffer
            tmpBufBias.Init(l0biasBufferManager, BT_PP_SIZE * KB_TO_BYTES);
        }
    }
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::InitGmTensor(
    CVSharedParams<isInfer, isPa> *cvSharedParams, __gm__ int64_t *actualSeqQlenAddr,
    __gm__ int64_t *actualSeqKvlenAddr)
{
    if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
        this->queryGm.offsetCalculator.Init(cvSharedParams->bSize, cvSharedParams->n2Size, cvSharedParams->gSize,
            cvSharedParams->s1Size, cvSharedParams->dSize);
        if constexpr (hasRope) {
            this->queryRopeGm.offsetCalculator.Init(cvSharedParams->bSize, cvSharedParams->n2Size,
                                                    cvSharedParams->gSize, cvSharedParams->s1Size,
                                                    cvSharedParams->dSizeRope);
        }
    } else {  // GM_Q_OUT_TND
        GlobalTensor<uint64_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr);
        if constexpr (isInfer) {
            this->queryGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->gSize, cvSharedParams->dSize,
                actualSeqQLen, cvSharedParams->actualSeqLengthsSize);
            if constexpr (hasRope) {
                this->queryRopeGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->gSize,
                    cvSharedParams->dSizeRope, actualSeqQLen, cvSharedParams->actualSeqLengthsSize);
            }
        } else {
            this->queryGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->gSize, cvSharedParams->dSize,
                actualSeqQLen, cvSharedParams->bSize);
            if constexpr (hasRope) {
                this->queryRopeGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->gSize,
                    cvSharedParams->dSizeRope, actualSeqQLen, cvSharedParams->bSize);
            }
        }
    }
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        this->keyGm.offsetCalculator.Init(cvSharedParams->bSize, cvSharedParams->n2Size, cvSharedParams->s2Size,
            cvSharedParams->dSize);
        this->valueGm.offsetCalculator.Init(cvSharedParams->bSize, cvSharedParams->n2Size, cvSharedParams->s2Size,
            cvSharedParams->dSizeV);
        if constexpr (enableKVPrefix) {
            this->keySharedPrefixGm.offsetCalculator.Init(1, cvSharedParams->n2Size, cvSharedParams->kvPrefixSize,
                                                          cvSharedParams->dSize);
            this->valueSharedPrefixGm.offsetCalculator.Init(1, cvSharedParams->n2Size, cvSharedParams->kvPrefixSize,
                                                            cvSharedParams->dSizeV);
        }
        if constexpr (hasRope) {
            this->keyRopeGm.offsetCalculator.Init(cvSharedParams->bSize, cvSharedParams->n2Size, cvSharedParams->s2Size,
                cvSharedParams->dSizeRope);
        }
    } else { // GM_KV_TND
        GlobalTensor<uint64_t> actualSeqKVLen;
        actualSeqKVLen.SetGlobalBuffer((__gm__ uint64_t *)actualSeqKvlenAddr);
        if constexpr (isInfer) {
            this->keyGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->dSize, actualSeqKVLen,
                cvSharedParams->actualSeqLengthsKVSize);
            this->valueGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->dSizeV, actualSeqKVLen,
                cvSharedParams->actualSeqLengthsKVSize);
            if constexpr (hasRope) {
                this->keyRopeGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->dSizeRope, actualSeqKVLen,
                    cvSharedParams->actualSeqLengthsKVSize);
            }
        } else {
            this->keyGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->dSize, actualSeqKVLen,
                cvSharedParams->bSize);
            this->valueGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->dSizeV, actualSeqKVLen,
                cvSharedParams->bSize);
            if constexpr (hasRope) {
                this->keyRopeGm.offsetCalculator.Init(cvSharedParams->n2Size, cvSharedParams->dSizeRope, actualSeqKVLen,
                    cvSharedParams->bSize);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::CalcS1Coord(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    // 计算s1方向偏移
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * s1BaseSize;
    if constexpr (isInfer) {
        coordInfo[runInfo.taskIdMod3].s1Coord += runInfo.queryLeftPaddingSize;  // 左padding
        // 推理无效行场景，s1方向起始跳过无效行
        coordInfo[runInfo.taskIdMod3].s1Coord += (runInfo.nextTokensPerBatch < 0) ? -runInfo.nextTokensPerBatch : 0;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::CalcS2Coord(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
    coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.boIdx;
    if constexpr (isInfer) {
        coordInfo[runInfo.taskIdMod3].s2Coord += runInfo.kvLeftPaddingSize;  // 左padding
        if constexpr (isFd) {
            coordInfo[runInfo.taskIdMod3].s2Coord += constInfo.sInnerLoopSize * runInfo.flashDecodeS2Idx;
        }
        if (constInfo.isKvContinuous == 0) {
            coordInfo[runInfo.taskIdMod3].curBIdx = 0;
            if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
                // 更新N2方向stride
                this->keyGm.offsetCalculator.Init(0, constInfo.n2Size, runInfo.s2InCurrentBatch,
                                                  constInfo.dSize);
                if constexpr (hasRope) {
                    this->keyRopeGm.offsetCalculator.Init(0, constInfo.n2Size, runInfo.s2InCurrentBatch,
                                                          constInfo.dSizeRope);
                }
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    CalcS1Coord(runInfo, constInfo);
    CalcS2Coord(runInfo, constInfo);

    if constexpr (isFp8) {
        IterateBmm1MLAFullQuant(outputBuf,  runInfo, constInfo);
    } else if constexpr(isInt8) {
        IterateBmm1MLAFullQuantInt8(outputBuf,  runInfo, constInfo);
    }

}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::IterateBmm2(mm2ResPos &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isMlaFullQuant) {
        IterateBmm2MLAFullQuant(outputBuf, inputBuf, runInfo, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::GetKvByTensorList(RunInfo<isInfer>& runInfo, 
    const ConstInfo<isInfer, hasRope> &constInfo,
    GlobalTensor<INPUT_T>& keyValueGm, GlobalTensor<INPUT_T>& tempKeyValueGm)
{
    if (constInfo.isKvContinuous != 0) {
        return;
    }
    ListTensorDesc kvListTensorDesc((__gm__ void*)keyValueGm.GetPhyAddr());
    __gm__ uint8_t* tempkvGmPtr =
        (__gm__ uint8_t*)kvListTensorDesc.GetDataPtr<__gm__ uint8_t>(runInfo.boIdx);
    tempKeyValueGm.SetGlobalBuffer((__gm__ INPUT_T*)tempkvGmPtr);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_T> FABlockCubeMlaFullquant<TEMPLATE_ARGS>::GetKeyGm(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer) {
        GlobalTensor<INPUT_T> tmpKeyGm = this->keyGm.gmTensor;
        GetKvByTensorList(runInfo, constInfo, this->keyGm.gmTensor, tmpKeyGm);
        return tmpKeyGm;
    } else {
        return this->keyGm.gmTensor;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_T> FABlockCubeMlaFullquant<TEMPLATE_ARGS>::GetValueGm(
    RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer) {
        GlobalTensor<INPUT_T> tmpValueGm = this->valueGm.gmTensor;
        GetKvByTensorList(runInfo, constInfo, this->valueGm.gmTensor, tmpValueGm);
        return tmpValueGm;
    } else {
        return this->valueGm.gmTensor;
    }
}

/* 针对MLA的bmm1*/
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::IterateBmm1MLAFullQuant(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    uint32_t dTypeRATIO = sizeof(bfloat16_t)/sizeof(INPUT_T);
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    uint32_t dstNzC0StrideQNope = (runInfo.s1RealSize + 31) >> 5 << 5;
    uint32_t offsetQRopeByElement = dstNzC0StrideQNope * constInfo.dSize / dTypeRATIO; // Rope在mm1A的偏移量（单位：元素）
    // 左矩阵复用 ,s2的第一次循环加载左矩阵
    // 加载左矩阵到L1 当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本快：搬运0
        mm1A = l1QBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用，MTE2开始
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, this->queryGm.gmTensor[runInfo.queryOffset],
            runInfo.s1RealSize, constInfo.dSize, constInfo.mm1Ka);

        LocalTensor<bfloat16_t> mm1ARopeTensor  = mm1A.GetTensor<bfloat16_t>(offsetQRopeByElement); 
        CopyToL1Nd2Nz<bfloat16_t>(mm1ARopeTensor, this->queryRopeGm.gmTensor[runInfo.qRopeOffset],
            runInfo.s1RealSize, constInfo.dSizeRope, constInfo.mm1RopeKa);
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非s2的第一次循环直接复用Q
        mm1A = l1QBuffers.GetPre();
        // 左矩阵复用时，s2循环内不需要MTE2同步等待
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知 
    }
    // 加载当前轮的右矩阵到L1
    mm1B = l1KBuffers.Get();
    mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用，MTE2开始
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
    uint32_t dstNzC0StrideK = (runInfo.s2RealSize + 31) >> 5 << 5;
    uint32_t offsetKRopeByElement = dstNzC0StrideK * constInfo.dSize / dTypeRATIO; // Rope在mm1A的偏移量（单位：元素）
    LocalTensor<bfloat16_t> mm1BRopeTensor  = mm1B.GetTensor<bfloat16_t>(offsetKRopeByElement);

    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.dIdx = 0;
        if constexpr (isFd) {
            startPos.s2Offset = runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize +  // FD分片起始
                        runInfo.s2LoopCount * s2BaseSize;  // 核心内循环偏移
        } else {
            startPos.s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;  // 非FD场景
        }
        PAShape nopeShape;//配置KNope的PA搬运参数
        nopeShape.blockSize = kvCacheBlockSize;
        nopeShape.headNum = constInfo.n2Size;
        nopeShape.headDim = constInfo.dSize;
        nopeShape.actHeadDim = constInfo.dSize;
        nopeShape.maxblockNumPerBatch = maxBlockNumPerBatch;
        nopeShape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5; // 31, 5: nope为Fp8，需要对齐到32
        nopeShape.copyRowNum = runInfo.s2RealSize;
        GlobalTensor<INPUT_T> mm1BGmTensor = this->keyGm.gmTensor;
        PAShape ropeShape = nopeShape; //配置KRope的PA搬运参数
        ropeShape.headDim = constInfo.dSizeRope;
        ropeShape.actHeadDim = constInfo.dSizeRope;
        ropeShape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4; // 15, 4: rope为Bf16，需要对齐到16
        GlobalTensor<bfloat16_t> mm1BRopeGm = this->keyRopeGm.gmTensor;
        //先搬运KNope,再搬运Rope
        GmCopyInToL1PA<INPUT_T>(mm1BTensor, mm1BGmTensor, blockTableGm, kvLayout, nopeShape, startPos);
        GmCopyInToL1PA<bfloat16_t>(mm1BRopeTensor, mm1BRopeGm, blockTableGm, kvLayout, ropeShape, startPos);
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx,
            runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        uint64_t gmRopeOffset = this->keyRopeGm.offsetCalculator.GetOffset(runInfo.boIdx,
            runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        //先搬运KNope,再搬运Rope
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
            constInfo.dSize, constInfo.mm1Kb);
        CopyToL1Nd2Nz<bfloat16_t>(mm1BRopeTensor, this->keyRopeGm.gmTensor[gmRopeOffset],
                                  runInfo.s2RealSize, constInfo.dSizeRope, constInfo.mm1RopeKb);
    }
    mm1B.Set<HardEvent::MTE2_MTE1>(); // MTE2结束，通知
    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A，MTE1开始，准备Matmul
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用,mmad开始
    // Nope的MatMul;其中，NopeBaseM可优化为s1RealSize
    MMParam param = {(uint32_t)runInfo.s1RealSize, (uint32_t)runInfo.s2RealSize, (uint32_t)(constInfo.dSize),
                     0,    // isLeftTranspose
                     1     // isRightTranspose 
                    };

    MatmulK<INPUT_T, INPUT_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
                mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(),
                param);
    // Rope的MatMul
    MMParam paramRope = {(uint32_t)runInfo.s1RealSize,
                         (uint32_t)runInfo.s2RealSize,
                         (uint32_t)(constInfo.dSizeRope),
                         0,    // isLeftTranspose
                         1,    // isRightTranspose
                         1,
                         0 // 累加Nope的L0C
                        };
    MatmulFull<bfloat16_t, bfloat16_t, T, 64, 128, 64, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<bfloat16_t>(offsetQRopeByElement), mm1B.GetTensor<bfloat16_t>(offsetKRopeByElement),
        mmL0ABuffers, mmL0BBuffers, mm1ResL0C.GetTensor<T>(), paramRope);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>(); //S2循环中，Q常驻L1
    }
    mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放 L1B
    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipec310Params; // L0C->UB
    fixpipec310Params.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipec310Params.mSize = (runInfo.s1RealSize + 1) >> 1 << 1;
    // 源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipec310Params.srcStride =
        (fixpipec310Params.mSize + 15) >> 4 << 4;  // 15, 4: L0C上matmul结果相邻连续数据片断间隔, 单位为16 *sizeof(T) ，对齐到16
    fixpipec310Params.dstStride =
        s2BaseSize;  // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipec310Params.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipec310Params.params.ndNum = 1;
    fixpipec310Params.params.srcNdStride = 0;
    fixpipec310Params.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(),
                                        fixpipec310Params);  // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>();
    outputBuf.SetCrossCore();
}

//MLA全量化新增的bmm2,L1上切N
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::IterateBmm2MLAFullQuant(mm2ResPos &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo) 
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();
    mm2A.WaitCrossCore();

    if constexpr (bmm2Write2Ub) {
        outputBuf.WaitCrossCore();
    }
    Buffer<BufferType::L1> mm2B = l1KBuffers.GetReused();
    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)s1BaseSize,  // singleM
                        (uint32_t)constInfo.dSizeV, // singleN
                        (uint32_t)runInfo.s2RealSize,  // singleK
                        false,    // isLeftTranspose
                        false     // isRightTranspose
                    };
    MatmulN<INPUT_T, INPUT_T, MLA_FULLQUANT_MM2_T, 128, 256, 128, ABLayout::MK, ABLayout::KN>(
        mm2A.GetTensor<INPUT_T>(),
        mm2B.GetTensor<INPUT_T>(),
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<MLA_FULLQUANT_MM2_T>(),
        param);

    mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    if constexpr (bmm2Write2Ub) {
        fixpipeParams.nSize = ((uint32_t)constInfo.dSizeV + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    } else {
        fixpipeParams.nSize = (uint32_t)constInfo.dSizeV; // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    }
    fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = ((s1BaseSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    if constexpr (bmm2Write2Ub) {
        fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
    } else {
        fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV; // dstGm 两行之间的间隔
    }
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<MLA_FULLQUANT_MM2_T, MLA_FULLQUANT_MM2_T, BMM2_FIXPIPE_CONFIG>(outputBuf.template GetTensor<MLA_FULLQUANT_MM2_T>(), mm2ResL0C.GetTensor<MLA_FULLQUANT_MM2_T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::MatmulFromValues(
const float avalue, const float bvalue, const LocalTensor<T> &cL0Tensor, const MMParam &param, RunInfo<isInfer> &runInfo)
{
    InitConstValueParams<float> paramsA(1, 512, 0, avalue); // n*k*4/32B=128*32*4/32
    InitConstValueParams<float> paramsB(1, 512, 0, bvalue); // n*k*4/32B=128*32*4/32
    // 新增的两个偏移
    uint32_t offsetAval = 1024; // 4k / 4 = 1024
    uint32_t offsetBval = 9216; // (4k+32k) / 4 = 9216

    // 在L1上声明并且初始化
    Buffer<BufferType::L1> ValueBufTemp = biasBufL1.Get();
    ValueBufTemp.Wait<HardEvent::MTE1_MTE2>();
    LocalTensor<float> aValL1Tensor = ValueBufTemp.GetTensor<float>(offsetAval);
    LocalTensor<float> bValL1Tensor = ValueBufTemp.GetTensor<float>(offsetBval);
    InitConstValue(aValL1Tensor, paramsA); // 在L1上初始化
    InitConstValue(bValL1Tensor, paramsB); // 在L1上初始化
    ValueBufTemp.Set<HardEvent::MTE2_MTE1>(); // 设置同步信号，MTE1等待MTE2搬运完
    ValueBufTemp.Wait<HardEvent::MTE2_MTE1>(); // 等待同步，MTE1等待MTE2搬运完，此时代表在L1上已经初始化完成，准备随时可以搬到L0

    // 在L0A上声明
    Buffer<BufferType::L0A> l0aBuffer = mmL0ABuffers.Get();
    LocalTensor<INPUT_T> L0ATensor = l0aBuffer.GetTensor<INPUT_T>();
    LocalTensor<float> L0ATensorFp32 = L0ATensor.template ReinterpretCast<float>(); //不是“类型转换”，而是 “重新解释内存内容”
    // 完成搬运：L1---L0A
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LoadDataToL0A(L0ATensorFp32, aValL1Tensor, param, 0, 32, 128); // 0:偏移为0；32:kSplitSize；128:mSplitSize
    l0aBuffer.Set<HardEvent::MTE1_M>();
    l0aBuffer.Wait<HardEvent::MTE1_M>();

    // 在L0B上声明
    Buffer<BufferType::L0B> l0bBuffer = mmL0BBuffers.Get();
    LocalTensor<INPUT_T> L0BTensor = l0bBuffer.GetTensor<INPUT_T>();
    LocalTensor<float> L0BTensorFp32 = L0BTensor.template ReinterpretCast<float>();
    // 完成搬运：L1---L0B
    l0bBuffer.Wait<HardEvent::M_MTE1>();
    LoadDataToL0B(L0BTensorFp32, bValL1Tensor, param, 0, 32, 128); // 0:偏移为0；32:kSplitSize；128:nSplitSize
    l0bBuffer.Set<HardEvent::MTE1_M>();
    l0bBuffer.Wait<HardEvent::MTE1_M>();
    
    // 开始mmad计算
    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = false;
    mmadParams.cmatrixSource = false;
    Mmad(cL0Tensor, L0ATensorFp32, L0BTensorFp32, mmadParams);
    // 添加一些同步信号，与最开始的wait对应
    l0aBuffer.Set<HardEvent::M_MTE1>();
    l0bBuffer.Set<HardEvent::M_MTE1>();
    ValueBufTemp.Set<HardEvent::MTE1_MTE2>();
}

/* 针对MLA INT8的bmm1*/
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeMlaFullquant<TEMPLATE_ARGS>::IterateBmm1MLAFullQuantInt8(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    uint32_t dTypeRATIO = sizeof(bfloat16_t)/sizeof(INPUT_T);
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    uint32_t dstNzC0StrideQNope = (runInfo.s1RealSize + 31) >> 5 << 5; //字节对齐，向上取整到32字节
    uint32_t offsetQRopeByElement = dstNzC0StrideQNope * constInfo.dSize / dTypeRATIO; //Rope在mm1A的偏移量（单位：元素）
    //********************新增偏执矩阵的计算 */
    Buffer<BufferType::C2> l0biasBuffer = tmpBufBias.Get();
    LocalTensor<int32_t> btC2Tensor = l0biasBuffer.GetTensor<int32_t>();//将在C2上位置
    if (isFirstIter) {
        isFirstIter = false;
        int32_t biasValue = 1260388352; // 1260388352:2^21 + 150 * 2^23
        InitConstValueParams<int32_t> params(1, 128, 0, biasValue); // 4096 / 32B = 128，同时完成一些结构体参数的配置
        Buffer<BufferType::L1> biasBufTemp = biasBufL1.Get();
        biasBufTemp.Wait<HardEvent::MTE1_MTE2>(); // 等待同步，MTE2等待MTE1搬运完
        LocalTensor<int32_t> biasL1Tensor = biasBufTemp.GetTensor<int32_t>(); //在L1中的位置
        InitConstValue(biasL1Tensor, params);// 利用结构体参数来完成对biasTensor（C1上的局部Localtensor）初始化
        biasBufTemp.Set<HardEvent::MTE2_MTE1>(); // 设置同步信号，MTE1等待MTE2搬运完
        biasBufTemp.Wait<HardEvent::MTE2_MTE1>(); // 等待同步，MTE1等待MTE2搬运完
        const uint32_t nSplitSize = 128; // 128:n方向切分，s2basesize = 128
        DataCopy(btC2Tensor, biasL1Tensor, nSplitSize * 2); //因此利用了Datacopy完成了数据从C1---C2的搬运
        biasBufTemp.Set<HardEvent::MTE1_MTE2>();
    }
    //********************结束偏置矩阵的计算 */
     // 左矩阵复用 ,s2的第一次循环加载左矩阵
    // 加载左矩阵到L1 当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本快：搬运0
        mm1A = l1QBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用，MTE2开始
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, this->queryGm.gmTensor[runInfo.queryOffset], runInfo.s1RealSize,
            constInfo.dSize, constInfo.mm1Ka);
 
        LocalTensor<bfloat16_t> mm1ARopeTensor  = mm1A.GetTensor<bfloat16_t>(offsetQRopeByElement); 
        CopyToL1Nd2Nz<bfloat16_t>(mm1ARopeTensor,this->queryRopeGm.gmTensor[runInfo.qRopeOffset], runInfo.s1RealSize,
            constInfo.dSizeRope, constInfo.mm1RopeKa); 
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非s2的第一次循环直接复用Q
        mm1A = l1QBuffers.GetPre();
        // 左矩阵复用时，s2循环内不需要MTE2同步等待
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知 
    }
    // 加载当前轮的右矩阵到L1
    mm1B = l1KBuffers.Get();
    mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用，MTE2开始
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
    uint32_t dstNzC0StrideKNope = (runInfo.s2RealSize + 31) >> 5 << 5;
    uint32_t offsetKRopeByElement = dstNzC0StrideKNope * constInfo.dSize / dTypeRATIO; //Rope在mm1A的偏移量（单位：元素）
    LocalTensor<bfloat16_t> mm1BRopeTensor  = mm1B.GetTensor<bfloat16_t>(offsetKRopeByElement); 
 
    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        if constexpr (isFd) {
            startPos.s2Offset = runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize +  // FD分片起始
                        runInfo.s2LoopCount * s2BaseSize;  // 核心内循环偏移
        } else {
            startPos.s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;  // 非FD场景
        }
        startPos.dIdx = 0;
        PAShape nopeShape;//配置KNope的PA搬运参数
        nopeShape.blockSize = kvCacheBlockSize;
        nopeShape.headNum = constInfo.n2Size;
        nopeShape.headDim = constInfo.dSize;
        nopeShape.actHeadDim = constInfo.dSize;
        nopeShape.maxblockNumPerBatch = maxBlockNumPerBatch;
        nopeShape.copyRowNum = runInfo.s2RealSize;
        nopeShape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5; // 31, 5: nope为Fp8，需要对齐到32
        GlobalTensor<INPUT_T> mm1BGmTensor = this->keyGm.gmTensor;
        PAShape ropeShape = nopeShape; //配置KRope的PA搬运参数
        ropeShape.headDim = constInfo.dSizeRope;
        ropeShape.actHeadDim = constInfo.dSizeRope;
        ropeShape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4; // 15, 4: rope为Bf16，需要对齐到16
        GlobalTensor<bfloat16_t> mm1BRopeGmTensor = this->keyRopeGm.gmTensor;
        //先搬运KNope,再搬运Rope
        GmCopyInToL1PA<INPUT_T>(mm1BTensor, mm1BGmTensor, blockTableGm, kvLayout, nopeShape, startPos);
        GmCopyInToL1PA<bfloat16_t>(mm1BRopeTensor, mm1BRopeGmTensor, blockTableGm, kvLayout, ropeShape, startPos);
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
            coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        uint64_t gmRopeOffset = this->keyRopeGm.offsetCalculator.GetOffset(runInfo.boIdx,
            runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        //先搬运KNope,再搬运Rope
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
            constInfo.dSize, constInfo.mm1Kb);
        CopyToL1Nd2Nz<bfloat16_t>(mm1BRopeTensor, this->keyRopeGm.gmTensor[gmRopeOffset],
                runInfo.s2RealSize, constInfo.dSizeRope, constInfo.mm1RopeKb); 
    }
    mm1B.Set<HardEvent::MTE2_MTE1>(); // MTE2结束，通知
 
    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A，MTE1开始，准备Matmul
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B
 
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用,mmad开始
    // Nope的MatMul;其中，NopeBaseM可优化为s1RealSize
    MMParam param = {(uint32_t)runInfo.s1RealSize,
                     (uint32_t)runInfo.s2RealSize,
                     (uint32_t)(constInfo.dSize), // singleK完整dsize, MatmulK内部会切K
                     0,    // isLeftTranspose
                     1     // isRightTranspose 
                    };
    // 64:S1方向基本块的大小；128:S2方向基本块的大小；256:Q/K沿着K轴Dim切分的大小为256
    MatmulKbias<INPUT_T, INPUT_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
                mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<int32_t>(),
                l0biasBuffer.GetTensor<int32_t>(),
                param);
    LocalTensor<int32_t> cL0Tensor = mm1ResL0C.GetTensor<int32_t>(); // 用int32_t的形式去接收，
    LocalTensor<T> cL0TensorFp32 = cL0Tensor.template ReinterpretCast<T>();//解释为另一种数据类型,解读方式不同
   
    // ROPE 计算
    {
        // 行向量与列向量的matmul
        float aVal = -2097152.0; // -2097152.0:-2^21，int32转fp32的算法固定值
        float bVal = 5.0; // 5.0:int32转fp32的算法固定值
        MMParam paramval = {(uint32_t)runInfo.s1RealSize,
                     (uint32_t)runInfo.s2RealSize,
                     1, // K轴固定为1，完成行向量与列向量的相乘
                     0,    // isLeftTranspose
                     1     // isRightTranspose 
                    };
            
        MatmulFromValues(aVal,bVal,cL0TensorFp32,paramval,runInfo);

        // Rope的MatMul
        MMParam paramRope = {(uint32_t)runInfo.s1RealSize,
                        (uint32_t)runInfo.s2RealSize,
                        (uint32_t)(constInfo.dSizeRope), 
                        0,    // isLeftTranspose
                        1,    // isRightTranspose
                        1,
                        0 //累加Nope的L0C
                        };
        // 64:S1方向基本块的大小；128:S2方向基本块的大小；64:Qrope/Krope在K轴方向的大小
        MatmulFull<bfloat16_t, bfloat16_t, T, 64, 128, 64, ABLayout::MK, ABLayout::KN>(
                    mm1A.GetTensor<bfloat16_t>(offsetQRopeByElement), mm1B.GetTensor<bfloat16_t>(offsetKRopeByElement),
                    mmL0ABuffers, mmL0BBuffers,
                    cL0TensorFp32,
                    paramRope);
        if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
            mm1A.Set<HardEvent::MTE1_MTE2>(); //S2循环中，Q常驻L1
        }
        mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
        mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
        mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
 
        outputBuf.WaitCrossCore();
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
        fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
        fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
        // 源NZ矩阵中相邻Z排布的起始地址偏移
        fixpipeParams.srcStride = (fixpipeParams.mSize + 15) >> 4 << 4; // 15, 4: L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) ，对齐到16
        fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
        fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
 
        Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), cL0TensorFp32, fixpipeParams); // 将matmul结果从L0C搬运到UB
        mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放
        outputBuf.SetCrossCore();
    }              
}

// 判断是否GS1合轴
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool FABlockCubeMlaFullquant<TEMPLATE_ARGS>::IsGS1Merge(ConstInfo<isInfer, hasRope> &constInfo)
{
    return (Q_FORMAT == GmFormat::BSNGD || Q_FORMAT == GmFormat::TNGD) && constInfo.isPfaGS1Merge;
}


TEMPLATES_DEF
class FABlockCubeMlaFullquantDummy {
public:
    static constexpr bool isFp8 = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::isFp8;
    static constexpr bool isInt8 = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::isInt8;
    static constexpr bool useDn = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::useDn;
    static constexpr bool useNz = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::useNz;
    static constexpr TPosition bmm2OutPos = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::bmm2Write2Ub;
    static constexpr bool splitD = FABlockCubeMlaFullquant<TEMPLATE_ARGS>::splitD;
    __aicore__ inline FABlockCubeMlaFullquantDummy() {};
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
                                         CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
                                         __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr,
                                         __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
                                         __gm__ uint8_t *actualSharedPrefixLen) {}
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void IterateBmm1Nz(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void InitDequantParams(__gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK,
        __gm__ uint8_t *deqScaleV) {}
    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };
DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeMlaFullquant);
DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeMlaFullquantDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)\
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_MLA_FULLQUANT_H_
