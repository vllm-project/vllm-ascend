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
 * \file flash_attention_score_block_cube.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
#define FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
#include "util_regbase.h"
#include "../offset_calculator.h"
#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../CopyInL1.h"

#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"
using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat() {
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else {
        return GmFormat::TNGD;
    }
}

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetKVGmFormat() {
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBND;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNSD;
    } else {
        return GmFormat::TND;
    }
}

/* ============确定Query的L1类型============= */
template <typename INPUT_T, uint32_t dBaseSize>
struct QL1BuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float> ||
        (!(std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>) && dBaseSize > 256),
        BuffersPolicySingleBuffer<BufferType::L1>,
        BuffersPolicyDB<BufferType::L1>>;
};

/* ============确定Key的L1类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct KVL1BuffSel {
    using Type = std::conditional_t<
        (!(std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>) && s2BaseSize == 256 && dBaseSize > 128),
        BuffersPolicySingleBuffer<BufferType::L1>,
        BuffersPolicyDB<BufferType::L1>>;
};

/* ============确定L0A的类型============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float>,
        BuffersPolicySingleBuffer<BufferType::L0A>,
        BuffersPolicyDB<BufferType::L0A>>;
};
/* ============确定L0B的类型============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<
        std::is_same_v<INPUT_T, float> || (s2BaseSize == 256 && dBaseSize > 128 && 
        !(std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, fp8_e5m2_t> ||
           std::is_same_v<INPUT_T, hifloat8_t>)),
        BuffersPolicySingleBuffer<BufferType::L0B>,
        BuffersPolicyDB<BufferType::L0B>>;
};
/* ============确定L0C的类型============= */
template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
struct L0CBuffSel {
    using Type = std::conditional_t<
        (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 && s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
        BuffersPolicy4buff<BufferType::L0C>,
        BuffersPolicyDB<BufferType::L0C>>;
};

/* ============确定bmm2ResBuffer的类型============= */
template <bool useDn, bool isFp8>
struct Bmm2ResBuffSel {
    using Type = std::conditional_t<(useDn && isFp8),
        BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>>;
};

TEMPLATES_DEF
class FABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dVBaseSize = (uint32_t)dVTemplateType;
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                                IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr bool isInt8 = IsSameType<INPUT_T, int8_t>::value;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;
    static constexpr bool useDn = IsDn(true, isInt8, pseMode, hasAtten, hasDrop,
                                       s1BaseSize == 128, dTemplateType, hasRope);
    static constexpr TPosition bmm2OutPos = GetC2Position(dVTemplateType,
                                                          UbOutCondition<INPUT_T>(IsSameType<INPUT_T, float>::value, pseMode, hasAtten, hasDrop,
                                                                               s1BaseSize == 64), (s2BaseSize == 256 && s1BaseSize == 64));
    static constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    static constexpr FixpipeConfig BMM1_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, true, !isInt8};
    static constexpr FixpipeConfig BMM2_FIXPIPE_CONFIG = {CO2Layout::ROW_MAJOR, bmm2Write2Ub, !isInt8};
    static constexpr uint32_t l1BaseD = isFp8 ? 256: ((IsSameType<INPUT_T, float>::value) ? (dBaseSize > 128 ? 96 : 128): 128);
    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    GlobalTensor<uint32_t> deqScaleQKGm;
    GlobalTensor<uint32_t> deqScaleVGm;

    __aicore__ inline FABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable, 
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
        CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
        __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *deqScaleQK, __gm__ uint8_t *deqScaleV, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(CVSharedParams<isInfer, isPa> *sharedParams, __gm__ int64_t *actualSeqQlenAddr,
        __gm__ int64_t *actualSeqKvlenAddr);
    __aicore__ inline void CalcS1Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void CalcS2Coord(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void GetKvByTensorList(RunInfo<isInfer> &runInfo, const ConstInfo<isInfer, hasRope> &constInfo,
        GlobalTensor<INPUT_T> &keyValueGm, GlobalTensor<INPUT_T> &tempKeyValueGm);
    __aicore__ inline GlobalTensor<INPUT_T> GetKeyGm(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline GlobalTensor<INPUT_T> GetValueGm(RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void IterateBmm1Nd(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm1NdL0Split(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm1NdL1SplitK(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void IterateBmm1DnSplitK(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm1Dn(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo);

    __aicore__ inline void InitQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *deqScaleQK, __gm__ uint8_t *deqScaleV);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2L1SplitN(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo);

    TPipe *tPipe;
    /* =====================GM变量==================== */
    __gm__ uint8_t *currentKey;    // pageattention需要
    __gm__ uint8_t *currentValue;  // pageattention需要
    __gm__ uint8_t *blocktablePtr; // pageattention需要
    GlobalTensor<int32_t> blockTableGm; // pageattention需要
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<layout>();
    static constexpr GmFormat KV_FORMAT = GetKVGmFormat<layout>();
    FaGmTensor<INPUT_T, Q_FORMAT> queryGm;
    FaGmTensor<INPUT_T, KV_FORMAT> keyGm;
    FaGmTensor<INPUT_T, KV_FORMAT> valueGm;
    FaGmTensor<INPUT_T, Q_FORMAT> queryRopeGm;
    FaGmTensor<INPUT_T, KV_FORMAT> keyRopeGm;

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

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用, GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    typename QL1BuffSel<INPUT_T, dBaseSize>::Type l1QBuffers;
    // mm1右矩阵K
    typename KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1KBuffers;
    // mm2右矩阵V
    typename KVL1BuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type l1VBuffers;
    // L0A
    using L0AType = typename L0ABuffSel<INPUT_T>::Type;
    L0AType mmL0ABuffers;
    // L0B
    using L0BType = typename L0BBuffSel<INPUT_T, s2BaseSize, dBaseSize>::Type;
    L0BType mmL0BBuffers;
    // L0C
    using L0CType = typename L0CBuffSel<INPUT_T, s1BaseSize, s2BaseSize, dVBaseSize>::Type;
    L0CType mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable, 
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope)
{
    tPipe = pipe;
    l1BufferManagerPtr = l1BuffMgr;
    this->queryGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)query);
    if constexpr (hasRope) {
        this->queryRopeGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
        this->keyRopeGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }
    if constexpr (!isInfer) {
        this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }
    if constexpr (isPa) {
        blocktablePtr = blockTable;
    }
    InitLocalBuffer();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitCubeInput(
    __gm__ uint8_t *key, __gm__ uint8_t *value, CVSharedParams<isInfer, isPa> *sharedParams,
    AttenMaskInfo *attenMaskInfo, __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr)
{
    if constexpr (isInfer) {
        if (sharedParams->fromFused) {
            ListTensorDesc keyListTensorDescInit((__gm__ void *)key);
            ListTensorDesc valueListTensorDescInit((__gm__ void *)value);
            currentKey = (__gm__ uint8_t *)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            currentValue = (__gm__ uint8_t *)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            if (sharedParams->isKvContinuous == 1) {
                this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
                this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
            } else {
                this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
                this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
            }
        } else {
            this->keyGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)key);
            this->valueGm.gmTensor.SetGlobalBuffer((__gm__ INPUT_T *)value);
        }
        attenMaskInfo->preTokens = sharedParams->preTokens;
        attenMaskInfo->nextTokens = sharedParams->nextTokens;
        attenMaskInfo->compressMode = sharedParams->compressMode;
        attenMaskInfo->attenMaskS1Size = sharedParams->attenMaskS1Size;
        attenMaskInfo->attenMaskS2Size = sharedParams->attenMaskS2Size;
        if constexpr (isPa) {
            this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blocktablePtr);
            this->kvCacheBlockSize = sharedParams->blockSize;
            this->maxBlockNumPerBatch = sharedParams->blockTableDim2;
            kvLayout = sharedParams->paLayoutType == 1 ? KVLAYOUT::BBH : KVLAYOUT::BNBD;
        }
    }
    InitGmTensor(sharedParams, actualSeqQlenAddr, actualSeqKvlenAddr);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *deqScaleQK,
    __gm__ uint8_t *deqScaleV, ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInt8) {
        this->InitQuant(constInfo, deqScaleQK, deqScaleV);
    } else {
        constInfo.deqScaleQKValue = 0x3f800000; // float32的1
        constInfo.deqScaleVValue = 0x3f800000;
    }
}
 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitQuant(ConstInfo<isInfer, hasRope> &constInfo, __gm__ uint8_t *deqScaleQK,
    __gm__ uint8_t *deqScaleV)
{
    if constexpr (isInt8) {
        if (deqScaleQK != nullptr) {
            deqScaleQKGm.SetGlobalBuffer((__gm__ uint32_t *)deqScaleQK);
            constInfo.deqScaleQKValue = deqScaleQKGm.GetValue(0);
        }
        if (deqScaleV != nullptr) {
            deqScaleVGm.SetGlobalBuffer((__gm__ uint32_t *)deqScaleV);
            constInfo.deqScaleVValue = deqScaleVGm.GetValue(0);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitLocalBuffer() {
    if constexpr ((dBaseSize > 256) || IsSameType<INPUT_T, float>::value) {
        /* Float32场景以及D大于256的其他dtype场景，Bmm1左矩阵不开DB + 驻留 + 复用 + L1切K 
           Bmm2左矩阵3 Buffer循环，Bmm1右矩阵和Bmm2右矩阵在L1上切D轴，并开启DoubleBuffer
           唯一不同的是D=256场景下的D轴只能切分到96，其余都可以切分到128，原因是D=256场景S1Base是128
        */
        constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(INPUT_T);
        /* fp8场景bmm1右矩阵在L1上全载*/
        constexpr uint32_t mm1RightSize = isFp8 ? (dBaseSize * s2BaseSize * sizeof(INPUT_T)) : (s2BaseSize * l1BaseD * sizeof(INPUT_T));
        constexpr uint32_t mm2RightSize = s2BaseSize * l1BaseD * sizeof(INPUT_T);
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
        l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
    } else {
        constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(INPUT_T);
        constexpr uint32_t mm2RightSize = (uint32_t)dVTemplateType * s2BaseSize * sizeof(INPUT_T);
        l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
        l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);
        l1VBuffers.Init((*l1BufferManagerPtr), mm2RightSize);
    }

    // L0A B C 当前写死，能否通过基础api获取
    l0aBufferManager.Init(tPipe, 65536); // 64 * 1024
    l0bBufferManager.Init(tPipe, 65536); // 64 * 1024
    l0cBufferManager.Init(tPipe, 262144); // 256 * 1024
    // L0A B C当前写死，要改成通过计算获取
    if (IsSameType<INPUT_T, float>::value) {
        mmL0ABuffers.Init(l0aBufferManager, 64 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 64 * 1024);
    } else {
        mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
        mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
    }

    if constexpr (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 && s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4) {
        mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_4) * KB_TO_BYTES);
    } else {
        mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / NUM_2) * KB_TO_BYTES);
    }
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::InitGmTensor(CVSharedParams<isInfer, isPa> *sharedParams,
    __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr)
{
    if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
        this->queryGm.offsetCalculator.Init(sharedParams->bSize, sharedParams->n2Size, sharedParams->gSize,
            sharedParams->s1Size, sharedParams->dSize);
        if constexpr (hasRope) {
            this->queryRopeGm.offsetCalculator.Init(sharedParams->bSize, sharedParams->n2Size, sharedParams->gSize,
                sharedParams->s1Size, sharedParams->dSizeRope);
        }
    } else {  // GM_Q_OUT_TND
        GlobalTensor<uint64_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ uint64_t *)actualSeqQlenAddr);
        if constexpr (isInfer) {
            this->queryGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->gSize, sharedParams->dSize,
                actualSeqQLen, sharedParams->actualSeqLengthsSize);
            if constexpr (hasRope) {
                this->queryRopeGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->gSize,
                    sharedParams->dSizeRope, actualSeqQLen, sharedParams->actualSeqLengthsSize);
            }
        } else {
            this->queryGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->gSize, sharedParams->dSize,
                actualSeqQLen, sharedParams->bSize);
            if constexpr (hasRope) {
                this->queryRopeGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->gSize,
                    sharedParams->dSizeRope, actualSeqQLen, sharedParams->bSize);
            }
        }
    }
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        this->keyGm.offsetCalculator.Init(sharedParams->bSize, sharedParams->n2Size, sharedParams->s2Size,
            sharedParams->dSize);
        this->valueGm.offsetCalculator.Init(sharedParams->bSize, sharedParams->n2Size, sharedParams->s2Size,
            sharedParams->dSizeV);
        if constexpr (hasRope) {
            this->keyRopeGm.offsetCalculator.Init(sharedParams->bSize, sharedParams->n2Size, sharedParams->s2Size,
                sharedParams->dSizeRope);
        }
    } else {  // GM_KV_TND
        GlobalTensor<uint64_t> actualSeqKVLen;
        actualSeqKVLen.SetGlobalBuffer((__gm__ uint64_t *)actualSeqKvlenAddr);
        if constexpr (isInfer) {
            this->keyGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->dSize, actualSeqKVLen,
                sharedParams->actualSeqLengthsKVSize);
            this->valueGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->dSizeV, actualSeqKVLen,
                sharedParams->actualSeqLengthsKVSize);
            if constexpr (hasRope) {
                this->keyRopeGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->dSizeRope, actualSeqKVLen,
                    sharedParams->actualSeqLengthsKVSize);
            }
        } else {
            this->keyGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->dSize, actualSeqKVLen,
                sharedParams->bSize);
            this->valueGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->dSizeV, actualSeqKVLen,
                sharedParams->bSize);
            if constexpr (hasRope) {
                this->keyRopeGm.offsetCalculator.Init(sharedParams->n2Size, sharedParams->dSizeRope, actualSeqKVLen,
                    sharedParams->bSize);
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::CalcS1Coord(RunInfo<isInfer> &runInfo,
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
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::CalcS2Coord(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
    coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.boIdx;
    if constexpr (isInfer) {
        coordInfo[runInfo.taskIdMod3].s2Coord += runInfo.kvLeftPaddingSize;  // 左padding
        if constexpr (isFd) {
            coordInfo[runInfo.taskIdMod3].s2Coord += runInfo.flashDecodeS2Idx * constInfo.sInnerLoopSize;
        }
        if (constInfo.isKvContinuous == 0) {
            coordInfo[runInfo.taskIdMod3].curBIdx = 0;
            if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
                // 更新N2方向stride
                this->keyGm.offsetCalculator.Init(0, constInfo.n2Size, runInfo.s2InCurrentBatch, constInfo.dSize);
                if constexpr (hasRope) {
                    this->keyRopeGm.offsetCalculator.Init(0, constInfo.n2Size, runInfo.s2InCurrentBatch,
                                                          constInfo.dSizeRope);
                }
            }
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    CalcS1Coord(runInfo, constInfo);
    CalcS2Coord(runInfo, constInfo);
    if constexpr (useDn) {
        IterateBmm1Dn(outputBuf, runInfo, constInfo);
    } else {
        IterateBmm1Nd(outputBuf, runInfo, constInfo);
    }
    
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm2L1SplitN(mm2ResPos &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo) 
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.Get();

    // dVTemplateType > 256, L1切N, 左矩阵不变，右矩阵每次循环搬运S2 * 128
    constexpr uint32_t baseN = l1BaseD;
    uint32_t nLoops = ((uint32_t)constInfo.dSizeV + baseN - 1) / baseN; // 尾块处理
    uint32_t realN = baseN;
    for (uint32_t n = 0; n < nLoops; ++n) {
        if (n == nLoops - 1) {
            uint32_t tailSize = (uint32_t)constInfo.dSizeV % baseN;
            realN = tailSize ? tailSize : baseN;
        }
        Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
        mm2B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
        LocalTensor<INPUT_T> mm2BTensor = mm2B.GetTensor<INPUT_T>();
        uint64_t gmNOffset = n * baseN;
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
            startPos.dIdx = n * baseN;
            PAShape shape;
            shape.blockSize = kvCacheBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.dSizeV;
            shape.actHeadDim = realN;
            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
            shape.copyRowNum = runInfo.s2RealSize;
            if constexpr (isFp8) {
                shape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5;
            } else {
                shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
            }
            GlobalTensor<INPUT_T> mm2BGmTensor = GetValueGm(runInfo, constInfo);
            GmCopyInToL1PA<INPUT_T>(mm2BTensor, mm2BGmTensor, blockTableGm, kvLayout, shape, startPos);
        } else {
            uint64_t gmOffset = runInfo.keyOffset;
            if (constInfo.dSize != constInfo.dSizeV) {
                gmOffset = this->valueGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
            }
            CopyToL1Nd2Nz<INPUT_T>(mm2BTensor, GetValueGm(runInfo, constInfo)[gmOffset + gmNOffset], runInfo.s2RealSize, realN, constInfo.mm2Kb);
        }
        mm2B.Set<HardEvent::MTE2_MTE1>(); // 通知

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
        MMParam param = {(uint32_t)s1BaseSize,  // singleM
                          realN, // singleN
                          (uint32_t)runInfo.s2RealSize,  // singleK
                          useDn,    // isLeftTranspose
                          false     // isRightTranspose
                        };
        mm2B.Wait<HardEvent::MTE2_MTE1>(); // 等待
        MatmulFull<INPUT_T, INPUT_T, T, 128, baseN, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<INPUT_T>(),
            mm2BTensor,
            mmL0ABuffers,
            mmL0BBuffers,
            mm2ResL0C.GetTensor<T>(),
            param);

        mm2B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
        mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
        mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        if constexpr (bmm2Write2Ub) {
            fixpipeParams.nSize = (realN + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
        } else {
            fixpipeParams.nSize = realN; // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
        }
        fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
        fixpipeParams.srcStride = ((s1BaseSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
        if constexpr (bmm2Write2Ub || splitD) {
            fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV; // dstGm 两行之间的间隔
        }
        fixpipeParams.dualDstCtl = 1;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        Fixpipe<T, T, BMM2_FIXPIPE_CONFIG>(outputBuf.template GetTensor<T>()[gmNOffset], mm2ResL0C.GetTensor<T>(),
            fixpipeParams); // 将matmul结果从L0C搬运到UB
        mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm2(mm2ResPos &outputBuf,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer && layout == LayOutTypeEnum::LAYOUT_BNSD) {
        if (constInfo.isKvContinuous == 0) {
            this->valueGm.offsetCalculator.Init(0, constInfo.n2Size, runInfo.s2InCurrentBatch, constInfo.dSizeV);
        }
    }
    if constexpr (IsSameType<INPUT_T, float>::value || (uint32_t)dVTemplateType > 256 || (uint32_t)dTemplateType > 256) {
        IterateBmm2L1SplitN(outputBuf, inputBuf, runInfo, constInfo);
    } else {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2A = inputBuf.GetCube();
        Buffer<BufferType::L1> mm2B = l1VBuffers.Get();
        mm2B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
        LocalTensor<INPUT_T> mm2BTensor = mm2B.GetTensor<INPUT_T>();
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
            PAShape shape;
            shape.blockSize = kvCacheBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.dSizeV;
            shape.actHeadDim = constInfo.dSizeV;
            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
            shape.copyRowNum = runInfo.s2RealSize;
            if constexpr (isFp8) {
                shape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5;
            } else {
                shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
            }
            GlobalTensor<INPUT_T> mm2BGmTensor = GetValueGm(runInfo, constInfo);
            GmCopyInToL1PA<INPUT_T>(mm2BTensor, mm2BGmTensor, blockTableGm, kvLayout, shape, startPos);
        } else {
            uint64_t gmOffset = runInfo.keyOffset;
            if (constInfo.dSize != constInfo.dSizeV) {
                gmOffset = this->valueGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx,
                    runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
            }
            CopyToL1Nd2Nz<INPUT_T>(mm2BTensor, GetValueGm(runInfo, constInfo)[gmOffset], runInfo.s2RealSize,
                constInfo.dSizeV, constInfo.mm2Kb);
        }
        mm2B.Set<HardEvent::MTE2_MTE1>(); // 通知

        Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
        mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
        MMParam param = {(uint32_t)s1BaseSize,  // singleM 128
                         (uint32_t)constInfo.dSizeV, // singleN 128
                         (uint32_t)runInfo.s2RealSize,  // singleK
                         useDn,    // isLeftTranspose
                         false     // isRightTranspose
                        };
        mm2B.Wait<HardEvent::MTE2_MTE1>(); // 等待
        using L0C_TYPE = int32_t;
        MatmulBase<INPUT_T, INPUT_T, L0C_TYPE, 128, (uint32_t)dVTemplateType, 128, ABLayout::MK, ABLayout::KN>(
            mm2A.GetTensor<INPUT_T>(),
            mm2BTensor,
            mmL0ABuffers,
            mmL0BBuffers,
            mm2ResL0C.GetTensor<L0C_TYPE>(),
            param);
        mm2B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B

        mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
        mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
        fixpipeParams.nSize = constInfo.dSizeV; // L0C上的bmm1结果矩阵N方向的size大小

        fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
        fixpipeParams.srcStride = ((s1BaseSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
        if constexpr (bmm2Write2Ub) {
            fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4;
        } else {
            fixpipeParams.dstStride = (uint32_t)constInfo.dSizeV; // dstGm 两行之间的间隔
        }
        fixpipeParams.dualDstCtl = 0;
        fixpipeParams.params.ndNum = 1;
        fixpipeParams.params.srcNdStride = 0;
        fixpipeParams.params.dstNdStride = 0;
        fixpipeParams.quantPre = QuantMode_t::DEQF16;
        fixpipeParams.deqScalar = constInfo.deqScaleVValue;
        Fixpipe<T, int32_t, BMM2_FIXPIPE_CONFIG>(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<int32_t>(),
                                                 fixpipeParams); // 将matmul结果从L0C搬运到UB
        mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::GetKvByTensorList(RunInfo<isInfer>& runInfo, 
    const ConstInfo<isInfer, hasRope> &constInfo,
    GlobalTensor<INPUT_T>& keyValueGm, GlobalTensor<INPUT_T>& tempKeyValueGm)
{
    if (constInfo.isKvContinuous != 0) {
        return;
    }
    ListTensorDesc keyValueListTensorDesc((__gm__ void*)keyValueGm.GetPhyAddr());
    __gm__ uint8_t* tempKeyValueGmPtr =
        (__gm__ uint8_t*)keyValueListTensorDesc.GetDataPtr<__gm__ uint8_t>(runInfo.boIdx);
    tempKeyValueGm.SetGlobalBuffer((__gm__ INPUT_T*)tempKeyValueGmPtr);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_T>
FABlockCube<TEMPLATE_ARGS>::GetKeyGm(RunInfo<isInfer> &runInfo, 
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer) {
        GlobalTensor<INPUT_T> tempKeyGm = this->keyGm.gmTensor;
        GetKvByTensorList(runInfo, constInfo, this->keyGm.gmTensor, tempKeyGm);
        return tempKeyGm;
    } else {
        return this->keyGm.gmTensor;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline GlobalTensor<INPUT_T>
FABlockCube<TEMPLATE_ARGS>::GetValueGm(RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    if constexpr (isInfer) {
        GlobalTensor<INPUT_T> tempValueGm = this->valueGm.gmTensor;
        GetKvByTensorList(runInfo, constInfo, this->valueGm.gmTensor, tempValueGm);
        return tempValueGm;
    } else {
        return this->valueGm.gmTensor;
    }
}

/* 针对S1Base=128, S2Base = 128, D > 128场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1NdL0Split(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    // 左矩阵复用 ,s2的第一次循环加载左矩阵
    // 加载左矩阵到L1 当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本快：搬运0
        mm1A = l1QBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx, coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, this->queryGm.gmTensor[gmOffset], runInfo.s1RealSize, constInfo.dSize, constInfo.mm1Ka);
        if constexpr (hasRope) {
            uint32_t dstNzC0Stride = (runInfo.s1RealSize + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数;
            if constexpr (isFp8) {
                dstNzC0Stride = (runInfo.s1RealSize + 31) >> 5 << 5; // fp8场景在L1上M方向32对齐，防止loadL12L0出现地址越界
            }
            uint64_t gmRopeOffset = this->queryRopeGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx,
                runInfo.goIdx, coordInfo[runInfo.taskIdMod3].s1Coord, 0);
            CopyToL1Nd2Nz<INPUT_T>(mm1ATensor[dstNzC0Stride * constInfo.dSize],
                this->queryRopeGm.gmTensor[gmRopeOffset], runInfo.s1RealSize, constInfo.dSizeRope, constInfo.mm1RopeKa); 
        }
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非s2的第一次循环直接复用Q
        mm1A = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知 // 是否可以省略
    }
    // 加载当前轮的右矩阵到L1
    mm1B = l1KBuffers.Get();
    mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
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
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        if constexpr (isFp8) {
            shape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5;
        } else {
            shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        }
        GlobalTensor<INPUT_T> mm1BGmTensor = this->keyGm.gmTensor;
        if constexpr (hasRope) {
            PAShape ropeShape = shape;
            ropeShape.headDim = constInfo.dSizeRope;
            ropeShape.actHeadDim = constInfo.dSizeRope;
            uint32_t dstNzC0Stride = shape.copyRowNumAlign;
            LocalTensor<INPUT_T> mm1BRopeTensor = mm1BTensor[dstNzC0Stride * constInfo.dSize];
            GlobalTensor<INPUT_T> mm1BRopeGmTensor = this->keyRopeGm.gmTensor;
            GmCopyInToL1HasRopePA<INPUT_T>(mm1BTensor, mm1BRopeTensor, mm1BGmTensor, mm1BRopeGmTensor, blockTableGm, kvLayout, shape, ropeShape, startPos);
        } else {
            GmCopyInToL1PA<INPUT_T>(mm1BTensor, mm1BGmTensor, blockTableGm, kvLayout, shape, startPos);
        }
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
            coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
            constInfo.dSize, constInfo.mm1Kb);
        if constexpr (hasRope) {
            uint32_t dstNzC0Stride = (runInfo.s2RealSize + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数;
            if constexpr (isFp8) {
                dstNzC0Stride = (runInfo.s2RealSize + 31) >> 5 << 5; // fp8场景在L1上M方向32对齐，防止loadL12L0出现地址越界
            }
            uint64_t gmRopeOffset = this->keyRopeGm.offsetCalculator.GetOffset(runInfo.boIdx,
                runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
            CopyToL1Nd2Nz<INPUT_T>(mm1BTensor[dstNzC0Stride * constInfo.dSize], this->keyRopeGm.gmTensor[gmRopeOffset],
                runInfo.s2RealSize, constInfo.dSizeRope, constInfo.mm1RopeKb); 
        }
    }
    mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s1RealSize,
                     (uint32_t)runInfo.s2RealSize,
                     (uint32_t)(constInfo.dSize + constInfo.dSizeRope), // singleK完整dsize, MatmulK内部会切K
                     0,    // isLeftTranspose
                     1     // isRightTranspose 
                    };

    // 这里base M N K不要写死
    if constexpr (s2BaseSize == 256) {
        MatmulN<INPUT_T, INPUT_T, T, 64, 128, 256, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
            mmL0ABuffers, mmL0BBuffers,
            mm1ResL0C.GetTensor<T>(),
            param);
    } else {
        if constexpr (isFp8) {
            MatmulK<INPUT_T, INPUT_T, T, 128, 128, 256, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
                mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(),
                param);
        } else {
            MatmulK<INPUT_T, INPUT_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
                mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
                mmL0ABuffers, mmL0BBuffers,
                mm1ResL0C.GetTensor<T>(),
                param);
        }
    }
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>();
    }
    mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放
}

/* 针对useDn=true, S1Base=128, S2Base = 128, 128 < D <= 256场景，L1全载，左矩阵驻留 + L0切D + L0Db*/
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1DnSplitK(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    // 右矩阵复用，S2的第一次循环加载右矩阵
    // 加载右矩阵到L1 ,当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本快：搬运0
        mm1B = l1QBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用
        LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, this->queryGm.gmTensor[gmOffset], runInfo.s1RealSize, constInfo.dSize,
            constInfo.mm1Ka);
        mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非s2的第一次循环直接复用Q
        mm1B = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知 // 是否可以省略
    }

    // 加载当前轮的左矩阵到L1
    mm1A = l1KBuffers.Get();
    mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用
    LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
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
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        GlobalTensor<INPUT_T> mm1AGmTensor = GetKeyGm(runInfo, constInfo);
        GmCopyInToL1PA<INPUT_T>(mm1ATensor, mm1AGmTensor, blockTableGm, kvLayout, shape, startPos);
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
            coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
            constInfo.dSize, constInfo.mm1Kb);
    }
    mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s2RealSize,
                     (uint32_t)runInfo.s1RealSize,
                     (uint32_t)(constInfo.dSize), // singleK完整dsize, MatmulK内部会切K
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };

    // 这里base M N K不要写死
    MatmulK<INPUT_T, INPUT_T, T, 128, 128, 128, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1Q
    }
    mm1A.Set<HardEvent::MTE1_MTE2>(); // 释放L1K

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); //等待L0C

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = (runInfo.s1RealSize + 31) >> 5 << 5; // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = runInfo.s2RealSize; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = fixpipeParams.nSize / 2; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 2; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1Nd(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    // 计算key的offset
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1 ,当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        mm1A = l1QBuffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, this->queryGm.gmTensor[gmOffset], runInfo.s1RealSize, constInfo.dSize,
            constInfo.mm1Ka);
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非S2的第一次循环直接复用Q
        mm1A = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    // 加载当前轮的右矩阵到L1
    mm1B = l1KBuffers.Get();
    mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
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
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        if constexpr (isFp8) {
            shape.copyRowNumAlign = (runInfo.s2RealSize + 31) >> 5 << 5;
        } else {
            shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        }
        GlobalTensor<INPUT_T> mm1BGmTensor = GetKeyGm(runInfo, constInfo);
        GmCopyInToL1PA<INPUT_T>(mm1BTensor, mm1BGmTensor, blockTableGm, kvLayout, shape, startPos);
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
            coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
            constInfo.dSize, constInfo.mm1Kb);
    }
    mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s1RealSize,  // singleM
                        (uint32_t)runInfo.s2RealSize,  // singleN
                        (uint32_t)(constInfo.dSize), // singleK
                        0,    // isLeftTranspose
                        1     // isRightTranspose
    };
    using L0C_TYPE = int32_t;
    MatmulBase<INPUT_T, INPUT_T, L0C_TYPE, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<L0C_TYPE>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }

    mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); //等待L0C

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 0; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.quantPre = QuantMode_t::DEQF16;
    fixpipeParams.deqScalar = constInfo.deqScaleQKValue;
    Fixpipe<T, L0C_TYPE, BMM1_FIXPIPE_CONFIG>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<L0C_TYPE>(),
                                              fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
}

/* 针对S1Base=128, S2Base = 128, D > 256场景，L1层面切K，且左矩阵单Buffer+驻留，右矩阵每次重新搬运。*/
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1NdL1SplitK(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo) 
{
    constexpr uint32_t baseK = l1BaseD;
    uint32_t kLoops = (constInfo.dSize + baseK - 1) / baseK; // 尾块处理
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); 
    Buffer<BufferType::L1> mm1A;
    uint32_t dstNzC0Stride = ((runInfo.s1RealSize + 15) >> 4 << 4);
    uint64_t l1BaseKOffset = baseK * dstNzC0Stride;
    mm1A = l1QBuffers.Get();
    if (unlikely(runInfo.s2LoopCount == 0)) {
        mm1A.Wait<HardEvent::MTE1_MTE2>();
    }
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
        coordInfo[runInfo.taskIdMod3].s1Coord, 0);
    runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
        coordInfo[runInfo.taskIdMod3].s2Coord, 0);
    for (uint32_t k = 0; k < kLoops; k++) {
        Buffer<BufferType::L1> mm1B;
        // 左矩阵复用, 但是每次只加载realK列
        uint32_t realK;
        if (k == kLoops - 1) {
            uint32_t tailSize = constInfo.dSize % baseK;
            realK = tailSize ? tailSize : baseK;
        } else {
            realK = baseK; // 单个ND矩阵的实际列数，单位为元素个数
        }
        if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本快：搬运0
            uint64_t gmKOffset = k * baseK;
            LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
            CopyToL1Nd2Nz<INPUT_T>(mm1ATensor[k * l1BaseKOffset], this->queryGm.gmTensor[gmOffset + gmKOffset],
                runInfo.s1RealSize, realK, constInfo.mm1Ka);
            mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
        } else { // 非s2的第一次循环直接复用Q
            mm1A = l1QBuffers.GetPre();
            // 左矩阵复用时，sinner循环内不需要MTE2同步等待
            mm1A.Set<HardEvent::MTE2_MTE1>();
        }
        // 加载当前轮的右矩阵到L1
        mm1B = l1KBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>();
        LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
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
            startPos.dIdx = k * baseK;
            PAShape shape;
            shape.blockSize = kvCacheBlockSize;
            shape.headNum = constInfo.n2Size;
            shape.headDim = constInfo.dSize;
            shape.actHeadDim = realK;
            shape.maxblockNumPerBatch = maxBlockNumPerBatch;
            shape.copyRowNum = runInfo.s2RealSize;
            shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
            GlobalTensor<INPUT_T> mm1BGmTensor = this->keyGm.gmTensor;
            GmCopyInToL1PA<INPUT_T>(mm1BTensor, mm1BGmTensor, blockTableGm, kvLayout, shape, startPos);
        } else {
            uint64_t gmKBOffset = k * baseK;
            CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset + gmKBOffset],
                runInfo.s2RealSize, realK, constInfo.mm1Kb);
        }
        mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知
        mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
        mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1B

        MMParam param = {(uint32_t)runInfo.s1RealSize,
                         (uint32_t)runInfo.s2RealSize,
                         realK, // singleK = realK
                         0,     // isLeftTranspose
                         1,     // isRightTranspose 
                         k == 0,
                         k == 0
                        };

        // 这里base M N K不要写死
        MatmulFull<INPUT_T, INPUT_T, T, 128, 128, baseK, ABLayout::MK, ABLayout::KN>(
            mm1A.GetTensor<INPUT_T>()[k * l1BaseKOffset], mm1BTensor,
            mmL0ABuffers, mmL0BBuffers,
            mm1ResL0C.GetTensor<T>(),
            param);

        mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1B
    }
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>();
    }
    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知

    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCube<TEMPLATE_ARGS>::IterateBmm1Dn(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf, RunInfo<isInfer> &runInfo,
    ConstInfo<isInfer, hasRope> &constInfo)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1> mm1B;
    // 右矩阵复用，S2的第一次循环加载右矩阵
    // 加载右矩阵到L1 ,当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        mm1B = l1QBuffers.Get();
        mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1BTensor, this->queryGm.gmTensor[gmOffset], runInfo.s1RealSize, constInfo.dSize,
            constInfo.mm1Ka);
        mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非S2的第一次循环直接复用Q
        mm1B = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        mm1B.Set<HardEvent::MTE2_MTE1>(); // 通知
    }
    // 加载当前轮的左矩阵到L1
    // 计算key的offset
    mm1A = l1KBuffers.Get();
    mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用L1B
    LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
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
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        GlobalTensor<INPUT_T> mm1AGmTensor = GetKeyGm(runInfo, constInfo);
        GmCopyInToL1PA<INPUT_T>(mm1ATensor, mm1AGmTensor, blockTableGm, kvLayout, shape, startPos);
    } else {
        runInfo.keyOffset = this->keyGm.offsetCalculator.GetOffset(coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx,
            coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<INPUT_T>(mm1ATensor, GetKeyGm(runInfo, constInfo)[runInfo.keyOffset], runInfo.s2RealSize,
            constInfo.dSize, constInfo.mm1Kb);
    }
    mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1K
    mm1B.Wait<HardEvent::MTE2_MTE1>(); // 等待L1Q

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s2RealSize,  // singleM
                     (uint32_t)runInfo.s1RealSize,  // singleN
                     (uint32_t)(constInfo.dSize), // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
    };
    using L0C_TYPE = int32_t;
    MatmulBase<INPUT_T, INPUT_T, L0C_TYPE, 128, 128, dBaseSize, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<L0C_TYPE>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放L1Q
    }
    mm1A.Set<HardEvent::MTE1_MTE2>(); // 释放L1K

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); //等待L0C

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = (runInfo.s1RealSize + 31) >> 5 << 5; // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = runInfo.s2RealSize; // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = fixpipeParams.nSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 0; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    fixpipeParams.quantPre = QuantMode_t::DEQF16;
    fixpipeParams.deqScalar = constInfo.deqScaleQKValue;
    Fixpipe<T, int32_t, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<int32_t>(),
                                              fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
}

TEMPLATES_DEF
class FABlockCubeDummy {
public:
    static constexpr bool isFp8 = FABlockCube<TEMPLATE_ARGS>::isFp8;
    static constexpr bool splitD = FABlockCube<TEMPLATE_ARGS>::splitD;
    static constexpr bool useDn = FABlockCube<TEMPLATE_ARGS>::useDn;
    static constexpr TPosition bmm2OutPos = FABlockCube<TEMPLATE_ARGS>::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = FABlockCube<TEMPLATE_ARGS>::bmm2Write2Ub;
    __aicore__ inline FABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable, 
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *key, __gm__ uint8_t *value,
        CVSharedParams<isInfer, isPa> *sharedParams, AttenMaskInfo *attenMaskInfo,
        __gm__ int64_t *actualSeqQlenAddr, __gm__ int64_t *actualSeqKvlenAddr) {}

    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer> &runInfo, ConstInfo<isInfer, hasRope> &constInfo) {}

    using mm2ResPos = typename std::conditional<bmm2Write2Ub, Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD>>::type;
    __aicore__ inline void IterateBmm2(mm2ResPos &outputBuf,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputBuf,RunInfo<isInfer> &runInfo,
        ConstInfo<isInfer, hasRope> &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *deqScaleQK, __gm__ uint8_t *deqScaleV, ConstInfo<isInfer, hasRope> &constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(FABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE)\
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
