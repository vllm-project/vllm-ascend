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
 * \file flash_attention_score_kernel_base.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_BASE_H_
#define FLASH_ATTENTION_SCORE_KERNEL_BASE_H_
#include "flash_attention_score_block_cube.h"
#include "flash_attention_score_block_vec_train.h"
#include "flash_attention_score_block_vec_infer.h"
#include "flash_attention_score_common_regbase.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "attenmask.h"

// 线上编包
#include "../matmul.h"
#include "../FixpipeOut.h"
#include "../CopyInL1.h"

#include "pse.h"
#include "infer_flash_attention_comm.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/utils/init_global_memory.h"

using matmul::MatmulType;
using namespace AscendC;
using namespace optiling;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;

namespace BaseApi {
template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
class FlashAttentionScoreKernelBase {
public:
    ARGS_TRAITS;
    __aicore__ inline FlashAttentionScoreKernelBase() {};

    __aicore__ inline void InitBaseAPI(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
                            __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, 
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, 
                            __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset,
                            __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen, 
                            __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax, 
                            __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                            __gm__ uint8_t *workspace, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void GetExtremeValue(T &negativeScalar, T &positiveScalar);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse,
                            __gm__ uint8_t *dropMask, __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask,
                            __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKv,
                            __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
                            __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *queryRope,
                            __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
                            __gm__ uint8_t *keyRope, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize,
                            __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink, __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum,
                            __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
                            const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitMMResBuf();
    __aicore__ inline void InitActualKVPrefixLen(__gm__ uint8_t *actualSharedPrefixLen);
    __aicore__ inline void ComputeConstexpr();
    __aicore__ inline void SetRunInfo(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount,
                                      int64_t s2LoopLimit, int64_t multiCoreInnerIdx);
    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam);
    __aicore__ inline void ComputeBmm1Tail(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam);
    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t &actualSeqQlen, int64_t &actualSeqKvLen);

    __aicore__ inline ChildClass* GetDerived() {
        return static_cast<ChildClass*>(this);
    }
    TPipe *pipe;

    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
    /* 编译期常量的基本块信息 */
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr bool isFp8 = CubeBlockType::isFp8;
    /* 是否使能dn的信息; 没有可选输入并且S2切分的时候使用dn，s2比较小的时候nd效果更好 */
    static constexpr bool useDn = CubeBlockType::useDn;
    /* HIFLOAT8场景 softmax计算使用Nz格式计算，vec1ResBuffer可以和bmm1ResBuffer进行复用*/
    static constexpr bool useNz = CubeBlockType::useNz;
    static constexpr TPosition bmm2OutPos = CubeBlockType::bmm2OutPos;
    static constexpr bool bmm2Write2Ub = CubeBlockType::bmm2Write2Ub;
    static constexpr bool splitD =  CubeBlockType::splitD;
    static constexpr uint64_t SYNC_MODE = 4;
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    /* 核间通道 */
    BufferManager<BufferType::GM> gmBufferManager;
    BuffersPolicy3buff<BufferType::GM, SyncType::CROSS_CORE_SYNC_FORWARD> bmm2ResGmBuffers;

    BufferManager<BufferType::UB> ubBufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    using bmm2ResBufferType = typename Bmm2ResBuffSel<useDn, isFp8>::Type;
    bmm2ResBufferType bmm2Buffers;

    // mm2左矩阵P
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers; 
    CVSharedParams<isInfer, isPa> sharedParams;
    /* GM信息 */
    using keyGmType = typename std::conditional<isInfer, GlobalTensor<INPUT_T>, int32_t>::type;
    keyGmType keyGm; // kv不连续场景需要使用来获取shape
    __gm__ int64_t *actualSeqQlenAddr;
    __gm__ int64_t *actualSeqKvlenAddr;
    /* 核Index信息 */
    int32_t aicIdx;

    /* 初始化后不变的信息 */
    ConstInfo<isInfer, hasRope> constInfo;
    AttenMaskInfo attenMaskInfo;
    PseInfo pseInfo;
    /* 其他正向独有的一些信息 */
    // Unpack参数
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t s1ScaleNumAcc;
    uint64_t s2ScaleNumAcc;

    // prefix
    int64_t actualKVPrefixLen = 0;

    /* 模板库Block */
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitBaseAPI(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
    __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV, __gm__ uint8_t *pScale,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, 
    __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    fa_base_matmul::idCounterNum = 0;
    constInfo.subBlockIdx = GetSubBlockIdx();
    if ASCEND_IS_AIC {
        this->aicIdx = GetBlockIdx();
    } else {
        constInfo.aivIdx = GetBlockIdx();
        this->aicIdx = constInfo.aivIdx >> 1;
        this->tilingData = tiling;
    }
    this->pipe = tPipe;
    vecBlock.InitVecBlock(tPipe, this->tilingData, this->sharedParams, this->aicIdx, constInfo.subBlockIdx, 
        attenMaskInfo, pseInfo);
    vecBlock.CleanOutput(softmaxLse, attentionOut, constInfo);
    /* cube侧不依赖sharedParams的scalar前置 */
    InitMMResBuf();
    if ASCEND_IS_AIC {
        cubeBlock.InitCubeBlock(pipe, &l1BufferManager, query, key, value, blockTable, queryRope, keyRope);
        /* wait kfc message */
        CrossCoreWaitFlag<SYNC_MODE, PIPE_S>(15);
        auto tempTilingSSbuf = reinterpret_cast<__ssbuf__ uint32_t*>(0); // 从ssbuf的0地址开始拷贝
        auto tempTiling = reinterpret_cast<uint32_t *>(&sharedParams);
        #pragma unroll
        for (int i = 0; i < sizeof(CVSharedParams<isInfer, isPa>) / sizeof(uint32_t); ++i, ++tempTilingSSbuf, ++tempTiling) {
            *tempTiling = *tempTilingSSbuf;
        }
    }

    this->InitActualKVPrefixLen(actualSharedPrefixLen);
    this->ComputeConstexpr();
    this->InitGlobalBuffer(query, key, value, pse, dropMask, paddingMask, attenMask, prefix, actualSeqLengths,
        actualSeqLengthsKv, deqScaleQ, deqScaleK, deqScaleV, pScale, postQuantScale, postQuantOffset,
        keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, blockTable, queryPaddingSize, 
        kvPaddingSize, learnableSink, softmaxMax, softmaxSum, softmaxOut, workspace, tiling, tPipe); // gm设置
    this->InitLocalBuffer();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitActualKVPrefixLen(__gm__ uint8_t *actualSharedPrefixLen) 
{
    if constexpr (isInfer) {
        if constexpr (enableKVPrefix) {
            constInfo.kvPrefixSize = this->sharedParams.kvPrefixSize;
            if (!this->sharedParams.isActualSharedPrefixLenNull) {
                this->actualKVPrefixLen = ((__gm__ int64_t *)actualSharedPrefixLen)[0];
            } else {
                this->actualKVPrefixLen = this->sharedParams.kvPrefixSize;
            }
            constInfo.actualKVPrefixSize = this->actualKVPrefixLen;
        }
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pse, __gm__ uint8_t *dropMask,
    __gm__ uint8_t *paddingMask, __gm__ uint8_t *attenMask, __gm__ uint8_t *prefix, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *deqScaleQ, __gm__ uint8_t *deqScaleK, __gm__ uint8_t *deqScaleV,
    __gm__ uint8_t *pScale, __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t*actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *learnableSink,
    __gm__ uint8_t *softmaxMax, __gm__ uint8_t *softmaxSum, __gm__ uint8_t *softmaxOut, __gm__ uint8_t *workspace,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling, TPipe *tPipe)
{
    // 初始化vector用到的global buffer
    if constexpr (isInfer) {
        keyGm.SetGlobalBuffer((__gm__ INPUT_T *)(key));
        if (constInfo.isQHasLeftPadding) {
            constInfo.queryRightPaddingSize = ((__gm__ int64_t *)queryPaddingSize)[0];
            if (constInfo.queryRightPaddingSize < 0) {
                constInfo.queryRightPaddingSize = 0;
            }
        }
        if (constInfo.isKVHasLeftPadding) {
            constInfo.kvRightPaddingSize = ((__gm__ int64_t *)kvPaddingSize)[0];
            if (constInfo.kvRightPaddingSize < 0) {
                constInfo.kvRightPaddingSize = 0;
            }
        }
    }
    if constexpr (hasAtten) {
        attenMaskInfo.prefixNAddr = prefix;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
        actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
        actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
    } else {
        if constexpr (isInfer) {
            if (!constInfo.isActualLenDimsNull) {
                actualSeqQlenAddr = (__gm__ int64_t *)actualSeqLengths;
            }
            if (!constInfo.isActualLenDimsKVNull) {
                actualSeqKvlenAddr = (__gm__ int64_t *)actualSeqLengthsKv;
            }
        }
    }

    vecBlock.InitDropOut(dropMask, workspace);
    uint64_t singleCoreOffset = 0;
    if constexpr (!bmm2Write2Ub) {
        int64_t bmm2ResBlock = this->sharedParams.dSizeV;
        if constexpr (splitD) {
            bmm2ResBlock = (int64_t)dVTemplateType;
        }
        int64_t mm2ResultSize = (s1BaseSize) * bmm2ResBlock; // 使用Cube计算的总大小，Gm上的数据按照实际的dSize存储
        int64_t mm2Offset = CeilDiv(mm2ResultSize, 128) * 128 * sizeof(T);
        int64_t vec2ResultSize = (s1BaseSize) * constInfo.dBasicBlock;
        int64_t vec2Offset = CeilDiv(vec2ResultSize, 128) * 128 * sizeof(T);
        singleCoreOffset = mm2Offset;
        if constexpr (splitD) {
            singleCoreOffset = mm2Offset + vec2Offset;
        }
        int64_t totalOffset = this->aicIdx * 3 * singleCoreOffset; // 3为preload次数
        // SameB模式下V0和V1调用IterateAll的时候填写的地址相同
        gmBufferManager.Init(workspace + totalOffset);
        bmm2ResGmBuffers.Init(gmBufferManager, mm2Offset);
        workspace += (totalOffset + mm2Offset * 3);
    }
    vecBlock.InitGlobalBuffer(pse, deqScaleQ, deqScaleK, deqScaleV, pScale, postQuantScale, postQuantOffset,
        prefix, attenMask, queryPaddingSize, kvPaddingSize, learnableSink, softmaxMax, softmaxSum, workspace, singleCoreOffset,
        this->aicIdx, constInfo);
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND && !isInfer) {
        if ASCEND_IS_AIV {
            int64_t actualS1Len;
            int64_t actualS2Len;
            for (int64_t i = 0; i < this->sharedParams.bSize; ++i) {
                this->GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
                if (actualS2Len <= 0 && actualS1Len != 0) {
                    int64_t accumSize = (i == 0) ? 0 : ((__gm__ int64_t *)this->actualSeqQlenAddr)[i - 1];
                    if (actualS1Len < 0 && accumSize > 0) {
                        actualS1Len = constInfo.s1Size - accumSize;
                        int64_t frontCoreNum = actualS1Len % (this->sharedParams.coreNum * 2);
                        int64_t splitFactor = frontCoreNum > 0 ? 1 : 0;
                        int64_t s1SizeInner = actualS1Len / this->sharedParams.coreNum / 2;
                        int64_t innerOffset1 = (s1SizeInner + splitFactor) * (constInfo.aivIdx >= frontCoreNum ?
                                            frontCoreNum : constInfo.aivIdx);
                        int64_t innerOffset2 = s1SizeInner * (constInfo.aivIdx >= frontCoreNum ?
                                            constInfo.aivIdx - frontCoreNum : 0);
                        accumSize = accumSize + innerOffset1 + innerOffset2;
                        actualS1Len = s1SizeInner + (constInfo.aivIdx >= frontCoreNum ? 0 : splitFactor);
                    }
                    GlobalTensor<OUTPUT_T> attentionOutEodGm = this->vecBlock.attentionOutGm[accumSize * this->constInfo.n2GDv];
                    GlobalTensor<float> softmaxMaxEodGm = this->vecBlock.softmaxMaxGm[accumSize * this->constInfo.n2G * 8];
                    GlobalTensor<float> softmaxSumEodGm = this->vecBlock.softmaxSumGm[accumSize * this->constInfo.n2G * 8];
                    AscendC::Fill<OUTPUT_T>(attentionOutEodGm, actualS1Len * this->constInfo.n2GDv, static_cast<OUTPUT_T>(0.0));
                    AscendC::Fill<float>(softmaxMaxEodGm, actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
                    AscendC::Fill<float>(softmaxSumEodGm, actualS1Len * this->constInfo.n2G * 8, static_cast<float>(0.0));
                    SyncAll();
                }
            }
        }
    }
    cubeBlock.InitCubeInput(key, value, &sharedParams, &attenMaskInfo, actualSeqQlenAddr, actualSeqKvlenAddr, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen);
    cubeBlock.InitDequantParams(deqScaleQ, deqScaleK, deqScaleV);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitMMResBuf()
{
    uint32_t mm1OutDtype = sizeof(T);
    if constexpr (useNz) {
        mm1OutDtype = sizeof(half);
    }
    uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * mm1OutDtype;
    constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(INPUT_T);
    l1BufferManager.Init(pipe, 524288); // 512 * 1024
    // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
    l1PBuffers.Init(l1BufferManager, mm2LeftSize);
    if constexpr (bmm2Write2Ub) {
        if constexpr (!(useDn && isFp8)) {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize * 2);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
            if ASCEND_IS_AIV {
                bmm2Buffers.Get().SetCrossCore();
                bmm2Buffers.Get().SetCrossCore();
            }
        } else {
            ubBufferManager.Init(pipe, mm1ResultSize * 2 + mm2ResultSize);
            bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
            if ASCEND_IS_AIV {
                bmm2Buffers.Get().SetCrossCore();
            }
        }
    } else {
        ubBufferManager.Init(pipe, mm1ResultSize * 2);
    }
    bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
    if ASCEND_IS_AIV {
        if constexpr (useNz) {
            bmm1Buffers.Get().SetCrossCore<true>();
            bmm1Buffers.Get().SetCrossCore<true>();
        } else {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
        }
    }
}
 
template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::InitLocalBuffer()
{
    vecBlock.InitLocalBuffer(pipe, constInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeConstexpr()
{
    constInfo.s1BaseSize = s1BaseSize;
    constInfo.s2BaseSize = s2BaseSize;
    // 计算轴的乘积

    constInfo.bSize = sharedParams.bSize;
    constInfo.t1Size = sharedParams.t1Size;
    constInfo.t2Size = sharedParams.t2Size;
    constInfo.n2Size = sharedParams.n2Size;
    constInfo.s1Size = sharedParams.s1Size;
    constInfo.s2Size = sharedParams.s2Size;
    constInfo.dSize = sharedParams.dSize;
    constInfo.dSizeV = sharedParams.dSizeV;
    constInfo.dBasicBlock = Align64Func((uint16_t)constInfo.dSizeV);
    if constexpr (hasRope) {
        constInfo.dSizeRope = sharedParams.dSizeRope;
    } else {
        constInfo.dSizeRope = 0;
    }
    constInfo.gSize = sharedParams.gSize;
    constInfo.s1OuterSize = sharedParams.s1OuterSize;
    constInfo.s1S2 = constInfo.s1Size * constInfo.s2Size;
    constInfo.gS1 = constInfo.gSize * constInfo.s1Size;
    constInfo.n2G = constInfo.n2Size * constInfo.gSize;

    constInfo.s1Dv = constInfo.s1Size * constInfo.dSizeV;
    constInfo.s2Dv = constInfo.s2Size * constInfo.dSizeV;
    constInfo.n2Dv = constInfo.n2Size * constInfo.dSizeV;
    constInfo.gDv = constInfo.gSize * constInfo.dSizeV;
    constInfo.gS1Dv = constInfo.gSize * constInfo.s1Dv;
    constInfo.n2S2Dv = constInfo.n2Size * constInfo.s2Dv;
    constInfo.n2GDv = constInfo.n2Size * constInfo.gDv;
    constInfo.s2BaseN2Dv = s2BaseSize * constInfo.n2Dv;
    constInfo.n2GS1Dv = constInfo.n2Size * constInfo.gS1Dv;
    constInfo.layoutType = sharedParams.layoutType;

    if constexpr (isInfer) {
        constInfo.s1D = constInfo.s1Size * constInfo.dSize;
        constInfo.gD = constInfo.gSize * constInfo.dSize;
        constInfo.n2GD = constInfo.n2Size * constInfo.gD; 
        constInfo.gS1D = constInfo.gS1 * constInfo.dSize;
    }

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.mm1RopeKa = constInfo.n2Size * constInfo.gSize * constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.n2Size * constInfo.dSizeRope;
        }

        constInfo.mm1Ka = constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (sharedParams.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (sharedParams.isPfaGS1Merge == 1) {
                    constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
                } else if (sharedParams.isGqa == 1) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
        // BSH/BSNGD
        constInfo.s1BaseN2GDv = s1BaseSize * constInfo.n2GDv;
        if constexpr (hasRope) {
            constInfo.mm1RopeKa = constInfo.n2Size * constInfo.gSize * constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.n2Size * constInfo.dSizeRope;
        }
        constInfo.mm1Ka = constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = constInfo.n2Dv;
        if constexpr (isInfer) {
            if (sharedParams.isGqa) {
                constInfo.mm1Ka = constInfo.dSize;
            }
        }
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride =
                (constInfo.n2G - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
            if constexpr (isInfer) {
                if (sharedParams.isPfaGS1Merge == 1) {
                    constInfo.attentionOutStride = (constInfo.n2G - constInfo.gSize) * constInfo.dSizeV * sizeof(OUTPUT_T);
                } else if (sharedParams.isGqa == 1) {
                    constInfo.attentionOutStride = 0;
                }
            }
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_SBH) {
        // SBH/SBNGD
        constInfo.bN2GDv = sharedParams.bSize * constInfo.n2GDv;
        constInfo.s1BaseBN2GDv = s1BaseSize * constInfo.bN2GDv;
        constInfo.s2BaseBN2Dv = sharedParams.bSize * constInfo.s2BaseN2Dv;
        if constexpr (hasRope) {
            constInfo.mm1RopeKa = sharedParams.bSize * constInfo.n2Size * constInfo.gSize * constInfo.dSizeRope;
            constInfo.mm1RopeKb = sharedParams.bSize * constInfo.n2Size * constInfo.dSizeRope;
        }
        constInfo.mm1Ka = sharedParams.bSize * constInfo.n2Size * constInfo.gSize * constInfo.dSize;
        constInfo.mm1Kb = sharedParams.bSize * constInfo.n2Size * constInfo.dSize;
        constInfo.mm2Kb = sharedParams.bSize * constInfo.n2Dv;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride =
                (sharedParams.bSize * constInfo.n2Size * constInfo.gSize - 1) * constInfo.dSizeV * sizeof(OUTPUT_T);
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
        // bnsd
        constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
        constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
        if constexpr (hasRope) {
            constInfo.mm1RopeKa = constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.dSizeRope;
        }
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.dSize;
        constInfo.mm2Kb = constInfo.dSizeV;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = 0;
        }
    } else if constexpr (layout == LayOutTypeEnum::LAYOUT_NTD) {
        // NG(BS)D
        constInfo.s1BaseDv = s1BaseSize * constInfo.dSizeV;
        constInfo.s2BaseDv = s2BaseSize * constInfo.dSizeV;
        if constexpr (hasRope) {
            constInfo.mm1RopeKa = constInfo.dSizeRope;
            constInfo.mm1RopeKb = constInfo.dSizeRope;
        }
        constInfo.mm1Ka = constInfo.dSize;
        constInfo.mm1Kb = constInfo.dSize;
        constInfo.mm2Kb = constInfo.dSizeV;
        if ASCEND_IS_AIV {
            constInfo.attentionOutStride = 0;
        }
    }

    if ASCEND_IS_AIV {
        auto &inputParamsRegbase = this->tilingData->inputParamsRegbase;
        if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            pseInfo.pseLayoutType = inputParamsRegbase.pseShapeType;
            pseInfo.pseType = inputParamsRegbase.pseType;
            pseInfo.pseBSize = inputParamsRegbase.pseBSize;
            pseInfo.pseS1Size = inputParamsRegbase.pseS1Size;
            pseInfo.pseS2Size = inputParamsRegbase.pseS2Size;
            pseInfo.pseEncodeType = (uint32_t)inputParamsRegbase.pseEncodeType;
            pseInfo.pseStride = (pseInfo.pseLayoutType == (uint32_t)PseLayoutTypeEnum::PSE_1S2) ? 0 : s2BaseSize;
            pseInfo.qStartIdx = inputParamsRegbase.qStartIdx;
            pseInfo.kvStartIdx = inputParamsRegbase.kvStartIdx;
            if (inputParamsRegbase.pseShapeType == (uint32_t)PseLayoutTypeEnum::PSE_1S2) {
                constInfo.gS2 = constInfo.gSize * constInfo.s2Size;
            }
        }

        if constexpr (hasAtten) {
            attenMaskInfo.preTokens = sharedParams.preTokens;
            attenMaskInfo.nextTokens = sharedParams.nextTokens;
            attenMaskInfo.compressMode = inputParamsRegbase.attenMaskCompressMode;
            attenMaskInfo.attenMaskShapeType = inputParamsRegbase.attenMaskShapeType;
            attenMaskInfo.attenMaskS1Size = inputParamsRegbase.attenMaskS1Size;
            attenMaskInfo.attenMaskS2Size = inputParamsRegbase.attenMaskS2Size;
            attenMaskInfo.bandIndex = inputParamsRegbase.bandIndex;
        }
        constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
    }

    GetDerived()->InitUniqueConstInfo();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::Process()
{
    GetDerived()->Process();
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::GetSeqQlenKvlenByBoidx(int64_t boIdx,
    int64_t &actualSeqQlen, int64_t &actualSeqKvlen)
{
    if (unlikely(boIdx == 0)) {
        actualSeqQlen = actualSeqQlenAddr[0];
        actualSeqKvlen = actualSeqKvlenAddr[0];
        return;
    }
    actualSeqQlen = actualSeqQlenAddr[boIdx] - actualSeqQlenAddr[boIdx - 1];
    actualSeqKvlen = actualSeqKvlenAddr[boIdx] - actualSeqKvlenAddr[boIdx - 1];
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeAxisIdx(
    int64_t multiCoreInnerIdx, RunParamStr<isInfer> &runParam)
{
    // 计算轴的idx
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
        int64_t actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);

        while (multiCoreInnerIdx >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += runParam.actualS1Size;
            this->s2SizeAcc += runParam.actualS2Size;
            runParam.b1SSOffset += runParam.actualS1Size * runParam.actualS2Size;
            if constexpr (hasDrop) {
                runParam.b1SSOffsetAlign16 += runParam.actualS1Size * Align(runParam.actualS2Size);
            }
            runParam.boIdx++;
            if (runParam.boIdx >= this->sharedParams.bSize) {
                break;
            }
            GetSeqQlenKvlenByBoidx(runParam.boIdx, runParam.actualS1Size, runParam.actualS2Size);
            actualS1Outersize = this->s1OuterSizeAcc + (CeilDiv(runParam.actualS1Size, this->s1BaseSize) * constInfo.n2G);
        }

        int64_t tmpS1Outersize = CeilDiv(runParam.actualS1Size, this->s1BaseSize);
        actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
        runParam.n2oIdx = actualS1Outersize / tmpS1Outersize / this->sharedParams.gSize;
        runParam.goIdx = actualS1Outersize / tmpS1Outersize % this->sharedParams.gSize;
        runParam.s1oIdx = actualS1Outersize % tmpS1Outersize;
    } else {
        runParam.boIdx = multiCoreInnerIdx / constInfo.n2GS1o;
        runParam.n2oIdx = multiCoreInnerIdx % constInfo.n2GS1o / constInfo.gS1o;
        runParam.goIdx = multiCoreInnerIdx % constInfo.gS1o / constInfo.s1OuterSize;
        runParam.s1oIdx = multiCoreInnerIdx % constInfo.s1OuterSize;
        runParam.b1SSOffset = runParam.boIdx * constInfo.s1S2;
        runParam.actualS1Size = constInfo.s1Size;
        runParam.actualS2Size = constInfo.s2Size;
        if constexpr (hasDrop) {
            runParam.b1SSOffsetAlign16 = runParam.boIdx * constInfo.s1Size * Align(constInfo.s2Size);
        }
    }
    runParam.s1RealSize = Min(s1BaseSize, runParam.actualS1Size - runParam.s1oIdx * s1BaseSize);
    if constexpr (useDn || useNz) {
        runParam.s1RealSizeAlign32 = (runParam.s1RealSize + 31) >> 5 << 5;
        runParam.halfS1RealSize = runParam.s1RealSize <= 16 ? runParam.s1RealSize : (runParam.s1RealSizeAlign32 >> 1);
    } else {
        runParam.halfS1RealSize = (runParam.s1RealSize + 1) >> 1;
    }
    runParam.firstHalfS1RealSize = runParam.halfS1RealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfS1RealSize = runParam.s1RealSize - runParam.halfS1RealSize;
    }
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::SetRunInfo(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, int64_t taskId, int64_t s2LoopCount, int64_t s2LoopLimit, int64_t multiCoreInnerIdx)
{
    runInfo.s2StartIdx = runParam.s2LineStartIdx;
    runInfo.s2EndIdx = runParam.s2LineEndIdx;
    runInfo.s2LoopCount = s2LoopCount;
    if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
        runInfo.s1oIdx = runParam.s1oIdx;
        runInfo.boIdx = runParam.boIdx;
        runInfo.n2oIdx = runParam.n2oIdx;
        runInfo.goIdx = runParam.goIdx;
        runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
        runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
        runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
    }
    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_NTD) {
        runInfo.boIdx = runParam.boIdx;
        runInfo.s1ScaleNumAcc = s1ScaleNumAcc;
        runInfo.s2ScaleNumAcc = s2ScaleNumAcc;
        runInfo.s1SizeAcc = s1SizeAcc;
        runInfo.s2SizeAcc = s2SizeAcc;
    } else {
        runInfo.s2SizeAcc = runInfo.boIdx * constInfo.s2Size;
    }
    runInfo.taskId = taskId;
    runInfo.taskIdMod2 = taskId & 1;
    runInfo.taskIdMod3 = taskId % 3;
    runInfo.s2LoopLimit = s2LoopLimit;

    if constexpr (isFd) {
        runInfo.flashDecodeS2Idx = this->aicIdx % constInfo.splitKVNum;
    }
    runInfo.actualS1Size = runParam.actualS1Size;
    runInfo.actualS2Size = runParam.actualS2Size;
    runInfo.attentionOutOffset = runParam.attentionOutOffset;
    runInfo.queryOffset = runParam.tensorQOffset;
    runInfo.qRopeOffset = runParam.qRopeNBGOffset;
    if constexpr (isInfer) {
        runInfo.sOuterOffset = runParam.sOuterOffset;
    }
    this->ComputeBmm1Tail(runInfo, runParam);
    GetDerived()->InitUniqueRunInfo(runParam, runInfo);
}

template <typename ChildClass, typename CubeBlockType, typename VecBlockType>
__aicore__ inline void FlashAttentionScoreKernelBase<ChildClass, CubeBlockType, VecBlockType>::ComputeBmm1Tail(
    RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam)
{
    // ------------------------S1 Base Related---------------------------
    runInfo.s1RealSize = runParam.s1RealSize;
    runInfo.s1RealSizeAlign32 = runParam.s1RealSizeAlign32;
    runInfo.halfS1RealSize = runParam.halfS1RealSize;
    runInfo.firstHalfS1RealSize = runParam.firstHalfS1RealSize;

    runInfo.vec2S1BaseSize = runInfo.halfS1RealSize;  // D>128 这里需要适配

    // ------------------------S2 Base Related----------------------------
    runInfo.s2RealSize = s2BaseSize;
    runInfo.s2AlignedSize = runInfo.s2RealSize;
    if constexpr (enableKVPrefix) {
        if ((runInfo.s2LoopCount + runInfo.s2StartIdx / s2BaseSize) < constInfo.prefixLoopCount) {
            if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > constInfo.actualKVPrefixSize) {
                runInfo.s2RealSize = constInfo.actualKVPrefixSize - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
                runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
            }
        } else {
            if (runInfo.s2StartIdx + (runInfo.s2LoopCount - constInfo.prefixLoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx - constInfo.actualKVPrefixSize) {
                runInfo.s2RealSize = (runInfo.s2EndIdx - constInfo.actualKVPrefixSize) - (runInfo.s2LoopCount - constInfo.prefixLoopCount) * runInfo.s2RealSize - runInfo.s2StartIdx;
                runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
            }
        }
    } else {
        if (runInfo.s2StartIdx + (runInfo.s2LoopCount + 1) * runInfo.s2RealSize > runInfo.s2EndIdx) {
            runInfo.s2RealSize = runInfo.s2EndIdx - runInfo.s2LoopCount * runInfo.s2RealSize - runInfo.s2StartIdx;
            runInfo.s2AlignedSize = Align(runInfo.s2RealSize);
        }
    }
}
}
#endif // FLASH_ATTENTION_SCORE_KERNEL_BASE_H_