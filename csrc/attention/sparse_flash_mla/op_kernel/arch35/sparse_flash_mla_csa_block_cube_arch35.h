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
 * \file sparse_flash_mla_csa_block_cube_arch35.h
 * \brief
 */
#ifndef SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H__ARCH35
#define SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H__ARCH35
#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "sparse_flash_mla_common_arch35.h"
#include "common/static_matmul.h"

#if __has_include("../../common/op_kernel/offset_calculator.h")
#include "../../common/op_kernel/offset_calculator.h"
#else
#include "../common/offset_calculator.h"
#endif
#if __has_include("../../common/op_kernel/matmul.h")
#include "../../common/op_kernel/matmul.h"
#else
#include "../common/matmul.h"
#endif
#if __has_include("../../common/op_kernel/FixpipeOut.h")
#include "../../common/op_kernel/FixpipeOut.h"
#else
#include "../common/FixpipeOut.h"
#endif
#if __has_include("../../common/op_kernel/CopyInL1.h")
#include "../../common/op_kernel/CopyInL1.h"
#else
#include "../common/CopyInL1.h"
#endif

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace SMLAKernel {
struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t s1Coord;
    uint32_t s2Coord;
};

template <SMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

template <SMLA_LAYOUT KV_LAYOUT_T>
__aicore__ inline constexpr GmFormat GetKvGmFormat()
{
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        return GmFormat::PA_BnBsND;
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        return GmFormat::TND;
    } else { // BSND
        return GmFormat::BSND;
    }
}

TEMPLATES_DEF
class CSABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;
    static constexpr uint32_t rightBufNum = 3;
    static constexpr uint32_t rightBufSingleSize = s2BaseSize * dBaseSize;
    static constexpr uint32_t rightBufTotalSize = rightBufSingleSize * rightBufNum;
    static constexpr uint32_t l1QBufNum = 3;                // L1 Q 三缓冲
    static constexpr uint32_t l1KBufNum = 3;                // L1 K 三缓冲
    static constexpr uint32_t qHalfNum = 2;                 // Q 沿 d 轴切半
    static constexpr uint32_t crossCoreMte2SyncFlagId = 15; // IS_SPLIT_G 核间 MTE2 同步 flag ID

    __aicore__ inline CSABlockCube(){};
    __aicore__ inline void InitLocalBuffer(uint32_t l1BaseAddr);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv,
                                            __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                            __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
                                            __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                            __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                            __gm__ uint8_t *seqUsedCmpKV, const ConstInfo &constInfo);

    // SWA/HCA场景
    __aicore__ inline void IterateLoadQK(RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);

    // CSA场景
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                         RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(StaticBuffer<T> &output, bool notLastTwoLoop, RunInfo &runInfoNext,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2(StaticBuffer<T> &outputBuf, StaticBuffer<Q_T> &l1PBuffer, const RunInfo &runInfo,
                                       const ConstInfo &constInfo);
    __aicore__ inline void FreeEvent();

private:
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                        __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv,
                                        __gm__ uint8_t *seqUsedOriKV, __gm__ uint8_t *seqUsedCmpKV,
                                        const ConstInfo &constInfo);
    __aicore__ inline void CalcS2Coord(const RunInfo &runInfo, const ConstInfo &constInfo);
    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void LoadKGmToL1(LocalTensor<KV_T> &inputRightTensor, const RunInfo &runInfo,
                                       const ConstInfo &constInfo);

    __aicore__ inline void IterateBmm1Impl(StaticBuffer<T> &outputBuf, bool notLastTwoLoop, RunInfo &runInfoNext,
                                           RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm2Impl(StaticBuffer<T> &outputBuf, StaticBuffer<Q_T> &l1PBuffer,
                                           const RunInfo &runInfo, const ConstInfo &constInfo);

    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKvGmFormat<KV_LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == SMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;
    static constexpr bool KV_WITH_ZERO_HEAD = (KV_LAYOUT_T == SMLA_LAYOUT::TND);
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> oriKvGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> cmpKvGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    FaGmTensor<KV_T, KV_FORMAT, int32_t, KV_WITH_ZERO_HEAD> curKvGm;
    GlobalTensor<int32_t> cuSeqlensQGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t l1QBufId = 0; // 3 buffer, 0-2 (轮转游标)
    uint32_t l1KLoadBufId = 0;
    uint32_t l1KMatmul1BufId = 0;
    uint32_t l1KMatmul2BufId = 0;
    /* =====================LocalBuffer变量==================== */
    StaticBuffer<Q_T> l1QBufs[3];
    StaticBuffer<Q_T> l1RightBufs[3];
    StaticBuffer<Q_T> l0ABufs[2];
    RingBuffer<Q_T> l0A;
    StaticBuffer<Q_T> l0BBufs[2];
    RingBuffer<Q_T> l0B;
    StaticBuffer<T> l0CBufs[2];
    RingBuffer<T> l0C;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitLocalBuffer(uint32_t l1BaseAddr)
{
    if ASCEND_IS_AIC {
        uint32_t l1Addr = l1BaseAddr;

        l1QBufs[0] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 0};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
        l1QBufs[1] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 1};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);
        l1QBufs[2] = {LocalTensor<Q_T>(TPosition::A1, l1Addr, L1Q_ELEM_PER_BUF), 2};
        l1Addr += L1Q_ELEM_PER_BUF * sizeof(Q_T);

        l1RightBufs[0] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 0};
        l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);
        l1RightBufs[1] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 1};
        l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);
        l1RightBufs[2] = {LocalTensor<Q_T>(TPosition::B1, l1Addr, L1_RIGHT_ELEM_PER_BLOCK), 2};
        l1Addr += L1_RIGHT_ELEM_PER_BLOCK * sizeof(Q_T);

        uint32_t l0aAddr = 0;
        l0ABufs[0] = {LocalTensor<Q_T>(TPosition::A2, l0aAddr, L0A_ELEM_PER_BUF), 0};
        l0aAddr += L0A_ELEM_PER_BUF * sizeof(Q_T);
        l0ABufs[1] = {LocalTensor<Q_T>(TPosition::A2, l0aAddr, L0A_ELEM_PER_BUF), 1};

        uint32_t l0bAddr = 0;
        l0BBufs[0] = {LocalTensor<Q_T>(TPosition::B2, l0bAddr, L0B_ELEM_PER_BUF), 0};
        l0bAddr += L0B_ELEM_PER_BUF * sizeof(Q_T);
        l0BBufs[1] = {LocalTensor<Q_T>(TPosition::B2, l0bAddr, L0B_ELEM_PER_BUF), 1};

        uint32_t l0cAddr = 0;
        l0CBufs[0] = {LocalTensor<T>(TPosition::CO1, l0cAddr, L0C_ELEM_PER_BUF), 0};
        l0cAddr += L0C_ELEM_PER_BUF * sizeof(T);
        l0CBufs[1] = {LocalTensor<T>(TPosition::CO1, l0cAddr, L0C_ELEM_PER_BUF), 1};

        l0A = RingBuffer<Q_T>(l0ABufs, 2);
        l0B = RingBuffer<Q_T>(l0BBufs, 2);
        l0C = RingBuffer<T>(l0CBufs, 2);

        SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(0));
        SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(1));
        SetFlag<HardEvent::M_MTE1>(INNERCORE_L0AB(0));
        SetFlag<HardEvent::M_MTE1>(INNERCORE_L0AB(1));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(0));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(1));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(2));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(0));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(1));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(2));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(
    __gm__ uint8_t *query, __gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv, __gm__ uint8_t *cmpSparseIndices,
    __gm__ uint8_t *oriBlockTable, __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *cuSeqlensQ,
    __gm__ uint8_t *cuSeqlensOriKv, __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
    __gm__ uint8_t *seqUsedCmpKV, const ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        this->oriKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)oriKv);
        if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
            this->oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);
        }
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                      TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
            this->cmpKvGm.gmTensor.SetGlobalBuffer((__gm__ KV_T *)cmpKv);
            if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
                this->cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
            }
        }
        if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
            this->cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
        }
        InitGmTensor(cuSeqlensQ, sequsedQ, cuSeqlensOriKv, cuSeqlensCmpKv, seqUsedOriKV, seqUsedCmpKV, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::FreeEvent()
{
    WaitFlag<HardEvent::M_MTE1>(INNERCORE_L0AB(0));
    WaitFlag<HardEvent::M_MTE1>(INNERCORE_L0AB(1));
    WaitFlag<HardEvent::FIX_M>(INNERCORE_L0C(0));
    WaitFlag<HardEvent::FIX_M>(INNERCORE_L0C(1));
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(0));
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(1));
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(2)); // 2 for l1q buffer id
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(0));
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(1));
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(2)); // 2 for l1kv buffer id
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                                                 __gm__ uint8_t *cuSeqlensOriKv,
                                                                 __gm__ uint8_t *cuSeqlensCmpKv,
                                                                 __gm__ uint8_t *seqUsedOriKV,
                                                                 __gm__ uint8_t *seqUsedCmpKV,
                                                                 const ConstInfo &constInfo)
{
    if constexpr (LAYOUT_T == SMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                            constInfo.dSize);
    } else { // SMLA_LAYOUT::TND
        cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        uint32_t sequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensQ, constInfo.bSize + 1, sequsedQ, sequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize, parser);
    }

    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.oriBlockSize, constInfo.dSize,
                                            this->oriBlockTableGm, constInfo.oriMaxBlockNumPerBatch);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.cmpBlockSize, constInfo.dSize,
                                                this->cmpBlockTableGm, constInfo.cmpMaxBlockNumPerBatch);
        }
    } else if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::TND) {
        uint32_t seqUsedOriKvSize = (seqUsedOriKV == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
        parser.Init(cuSeqlensOriKv, constInfo.actualSeqLenSize + 1, seqUsedOriKV, seqUsedOriKvSize);
        this->oriKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize, parser);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                      TEMPLATE_MODE != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) {
            uint32_t seqUseCmpKvSize = (seqUsedCmpKV == nullptr) ? 0 : constInfo.bSize;
            ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> parser;
            parser.Init(cuSeqlensCmpKv, constInfo.actualSeqLenSize + 1, seqUsedCmpKV, seqUseCmpKvSize);
            this->cmpKvGm.offsetCalculator.Init(constInfo.n2Size, constInfo.dSize, parser);
        }
    } else {
        // BSND不需要初始化
        this->oriKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.s2Size, constInfo.dSize);
        if constexpr (TEMPLATE_MODE != SMLATemplateMode::SWA_TEMPLATE_MODE) {
            this->cmpKvGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.cmpS2Size,
                                                constInfo.dSize); // 替换constInfo.s2Size / constInfo.cmpRatio
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CalcS2Coord(const RunInfo &runInfo, const ConstInfo &constInfo)
{
    // 计算s2方向偏移
    coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.boIdx;
    if (runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx) {
        kvCacheBlockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord =
            runInfo.s2StartIdx + (runInfo.s2LoopCount - runInfo.oriKvLoopEndIdx) * s2BaseSize;
        blockTableGm = cmpBlockTableGm;
        curKvGm = cmpKvGm;
    } else {
        kvCacheBlockSize = constInfo.oriBlockSize;
        maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
        coordInfo[runInfo.taskIdMod3].s2Coord = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
        blockTableGm = oriBlockTableGm;
        curKvGm = oriKvGm;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
                                                                 runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < qHalfNum; i++) {
        uint32_t curL1QBufId = (l1QBufId + i) % l1QBufNum;
        WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(curL1QBufId));
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QBufs[curL1QBufId].tensor, this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize,
                           constInfo.dSize >> 1, constInfo.mm1Ka);
        SetFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1Q(curL1QBufId));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::LoadKGmToL1(LocalTensor<KV_T> &inputRightTensor,
                                                                const RunInfo &runInfo, const ConstInfo &constInfo)
{
    if constexpr (KV_LAYOUT_T == SMLA_LAYOUT::PA_BBND) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = coordInfo[runInfo.taskIdMod3].s2Coord;
        startPos.dIdx = 0;
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = constInfo.n2Size;
        shape.headDim = constInfo.dSize;
        shape.actHeadDim = constInfo.dSize;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4; // 15，4：进行16字节对齐处理
        shape.pageStride = runInfo.isCmp ? constInfo.cmpKeyStride0 : constInfo.oriKeyStride0;
        GmCopyInToL1PA<KV_T>(inputRightTensor, curKvGm.gmTensor, blockTableGm, KVLAYOUT::BBH, shape, startPos);
    } else {
        int64_t keyOffset = this->curKvGm.offsetCalculator.GetOffset(
            coordInfo[runInfo.taskIdMod3].curBIdx, runInfo.n2oIdx, coordInfo[runInfo.taskIdMod3].s2Coord, 0);
        CopyToL1Nd2Nz<KV_T>(inputRightTensor, curKvGm.gmTensor[keyOffset], runInfo.s2RealSize, constInfo.dSize,
                            constInfo.mm1Kb);
    }
}

// SWA/HCA场景: K从GM直接搬运
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(RunInfo &runInfo, ConstInfo &constInfo,
                                                                  bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    // 加载当前轮的右矩阵到L1
    CalcS2Coord(runInfo, constInfo);
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(l1KLoadBufId));
    LocalTensor<KV_T> dst = l1RightBufs[runInfo.taskIdMod3].tensor;
    LoadKGmToL1(dst, runInfo, constInfo);
    SetFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1KV(l1KLoadBufId));
    l1KLoadBufId = (l1KLoadBufId + 1) % l1KBufNum;
}

// CSA场景: K来源取决于isSparse
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, RunInfo &runInfo, ConstInfo &constInfo,
    bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    // ORI_SPARSE、ORI_CMP_SPARSE及CSA的cmpKv为v0稀疏搬运，CSA的oriKv为cube搬运
    bool isSparse = false;
    if constexpr (TEMPLATE_MODE == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                  TEMPLATE_MODE == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        isSparse = true;
    } else if constexpr (TEMPLATE_MODE == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        isSparse = runInfo.isCmp ? true : false;
    }

    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(l1KLoadBufId));
    LocalTensor<Q_T> dst = l1RightBufs[runInfo.taskIdMod3].tensor;
    if (!isSparse) {
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
            CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
        }
        // cube直接从kv cache搬运K
        CalcS2Coord(runInfo, constInfo);
        LoadKGmToL1(dst, runInfo, constInfo);
    } else {
        v0ResGm.WaitCrossCore();
        if constexpr (IS_SPLIT_G) {
            CrossCoreSetFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
            CrossCoreWaitFlag<0, PIPE_MTE2>(crossCoreMte2SyncFlagId);
        }
        GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
        CopyToL1Nd2Nz<Q_T>(dst, v0ResGmTensor, runInfo.s2RealSize, constInfo.dSize, constInfo.mm1Kb);
    }
    SetFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1KV(l1KLoadBufId));
    l1KLoadBufId = (l1KLoadBufId + 1) % l1KBufNum;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1(StaticBuffer<T> &outputBuf, bool notLastTwoLoop,
                                                                RunInfo &runInfoNext, RunInfo &runInfo,
                                                                ConstInfo &constInfo)
{
    IterateBmm1Impl(outputBuf, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2(StaticBuffer<T> &outputBuf,
                                                                StaticBuffer<Q_T> &l1PBuffer, const RunInfo &runInfo,
                                                                const ConstInfo &constInfo)
{
    IterateBmm2Impl(outputBuf, l1PBuffer, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1Impl(StaticBuffer<T> &outputBuf, bool notLastTwoLoop,
                                                                    RunInfo &runInfoNext, RunInfo &runInfo,
                                                                    ConstInfo &constInfo)
{
    LocalTensor<Q_T> curL1RightTensor = l1RightBufs[runInfo.taskIdMod3].tensor;
    WaitFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1KV(l1KMatmul1BufId));
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % l1KBufNum;

    StaticBuffer<T> &cBuf = l0C.GetNext();
    WaitFlag<HardEvent::FIX_M>(INNERCORE_L0C(cBuf.idx));
    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),    // singleM
        static_cast<uint32_t>(runInfo.s2RealSize),   // singleN
        static_cast<uint32_t>(constInfo.dSize >> 1), // singleK
        0,                                           // isLeftTranspose
        1                                            // isRightTranspose
    };
    uint32_t curL1QBufId = l1QBufId;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1Q(curL1QBufId));
    }

    // m,n不切，k切128，mm1B直接用tensor的数据
    MatmulKStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QBufs[curL1QBufId].tensor, curL1RightTensor, l0A, l0B, cBuf.tensor, param);

    curL1QBufId = (curL1QBufId + 1) % l1QBufNum;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1Q(curL1QBufId));
    }
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;

    // m,n不切，k切128, mm1B直接用tensor的数据
    MatmulKStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QBufs[curL1QBufId].tensor, curL1RightTensor[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)], l0A,
        l0B, cBuf.tensor, param);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(l1QBufId));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(curL1QBufId));
        l1QBufId = (l1QBufId + qHalfNum) % l1QBufNum;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    SetFlag<HardEvent::M_FIX>(INNERCORE_L0C(cBuf.idx));
    WaitFlag<HardEvent::M_FIX>(INNERCORE_L0C(cBuf.idx));

    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx) + AIV0_AIV1_OFFSET);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) //
    // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dstStride = s2BaseSize;
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, cBuf.tensor, fixpipeParams);
    SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(cBuf.idx));
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx) + AIV0_AIV1_OFFSET);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2Impl(StaticBuffer<T> &outputBuf,
                                                                    StaticBuffer<Q_T> &l1PBuffer,
                                                                    const RunInfo &runInfo, const ConstInfo &constInfo)
{
    LocalTensor<Q_T> curL1RightTensor = l1RightBufs[runInfo.taskIdMod3].tensor;
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P(l1PBuffer.idx));
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P(l1PBuffer.idx) + AIV0_AIV1_OFFSET);

    StaticBuffer<T> &cBuf = l0C.GetNext();
    WaitFlag<HardEvent::FIX_M>(INNERCORE_L0C(cBuf.idx));
    MMParam param = {
        static_cast<uint32_t>(runInfo.mRealSize),  // singleM
        static_cast<uint32_t>(constInfo.dSizeV),   // singleN 512
        static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
        0,                                         // isLeftTranspose
        0                                          // isRightTranspose
    };
    MatmulNStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.tensor, curL1RightTensor, l0A, l0B, cBuf.tensor, param);

    SetFlag<HardEvent::M_FIX>(INNERCORE_L0C(cBuf.idx));
    WaitFlag<HardEvent::M_FIX>(INNERCORE_L0C(cBuf.idx));
    SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(l1KMatmul2BufId));
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % l1KBufNum;

    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
    // L0C→UB;FixpipeParamsM300:L0C→UB
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize);
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, cBuf.tensor, fixpipeParams); // 将matmul结果从L0C搬运到UB
    SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(cBuf.idx));

    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
}

TEMPLATES_DEF
class CSABlockCubeDummy {
public:
    __aicore__ inline CSABlockCubeDummy(){};
    __aicore__ inline void InitLocalBuffer(uint32_t l1BaseAddr) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *oriKv, __gm__ uint8_t *cmpKv,
                                            __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                            __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
                                            __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensOriKv,
                                            __gm__ uint8_t *cuSeqlensCmpKv, __gm__ uint8_t *seqUsedOriKV,
                                            __gm__ uint8_t *seqUsedCmpKV, const ConstInfo &constInfo)
    {}
    __aicore__ inline void FreeEvent() {}
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(CSABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(CSABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace SMLAKernel
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
