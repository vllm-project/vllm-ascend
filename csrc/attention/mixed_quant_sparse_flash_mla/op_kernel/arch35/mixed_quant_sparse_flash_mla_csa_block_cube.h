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
 * \file mixed_quant_sparse_flash_mla_csa_block_cube.h
 * \brief
 */
#ifndef MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
#define MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H

#include "../vendor/c6240b268/attention/common/op_kernel/offset_calculator.h"
#include "../vendor/c6240b268/attention/common/op_kernel/matmul.h"
#include "../vendor/c6240b268/attention/common/op_kernel/FixpipeOut.h"
#include "../vendor/c6240b268/attention/common/op_kernel/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"
#include "util_regbase.h"
#include "mixed_quant_sparse_flash_mla_common_arch35.h"
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/static_matmul.h"
#include "../vendor/c6240b268/attention/sparse_flash_mla/op_kernel/arch35/common/cube_local_buffer.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
template <QSMLA_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == QSMLA_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
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

    __aicore__ inline CSABlockCube(){};
    __aicore__ inline void InitLocalBuffer(uint32_t l1BaseAddr);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                            const ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void IterateLoadQK(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                         RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo,
                                         bool isFirstLoop);
    __aicore__ inline void IterateBmm1(StaticBuffer<T> &outputBuf,
                                       Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       bool notLastTwoLoop, RunInfo<HIGH_PERF> &runInfoNext,
                                       RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void FreeEvent();

    __aicore__ inline void IterateBmm2(StaticBuffer<T> &outputBuf, StaticBuffer<Q_T> &l1PBuffer,
                                       RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);

private:
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                        const ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void CopyQGmToL1(RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    __aicore__ inline void IterateBmm1CSA(StaticBuffer<T> &outputBuf,
                                          Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          bool notLastTwoLoop, RunInfo<HIGH_PERF> &runInfoNext,
                                          RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2CSA(StaticBuffer<T> &outputBuf, StaticBuffer<Q_T> &l1PBuffer,
                                          RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo);
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr bool Q_WITH_ZERO_HEAD = (LAYOUT_T == QSMLA_LAYOUT::TND);
    FaGmTensor<Q_T, Q_FORMAT, int32_t, Q_WITH_ZERO_HEAD> queryGm;

    /* =====================运行时变量==================== */
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
    AttentionCommon::InitCubeLocalBuffer<Q_T, T>(l1QBufs, l1RightBufs, l0ABufs, l0A, l0BBufs, l0B, l0CBufs, l0C,
                                                 l1BaseAddr);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *cuSeqlensQ,
                                                                     __gm__ uint8_t *sequsedQ,
                                                                     const ConstInfo<HIGH_PERF> &constInfo)
{
    if ASCEND_IS_AIC {
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitGmTensor(cuSeqlensQ, sequsedQ, constInfo);
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
                                                                 const ConstInfo<HIGH_PERF> &constInfo)
{
    if constexpr (LAYOUT_T == QSMLA_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize, constInfo.s1Size,
                                            constInfo.qInputD);
    } else { // QSMLA_LAYOUT::TND
        uint32_t mqsmlaSequsedQSize = (sequsedQ == nullptr) ? 0 : constInfo.bSize;
        ActualSeqLensParser<ActualSeqLensMode::ACCUM, int32_t, true> mqsmlaParser;
        mqsmlaParser.Init(cuSeqlensQ, constInfo.bSize + 1, sequsedQ, mqsmlaSequsedQSize);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.qInputD, mqsmlaParser);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    StaticBuffer<T> &outputBuf, Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo<HIGH_PERF> &runInfoNext, RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo)
{
    IterateBmm1CSA(outputBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2(StaticBuffer<T> &outputBuf,
                                                                StaticBuffer<Q_T> &l1PBuffer,
                                                                RunInfo<HIGH_PERF> &runInfo,
                                                                ConstInfo<HIGH_PERF> &constInfo)
{
    IterateBmm2CSA(outputBuf, l1PBuffer, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo<HIGH_PERF> &runInfo,
                                                                ConstInfo<HIGH_PERF> &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
                                                                 runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2U; i++) {
        uint32_t mqsmlaCurL1QBufId = (l1QBufId + i) % 3U;
        WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(mqsmlaCurL1QBufId));
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        if (constInfo.hasRopeInput) {
            CopyToL1Nd2Nz<Q_T>(l1QBufs[mqsmlaCurL1QBufId].tensor, this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize,
                               constInfo.dSize >> 1, constInfo.mm1Ka);
        } else {
            // MTE1_MTE2 above released this buffer. Initialize every NZ element on reuse.
            auto zeroTensor = l1QBufs[mqsmlaCurL1QBufId].tensor.template ReinterpretCast<half>();
            InitConstValueParams<half> zeroParams;
            zeroParams.repeatTimes = 1;
            zeroParams.blockNum = Align16Func(runInfo.mRealSize) * (constInfo.dSize >> 1) * sizeof(Q_T) / 32;
            zeroParams.dstGap = 0;
            zeroParams.initValue = static_cast<half>(0);
            InitConstValue(zeroTensor, zeroParams);
            PipeBarrier<PIPE_MTE2>();
            // Q GM: 448 columns. Internal halves: 256 and 192+64 zeros.
            uint32_t copyD = i == 0 ? 256U : 192U;
            CopyToL1Nd2Nz<Q_T>(l1QBufs[mqsmlaCurL1QBufId].tensor, this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize,
                               copyD, constInfo.mm1Ka);
        }
        SetFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1Q(mqsmlaCurL1QBufId));
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, RunInfo<HIGH_PERF> &runInfo,
    ConstInfo<HIGH_PERF> &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }
    WaitFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(l1KLoadBufId));
    LocalTensor<Q_T> dst = l1RightBufs[runInfo.taskIdMod3].tensor;
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE2>(15U);
        CrossCoreWaitFlag<0, PIPE_MTE2>(15U);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * constInfo.dSize);
    SetFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1KV(l1KLoadBufId));
    l1KLoadBufId = (l1KLoadBufId + 1) % 3U;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm1CSA(
    StaticBuffer<T> &outputBuf, Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo<HIGH_PERF> &runInfoNext, RunInfo<HIGH_PERF> &runInfo, ConstInfo<HIGH_PERF> &constInfo)
{
    WaitFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1KV(l1KMatmul1BufId));
    l1KMatmul1BufId = (l1KMatmul1BufId + 1) % 3U;

    StaticBuffer<T> &mqsmlaCBuf = l0C.GetNext();
    WaitFlag<HardEvent::FIX_M>(INNERCORE_L0C(mqsmlaCBuf.idx));
    MMParam mqsmlaParam = {
        static_cast<uint32_t>(runInfo.mRealSize),    // singleM
        static_cast<uint32_t>(runInfo.s2RealSize),   // singleN
        static_cast<uint32_t>(constInfo.dSize >> 1), // singleK
        0,                                           // isLeftTranspose
        1                                            // isRightTranspose
    };
    uint32_t mqsmlaCurL1QBufId = l1QBufId;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1Q(mqsmlaCurL1QBufId));
    }
    LocalTensor<Q_T> curL1RightTensor = l1RightBufs[runInfo.taskIdMod3].tensor;
    MatmulKStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QBufs[mqsmlaCurL1QBufId].tensor, curL1RightTensor, l0A, l0B, mqsmlaCBuf.tensor, mqsmlaParam);

    mqsmlaCurL1QBufId = (mqsmlaCurL1QBufId + 1) % 3U;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        WaitFlag<HardEvent::MTE2_MTE1>(INNERCORE_L1Q(mqsmlaCurL1QBufId));
    }
    mqsmlaParam.singleK = constInfo.dSize - mqsmlaParam.singleK;
    mqsmlaParam.isOutKFisrt = false;
    MatmulKStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QBufs[mqsmlaCurL1QBufId].tensor, curL1RightTensor[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
        l0A, l0B, mqsmlaCBuf.tensor, mqsmlaParam);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(l1QBufId));
        SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1Q(mqsmlaCurL1QBufId));
        l1QBufId = (l1QBufId + 2U) % 3U;
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    SetFlag<HardEvent::M_FIX>(INNERCORE_L0C(mqsmlaCBuf.idx));
    WaitFlag<HardEvent::M_FIX>(INNERCORE_L0C(mqsmlaCBuf.idx));

    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx) + AIV0_AIV1_OFFSET);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> mqsmlaFixpipeParams; // L0C→UB
    // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    mqsmlaFixpipeParams.mSize = Align2Func(runInfo.mRealSize);
    // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    mqsmlaFixpipeParams.nSize = Align8Func(runInfo.s2RealSize);
    // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) //
    // 源Nz矩阵中相邻大Z排布的起始地址偏移
    mqsmlaFixpipeParams.srcStride = Align16Func(mqsmlaFixpipeParams.mSize);
    mqsmlaFixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到,
                                                // ND方案(S1*S2)时脏数据用mask剔除
    mqsmlaFixpipeParams.params.ndNum = 1;
    mqsmlaFixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    mqsmlaFixpipeParams.params.srcNdStride = 0;
    mqsmlaFixpipeParams.params.dstNdStride = 0;

    // 将matmul结果从L0C搬运到UB
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, mqsmlaCBuf.tensor, mqsmlaFixpipeParams);
    SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(mqsmlaCBuf.idx));
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx));
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM1(outputBuf.idx) + AIV0_AIV1_OFFSET);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void CSABlockCube<TEMPLATE_ARGS>::IterateBmm2CSA(StaticBuffer<T> &outputBuf,
                                                                   StaticBuffer<Q_T> &l1PBuffer,
                                                                   RunInfo<HIGH_PERF> &runInfo,
                                                                   ConstInfo<HIGH_PERF> &constInfo)
{
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P(l1PBuffer.idx));
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(CROSSCORE_L1P(l1PBuffer.idx) + AIV0_AIV1_OFFSET);

    StaticBuffer<T> &mqsmlaCBuf = l0C.GetNext();
    WaitFlag<HardEvent::FIX_M>(INNERCORE_L0C(mqsmlaCBuf.idx));
    MMParam mqsmlaParam = {
        static_cast<uint32_t>(runInfo.mRealSize),  // singleM 64
        static_cast<uint32_t>(constInfo.dSizeV),   // singleN 512
        static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
        0,                                         // isLeftTranspose
        0                                          // isRightTranspose
    };
    LocalTensor<Q_T> curL1RightTensor = l1RightBufs[runInfo.taskIdMod3].tensor;
    MatmulNStatic<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.tensor, curL1RightTensor, l0A, l0B, mqsmlaCBuf.tensor, mqsmlaParam);

    SetFlag<HardEvent::M_FIX>(INNERCORE_L0C(mqsmlaCBuf.idx));
    WaitFlag<HardEvent::M_FIX>(INNERCORE_L0C(mqsmlaCBuf.idx));
    SetFlag<HardEvent::MTE1_MTE2>(INNERCORE_L1KV(l1KMatmul2BufId));
    l1KMatmul2BufId = (l1KMatmul2BufId + 1) % 3; // 3：循环使用三个L1 KV缓冲区

    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> mqsmlaFixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    mqsmlaFixpipeParams.nSize =
        Align8Func(constInfo.dSizeV); // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    mqsmlaFixpipeParams.mSize = Align2Func(
        runInfo
            .mRealSize); // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    mqsmlaFixpipeParams.srcStride = Align16Func(
        mqsmlaFixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    mqsmlaFixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    mqsmlaFixpipeParams.dualDstCtl = 1;
    mqsmlaFixpipeParams.params.ndNum = 1;
    mqsmlaFixpipeParams.params.srcNdStride = 0;
    mqsmlaFixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.tensor, mqsmlaCBuf.tensor, mqsmlaFixpipeParams);
    SetFlag<HardEvent::FIX_M>(INNERCORE_L0C(mqsmlaCBuf.idx));

    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2);
    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(CROSSCORE_BMM2 + AIV0_AIV1_OFFSET);
}

TEMPLATES_DEF
class CSABlockCubeDummy {
public:
    __aicore__ inline CSABlockCubeDummy(){};
    __aicore__ inline void InitLocalBuffer(uint32_t l1BaseAddr) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *query, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedQ,
                                            const ConstInfo<HIGH_PERF> &constInfo)
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
} // namespace BaseApi
#endif // MIXED_QUANT_SPARSE_FLASH_MLA_CSA_BLOCK_CUBE_H
