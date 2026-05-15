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
 * \file sparse_attn_sharedkv_scfa_block_cube.h
 * \brief
 */
#ifndef SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H_
#define SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H_
#include "common/offset_calculator.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {
template <SAS_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat() {
    if constexpr (LAYOUT == SAS_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class SCFABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;

    __aicore__ inline SCFABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers, 
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo);
    __aicore__ inline void CalcS1Coord(RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm1SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT> queryGm;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];

    /* =====================LocalBuffer变量==================== */
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;

    // D小于等于256 mm1左矩阵Q，GS1循环内左矩阵复用, GS1循环间开pingpong；D大于256使用单块Buffer，S1循环间驻留；fp32场景单块不驻留
    BuffersPolicySingleBuffer<BufferType::L1> l1QBuffers;

    // mm1右矩阵K
    BuffersPolicy3buff<BufferType::L1> l1KBuffers;

    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        l1BufferManagerPtr = l1BuffMgr;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitLocalBuffer() {
    constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    constexpr uint32_t mm1RightSize = dBaseSize * s2BaseSize * sizeof(Q_T);
    l1QBuffers.Init((*l1BufferManagerPtr), mm1LeftSize);
    l1KBuffers.Init((*l1BufferManagerPtr), mm1RightSize);

    // L0A B C 当前写死，能否通过基础api获取
    l0aBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0bBufferManager.Init(tPipe, L0AB_SHARED_SIZE_64K);
    l0cBufferManager.Init(tPipe, L0C_SHARED_SIZE_256K);

    mmL0ABuffers.Init(l0aBufferManager, BUFFER_SIZE_16K); // db类型，填入数值是总大小的一半
    mmL0BBuffers.Init(l0bBufferManager, BUFFER_SIZE_32K);
    mmL0CBuffers.Init(l0cBufferManager, BUFFER_SIZE_128K);
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // SAS_LAYOUT::TND
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            actualSeqQLen, constInfo.actualSeqLenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CalcS1Coord(RunInfo &runInfo,
    ConstInfo &constInfo)
{
    // 计算s1方向偏移
    coordInfo[runInfo.taskIdMod3].s1Coord = runInfo.s1oIdx * runInfo.qSNumInOneBlock;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    CalcS1Coord(runInfo, constInfo);

    IterateBmm1SCFA(outputBuf, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    IterateBmm2SCFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    Buffer<BufferType::L1> inputLeftBuf;
    // 左矩阵复用，S2的第一次循环加载左矩阵
    // 加载左矩阵到L1, 全载
    // query对ori_kv, cmp_kv都一样，无需区分
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运Q
        inputLeftBuf = l1QBuffers.Get();
        inputLeftBuf.Wait<HardEvent::MTE1_MTE2>(); // 占用L1A
        LocalTensor<Q_T> inputLeftTensor = inputLeftBuf.GetTensor<Q_T>();

        uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
            coordInfo[runInfo.taskIdMod3].s1Coord, 0);
        CopyToL1Nd2Nz<Q_T>(inputLeftTensor, this->queryGm.gmTensor[gmOffset], runInfo.mRealSize, constInfo.dSize,
            constInfo.mm1Ka);

        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非S2的第一次循环直接复用Q
        inputLeftBuf = l1QBuffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        inputLeftBuf.Set<HardEvent::MTE2_MTE1>(); // 通知
    }

    // 加载当前轮的右矩阵到L1
    inputRightBuf.WaitCrossCore();    // 核间同步，这里需要根据V0操作处理同步，确保取tensor时，数据已经准备好

    inputLeftBuf.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                        static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                        static_cast<uint32_t>(constInfo.dSize),   // singleK
                        0,    // isLeftTranspose
                        1     // isRightTranspose
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(  // m,n不切，k切128
        inputLeftBuf.GetTensor<Q_T>(), inputRightBuf.GetTensor<Q_T>(),                // mm1B直接用tensor的数据
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        inputLeftBuf.Set<HardEvent::MTE1_MTE2>(); // 释放L1A
    }

    mm1ResL0C.Set<HardEvent::M_FIX>();    // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>();   // 等待L0C

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize); // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize); // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放L0C
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
    ConstInfo &constInfo)
{
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> inputLeftBuf = inputLeftBuffers.Get(); // P直接用无需搬运
    inputLeftBuf.WaitCrossCore();

    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        inputLeftBuf.GetTensor<Q_T>(),
        inputRightBuf.GetTensor<Q_T>(),
        mmL0ABuffers,
        mmL0BBuffers,
        mm2ResL0C.GetTensor<T>(),
        param);

    inputLeftBuf.SetCrossCore();
    inputRightBuf.SetCrossCore();   // bmm2才释放KV，在这里释放

    mm2ResL0C.Set<HardEvent::M_FIX>();  // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    outputBuf.WaitCrossCore(); //占用
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.mSize = s1BaseSize;                      // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = Align16Func(s1BaseSize);     // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class SCFABlockCubeDummy {
public:
    __aicore__ inline SCFABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, BufferManager<BufferType::L1> *l1BufferManagerPtr, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo) {}
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf,
        RunInfo &runInfo, ConstInfo &constInfo) {}
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputRightBuf, RunInfo &runInfo,
        ConstInfo &constInfo) {}
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
    }

DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // FLASH_ATTENTION_SCORE_BLOCK_CUBE_H_
