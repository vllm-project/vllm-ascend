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
 * \file flash_attention_score_antiquant_block_cube.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_ANTIQUANT_BLOCK_CUBE_H_
#define FLASH_ATTENTION_SCORE_ANTIQUANT_BLOCK_CUBE_H_
#include "flash_attention_score_tiling_regbase.h"
using namespace regbaseutil;
using namespace optiling;
namespace BaseApi {
ANTIQUANT_TEMPLATES_DEF
class FABlockCubeAntiquant {
public:
    /*Compile Static Variable*/
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dBaseSize = (uint32_t)dTemplateType;
    static constexpr uint32_t dBaseSizediv4 = dBaseSize / 4;
    static constexpr uint32_t dTemplateAlign64 = Align64FuncAntiquantup((uint16_t)dVTemplateType);
    static constexpr uint32_t mm1ResultSize = s1BaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t mm2ResultSize = s1BaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t mm2LeftSize = s1BaseSize * s2BaseSize * sizeof(Q_T);
    static constexpr uint32_t mm1LeftSize = s1BaseSize * dBaseSize * sizeof(Q_T);
    static constexpr uint32_t kvAntiquantResSize = s2BaseSize * dTemplateAlign64 * sizeof(Q_T);
    static constexpr bool useDn = false;

    /*Public Function*/
    __aicore__ inline FABlockCubeAntiquant() {};
    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        TPipe *tPipe, BufferManager<BufferType::L1> *l1BuffMgr);
    __aicore__ inline void SendCrossCoreFlag();
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<Q_T> &l1Tensor, const GlobalTensor<Q_T> &gmTensor, 
        uint32_t ndNum, uint32_t nValue, uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue, uint32_t dstNzC0Stride, uint32_t dstNzMatrixStride);
    __aicore__ inline void PrepareMm1Input(Buffer<BufferType::L1> &mm1A, RunInfo<isInfer> &runInfo,
        RunParamStr<isInfer> &runParam, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm1(RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, const int64_t &subTaskId,
        Buffer<BufferType::L1> &mm1B, LocalTensor<T> &outputTensor, ConstInfo<isInfer, hasRope> &constInfo);
    __aicore__ inline void IterateBmm2(const int64_t &subTaskId, const RunInfo<isInfer> &runInfo, Buffer<BufferType::L1> &inputBufA,
    Buffer<BufferType::L1> &inputBufB, LocalTensor<T> &outputTensor, ConstInfo<isInfer, hasRope> &constInfo);

    GlobalTensor<Q_T> queryGm;

private:
    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L1> mm1AL1Buffers;
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;
    TPipe *tPipe;
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData;
};

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitCubeBlock(
    __gm__ uint8_t *query, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
    TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr)
{
    this->tPipe = pipe;
    this->tilingData = tiling;
    this->l1BufferManagerPtr = l1BuffMgr;
    this->queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::SendCrossCoreFlag()
{
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[0]);
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[0]); // 一个aic对应两个aiv,  一共32个核，所以是16
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[1]);
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[1]); // 一个aic对应两个aiv,  一共32个核，所以是16
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::InitLocalBuffer()
{
    mm1AL1Buffers.Init((*l1BufferManagerPtr), mm1LeftSize);
    l0aBufferManager.Init(this->tPipe, L0A_SIZE * KB_TO_BYTES); // 初始化l0a的总内存为64K
    l0bBufferManager.Init(this->tPipe, L0B_SIZE * KB_TO_BYTES); // 初始化l0b的总内存为64K
    l0cBufferManager.Init(this->tPipe, L0C_SIZE * KB_TO_BYTES); // 初始化l0c的总内存为256K
    mmL0ABuffers.Init(l0aBufferManager, (L0A_SIZE / DOUBLE_BUFFER_NUM) * KB_TO_BYTES);  // 双缓冲 l0a内存大小为32K
    mmL0BBuffers.Init(l0bBufferManager, (L0B_SIZE / DOUBLE_BUFFER_NUM) * KB_TO_BYTES);  // 双缓冲 l0b内存大小32K
    mmL0CBuffers.Init(l0cBufferManager, (L0C_SIZE / DOUBLE_BUFFER_NUM) * KB_TO_BYTES); // 双缓冲 l0c内存大小128K
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::CopyToL1Nd2Nz(
    const LocalTensor<Q_T> &l1Tensor, const GlobalTensor<Q_T> &gmTensor, 
    uint32_t ndNum, uint32_t nValue, uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue, uint32_t dstNzC0Stride, uint32_t dstNzMatrixStride)
{
    Nd2NzParams gm2L1Nd2NzParams;

    gm2L1Nd2NzParams.ndNum = ndNum; // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue; // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue; // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = srcNdMatrixStride; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.dstNzC0Stride = (dstNzC0Stride + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数 15 >> 4 << 4 is Align 16
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride = dstNzMatrixStride; // 两个NZ矩阵，起始地址之间的偏移，单位为元素数量
    
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::PrepareMm1Input(
    Buffer<BufferType::L1> &mm1A, RunInfo<isInfer> &runInfo,
    RunParamStr<isInfer> &runParam, ConstInfo<isInfer, hasRope> &constInfo)
{
    if (unlikely(runInfo.s2LoopCount == 0)) {
        mm1A = mm1AL1Buffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用
        LocalTensor<Q_T> mm1ATensor = mm1A.GetTensor<Q_T>();

        if constexpr (isInfer) {
            if ((constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH) ||
                constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_TND)) &&
                constInfo.isPfaGS1Merge) {
                CopyToL1Nd2Nz(mm1ATensor, this->queryGm[runParam.tensorQOffset], constInfo.s1Size,
                    constInfo.gSize, constInfo.dSize, constInfo.n2Size * constInfo.gSize * constInfo.dSize,
                    constInfo.dSize, runInfo.s1RealSize, constInfo.gSize * 32 / sizeof(Q_T));
            } else {
                CopyToL1Nd2Nz(mm1ATensor, this->queryGm[runParam.tensorQOffset], 1,
                    runInfo.s1RealSize, constInfo.dSize, 0, constInfo.mm1Ka, runInfo.s1RealSize, 0);
            }
        } else {
            CopyToL1Nd2Nz(mm1ATensor, this->queryGm[runParam.tensorQOffset], 1,
                runInfo.s1RealSize, constInfo.dSize, 0, constInfo.mm1Ka, runInfo.s1RealSize, 0);
        }
        
        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else {
        mm1A = mm1AL1Buffers.GetPre();
        mm1A.Set<HardEvent::MTE2_MTE1>();
    }
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::IterateBmm1(
        RunInfo<isInfer> &runInfo, RunParamStr<isInfer> &runParam, const int64_t &subTaskId,
        Buffer<BufferType::L1> &mm1B, LocalTensor<T> &outputTensor, ConstInfo<isInfer, hasRope> &constInfo)
{
    Buffer<BufferType::L1> mm1A;
    PrepareMm1Input(mm1A, runInfo, runParam, constInfo);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(VC_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + VC_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer

    mm1A.Wait<HardEvent::MTE2_MTE1>();

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s1RealSize,
                    (uint32_t)runInfo.s2RealSize,
                    (uint32_t)(constInfo.dSize),
                    0,
                    1
                    };
    MatmulK<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseSizediv4, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<Q_T>(), mm1B.GetTensor<Q_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        mm1A.Set<HardEvent::MTE1_MTE2>();
    }

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(VC_MM1RES_EVENT[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + VC_MM1RES_EVENT[runInfo.taskIdMod2]); // 16 is Vec num

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐 7 >> 3 <<3 is Align
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) 15 is align
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    if constexpr (isInfer) {
        bool isS1Odd = constInfo.s1Size % 2 != 0; // BSNGD/TNGD GS1合轴时，若s1为奇数且开启双目标模式，扩展M维度对齐g，避免计算中间块
        if ((constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_BSH) || constInfo.layoutType == static_cast<uint8_t>(LayOutTypeEnum::LAYOUT_TND)) && constInfo.isPfaGS1Merge && isS1Odd) {
            fixpipeParams.mSize = runInfo.s1RealSize + constInfo.gSize;
        }
    }

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputTensor, mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CV_MM1RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + CV_MM1RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG, 16 is Vec num

    mm1ResL0C.Set<HardEvent::FIX_M>();
}

ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeAntiquant<ANTIQUANT_TEMPLATE_ARGS>::IterateBmm2(
    const int64_t &subTaskId, const RunInfo<isInfer> &runInfo, Buffer<BufferType::L1> &inputBufA,
    Buffer<BufferType::L1> &inputBufB, LocalTensor<T> &outputTensor, ConstInfo<isInfer, hasRope> &constInfo)
{
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(VC_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + VC_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer

    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)s1BaseSize,  // singleM 64
                    (uint32_t)constInfo.dSizeV,  // singleN 512
                    (uint32_t)runInfo.s2RealSize,  // singleK 128
                    false,    // isLeftTranspose
                    false     // isRightTranspose 
                    };
    MatmulN<Q_T, Q_T, T, s1BaseSize, dBaseSizediv4, s2BaseSize, ABLayout::MK, ABLayout::KN>(
                                inputBufA.GetTensor<Q_T>(), 
                                inputBufB.GetTensor<Q_T>(),
                                mmL0ABuffers,
                                mmL0BBuffers,
                                mm2ResL0C.GetTensor<T>(),
                                param);
    // inputBufA.Set<HardEvent::MTE1_MTE2>(); // 释放
    // inputBufB.Set<HardEvent::MTE1_MTE2>(); // 释放

    mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待

    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(CV_L1_EVENT[subTaskId % 2]); // 2 is double buffer
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(16 + CV_L1_EVENT[subTaskId % 2]); // 16 is Vec num, 2 is double buffer
    
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(VC_MM2RES_EVENT[runInfo.taskIdMod2]);
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(16 + VC_MM2RES_EVENT[runInfo.taskIdMod2]); // 16 is Vec num

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (constInfo.dSizeV +7) >> 3 << 3; // 7 >> 3 << 3 8位对齐
    fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) / 16) * 16; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) 15 is align
    fixpipeParams.dstStride = ((uint32_t)dVTemplateType + 15) >> 4 << 4; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除 15 >> 4 << 4 is align
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputTensor, mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB
    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(CV_MM2RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + CV_MM2RES_EVENT[runInfo.taskIdMod2]);  // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG, 16 is aiv num
}


ANTIQUANT_TEMPLATES_DEF
class FABlockCubeAntiquantDummy {
public:
    __aicore__ inline void InitCubeBlock(__gm__ uint8_t *query, const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        TPipe *pipe, BufferManager<BufferType::L1> *l1BuffMgr) {}
    __aicore__ inline void SendCrossCoreFlag() {}
    __aicore__ inline void InitLocalBuffer() {}
};

template <typename T>
struct CubeBlockTraitsAntiquant;

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE_ANTIQUANT(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST_ANTIQAUNT(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS_ANTIQUANT(CUBE_BLOCK_CLASS_ANTIQUANT) \
    ANTIQUANT_TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraitsAntiquant<CUBE_BLOCK_CLASS_ANTIQUANT<ANTIQUANT_TEMPLATE_ARGS>> { \
        ANTIQUANT_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE_ANTIQUANT) \
        ANTIQUANT_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST_ANTIQAUNT) \
    }

DEFINE_CUBE_BLOCK_TRAITS_ANTIQUANT(FABlockCubeAntiquant);
DEFINE_CUBE_BLOCK_TRAITS_ANTIQUANT(FABlockCubeAntiquantDummy);

#define GEN_ARGS_TYPE_ANTIQUANT(name, ...) using name = typename CubeBlockTraitsAntiquant<AntiquantCubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST_ANTIQUANT(name, type, ...) static constexpr type name = CubeBlockTraitsAntiquant<AntiquantCubeBlockType>::name##Traits;
#define ARGS_TRAITS_ANTIQUANT \
    ANTIQUANT_CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE_ANTIQUANT)\
    ANTIQUANT_CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST_ANTIQUANT)
}
#endif