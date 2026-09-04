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
 * \file flash_attention_block_cube_noquant_mla.h
 * \brief
 */

#ifndef FLASH_ATTENTION_BLOCK_CUBE_NOQUANT_MLA_H_
#define FLASH_ATTENTION_BLOCK_CUBE_NOQUANT_MLA_H_

#include "kernel_operator.h"
#include "../matmul.h"
#include "kernel_tiling/kernel_tiling.h"
#include "infer_flash_attention_comm.h"
#include "flash_attention_score_common_regbase.h"
#include "flash_attention_score_block_cube.h"

using namespace fa_base_matmul;
using namespace BaseApi;

TEMPLATES_DEF
class FABlockCubeNoquantMla {
public:
    __aicore__ inline FABlockCubeNoquantMla() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe,
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
        BufferManager<BufferType::L1>* l1BuffMgr,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>* mm12Bmm2AL1BuffersPtr);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer>& runInfo, RunParamStr<isInfer>& runParam, bool isLast, ConstInfo<isInfer, hasRope>& constInfo);
    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        RunInfo<isInfer>& runInfo, ConstInfo<isInfer, hasRope>& constInfo);

private:
    __aicore__ inline void InitInput(TPipe *pipe, __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        const FlashAttentionScoreSimplifiedTilingData *__restrict tiling);
    __aicore__ inline void InitLocalBuffer(BufferManager<BufferType::L1>* l1BuffMgr,
        BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>* mm12Bmm2AL1BuffMgr);
    __aicore__ inline int64_t GetQueryRopeOffset(RunInfo<isInfer>& runInfo, ConstInfo<isInfer, hasRope>& constInfo);
    __aicore__ inline int64_t GetKeyRopeOffset(RunInfo<isInfer>& runInfo, ConstInfo<isInfer, hasRope>& constInfo);

public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr uint32_t s1BaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr bool hasPse = pseMode != PseTypeEnum::PSE_NONE_TYPE;
    static constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;

private:
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = nullptr;
    TPipe *tPipe = nullptr;

    GlobalTensor<INPUT_T> queryGm;
    __gm__ uint8_t* currentKey;
    GlobalTensor<INPUT_T> keyGm;
    __gm__ uint8_t* currentValue;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> queryRopeGm;
    GlobalTensor<INPUT_T> keyRopeGm;
    // block_table
    GlobalTensor<int32_t> blockTableGm;
    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    KVLAYOUT kvLayout;

    BufferManager<BufferType::L1>* l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>* mm12Bmm2AL1BuffersPtr;
    // mm1左矩阵，GS1循环内左矩阵复用，GS1循环间不开pingpong
    BuffersPolicySingleBuffer<BufferType::L1> mm1AL1Buffers;
    // L0A
    BuffersPolicyDB<BufferType::L0A> mmL0ABuffers;
    // L0B
    BuffersPolicyDB<BufferType::L0B> mmL0BBuffers;
    // L0C
    BuffersPolicyDB<BufferType::L0C> mmL0CBuffers;

    static constexpr uint64_t kvHeadNum = 1ULL;
    static constexpr uint64_t headDim = 512ULL;
    static constexpr uint64_t headDimRope = 64ULL;

    // keyOffset记录，value总是可以使用上一次key的offset
    int64_t keyRopeOffset[3];

    static constexpr bool mm2RightStillInL1 = true;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeNoquantMla<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling,
    BufferManager<BufferType::L1>* l1BuffMgr,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>* mm12Bmm2AL1BuffMgr)
{
    InitInput(pipe, query, key, value, blockTable, queryRope, keyRope, tiling);
    InitLocalBuffer(l1BuffMgr, mm12Bmm2AL1BuffMgr);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeNoquantMla<TEMPLATE_ARGS>::InitInput(
    TPipe *pipe, __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    const FlashAttentionScoreSimplifiedTilingData *__restrict tiling)
{
    this->tilingData = tiling;
    this->tPipe = pipe;

    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T *)query);
    ListTensorDesc keyListTensorDescInit((__gm__ void*)key);
    ListTensorDesc valueListTensorDescInit((__gm__ void*)value);
    currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    if (this->tilingData->inputParamsRegbase.isKvContinuous == 1) {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)currentKey);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)currentValue);
    } else {
        this->keyGm.SetGlobalBuffer((__gm__ INPUT_T *)key);
        this->valueGm.SetGlobalBuffer((__gm__ INPUT_T *)value);
    }
    if constexpr (isPa) {
        this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        this->kvCacheBlockSize = this->tilingData->inputParamsRegbase.blockSize;
        this->maxBlockNumPerBatch = this->tilingData->inputParamsRegbase.blockTableDim2;
        if (this->tilingData->inputParamsRegbase.paLayoutType == 2) { // NZ下paLayoutType == 2
            kvLayout = KVLAYOUT::NZ;
        } else {
            kvLayout = this->tilingData->inputParamsRegbase.paLayoutType == 1 ? KVLAYOUT::BBH : KVLAYOUT::BNBD;
        }
    }
    if constexpr (hasRope) {
        this->queryRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)queryRope);
        this->keyRopeGm.SetGlobalBuffer((__gm__ INPUT_T *)keyRope);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeNoquantMla<TEMPLATE_ARGS>::InitLocalBuffer(BufferManager<BufferType::L1>* l1BuffMgr,
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>* mm12Bmm2AL1BuffMgr)
{
    if ASCEND_IS_AIC {
        l1BufferManagerPtr = l1BuffMgr;
        l0aBufferManager.Init(tPipe, 65536);
        l0bBufferManager.Init(tPipe, 65536);
        l0cBufferManager.Init(tPipe, 262144);

        // s1=64, s2=128分核方案，mm1和mm2结果全部在ub上
        if constexpr(s1BaseSize == 64 && s2BaseSize == 128) {
            // 保存p结果的L1内存必须放在第一个L1 policy上，保证和vec申请的地址相同
            mm12Bmm2AL1BuffersPtr = mm12Bmm2AL1BuffMgr; // L1P与L1K_rope复用
            mm1AL1Buffers.Init(*l1BufferManagerPtr, (uint32_t)dTemplateType * s1BaseSize * 2);
            // L0A B C当前写死，要改成通过计算获取
            mmL0ABuffers.Init(l0aBufferManager, 32 * 1024);
            mmL0BBuffers.Init(l0bBufferManager, 32 * 1024);
            mmL0CBuffers.Init(l0cBufferManager, 128 * 1024);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeNoquantMla<TEMPLATE_ARGS>::IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    RunInfo<isInfer>& runInfo, RunParamStr<isInfer>& runParam, bool isLast, ConstInfo<isInfer, hasRope>& constInfo)
{
    Buffer<BufferType::L1> mm1A;
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm1B;
    // 左矩阵复用 ,s2的第一次循环加载左矩阵
    // 加载左矩阵到L1 当前使用全载方式
    if (unlikely(runInfo.s2LoopCount == 0)) { // sOuter循环第一个基本块：搬运0
        mm1A = mm1AL1Buffers.Get();
        mm1A.Wait<HardEvent::MTE1_MTE2>(); // 占用
        LocalTensor<INPUT_T> mm1ATensor = mm1A.GetTensor<INPUT_T>();
        Nd2NzParams Gm2L1Nd2NzParams;
        Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
        Gm2L1Nd2NzParams.nValue = runInfo.s1RealSize; // 单个ND矩阵的实际行数，单位为元素个数
        Gm2L1Nd2NzParams.dValue = constInfo.dSize; // 单个ND矩阵的实际列数，单位为元素个数
        Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.srcDValue = constInfo.mm1Ka; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
        DataCopy(mm1ATensor, this->queryGm[runParam.tensorQOffset], Gm2L1Nd2NzParams);

        // 拷贝 Qrope
        if constexpr (hasRope) {
            int64_t queryRopeOffset = GetQueryRopeOffset(runInfo, constInfo);
            Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
            Gm2L1Nd2NzParams.nValue = runInfo.s1RealSize; // 单个ND矩阵的实际行数，单位为元素个数
            Gm2L1Nd2NzParams.dValue = constInfo.dSizeRope; // 单个ND矩阵的实际列数，单位为元素个数
            Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.srcDValue = constInfo.mm1RopeKa; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
            DataCopy(mm1ATensor[Gm2L1Nd2NzParams.dstNzC0Stride * constInfo.dSize], this->queryRopeGm[queryRopeOffset], Gm2L1Nd2NzParams);
        }

        mm1A.Set<HardEvent::MTE2_MTE1>(); // 通知
    } else { // 非s2的第一次循环直接复用Q
        mm1A = mm1AL1Buffers.GetPre();
        // 左矩阵复用时，sinner循环内不需要MTE2同步等待
        // mm1A.Wait<HardEvent::MTE1_MTE2>();
        mm1A.Set<HardEvent::MTE2_MTE1>();
    }

    // 加载当前轮的右矩阵到L1
    mm1B = mm12Bmm2AL1BuffersPtr->Get();
    // mm1B.Wait<HardEvent::MTE1_MTE2>(); // 占用
    WaitFlag<HardEvent::MTE1_MTE2>(mm1B.GetEventID<HardEvent::MTE1_MTE2>());
    LocalTensor<INPUT_T> mm1BTensor = mm1B.GetTensor<INPUT_T>();
    if constexpr (isPa) {
        Position startPos;
        startPos.bIdx = runInfo.boIdx;
        startPos.n2Idx = runInfo.n2oIdx;
        startPos.s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * s2BaseSize;
        startPos.dIdx = 0; 
        PAShape shape;
        shape.blockSize = kvCacheBlockSize;
        shape.headNum = kvHeadNum;
        shape.headDim = headDim;
        shape.actHeadDim = headDim;
        shape.maxblockNumPerBatch = maxBlockNumPerBatch;
        shape.copyRowNum = runInfo.s2RealSize;
        shape.copyRowNumAlign = (runInfo.s2RealSize + 15) >> 4 << 4;
        shape.pageStride = this->tilingData->inputParamsRegbase.keyStrides.bnStride;
        shape.n2Stride = this->tilingData->inputParamsRegbase.keyStrides.n2Stride;
        PAShape ropeShape = shape;
        ropeShape.headDim = headDimRope;
        ropeShape.actHeadDim = headDimRope;
        ropeShape.pageStride = this->tilingData->inputParamsRegbase.kRopeStrides.bnStride;
        ropeShape.n2Stride = this->tilingData->inputParamsRegbase.kRopeStrides.n2Stride;
        uint32_t dstNzC0Stride = (runInfo.s2RealSize + 15) >> 4 << 4;
        LocalTensor<INPUT_T> mm1BRopeTensor = mm1BTensor[dstNzC0Stride * constInfo.dSize];
        GlobalTensor<INPUT_T> mm1BNopeGmTensor = this->keyGm;
        GlobalTensor<INPUT_T> mm1BRopeGmTensor = this->keyRopeGm;
        GmCopyInToL1HasRopePA<INPUT_T>(mm1BTensor, mm1BRopeTensor, mm1BNopeGmTensor, mm1BRopeGmTensor, blockTableGm, kvLayout, shape, ropeShape, startPos);
    } else {
        Nd2NzParams Gm2L1Nd2NzParams;
        Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
        Gm2L1Nd2NzParams.nValue = runInfo.s2RealSize; // 单个ND矩阵的实际行数，单位为元素个数
        Gm2L1Nd2NzParams.dValue = constInfo.dSize; // 单个ND矩阵的实际列数，单位为元素个数
        Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.srcDValue = constInfo.mm1Kb; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
        Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
        Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
        DataCopy(mm1BTensor, this->keyGm[runInfo.keyOffset], Gm2L1Nd2NzParams);

        // 拷贝 Krope
        if constexpr (hasRope) {
            keyRopeOffset[runInfo.taskIdMod3] = GetKeyRopeOffset(runInfo, constInfo);
            Nd2NzParams Gm2L1Nd2NzParams;
            Gm2L1Nd2NzParams.ndNum = 1; // ND矩阵的个数
            Gm2L1Nd2NzParams.nValue = runInfo.s2RealSize; // 单个ND矩阵的实际行数，单位为元素个数
            Gm2L1Nd2NzParams.dValue = constInfo.dSizeRope; // 单个ND矩阵的实际列数，单位为元素个数
            Gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.srcDValue = constInfo.mm1RopeKb; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
            Gm2L1Nd2NzParams.dstNzC0Stride = (Gm2L1Nd2NzParams.nValue + 15) >> 4 << 4; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
            Gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
            DataCopy(mm1BTensor[Gm2L1Nd2NzParams.dstNzC0Stride * constInfo.dSize], this->keyRopeGm[keyRopeOffset[runInfo.taskIdMod3]], Gm2L1Nd2NzParams);
        }
    }
    SetFlag<HardEvent::MTE2_MTE1>(mm1B.GetEventID<HardEvent::MTE2_MTE1>());

    mm1A.Wait<HardEvent::MTE2_MTE1>(); // 等待L1A
    WaitFlag<HardEvent::MTE2_MTE1>(mm1B.GetEventID<HardEvent::MTE2_MTE1>());

    Buffer<BufferType::L0C> mm1ResL0C = mmL0CBuffers.Get();
    mm1ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)runInfo.s1RealSize,  // singleM 64
                     (uint32_t)runInfo.s2RealSize,  // singleN 128
                     (uint32_t)(constInfo.dSize + constInfo.dSizeRope), // singleK 576
                     0,    // isLeftTranspose
                     1     // isRightTranspose 
                    };
    
    // 这里base M N K不要写死
    MatmulK<INPUT_T, INPUT_T, T, s1BaseSize, s2BaseSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        mm1A.GetTensor<INPUT_T>(), mm1B.GetTensor<INPUT_T>(),
        mmL0ABuffers, mmL0BBuffers,
        mm1ResL0C.GetTensor<T>(),
        param);
    if (unlikely(runInfo.s2LoopCount == runParam.s2LoopEndIdx - 1)) {
        mm1A.Set<HardEvent::MTE1_MTE2>();
    }
    // mm1B.Set<HardEvent::MTE1_MTE2>(); // 释放

    mm1ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm1ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    
    outputBuf.WaitCrossCore();

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = (runInfo.s2RealSize + 7) >> 3 << 3; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = (runInfo.s1RealSize + 1) >> 1 << 1; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm1ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    mm1ResL0C.Set<HardEvent::FIX_M>(); // 释放
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t FABlockCubeNoquantMla<TEMPLATE_ARGS>::GetQueryRopeOffset(RunInfo<isInfer>& runInfo, ConstInfo<isInfer, hasRope>& constInfo)
{
    // 计算gm上的offset
    int64_t bOffsetRope = 0;
    // s1需要考虑inner轴的影响
    int64_t s1OffsetRope = 0;
    int64_t n2OffsetRope = 0;
    int64_t gOffsetRope = 0;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffsetRope = runInfo.s1SizeAcc * constInfo.n2GDR;
        s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
        n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
        gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
    }  else {
        if constexpr (layout == LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSNGD
            bOffsetRope = runInfo.boIdx * constInfo.n2GS1DR;
            s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.gDR;
            gOffsetRope = runInfo.goIdx * constInfo.dSizeRope;
        } else if constexpr (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            // bnsd
            bOffsetRope = runInfo.boIdx * constInfo.n2GS1DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.gS1DR;
            gOffsetRope = runInfo.goIdx * constInfo.s1DR;
            s1OffsetRope = runInfo.s1oIdx * constInfo.s1BaseDR;
        }
    } 
    int64_t ret = bOffsetRope + n2OffsetRope + gOffsetRope + s1OffsetRope;
    if ((layout == LayOutTypeEnum::LAYOUT_TND || layout == LayOutTypeEnum::LAYOUT_BSH) && runInfo.nextTokensPerBatch < 0) {
        ret += (-runInfo.nextTokensPerBatch) * constInfo.dSizeRope;
    }
    return ret;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t FABlockCubeNoquantMla<TEMPLATE_ARGS>::GetKeyRopeOffset(RunInfo<isInfer>& runInfo, ConstInfo<isInfer, hasRope>& constInfo)
{
    // 计算gm上的offset
    int64_t bOffsetRope = 0;
    int64_t n2OffsetRope = 0;
    int64_t s2OffsetRope = 0;

    if constexpr (layout == LayOutTypeEnum::LAYOUT_TND) {
        // (BS)ND
        bOffsetRope = runInfo.s2SizeAcc * constInfo.n2DR;
        s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + runInfo.s2LoopCount * constInfo.s2BaseN2DR;
        n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
    } else {
        if (layout == LayOutTypeEnum::LAYOUT_BSH) {
            // BSH/BSND
            bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
            s2OffsetRope = runInfo.s2StartIdx * constInfo.n2DR + runInfo.s2LoopCount * constInfo.s2BaseN2DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.dSizeRope;
        } else if (layout == LayOutTypeEnum::LAYOUT_BNSD) {
            // BNSD
            bOffsetRope = runInfo.boIdx * constInfo.n2S2DR;
            n2OffsetRope = runInfo.n2oIdx * constInfo.s2DR;
            s2OffsetRope = runInfo.s2StartIdx * constInfo.dSizeRope + runInfo.s2LoopCount * constInfo.s2BaseDR;
        }
    }
    return bOffsetRope + n2OffsetRope + s2OffsetRope;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void FABlockCubeNoquantMla<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    RunInfo<isInfer>& runInfo, ConstInfo<isInfer, hasRope>& constInfo)
{
    // 获取 mm2 的左右矩阵
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> mm2AB;
    if (mm2RightStillInL1) {
        mm2AB = mm12Bmm2AL1BuffersPtr->GetReused();
    }
    mm2AB.WaitCrossCore();

    uint64_t VDsize = (uint32_t)dVTemplateType;
    Buffer<BufferType::L0C> mm2ResL0C = mmL0CBuffers.Get();
    mm2ResL0C.Wait<HardEvent::FIX_M>(); // 占用
    MMParam param = {(uint32_t)s1BaseSize,  // singleM 64
                     (uint32_t)constInfo.dSizeV,  // singleN 512
                     (uint32_t)runInfo.s2RealSize,  // singleK 128
                     false,    // isLeftTranspose
                     false     // isRightTranspose 
                    };
    // 这里base M N K不要写死
    MatmulN<INPUT_T, INPUT_T, T, 64, 128, 128, ABLayout::MK, ABLayout::KN>(
                                mm2AB.GetTensor<INPUT_T>(s2BaseSize * VDsize), 
                                mm2AB.GetTensor<INPUT_T>(),
                                mmL0ABuffers,
                                mmL0BBuffers,
                                mm2ResL0C.GetTensor<T>(),
                                param);
    // mm2AB.Set<HardEvent::MTE1_MTE2>(); // 释放
    SetFlag<HardEvent::MTE1_MTE2>(mm2AB.GetEventID<HardEvent::MTE1_MTE2>());

    mm2ResL0C.Set<HardEvent::M_FIX>(); // 通知
    mm2ResL0C.Wait<HardEvent::M_FIX>(); // 等待L0C
    
    outputBuf.WaitCrossCore();

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C->UB
    fixpipeParams.nSize = constInfo.dSizeV; // L0C上的bmm1结果矩阵N方向的size大小；同mmadParams.n；8个元素（32B)对齐
    fixpipeParams.mSize = s1BaseSize; // 有效数据不足16行，只需输出部分行即可;L0C上的bmm1结果矩阵M方向的size大小必须是偶数
    fixpipeParams.srcStride = ((fixpipeParams.mSize + 15) >> 4) << 4; // L0C上matmul结果相邻连续数据片断间隔（前面一个数据块的头与后面数据块的头的间隔），单位为16 *sizeof(T) //源NZ矩阵中相邻Z排布的起始地址偏移
    fixpipeParams.dstStride = (fixpipeParams.nSize + 15) >> 4 << 4; // mmResUb上两行之间的间隔，单位：element。 // 128：根据比对dump文件得到，ND方案(S1 * S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分， M / 2 * N写入每个UB，M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(), mm2ResL0C.GetTensor<T>(), fixpipeParams); // 将matmul结果从L0C搬运到UB

    mm2ResL0C.Set<HardEvent::FIX_M>(); // 释放
    outputBuf.SetCrossCore();
}

DEFINE_CUBE_BLOCK_TRAITS(FABlockCubeNoquantMla);

#endif
