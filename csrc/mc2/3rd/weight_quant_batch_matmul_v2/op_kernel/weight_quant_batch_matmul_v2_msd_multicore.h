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
 * \file weight_quant_batch_matmul_v2_msd_multicore.h
 * \brief
 */
#ifndef WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_MULTICORE_H
#define WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_MULTICORE_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "tool.h"
#include "weight_quant_batch_matmul_v2_constant.h"

using AscendC::AIC;
using AscendC::AIV;
using AscendC::AscendAntiQuant;
using AscendC::BlockReduceMax;
using AscendC::BlockReduceSum;
using AscendC::Ceil;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::Nd2NzParams;
using AscendC::ONE_BLK_SIZE;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::ReduceLastND;
using AscendC::ReduceOrder;
using AscendC::RESERVED_WORKSPACE;
using AscendC::SetAtomicAdd;
using AscendC::SetAtomicMax;
using AscendC::SetAtomicNone;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SyncAll;
using AscendC::TBuf;
using AscendC::TEventID;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::TQue;
using AscendC::WaitFlag;
using matmul::MatmulImpl;
using matmul::MatmulType;
using Mc2WeightQuantBatchMatmulV2::CeilAlign;
using Mc2WeightQuantBatchMatmulV2::CeilDiv;
using Mc2WeightQuantBatchMatmulV2::Min;
using Mc2WeightQuantBatchMatmulV2::SYNC_MODE2;

#if defined(__CCE_KT_TEST__)
using AscendC::ProcessLock;
#endif
using Mc2WeightQuantBatchMatmulV2::Mc2QuantType;
using Mc2WeightQuantBatchMatmulV2::PrecisionType;

namespace Mc2WeightQuantBatchMatmulV2Msd {
static constexpr int32_t SYNC_VECTOR_CUBE_FLAG = 1;
static constexpr int32_t SYNC_CUBE_VECTOR_FLAG = 2;
static constexpr float EXPAND_FACTOR_1 = 127.499;
static constexpr float EXPAND_FACTOR_2 = 127.499 * 127.499 * 2;

// 根据《Ascend C API参考》DataCopy相关限制, Nd2NzParams相关
static constexpr int32_t NVALUE_MAX = 16384;
static constexpr int32_t DVALUE_MAX = 65535;
static constexpr int32_t SRC_DVALUE_MIN = 1;
static constexpr int32_t SRC_DVALUE_MAX = 65535;
static constexpr int32_t DST_NZ_C0_STRIDE_MIN = 1;
static constexpr int32_t DST_NZ_C0_STRIDE_MAX = 16384;
// 根据《Ascend C API参考》DataCopy相关限制, DataCopyParams相关
static constexpr int32_t BLOCK_COUNT_MIN = 1;
static constexpr int32_t BLOCK_COUNT_MAX = 4095;
static constexpr int32_t BLOCK_LEN_MIN = 1;
static constexpr int32_t BLOCK_LEN_MAX = 65535;
// 根据《Ascend C API参考》校验DataCopy参数范围
__aicore__ inline void CheckDataCopyParams(uint32_t blockCount, uint32_t blockLen)
{
    ASCENDC_ASSERT(blockCount >= BLOCK_COUNT_MIN, { KERNEL_LOG(KERNEL_ERROR, "blockCount should >= 1"); });
    ASCENDC_ASSERT(blockCount <= BLOCK_COUNT_MAX, { KERNEL_LOG(KERNEL_ERROR, "blockCount should <= 65535"); });
    ASCENDC_ASSERT(blockLen >= BLOCK_LEN_MIN, { KERNEL_LOG(KERNEL_ERROR, "blockLen should >= 1"); });
    ASCENDC_ASSERT(blockLen <= BLOCK_LEN_MAX, { KERNEL_LOG(KERNEL_ERROR, "blockLen should <= 65535"); });
}
__aicore__ inline void CheckDataCopyNd2nzParams(uint32_t nValue, uint32_t dValue, uint32_t srcDValue,
                                                uint32_t dstNzC0Stride)
{
    ASCENDC_ASSERT(nValue <= NVALUE_MAX, { KERNEL_LOG(KERNEL_ERROR, "nValue should <= 16384"); });
    ASCENDC_ASSERT(dValue <= DVALUE_MAX, { KERNEL_LOG(KERNEL_ERROR, "dValue should <= 65535"); });
    ASCENDC_ASSERT(srcDValue >= SRC_DVALUE_MIN, { KERNEL_LOG(KERNEL_ERROR, "srcDValue should >= 1 "); });
    ASCENDC_ASSERT(srcDValue <= SRC_DVALUE_MAX, { KERNEL_LOG(KERNEL_ERROR, "srcDValue should <= 65535"); });
    ASCENDC_ASSERT(dstNzC0Stride >= DST_NZ_C0_STRIDE_MIN, { KERNEL_LOG(KERNEL_ERROR, "dstNzC0Stride should >= 1"); });
    ASCENDC_ASSERT(dstNzC0Stride <= DST_NZ_C0_STRIDE_MAX,
                   { KERNEL_LOG(KERNEL_ERROR, "dstNzC0Stride should <= 16384"); });
}

__aicore__ inline void SetDataCopyNd2nzParams(Nd2NzParams &nd2nzParams, uint32_t nValue, uint32_t dValue,
                                              uint32_t srcDValue, uint32_t dstNzC0Stride)
{
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = nValue;
    nd2nzParams.dValue = dValue;
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.srcDValue = srcDValue;
    nd2nzParams.dstNzC0Stride = dstNzC0Stride;
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t fullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (fullDim0 - dim0) * sizeof(T);
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    if (dim0 % (ONE_BLK_SIZE / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(dim0, static_cast<uint32_t>(32 / sizeof(T))) - dim0;
        padParams.paddingValue = 0;
    }
    SHORT_MIX_LOG("blockCount %d blockLen %d srcStride %d dstStride %d rightPadding %d", params.blockCount,
                  params.blockLen, params.srcStride, params.dstStride, padParams.rightPadding);
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    SHORT_MIX_LOG("dim1 %d dim0 %d dstFullDim0 %d blockCount %d blockLen %d srcStride %d dstStride %d", dim1, dim0,
                  dstFullDim0, params.blockCount, params.blockLen, params.srcStride, params.dstStride);
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void RowMaxFixTmpTensor(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                          const LocalTensor<T> &tmpTensor, uint32_t m, uint32_t k)
{
    // 假设k已对齐
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    for (uint32_t idxM = 0; idxM < m; ++idxM) {
        if (k >= 2 * elemsPerRepeat) {
            Max(tmpTensor[idxM * elemsPerRepeat], src[idxM * k], src[idxM * k + elemsPerRepeat], elemsPerRepeat);
        } else if (k > elemsPerRepeat) {
            DataCopy(tmpTensor[idxM * elemsPerRepeat], src[idxM * k], elemsPerRepeat);
            Max(tmpTensor[idxM * elemsPerRepeat], src[idxM * k], src[idxM * k + elemsPerRepeat], k - elemsPerRepeat);
        } else {
            DataCopy(tmpTensor[idxM * elemsPerRepeat], src[idxM * k], k);
        }
        PipeBarrier<PIPE_V>();

        uint32_t idxK = 2;
        for (; idxK < k / elemsPerRepeat; ++idxK) {
            Max(tmpTensor[idxM * elemsPerRepeat], tmpTensor[idxM * elemsPerRepeat],
                src[idxM * k + idxK * elemsPerRepeat], elemsPerRepeat);
            PipeBarrier<PIPE_V>();
        }

        int32_t tailK = k - idxK * elemsPerRepeat;
        if (tailK > 0) {
            Max(tmpTensor[idxM * elemsPerRepeat], tmpTensor[idxM * elemsPerRepeat],
                src[idxM * k + idxK * elemsPerRepeat], tailK);
            PipeBarrier<PIPE_V>();
        }
    }
    if (k >= elemsPerRepeat) {
        BlockReduceMax<float>(tmpTensor, tmpTensor, m, elemsPerRepeat, 8, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float>(dst, tmpTensor, 1, 8 * m, 8, 1, 8);
    } else {
        ReduceMax(dst, tmpTensor, tmpTensor, k, m, CeilDiv(k, 32U));
    }

    PipeBarrier<PIPE_V>();
    Brcb(dst, dst, (m + 7) / 8, {1, 8});
}

template <typename T>
__aicore__ inline void RowSumFixTmpTensor(const LocalTensor<T> &dst, const LocalTensor<T> &src,
                                          const LocalTensor<T> &tmpTensor, uint32_t m, uint32_t k)
{
    // 假设k已对齐
    for (uint32_t idxM = 0; idxM < m; ++idxM) {
        if (k >= 128) {
            Add(tmpTensor[idxM * 64], src[idxM * k], src[idxM * k + 64], 64);
        } else if (k > 64) {
            DataCopy(tmpTensor[idxM * 64], src[idxM * k], 64);
            Add(tmpTensor[idxM * 64], src[idxM * k], src[idxM * k + 64], k - 64);
        } else {
            DataCopy(tmpTensor[idxM * 64], src[idxM * k], k);
        }
        PipeBarrier<PIPE_V>();

        uint32_t idxK = 2;
        for (; idxK < k / 64; ++idxK) {
            Add(tmpTensor[idxM * 64], tmpTensor[idxM * 64], src[idxM * k + idxK * 64], 64);
            PipeBarrier<PIPE_V>();
        }

        int32_t tailK = k - idxK * 64;
        if (tailK > 0) {
            Add(tmpTensor[idxM * 64], tmpTensor[idxM * 64], src[idxM * k + idxK * 64], tailK);
            PipeBarrier<PIPE_V>();
        }
    }
    if (k >= 64) {
        BlockReduceSum<float>(tmpTensor, tmpTensor, m, 64, 8, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float>(dst, tmpTensor, 1, 8 * m, 8, 1, 8);
    } else {
        ReduceSum(dst, tmpTensor, tmpTensor, k, m, CeilDiv(k, 32U));
    }
    PipeBarrier<PIPE_V>();
    Brcb(dst, dst, (m + 7) / 8, {1, 8});
}

template <typename T>
__aicore__ inline void RowMaxInplace(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t m, uint32_t k)
{
    // k must be aligned
    // numRepeatK > 1
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t numRepeatK = k / elemsPerRepeat;
    uint32_t tailK = k - numRepeatK * elemsPerRepeat;
    uint32_t numProcessK = numRepeatK >> 1;
    uint32_t offsetPrevHalf = 0;
    uint32_t offsetPostHalf = numProcessK * elemsPerRepeat;
    uint32_t offset;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    while (numRepeatK > 1) {
        repeatParams.dstRepStride = (k - offsetPostHalf) * (sizeof(T) >> 5);
        repeatParams.src0RepStride = (k - offsetPrevHalf) * (sizeof(T) >> 5);
        repeatParams.src1RepStride = repeatParams.dstRepStride;
        for (uint32_t i = 0; i < numProcessK; ++i) {
            offset = i * elemsPerRepeat;
            Max(src[offsetPostHalf + offset], src[offsetPrevHalf + offset], src[offsetPostHalf + offset],
                elemsPerRepeat, m, repeatParams);
        }
        numRepeatK -= numProcessK;
        offsetPrevHalf = offsetPostHalf;
        numProcessK = numRepeatK >> 1;
        offsetPostHalf += numProcessK * elemsPerRepeat;
        PipeBarrier<PIPE_V>();
    }
    if (tailK > 0) {
        offsetPostHalf += elemsPerRepeat;
        repeatParams.dstRepStride = (k - offsetPrevHalf) * (sizeof(T) >> 5);
        repeatParams.src0RepStride = (k - offsetPostHalf) * (sizeof(T) >> 5);
        repeatParams.src1RepStride = repeatParams.dstRepStride;
        Max(src[offsetPrevHalf], src[offsetPostHalf], src[offsetPrevHalf], tailK, m, repeatParams);
        PipeBarrier<PIPE_V>();
    }

    // dst, src, repeat, mask, dstRepStride, srcBlkStride, srcRepStride
    auto lastTensor = src[offsetPrevHalf];
    BlockReduceMax<float>(lastTensor, lastTensor, m, 64, 8, 1, 8);
    PipeBarrier<PIPE_V>();
    BlockReduceMax<float>(dst, lastTensor, 1, 8 * m, 8, 1, 8);
    PipeBarrier<PIPE_V>();
    Brcb(dst, dst, (m + 7) / 8, {1, 8});
}

template <typename T>
__aicore__ inline void RowSumInplace(const LocalTensor<T> &dst, const LocalTensor<T> &src, uint32_t m, uint32_t k)
{
    // k must be aligned
    // numRepeatK > 1
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t numRepeatK = k / elemsPerRepeat;
    uint32_t tailK = k - numRepeatK * elemsPerRepeat;
    uint32_t numProcessK = numRepeatK >> 1;
    uint32_t offsetPrevHalf = 0;
    uint32_t offsetPostHalf = numProcessK * elemsPerRepeat;
    uint32_t offset;

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    while (numRepeatK > 1) {
        repeatParams.dstRepStride = (k - offsetPostHalf) * (sizeof(T) >> 5);
        repeatParams.src0RepStride = (k - offsetPrevHalf) * (sizeof(T) >> 5);
        repeatParams.src1RepStride = repeatParams.dstRepStride;
        for (uint32_t i = 0; i < numProcessK; ++i) {
            offset = i * elemsPerRepeat;
            Add(src[offsetPostHalf + offset], src[offsetPrevHalf + offset], src[offsetPostHalf + offset],
                elemsPerRepeat, m, repeatParams);
        }
        numRepeatK -= numProcessK;
        offsetPrevHalf = offsetPostHalf;
        numProcessK = numRepeatK >> 1;
        offsetPostHalf += numProcessK * elemsPerRepeat;
        PipeBarrier<PIPE_V>();
    }
    if (tailK > 0) {
        offsetPostHalf += elemsPerRepeat;
        repeatParams.dstRepStride = (k - offsetPrevHalf) * (sizeof(T) >> 5);
        repeatParams.src0RepStride = (k - offsetPostHalf) * (sizeof(T) >> 5);
        repeatParams.src1RepStride = repeatParams.dstRepStride;
        Add(src[offsetPrevHalf], src[offsetPostHalf], src[offsetPrevHalf], tailK, m, repeatParams);
        PipeBarrier<PIPE_V>();
    }

    // dst, src, repeat, mask, dstRepStride, srcBlkStride, srcRepStride
    auto lastTensor = src[offsetPrevHalf];
    BlockReduceSum<float>(lastTensor, lastTensor, m, 64, 8, 1, 8);
    PipeBarrier<PIPE_V>();
    BlockReduceSum<float>(dst, lastTensor, 1, 8 * m, 8, 1, 8);
    PipeBarrier<PIPE_V>();
    Brcb(dst, dst, (m + 7) / 8, {1, 8});
}

template <typename T>
__aicore__ inline void DivMC(const LocalTensor<T> dst, const LocalTensor<T> src0, const LocalTensor<T> src1,
                             uint32_t dim0, uint32_t dim1)
{
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t mainRepeatN = dim1 / elemsPerRepeat;
    uint32_t offset;
    if (mainRepeatN > dim0) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        for (uint32_t idxM = 0; idxM < dim0; ++idxM) {
            offset = idxM * dim1;
            Muls(dst[offset], src0[offset], 1.0f / src1(idxM << 3), dim1);
        }
    } else {
        // (m, n) <- (m, n) * (m, 8)
        uint32_t tailN = dim1 % elemsPerRepeat;
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = dim1 * sizeof(T) / 32;
        repeatParams.src0RepStride = repeatParams.dstRepStride;
        repeatParams.src1RepStride = 1;
        for (uint32_t idxN = 0; idxN < mainRepeatN; ++idxN) {
            offset = idxN * elemsPerRepeat;
            Div(dst[offset], src0[offset], src1, elemsPerRepeat, dim0, repeatParams);
        }
        if (tailN > 0) {
            offset = mainRepeatN * elemsPerRepeat;
            Div(dst[offset], src0[offset], src1, tailN, dim0, repeatParams);
        }
    }
}

template <typename T>
__aicore__ inline void MulMC(const LocalTensor<T> dst, const LocalTensor<T> src0, const LocalTensor<T> src1,
                             uint32_t dim0, uint32_t dim1)
{
    // (m, n) <- (m, n) * (m, 8)
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t mainRepeatN = dim1 / elemsPerRepeat;
    uint32_t tailN = dim1 % elemsPerRepeat;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstRepStride = dim1 * sizeof(T) / 32;
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1RepStride = 1;
    for (uint32_t idxN = 0; idxN < mainRepeatN; ++idxN) {
        Mul(dst[idxN * elemsPerRepeat], src0[idxN * elemsPerRepeat], src1, elemsPerRepeat, dim0, repeatParams);
    }
    if (tailN > 0) {
        Mul(dst[mainRepeatN * elemsPerRepeat], src0[mainRepeatN * elemsPerRepeat], src1, tailN, dim0, repeatParams);
    }
}

template <typename T>
__aicore__ inline void AddMV(const LocalTensor<T> dst, const LocalTensor<T> src0, const LocalTensor<T> src1,
                             uint32_t dim0, uint32_t dim1)
{
    // (m, n) <- (m, n) * (1, n)
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t mainRepeatN = dim1 / elemsPerRepeat;
    uint32_t tailN = dim1 % elemsPerRepeat;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = dim1 * sizeof(T) / 32;
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1RepStride = 0;
    uint32_t offsetN;
    for (uint32_t idxN = 0; idxN < mainRepeatN; ++idxN) {
        offsetN = idxN * elemsPerRepeat;
        Add(dst[offsetN], src0[offsetN], src1[offsetN], elemsPerRepeat, dim0, repeatParams);
    }
    if (tailN > 0) {
        offsetN = mainRepeatN * elemsPerRepeat;
        Add(dst[offsetN], src0[offsetN], src1[offsetN], tailN, dim0, repeatParams);
    }
}

template <typename T>
__aicore__ inline void MulMV(const LocalTensor<T> dst, const LocalTensor<T> src0, const LocalTensor<T> src1,
                             uint32_t dim0, uint32_t dim1)
{
    // (m, n) <- (m, n) * (1, n)
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t mainRepeatN = dim1 / elemsPerRepeat;
    uint32_t tailN = dim1 % elemsPerRepeat;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = dim1 * sizeof(T) / 32;
    repeatParams.src0RepStride = repeatParams.dstRepStride;
    repeatParams.src1RepStride = 0;
    uint32_t offsetN;
    for (uint32_t idxN = 0; idxN < mainRepeatN; ++idxN) {
        offsetN = idxN * elemsPerRepeat;
        Mul(dst[offsetN], src0[offsetN], src1[offsetN], elemsPerRepeat, dim0, repeatParams);
    }
    if (tailN > 0) {
        offsetN = mainRepeatN * elemsPerRepeat;
        Mul(dst[offsetN], src0[offsetN], src1[offsetN], tailN, dim0, repeatParams);
    }
}

template <typename T>
__aicore__ inline void OuterProduct(const LocalTensor<T> dst, const LocalTensor<T> src0, const LocalTensor<T> src1,
                                    uint32_t dim0, uint32_t dim1)
{
    // (m, 8) (1, n) -> (m, n)
    constexpr uint32_t elemsPerRepeat = 256 / sizeof(T);
    uint32_t mainRepeatN = dim1 / elemsPerRepeat;
    uint32_t tailN = dim1 % elemsPerRepeat;
    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = dim1 * sizeof(T) / 32;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = 0;
    for (uint32_t idxN = 0; idxN < mainRepeatN; ++idxN) {
        Mul(dst[idxN * elemsPerRepeat], src0, src1[idxN * elemsPerRepeat], elemsPerRepeat, dim0, repeatParams);
    }
    if (tailN > 0) {
        Mul(dst[mainRepeatN * elemsPerRepeat], src0, src1[mainRepeatN * elemsPerRepeat], tailN, dim0, repeatParams);
    }
}

__aicore__ inline void NotifyCube()
{
    if ASCEND_IS_AIC {
        return;
    }

    SyncAll();
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_VECTOR_CUBE_FLAG);
}

__aicore__ inline void NotifyVector()
{
    if ASCEND_IS_AIV {
        return;
    }

    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_CUBE_VECTOR_FLAG);
}

__aicore__ inline void WaitForVector()
{
    if ASCEND_IS_AIV {
        return;
    }

    CrossCoreWaitFlag(SYNC_VECTOR_CUBE_FLAG);
}

__aicore__ inline void WaitForCube()
{
    if ASCEND_IS_AIC {
        return;
    }

    CrossCoreWaitFlag(SYNC_CUBE_VECTOR_FLAG);
}

template <typename T, PrecisionType precisionType = PrecisionType::NONE>
class Mc2PreprocessKernel {
public:
    __aicore__ inline Mc2PreprocessKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2MsdTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();

    TQue<QuePosition::VECIN, 1> inQue_;
    TQue<QuePosition::VECOUT, 1> outQueA1A2_, outQueMax_, outQueSum_;
    TBuf<> aF32_;
    TBuf<> aAbs_;
    TBuf<> tmpBuf_;
    TBuf<> a12F16_;
    TBuf<> sumTmp_;

    GlobalTensor<T> xGlobal_;
    GlobalTensor<int8_t> workspaceXS8Global_;
    GlobalTensor<float> reduceMaxWorkspaceGm_;
    GlobalTensor<float> reduceSumWorkspaceGm_;

    TPipe *pipe;

    const Mc2WeightQuantBatchMatmulV2MsdTilingData *tiling_;

    uint32_t curBlockIdx_;

    uint32_t mSize_;
    uint32_t kSize_;
    uint32_t usedVecNum_;

    LocalTensor<float> xF32InUb0;
    LocalTensor<float> xF32InUb1;
    LocalTensor<half> aF16InUb;
    LocalTensor<float> tmpTensor;
};

template <typename T, PrecisionType precisionType>
__aicore__ inline void
Mc2PreprocessKernel<T, precisionType>::Init(GM_ADDR x, GM_ADDR workspace,
                                            const Mc2WeightQuantBatchMatmulV2MsdTilingData *tilingData, TPipe *tPipe)
{
    pipe = tPipe;
    tiling_ = tilingData;

    mSize_ = tiling_->mSize;
    kSize_ = tiling_->kSize;
    usedVecNum_ = tiling_->preProcessUsedVecNum;

    curBlockIdx_ = GetBlockIdx();

    // TBuf
    pipe->InitBuffer(aF32_, kSize_ * sizeof(float));
    pipe->InitBuffer(aAbs_, kSize_ * sizeof(float));
    // 64 * 4 64个fp32的数
    pipe->InitBuffer(tmpBuf_, 256);
    pipe->InitBuffer(a12F16_, kSize_ * sizeof(half));
    // Queue
    pipe->InitBuffer(inQue_, 1, kSize_ * sizeof(T));

    pipe->InitBuffer(outQueA1A2_, 1, kSize_ * sizeof(int8_t));
    pipe->InitBuffer(outQueMax_, 1, 256);
    pipe->InitBuffer(outQueSum_, 1, 256);
    pipe->InitBuffer(sumTmp_, 256);

    xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x));
    uint32_t sizeA1A2 = CeilAlign(static_cast<uint32_t>(2 * mSize_ * kSize_), static_cast<uint32_t>(512));
    workspaceXS8Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace), sizeA1A2);

    uint32_t sizeReduce = CeilAlign(static_cast<uint32_t>(8 * mSize_ * sizeof(float)), static_cast<uint32_t>(512));
    reduceMaxWorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workspace + sizeA1A2));
    reduceSumWorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workspace + sizeA1A2 + sizeReduce));
}

template <typename T, PrecisionType precisionType>
__aicore__ inline void Mc2PreprocessKernel<T, precisionType>::Process()
{
    if ASCEND_IS_AIC {
        return;
    }
    if (curBlockIdx_ >= usedVecNum_) {
        return;
    }
    uint32_t singleM = 1;
    uint32_t singleMLoop = 1;
    if (mSize_ > usedVecNum_) {
        singleM = (mSize_ + usedVecNum_ - 1) / usedVecNum_;
        uint32_t singleMTail = mSize_ % singleM;
        singleMLoop = singleM;
        if (curBlockIdx_ == usedVecNum_ - 1 && singleMTail != 0) {
            singleMLoop = singleMTail;
        }
    }

    uint32_t kSizeAligned = CeilAlign(kSize_, static_cast<uint32_t>(16));
    uint32_t baseM = 1;
    for (uint32_t offsetM = 0; offsetM < singleMLoop; offsetM += baseM) {
        auto xB16 = inQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams;
        DataCopyExtParams copyParams;
        copyParams.blockLen = kSize_ * sizeof(T);
        copyParams.blockCount = baseM;
        DataCopyPad(xB16, xGlobal_[(singleM * curBlockIdx_ + offsetM) * kSize_], copyParams, padParams);
        inQue_.EnQue(xB16);
        xB16 = inQue_.DeQue<T>();
        auto xF32 = aF32_.Get<float>();
        Cast(xF32, xB16, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        inQue_.FreeTensor(xB16);

        auto xAbs = aAbs_.Get<float>();
        Abs(xAbs, xF32, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        // (m, k) -> (m, 8)
        auto xMax = outQueMax_.AllocTensor<float>();
        auto xSum = outQueSum_.AllocTensor<float>();
        tmpTensor = tmpBuf_.Get<float>();

        // 实际测试大于4096时上面部分性能优于下部分
        if (kSizeAligned >= 4096) {
            // binary method
            RowMaxInplace(xMax, xAbs, baseM, kSizeAligned);
            PipeBarrier<PIPE_V>();
            DataCopy(xAbs, xF32, baseM * kSizeAligned);
            PRINT_DATA(xMax, baseM * 8, 1, 8, "xMax");
            if constexpr (precisionType != PrecisionType::HIGH_PRECISION) {
                RowSumInplace(xSum, xAbs, baseM, kSizeAligned);
            }
        } else {
            // for loop
            RowMaxFixTmpTensor(xMax, xAbs, tmpTensor, baseM, kSizeAligned);
            PipeBarrier<PIPE_V>();
            PRINT_DATA(xMax, baseM * 8, 1, 8, "xMax");
            if constexpr (precisionType != PrecisionType::HIGH_PRECISION) {
                RowSumFixTmpTensor(xSum, xF32, tmpTensor, baseM, kSizeAligned);
            }
        }
        PipeBarrier<PIPE_V>();
        PRINT_DATA(xSum, baseM * 8, 1, 8, "xSum");

        DivMC(xF32, xF32, xMax, baseM, kSizeAligned);
        PipeBarrier<PIPE_V>();
        Muls(xAbs, xF32, EXPAND_FACTOR_1, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Cast(xAbs, xAbs, RoundMode::CAST_ROUND, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        auto a12F16 = a12F16_.Get<half>();
        Cast(a12F16, xAbs, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        auto a1S8 = outQueA1A2_.AllocTensor<int8_t>();
        Cast(a1S8, a12F16, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams aOutParams;
        aOutParams.blockLen = kSize_;
        aOutParams.blockCount = baseM;
        aOutParams.dstStride = 0;
        outQueA1A2_.EnQue(a1S8);
        a1S8 = outQueA1A2_.DeQue<int8_t>();
        DataCopyPad(workspaceXS8Global_[(singleM * curBlockIdx_ + offsetM) * kSize_], a1S8, aOutParams);
        outQueA1A2_.FreeTensor(a1S8);

        Muls(xAbs, xAbs, static_cast<float>(1 / EXPAND_FACTOR_1), baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        if constexpr (precisionType == PrecisionType::HIGH_PRECISION) {
            RowSumInplace(xSum, xAbs, baseM, kSizeAligned);
            PipeBarrier<PIPE_V>();
            // RowSumInplace 会修改 xAbs, 需要恢复数据
            Muls(xAbs, xF32, EXPAND_FACTOR_1, baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
            Cast(xAbs, xAbs, RoundMode::CAST_ROUND, baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
            Muls(xAbs, xAbs, static_cast<float>(1 / EXPAND_FACTOR_1), baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
        }
        Sub(xAbs, xF32, xAbs, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Muls(xAbs, xAbs, EXPAND_FACTOR_2, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Cast(xAbs, xAbs, RoundMode::CAST_ROUND, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        Cast(a12F16, xAbs, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        auto a2S8 = outQueA1A2_.AllocTensor<int8_t>();
        Cast(a2S8, a12F16, RoundMode::CAST_NONE, baseM * kSizeAligned);
        PipeBarrier<PIPE_V>();
        outQueA1A2_.EnQue<int8_t>(a2S8);
        a2S8 = outQueA1A2_.DeQue<int8_t>();
        DataCopyPad(workspaceXS8Global_[mSize_ * kSize_ + (singleM * curBlockIdx_ + offsetM) * kSize_], a2S8,
                    aOutParams);
        outQueA1A2_.FreeTensor(a2S8);

        outQueMax_.EnQue(xMax);
        xMax = outQueMax_.DeQue<float>();
        DataCopy(reduceMaxWorkspaceGm_[curBlockIdx_ * singleM * 8 + offsetM * 8], xMax, baseM * 8);
        if constexpr (precisionType == PrecisionType::HIGH_PRECISION) {
            Muls(xAbs, xAbs, static_cast<float>(1 / EXPAND_FACTOR_2), baseM * kSizeAligned);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> sumTemp = sumTmp_.Get<float>();
            RowSumInplace(sumTemp, xAbs, baseM, kSizeAligned);
            PipeBarrier<PIPE_V>();
            Add(xSum, xSum, sumTemp, xSum.GetSize() / sizeof(float));
            PipeBarrier<PIPE_V>();
            Mul(xSum, xSum, xMax, xSum.GetSize() / sizeof(float));
        }
        outQueMax_.FreeTensor(xMax);
        outQueSum_.EnQue(xSum);
        xSum = outQueSum_.DeQue<float>();
        DataCopy(reduceSumWorkspaceGm_[curBlockIdx_ * singleM * 8 + offsetM * 8], xSum, baseM * 8);
        outQueSum_.FreeTensor(xSum);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType = PrecisionType::NONE>
class Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel {
public:
    __aicore__ inline Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                const Mc2WeightQuantBatchMatmulV2MsdTilingData *tilingData, TPipe *tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessCube(uint32_t cubeSingleCoreN, uint32_t curCubeSingleCoreN);
    __aicore__ inline void LoadMaxSumToUb(uint32_t m);
    __aicore__ inline void BL1PreLoad(TPipe *tPipe);
    __aicore__ inline void BL1PreLoadNd(TPipe *tPipe);
    __aicore__ inline void BL1PreLoadNz(TPipe *tPipe);
    __aicore__ inline void LoadAntiQuantOffsetScaleToUb(uint32_t offsetN, uint32_t n);

    using InputXType = MatmulType<TPosition::GM, CubeFormat::ND, int8_t, aTrans>;
    using InputWType = MatmulType<TPosition::GM, weightFormat, int8_t, bTrans>;
    using OutputYType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    using InputBiasType = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    MatmulImpl<InputXType, InputWType, OutputYType, InputBiasType, CFG_MDL> mmObj;

    TPipe *pipe_;
    const Mc2WeightQuantBatchMatmulV2MsdTilingData *tiling_;

    GlobalTensor<int8_t> workspaceXS8Global_;
    GlobalTensor<int32_t> workspaceCS32Global_;
    GlobalTensor<int8_t> wGlobal_;
    GlobalTensor<xType> antiQuantOffsetGlobal_;
    GlobalTensor<xType> antiQuantScaleGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<uint64_t> quantScaleGlobal_;
    GlobalTensor<yType> yGlobal_;

    GlobalTensor<float> reduceMaxWorkspaceGm_;
    GlobalTensor<float> reduceSumWorkspaceGm_;

    TQue<QuePosition::VECIN, 1> inQueMaxSum_, inQueAntiQuantOffset_, inQueAntiQuantScale_, inQueBias_, inQueC1_,
        inQueC2_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    TBuf<> bufFix_;
    TBuf<> bufOuterProductF32_;
    TBuf<> bufC1F32_;
    TBuf<> bufC2F32_;
    TBuf<TPosition::B1> bL1Tbuf_;

    LocalTensor<float> tnsMax_;
    LocalTensor<float> tnsSum_;
    LocalTensor<float> tnsAntiQuantOffsetF32_;
    LocalTensor<float> tnsAntiQuantOffsetMnF32_;
    LocalTensor<xType> tnsAntiQuantOffsetB16_;
    LocalTensor<float> tnsAntiQuantScaleF32_;
    LocalTensor<xType> tnsAntiQuantScaleB16_;
    LocalTensor<float> tnsBiasF32_;
    LocalTensor<int32_t> tnsC1S32_;
    LocalTensor<int32_t> tnsC2S32_;
    LocalTensor<float> tnsC1F32_;
    LocalTensor<float> tnsC2F32_;

    // 由公式 6k+8*N+22*m*n < 196352 中带入 N=1024, 其中 m*n 部分就为如下初始值
    uint32_t ubCalcShape_ = 7712;

    int32_t curBlockIdx_;

    uint32_t sizeReduce;
    uint32_t elemsReduce;
    uint32_t elemsC1C2;
};

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    precisionType>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR quantScale,
                         GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                         const Mc2WeightQuantBatchMatmulV2MsdTilingData *tilingData, TPipe *tPipe)
{
    tiling_ = tilingData;
    Mc2PreprocessKernel<xType, precisionType> op;
    curBlockIdx_ = GetBlockIdx();

    uint32_t m = tiling_->mSize;
    uint32_t k = tiling_->kSize;
    uint32_t n = tiling_->nSize;
    wGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight), k * n);
    op.Init(x, workspace, tilingData, tPipe);
    op.Process();
    BL1PreLoad(tPipe);
    tPipe->Reset();

    if ASCEND_IS_AIC {
        mmObj.SetSubBlockIdx(0);
        mmObj.Init(&tiling_->matmulTiling, tPipe);
    }

    NotifyCube();

    pipe_ = tPipe;

    uint64_t alignedOffset = 0;
    uint64_t alignedSize = CeilAlign(2 * m * k, static_cast<uint32_t>(512));
    workspaceXS8Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(workspace), alignedSize);

    alignedOffset += alignedSize;
    // 8 * m 为 reduce 大小
    alignedSize = CeilAlign(static_cast<uint64_t>(8 * m * sizeof(float)), static_cast<uint64_t>(512));
    reduceMaxWorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workspace + alignedOffset));

    alignedOffset += alignedSize;
    reduceSumWorkspaceGm_.SetGlobalBuffer((__gm__ float *)(workspace + alignedOffset));

    alignedOffset += alignedSize;
    alignedSize = tiling_->cubeBlockDimN *
                  CeilAlign(static_cast<uint64_t>(2 * m * tiling_->matmulTiling.singleCoreN * sizeof(int32_t)),
                            static_cast<uint64_t>(512));
    elemsC1C2 = alignedSize / tiling_->cubeBlockDimN / sizeof(float);
    workspaceCS32Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace + alignedOffset), alignedSize);

    antiQuantScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantScale), n);
    antiQuantOffsetGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ xType *>(antiquantOffset), n);
    if (tiling_->hasBias) {
        biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), n);
    }
    yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ yType *>(y), m * n);

    uint32_t antiQuantOffsetCalcShape = 1024;
    if (tiling_->matmulTiling.singleCoreN > antiQuantOffsetCalcShape) {
        antiQuantOffsetCalcShape = tiling_->matmulTiling.singleCoreN;
        // 186=192k-6k(reduceSum和reduceMax)，预留了最大的m可用空间
        // 20 为 offset+scale+bias 占用的空间，f16->f32, f16->f32, f32->f32
        // 22*m*n 为计算中各个的汇总，分别为 (IN Que) 4B 4B, (OUT Que) 2B 和 (TBuf) 4B 4B 4B
        // / 32 * 32 为向下取整
        ubCalcShape_ = (186 * 1024 - 20 * antiQuantOffsetCalcShape - 256) / 22 / 32 * 32;
    }

    // 初始化UB Tbuffer，3 为 offset、scale、bias
    pipe_->InitBuffer(bufFix_, 3 * antiQuantOffsetCalcShape * sizeof(float));
    tnsAntiQuantOffsetF32_ = bufFix_.Get<float>();
    tnsAntiQuantScaleF32_ =
        bufFix_.GetWithOffset<float>(antiQuantOffsetCalcShape, antiQuantOffsetCalcShape * sizeof(float));
    tnsBiasF32_ = bufFix_.GetWithOffset<float>(antiQuantOffsetCalcShape, 2 * antiQuantOffsetCalcShape * sizeof(float));

    pipe_->InitBuffer(bufOuterProductF32_, ubCalcShape_ * sizeof(float));
    tnsAntiQuantOffsetMnF32_ = bufOuterProductF32_.Get<float>();
    pipe_->InitBuffer(bufC1F32_, ubCalcShape_ * sizeof(float));
    tnsC1F32_ = bufC1F32_.Get<float>();
    pipe_->InitBuffer(bufC2F32_, ubCalcShape_ * sizeof(float));
    tnsC2F32_ = bufC2F32_.Get<float>();

    // 初始化UB QUEUE
    sizeReduce = CeilAlign(static_cast<uint32_t>(8 * m * sizeof(float)), static_cast<uint32_t>(512));
    elemsReduce = sizeReduce / sizeof(float);
    pipe_->InitBuffer(inQueMaxSum_, 1, sizeReduce * 2);
    pipe_->InitBuffer(inQueAntiQuantOffset_, 1, antiQuantOffsetCalcShape * sizeof(half));
    pipe_->InitBuffer(inQueAntiQuantScale_, 1, antiQuantOffsetCalcShape * sizeof(half));
    if (tiling_->hasBias) {
        pipe_->InitBuffer(inQueBias_, 1, antiQuantOffsetCalcShape * sizeof(float));
    }
    pipe_->InitBuffer(inQueC1_, 1, ubCalcShape_ * sizeof(float));
    pipe_->InitBuffer(inQueC2_, 1, ubCalcShape_ * sizeof(float));

    pipe_->InitBuffer(outQue_, 1, ubCalcShape_ * sizeof(half));
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                              hasAntiQuantOffset, quantType, weightFormat,
                                              precisionType>::BL1PreLoad(TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        return;
    }
    if (curBlockIdx_ > tiling_->cubeBlockDimN - 1) { // aic和aiv需要的数量不一定正好是1:2， 防止访问越界
        return;
    }
    if constexpr (IsSameType<wType, int4b_t>::value) {
        return;
    }

    if constexpr (weightFormat == CubeFormat::NZ) {
        BL1PreLoadNz(tPipe);
    } else {
        BL1PreLoadNd(tPipe);
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                              hasAntiQuantOffset, quantType, weightFormat,
                                              precisionType>::BL1PreLoadNd(TPipe *tPipe)
{
    uint64_t weightOffset;
    uint32_t nValue;
    uint32_t dValue;
    uint32_t dstNzC0Stride;
    uint32_t srcDValue;
    uint64_t bGlobalStep;
    uint32_t bL1KSize = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;
    bL1KSize = Min(static_cast<uint64_t>(bL1KSize), tiling_->kSize);
    uint64_t nBlockOffset = curBlockIdx_ * tiling_->matmulTiling.singleCoreN;
    uint32_t bL1NSize = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN;
    tPipe->InitBuffer(bL1Tbuf_, CeilAlign(bL1NSize, 32U) * bL1KSize);
    if (curBlockIdx_ + 1 == tiling_->cubeBlockDimN) {
        bL1NSize = nBlockOffset + bL1NSize >= tiling_->nSize ? tiling_->nSize - nBlockOffset : bL1NSize;
    }
    if constexpr (bTrans) {
        weightOffset = curBlockIdx_ * tiling_->matmulTiling.singleCoreN * tiling_->kSize;
        nValue = bL1NSize;
        dValue = bL1KSize;
        srcDValue = tiling_->kSize;
        bGlobalStep = bL1KSize;
    } else {
        weightOffset = curBlockIdx_ * tiling_->matmulTiling.singleCoreN;
        nValue = bL1KSize;
        dValue = bL1NSize;
        srcDValue = tiling_->nSize;
        bGlobalStep = bL1KSize * tiling_->nSize;
    }
    // 最小分形的大小，我们的分形一般有 (16, 16), (16, 32), (16, 8), 这个16U是指的基本分形的外轴16
    dstNzC0Stride = CeilAlign(nValue, 16U);

    Nd2NzParams nd2nzParams;
    CheckDataCopyNd2nzParams(nValue, dValue, srcDValue, dstNzC0Stride);
    SetDataCopyNd2nzParams(nd2nzParams, nValue, dValue, srcDValue, dstNzC0Stride);

    LocalTensor<int8_t> bL1Tensor = bL1Tbuf_.template Get<int8_t>();
    DataCopy(bL1Tensor, wGlobal_[weightOffset], nd2nzParams);
    uint64_t kOffset = 0;
    if constexpr (bTrans) {
        kOffset = bL1KSize;
    }
    for (uint32_t times = 1; times < tiling_->preloadTimes; times++) {
        if (kOffset >= tiling_->kSize) {
            return;
        }
        if constexpr (bTrans) {
            nd2nzParams.dValue = kOffset + bL1KSize > tiling_->kSize ? tiling_->kSize - kOffset : nd2nzParams.dValue;
        }
        DataCopy(bL1Tensor, wGlobal_[weightOffset + bGlobalStep * times], nd2nzParams);
        kOffset += bL1KSize;
    }
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                              hasAntiQuantOffset, quantType, weightFormat,
                                              precisionType>::BL1PreLoadNz(TPipe *tPipe)
{
    uint64_t weightOffset;
    uint32_t hValue;
    uint32_t wValue;
    uint32_t hSize;
    uint64_t bGlobalStep;

    uint32_t bL1KSize = tiling_->matmulTiling.baseK * tiling_->matmulTiling.stepKb;
    bL1KSize = Min(static_cast<uint64_t>(bL1KSize), tiling_->kSize);

    uint64_t nBlockOffset = curBlockIdx_ * tiling_->matmulTiling.singleCoreN;
    uint32_t bL1NSize = tiling_->matmulTiling.baseN * tiling_->matmulTiling.stepN;
    tPipe->InitBuffer(bL1Tbuf_, CeilAlign(bL1NSize, 32U) * bL1KSize);
    if (curBlockIdx_ + 1 == tiling_->cubeBlockDimN) {
        bL1NSize = nBlockOffset + bL1NSize >= tiling_->nSize ? tiling_->nSize - nBlockOffset : bL1NSize;
    }

    uint64_t w0 = 32 / sizeof(wType);
    if constexpr (bTrans) {
        weightOffset = nBlockOffset * w0;
        hValue = bL1NSize;
        wValue = bL1KSize;
        hSize = tiling_->nSize;
        bGlobalStep = (wValue / w0 * tiling_->nSize) * w0 + (wValue % w0);
    } else {
        weightOffset = (nBlockOffset / w0 * tiling_->kSize) * w0 + (nBlockOffset % w0);
        hValue = bL1KSize;
        wValue = bL1NSize;
        hSize = tiling_->kSize;
        bGlobalStep = bL1KSize * w0;
    }

    CheckDataCopyParams(wValue / w0, hValue);
    // tiliing里限制死了maxN=32000 maxK=13696 这里计算出来DataCopyParams的各参数不会超范围
    DataCopyParams param;
    param.blockCount = wValue / w0;
    param.blockLen = hValue; // NZ格式 w方向就是32Byte为单位
    param.srcStride = hSize - hValue;
    param.dstStride = 0;
    LocalTensor<int8_t> bL1Tensor = bL1Tbuf_.template Get<int8_t>();
    DataCopy(bL1Tensor, wGlobal_[weightOffset], param);
    for (uint32_t times = 1; times < tiling_->preloadTimes; times++) {
        DataCopy(bL1Tensor, wGlobal_[weightOffset + bGlobalStep * times], param);
    }
    PipeBarrier<PIPE_MTE2>(); // matmul开始的时候,preload可能还未完成，造成精度问题。实测不加同步会有随机精度问题
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<
    xType, wType, biasType, yType, aTrans, bTrans, antiQuantType, hasAntiQuantOffset, quantType, weightFormat,
    precisionType>::ProcessCube(uint32_t cubeSingleCoreN, uint32_t curCubeSingleCoreN)
{
    // (m, k)
    mmObj.SetTensorA(workspaceXS8Global_, aTrans);
    // (k, n)
    uint64_t idxWeight;
    if constexpr (weightFormat == CubeFormat::NZ) {
        uint64_t startOffsetN = curBlockIdx_ * cubeSingleCoreN;
        uint64_t w0 = 32 / sizeof(wType);
        idxWeight = bTrans ? startOffsetN * w0 :                                         // (n, k)
                             (startOffsetN / w0 * tiling_->kSize) * w0 + (startOffsetN % w0); // (k, n)

    } else {
        if (bTrans) {
            // (n, k)
            idxWeight = curBlockIdx_ * cubeSingleCoreN * tiling_->kSize;
        } else {
            // (k, n)
            idxWeight = curBlockIdx_ * cubeSingleCoreN;
        }
    }
    mmObj.SetTensorB(wGlobal_[idxWeight], bTrans);
    // 2 为 ORDER
    mmObj.SetTail(2 * tiling_->mSize, curCubeSingleCoreN);

    /// (m, n)
    WaitForVector();
    mmObj.IterateAll(workspaceCS32Global_[curBlockIdx_ * cubeSingleCoreN]);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                              hasAntiQuantOffset, quantType, weightFormat,
                                              precisionType>::LoadMaxSumToUb(uint32_t n)
{
    tnsMax_ = inQueMaxSum_.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams;
    DataCopyExtParams aMaxInParams;
    if constexpr (hasAntiQuantOffset) {
        // 有 offset 才需要使用 ReduceSum 结果
        aMaxInParams.blockLen = 2 * sizeReduce;
    } else {
        aMaxInParams.blockLen = sizeReduce;
    }
    aMaxInParams.blockCount = 1;
    DataCopyPad(tnsMax_, reduceMaxWorkspaceGm_, aMaxInParams, padParams);
    inQueMaxSum_.EnQue(tnsMax_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                              hasAntiQuantOffset, quantType, weightFormat,
                                              precisionType>::LoadAntiQuantOffsetScaleToUb(uint32_t offsetN, uint32_t n)
{
    DataCopyPadExtParams<xType> padParams;
    DataCopyExtParams copyParams;
    copyParams.blockLen = n * sizeof(xType);
    copyParams.blockCount = 1;
    uint32_t nAligned = CeilAlign(n, static_cast<uint32_t>(8));
    if constexpr (hasAntiQuantOffset) {
        tnsAntiQuantOffsetB16_ = inQueAntiQuantOffset_.AllocTensor<xType>();
        DataCopyPad(tnsAntiQuantOffsetB16_, antiQuantOffsetGlobal_[offsetN], copyParams, padParams);
        inQueAntiQuantOffset_.EnQue(tnsAntiQuantOffsetB16_);
        tnsAntiQuantOffsetB16_ = inQueAntiQuantOffset_.DeQue<xType>();
        Cast(tnsAntiQuantOffsetF32_, tnsAntiQuantOffsetB16_, RoundMode::CAST_NONE, nAligned);
        inQueAntiQuantOffset_.FreeTensor(tnsAntiQuantOffsetB16_);
    }
    tnsAntiQuantScaleB16_ = inQueAntiQuantScale_.AllocTensor<xType>();
    DataCopyPad(tnsAntiQuantScaleB16_, antiQuantScaleGlobal_[offsetN], copyParams, padParams);
    inQueAntiQuantScale_.EnQue(tnsAntiQuantScaleB16_);
    tnsAntiQuantScaleB16_ = inQueAntiQuantScale_.DeQue<xType>();
    Cast(tnsAntiQuantScaleF32_, tnsAntiQuantScaleB16_, RoundMode::CAST_NONE, nAligned);
    inQueAntiQuantScale_.FreeTensor(tnsAntiQuantScaleB16_);
}

template <typename xType, typename wType, typename biasType, typename yType, bool aTrans, bool bTrans,
          Mc2QuantType antiQuantType, bool hasAntiQuantOffset, Mc2QuantType quantType, CubeFormat weightFormat,
          PrecisionType precisionType>
__aicore__ inline void
Mc2WeightQuantBatchMatmulV2MsdMultiCoreKernel<xType, wType, biasType, yType, aTrans, bTrans, antiQuantType,
                                              hasAntiQuantOffset, quantType, weightFormat, precisionType>::Process()
{
    if constexpr (IsSameType<wType, int4b_t>::value) {
        return;
    }
    if constexpr (IsSameType<yType, int8_t>::value) {
        return;
    }
    uint32_t m = tiling_->mSize;
    uint32_t k = tiling_->kSize;
    uint32_t n = tiling_->nSize;
    uint32_t cubeSingleCoreN = tiling_->matmulTiling.singleCoreN;
    uint32_t cubeBlockDimN = tiling_->cubeBlockDimN;
    uint32_t cubeSingleCoreNTail = n - (cubeBlockDimN - 1) * cubeSingleCoreN;
    uint32_t curCubeSingleCoreN = cubeSingleCoreN;

    if ASCEND_IS_AIC {
        if (curBlockIdx_ >= cubeBlockDimN) {
            return;
        }
        if (curBlockIdx_ + 1 == cubeBlockDimN) {
            curCubeSingleCoreN = cubeSingleCoreNTail;
        }

        ProcessCube(cubeSingleCoreN, curCubeSingleCoreN);
        NotifyVector();
    } else {
        uint32_t cubeSingleCoreM = tiling_->matmulTiling.singleCoreM;

        uint32_t curCubeBlockIdx = curBlockIdx_ >> 1;
        if (curCubeBlockIdx >= cubeBlockDimN) {
            return;
        }
        if (curCubeBlockIdx + 1 == cubeBlockDimN) {
            curCubeSingleCoreN = cubeSingleCoreNTail;
        }

        uint32_t subBlockIdx = GetSubBlockIdx();
        LoadMaxSumToUb(m);
        LoadAntiQuantOffsetScaleToUb(curCubeBlockIdx * cubeSingleCoreN, curCubeSingleCoreN);

        tnsMax_ = inQueMaxSum_.DeQue<float>();
        if constexpr (hasAntiQuantOffset) {
            tnsSum_ = tnsMax_[elemsReduce];
        }
        uint32_t maxBaseM = ubCalcShape_ / curCubeSingleCoreN;
        uint32_t startM, endM, baseM, curBaseM;
        if (subBlockIdx == 0) {
            startM = 0;
            endM = cubeSingleCoreM / 2;
            baseM = maxBaseM > endM ? endM : maxBaseM;
        } else {
            startM = cubeSingleCoreM / 2;
            endM = cubeSingleCoreM - startM;
            baseM = maxBaseM > (endM - startM) ? (endM - startM) : maxBaseM;
        }

        // 向上取8的整数，8是因为fp32类型，ub上一个block刚好是8个元素
        uint32_t cubeSingleCoreNAligned = (curCubeSingleCoreN + 7) & (~7);
        uint32_t offsetGm;
        curBaseM = baseM;
        for (uint32_t offsetM = startM; offsetM < endM; offsetM += baseM) {
            curBaseM = offsetM + baseM >= endM ? endM - offsetM : curBaseM;

            offsetGm = curCubeBlockIdx * cubeSingleCoreN + offsetM * n;
            if constexpr (hasAntiQuantOffset) {
                OuterProduct(tnsAntiQuantOffsetMnF32_, tnsSum_[offsetM * 8], tnsAntiQuantOffsetF32_, curBaseM,
                             cubeSingleCoreNAligned);
            }

            if (offsetM == startM) {
                WaitForCube();
            }

            tnsC1S32_ = inQueC1_.AllocTensor<int32_t>();
            SHORT_MIX_LOG("offsetGm %d curCubeBlockIdx %d elemsC1C2 %d offsetM %d curCubeSingleCoreN %d", offsetGm,
                          curCubeBlockIdx, elemsC1C2, offsetM, curCubeSingleCoreN);
            DataCopyPad2D(tnsC1S32_, workspaceCS32Global_[offsetGm], curBaseM, curCubeSingleCoreN, n);
            inQueC1_.EnQue(tnsC1S32_);
            inQueC1_.DeQue<int32_t>();
            Cast(tnsC1F32_, tnsC1S32_, RoundMode::CAST_NONE, curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();
            inQueC1_.FreeTensor(tnsC1S32_);
            Muls(tnsC1F32_, tnsC1F32_, static_cast<float>(1 / EXPAND_FACTOR_1), curBaseM * cubeSingleCoreNAligned);

            tnsC2S32_ = inQueC2_.AllocTensor<int32_t>();
            DataCopyPad2D(tnsC2S32_, workspaceCS32Global_[m * n + offsetGm], curBaseM, curCubeSingleCoreN, n);
            inQueC2_.EnQue(tnsC2S32_);
            inQueC2_.DeQue<int32_t>();
            Cast(tnsC2F32_, tnsC2S32_, RoundMode::CAST_NONE, curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();
            inQueC2_.FreeTensor(tnsC2S32_);
            Muls(tnsC2F32_, tnsC2F32_, static_cast<float>(1 / EXPAND_FACTOR_2), curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            Add(tnsC1F32_, tnsC1F32_, tnsC2F32_, curBaseM * cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            // c * amax
            MulMC(tnsC1F32_, tnsC1F32_, tnsMax_[offsetM * 8], curBaseM, cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            if constexpr (hasAntiQuantOffset) {
                // c * amax + offset
                Add(tnsC1F32_, tnsAntiQuantOffsetMnF32_, tnsC1F32_, curBaseM * cubeSingleCoreNAligned);
                PipeBarrier<PIPE_V>();
            }

            // (c * amax + offset) * scale
            MulMV(tnsC1F32_, tnsC1F32_, tnsAntiQuantScaleF32_, curBaseM, cubeSingleCoreNAligned);
            PipeBarrier<PIPE_V>();

            if (tiling_->hasBias) {
                if (offsetM == startM) {
                    LocalTensor<biasType> tnsBias = inQueBias_.AllocTensor<biasType>();
                    DataCopyPad2D(tnsBias, biasGlobal_[curCubeBlockIdx * cubeSingleCoreN], 1, curCubeSingleCoreN, n);
                    inQueBias_.EnQue(tnsBias);
                    inQueBias_.DeQue<biasType>();
                    if constexpr (IsSameType<biasType, float>::value) {
                        tnsBiasF32_ = tnsBias;
                    } else {
                        Cast(tnsBiasF32_, tnsBias, RoundMode::CAST_NONE, curCubeSingleCoreN);
                        PipeBarrier<PIPE_V>();
                    }
                    inQueBias_.FreeTensor(tnsBias);
                }
                AddMV(tnsC1F32_, tnsC1F32_, tnsBiasF32_, curBaseM, cubeSingleCoreNAligned);
                PipeBarrier<PIPE_V>();
            }

            if constexpr (!IsSameType<yType, int8_t>::value) {
                LocalTensor<yType> tnsCB16 = outQue_.AllocTensor<yType>();
                Cast(tnsCB16, tnsC1F32_, RoundMode::CAST_ROUND, curBaseM * cubeSingleCoreNAligned);
                PipeBarrier<PIPE_V>();
                outQue_.EnQue(tnsCB16);
                tnsCB16 = outQue_.DeQue<yType>();
                DataCopyPad2D(yGlobal_[offsetM * n + curCubeBlockIdx * cubeSingleCoreN], tnsCB16, curBaseM,
                              curCubeSingleCoreN, cubeSingleCoreNAligned, n);
                outQue_.FreeTensor(tnsCB16);
            }
        }
    }
}
} // namespace Mc2WeightQuantBatchMatmulV2Msd

#endif // WEIGHT_QUANT_BATCH_MATMUL_V2_MSD_MULTICORE_H
