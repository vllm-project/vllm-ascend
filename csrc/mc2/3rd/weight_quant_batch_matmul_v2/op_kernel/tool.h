/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file tool.h
 * \brief
 */
#ifndef TOOL_H
#define TOOL_H

#include <limits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using AscendC::AIC;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockNum;
using AscendC::GetUserWorkspace;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::InitOutput;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::ONE_REPEAT_BYTE_SIZE;
using AscendC::SetFlag;
using AscendC::TEventID;
using AscendC::ToFloat;
using AscendC::TPosition;
using AscendC::WaitFlag;
using matmul::MatmulType;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
using AscendC::VECTOR_REG_WIDTH;
#endif

#if defined(__CCE_KT_TEST__)
#include <sys/types.h>
#include <unistd.h>

#include <sstream>
#include <string>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

constexpr int DEBUG_T_WIDTH = 10;
constexpr int DEBUG_T_PRECISION = 6;
constexpr int DEBUG_HALF_WIDTH = 14;
constexpr int DEBUG_HALF_PRECISION = 4;

template <typename T>
std::string Mc2DoPrintData(const LocalTensor<T> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_T_WIDTH) << std::setprecision(DEBUG_T_PRECISION) << data[localCount * stride] << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

template <>
std::string Mc2DoPrintData(const LocalTensor<uint8_t> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << (int)data[localCount * stride] << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

template <>
std::string Mc2DoPrintData(const LocalTensor<int8_t> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << (int)data[localCount * stride] << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

template <>
std::string Mc2DoPrintData(const LocalTensor<half> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_HALF_PRECISION) << data[localCount * stride].ToFloat() << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

template <typename T>
std::string Mc2DoPrintData(const GlobalTensor<T> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_T_WIDTH) << std::setprecision(DEBUG_T_PRECISION) << data[localCount * stride] << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

std::string Mc2DoPrintData(const GlobalTensor<int8_t> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << (int)data[localCount * stride] << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

template <>
std::string Mc2DoPrintData(const GlobalTensor<half> &tensor, size_t count, size_t stride, size_t elementsPerRow,
                           const std::string &block_id, const std::string &core_type)
{
    auto data = tensor.GetPhyAddr();
    std::ostringstream oss;
    for (size_t localCount = 0; localCount < count; ++localCount) {
        if (localCount == 0) {
            oss << "[" << block_id << "][" << core_type << "] ";
        }
        oss << std::setw(DEBUG_HALF_PRECISION) << data[localCount * stride].ToFloat() << " ";
        if ((localCount % elementsPerRow == elementsPerRow - 1) && (localCount != count - 1)) {
            oss << std::endl << "[" << block_id << "][" << core_type << "] ";
        }
    }
    oss << std::endl;
    return oss.str();
}

#define PRINT_DATA(data, count, stride, elementsPerRow, format, ...)                                                   \
    do {                                                                                                               \
        uint64_t coreId = AscendC::GetBlockIdx();                                                                      \
        std::string core_type = "";                                                                                    \
        std::string block_id = "Block_";                                                                               \
        if (g_coreType == AscendC::AIC_TYPE) {                                                                         \
            core_type = "AIC_";                                                                                        \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                                  \
            core_type = "AIV_";                                                                                        \
            coreId = coreId / 2;                                                                                       \
        } else {                                                                                                       \
            core_type = "MIX_";                                                                                        \
        }                                                                                                              \
        core_type += std::to_string(AscendC::GetSubBlockIdx());                                                        \
        block_id += std::to_string(coreId);                                                                            \
        printf("[%s][%s][%s:%d][%s][%ld] " format "\n%s\n", block_id.c_str(), core_type.c_str(), FILENAME, __LINE__,   \
               __FUNCTION__, (long)getpid(), ##__VA_ARGS__,                                                            \
               Mc2DoPrintData(data, count, stride, elementsPerRow, block_id, core_type).c_str());                      \
    } while (0)

#define SHORT_MIX_LOG(format, ...)                                                                                     \
    do {                                                                                                               \
        uint64_t coreId = AscendC::GetBlockIdx();                                                                      \
        std::string core_type = "";                                                                                    \
        std::string block_id = "Block_";                                                                               \
        if (g_coreType == AscendC::AIC_TYPE) {                                                                         \
            core_type = "AIC_";                                                                                        \
        } else if (g_coreType == AscendC::AIV_TYPE) {                                                                  \
            core_type = "AIV_";                                                                                        \
            coreId = coreId / 2;                                                                                       \
        } else {                                                                                                       \
            core_type = "MIX_";                                                                                        \
        }                                                                                                              \
        core_type += std::to_string(AscendC::GetSubBlockIdx());                                                        \
        block_id += std::to_string(coreId);                                                                            \
        printf("[%s][%s][%s:%d][%s][%ld] " format "\n", block_id.c_str(), core_type.c_str(), FILENAME, __LINE__,       \
               __FUNCTION__, (long)getpid(), ##__VA_ARGS__);                                                           \
    } while (0)

#else

#define SHORT_MIX_LOG(format, ...)
#define PRINT_DATA(format, ...)

#endif
namespace Mc2WeightQuantBatchMatmulV2 {
static constexpr uint64_t SYNC_MODE0 = 0;
static constexpr uint64_t SYNC_MODE2 = 2;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t SYNC_AIV_ONLY_ALL_FLAG = 6;
static constexpr uint64_t SYNC_AIC_ONLY_ALL_FLAG = 7;
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t SYNC_AIC_FIX_AIV_VF_FLAG = 3;
static constexpr uint64_t SYNC_AIV_MTE3_AIC_FIX_FLAG = 4;

static constexpr uint64_t SYNC_AIV_ONLY_CONFIG = 1 | (SYNC_MODE0 << 4) | (SYNC_AIV_ONLY_ALL_FLAG << 8);
static constexpr uint64_t SYNC_AIC_ONLY_CONFIG = 1 | (SYNC_MODE0 << 4) | (SYNC_AIC_ONLY_ALL_FLAG << 8);
static constexpr uint64_t SYNC_AIV_AIC_CONFIG = 1 | (SYNC_MODE2 << 4) | (SYNC_AIV_AIC_FLAG << 8);
static constexpr uint64_t SYNC_AIC_AIV_CONFIG = 1 | (SYNC_MODE2 << 4) | (SYNC_AIC_AIV_FLAG << 8);

static constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;
static constexpr int32_t DOUBLE_BUFFER_NUM = 2;
static constexpr int32_t SINGLE_BUFFER_NUM = 1;

static constexpr int32_t FP32_MAX_MASK_SIZE = 64;
static constexpr int32_t FP32_MASK_BLK_NUM = 8;
static constexpr uint32_t FP16_MASK_SIZE = 128;
static constexpr uint32_t FP32_BLOCK_SIZE = 8;
static constexpr uint32_t FP16_BLOCK_SIZE = 16;
static constexpr uint32_t INT4_BLOCK_SIZE = 64;
static constexpr uint32_t INT8_BLOCK_SIZE = 32;

static constexpr uint32_t UINT64_DATA_BENCHMARK = 128;
static constexpr uint32_t FLOAT_DATA_BENCHMARK = 256;
static constexpr uint32_t HALF_DATA_BENCHMARK = 512;
static constexpr uint32_t INT8_DATA_BENCHMARK = 1024;
static constexpr uint32_t INT4_DATA_BENCHMARK = 2048;
static constexpr uint64_t FLAG_ID_MAX = 16;

static constexpr uint64_t FRAC_SIZE_HALF = 256;
// CUBE分型信息
static constexpr uint32_t INT8_FRAC_SIZE = 512;
static constexpr uint32_t INT8_ONE_BLK_SIZE = 32;

// LoadDataWithTranspose对于int8每次可处理两个分型
static constexpr uint32_t INT8_FRAC_NUM = 2;

// datacopy dstLocal位于C2PIPE2GM时，单位为128B;位于C2时，单位为64B
static constexpr uint32_t FIXP_BLK_SIZE = 128;
static constexpr uint32_t BT_BLK_SIZE = 64;

// vector指令一个repeat最多处理256B，包含8个Block，repeat_stride最大为8
static constexpr uint32_t VEC_REPEAT_MAX_STRIDE = 8;

static constexpr uint32_t L1_MAX_SIZE_910B = 512 * 1024;
static constexpr uint32_t L0A_MAX_SIZE_910B = 64 * 1024;
static constexpr uint32_t L0B_MAX_SIZE_910B = 64 * 1024;
static constexpr uint32_t L0C_MAX_SIZE_910B = 128 * 1024;
static constexpr uint32_t BIAS_TABLE_MAX_SIZE_910B = 1024;
static constexpr uint32_t FIXPIPE_TABLE_MAX_SIZE_910B = 2048;

static constexpr uint64_t MX_GROUPSIZE = 32;

static constexpr int32_t C0_SIZE_B8 = 32;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
static constexpr uint64_t VEC_MAX_ELEM_B16 = VECTOR_REG_WIDTH / sizeof(half);
#endif

template <typename T>
__aicore__ inline T CeilAlign(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

__aicore__ inline uint32_t CeilAlign(uint32_t a, uint32_t b)
{
    ASCENDC_ASSERT(a <= (std::numeric_limits<uint32_t>::max() - b),
                   { KERNEL_LOG(KERNEL_ERROR, "CeilAlign uint32 over limit."); });
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline T CeilDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T FloorDiv(T a, T b)
{
    ASCENDC_ASSERT(b != 0, { KERNEL_LOG(KERNEL_ERROR, "Division by zero error!"); });
    return a / b;
}

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t fullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (fullDim0 - dim0) * sizeof(T);
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    if (dim0 % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(dim0, static_cast<uint32_t>(32 / sizeof(T))) - dim0;
        padParams.paddingValue = 0;
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t blockCount,
                                     uint32_t blockLen, uint32_t dstInnerLength, uint32_t srcInnerLength)
{
#if defined(__CCE_KT_TEST__)
    ASCENDC_ASSERT(dstInnerLength >= blockLen, {
        KERNEL_LOG(KERNEL_ERROR, "dstInnerLength[%d] should be larger than blockLen[%d].", dstInnerLength, blockLen);
    });
#endif
    DataCopyExtParams params;
    params.blockCount = blockCount;
    params.blockLen = blockLen * sizeof(T);
    params.srcStride = (srcInnerLength - blockLen) * sizeof(T);
    params.dstStride = (dstInnerLength - blockLen) * sizeof(T) / ONE_BLK_SIZE;
    DataCopyPadExtParams<T> padParams;
    if (blockLen % (32 / sizeof(T)) != 0) {
        padParams.isPad = true;
        padParams.rightPadding = CeilAlign(blockLen, static_cast<uint32_t>(32 / sizeof(T))) - blockLen;
        padParams.paddingValue = 0;
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        // 4bit场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
#else
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
        padParams.rightPadding = padParams.rightPadding >> 1;
    }
#endif
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = 0;
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if constexpr (IsSameType<T, int4b_t>::value || IsSameType<T, fp4x2_e2m1_t>::value ||
                  IsSameType<T, fp4x2_e1m2_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
#else
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
#endif
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t dim1,
                                     uint32_t dim0, uint32_t srcFullDim0, uint32_t dstFullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = CeilDiv((srcFullDim0 - dim0) * sizeof(T), static_cast<uint64_t>(ONE_BLK_SIZE));
    params.dstStride = (dstFullDim0 - dim0) * sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        // int4场景下， 跳转的步长、数据长度等需要除2
        params.blockLen = params.blockLen >> 1;
        params.srcStride = params.srcStride >> 1;
        params.dstStride = params.dstStride >> 1;
    }
    SHORT_MIX_LOG("dim1 %d dim0 %d dstFullDim0 %d blockCount %d blockLen %d srcStride %d dstStride %d", dim1, dim0,
                  dstFullDim0, params.blockCount, params.blockLen, params.srcStride, params.dstStride);
    DataCopyPad(dst, src, params);
}

template <typename T>
__aicore__ inline void InitAtomicAddr(const GlobalTensor<T> dst, uint64_t initTotalSize, int32_t curBlockIdx)
{
    if ASCEND_IS_AIC {
        return;
    }
    static constexpr uint64_t addrAlignBlock = 512 / sizeof(T);
    uint64_t initBaseSize = CeilAlign(CeilDiv(initTotalSize, static_cast<uint64_t>(GetBlockNum() * 2)), addrAlignBlock);
    uint64_t initOffset = curBlockIdx * initBaseSize;
    if (initOffset < initTotalSize) {
        uint64_t initRealSize = initBaseSize;
        if (initOffset + initRealSize > initTotalSize) {
            initRealSize = initTotalSize - initOffset;
        }
        InitOutput<T>(dst[initOffset], initRealSize, (T)0.0);

        // 后续运算的mte2要等当前initOutput的v和mte3结束才行
        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>();
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        eventId = GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }
}

template <typename T>
constexpr int32_t GetBlockSize()
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return INT4_BLOCK_SIZE;
    }
    return INT8_BLOCK_SIZE;
}

template <HardEvent event>
class SyncProcessor {
public:
    __aicore__ inline SyncProcessor()
    {
    }
    TEventID eventIds_[DOUBLE_BUFFER_NUM];
    uint32_t doubleBufferNum_ = DOUBLE_BUFFER_NUM;
    uint64_t setTaskId_ = 0;
    uint64_t waitTaskId_ = 0;
    uint64_t taskId_ = 0;

    __aicore__ inline void Init(int32_t doubleBufferNum)
    {
        doubleBufferNum_ = doubleBufferNum;
        if (doubleBufferNum_ > DOUBLE_BUFFER_NUM) {
            doubleBufferNum_ = DOUBLE_BUFFER_NUM;
        }
        for (int32_t eventIdx = 0; eventIdx < doubleBufferNum_; eventIdx++) {
            eventIds_[eventIdx] = GetTPipePtr()->AllocEventID<event>();
        }
    };

    __aicore__ uint64_t GetBufferId()
    {
        uint64_t bufferId = taskId_ % doubleBufferNum_;
        taskId_++;
        return bufferId;
    }

    __aicore__ inline void SetSyncFlag()
    {
        TEventID eventIds[DOUBLE_BUFFER_NUM] = {eventIds_[0], eventIds_[1]};
        SetFlag<event>(eventIds[setTaskId_ & (doubleBufferNum_ - 1)]);
        setTaskId_++;
    };

    __aicore__ inline void WaitSyncFlag()
    {
        if (setTaskId_ < doubleBufferNum_) {
            return;
        }

        TEventID eventIds[DOUBLE_BUFFER_NUM] = {eventIds_[0], eventIds_[1]};
        WaitFlag<event>(eventIds[waitTaskId_ & (doubleBufferNum_ - 1)]);
        waitTaskId_++;
    };

    __aicore__ inline void Destory()
    {
        TEventID eventIds[DOUBLE_BUFFER_NUM] = {eventIds_[0], eventIds_[1]};
        for (; waitTaskId_ < setTaskId_; waitTaskId_++) {
            WaitFlag<event>(eventIds[waitTaskId_ & (doubleBufferNum_ - 1)]);
        }

        for (int32_t eventIdx = 0; eventIdx < doubleBufferNum_; eventIdx++) {
            GetTPipePtr()->ReleaseEventID<event>(eventIds[eventIdx]);
        }
        setTaskId_ = 0;
        waitTaskId_ = 0;
        taskId_ = 0;
    };
};

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

} // namespace Mc2WeightQuantBatchMatmulV2
#endif
