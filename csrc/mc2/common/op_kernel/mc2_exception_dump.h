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
 * \file mc2_exception_dump.h
 * \brief ExceptionDump 通用引擎。
 *        Header / DumpParams / 常量定义在 namespace 级别，kernel 与 host 共享。
 *        ExceptionDump<Policy> 模板类提供 kernel 侧 Init/UpdateStage/Dump 实现。
 */
#ifndef MC2_EXCEPTION_DUMP_H
#define MC2_EXCEPTION_DUMP_H

namespace MC2ExceptionDump {

enum class OpType : int32_t {
    OP_TYPE_MEGA_MOE = 0, // mega moe算子OpType值
    OP_TYPE_END = 1,      // 结束标志
};

static constexpr size_t MAX_DUMP_ENTRIES = 1024;
static constexpr size_t MAX_DUMP_TILING_SIZE = 2048;
static constexpr int32_t HEADER_MAGIC = 0x5A5A5A5A;

// Device 内存布局固定区域大小（均为 64 字节对齐，满足 cacheline 要求）。
// Host 侧 dump 时按实际大小搬运并密排，不包含 padding。
static constexpr size_t DUMP_HEADER_REGION_SIZE = 512;
static constexpr size_t DUMP_TILING_REGION_SIZE = 2048;
static constexpr size_t DUMP_BLOCK_STAGE_REGION_SIZE = 20480;

struct alignas(8) Header {
    int32_t headerMagic;     // 0x5A5A5A5A，校验 dump 有效性
    int32_t majorVersion;    // CANN 主版本
    int32_t minorVersion;    // CANN 次版本
    int32_t patchVersion;    // CANN 补丁版本
    int32_t opType;          // 算子枚举值（OpType）
    int32_t stageNum;        // 阶段总数 = StageEnumT::END
    uint32_t tilingDataSize; // sizeof(TilingDataT)，供 host 计算 BlockStage 偏移
    uint32_t numBlocks;      // GetBlockNum() * GetSubBlockNum()，供 host 计算 BlockStage/DumpParams 偏移
    uint64_t dumpCount;      // 本次执行 Dump 调用次数（Init 时清零，每次 Dump 自增）
    uint64_t execTimes[static_cast<int32_t>(OpType::OP_TYPE_END)]; // 通信域执行次数
};

// Dump 参数：host 侧按此结构解析并执行实际数据搬运
//   dumpAddr    -- 起始 GM 地址
//   blockCount  -- block 数量
//   blockLen    -- 每 block 的字节数
//   srcStride   -- 相邻 block 起点间的字节数（含 blockLen 自身）
struct alignas(8) DumpParams {
    uintptr_t dumpAddr;
    size_t blockCount;
    size_t blockLen;
    size_t srcStride;
};

#if defined(__CCE_AICORE__)
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

template <typename Policy>
class ExceptionDump {
public:
    using TilingDataT = typename Policy::TilingDataT;
    using StageEnumT = typename Policy::StageEnumT;

    // BlockStage 依赖模板参数 StageEnumT::END，保留在模板类内部。
    // Host 侧通过 Header.stageNum 运行时计算 sizeof(BlockStage) = 8 + stageNum * 8。
    struct alignas(8) BlockStage {
        uint64_t currentStage;                                       // 算子当前执行阶段
        uint64_t stageCycle[static_cast<uint32_t>(StageEnumT::END)]; // 算子各阶段时钟周期
    };

    __aicore__ inline ExceptionDump() = default;
    __aicore__ inline ExceptionDump(GM_ADDR baseAddr, GM_ADDR tilingAddr)
    {
        Init(baseAddr, tilingAddr);
    }

    __aicore__ inline void Init(GM_ADDR baseAddr, GM_ADDR tilingAddr)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        aivId_ = GetBlockIdx();
        header_ = reinterpret_cast<__gm__ Header *>(baseAddr);
        blockStage_ = reinterpret_cast<__gm__ BlockStage *>(baseAddr + DUMP_HEADER_REGION_SIZE +
                                                            DUMP_TILING_REGION_SIZE + aivId_ * sizeof(BlockStage));

        dumpParams_ = reinterpret_cast<__gm__ DumpParams *>(baseAddr + DUMP_HEADER_REGION_SIZE +
                                                            DUMP_TILING_REGION_SIZE + DUMP_BLOCK_STAGE_REGION_SIZE);
        dumpParamsEnd_ = dumpParams_ + MAX_DUMP_ENTRIES;
        blockStageGlobal_.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(blockStage_));

        if (aivId_ == 0) {
            DumpTilingData(baseAddr + DUMP_HEADER_REGION_SIZE, tilingAddr, sizeof(TilingDataT));
            header_->headerMagic = HEADER_MAGIC;
            header_->majorVersion = ASC_DEVKIT_MAJOR;
            header_->minorVersion = ASC_DEVKIT_MINOR;
            header_->patchVersion = ASC_DEVKIT_PATCH;
            header_->opType = static_cast<int32_t>(Policy::OP_TYPE);
            header_->stageNum = static_cast<uint32_t>(StageEnumT::END);
            header_->tilingDataSize = static_cast<uint32_t>(sizeof(TilingDataT));
            header_->numBlocks = GetBlockNum() * GetSubBlockNum();
            header_->dumpCount = 0;
            header_->execTimes[static_cast<size_t>(Policy::OP_TYPE)]++;
        }
    }

    __aicore__ inline void UpdateStage(typename Policy::StageEnumT stage)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (unlikely(blockStage_ == nullptr)) {
            return;
        }
        uint32_t stageId = static_cast<uint32_t>(stage);
        if (likely(stageId < static_cast<uint32_t>(StageEnumT::END))) {
            blockStage_->currentStage = stageId;
            blockStage_->stageCycle[stageId] = GetSystemCycle();
            DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(blockStageGlobal_);
        }
    }

    __aicore__ inline void Dump(GM_ADDR dumpAddr, size_t size)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (aivId_ != 0) {
            return;
        }
        if (unlikely(dumpParams_ == nullptr || dumpParams_ >= dumpParamsEnd_)) {
            return;
        }
        dumpParams_->dumpAddr = reinterpret_cast<uintptr_t>(dumpAddr);
        dumpParams_->blockCount = 1;
        dumpParams_->blockLen = size;
        dumpParams_->srcStride = 0;
        GlobalTensor<uint8_t> dumpParamsGlobal;
        dumpParamsGlobal.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(dumpParams_));
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dumpParamsGlobal);
        dumpParams_++;
        header_->dumpCount++;
    }

    __aicore__ inline void Dump(GM_ADDR dumpAddr, size_t blockCount, size_t blockLen, size_t srcStride)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if (aivId_ != 0) {
            return;
        }
        if (unlikely(dumpParams_ == nullptr || dumpParams_ >= dumpParamsEnd_)) {
            return;
        }
        dumpParams_->dumpAddr = reinterpret_cast<uintptr_t>(dumpAddr);
        dumpParams_->blockCount = blockCount;
        dumpParams_->blockLen = blockLen;
        dumpParams_->srcStride = srcStride;
        GlobalTensor<uint8_t> dumpParamsGlobal;
        dumpParamsGlobal.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(dumpParams_));
        DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dumpParamsGlobal);
        dumpParams_++;
        header_->dumpCount++;
    }

private:
    __aicore__ inline void DumpTilingData(GM_ADDR dumpAddr, GM_ADDR tilingAddr, size_t tilingSize)
    {
        if (tilingAddr == nullptr || tilingSize == 0) {
            return;
        }
        size_t copySize = tilingSize;
        if (copySize > MAX_DUMP_TILING_SIZE) {
            copySize = MAX_DUMP_TILING_SIZE;
        }
        LocalTensor<uint8_t> tilingLocal{TPosition::LCM, 0, static_cast<uint32_t>(copySize)};
        GlobalTensor<uint8_t> dumpTilingGlobal;
        GlobalTensor<uint8_t> tilingGlobal;
        dumpTilingGlobal.SetGlobalBuffer(dumpAddr, copySize);
        tilingGlobal.SetGlobalBuffer(tilingAddr, copySize);
        DataCopyPad(tilingLocal, tilingGlobal, {1, static_cast<uint16_t>(copySize), 0, 0, 0}, {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopyPad(dumpTilingGlobal, tilingLocal, {1, static_cast<uint16_t>(copySize), 0, 0, 0});
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    uint32_t aivId_{0};
    __gm__ Header *header_{nullptr};
    __gm__ BlockStage *blockStage_{nullptr};
    __gm__ DumpParams *dumpParams_{nullptr};
    __gm__ DumpParams *dumpParamsEnd_{nullptr};
    GlobalTensor<uint8_t> blockStageGlobal_;
};
#endif
} // namespace MC2ExceptionDump
#endif // MC2_EXCEPTION_DUMP_H
