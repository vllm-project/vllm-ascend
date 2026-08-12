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
 * \file gm_ub_gm_copy.h
 * \brief 实现 GM 到 UB 再到 GM 的数据拷贝流程 (GM -> UB -> GM)。
 */

#ifndef GM_UB_GM_COPY_H
#define GM_UB_GM_COPY_H

#include "reduce_sum_utils.h"

namespace GmUbGmCopyImpl {

using namespace AscendC;

constexpr static uint32_t UB_BUFFER_NUM = 2; // 使用到的UB buffer 个数, 当前为2， 即 double buffer dataQueue_
constexpr static uint32_t MAX_PER_BLOCK_NUM = 1024UL * 15UL; // 经验最优上限, 限制UB每块最大可搬运的元素数
constexpr static uint32_t UB_ALIGN_BYTES = 32U;              // UB搬运需按32B对齐
constexpr uint32_t BUFFER_NUM = 2U;                          // 用于double buffer

template <typename DataType>
class GmUbGmCopy {
public:
    __aicore__ inline GmUbGmCopy(){};
    __aicore__ inline void Init(uint64_t dataSize, uint64_t aivNum, GM_ADDR srcAddr, GM_ADDR dstAddr, TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams(uint64_t dataSize, uint64_t aivNum);
    __aicore__ inline void InitBuffers(GM_ADDR srcAddr, GM_ADDR dstAddr, TPipe *tPipe);
    __aicore__ inline void GmUbGmDataBlockCopy(uint64_t curOffset, uint64_t elemsPerBlock, bool isTail);
    __aicore__ inline void ComputeTailAivId();

    GlobalTensor<DataType> srcGm_;
    GlobalTensor<DataType> dstGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQueue_;

    uint64_t totalBlockNums_{0};
    uint64_t round_{0};
    uint64_t tailBlockNums_{0};
    uint64_t aivNum_{0};
    uint64_t aivId_{0};
    uint64_t assignedBlockNums_{0};
    uint64_t blockIdx_{0};
    uint64_t curAivOffset_{0};
    uint64_t perBlockNum_{0};
    uint64_t lastAivId_{0};
    uint64_t tailBytes_{0};
    uint64_t tailPerBlockNumAlign_{0};
};

/**
 * @brief 统一初始化入口：完成参数计算与 GM/UB Tensor 设置。
 *
 * @param dataSize   需要搬运的总的数据量（单位：sizeof(dataType)）。
 * @param aivNum     启用的 AIV 核数量。
 * @param srcAddr    全局内存中源数据起始地址（All2All 接收缓冲区）。
 * @param dstAddr    全局内存中目标输出起始地址（Reduce-Sum 结果写回位置）。
 * @param tPipe      Tensor Pipe 对象，用于申请和管理 UB 缓冲区。
 */
template <typename DataType>
__aicore__ inline void GmUbGmCopy<DataType>::Init(uint64_t dataSize, uint64_t aivNum, GM_ADDR srcAddr, GM_ADDR dstAddr,
                                                  TPipe *tPipe)
{
    // 初始化计算参数 (核分配策略等)
    InitParams(dataSize, aivNum);

    // 初始化内存地址与 UB 缓冲区
    InitBuffers(srcAddr, dstAddr, tPipe);
}

/**
 * @brief 初始化所需的参数，包括核分配策略等。
 *
 * @param dataSize 需要搬运的数据量（数据个数）
 * @param aivNum  AIV 核数量
 *
 */
template <typename DataType>
__aicore__ inline void GmUbGmCopy<DataType>::InitParams(uint64_t dataSize, uint64_t aivNum)
{
    // 读取配置
    aivNum_ = aivNum; // AIV数量

    // 计算理论UB每块可搬运的最大容量（向下 32B 对齐）
    uint64_t maxPerBlockNum_ = FloorAlign(TOTAL_UB_SIZE / UB_BUFFER_NUM, UB_ALIGN_BYTES) / sizeof(DataType);

    // 限制UB每次搬运数据块大小。
    perBlockNum_ = MIN(MAX_PER_BLOCK_NUM, maxPerBlockNum_);

    // 分核
    totalBlockNums_ = CeilDiv(dataSize, perBlockNum_); // 数据需要搬运的总块数

    // 尾块搬运大小
    tailBytes_ =
        BlockAlignMod(dataSize, perBlockNum_) * sizeof(DataType); // 即计算分卡后每片的最后一个搬运数据块的字节大小
    tailPerBlockNumAlign_ = CeilAlign(tailBytes_, UB_ALIGN_BYTES) /
                            sizeof(DataType); // 即计算分卡后每片的最后一个搬运数据块的大小, 向上32B对齐

    // 核分配策略
    round_ = totalBlockNums_ / aivNum_;         // 计算数据分核搬运需要的轮次数
    tailBlockNums_ = totalBlockNums_ % aivNum_; // 搬运的尾块数
    aivId_ = GetBlockIdx();                     // 获取当前核Id
    assignedBlockNums_ =
        (aivId_ < tailBlockNums_) ? (round_ + 1) : round_; // 当前核分配到的数据块数量，顺序分核，序号小的核多搬一轮
    blockIdx_ =
        aivId_ * round_ + (aivId_ < tailBlockNums_ ? aivId_ : tailBlockNums_); // 计算当前核分派到的首个数据块序列号
    curAivOffset_ = blockIdx_ * perBlockNum_; // 当前核偏移，块数 * 一块有多少数据，得到要搬第几个数据
    ComputeTailAivId();                       // 计算最后一个核的id
}

/**
 * @brief 初始化 GM 源/目标地址及 UB 内部缓冲区 (TQueBind 队列)。
 *
 * @param srcAddr 全局内存中源数据起始地址
 * @param dstAddr 全局内存中目标输出起始地址（结果写回位置）
 * @param tPipe Tensor Pipe 对象，用于申请和管理 UB 缓冲区
 *
 */
template <typename DataType>
__aicore__ inline void GmUbGmCopy<DataType>::InitBuffers(GM_ADDR srcAddr, GM_ADDR dstAddr, TPipe *tPipe)
{
    srcGm_.SetGlobalBuffer((__gm__ DataType *)srcAddr);
    dstGm_.SetGlobalBuffer((__gm__ DataType *)dstAddr);

    tPipe->InitBuffer(dataQueue_, BUFFER_NUM, perBlockNum_ * sizeof(DataType));
}

/**
 * @brief 从 GM 指定偏移读取一块数据到 UB，再搬回指定偏移的另一处 GM
 *
 * @param curOffset 当前要读取的数据在 GM 中的偏移（元素单位）
 * @param elemsPerBlock 本次搬运的元素个数（主块或尾块大小）
 * @param isTail 是否为尾块，若是则写回时使用 DataCopyPad 防止越界
 */
template <typename DataType>
__aicore__ inline void GmUbGmCopy<DataType>::GmUbGmDataBlockCopy(uint64_t curOffset, uint64_t elemsPerBlock,
                                                                 bool isTail)
{
    LocalTensor<DataType> tmpTensor = dataQueue_.AllocTensor<DataType>();

    // 从 GM 读到 UB
    DataCopy(tmpTensor, srcGm_[curOffset], elemsPerBlock);

    // 入队 (等待读完成)
    dataQueue_.EnQue(tmpTensor);

    // 出队 (阻塞直到读完成)
    tmpTensor = dataQueue_.DeQue<DataType>();

    // 从 UB 写到 GM
    if (isTail) {
        // 尾块处理：使用 Pad 拷贝防止越界
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        // blockLen 单位是 Byte，需传入实际有效字节数
        copyOutParams.blockLen = static_cast<uint32_t>(tailBytes_);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        copyOutParams.rsv = 0;

        DataCopyPad(dstGm_[curOffset], tmpTensor, copyOutParams);
    } else {
        // 主块处理：使用高性能的DataCopy拷贝
        DataCopy(dstGm_[curOffset], tmpTensor, elemsPerBlock);
    }

    // 4. 释放资源
    dataQueue_.FreeTensor<DataType>(tmpTensor);
}

/**
 * @brief 执行完整的数据搬运流程：遍历当前核分配到的所有数据块，通过 UB 中转将数据从 GM 源地址拷贝至 GM 目的地址。
 *
 * @note 尾块由最后一个 AIV 的最后一轮搬运特殊处理，并使用 DataCopyPad 防止越界。
 */
template <typename DataType>
__aicore__ inline void GmUbGmCopy<DataType>::Process()
{
    const uint64_t lastAivBlockIdx = assignedBlockNums_ - 1;
    const bool isLastAiv = (aivId_ == lastAivId_);

    // 遍历当前核分配到的数据块
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        // 判定是否为全局最后一个数据块 (Tail Block)
        const bool isTailBlock = isLastAiv && (curBlock == lastAivBlockIdx);
        // 如果是尾块，使用对齐后的尾块长度；否则使用标准块长度
        const uint64_t copyBlockNum = isTailBlock ? tailPerBlockNumAlign_ : perBlockNum_;
        // 计算当前块的偏移
        uint64_t curBlockOffset = curAivOffset_ + curBlock * perBlockNum_;

        // 执行单次搬运
        GmUbGmDataBlockCopy(curBlockOffset, copyBlockNum, isTailBlock);
    }
}

/**
 * @brief 计算负责处理尾部数据的AIV核ID
 *
 * @note 当数据量较小时，此时计算仅仅只有一轮（round_ == 0），此时处理尾块的aiv并非最后一个，
 *       需根据当前数据块数量计算。
 */
template <typename DataType>
__aicore__ inline void GmUbGmCopy<DataType>::ComputeTailAivId()
{
    if (round_ == 0) {
        // 小数据量时，如果只有一轮搬运，负责尾块的aiv由此时计算的总块数决定
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        // 轮次大于一轮时，负责尾块的aiv必定是最后一个
        lastAivId_ = aivNum_ - 1;
    }
}
} // namespace GmUbGmCopyImpl
#endif // GM_UB_GM_COPY_H
