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
 * \file reduce_sum_cast_fp32.h
 * \brief 使用AIV 进行 reduce_sum, 累加过程Cast成fp32 kernel代码公共部分
 */

#ifndef REDUCE_SUM_CAST_FP32_H
#define REDUCE_SUM_CAST_FP32_H

#include "reduce_sum_utils.h"

namespace AiVReduceSumCastFp32Impl {

using namespace AscendC;

constexpr static uint32_t UB_BUFFER_NUM = 6;                 // 使用到的UB buffer 个数
constexpr static uint32_t MAX_PER_BLOCK_NUM = 1024UL * 15UL; // 经验最优上限, 限制UB每块最大可搬运的元素数
constexpr static uint32_t UB_ALIGN_BYTES = 32U;              // UB搬运需按32B对齐
constexpr uint32_t BUFFER_NUM = 2U;                          // 用于double buffer

template <typename DataType>
class ReduceSumForAlltoAll {
public:
    __aicore__ inline ReduceSumForAlltoAll(){};
    __aicore__ inline void Init(uint64_t outputSize, uint64_t stride, uint64_t rankDim, uint64_t aivNum,
                                GM_ADDR srcAddr, GM_ADDR dstAddr, TPipe *tPipe);
    __aicore__ inline void ExecuteReduceSum();

private:
    __aicore__ inline void InitParams(uint64_t outputSize, uint64_t stride, uint64_t rankDim, uint64_t aivNum);
    __aicore__ inline void InitBuffers(GM_ADDR srcAddr, GM_ADDR dstAddr, TPipe *tPipe);
    __aicore__ inline void ReadDataBlockReduceSum(uint64_t curOffset, uint64_t count);
    __aicore__ inline void CopyResultToOutput(uint64_t outOffsetGM, LocalTensor<float> &localResultTensor,
                                              uint64_t count, bool isTail);
    __aicore__ inline void ComputeTailAivId();

    TBuf<> sumBuf_;                 // 用于Reduce_sum 求和
    TBuf<> castBuf_;                // 用于把输入Cast成float32
    LocalTensor<float> sumTensor_;  // 累加用float32
    LocalTensor<float> castTensor_; // 转换Cast后的数据
    GlobalTensor<DataType> srcGm_;
    GlobalTensor<DataType> dstGm_;
    TQue<QuePosition::VECIN, 1> vecInQueue_;
    TQue<QuePosition::VECOUT, 1> vecOutQueue_;

    uint64_t sliceSize_{0};
    uint64_t strideSize_{0};
    uint64_t totalBlockNums_{0};
    uint64_t round_{0};
    uint64_t tailBlockNums_{0};
    uint64_t rankDim_{0};
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
 * @param outputSize 期望输出的数据量。与每次 All2All 的通信量相同，即 sliceM * N（单位：sizeof(dataType)）。
 * @param stride     求和数据块间的偏移量, 与 All2All通信的数据块间隔相同（单位：sizeof(dataType)）。
 *                   - stride = 0：表示相邻数据块地址连续；
 *                   - stride > 0：表示相邻数据块起始地址的偏移量为 stride。
 * @param rankDim    参与 All2All 的卡数（Rank 数量）。
 * @param aivNum     启用的 AIV 核数量。
 * @param srcAddr    全局内存中源数据起始地址（All2All 接收缓冲区）。
 * @param dstAddr    全局内存中目标输出起始地址（Reduce-Sum 结果写回位置）。
 * @param tPipe      Tensor Pipe 对象，用于申请和管理 UB 缓冲区。
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::Init(uint64_t outputSize, uint64_t stride, uint64_t rankDim,
                                                            uint64_t aivNum, GM_ADDR srcAddr, GM_ADDR dstAddr,
                                                            TPipe *tPipe)
{
    // 初始化计算参数 (分片大小、块数、核分配策略等)
    InitParams(outputSize, stride, rankDim, aivNum);

    // 初始化内存地址与 UB 缓冲区
    InitBuffers(srcAddr, dstAddr, tPipe);
}

/**
 * @brief 初始化 ReduceSum 所需的参数，包括分片大小、块数、核分配策略等。
 *
 * @param outputSize 期望输出的数据量（数据个数）
 * @param stride 期望求和的个个数据块间的偏移量。(数据个数)
 * @param rankDim 参与 All2All 的卡数（rank 数量）
 * @param aivNum  AIV 核数量
 *
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::InitParams(uint64_t outputSize, uint64_t stride,
                                                                  uint64_t rankDim, uint64_t aivNum)
{
    // 读取配置
    rankDim_ = rankDim; // 卡数
    aivNum_ = aivNum;   // AIV数量

    // 计算理论UB每块可搬运的最大容量（向下 32B 对齐）
    uint64_t maxPerBlockNum_ = FloorAlign(TOTAL_UB_SIZE / UB_BUFFER_NUM, UB_ALIGN_BYTES) / sizeof(DataType);

    // 限制UB每次搬运数据块大小。
    perBlockNum_ = MIN(MAX_PER_BLOCK_NUM, maxPerBlockNum_);

    // 分片大小与总块数
    sliceSize_ = outputSize;                             // 每张卡分片的数据大小，与输出大小一致
    strideSize_ = (stride == 0U) ? sliceSize_ : stride;  // 累加数据块间的数据量偏移，即卡间偏移
    totalBlockNums_ = CeilDiv(sliceSize_, perBlockNum_); // 1/rank 数据需要搬运的总块数

    // 尾块搬运大小
    tailBytes_ =
        BlockAlignMod(sliceSize_, perBlockNum_) * sizeof(DataType); // 即计算分卡后每片的最后一个搬运数据块的字节大小
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
 * @brief 初始化 GM 源/目标地址及 UB 内部缓冲区（sum buffer 与 double buffer 队列）。
 *
 * @param srcAddr 全局内存中源数据起始地址（All2All 接收缓冲区）
 * @param dstAddr 全局内存中目标输出起始地址（Reduce-Sum 结果写回位置）
 * @param tPipe Tensor Pipe 对象，用于申请和管理 UB 缓冲区
 *
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::InitBuffers(GM_ADDR srcAddr, GM_ADDR dstAddr, TPipe *tPipe)
{
    srcGm_.SetGlobalBuffer((__gm__ DataType *)srcAddr);
    dstGm_.SetGlobalBuffer((__gm__ DataType *)dstAddr);

    tPipe->InitBuffer(vecInQueue_, BUFFER_NUM, perBlockNum_ * sizeof(DataType));  // srcBuf_ -> UB 拷贝队列
    tPipe->InitBuffer(vecOutQueue_, BUFFER_NUM, perBlockNum_ * sizeof(DataType)); // UB -> dstBuf_ 拷贝队列
    tPipe->InitBuffer(sumBuf_, perBlockNum_ * sizeof(float));                     // Reduce-sum buffer
    tPipe->InitBuffer(castBuf_, perBlockNum_ * sizeof(float));                    // CastToFloat32 buffer
    sumTensor_ = sumBuf_.Get<float>();
    castTensor_ = castBuf_.Get<float>();
}

/**
 * @brief 从 GM 指定偏移读取一块数据到 UB，并将其累加到当前 sumTensor_ 中。
 *
 * @param curOffset 当前要读取的数据在 GM 中的偏移（元素单位）
 * @param count 本次搬运的元素个数（主块或尾块大小）
 *
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::ReadDataBlockReduceSum(uint64_t curOffset, uint64_t count)
{
    LocalTensor<DataType> inTensor = vecInQueue_.template AllocTensor<DataType>();

    // GM -> UB: 从 srcBuf_ 拷贝数据
    DataCopy(inTensor, srcGm_[curOffset], count);

    vecInQueue_.EnQue(inTensor);
    inTensor = vecInQueue_.DeQue<DataType>();

    // Cast 成 float32
    if constexpr (AscendC::IsSameType<DataType, float>::value) {
        DataCopy(castTensor_, inTensor, count); // 左手倒右手一波
    } else {
        Cast(castTensor_, inTensor, RoundMode::CAST_NONE, count); // Cast 成 float32
    }

    // 释放临时 buffer
    vecInQueue_.FreeTensor(inTensor);

    // 执行累加: sumTensor_ += castTensor_
    Add(sumTensor_, sumTensor_, castTensor_, count);
    PipeBarrier<PIPE_V>();
}

/**
 * @brief 执行完整的多卡 Reduce-Sum 流程：遍历分配块，累加各 rank 分片后写回 GM。
 *
 * @note 对每个数据块清零 sumTensor_，循环读取 rankDim_ 个分片累加，最后将结果写入 dstGm_。
 *       尾块由最后一个 AIV 的最后一轮搬运特殊处理。
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::ExecuteReduceSum()
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

        // 1. 清零累加 SumTensor
        Duplicate<float>(sumTensor_, static_cast<float>(0.0), perBlockNum_);
        PipeBarrier<PIPE_V>();

        // 2. 对每个 rank 的 slice 进行累加
        for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
            uint64_t srcSliceOffset = curRank * strideSize_; // 属于不同卡的数据块间的偏移量
            uint64_t curOffset = srcSliceOffset + curBlockOffset;
            ReadDataBlockReduceSum(curOffset, copyBlockNum); // 从 recvBuf_ 读取数据做reduce sum
        }

        // 3. 将结果写入 output (UB -> GM), 如果是非float数据类型需要先转换成目标数据类型
        CopyResultToOutput(curBlockOffset, sumTensor_, copyBlockNum, isTailBlock);
    }
}

/**
 * @brief 将计算得到的UB上的结果Tensor的数据复制到GM上的OutPutTensor，并进行类型转换（如果需要）。
 *
 * @param outputOffset 输出OutputTensor的GM偏移量，用于指定目标位置。
 * @param sourceTensor 本地UB上计算结果Tensor，包含计算完成的数据。
 * @param count 当前每次处理数据块的元素数量
 * @param isTail 是否为尾块搬运
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::CopyResultToOutput(uint64_t outOffsetGM,
                                                                          LocalTensor<float> &localResultTensor,
                                                                          uint64_t count, bool isTail)
{
    LocalTensor<DataType> outTensor = vecOutQueue_.AllocTensor<DataType>();

    if constexpr (AscendC::IsSameType<DataType, float>::value) {
        DataCopy(outTensor, localResultTensor, count); // 左手倒右手一波
    } else {
        Cast(outTensor, localResultTensor, RoundMode::CAST_RINT, count); // Cast成目标格式
    }

    vecOutQueue_.EnQue(outTensor);
    outTensor = vecOutQueue_.DeQue<DataType>();

    // 搬出时，尾块用dataCopyPad, 防止32B向上对齐越界OutPut
    if (isTail) {
        // 尾块处理：使用 Pad 拷贝防止越界
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = static_cast<uint32_t>(tailBytes_); // 单位为Byte
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        copyOutParams.rsv = 0;
        DataCopyPad(dstGm_[outOffsetGM], outTensor, copyOutParams);
    } else {
        // 主块处理：使用高性能的DataCopy拷贝
        DataCopy(dstGm_[outOffsetGM], outTensor, count);
    }

    vecOutQueue_.FreeTensor(outTensor);
}

/**
 * @brief 计算负责处理尾部数据的AIV核ID
 *
 * @note 当数据量较小时，此时计算仅仅只有一轮（round_ == 0），此时处理尾块的aiv并非最后一个，
 *       需根据当前数据块数量计算。
 */
template <typename DataType>
__aicore__ inline void ReduceSumForAlltoAll<DataType>::ComputeTailAivId()
{
    if (round_ == 0) {
        // 小数据量时，如果只有一轮搬运，负责尾块的aiv由此时计算的总块数决定
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        // 轮次大于一轮时，负责尾块的aiv必定是最后一个
        lastAivId_ = aivNum_ - 1;
    }
}
} // namespace AiVReduceSumCastFp32Impl
#endif // REDUCE_SUM_CAST_FP32_H
