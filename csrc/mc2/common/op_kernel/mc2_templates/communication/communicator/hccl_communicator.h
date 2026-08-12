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
 * \file hccl_communicator.h
 * \brief
 */

#ifndef MC2_HCCL_COMMUNICATOR_H
#define MC2_HCCL_COMMUNICATOR_H

#include "../primitives/hccl_primitives.h"

namespace MC2KernelTemplate {
/**
 * AICSync:通信控制核，true代表cube核，false代表vector核
 * ServerType:通信控制方式，ccu/mte/aicpu等
 * ContextType:上下文数据类型
 * TilingDataType:tilingdata具体数据类型
 * Primitive:通信原语的通信实现
 * SendCnt:每轮发送的次数
 * RecvCnt:每轮等待发送完毕的次数
 */
template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
class HcclCommunicator {
public:
    // 通信器构造方法
    __aicore__ inline HcclCommunicator(TilingDataType *tiling) : tiling_(tiling){};
    // 初始化通信器参数
    __aicore__ inline void Init();
    // 获取上下文指针
    __aicore__ inline ContextType *GetContextPtr();
    // 提前准备通信任务，与Process方法配套使用，一次PrepareAll多次Process
    __aicore__ inline void PrepareAll(uint32_t taskCnt);
    // 执行一次通信任务，与PrepareAll方法配套使用
    __aicore__ inline void Process(uint32_t taskIndex);
    // 执行一次通信任务，与AddIndex方法配套使用
    __aicore__ inline void ProcessWithPrepare(uint32_t taskIndex);
    // 更新通信任务起始索引，与ProcessWithPrepare方法配套使用，先多次ProcessWithPrepare，后一次AddIndex
    __aicore__ inline void AddIndex(uint32_t taskCnt);
    // 提前准备通信任务，不带taskCnt的方法需要额外准备repeat参数，repeat次数与taskCnt次数一致
    __aicore__ inline void PrepareAll();
    // 按顺序repeat执行准备的通信任务
    __aicore__ inline void Process();
    // 释放通信器资源
    __aicore__ inline void End();

private:
    enum Communicationtype {
        COMMUNICATION_WAIT_ONE,
        COMMUNICATION_SEND_ONE
    };
    TilingDataType *tiling_;
    AscendC::Hccl<ServerType> hccl_;
    ContextType context_;
    Primitive<AscendC::Hccl<ServerType>> primitive_;
    uint32_t startIndex_ = 0;
    uint32_t endIndex_ = 0;
    bool notifyFlag_ = false;
    Communicationtype communicationType_ = COMMUNICATION_WAIT_ONE;
    static constexpr uint8_t MAX_HCCL_HANDLE_ = 63; // hccl只支持最多63个任务并行
    AscendC::HcclHandle hTasks_[MAX_HCCL_HANDLE_];
    bool taskSuccess_[MAX_HCCL_HANDLE_];
};

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Init()
{
    notifyFlag_ = false;
    if ASCEND_IS_AIV {
        if (!AICSync && AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true;
        }
    } else if ASCEND_IS_AIC {
        if (AICSync && AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true;
        }
    }

    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tiling_->mc2InitTiling));
    hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
    if constexpr (SendCnt == 1U && RecvCnt == 0U) {
        communicationType_ = Communicationtype::COMMUNICATION_SEND_ONE;
    } else if constexpr (SendCnt == 0U && RecvCnt == 1U) {
        communicationType_ = Communicationtype::COMMUNICATION_WAIT_ONE;
    }

    for (uint8_t i = 0; i < MAX_HCCL_HANDLE_; ++i) {
        taskSuccess_[i] = true;
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline ContextType *
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::GetContextPtr()
{
    return &context_;
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::PrepareAll(
    uint32_t taskCnt)
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    for (uint32_t i = 0; i < taskCnt; i++) {
        hTasks_[endIndex_ + i] = primitive_.Prepare(&hccl_, &context_, i);
    }
    // task队列索引更新
    AddIndex(taskCnt);
    // 如果是先通后算就全量启动通信
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        for (uint32_t i = 0; i < taskCnt; i++) {
            hccl_.Commit(hTasks_[startIndex_ + i]);
            taskSuccess_[startIndex_ + i] = false;
        }
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Process(
    uint32_t taskIndex)
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        hccl_.Wait(hTasks_[startIndex_ + taskIndex]);
        taskSuccess_[startIndex_ + taskIndex] = true;
    } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        hccl_.Commit(hTasks_[startIndex_ + taskIndex]);
        taskSuccess_[startIndex_ + taskIndex] = false;
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::ProcessWithPrepare(
    uint32_t taskIndex)
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        AscendC::HcclHandle handleId = primitive_.SyncSend(&hccl_, &context_, taskIndex);
        hccl_.Wait(handleId);
    } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        hTasks_[endIndex_ + taskIndex] = primitive_.SyncSend(&hccl_, &context_, taskIndex);
        taskSuccess_[endIndex_ + taskIndex] = false;
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::AddIndex(
    uint32_t taskCnt)
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    // 更新全局变量
    startIndex_ = endIndex_;
    endIndex_ += taskCnt;
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::PrepareAll()
{
    // 只有通信核参与通信
    if (!notifyFlag_) {
        return;
    }
    hTasks_[endIndex_] = primitive_.Prepare(&hccl_, &context_, 0);
    for (uint32_t i = 1; i < context_.repeat; ++i) {
        hTasks_[endIndex_ + i] = hTasks_[endIndex_];
    }
    // task队列索引更新
    AddIndex(context_.repeat);
    // 如果是先通后算就全量启动通信
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        for (uint32_t i = 0; i < context_.repeat; ++i) {
            hccl_.Commit(hTasks_[startIndex_ + i]);
            taskSuccess_[startIndex_ + i] = false;
        }
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Process()
{
    // 执行repeat次通信子任务中的一个任务
    Process(0);
    // task队列索引更新
    startIndex_ += 1;
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::End()
{
    // 全量等待通信结果
    if (notifyFlag_) {
        for (uint32_t i = 0; i < endIndex_; ++i) {
            if (!taskSuccess_[i]) {
                hccl_.Wait(hTasks_[i]);
                taskSuccess_[i] = true;
            }
        }
    }

    if (notifyFlag_) {
        hccl_.Finalize();
    }
}

#ifndef DEFINE_MC2_HCCL_FOR_COMMUNICATION
#define DEFINE_MC2_HCCL_FOR_COMMUNICATION(AICSync, ServerType, HcclContextType, TilingDataType, Primitive,             \
                                          sendCntPerTask, recvCntPerTask, CommunicationType)                           \
    using CommunicationType =                                                                                          \
        MC2KernelTemplate::HcclCommunicator<AICSync, ServerType, HcclContextType, TilingDataType, Primitive,           \
                                            sendCntPerTask, recvCntPerTask>
#endif
}; // namespace MC2KernelTemplate

#endif
