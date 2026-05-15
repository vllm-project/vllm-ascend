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
 * \file buffer.h
 * \brief同步管理
 */
#ifndef BUFFER_H
#define BUFFER_H
#include<type_traits>
#include"lib/matmul_intf.h"
#include"kernel_event.h"
#include"kernel_common.h"
#include"kernel_tpipe.h"
using namespace AscendC;
namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 16)

// 核间同步中，AIC(flagId 0-10)对应AIV0(flagId 0-10)，对应AIV1(flagId 16-26)
#define AIV0_AIV1_OFFSET 16

enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
    UB = 4,
    GM = 5,
};

enum class SyncType {
    NO_SYNC,
    INNER_CORE_SYNC,
    CROSS_CORE_SYNC_FORWARD,
    CROSS_CORE_SYNC_BOTH,
};

constexpr uint32_t INVALID_CROSS_CORE_EVENT_ID = 16;
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;

template<BufferType Type>
struct BufferInfo{
    // Cons 消费者，Prod 生产者
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus() {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE2_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::M_FIX;
        }
    }

    __aicore__ const static constexpr HardEvent ProdWaitConsStatus() {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE1_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::FIX_M;
        }
    }

    __aicore__ const static constexpr TPosition GetTPosition() {
        if constexpr (Type == BufferType::L1) {
            return TPosition::A1;
        } else if constexpr (Type == BufferType::L0A) {
            return TPosition::A2;
        } else if constexpr (Type == BufferType::L0B) {
            return TPosition::B2;
        } else if constexpr (Type == BufferType::L0C) {
            return TPosition::CO1;
        } else if constexpr (Type == BufferType::UB) {
            return TPosition::VECIN;
        } else if constexpr (Type == BufferType::GM) {
            return TPosition::GM;
        }
    }

    static constexpr HardEvent EventP2C = ConsWaitProdStatus(); // 生产者到消费者方向的HardEvent：消费者等生产者提供/生产者通知消费者已生成
    static constexpr HardEvent EventC2P = ProdWaitConsStatus(); // 消费者到生产者方向的HardEvent：生产者等消费者消耗/消费者通知生产者已消耗’
    static constexpr TPosition Position = GetTPosition();
};

// buffer绑定生产者、消费者关系
// L1 buffer的生产者为MTE2或者MTE3，消费者为MTE1
// L0A buffer的生产者为MTE1，消费者为M
// L0B buffer的生产者为MTE1，消费者为M
// L0C buffer的生产者为M，消费者为FIX
template<BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Buffer {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;

    template <typename T>
    using TargetTensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<T>, LocalTensor<T>>;
public:
    __aicore__ inline Buffer() {}
    __aicore__ inline Buffer(TensorType tensor, uint32_t size) {
        tensor_ = tensor;
        size_ = size;
        if constexpr (syncType == SyncType::CROSS_CORE_SYNC_FORWARD) {
            id0_ = MAKE_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
            id0_ = MAKE_ID;
            id1_ = MAKE_ID;
        } else {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        }
    }

    __aicore__ inline void Init() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
                SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    __aicore__ inline void UnInit() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 确保只能被调用一次
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Wait() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    WaitFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 消费者等待生产者完成生产
                } else {
                    WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 生产者等待消费者完成消费
                }
            }
        }
    }

    template<HardEvent EventType>
    __aicore__ inline void Set() {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    SetFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 生产者通知消费者已完成生产
                } else {
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 消费者通知生产者已完成消费
                }
            }
        }
    }

    __aicore__ inline void WaitCrossCore() {
        if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id0_);
            }
        } else if constexpr (bufferType == BufferType::L1) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                }
            }
        }
    }

    __aicore__ inline void SetCrossCore() {
        if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id1_);
            }
        } else if constexpr (bufferType == BufferType::L1) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_);
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_ + AIV0_AIV1_OFFSET);
                }
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
            }
        }
    }

    template<typename T>
    __aicore__ inline TargetTensorType<T> GetTensor() {
        return tensor_.template ReinterpretCast<T>();
    }

    template<typename T>
    __aicore__ inline TargetTensorType<T> GetTensor(uint64_t startindex) {
        TargetTensorType<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    TensorType tensor_;
    uint32_t size_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
    uint32_t id0_;      // 用作正向同步：生产者通知消费者，或者消费者等待生产者；
    uint32_t id1_;      // 用作反向同步：消费者通知生产者，或者生产者等待消费者；
};
}
#endif