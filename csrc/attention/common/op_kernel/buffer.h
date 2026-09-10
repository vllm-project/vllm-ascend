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
#include <type_traits>
#include "lib/matmul_intf.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;
namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 11)

// 核间同步中，AIC(flagId 0-10)对应AIV0(flagId 0-10)，对应AIV1(flagId 16-26)
#define AIV0_AIV1_OFFSET 16

enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
    UB = 4,
    GM = 5,
    C2 = 6,
};

enum class SyncType {
    NO_SYNC,
    INNER_CORE_SYNC,
    CROSS_CORE_SYNC_FORWARD,
    CROSS_CORE_SYNC_BOTH,
    CROSS_CORE_SYNC_BACKWARD,
};

enum class SyncMode {
    SET_WAIT_FLAG,
    LOCK_UNLOCK,
};

enum class IdSource {
    INTERNAL, // ID由接口内部分配
    EXTERNAL, // ID由接口外部传入（用户指定）
};

constexpr uint32_t INVALID_CROSS_CORE_EVENT_ID = 16;
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;

template <BufferType Type>
struct BufferInfo {
    // Cons 消费者，Prod 生产者
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus()
    {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE2_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::M_FIX;
        } else if constexpr (Type == BufferType::C2) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::GM) {
            return HardEvent::MTE2_S;
        }
    }

    __aicore__ const static constexpr HardEvent ProdWaitConsStatus()
    {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE1_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::FIX_M;
        } else if constexpr (Type == BufferType::C2) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::GM) {
            return HardEvent::S_MTE2;
        }
    }

    __aicore__ const static constexpr pipe_t GetProdPipe()
    {
        if constexpr (Type == BufferType::L1) {
            return PIPE_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return PIPE_M;
        }
    }

    __aicore__ const static constexpr pipe_t GetConsPipe()
    {
        if constexpr (Type == BufferType::L1) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return PIPE_M;
        } else if constexpr (Type == BufferType::L0B) {
            return PIPE_M;
        } else if constexpr (Type == BufferType::L0C) {
            return PIPE_FIX;
        }
    }

    __aicore__ const static constexpr TPosition GetTPosition()
    {
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
        } else if constexpr (Type == BufferType::C2) {
            return TPosition::C2;
        }
    }

    static constexpr HardEvent EventP2C =
        ConsWaitProdStatus(); // 生产者到消费者方向的HardEvent：消费者等生产者提供/生产者通知消费者已生成
    static constexpr HardEvent EventC2P =
        ProdWaitConsStatus(); // 消费者到生产者方向的HardEvent：生产者等消费者消耗/消费者通知生产者已消耗
    static constexpr pipe_t ProdPipe = GetProdPipe(); // 使用Mutex时的生产者PIPE
    static constexpr pipe_t ConsPipe = GetConsPipe(); // 使用Mutex时的消费者PIPE
    static constexpr TPosition Position = GetTPosition();
};

// buffer绑定生产者、消费者关系
// L1 buffer的生产者为MTE2或者MTE3，消费者为MTE1
// L0A buffer的生产者为MTE1，消费者为M
// L0B buffer的生产者为MTE1，消费者为M
// L0C buffer的生产者为M，消费者为FIX
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG>
class Buffer {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;

    template <typename T>
    using TargetTensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<T>, LocalTensor<T>>;

public:
    __aicore__ inline Buffer()
    {
    }
    __aicore__ inline Buffer(TensorType tensor, uint32_t size)
    {
        tensor_ = tensor;
        size_ = size;
        if constexpr (syncType == SyncType::CROSS_CORE_SYNC_FORWARD) {
            id0_ = MAKE_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = MAKE_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
            id0_ = MAKE_ID;
            id1_ = MAKE_ID;
        } else {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        }
    }

    template <IdSource idSource = IdSource::INTERNAL>
    __aicore__ inline void Init()
    {
        static_assert(idSource == IdSource::INTERNAL, "idSource should IdSource::INTERNAL.");
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && idSource == IdSource::INTERNAL) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
                    c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    mutexId_ = AllocMutexID();
#endif
                }
            }
        }
    }

    template <IdSource idSource = IdSource::EXTERNAL>
    __aicore__ inline void Init(uint32_t id)
    {
        static_assert(idSource == IdSource::EXTERNAL, "idSource should IdSource::EXTERNAL.");
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && idSource == IdSource::EXTERNAL) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    // 静态Tensor场景, 用户自己指定EventId
                    p2cEventId_ = id;
                    c2pEventId_ = id;
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    mutexId_ = id;
#endif
                }
            }
        }
    }

    template <IdSource idSource = IdSource::INTERNAL>
    __aicore__ inline void UnInit()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                    if constexpr (idSource == IdSource::INTERNAL) {
                        GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(
                            p2cEventId_); // 确保只能被调用一次
                        GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                    }
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    if constexpr (idSource == IdSource::INTERNAL) {
                        ReleaseMutexID(mutexId_);
                    }
#endif
                }
            }
        }
    }

    template <pipe_t pipe>
    __aicore__ inline void Lock()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && syncMode == SyncMode::LOCK_UNLOCK) {
                Mutex::Lock<pipe>(mutexId_);
            }
        }
#endif
    }

    template <pipe_t pipe>
    __aicore__ inline void Unlock()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && syncMode == SyncMode::LOCK_UNLOCK) {
                Mutex::Unlock<pipe>(mutexId_);
            }
        }
#endif
    }

    template <HardEvent EventType>
    __aicore__ inline void Wait()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        WaitFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 消费者等待生产者完成生产
                    } else {
                        WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 生产者等待消费者完成消费
                    }
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        Mutex::Lock<BufferInfo<bufferType>::ConsPipe>(mutexId_); // 消费者加锁
                    } else {
                        Mutex::Lock<BufferInfo<bufferType>::ProdPipe>(mutexId_); // 生产者加锁
                    }
#endif
                }
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline void Set()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        SetFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 生产者通知消费者已完成生产
                    } else {
                        SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 消费者通知生产者已完成消费
                    }
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        Mutex::Unlock<BufferInfo<bufferType>::ProdPipe>(mutexId_); // 生产者解锁
                    } else {
                        Mutex::Unlock<BufferInfo<bufferType>::ConsPipe>(mutexId_); // 消费者解锁
                    }
#endif
                }
            }
        }
    }

    template <IdSource idSource = IdSource::INTERNAL>
    __aicore__ inline void SetEventID()
    {
        static_assert(idSource == IdSource::INTERNAL, "idSource should IdSource::INTERNAL.");
        if ASCEND_IS_AIC {
            if constexpr (idSource == IdSource::INTERNAL && syncMode == SyncMode::SET_WAIT_FLAG) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline TEventID GetEventID()
    {
        if ASCEND_IS_AIC {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                return p2cEventId_; // 生产者通知消费者已完成生产
            } else {
                return c2pEventId_; // 消费者通知生产者已完成消费
            }
        }
    }

    __aicore__ inline void SetCrossCoreID(uint32_t id0, uint32_t id1)
    {
        id0_ = id0;
        id1_ = id1;
    }

    template <bool isReuse = false>
    __aicore__ inline void WaitCrossCore()
    {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id0_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (isReuse) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
                } else {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id0_);
                }
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

    template <bool isReuse = false>
    __aicore__ inline void SetCrossCore()
    {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (isReuse) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                } else {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id1_);
                }
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

    template <typename T>
    __aicore__ inline TargetTensorType<T> GetTensor()
    {
        return tensor_.template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline TargetTensorType<T> GetTensor(uint64_t startindex)
    {
        TargetTensorType<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    TensorType tensor_;
    uint32_t size_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
    uint32_t id0_; // 用作正向同步：生产者通知消费者，或者消费者等待生产者；
    uint32_t id1_; // 用作反向同步：消费者通知生产者，或者生产者等待消费者；
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    MutexID mutexId_;
#endif
};
} // namespace fa_base_matmul
#endif
