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
 * \file hccl_shmem.hpp
 * \brief Templated HCCL shared-memory helper.
 *        Parameterized on `IS_A2`:
 *          - IS_A2 == true  : Atlas A2 (910B), context type is `HcclA2CombineOpParam`
 *          - IS_A2 == false : Atlas A3 (910_93), context type is `HcclOpResParam`
 *        Unifies the three legacy hccl_shmem.hpp variants under
 *        mega_moe / mega_moe_bf16 / mega_moe_w4_a8
 *        so that the unified op_kernel can compile for both platforms.
 */
#ifndef HCCL_SHMEM_HPP
#define HCCL_SHMEM_HPP


#include "kernel_operator.h"
#include "const_args.hpp"
#include "mc2_moe_context.h"

#if __has_include("../../../common/moe_distribute_base.h")
#include "../../../common/moe_distribute_base.h"
#else
#include "../../../../common/op_kernel/moe_distribute_base.h"
#endif

#define FORCE_INLINE_AICORE inline __attribute__((always_inline)) __aicore__
constexpr int32_t SHMEM_MEM = 700 * MB_SIZE;

constexpr uint16_t SEND_SYNC_EVENT_ID = 9;
constexpr uint16_t RECV_SYNC_EVENT_ID = 10;

constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
// CrossRankSyncV2 的 self state 从 SELF_STATE_OFFSET 开始，并按每个 block
// UB_ALIGN 个 int32_t（128B）保存一个状态。为该区域固定预留 256KB，
// 避免小 EP 场景下仅按 EP * STATE_OFFSET 预留导致访问越过 winSize。
constexpr uint32_t SELF_STATE_REGION_SIZE = 256 * 1024;
constexpr uint32_t STATE_OFFSET = 512;
constexpr uint32_t SYNC_STATE_RESERVED_SIZE = SELF_STATE_OFFSET + SELF_STATE_REGION_SIZE;

template <typename T>
FORCE_INLINE_AICORE void gm_store(__gm__ T *addr, T val)
{
    *((__gm__ T *)addr) = val;
}

template <typename T>
FORCE_INLINE_AICORE T gm_load(__gm__ T *cache)
{
    return *((__gm__ T *)cache);
}

template <typename T>
FORCE_INLINE_AICORE void gm_dcci(__gm__ T *addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(reinterpret_cast<GM_ADDR>(addr));

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

FORCE_INLINE_AICORE int64_t gm_signal_wait_until_eq_for_barrier(__gm__ int64_t *sig_addr, int64_t cmp_val)
{
    do {
        gm_dcci((__gm__ uint8_t *)sig_addr);
        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);
    return -1;
}

FORCE_INLINE_AICORE void gm_signal_wait_until_ne(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        AscendC::LocalTensor<int32_t> ub;
        ub.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        ub.address_.bufferAddr = 0;
        AscendC::GlobalTensor<int32_t> sig;
        sig.SetGlobalBuffer(sig_addr);
        AscendC::DataCopy(ub, sig, 8);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (ub(0) != cmp_val) {
            return;
        }
    } while (true);
    return;
}

FORCE_INLINE_AICORE void AIVRDMAPostSend(GM_ADDR srcDmaAddr, GM_ADDR destDmaAddr, uint64_t destRankId,
                                         uint64_t messageLen, __gm__ HcclAiRMAInfo *QpInfo,
                                         AscendC::LocalTensor<uint64_t> &ubLocal,
                                         AscendC::LocalTensor<uint32_t> &ubLocalHead)
{
    auto qpNum = QpInfo->qpNum;
    auto qp_ctx_entry =
        (__gm__ HcclAiRMAWQ *)(QpInfo->sqPtr + destRankId * qpNum * static_cast<uint64_t>(QpInfo->sizeOfAiRMAWQ));
    auto mem_info_table = QpInfo->memPtr;
    auto sizeof_memdetail = QpInfo->sizeOfAiRMAMem;
    auto sqBaseAddr = qp_ctx_entry->bufAddr;
    auto wqeSize = qp_ctx_entry->wqeSize;
    auto curHardwareHead = qp_ctx_entry->headAddr;
    cacheWriteThrough((__gm__ uint8_t *)curHardwareHead, 8);
    uint64_t curHead = *(__gm__ uint32_t *)(curHardwareHead);
    auto curHardwareTailAddr = qp_ctx_entry->tailAddr;
    uint64_t shift = 15U;
    auto QP_DEPTH = qp_ctx_entry->depth;

    AscendC::PipeBarrier<PIPE_ALL>();

    while (true) {
        cacheWriteThrough((__gm__ uint8_t *)curHardwareTailAddr, 8);
        if ((curHead - *(__gm__ uint32_t *)(curHardwareTailAddr)) < QP_DEPTH - 1) {
            break;
        }
        int64_t systemCycleAfter = AscendC::GetSystemCycle();
        (void)systemCycleAfter;
    }

    __gm__ uint8_t *wqeAddr = (__gm__ uint8_t *)(sqBaseAddr + wqeSize * (curHead % QP_DEPTH));

    uint64_t ownBit = (curHead >> shift) & 1U;
    uint32_t byte_4 = 3U;
    byte_4 |= ((~ownBit) << 7U) & (1U << 7U);
    byte_4 |= 1U << 8U;

    *(__gm__ uint32_t *)(wqeAddr) = byte_4;
    *(__gm__ uint32_t *)(wqeAddr + 4) = messageLen;
    *(__gm__ uint32_t *)(wqeAddr + 8) = 0;
    *(__gm__ uint32_t *)(wqeAddr + 12) = 1U << 24U;
    *(__gm__ uint32_t *)(wqeAddr + 16) = 0;
    __gm__ HcclAiRMAMemInfo *memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(wqeAddr + 20) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::REMOTE_INPUT)))
            ->key;
    *(__gm__ uint64_t *)(wqeAddr + 24) = (uint64_t)(destDmaAddr);

    // Setup SGE and write to GM
    __gm__ uint8_t *sgeAddr = wqeAddr + sizeof(struct hns_roce_rc_sq_wqe);
    *(__gm__ uint32_t *)(sgeAddr) = messageLen;
    memDetail = (__gm__ HcclAiRMAMemInfo *)(mem_info_table + sizeof_memdetail * destRankId);
    *(__gm__ uint32_t *)(sgeAddr + sizeof(uint32_t)) =
        ((__gm__ MemDetails *)(memDetail->memDetailPtr +
                               memDetail->sizeOfMemDetails * static_cast<uint32_t>(HcclAiRMAMemType::LOCAL_OUTPUT)))
            ->key;
    *(__gm__ uint64_t *)(sgeAddr + 2 * sizeof(uint32_t)) = (uint64_t)(srcDmaAddr);

    cacheWriteThrough(wqeAddr, sizeof(struct hns_roce_rc_sq_wqe) + sizeof(struct hns_roce_lite_wqe_data_seg));
    AscendC::PipeBarrier<PIPE_ALL>();
    curHead++;

    uint64_t doorBellInfo = 0;
    doorBellInfo |= qp_ctx_entry->wqn;
    doorBellInfo |= 0UL << 24UL;
    doorBellInfo |= (curHead % 65536UL) << 32UL;
    doorBellInfo |= static_cast<uint64_t>(qp_ctx_entry->sl) << 48UL;

    __gm__ uint64_t *doorBellAddr = (__gm__ uint64_t *)(qp_ctx_entry->dbAddr);
    AscendC::PipeBarrier<PIPE_ALL>();

    ubLocal.SetValue(0, doorBellInfo);
    AscendC::GlobalTensor<uint64_t> DBGlobalTensor;
    DBGlobalTensor.SetGlobalBuffer(doorBellAddr);
    AscendC::DataCopyExtParams copyParams{1, 1 * sizeof(uint64_t), 0, 0, 0};
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(DBGlobalTensor, ubLocal, copyParams);
    AscendC::PipeBarrier<PIPE_ALL>();

    ubLocalHead.SetValue(0, static_cast<uint32_t>(curHead));
    AscendC::GlobalTensor<uint32_t> HeadGlobalTensor;
    HeadGlobalTensor.SetGlobalBuffer((__gm__ uint32_t *)curHardwareHead);
    AscendC::DataCopyExtParams copyParamsHead{1, 1 * sizeof(uint32_t), 0, 0, 0};
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(HeadGlobalTensor, ubLocalHead, copyParamsHead);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <bool IS_A2>
class HcclShmem {
public:
    std::conditional_t<IS_A2, __gm__ HcclA2CombineOpParam *, __gm__ HcclOpResParam *> WinContext_{nullptr};
    AscendC::LocalTensor<int32_t> ub;
    FORCE_INLINE_AICORE
    HcclShmem()
    {
    }
    FORCE_INLINE_AICORE
    void initShmem(__gm__ Mc2Aclnn::Mc2MoeContext *mc2Context)
    {
        GM_ADDR kfcContextAddr = (GM_ADDR)mc2Context->kfcContextAddr;
        if constexpr (IS_A2) {
            WinContext_ = (__gm__ HcclA2CombineOpParam *)kfcContextAddr;
            m_rank = WinContext_->rankId;
            m_rankSize = WinContext_->rankNum;
        } else {
            WinContext_ = (__gm__ HcclOpResParam *)kfcContextAddr;
            m_rank = WinContext_->localUsrRankId;
            m_rankSize = WinContext_->rankSize;
        }
        m_segmentSize = WinContext_->winSize;
        if constexpr (IS_A2) {
            // A2 只使用按 rank 分配的 CrossRankSync 状态区。
            m_tailReservedSize = static_cast<size_t>(m_rankSize) * STATE_OFFSET;
        } else {
            // A3 尾部同时容纳：
            // 1. [0, rankSize * STATE_OFFSET) 的 per-rank 同步状态；
            // 2. [SELF_STATE_OFFSET, SELF_STATE_OFFSET + SELF_STATE_REGION_SIZE)
            //    的 per-block 本轮状态。
            // 当前支持的最大 rankSize 为 128，per-rank 区最多 64KB，不会与
            // 256KB 起始的 self-state 区重叠。
            m_tailReservedSize =
                static_cast<size_t>(SYNC_STATE_RESERVED_SIZE) > static_cast<size_t>(m_rankSize) * STATE_OFFSET ?
                    static_cast<size_t>(SYNC_STATE_RESERVED_SIZE) :
                    static_cast<size_t>(m_rankSize) * STATE_OFFSET;
        }
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator()() const
    { // No parameters: return pointer to local peermem
        if constexpr (IS_A2) {
            if (WinContext_->multiFlag == 0U) {
                return (GM_ADDR)(WinContext_->windowsIn[WinContext_->rankId]);
            } else {
                return (GM_ADDR)(WinContext_->data[WinContext_->rankId].localInput.addr);
            }
        } else {
            return (GM_ADDR)(WinContext_->localWindowsIn);
        }
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator()(int32_t index) const
    { // With index parameter: return pointer to the base address of remote peermem
        if constexpr (IS_A2) {
            if (WinContext_->multiFlag == 0U) {
                return (GM_ADDR)(WinContext_->windowsIn[index]);
            } else {
                if (index == WinContext_->rankId) {
                    return (GM_ADDR)(WinContext_->data[index].localInput.addr);
                } else {
                    return (GM_ADDR)(WinContext_->data[index].remoteInput.addr);
                }
            }
        } else {
            return (GM_ADDR)((index == m_rank) ?
                                 WinContext_->localWindowsIn :
                                 ((HcclRankRelationResV2 *)(WinContext_->remoteRes[index].nextDevicePtr))->windowsIn);
        }
    }

    FORCE_INLINE_AICORE
    GM_ADDR operator()(int64_t offset, int32_t rankId) const
    {
        if (offset < 0 || offset >= m_segmentSize) {
            return nullptr;
        }
        if (rankId < 0 || rankId >= m_rankSize) {
            return nullptr;
        }
        if constexpr (IS_A2) {
            if (WinContext_->multiFlag == 0U) {
                return (GM_ADDR)(WinContext_->windowsIn[rankId] + offset);
            } else {
                if (rankId == WinContext_->rankId) {
                    return (GM_ADDR)(WinContext_->data[rankId].localInput.addr + offset);
                } else {
                    return (GM_ADDR)(WinContext_->data[rankId].remoteInput.addr + offset);
                }
            }
        } else {
            return (GM_ADDR)((rankId == m_rank) ?
                                 WinContext_->localWindowsIn :
                                 ((HcclRankRelationResV2 *)(WinContext_->remoteRes[rankId].nextDevicePtr))->windowsIn) +
                   offset;
        }
    }

    // A2-only helpers: return windowsOut addresses.
    // For A3 these return nullptr (W4A8/INT8 legacy never used these).
    FORCE_INLINE_AICORE
    GM_ADDR windowsOutAddr() const
    {
        if constexpr (IS_A2) {
            if (WinContext_->multiFlag == 0U) {
                return (GM_ADDR)(WinContext_->windowsOut[WinContext_->rankId]);
            } else {
                return (GM_ADDR)(WinContext_->data[WinContext_->rankId].localOutput.addr);
            }
        } else {
            return nullptr;
        }
    }

    FORCE_INLINE_AICORE
    GM_ADDR windowsOutAddr(int32_t rankId) const
    {
        if constexpr (IS_A2) {
            if (WinContext_->multiFlag == 0U) {
                return (GM_ADDR)(WinContext_->windowsOut[rankId]);
            } else {
                if (rankId == WinContext_->rankId) {
                    return (GM_ADDR)(WinContext_->data[rankId].localOutput.addr);
                } else {
                    return (GM_ADDR)(WinContext_->data[rankId].remoteOutput.addr);
                }
            }
        } else {
            return nullptr;
        }
    }

    FORCE_INLINE_AICORE
    GM_ADDR windowsOutAddr(int64_t offset, int32_t rankId) const
    {
        if constexpr (IS_A2) {
            if (WinContext_->multiFlag == 0U) {
                return (GM_ADDR)(WinContext_->windowsOut[rankId] + offset);
            } else {
                if (rankId == WinContext_->rankId) {
                    return (GM_ADDR)(WinContext_->data[rankId].localOutput.addr + offset);
                } else {
                    return (GM_ADDR)(WinContext_->data[rankId].remoteOutput.addr + offset);
                }
            }
        } else {
            return nullptr;
        }
    }

    FORCE_INLINE_AICORE
    size_t SegmentSize() const
    {
        return m_segmentSize;
    }

    FORCE_INLINE_AICORE
    size_t TailReservedSize() const
    {
        return m_tailReservedSize;
    }

    FORCE_INLINE_AICORE
    int32_t Rank() const
    {
        return m_rank;
    }

    FORCE_INLINE_AICORE
    int32_t RankSize() const
    {
        return m_rankSize;
    }

    FORCE_INLINE_AICORE
    ~HcclShmem()
    {
    }

    FORCE_INLINE_AICORE
    void CrossRankSync()
    {
        uint64_t flag_offset = ((m_segmentSize - m_tailReservedSize)) / sizeof(int64_t);
        __gm__ int64_t *sync_counter = (__gm__ int64_t *)(*this)() + flag_offset;
        __gm__ int64_t *sync_base = (__gm__ int64_t *)(*this)() + flag_offset + m_rankSize * 8;
        int64_t count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        for (int i = vec_id; i < m_rankSize; i += vec_size) {
            __gm__ int64_t *sync_remote = (__gm__ int64_t *)((*this)(i)) + flag_offset + m_rank * 8;
            gm_store(sync_remote, count);
            gm_dcci((__gm__ uint8_t *)sync_remote);
            auto sync_check = sync_counter + i * 8;
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
    }

    // Cross-server-capable overload of CrossRankSync.
    // Same counter-based barrier protocol as the no-arg version, but routes the
    // remote write through RDMA when the destination rank lives on another server.
    //
    // ctrBuffer is a caller-owned UB scratch laid out as (in int32 units):
    //   [0 .. 7]                            -> payload (count is stored at [0]);
    //                                          full 32-byte block is the RDMA payload
    //   [UB_ALIGN/4 .. 2*UB_ALIGN/4 - 1]    -> rdmaUbLocal  (uint64 doorbell scratch)
    //   [2*UB_ALIGN/4 .. 3*UB_ALIGN/4 - 1]  -> rdmaUbLocalHead (uint32 head scratch)
    // i.e. needs at least 3 * UB_ALIGN = 96 bytes of UB and must be free for the
    // duration of this call (matches how CrossRankSyncV2Set lays out its scratch).
    FORCE_INLINE_AICORE
    void CrossRankSync(AscendC::LocalTensor<int32_t> ctrBuffer)
    {
        uint64_t flag_offset_i64 = ((m_segmentSize - m_tailReservedSize)) / sizeof(int64_t);
        uint64_t flag_offset_bytes = ((m_segmentSize - m_tailReservedSize));
        __gm__ int64_t *sync_counter = (__gm__ int64_t *)(*this)() + flag_offset_i64;
        __gm__ int64_t *sync_base = (__gm__ int64_t *)(*this)() + flag_offset_i64 + m_rankSize * 8;
        gm_dcci((__gm__ uint8_t *)sync_base);
        int64_t count = gm_load(sync_base) + 1;
        int vec_id = AscendC::GetBlockIdx();
        int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

        AscendC::LocalTensor<uint64_t> rdmaUbLocal;
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead;
        AscendC::GlobalTensor<int32_t> gmCrossServerPayload;
        if constexpr (IS_A2) {
            rdmaUbLocal = ctrBuffer[UB_ALIGN / sizeof(int32_t)].template ReinterpretCast<uint64_t>();
            rdmaUbLocalHead = ctrBuffer[2 * UB_ALIGN / sizeof(int32_t)].template ReinterpretCast<uint32_t>();
            gmCrossServerPayload.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
                windowsOutAddr() + flag_offset_bytes + static_cast<uint64_t>(m_rank) * 8 * sizeof(int64_t)));
            // Only this core's stride may include cross-server targets; skip UB→GM staging if none.
            bool needRdmaStage = false;
            int32_t const localServerId = m_rank / SERVER_RANK_SIZE_A2;
            for (int j = vec_id; j < m_rankSize; j += vec_size) {
                if ((j / SERVER_RANK_SIZE_A2) != localServerId) {
                    needRdmaStage = true;
                    break;
                }
            }
            if (needRdmaStage) {
                ctrBuffer.template ReinterpretCast<int64_t>().SetValue(0, count);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
                AscendC::DataCopy(gmCrossServerPayload, ctrBuffer, 8);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        } else {
            ctrBuffer.template ReinterpretCast<int64_t>().SetValue(0, count);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        }

        // Phase 1: post every outbound write. RDMA is async, so we must finish
        // posting all WQEs before any local poll (otherwise an intra-server peer
        // may already have arrived while our own cross-server WQEs are still
        // sitting unposted, and the producer side never makes forward progress).
        for (int i = vec_id; i < m_rankSize; i += vec_size) {
            __gm__ int64_t *sync_remote = (__gm__ int64_t *)((*this)(i)) + flag_offset_i64 + m_rank * 8;
            if constexpr (IS_A2) {
                int32_t localServerId = m_rank / SERVER_RANK_SIZE_A2;
                int32_t dstServerId = i / SERVER_RANK_SIZE_A2;
                if (dstServerId != localServerId) {
                    AIVRDMAPostSend((GM_ADDR)gmCrossServerPayload.GetPhyAddr(), (GM_ADDR)sync_remote,
                                    static_cast<uint64_t>(i), 4 * sizeof(int64_t),
                                    reinterpret_cast<__gm__ HcclAiRMAInfo *>(WinContext_->aiRMAInfo), rdmaUbLocal,
                                    rdmaUbLocalHead);
                    continue;
                }
            }
            gm_store(sync_remote, count);
            gm_dcci((__gm__ uint8_t *)sync_remote);
        }

        // Phase 2: wait for the same set of source ranks to land their count in
        // our local sync slots. Works uniformly for intra-server (gm_store) and
        // inter-server (RDMA) producers because both ultimately materialize the
        // same int64 at sync_counter[i*8].
        for (int i = vec_id; i < m_rankSize; i += vec_size) {
            auto sync_check = sync_counter + i * 8;
            gm_signal_wait_until_eq_for_barrier(sync_check, count);
        }

        AscendC::SyncAll<true>();
        gm_store(sync_base, count);
        gm_dcci((__gm__ uint8_t *)sync_base);
    }

    FORCE_INLINE_AICORE
    void InitStatusTargetSum()
    {
        using namespace AscendC;
        uint64_t flag_offset = ((m_segmentSize - m_tailReservedSize)) + SELF_STATE_OFFSET;
        uint32_t coreIdx = GetBlockIdx();
        GlobalTensor<int32_t> selfStatusTensor;
        selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)((*this)() + flag_offset));
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
        int32_t state = selfStatusTensor(coreIdx * UB_ALIGN);
        if (state == 0) {
            sumTarget_ = static_cast<float>(1.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0x3F800000; // 1.0f
            epStateValue_ = 0x3F800000;                        // 1.0f
        } else {
            sumTarget_ = static_cast<float>(0.0);
            selfStatusTensor(coreIdx * UB_ALIGN) = 0;
            epStateValue_ = 0;
        }
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            selfStatusTensor[coreIdx * UB_ALIGN]);
        __asm__ __volatile__("");
    }

    FORCE_INLINE_AICORE
    void CrossRankSyncV2Set(AscendC::LocalTensor<int32_t> ctrBuffer)
    {
        uint32_t stateOffset_ = STATE_OFFSET;

        uint64_t flag_offset = ((m_segmentSize - m_tailReservedSize)) + m_rank * stateOffset_;
        int vec_size = get_block_num();
        int vec_id = get_block_idx();

        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
        pipe_barrier(PIPE_ALL);

        ctrBuffer.SetValue(0, epStateValue_);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::LocalTensor<uint64_t> rdmaUbLocal;
        AscendC::LocalTensor<uint32_t> rdmaUbLocalHead;
        if constexpr (IS_A2) {
            rdmaUbLocal = ctrBuffer[UB_ALIGN / sizeof(int32_t)].template ReinterpretCast<uint64_t>();
            rdmaUbLocalHead = ctrBuffer[2 * UB_ALIGN / sizeof(int32_t)].template ReinterpretCast<uint32_t>();
        }
        for (uint32_t dstEpIdx = vec_id; dstEpIdx < m_rankSize; dstEpIdx += vec_size) {
            AscendC::GlobalTensor<int32_t> gmDstStates;
            gmDstStates.SetGlobalBuffer((__gm__ int32_t *)((*this)(flag_offset, dstEpIdx)));
            if constexpr (IS_A2) {
                int32_t localServerId = m_rank / SERVER_RANK_SIZE_A2;
                int32_t dstServerId = dstEpIdx / SERVER_RANK_SIZE_A2;
                if (dstServerId != localServerId) {
                    AscendC::GlobalTensor<int32_t> gmLocalOutputState;
                    gmLocalOutputState.SetGlobalBuffer(
                        reinterpret_cast<__gm__ int32_t *>(windowsOutAddr() + flag_offset));
                    AscendC::DataCopy(gmLocalOutputState, ctrBuffer, 8);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AIVRDMAPostSend((GM_ADDR)gmLocalOutputState.GetPhyAddr(), (GM_ADDR)gmDstStates.GetPhyAddr(),
                                    static_cast<uint64_t>(dstEpIdx), 8 * sizeof(int32_t),
                                    reinterpret_cast<__gm__ HcclAiRMAInfo *>(WinContext_->aiRMAInfo), rdmaUbLocal,
                                    rdmaUbLocalHead);
                    continue;
                }
            }
            AscendC::DataCopy(gmDstStates, ctrBuffer, 8);
        }
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);
    }

    FORCE_INLINE_AICORE
    void CrossRankSyncV2Wait(AscendC::LocalTensor<float> statusTensor, AscendC::LocalTensor<float> gatherMaskOutTensor,
                             AscendC::LocalTensor<uint32_t> gatherTmpTensor,
                             AscendC::LocalTensor<float> statusSumOutTensor)
    {
        uint64_t flag_offset = ((m_segmentSize - m_tailReservedSize));
        int vec_size = get_block_num();
        int vec_id = get_block_idx();
        uint32_t stateOffset_ = STATE_OFFSET;

        uint32_t sendRankNum_ = m_rankSize / vec_size;
        uint32_t remainderRankNum = m_rankSize % vec_size;
        uint32_t startRankId_ = sendRankNum_ * vec_id;
        if (vec_id < remainderRankNum) {
            sendRankNum_++;
            startRankId_ += vec_id;
        } else {
            startRankId_ += remainderRankNum;
        }
        uint32_t endRankId_ = startRankId_ + sendRankNum_;
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);

        AscendC::GlobalTensor<float> epStatusSpaceGlobalTensor_;
        epStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)((*this)() + flag_offset));

        if (startRankId_ < m_rankSize) {
            AscendC::PipeBarrier<PIPE_ALL>();
            gatherTmpTensor.SetValue(0, 1);
            uint32_t mask = 1;
            uint64_t rsvdCnt = 0;
            AscendC::DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1, static_cast<uint16_t>(15), 0};

            float sumOfFlag = static_cast<float>(-1.0);
            float minTarget = (sumTarget_ * sendRankNum_) - (float)0.5;
            float maxTarget = (sumTarget_ * sendRankNum_) + (float)0.5;
            AscendC::SumParams sumParams{1, sendRankNum_, sendRankNum_};

            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);

            while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
                AscendC::DataCopy<float>(
                    statusTensor, epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)], intriParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, mask,
                           {1, (uint16_t)sendRankNum_, 1, 0}, rsvdCnt);

                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                sumOfFlag = statusSumOutTensor.GetValue(0);
            }
        }

        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);

        // unpermute
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
    }

    FORCE_INLINE_AICORE
    __gm__ int32_t *SyncBaseAddr()
    {
        uint64_t flag_offset = (m_segmentSize - MB_SIZE) / sizeof(int32_t);
        return (__gm__ int32_t *)(*this)() + flag_offset + 2048;
    }

    // 将全卡同步函数实际操作的本地 peermem 内存区域通过 exceptionDump 进行 dump。
    // 使用带 stride 的 Dump 跳过空洞，避免 dump 整个 1MB。
    //
    // 布局参考（相对 flag_offset = m_segmentSize - MB_SIZE）：
    //   CrossRankSync（本 rank peermem 收到所有 rank 的 counter，按 world_size 存）：
    //     - sync_counter[rank][16] int32_t（每 rank 64B，仅 [0]4B 被读写）
    //     - sync_base: 1 个 int32_t（位于偏移 2048*4=8192B 处）
    //   CrossRankSyncV2（per-rank status 按 world_size 存，selfStatus 按本 rank AIV 核数存）：
    //     - InitStatusTargetSum 写 selfStatus[GetBlockIdx() * UB_ALIGN]（4B，UB_ALIGN=32）
    //     - CrossRankSyncV2Set 写 per-rank status[m_rank*STATE_OFFSET]（32B）到其他 rank
    //     - CrossRankSyncV2Wait 读 per-rank status[startRankId*STATE_OFFSET]（4B，stride 64B）
    //
    // 模板参数 ExceptionDumpT 需提供：
    //   void Dump(GM_ADDR dumpAddr, size_t blockCount, size_t blockLen, size_t srcStride)
    template <typename ExceptionDumpT>
    FORCE_INLINE_AICORE void DumpSyncRegions(ExceptionDumpT &exceptionDump)
    {
        uint64_t flagOffsetBytes = m_segmentSize - m_tailReservedSize;
        GM_ADDR base = (*this)() + flagOffsetBytes;

        // Region 0: CrossRankSync sync_counter[rank]，数据类型 int64_t
        // 访问模式：[rank*64B, rank*64B+8B)，共 m_rankSize 个 block，stride=64B
        exceptionDump.Dump(base, static_cast<size_t>(m_rankSize), sizeof(int64_t), 64U);

        // Region 1: CrossRankSync sync_base，单 int64_t
        // 地址 = base + m_rankSize * 64（sync_counter 之后紧接着 sync_base）
        exceptionDump.Dump(base + static_cast<uint64_t>(m_rankSize) * 64U, sizeof(int64_t));

        // Region 2: CrossRankSyncV2 per-rank status，每 rank 32B（V2Set 写 8 int32_t）
        // 访问模式：[rank*512B, rank*512B+32B)，共 m_rankSize 个 block，stride=STATE_OFFSET
        exceptionDump.Dump(base, static_cast<size_t>(m_rankSize), UB_ALIGN, static_cast<size_t>(STATE_OFFSET));

        // Region 3: CrossRankSyncV2 selfStatus，本 rank 各 AIV 核的 init 状态
        // InitStatusTargetSum 用 GetBlockIdx() 索引，范围 [0, get_block_num())
        // selfStatusTensor[coreIdx * UB_ALIGN] 中 UB_ALIGN 是 int32_t 元素索引，
        // 实际字节间距 = UB_ALIGN * sizeof(int32_t) = 128B
        exceptionDump.Dump(base + SELF_STATE_OFFSET, static_cast<size_t>(GetBlockNum()), sizeof(int32_t),
                           static_cast<size_t>(UB_ALIGN * sizeof(int32_t)));
    }

private:
    GM_ADDR symmetricPtr;
    int32_t m_rank;
    int32_t m_rankSize;
    size_t m_segmentSize;
    size_t m_tailReservedSize{0}; // 尾部 CrossRankSync 保留空间
    float sumTarget_{0.0};
    int32_t epStateValue_;
};

#endif
