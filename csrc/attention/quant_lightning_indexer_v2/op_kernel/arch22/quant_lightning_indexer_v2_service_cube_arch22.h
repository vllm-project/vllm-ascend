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
 * \file quant_lightning_indexer_v2_service_cube_arch22.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
#define QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_lightning_indexer_v2_common_arch22.h"

namespace QLIV2Kernel {
using namespace QLIV2Common;
struct MmInfo {
    int64_t s2L0LoopId;
    int64_t s1gL0LoopId;
    int64_t s2L0RealSize;
    int64_t s2GmOffset;
};

template <typename QLIV2T>
class QLIV2Matmul {
public:
    using Q_T = typename QLIV2T::queryType;
    using K_T = typename QLIV2T::keyType;

    __aicore__ inline QLIV2Matmul(){};
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm, const GlobalTensor<K_T>& keyGm,
                                               const GlobalTensor<Q_T>& queryGm, const GlobalTensor<float>& mm1ResGm,
                                               const GlobalTensor<half>& weightWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo& constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QLIV2Common::RunInfo& runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t DOUBLE_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;
    // K 每 2 对只装一次：K 槽在 0/1 之间轮转，第二对 QK 后归还；S 仍使用全部 4 个 L0B 槽，
    // 只在每个 WS 内跳过当前仍被 K 占用的槽。
    static constexpr uint64_t K_L0B_SLOT_NUM = 2;
    static constexpr uint64_t S_L0B_SLOT_NUM = L0AB_BUF_NUM;

    // 优化：Q 与 W 的 L0A 常驻槽共用一个 M→MTE1 守卫 id（PIN_MTE1_EVENT），
    // 每个行块只在首尾各 wait/set 一次 —— 内层循环的 M→MTE1 等待条数与基线完全相同（每对 3 条），
    // 从而避开"新增 M→MTE1 等待即挂死"的硬件限制（见 README §15.3）。
    // L0A 布局：槽0 = 本行块 4 个 s1 行的 W（每行 2KB，共 8KB）；槽1 空闲；槽2/3 = 两个 M 块的 Q（各 16KB）。
    static constexpr uint64_t Q_SLOT_BASE = 2;
    static constexpr uint64_t W_SLOT_HALF_BASE = 0;       // W 区在 L0A 的 half 元素起点
    static constexpr uint64_t W_ROW_STRIDE_HALF = 1024;   // 每行 W 占 1024 个 half（=2KB）
    static constexpr uint32_t PIN_MTE1_EVENT = EVENT_ID7; // Q+W 常驻区共用守卫（ids 7）

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QW_MTE1_MTE2_EVENT = EVENT_ID5; // KEY_MTE1_MTE2_EVENT + DOUBLE_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t FIX_MTE1_EVENT = EVENT_ID4;
    // S（L1 上的中间结果）双缓冲 → 3 块 + 消费滞后 2 对：让"S 就绪"(FIX_MTE1) 等待落在
    // 两对之前置位的旗标上（被更深的流水吸收）。旗标 id 用 4/5/6（FIX→MTE1 方向 id 6 空闲；
    // 事件 id 按流水方向隔离，与 M_MTE1 方向的同号 id 不冲突）。
    // 注意：缓冲槽与旗标 id 都必须用**全局对号**（成员 sL1BufIdx_，跨 行块/s2 内循环 连续），
    // 用本 call 内的局部游标会与写侧的全局游标错位 —— 见 scripts/prof_output/naie273qli/
    // model_s_pipeline.py（协议离线模型：局部游标版会出 "Wait 无可用 Set" 的挂死形状）。
    static constexpr uint64_t S_L1_BUF_NUM = 3;
    static constexpr uint64_t S_L1_BUF_LAG = S_L1_BUF_NUM - 1;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S1G_BASIC_BLOCK_L1 = 256;

    static constexpr uint64_t S1G_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK;
    static constexpr uint64_t SL1_BUFFER_OFFSET = S1G_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t WEIGHT_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * BLOCK_CUBE;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_FP16_16K = 16 * 512;
    static constexpr uint64_t L0C_BUFFER_OFFSET = 64 * 256;

private:
    __aicore__ inline void WeightDmaCopy(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0RealSize, uint64_t kSlot);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize,
                                          uint64_t qSlot);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const QLIV2Common::RunInfo& runInfo, uint64_t dstSlot);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize, uint64_t sL1SlotIdx,
                                      int64_t mStartPt, uint64_t sSlot);
    __aicore__ inline void LoadWeightToL0a(uint64_t l1ElemOffset, uint64_t l0aHalfOffset);
    __aicore__ inline void LoadAllWeightRowsToL0a();
    __aicore__ inline void ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset, uint64_t wRowIdx,
                                     uint64_t sSlot);
    __aicore__ inline void FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                       uint64_t s2GmOffset, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, uint64_t qSlot, uint64_t kSlot);
    // sL1SlotIdx：已取模的 S 缓冲槽号（0..S_L1_BUF_NUM-1），不是运行计数器
    __aicore__ inline void ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1SlotIdx,
                                     const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                     const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo);
    __aicore__ inline void CalcMmInfo(MmInfo& mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt, const MmInfo& lastMmInfo,
                                      const QLIV2Common::RunInfo& runInfo);
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIV2T::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIV2T::keyLayout;
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<half> weightGm_;
    GlobalTensor<float> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;
    TBuf<TPosition::A1> bufWeightL1_;
    LocalTensor<half> weightL1_;
    TBuf<TPosition::B1> bufSL1_;
    LocalTensor<half> sL1_;

    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<Q_T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<K_T> l0b_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t qwL1Mte2BufIdx_ = 0;
    uint64_t sL1BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;
    uint64_t sL0BufIdx_ = 0;       // S 的 L0B 4 槽环游标
    bool kPairFilledNext_ = false; // (旧 kp3 对填充，保留兼容)
    bool kPrefetched_ = false;     // 本组 K 已在上一组预取到 L1

    ConstInfo constInfo_;
};

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitParams(const ConstInfo& constInfo)
{
    constInfo_ = constInfo;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(bufQL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufWeightL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * BLOCK_CUBE * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();
    pipe->InitBuffer(bufSL1_, S_L1_BUF_NUM * S2_BASIC_BLOCK_L0 * S1G_BASIC_BLOCK_L0 * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    pipe->InitBuffer(bufL0A_, 64 * 1024);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, 64 * 1024);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, 128 * 1024);
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm,
                                                                const GlobalTensor<K_T>& keyGm,
                                                                const GlobalTensor<Q_T>& queryGm,
                                                                const GlobalTensor<float>& mm1ResGm,
                                                                const GlobalTensor<half>& weightWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    weightGm_ = weightWorkspaceGm;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ProcessWs(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t sL1SlotIdx,
                                                      const MmInfo& mmInfo, const QLIV2Common::RunInfo& runInfo)
{
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0cBufIdx_ & (DOUBLE_BUF_NUM - 1)));
    // S 用全部 4 个 L0B 槽，但必须跳过本迭代 ProcessQk 占用/正在装载的 K 槽。
    // 对 s1g=0 的 WS，下一组 K 已在同迭代装好并要占住两对；对 s1g=1 的 WS，本组 K 也要
    // 到本次 QK 结束才释放。若 S 落进该槽，S 装载会等 QK 的 MMAD，WS 被整体推到 QK 之后。
    const uint64_t avoidSlot = (mmInfo.s2L0LoopId + 1) % K_L0B_SLOT_NUM;
    // W 行号按块递增，除法只在循环外做一次（原来每块一次 DIV）。
    uint64_t wRowIdx = static_cast<uint64_t>(s1gL1Offset / static_cast<int64_t>(constInfo_.gSize));
    for (int64_t s1gOffset = 0; s1gOffset < s1gL0RealSize; s1gOffset += constInfo_.gSize) {
        uint64_t sSlot = sL0BufIdx_ % S_L0B_SLOT_NUM;
        if (sSlot == avoidSlot) {
            sSlot = (sSlot + 1) % S_L0B_SLOT_NUM;
        }
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + sSlot);
        LoadSToL0b(s1gL0RealSize, mmInfo.s2L0RealSize, sL1SlotIdx, s1gOffset, sSlot);
        ComputeWs(s1gL0RealSize, mmInfo.s2L0RealSize, s1gOffset, wRowIdx, sSlot);
        ++wRowIdx;
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + sSlot);
        sL0BufIdx_ = sSlot + 1;
    }

    FixpResToGm(s1gL0RealSize / constInfo_.gSize, mmInfo.s2L0RealSize, s1gL1Offset / constInfo_.gSize,
                mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0, runInfo);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0cBufIdx_ & (DOUBLE_BUF_NUM - 1)));
    l0cBufIdx_++;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset,
                                                      uint64_t s1L0LoopCnt, const MmInfo& mmInfo,
                                                      const QLIV2Common::RunInfo& runInfo)
{
    // K 填充滚动预取：每组的 K 在上一组就填到另一块 L1 槽；当前组只等上一组已发的
    // 填充完成，不再等本组填充。本组 K 的 L0B 装载释放后，再为下一组发填充（与 QK 重叠）。
    uint64_t curKIdx = keyL1BufIdx_ & (DOUBLE_BUF_NUM - 1);
    uint64_t myGmOffsetK = 0;
    bool prefetchNextK = false;
    if (mmInfo.s1gL0LoopId == 0) {
        if constexpr (K_LAYOUT_T == LI_LAYOUT::PA_BBND) {
            myGmOffsetK = runInfo.s2Idx * constInfo_.s2BaseSize + mmInfo.s2GmOffset;
            if (!kPrefetched_) {
                WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + curKIdx);
                KeyNd2NzForPA(mmInfo.s2L0RealSize, myGmOffsetK, runInfo, curKIdx);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT + curKIdx);
            }
            kPrefetched_ = false;
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT + curKIdx);
            const uint64_t s2Tail =
                runInfo.s2Idx * constInfo_.s2BaseSize + static_cast<uint64_t>(runInfo.actualSingleProcessSInnerSize);
            prefetchNextK =
                mmInfo.s2L0RealSize == S2_BASIC_BLOCK_L0 &&
                ((myGmOffsetK % constInfo_.kCacheBlockSize) + 2 * S2_BASIC_BLOCK_L0 <= constInfo_.kCacheBlockSize) &&
                (myGmOffsetK + 2 * S2_BASIC_BLOCK_L0 <= s2Tail);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + curKIdx);
            KeyNd2Nz(mmInfo.s2L0RealSize, mmInfo, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
            WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        }
    }

    // 常驻区（Q + W）：Q/W 只取决于查询行、与键无关 ⇒ 真正的装载只需按**行块**做一次
    // （L1 源缓冲同样是每行块填一次：isFirstS2InnerLoop，见 kernel_arch22.h:609）；
    // 但**握手仍保持"每基块一次"**（与已验证版逐字一致，旗标配平不变），只是后续基块跳过装载体。
    const uint64_t qSlot = Q_SLOT_BASE + mmInfo.s1gL0LoopId;
    if (mmInfo.s2L0LoopId == 0) {
        if (mmInfo.s1gL0LoopId == 0) {
            WaitFlag<HardEvent::M_MTE1>(PIN_MTE1_EVENT); // 每基块一次（保持与原协议同构）
            if (runInfo.isFirstS2InnerLoop) {
                LoadAllWeightRowsToL0a(); // 仅行块首块真装
            }
        }
        if (runInfo.isFirstS2InnerLoop) {
            LoadQueryToL0a(s1gL1Offset, runInfo.actMBaseSize, s1gL0RealSize, qSlot);
        }
    }
    // K 槽按 s2 组号在 0/1 之间轮转；两对 QK 共用，第二对 QK 后归还。
    const uint64_t kSlot = mmInfo.s2L0LoopId % K_L0B_SLOT_NUM;
    if (mmInfo.s1gL0LoopId == 0) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + kSlot);
        LoadKeyToL0b(mmInfo.s2L0RealSize, kSlot);
        // 当前 K 的 L1 只被这一次 L0B 装载消费，装完即可释放；随后为下一组发填充。
        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + curKIdx);
        keyL1BufIdx_++;
        if constexpr (K_LAYOUT_T == LI_LAYOUT::PA_BBND) {
            if (prefetchNextK) {
                const uint64_t nextKIdx = keyL1BufIdx_ & (DOUBLE_BUF_NUM - 1);
                WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + nextKIdx);
                KeyNd2NzForPA(S2_BASIC_BLOCK_L0, myGmOffsetK + S2_BASIC_BLOCK_L0, runInfo, nextKIdx);
                SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT + nextKIdx);
                kPrefetched_ = true;
            }
        }
    }

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0cBufIdx_ & (DOUBLE_BUF_NUM - 1)));
    ComputeQk(s1gL0RealSize, mmInfo.s2L0RealSize, qSlot, kSlot);
    if (mmInfo.s1gL0LoopId + 1 >= s1L0LoopCnt) {
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + kSlot);
    }

    FixpSToL1(s1gL0RealSize, mmInfo.s2L0RealSize);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + (l0cBufIdx_ & (DOUBLE_BUF_NUM - 1)));
    l0cBufIdx_++;
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::CalcMmInfo(MmInfo& mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt,
                                                       const MmInfo& lastMmInfo, const QLIV2Common::RunInfo& runInfo)
{
    // s1L0LoopCnt 实际只有 1 或 2（基块最多 256 行）：用移位/掩码替代 DIV/REM。
    if (s1L0LoopCnt == 2) {
        mmInfo.s2L0LoopId = loopIdx >> 1;
        mmInfo.s1gL0LoopId = loopIdx & 1;
    } else if (s1L0LoopCnt == 1) {
        mmInfo.s2L0LoopId = loopIdx;
        mmInfo.s1gL0LoopId = 0;
    } else {
        mmInfo.s2L0LoopId = loopIdx / s1L0LoopCnt;
        mmInfo.s1gL0LoopId = loopIdx % s1L0LoopCnt;
    }

    if (mmInfo.s1gL0LoopId == 0) {
        mmInfo.s2GmOffset = mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0;
        mmInfo.s2L0RealSize = mmInfo.s2GmOffset + S2_BASIC_BLOCK_L0 > runInfo.actualSingleProcessSInnerSize ?
                                  runInfo.actualSingleProcessSInnerSize - mmInfo.s2GmOffset :
                                  S2_BASIC_BLOCK_L0;
    } else {
        mmInfo.s2L0RealSize = lastMmInfo.s2L0RealSize;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeMm1(const QLIV2Common::RunInfo& runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + (qwL1Mte2BufIdx_ & (DOUBLE_BUF_NUM - 1)));
        QueryNd2Nz(runInfo.actMBaseSize, runInfo); // 256 * 128 // L1BasicBlock
        WeightDmaCopy(runInfo.actMBaseSize, runInfo);
    }
    int64_t loopIdx = 0;
    int64_t s2L0LoopCnt = CeilDiv(runInfo.actualSingleProcessSInnerSize, S2_BASIC_BLOCK_L0); // 2048取128
    int64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize, S1G_BASIC_BLOCK_L0);                 // 256取128
    int64_t s1gL1Offset[2] = {0, static_cast<int64_t>(S1G_BASIC_BLOCK_L0)};
    int64_t s1gL0RealSize[2] = {s1L0LoopCnt > 1 ? static_cast<int64_t>(S1G_BASIC_BLOCK_L0) : runInfo.actMBaseSize,
                                runInfo.actMBaseSize - s1gL1Offset[1]};
    MmInfo mmInfo[2];
    CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);

    // 3 块 S 流水：生产超前消费 S_L1_BUF_LAG(=2) 对，让 "S 就绪"(FIX_MTE1) 等待落在两对之前的旗标上。
    //   * 缓冲槽与旗标 id 一律用**全局对号** gIdx = globalBase + 本 call 内的对号；globalBase 取成员
    //     sL1BufIdx_（每生产一对 +1、跨 行块/s2 内循环 连续），使写侧与读侧看到同一个槽、同一个 id。
    //     若读侧改用本 call 内的局部游标，call 边界处会与写侧错位（Wait 等一个尚未置位的 id ⇒ 挂死），
    //     这正是 2026-10-15 那版 507014 的来源，见 scripts/prof_output/naie273qli/model_s_pipeline.py。
    //   * 消费所需的对信息取自快照环 ringInfo[本 call 内对号 % S_L1_BUF_NUM]（生产时写入）：mmInfo 只有
    //     2 个槽，消费第 i 对时其槽可能已被第 i+1 对覆盖，不能靠重算取得（会读到错的上一次对信息）。
    //   * 旗标计数与基线一致：每对 1 次 set（生产）+ 1 次 wait（消费）。
    const int64_t totalLoop = s2L0LoopCnt * s1L0LoopCnt;
    const int64_t earlyCnt =
        totalLoop < static_cast<int64_t>(S_L1_BUF_LAG) ? totalLoop : static_cast<int64_t>(S_L1_BUF_LAG);
    const uint64_t globalBase = sL1BufIdx_; // 本 call 首对的全局对号
    MmInfo ringInfo[S_L1_BUF_NUM];
    int64_t consumeIdx = 0; // 本 call 内下一个待消费的对号（= 已消费对数）

    for (int64_t w = 0; w < earlyCnt; ++w) { // 预热：只生产、不消费
        CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);
        ringInfo[loopIdx % S_L1_BUF_NUM] = mmInfo[loopIdx & 1];
        ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId], s1gL1Offset[mmInfo[loopIdx & 1].s1gL0LoopId],
                  s1L0LoopCnt, mmInfo[loopIdx & 1], runInfo);
        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % S_L1_BUF_NUM);
        sL1BufIdx_++;
        loopIdx++;
    }

    while (loopIdx < totalLoop) {
        CalcMmInfo(mmInfo[loopIdx & 1], loopIdx, s1L0LoopCnt, mmInfo[(loopIdx + 1) & 1], runInfo);
        ringInfo[loopIdx % S_L1_BUF_NUM] = mmInfo[loopIdx & 1];

        ProcessQk(s1gL0RealSize[mmInfo[loopIdx & 1].s1gL0LoopId], s1gL1Offset[mmInfo[loopIdx & 1].s1gL0LoopId],
                  s1L0LoopCnt, mmInfo[loopIdx & 1], runInfo);

        SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % S_L1_BUF_NUM);
        sL1BufIdx_++;

        const uint64_t gIdx = globalBase + static_cast<uint64_t>(consumeIdx);
        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + gIdx % S_L1_BUF_NUM);
        ProcessWs(s1gL0RealSize[ringInfo[consumeIdx % S_L1_BUF_NUM].s1gL0LoopId],
                  s1gL1Offset[ringInfo[consumeIdx % S_L1_BUF_NUM].s1gL0LoopId], gIdx % S_L1_BUF_NUM,
                  ringInfo[consumeIdx % S_L1_BUF_NUM], runInfo);
        consumeIdx++;
        loopIdx++;
    }

    for (int64_t t = 0; t < earlyCnt; ++t) { // 尾部排空：把预热未消费的 S 消费完
        const uint64_t gIdx = globalBase + static_cast<uint64_t>(consumeIdx);
        WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + gIdx % S_L1_BUF_NUM);
        ProcessWs(s1gL0RealSize[ringInfo[consumeIdx % S_L1_BUF_NUM].s1gL0LoopId],
                  s1gL1Offset[ringInfo[consumeIdx % S_L1_BUF_NUM].s1gL0LoopId], gIdx % S_L1_BUF_NUM,
                  ringInfo[consumeIdx % S_L1_BUF_NUM], runInfo);
        consumeIdx++;
    }

    // 常驻区（Q+W）释放：此刻本行块所有读 Q/W 的 MMAD 均已发完（M 流水保序），
    // 事件触发后下一个行块可覆盖这两个区；与行块首的 1 次 wait 一一对应。
    SetFlag<HardEvent::M_MTE1>(PIN_MTE1_EVENT);

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + (qwL1Mte2BufIdx_ & (DOUBLE_BUF_NUM - 1)));
        qwL1Mte2BufIdx_++;
    }
}

// blkNum, blkSize, N2, D
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                          const QLIV2Common::RunInfo& runInfo, uint64_t dstSlot)
{
    uint64_t s2L1Offset = 0;
    // A11: key 0 轴非连续寻址 — 块基址 = 表值 x keyStride0 (host 下发的真实 stride)。
    // keyStride0 == 0 (旧调用/非 PA 紧凑) 时兜底原紧凑公式, 现网行为 bit 级不变 (R8)
    uint64_t blkStride = constInfo_.keyStride0 != 0 ?
                             static_cast<uint64_t>(constInfo_.keyStride0) :
                             static_cast<uint64_t>(constInfo_.kCacheBlockSize) * constInfo_.kHeadNum *
                                 constInfo_.headDim;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset =
            blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) * blkStride +
            s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                              s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size; // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[dstSlot * KEY_BUFFER_OFFSET + s2L1Offset * S8_BLOCK_CUBE], keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo& mmInfo,
                                                     const QLIV2Common::RunInfo& runInfo)
{
    uint64_t dStride = constInfo_.headDim;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::BSND || K_LAYOUT_T == LI_LAYOUT::TND) {
        dStride = constInfo_.headDim * constInfo_.kHeadNum; // constInfo_.kHeadNum
    }
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = dStride;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[((keyL1BufIdx_ & (DOUBLE_BUF_NUM - 1))) * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + mmInfo.s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// batch, s1, g, 1
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::WeightDmaCopy(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo)
{
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = s1gL1RealSize;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(weightL1_[((qwL1Mte2BufIdx_ & (DOUBLE_BUF_NUM - 1))) * WEIGHT_BUFFER_OFFSET],
             weightGm_[runInfo.loop % DOUBLE_BUF_NUM * BLOCK_CUBE * constInfo_.mBaseSize], copyInParams);
}

// batch, s1, n2, g, d
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::QueryNd2Nz(uint64_t s1gL1RealSize, const QLIV2Common::RunInfo& runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize; // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[((qwL1Mte2BufIdx_ & (DOUBLE_BUF_NUM - 1))) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset], nd2nzPara);
}

// s1g, d
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                           uint64_t s1gL0RealSize, uint64_t qSlot)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE); // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                         // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;         // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = CeilAlign(s1gL0RealSize, BLOCK_CUBE); // M height维度目的
    loadData3DParams.kExtension = constInfo_.headDim;                   // K   width维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(l0a_[qSlot * L0AB_BUFFER_OFFSET_S8_16K],
                                   queryL1_[((qwL1Mte2BufIdx_ & (DOUBLE_BUF_NUM - 1))) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

// s1, g, s2  -->  2 * 64* 128
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadSToL0b(uint64_t s1gL1RealSize, uint64_t s2L0RealSize,
                                                       uint64_t sL1SlotIdx, int64_t mStartPt, uint64_t sSlot)
{
    LoadData3DParamsV2<half> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = S1G_BASIC_BLOCK_L0 / BLOCK_CUBE;             // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                                  // Win=M0
    loadData3DParams.channelSize = CeilAlign(s2L0RealSize, BLOCK_CUBE); // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = constInfo_.gSize;                    // M height维度目的
    loadData3DParams.kExtension = CeilAlign(s2L0RealSize, BLOCK_CUBE); // K   width维度目的
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 1;
    loadData3DParams.fMatrixCtrl = 0;

    loadData3DParams.mStartPt = mStartPt;
    LoadData<half, LOAD3DV2_CONFIG>(l0b_.template ReinterpretCast<half>()[sSlot * L0AB_BUFFER_OFFSET_FP16_16K],
                                    sL1_[(sL1SlotIdx % S_L1_BUF_NUM) * SL1_BUFFER_OFFSET], loadData3DParams);
}

// s1,g,1(16), 2,64,16
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadWeightToL0a(uint64_t l1ElemOffset, uint64_t l0aHalfOffset)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(constInfo_.gSize, BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(l0a_.template ReinterpretCast<half>()[l0aHalfOffset],
             weightL1_[((qwL1Mte2BufIdx_ & (DOUBLE_BUF_NUM - 1))) * WEIGHT_BUFFER_OFFSET + l1ElemOffset],
             loadData2DParams);
}

// 行块开头一次性装载本行块 4 个 s1 行的 W 到 L0A 常驻槽（每行 2KB，共 8KB）
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadAllWeightRowsToL0a()
{
    for (uint64_t j = 0; j < (S1G_BASIC_BLOCK_L0 * 2 / constInfo_.gSize); j++) {
        LoadWeightToL0a(j * constInfo_.gSize * BLOCK_CUBE, W_SLOT_HALF_BASE + j * W_ROW_STRIDE_HALF);
    }
}

// s2, d -> 128,128
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::LoadKeyToL0b(uint64_t s2L0RealSize, uint64_t kSlot)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[kSlot * L0AB_BUFFER_OFFSET_S8_16K],
             keyL1_[((keyL1BufIdx_ & (DOUBLE_BUF_NUM - 1))) * KEY_BUFFER_OFFSET], loadData2DParams);
}

// A: s1,g,1(16) B: s1,g,s2  C: s1, 1(16), s2
template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeWs(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, int64_t s1gOffset,
                                                      uint64_t wRowIdx, uint64_t sSlot)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = BLOCK_CUBE;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.gSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_.template ReinterpretCast<float>()[((l0cBufIdx_ & (DOUBLE_BUF_NUM - 1))) * L0C_BUFFER_OFFSET +
                                                s1gOffset * S2_BASIC_BLOCK_L0],
         l0a_.template ReinterpretCast<half>()[W_SLOT_HALF_BASE + wRowIdx * W_ROW_STRIDE_HALF],
         l0b_.template ReinterpretCast<half>()[sSlot * L0AB_BUFFER_OFFSET_FP16_16K], mmadParams);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, uint64_t qSlot,
                                                      uint64_t kSlot)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[((l0cBufIdx_ & (DOUBLE_BUF_NUM - 1))) * L0C_BUFFER_OFFSET], l0a_[qSlot * L0AB_BUFFER_OFFSET_S8_16K],
         l0b_[kSlot * L0AB_BUFFER_OFFSET_S8_16K], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;
    params.reluPre = 1;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);
    DataCopy(sL1_[(sL1BufIdx_ % S_L1_BUF_NUM) * SL1_BUFFER_OFFSET],
             cL0_[((l0cBufIdx_ & (DOUBLE_BUF_NUM - 1))) * L0C_BUFFER_OFFSET], params);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize,
                                                        uint64_t s1GmOffset, uint64_t s2GmOffset,
                                                        const QLIV2Common::RunInfo& runInfo)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = constInfo_.s2BaseSize;
    intriParams.srcStride = 16;
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(s1L0RealCount, CeilDiv(constInfo_.gSize, BLOCK_CUBE) * S2_BASIC_BLOCK_L0 / BLOCK_CUBE,
                                 2048);
    AscendC::DataCopy(mm1ResGm_[((runInfo.loop & 1)) * constInfo_.mBaseSize / constInfo_.gSize * constInfo_.s2BaseSize +
                                s1GmOffset * intriParams.dstStride + s2GmOffset],
                      cL0_.template ReinterpretCast<float>()[((l0cBufIdx_ & (DOUBLE_BUF_NUM - 1))) * L0C_BUFFER_OFFSET],
                      intriParams);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::M_MTE1>(PIN_MTE1_EVENT); // Q+W 常驻区守卫播种

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename QLIV2T>
__aicore__ inline void QLIV2Matmul<QLIV2T>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::M_MTE1>(PIN_MTE1_EVENT); // 与播种一一对应的回收

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}
} // namespace QLIV2Kernel
#endif // QUANT_LIGHTNING_INDEXER_V2_SERVICE_CUBE_H
