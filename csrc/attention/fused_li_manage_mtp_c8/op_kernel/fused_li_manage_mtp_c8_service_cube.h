/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_li_manage_mtp_c8_service_cube.h
 * \brief C8 cube service: int8 QK Mmad followed by the on-chip fp16 head
 * reduction Mmad (quant_lightning_indexer arch22 two-mma structure, ported
 * from the in-repo fused_li_manage_c8 template). Per-head scores stay on
 * chip (L0C int32 -> fixpipe DEQF16 x2^-10 + relu -> L1 fp16 -> mma2); only
 * the reduced fp32 score rows [s1, s2] are written to GM. The MTP gS1 block
 * (up to s1BaseSize=8 query rows x gSize heads) walks S1G_BASIC_BLOCK_L0
 * (128 s1g) L0 tiles like lightning_indexer_quant: one mma1 covers the
 * whole tile, ProcessWs loops the s1 rows of the tile, and one multi-row
 * nz2nd fixpipe writes all of them (repeat = 128/gSize per tile).
 */
#ifndef FUSED_LI_MANAGE_MTP_C8_SERVICE_CUBE_H
#define FUSED_LI_MANAGE_MTP_C8_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_li_manage_mtp_c8_common.h"

namespace LIMtpC8Kernel {
using namespace LIMtpC8Common;

template <typename LIT>
class LIMatmulMtpC8 {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    // mma2 output (and the GM score workspace) is fp32.
    using MM1_OUT_T = float;

    __aicore__ inline LIMatmulMtpC8(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<MM1_OUT_T> &mm1ResGm,
                                               const GlobalTensor<half> &weightWorkspaceGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LIMtpC8Common::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t L0_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    // FIX_M guards the L0C double buffer shared by mma1 (int32) and mma2
    // (fp32 reinterpret); M_FIX orders each Mmad before its Fixp drain (no
    // hardware scoreboard for M->FIX on L0C here). FIX_MTE1 publishes the
    // fp16 per-head scores in L1 to the mma2 operand load. Mirrors
    // quant_lightning_indexer (arch22) / fused_li_manage_c8.
    // Note: FIX_M sharing EVENT_ID2 with MTE2_MTE1/MTE1_M/KEY_MTE1_MTE2 is
    // intentional (template and arch22 do the same) — cross-pipe event IDs
    // are scoped per pipe-pair, not pooled globally (verified: moving FIX_M
    // to a free ID changed nothing).
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_MTE1_EVENT = EVENT_ID4;
    // MTE1_FIX closes the sL1 slot recycle loop: FIX_MTE1 only publishes a
    // written slot to LoadSToL0b; without a read->rewrite edge the next
    // FixpSToL1 (issued one flattened-iteration later) can overwrite the slot
    // while the deferred ProcessWs is still loading its later row groups
    // from L1.  Each Ws sets it after the row-group loads; FixpSToL1 waits
    // it before overwriting the same parity slot.
    static constexpr uint32_t MTE1_FIX_EVENT = EVENT_ID1;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S2_BASIC_BLOCK = 256;

    // int8 NZ fractal: C0 = 32 elements (32B / 1B); the N dim stays 16.
    // Mirrors S8_BLOCK_CUBE in the reference lightning_indexer_quant kernel.
    static constexpr uint64_t S8_BLOCK_CUBE = 32;

    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;
    // sL1 tile pitch: one fp16 per-head score tile [S1G_BASIC_BLOCK_L0, S2_BASIC_BLOCK_L0],
    // double buffered. S1G_BASIC_BLOCK_L0 is also the L0 tile size (lightning_
    // indexer_quant shape): 128 s1g = S1G_BASIC_BLOCK_L0/gSize s1 rows per tile
    // (4 rows @ H=32, 2 @ H=64); each row's mma2 C区 sits gSize*S2_BASIC_BLOCK_L0
    // elements apart so the rows exactly tile one L0C slot. The 2026-09-07 P2
    // change reverted the §7.2 dump-motivated 1-row clamp: that dump channel
    // itself races (§7.2), and lightning_indexer_quant runs the multi-row nz2nd
    // fixpipe in production (B/F topk cross-check 100%, §8.6 of the report).
    static constexpr uint64_t S1G_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t SL1_BUFFER_OFFSET = S1G_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    // V1 诊断开关: false = 逐行 repeat=1 fixpipe (隔离多 repeat 机制)。
    // 定论(2026-09-07): 多行失效根因是 nz2nd dstNdStride 误抄 F 的字面量 2048 —
    // 该参是"repeat 间 GM 步进(元素)"，必须等于本题 chunk 缓冲行距 s2BaseSize(2048);
    // 2048 时 repeat 落到行 0/4/8/12, 行1-3 从未被写(AIV 读到零/陈旧), 且第1个
    // repeat 溢出到奇偶对半区(表现为 row0 偶发污染)。V1 逐行形态因 repeat=1
    // 不步进而恰好正确, 用于隔离验证; 生产用多行形态(与 F 一致, 每 tile 1 条 fixpipe)。
    static constexpr bool P2_MULTIROW_NZ2ND = true;
    // Uniform 16KB L0A/L0B slots (QLI layout): int8 uses 16K elements, fp16 8K fp16 elements.
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_FP16_16K = 16 * 512;
    static constexpr uint64_t L0C_BUFFER_OFFSET = S1G_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void ProcessQk(uint64_t s1gL1Offset, uint64_t s1gL0RealSize, uint64_t s2L1Offset,
                                     uint64_t s2L0RealSize, bool syncKeyMte2,
                                     const LIMtpC8Common::RunInfo &runInfo);
    __aicore__ inline void ProcessWs(uint64_t s2GmOffset, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                     uint64_t s1gL1Offset, uint64_t s1gL0RealSize,
                                     const LIMtpC8Common::RunInfo &runInfo);
    __aicore__ inline void FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize, uint64_t s1GmOffset,
                                       uint64_t s2GmOffset, const LIMtpC8Common::RunInfo &runInfo,
                                       uint64_t rowIdx = 0);
    __aicore__ inline void ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void ComputeWs(uint64_t s2L0RealSize, uint64_t s1IdxInTile);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L0RealSize);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0RealSize, uint64_t actMBaseSize);
    __aicore__ inline void LoadSToL0b(uint64_t s2L0RealSize, uint64_t sL1BufIdx, uint64_t s1gOffset);
    __aicore__ inline void LoadWeightToL0a(uint64_t s1gWeightOffset);
    __aicore__ inline void QueryNd2Nz(const LIMtpC8Common::RunInfo &runInfo);
    __aicore__ inline void WeightDmaCopy(const LIMtpC8Common::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                         const LIMtpC8Common::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<MM1_OUT_T> mm1ResGm_;
    GlobalTensor<half> weightGm_;

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
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;
    uint64_t sL1BufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::InitBuffers(TPipe *pipe)
{
    // Single NZ buffer holding the whole gS1 block (actMBaseSize <= mBaseSize
    // = 8*gSize <= 512 rows -> <= 64KB for H=64).
    uint64_t queryL1Elems =
        LIMtpC8Common::Align(static_cast<uint64_t>(constInfo_.mBaseSize), static_cast<uint64_t>(BLOCK_CUBE)) *
        D_BASIC_BLOCK;
    pipe->InitBuffer(bufQL1_, queryL1Elems * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * KEY_BUFFER_OFFSET * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();
    // Brcb'd (w * q_scale) operand: mBaseSize blocks of BLOCK_CUBE fp16 elements.
    pipe->InitBuffer(bufWeightL1_, constInfo_.mBaseSize * BLOCK_CUBE * sizeof(half));
    weightL1_ = bufWeightL1_.Get<half>();
    pipe->InitBuffer(bufSL1_, 2 * SL1_BUFFER_OFFSET * sizeof(half));
    sL1_ = bufSL1_.Get<half>();

    pipe->InitBuffer(bufL0A_, L0AB_BUF_NUM * L0AB_BUFFER_OFFSET_S8_16K);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, L0AB_BUF_NUM * L0AB_BUFFER_OFFSET_S8_16K);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * L0C_BUFFER_OFFSET * sizeof(int32_t));
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename LIT>
__aicore__ inline void
LIMatmulMtpC8<LIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                        const GlobalTensor<Q_T> &queryGm, const GlobalTensor<MM1_OUT_T> &mm1ResGm,
                                        const GlobalTensor<half> &weightWorkspaceGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
    weightGm_ = weightWorkspaceGm;
}

// Flattened tile walk: s2 outer (S2_BASIC_BLOCK_L0 columns per L0 tile, key
// L1 double buffered across 256-row L1 tiles), s1g inner (S1G_BASIC_BLOCK_L0
// s1g per L0 tile). QK(iteration i) interleaves with WS(iteration i-1) through
// the FIX_MTE1 sL1 double buffer, exactly like fused_li_manage_c8 ComputeMm1.
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::ComputeMm1(const LIMtpC8Common::RunInfo &runInfo)
{
    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    uint64_t tileCnt = LIMtpC8Common::CeilDiv(s2ProcessSize, S2_BASIC_BLOCK_L0);
    uint64_t s1gL0TileSize = S1G_BASIC_BLOCK_L0;
    uint64_t s1L0LoopCnt = LIMtpC8Common::CeilDiv(static_cast<uint64_t>(runInfo.actMBaseSize), s1gL0TileSize);

    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);
        // In-order MTE2 pipe: the first even tile's MTE2_MTE1 pair below also
        // covers these query/weight loads (same pattern as QLI arch22).
        QueryNd2Nz(runInfo);
        WeightDmaCopy(runInfo);
    }

    bool firstIter = true;
    uint64_t prevS2GmOffset = 0;
    uint64_t prevS2L0RealSize = 0;
    uint64_t prevS1gL1Offset = 0;
    uint64_t prevS1gL0RealSize = 0;
    for (uint64_t tileIdx = 0; tileIdx < tileCnt; ++tileIdx) {
        uint64_t s2L1Offset = tileIdx % 2 * S2_BASIC_BLOCK_L0;
        uint64_t s2L1RealSize = 0;
        if (tileIdx % 2 == 0) {
            uint64_t l1TileStart = tileIdx * S2_BASIC_BLOCK_L0;
            s2L1RealSize = l1TileStart + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - l1TileStart
                                                                        : S2_BASIC_BLOCK;
            WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + l1TileStart, runInfo);
            SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        } else {
            uint64_t l1TileStart = (tileIdx - 1) * S2_BASIC_BLOCK_L0;
            s2L1RealSize = l1TileStart + S2_BASIC_BLOCK > s2ProcessSize ? s2ProcessSize - l1TileStart
                                                                        : S2_BASIC_BLOCK;
        }
        (void)s2L1RealSize;
        uint64_t s2L0RealSize = tileIdx * S2_BASIC_BLOCK_L0 + S2_BASIC_BLOCK_L0 > s2ProcessSize
                                    ? s2ProcessSize - tileIdx * S2_BASIC_BLOCK_L0
                                    : S2_BASIC_BLOCK_L0;
        for (uint64_t s1gL0LoopId = 0; s1gL0LoopId < s1L0LoopCnt; ++s1gL0LoopId) {
            uint64_t s1gL0RealSize =
                s1gL0LoopId * s1gL0TileSize + s1gL0TileSize > runInfo.actMBaseSize
                    ? runInfo.actMBaseSize - s1gL0LoopId * s1gL0TileSize
                    : s1gL0TileSize;
            uint64_t s1gL1Offset = s1gL0LoopId * s1gL0TileSize;
            // The MTE2_MTE1 pair is set once per L1 tile (even tileIdx); only
            // its first s1g consumer waits it.
            ProcessQk(s1gL1Offset, s1gL0RealSize, s2L1Offset, s2L0RealSize,
                      tileIdx % 2 == 0 && s1gL0LoopId == 0, runInfo);
            SetFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % 2);
            sL1BufIdx_++;
            if (!firstIter) {
                // Post-increment parity: two increments since the pending
                // tile was published, so sL1BufIdx_ % 2 selects it.
                WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + sL1BufIdx_ % 2);
                ProcessWs(prevS2GmOffset, prevS2L0RealSize, sL1BufIdx_, prevS1gL1Offset, prevS1gL0RealSize,
                          runInfo);
            }
            firstIter = false;
            prevS2GmOffset = tileIdx * S2_BASIC_BLOCK_L0;
            prevS2L0RealSize = s2L0RealSize;
            prevS1gL1Offset = s1gL1Offset;
            prevS1gL0RealSize = s1gL0RealSize;
        }
        if (tileIdx % 2 == 1 || tileIdx + 1 == tileCnt) {
            SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
            keyL1BufIdx_++;
        }
    }

    WaitFlag<HardEvent::FIX_MTE1>(FIX_MTE1_EVENT + (sL1BufIdx_ + 1) % 2);
    ProcessWs(prevS2GmOffset, prevS2L0RealSize, sL1BufIdx_ - 1, prevS1gL1Offset, prevS1gL0RealSize, runInfo);

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);
    }
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::ProcessQk(uint64_t s1gL1Offset, uint64_t s1gL0RealSize,
                                                     uint64_t s2L1Offset, uint64_t s2L0RealSize, bool syncKeyMte2,
                                                     const LIMtpC8Common::RunInfo &runInfo)
{
    (void)runInfo;
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    if (syncKeyMte2) {
        // One Set/Wait pair per L1 tile; also covers the segment-start
        // query/weight MTE2 via pipe order.
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    }
    LoadQueryToL0a(s1gL1Offset, s1gL0RealSize, runInfo.actMBaseSize);
    LoadKeyToL0b(s2L1Offset, s2L0RealSize);
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    ComuteL0c(s1gL0RealSize, s2L0RealSize);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    FixpSToL1(s1gL0RealSize, s2L0RealSize);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    l0BufIdx_++;
    l0cBufIdx_++;
}

// Head reduction for one (s1g tile, s2 tile): one mma2 per s1 row of the
// tile (arch22 per-s1 ProcessWs loop), then one fixpipe GM row per s1.
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::ProcessWs(uint64_t s2GmOffset, uint64_t s2L0RealSize, uint64_t sL1BufIdx,
                                                     uint64_t s1gL1Offset, uint64_t s1gL0RealSize,
                                                     const LIMtpC8Common::RunInfo &runInfo)
{
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    uint64_t gSize = constInfo_.gSize;
    for (uint64_t s1gOffset = 0; s1gOffset < s1gL0RealSize; s1gOffset += gSize) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
        LoadSToL0b(s2L0RealSize, sL1BufIdx, s1gOffset);
        LoadWeightToL0a(s1gL1Offset + s1gOffset);
        ComputeWs(s2L0RealSize, s1gOffset / gSize);
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
        l0BufIdx_++;
    }
    // Both group L0B loads of this sL1 slot have been issued; the flag fires
    // once the MTE1 queue drains, releasing the slot for the next FixpSToL1.
    SetFlag<HardEvent::MTE1_FIX>(MTE1_FIX_EVENT + sL1BufIdx % 2);
    if (P2_MULTIROW_NZ2ND) {
        // 每 ProcessWs 一条多行 nz2nd fixpipe (repeat = tile 内 s1 行数 =
        // 128/gSize)，与 lightning_indexer_quant 生产形态一致。
        FixpResToGm(s1gL0RealSize / gSize, s2L0RealSize, s1gL1Offset / gSize, s2GmOffset, runInfo);
    } else {
        // V1 诊断: 逐行 repeat=1 nz2nd (与 §7 旧形态同调用量级), 源指针逐行
        // 步进到该行 mma2 C 区 — 用于隔离 "多 repeat 机制" vs "mma2 源数据"。
        for (uint64_t r = 0; r < s1gL0RealSize / gSize; ++r) {
            FixpResToGm(1, s2L0RealSize, s1gL1Offset / gSize + r, s2GmOffset, runInfo, r);
        }
    }
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % L0_BUF_NUM);
    l0cBufIdx_++;
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                         const LIMtpC8Common::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset =
            static_cast<uint64_t>(blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId)) *
                constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
            s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = (s2L1RealSize <= S2_BASIC_BLOCK_L0 || s2L1Offset >= S2_BASIC_BLOCK_L0) ?
                                  s2L1RealSize - s2L1Offset :
                                  S2_BASIC_BLOCK_L0 - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ?
                         constInfo_.kCacheBlockSize - s2BlkOffset :
                         s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                                      CeilAlign(s2L1RealSize - S2_BASIC_BLOCK_L0, (uint64_t)BLOCK_CUBE) :
                                      (s2L1RealSize > S2_BASIC_BLOCK_L0 ?
                                           S2_BASIC_BLOCK_L0 :
                                           CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE));
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET +
                        (s2L1Offset >= S2_BASIC_BLOCK_L0 ?
                             S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 + (s2L1Offset - S2_BASIC_BLOCK_L0) * S8_BLOCK_CUBE :
                             s2L1Offset * S8_BLOCK_CUBE)],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

// TND query: the whole gS1 block (actMBaseSize = actS1Size * gSize rows)
// lands in one NZ L1 buffer.
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::QueryNd2Nz(const LIMtpC8Common::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = runInfo.actMBaseSize;
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(runInfo.actMBaseSize, (uint64_t)BLOCK_CUBE);
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1_, queryGm_[runInfo.tensorQueryOffset], nd2nzPara);
}

// Brcb'd (w * q_scale) fp16 operand prepared by the vector side (vec0):
// actMBaseSize blocks of BLOCK_CUBE identical fp16 elements, one per-core buffer.
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::WeightDmaCopy(const LIMtpC8Common::RunInfo &runInfo)
{
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = runInfo.actMBaseSize; // actMBaseSize blocks of 32B (16 fp16 elements)
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(weightL1_, weightGm_[0], copyInParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL0RealSize,
                                                          uint64_t actMBaseSize)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    loadData3DParams.l1H = CeilDiv(actMBaseSize, static_cast<uint64_t>(BLOCK_CUBE));
    loadData3DParams.l1W = BLOCK_CUBE;
    loadData3DParams.channelSize = constInfo_.headDim;

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;

    loadData3DParams.mExtension = s1gL0RealSize;
    loadData3DParams.kExtension = constInfo_.headDim;
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

    LoadData<Q_T, LOAD3DV2_CONFIG>(
        l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
        queryL1_, loadData3DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::LoadKeyToL0b(uint64_t s2L1Offset, uint64_t s2L0RealSize)
{
    uint64_t keyL1Offset = s2L1Offset >= S2_BASIC_BLOCK_L0 ? S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 : 0;
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + keyL1Offset], loadData2DParams);
}

// fp16 per-head scores [s1g rows, s2] in L1 -> transposed B operand
// [k=gSize heads, n=s2] for one s1 of the tile. Mirrors QLI arch22
// LoadSToL0b with mStartPt selecting the s1 row inside the tile.
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::LoadSToL0b(uint64_t s2L0RealSize, uint64_t sL1BufIdx, uint64_t s1gOffset)
{
    LoadData3DParamsV2<half> loadData3DParams;
    loadData3DParams.l1H = S1G_BASIC_BLOCK_L0 / BLOCK_CUBE;              // 128 rows in 16-row fractals
    loadData3DParams.l1W = BLOCK_CUBE;
    loadData3DParams.channelSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);  // Cin = s2

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    loadData3DParams.mExtension = constInfo_.gSize;                      // M height = head (k of mma2)
    loadData3DParams.kExtension = CeilAlign(s2L0RealSize, BLOCK_CUBE);   // K width = s2 (n of mma2)
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

    loadData3DParams.mStartPt = s1gOffset;
    LoadData<half, LOAD3DV2_CONFIG>(
        l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
        sL1_[(sL1BufIdx % 2) * SL1_BUFFER_OFFSET], loadData3DParams);
}

// A operand for the head reduction of one s1: [m=BLOCK_CUBE, k=gSize] where
// every m row holds the same w*q_scale vector for that (s1, g) block
// (Brcb'd by the vector side, transposed here into NZ fractals). Mirrors QLI
// arch22 LoadWeightToL0a; the source block index is the global s1g row.
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::LoadWeightToL0a(uint64_t s1gWeightOffset)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(constInfo_.gSize, static_cast<uint64_t>(BLOCK_CUBE));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = true;
    LoadData(l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
             weightL1_[s1gWeightOffset * BLOCK_CUBE], loadData2DParams);
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, static_cast<uint64_t>(BLOCK_CUBE));
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    // No unitFlag: the int8 unit-flag accumulate path produced 4x k-dim
    // over-accumulation on A3; the reference quant kernel uses plain Mmad.
    Mmad(cL0_[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET],
         l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
         l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

// Head reduction for one s1: [m=BLOCK_CUBE, k=gSize] x [k=gSize, n=s2] ->
// fp32 L0C row placed at s1IdxInTile * gSize * S2_BASIC_BLOCK_L0 (the
// lightning_indexer_quant s1gOffset * S2_BASIC_BLOCK_L0 layout: rows are
// gSize * S2_BASIC_BLOCK_L0 elements apart, 128/gSize rows per L0C slot).
// Only m row 0 carries the reduced score (all rows are identical by
// construction of the broadcast A operand).
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::ComputeWs(uint64_t s2L0RealSize, uint64_t s1IdxInTile)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = BLOCK_CUBE;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.gSize;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET +
                                                s1IdxInTile * constInfo_.gSize * S2_BASIC_BLOCK_L0],
         l0a_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
         l0b_.template ReinterpretCast<half>()[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_FP16_16K],
         mmadParams);
}

// int32 per-head scores -> fp16 in L1. DEQF16 applies the 2^-10 range
// normalization (SetFixpipePreQuantFlag, float bits 0x3a800000) so the values
// stay inside fp16 range; the uniform positive factor does not change top-k
// order. relu matches the reference fixpipe point (pre-reduction).
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::FixpSToL1(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    // The deferred ProcessWs of two iterations ago may still be draining this
    // parity slot's L1 reads; hold the FIX overwrite until MTE1 is empty.
    WaitFlag<HardEvent::MTE1_FIX>(MTE1_FIX_EVENT + sL1BufIdx_ % 2);
    DataCopyCO12DstParams params;
    params.mSize = CeilAlign(s1gL0RealSize, static_cast<uint64_t>(BLOCK_CUBE));
    params.nSize = CeilAlign(s2L0RealSize, BLOCK_CUBE);
    params.dstStride = S1G_BASIC_BLOCK_L0;
    params.srcStride = params.mSize;
    params.quantPre = QuantMode_t::DEQF16;
    params.reluPre = 1;
    params.channelSplit = 0;
    params.nz2ndEn = 0;
    SetFixpipePreQuantFlag(0x3a800000);
    DataCopy(sL1_[(sL1BufIdx_ % 2) * SL1_BUFFER_OFFSET],
             cL0_[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET], params);
}

// Reduced fp32 score rows -> GM: 每 tile 一条多行 nz2nd fixpipe (repeat =
// 行数, 源=L0C buffer 基址, 逐 repeat 按 blockNum*256 步进到下一行 C 区),
// 落到每核 [s1BaseSize, s2BaseSize] chunk 缓冲 (loop 奇偶双缓冲)。参数组
// 照抄 lightning_indexer_quant FixpResToGm (生产多行形态)。
template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::FixpResToGm(uint64_t s1L0RealCount, uint64_t s2L0RealSize,
                                                       uint64_t s1GmOffset, uint64_t s2GmOffset,
                                                       const LIMtpC8Common::RunInfo &runInfo, uint64_t rowIdx)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = constInfo_.s2BaseSize;
    intriParams.srcStride = BLOCK_CUBE;
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.reluPre = 0;
    // nz2nd 三参 = (ndNum, srcNdStride, dstNdStride)(ND_PARA 寄存器, 见 dav_c220
    // kernel_operator_fixpipe_impl.h): ndNum = 行数; srcNdStride = repeat 间 L0C 源
    // 步进, 单位 256 元素 fractal, CeilDiv(gSize,16)*128/16 使每 repeat 恰好步进
    // 一行 C 区 (gSize*128 元素, H32: 16 fractals=4096 元素), 与 ComputeWs 的
    // s1IdxInTile*gSize*S2 布局严格互逆; dstNdStride = repeat 间 GM 步进(元素),
    // 必须等于本题 chunk 缓冲行距 s2BaseSize=2048 (F 传 2048 是因为其 s2BaseSize
    // 恰为 2048 — 不可按字面量抄)。
    AscendC::SetFixpipeNz2ndFlag(
        s1L0RealCount,
        CeilDiv(constInfo_.gSize, static_cast<uint64_t>(BLOCK_CUBE)) * S2_BASIC_BLOCK_L0 /
            static_cast<uint64_t>(BLOCK_CUBE),
        constInfo_.s2BaseSize);
    AscendC::DataCopy(
        mm1ResGm_[(runInfo.loop % 2) * constInfo_.s1BaseSize * constInfo_.s2BaseSize +
                  s1GmOffset * constInfo_.s2BaseSize + s2GmOffset],
        cL0_.template ReinterpretCast<float>()[(l0cBufIdx_ % L0_BUF_NUM) * L0C_BUFFER_OFFSET +
                                               rowIdx * constInfo_.gSize * S2_BASIC_BLOCK_L0],
        intriParams);
    // 防线(必须保留, 2026-09-07 实证): 强制 FIX 队列 drain。移除对照实验
    // (精度全绿) 但墙钟 +6.5us (B5_C12288 209.9→216.4/217.0, E/F 锚点不动,
    // pipe 计时全平)——syncC1V1 挂在 PIPE_FIX 上, 该 drain 客观上使跨核 flag
    // 触发节拍提前; F 无此 barrier 是因其握手节拍不同, 不可照抄。
    PipeBarrier<PIPE_FIX>();
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::AllocEventID()
{
    // No SetMMLayoutTransform: the int8 quant LI kernels (v1/v2) run the MM in
    // the default layout mode; transform mode misreads int8 NZ fractals (C0=32).
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);

    // One credit per sL1 parity slot: the first FixpSToL1 of each slot has no
    // preceding ProcessWs to wait for.
    SetFlag<HardEvent::MTE1_FIX>(MTE1_FIX_EVENT + 0);
    SetFlag<HardEvent::MTE1_FIX>(MTE1_FIX_EVENT + 1);
}

template <typename LIT>
__aicore__ inline void LIMatmulMtpC8<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);

    WaitFlag<HardEvent::MTE1_FIX>(MTE1_FIX_EVENT + 0);
    WaitFlag<HardEvent::MTE1_FIX>(MTE1_FIX_EVENT + 1);
}

} // namespace LIMtpC8Kernel
#endif // FUSED_LI_MANAGE_MTP_C8_SERVICE_CUBE_H
