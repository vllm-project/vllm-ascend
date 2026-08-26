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
 * \file compress_norm_rope_kernel_c4.h
 * \brief Kernel A：C4（cmpRatio=4, coff=2, OVERLAP）——组流式 + 双缓冲流水
 *
 * 窗口 8 行 = [前一组 4 行 coff0 | 当前组 4 行 coff1]：
 *   - 左半（前一组 coff0）：直接复用上一组双缓冲驻留 UB 的 cur 缓冲（零拷贝、零 GM 重读）；
 *     本核首个任务的跨组前驱直接读 mm GM（只读输入，无跨核依赖）；
 *     历史部分（p < start_pos）读分页 state_cache；p < 0 填 0（kv）/ -inf（score）。
 *   - 右半（当前组 coff1）：本组已 cast/+ape 的 fp32 行（UB→UB 拼接）。
 *
 * 流水结构（每核组内，无 SyncAll、无跨核 flag、无 GM workspace）：
 *   迭代 i:  MTE2 预取组 i+1 mm ║ V 处理组 i（cast/+ape/窗口/softmax/sum/RmsNorm/RoPE/cast）
 *            ║ MTE3 写组 i state + cmp 行
 *   压缩行后处理（UB 内融合）：ColumnSum 得全 512 维 → RmsNorm(gamma 常驻)
 *            → RoPE（cos/sin 行号=compressedCnt_，作用于行尾 ropeHeadDim 维）→ CAST_RINT
 *   事件协议（每 buffer 独立 id，配平：wait(b) ⟸ 该 buffer 上一次 set，1:1；结束 drain）：
 *   - EVENT_ID0 evtMte2ToV_  : 本组 mm load → cast（每迭代自配对）；S_V 方向同号仅 Init 用一次
 *   - EVENT_ID1 evtVToMte3_  : cast/+ape → state 写 / cmp 写（每迭代自配对）
 *   - EVENT_ID2/3 idVToMTE2  : cast(b) 完成 → stage[b] 可再写（load 前按 stageUsed wait）
 *   - EVENT_ID4/5 idMTE3ToV  : save/cmpout(b) 完成 → cur[b]/outStage[b] 可再写（按 curUsed wait）
 *   - EVENT_ID6/7            : 窗口装配自配对 + rope cos/sin 加载自配对（时序上先后串行复用）
 */

#ifndef COMPRESS_NORM_ROPE_KERNEL_C4_H
#define COMPRESS_NORM_ROPE_KERNEL_C4_H

#include "compress_norm_rope_comm.h"
#include "compress_norm_rope_tiling_data.h"
#include "compress_norm_rope_tools.h"
#include "compress_norm_rope_vector_comm.h"

namespace CompressNormRope {

using optiling::CompressNormRopeTilingData;

template <typename T, typename T_NORM, typename T_ROPE>
class CompressNormRopeKernelC4 {
public:
    __aicore__ inline void Init(TPipe *pipe, const CompressNormRopeTilingData *tilingData,
                                __gm__ uint8_t *mmKv, __gm__ uint8_t *mmScore, __gm__ uint8_t *stateCache,
                                __gm__ uint8_t *ape, __gm__ uint8_t *normWeight, __gm__ uint8_t *ropeSin,
                                __gm__ uint8_t *ropeCos, __gm__ uint8_t *stateBlockTable,
                                __gm__ uint8_t *cuSeqlens, __gm__ uint8_t *seqUsed, __gm__ uint8_t *startPos,
                                __gm__ uint8_t *cmpKvOut)
    {
        pipe_ = pipe;
        batchSize_ = tilingData->batchSize;
        headDim_ = tilingData->headDim;
        cmpRatio_ = tilingData->cmpRatio;
        blockSize_ = tilingData->blockSize;
        maxBlockNumPerBatch_ = tilingData->maxBlockNumPerBatch;
        stateStride0_ = tilingData->stateCacheStrideDim0;
        mmKvStride0_ = tilingData->mmKvStrideDim0;
        mmScoreStride0_ = tilingData->mmScoreStrideDim0;
        ropeHeadDim_ = tilingData->ropeHeadDim;
        rotaryMode_ = tilingData->rotaryMode;
        normEps_ = tilingData->normEps;
        rowLen_ = COFF_NUM * headDim_;   // mm 行元素数（coff0|coff1）
        stateRowLen_ = 2 * rowLen_;      // state 行元素数（kv|score）

        uint32_t coreIdx = GetBlockIdx();
        taskStart_ = coreIdx * tilingData->taskPerCore +
                     (coreIdx < tilingData->taskRem ? coreIdx : tilingData->taskRem);
        taskEnd_ = taskStart_ + tilingData->taskPerCore + (coreIdx < tilingData->taskRem ? 1 : 0);

        mmKvGm_.SetGlobalBuffer((__gm__ T *)mmKv);
        mmScoreGm_.SetGlobalBuffer((__gm__ T *)mmScore);
        stateGm_.SetGlobalBuffer((__gm__ float *)stateCache);
        apeGm_.SetGlobalBuffer((__gm__ float *)ape);
        normWeightGm_.SetGlobalBuffer((__gm__ T_NORM *)normWeight);
        ropeSinGm_.SetGlobalBuffer((__gm__ T_ROPE *)ropeSin);
        ropeCosGm_.SetGlobalBuffer((__gm__ T_ROPE *)ropeCos);
        sbtGm_.SetGlobalBuffer((__gm__ int32_t *)stateBlockTable);
        cmpKvOutGm_.SetGlobalBuffer((__gm__ T *)cmpKvOut);

        iter_.Init(cuSeqlens, seqUsed, startPos, batchSize_, cmpRatio_);

        pipe_->InitBuffer(apeBuf_, cmpRatio_ * rowLen_ * sizeof(float));
        pipe_->InitBuffer(kvStageBuf0_, cmpRatio_ * rowLen_ * sizeof(T));
        pipe_->InitBuffer(kvStageBuf1_, cmpRatio_ * rowLen_ * sizeof(T));
        pipe_->InitBuffer(scoreStageBuf0_, cmpRatio_ * rowLen_ * sizeof(T));
        pipe_->InitBuffer(scoreStageBuf1_, cmpRatio_ * rowLen_ * sizeof(T));
        pipe_->InitBuffer(kvCurBuf0_, cmpRatio_ * rowLen_ * sizeof(float));
        pipe_->InitBuffer(kvCurBuf1_, cmpRatio_ * rowLen_ * sizeof(float));
        pipe_->InitBuffer(scoreCurBuf0_, cmpRatio_ * rowLen_ * sizeof(float));
        pipe_->InitBuffer(scoreCurBuf1_, cmpRatio_ * rowLen_ * sizeof(float));
        pipe_->InitBuffer(kvWinBuf_, COFF_NUM * cmpRatio_ * headDim_ * sizeof(float));
        pipe_->InitBuffer(scoreWinBuf_, COFF_NUM * cmpRatio_ * headDim_ * sizeof(float));
        pipe_->InitBuffer(tmpBuf_, COFF_NUM * cmpRatio_ * headDim_ * sizeof(float));
        pipe_->InitBuffer(cmpAccumBuf_, NORM_BATCH * headDim_ * sizeof(float));
        pipe_->InitBuffer(outStageBuf0_, NORM_BATCH * headDim_ * sizeof(T));
        pipe_->InitBuffer(outStageBuf1_, NORM_BATCH * headDim_ * sizeof(T));
        pipe_->InitBuffer(kvLeftBuf_, cmpRatio_ * headDim_ * sizeof(T));
        pipe_->InitBuffer(scoreLeftBuf_, cmpRatio_ * headDim_ * sizeof(T));
        pipe_->InitBuffer(gammaBuf_, headDim_ * sizeof(float));
        pipe_->InitBuffer(ropeStageBuf_, 3 * NORM_BATCH * ropeHeadDim_ * sizeof(float));
        pipe_->InitBuffer(gatherOffsetBuf_, ropeHeadDim_ * sizeof(int32_t));

        evtMte2ToV_ = static_cast<event_t>(EVENT_ID0);
        evtVToMte3_ = static_cast<event_t>(EVENT_ID1);
        eventIdVToMTE2_[0] = static_cast<event_t>(EVENT_ID2);
        eventIdVToMTE2_[1] = static_cast<event_t>(EVENT_ID3);
        eventIdMTE3ToV_[0] = static_cast<event_t>(EVENT_ID4);
        eventIdMTE3ToV_[1] = static_cast<event_t>(EVENT_ID5);
        evtVToMte2Win_ = static_cast<event_t>(EVENT_ID6);
        evtMte2ToVWin_ = static_cast<event_t>(EVENT_ID7);
        evtSToV_ = static_cast<event_t>(EVENT_ID0); // S_V 方向与 MTE2_V 物理 flag 独立，仅 Init 用一次
    }

    // gamma 常驻 + INTERLEAVE 偏移表（启动一次性；复用 evtMte2ToV_ 顺序自配对）
    __aicore__ inline void InitNormRope()
    {
        LocalTensor<float> gammaUb = gammaBuf_.Get<float>();
        if constexpr (std::is_same_v<T_NORM, float>) {
            DataCopy(gammaUb, normWeightGm_, headDim_);
            SetFlag<HardEvent::MTE2_V>(evtMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2ToV_);
        } else {
            // bf16/fp16 输入：原始数据 copy 到 gamma fp32 buffer 后半段（T_NORM view 偏移
            // headDim_，字节 1024 起 = 后半段），一次 Cast 到前半段 fp32（源/目标不重叠）
            LocalTensor<T_NORM> gammaRawUb = gammaBuf_.Get<T_NORM>();
            DataCopy(gammaRawUb[headDim_], normWeightGm_, headDim_);
            SetFlag<HardEvent::MTE2_V>(evtMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(evtMte2ToV_);
            Cast(gammaUb, gammaRawUb[headDim_], RoundMode::CAST_NONE, headDim_);
        }
        if (rotaryMode_ == (uint32_t)ROTARY_MODE::INTERLEAVE) {
            LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf_.Get<int32_t>();
            SetGatherSrcOffset<float>(gatherOffsetUb, ropeHeadDim_, evtSToV_);
        }
    }

    __aicore__ inline void Process()
    {
        if (taskStart_ >= taskEnd_) {
            return;
        }
        compressedCnt_ = iter_.Locate(taskStart_);

        LocalTensor<float> apeUb = apeBuf_.Get<float>();
        DataCopy(apeUb, apeGm_, cmpRatio_ * rowLen_);
        SetFlag<HardEvent::MTE2_V>(evtMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToV_);
        InitNormRope();

        LocalTensor<T> kvStage[2] = {kvStageBuf0_.Get<T>(), kvStageBuf1_.Get<T>()};
        LocalTensor<T> scoreStage[2] = {scoreStageBuf0_.Get<T>(), scoreStageBuf1_.Get<T>()};
        LocalTensor<float> kvCur[2] = {kvCurBuf0_.Get<float>(), kvCurBuf1_.Get<float>()};
        LocalTensor<float> scoreCur[2] = {scoreCurBuf0_.Get<float>(), scoreCurBuf1_.Get<float>()};
        LocalTensor<T> outStage[2] = {outStageBuf0_.Get<T>(), outStageBuf1_.Get<T>()};

        // prime：每 ping-pong id 先 Set 一次（与前两次迭代的 wait 配对，并配平末尾 drain）
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2_[0]);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2_[1]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[0]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[1]);

        uint32_t buf = 0;
        uint32_t prevBIdx = batchSize_; // 上一个处理组的 bIdx（无效初值 → 首组 prevContinues=false）
        for (uint32_t task = taskStart_; task < taskEnd_; task++) {
            GroupInfo g;
            if (!FetchGroup(g)) {
                break;
            }
            uint32_t nbuf = buf ^ 1;
            // 上一组与本组同 batch 相邻 → 窗口左半可复用其 cur 缓冲（kvCur[nbuf]）
            bool prevContinues = (prevBIdx == g.bIdx) && (g.nTok > 0);
            if (g.nTok > 0) {
                // ── 1. 等 stage[buf] 可写（cast(g_{i-2}) 完成；前两次迭代由 prime 配对）──
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2_[buf]);
                LoadMm(g, kvStage[buf], scoreStage[buf]);
                // ── 2. 等 cur[buf]/outStage[buf] 可写（save/cmpout(g_{i-2}) 完成）──
                WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[buf]);
                // ── 3. 等本组 mm load 完成，cast fp32；score 按组内位置加 ape ──
                SetFlag<HardEvent::MTE2_V>(evtMte2ToV_);
                WaitFlag<HardEvent::MTE2_V>(evtMte2ToV_);
                Cast(kvCur[buf], kvStage[buf], RoundMode::CAST_NONE, g.nTok * rowLen_);
                Cast(scoreCur[buf], scoreStage[buf], RoundMode::CAST_NONE, g.nTok * rowLen_);
                PipeBarrier<PIPE_V>();
                AddApe(scoreCur[buf], g, apeUb);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2_[buf]); // stage[buf] 可再写
                // ── 4. 写 state：kv 存 raw fp32，score 存 +ape fp32 ──
                SetFlag<HardEvent::V_MTE3>(evtVToMte3_);
                WaitFlag<HardEvent::V_MTE3>(evtVToMte3_);
                SaveState(kvCur[buf], g, 0);
                SaveState(scoreCur[buf], g, 1);
                // ── 5. 窗口装配 + 逐列 softmax + 加权求和（直写行累积槽）──
                if (g.produce) {
                    AssembleWindow(g, kvCur[buf], scoreCur[buf], kvCur[nbuf], scoreCur[nbuf], prevContinues,
                                   apeUb);
                    LocalTensor<float> kvWin = kvWinBuf_.Get<float>();
                    LocalTensor<float> scoreWin = scoreWinBuf_.Get<float>();
                    LocalTensor<float> tmpUb = tmpBuf_.Get<float>();
                    LocalTensor<float> cmpAccum = cmpAccumBuf_.Get<float>();
                    ColumnSoftMax(scoreWin, scoreWin, tmpUb, COFF_NUM * cmpRatio_, headDim_);
                    PipeBarrier<PIPE_V>();
                    Mul(kvWin, kvWin, scoreWin, COFF_NUM * cmpRatio_ * headDim_);
                    PipeBarrier<PIPE_V>();
                    // 压缩行 fp32 直接落累积槽；RmsNorm+RoPE 按 NORM_BATCH 行批量做（摊销 barrier）
                    ColumnSum(cmpAccum[accSlots_ * headDim_], kvWin, tmpUb, COFF_NUM * cmpRatio_, headDim_);
                    PipeBarrier<PIPE_V>();
                    accSlots_++;
                    compressedCnt_++;
                    if (accSlots_ == NORM_BATCH) {
                        FlushNormRope(outStage[buf], compressedCnt_ - NORM_BATCH, NORM_BATCH);
                        accSlots_ = 0;
                    }
                }
                // 本组 MTE3 读取（save + cmpout）已全部发射，供复用该 buffer 的 iter i+2 等待
                SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[buf]);
            }
            prevBIdx = g.bIdx;
            buf = nbuf;
        }
        // 尾部不足一批的行：flush（先等同 buf 上一任务 cmpout MTE3 完成再重写 outStage）
        if (accSlots_ > 0) {
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[buf]);
            FlushNormRope(outStage[buf], compressedCnt_ - accSlots_, accSlots_);
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[buf]); // 供 drain 消费，配平 1:1
            accSlots_ = 0;
        }
        // drain：与 prime 配对，消费末尾未消费的 set（全局 Set:Wait 严格 1:1）
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2_[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2_[1]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[0]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV_[1]);
    }

private:
    static constexpr uint32_t COFF_NUM = 2;
    static constexpr uint32_t NORM_BATCH = 4; // RmsNorm/RoPE 批量行数（摊销 barrier；UB 受限）

    // 批量 RmsNorm + RoPE + cast + 写出：行 firstRow..firstRow+cnt-1（GM 行号连续）
    // 契约：调用点已保证 outStage 可写（任务顶部 id4/5 wait 或尾部 flush 的显式 wait）
    __aicore__ inline void FlushNormRope(const LocalTensor<T> &outStage, uint32_t firstRow, uint32_t cnt)
    {
        LocalTensor<float> cmpAccum = cmpAccumBuf_.Get<float>();
        LocalTensor<float> tmpUb = tmpBuf_.Get<float>();
        LocalTensor<float> gammaUb = gammaBuf_.Get<float>();
        LocalTensor<float> cosUb = ropeStageBuf_.Get<float>();
        LocalTensor<float> sinUb = cosUb[NORM_BATCH * ropeHeadDim_];
        LocalTensor<uint32_t> gatherOff = gatherOffsetBuf_.Get<uint32_t>();
        // cos/sin 连续行一次载入；ropeStage 复用等上一轮 rope V 完成
        SetFlag<HardEvent::V_MTE2>(evtVToMte2Win_);
        WaitFlag<HardEvent::V_MTE2>(evtVToMte2Win_);
        uint64_t ropeOff = (uint64_t)firstRow * ropeHeadDim_;
        uint32_t cntRope = cnt * ropeHeadDim_;
        if constexpr (std::is_same_v<T_ROPE, float>) {
            DataCopy(cosUb, ropeCosGm_[ropeOff], cntRope);
            DataCopy(sinUb, ropeSinGm_[ropeOff], cntRope);
        } else {
            // bf16/fp16 输入：cos/sin 原始数据 copy 到 buffer 后两块（T_ROPE view 偏移
            // 4/5 * NORM_BATCH*ropeHeadDim_），各一次 Cast 到前两块 fp32（源/目标不重叠）
            LocalTensor<T_ROPE> ropeRawUb = ropeStageBuf_.Get<T_ROPE>();
            DataCopy(ropeRawUb[4 * NORM_BATCH * ropeHeadDim_], ropeCosGm_[ropeOff], cntRope);
            DataCopy(ropeRawUb[5 * NORM_BATCH * ropeHeadDim_], ropeSinGm_[ropeOff], cntRope);
        }
        SetFlag<HardEvent::MTE2_V>(evtMte2ToVWin_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToVWin_);
        if constexpr (!std::is_same_v<T_ROPE, float>) {
            LocalTensor<T_ROPE> ropeRawUb = ropeStageBuf_.Get<T_ROPE>();
            Cast(cosUb, ropeRawUb[4 * NORM_BATCH * ropeHeadDim_], RoundMode::CAST_NONE, cntRope);
            Cast(sinUb, ropeRawUb[5 * NORM_BATCH * ropeHeadDim_], RoundMode::CAST_NONE, cntRope);
        }
        RmsNormParam normParams{1.0f / (float)(int32_t)headDim_, normEps_, cnt, headDim_};
        RmsNorm(cmpAccum, cmpAccum, gammaUb, tmpUb, normParams);
        PipeBarrier<PIPE_V>();
        uint64_t baseAddr = headDim_ - ropeHeadDim_;
        if (rotaryMode_ == (uint32_t)ROTARY_MODE::INTERLEAVE) {
            RotaryPosEmb<ROTARY_MODE::INTERLEAVE>(cmpAccum, cmpAccum, cosUb, sinUb, tmpUb, gatherOff, cnt,
                                                  ropeHeadDim_, headDim_, baseAddr);
        } else {
            RotaryPosEmb<ROTARY_MODE::HALF>(cmpAccum, cmpAccum, cosUb, sinUb, tmpUb, gatherOff, cnt,
                                            ropeHeadDim_, headDim_, baseAddr);
        }
        PipeBarrier<PIPE_V>();
        Cast(outStage, cmpAccum, RoundMode::CAST_RINT, cnt * headDim_);
        SetFlag<HardEvent::V_MTE3>(evtVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(evtVToMte3_);
        DataCopy(cmpKvOutGm_[(uint64_t)firstRow * headDim_], outStage, cnt * headDim_);
    }

    __aicore__ inline bool FetchGroup(GroupInfo &info)
    {
        if (iter_.IsEnd()) {
            return false;
        }
        iter_.GetGroupInfo(info);
        iter_.Next();
        return true;
    }

    __aicore__ inline void LoadMm(const GroupInfo &g, const LocalTensor<T> &kvStage,
                                  const LocalTensor<T> &scoreStage)
    {
        uint64_t kvSrcOff = (uint64_t)g.xRow * mmKvStride0_;
        uint64_t scoreSrcOff = (uint64_t)g.xRow * mmScoreStride0_;
        DataCopyAlignGmToUb(kvStage, mmKvGm_[kvSrcOff], g.nTok, rowLen_, mmKvStride0_, rowLen_);
        DataCopyAlignGmToUb(scoreStage, mmScoreGm_[scoreSrcOff], g.nTok, rowLen_, mmScoreStride0_, rowLen_);
    }

    __aicore__ inline void AddApe(const LocalTensor<float> &scoreCur, const GroupInfo &g,
                                  const LocalTensor<float> &apeUb)
    {
        uint32_t apeRow0 = g.tokStart % cmpRatio_;
        if (apeRow0 == 0 && g.nTok == cmpRatio_) {
            Add(scoreCur, scoreCur, apeUb, g.nTok * rowLen_);
        } else {
            for (uint32_t i = 0; i < g.nTok; i++) {
                uint32_t apeRow = (apeRow0 + i) % cmpRatio_;
                Add(scoreCur[i * rowLen_], scoreCur[i * rowLen_], apeUb[apeRow * rowLen_], rowLen_);
            }
        }
    }

    // 窗口装配：win rows [0,r) = 左半（位置 gStart-r..gStart-1，coff0）
    //           win rows [r,2r) = 右半（位置 gStart..gStart+r-1，coff1）
    __aicore__ inline void AssembleWindow(const GroupInfo &g, const LocalTensor<float> &kvCur,
                                          const LocalTensor<float> &scoreCur, const LocalTensor<float> &kvPrevCur,
                                          const LocalTensor<float> &scorePrevCur, bool prevContinues,
                                          const LocalTensor<float> &apeUb)
    {
        LocalTensor<float> kvWin = kvWinBuf_.Get<float>();
        LocalTensor<float> scoreWin = scoreWinBuf_.Get<float>();
        uint32_t r = cmpRatio_;
        uint32_t gStart = g.gStart;
        uint32_t P = g.P;

        // ── 左半 ──
        int64_t leftFrom = (int64_t)gStart - (int64_t)r;
        uint32_t pos = leftFrom < 0 ? 0 : (uint32_t)leftFrom;
        uint32_t row = 0;
        if (leftFrom < 0) {
            // p < 0：kv 填 0、score 填 -inf（softmax 后贡献为 0）
            uint32_t cntFill = (uint32_t)(-leftFrom);
            Duplicate(kvWin, FLOAT_ZERO, cntFill * headDim_);
            Duplicate(scoreWin, SOFTMAX_MIN_VALUE, cntFill * headDim_);
            PipeBarrier<PIPE_V>();
            row = cntFill;
        }
        uint32_t stateEndL = gStart < P ? gStart : P; // 历史行部分（p < P 且 p >= 0）
        uint32_t cntPrev = gStart - (pos < stateEndL ? stateEndL : pos);
        // state 历史读（MTE2 写 win）需等上一轮 V 读完 win
        if (pos < stateEndL || g.headHolder > 0) {
            SetFlag<HardEvent::V_MTE2>(evtVToMte2Win_);
            WaitFlag<HardEvent::V_MTE2>(evtVToMte2Win_);
        }
        if (pos < stateEndL) {
            ReadStateRows(kvWin[row * headDim_], stateEndL - pos, pos, 0, 0, g);
            ReadStateRows(scoreWin[row * headDim_], stateEndL - pos, pos, 0, 1, g);
            row += stateEndL - pos;
            pos = stateEndL;
        }
        if (cntPrev > 0) {
            if (prevContinues) {
                // 左半 = 上一组 cur 的 coff0 半区（零拷贝复用双缓冲）
                DataCopyAlignUbToUb(kvWin[row * headDim_], kvPrevCur, cntPrev, headDim_, rowLen_, headDim_);
                DataCopyAlignUbToUb(scoreWin[row * headDim_], scorePrevCur, cntPrev, headDim_, rowLen_, headDim_);
            } else {
                // 本核首个任务的跨组前驱：直接读 mm GM（raw，score 补 ape）
                LoadMmLeftHalf(kvWin[row * headDim_], scoreWin[row * headDim_], pos, cntPrev, g, apeUb);
            }
            row += cntPrev;
        }

        // ── 右半 ──
        uint32_t rrow = r;
        if (g.headHolder > 0) {
            ReadStateRows(kvWin[rrow * headDim_], g.headHolder, gStart, 1, 0, g);
            ReadStateRows(scoreWin[rrow * headDim_], g.headHolder, gStart, 1, 1, g);
            rrow += g.headHolder;
        }
        DataCopyAlignUbToUb(kvWin[rrow * headDim_], kvCur[headDim_], g.nTok, headDim_, rowLen_, headDim_);
        DataCopyAlignUbToUb(scoreWin[rrow * headDim_], scoreCur[headDim_], g.nTok, headDim_, rowLen_, headDim_);

        // state 读（MTE2）→ 后续 V 计算
        SetFlag<HardEvent::MTE2_V>(evtMte2ToVWin_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToVWin_);
        PipeBarrier<PIPE_V>();
    }

    // 从 mm GM 读左半前驱行（bf16 raw → fp32；score 按位置补 ape 的 coff0 半区）
    __aicore__ inline void LoadMmLeftHalf(const LocalTensor<float> &kvDst, const LocalTensor<float> &scoreDst,
                                          uint32_t posStart, uint32_t rowCnt, const GroupInfo &g,
                                          const LocalTensor<float> &apeUb)
    {
        LocalTensor<T> kvLeft = kvLeftBuf_.Get<T>();
        LocalTensor<T> scoreLeft = scoreLeftBuf_.Get<T>();
        // 专用 scratch 的上一个读者是 V（上一次 cast），self-sync 后即可让 MTE2 写入
        SetFlag<HardEvent::V_MTE2>(evtVToMte2Win_);
        WaitFlag<HardEvent::V_MTE2>(evtVToMte2Win_);
        uint32_t xRowL = g.xRow - (g.tokStart - posStart); // = cu[b] + posStart - P
        DataCopyAlignGmToUb(kvLeft, mmKvGm_[(uint64_t)xRowL * mmKvStride0_], rowCnt, headDim_,
                            mmKvStride0_, headDim_);
        DataCopyAlignGmToUb(scoreLeft, mmScoreGm_[(uint64_t)xRowL * mmScoreStride0_], rowCnt, headDim_,
                            mmScoreStride0_, headDim_);
        SetFlag<HardEvent::MTE2_V>(evtMte2ToVWin_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToVWin_);
        Cast(kvDst, kvLeft, RoundMode::CAST_NONE, rowCnt * headDim_);
        Cast(scoreDst, scoreLeft, RoundMode::CAST_NONE, rowCnt * headDim_);
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < rowCnt; j++) {
            uint32_t apeRow = (posStart + j) % cmpRatio_;
            Add(scoreDst[j * headDim_], scoreDst[j * headDim_], apeUb[apeRow * rowLen_], headDim_);
        }
        PipeBarrier<PIPE_V>();
    }

    // 从分页 state 读历史行（fp32，coffHalf: 0=coff0, 1=coff1；stateIdx: 0=kv, 1=score）
    __aicore__ inline void ReadStateRows(const LocalTensor<float> &dst, uint32_t rowCnt, uint32_t posStart,
                                         uint32_t coffHalf, uint32_t stateIdx, const GroupInfo &g)
    {
        uint32_t p = posStart;
        uint32_t done = 0;
        while (done < rowCnt) {
            uint32_t blk = p / blockSize_;
            uint32_t rowInBlk = p % blockSize_;
            int32_t blockId = sbtGm_.GetValue(g.bIdx * maxBlockNumPerBatch_ + blk);
            uint32_t seg = blockSize_ - rowInBlk;
            if (done + seg > rowCnt) {
                seg = rowCnt - done;
            }
            uint64_t srcOff = (uint64_t)blockId * stateStride0_ + rowInBlk * stateRowLen_ + stateIdx * rowLen_ +
                              coffHalf * headDim_;
            DataCopyAlignGmToUb(dst[done * headDim_], stateGm_[srcOff], seg, headDim_, stateRowLen_, headDim_);
            done += seg;
            p += seg;
        }
    }

    // 写本组 token 到分页 state（fp32 全行 rowLen 列；blockId==0 跳过）
    __aicore__ inline void SaveState(const LocalTensor<float> &src, const GroupInfo &g, uint32_t stateIdx)
    {
        uint32_t p = g.tokStart;
        uint32_t done = 0;
        while (done < g.nTok) {
            uint32_t blk = p / blockSize_;
            uint32_t rowInBlk = p % blockSize_;
            int32_t blockId = sbtGm_.GetValue(g.bIdx * maxBlockNumPerBatch_ + blk);
            uint32_t seg = blockSize_ - rowInBlk;
            if (done + seg > g.nTok) {
                seg = g.nTok - done;
            }
            if (blockId != 0) {
                uint64_t dstOff =
                    (uint64_t)blockId * stateStride0_ + rowInBlk * stateRowLen_ + stateIdx * rowLen_;
                DataCopyAlignUbToGm(stateGm_[dstOff], src[done * rowLen_], seg, rowLen_, rowLen_, stateRowLen_);
            }
            done += seg;
            p += seg;
        }
    }

    TPipe *pipe_ = nullptr;
    GroupIterator iter_;

    GlobalTensor<T> mmKvGm_;
    GlobalTensor<T> mmScoreGm_;
    GlobalTensor<float> stateGm_;
    GlobalTensor<float> apeGm_;
    GlobalTensor<T_NORM> normWeightGm_;
    GlobalTensor<T_ROPE> ropeSinGm_;
    GlobalTensor<T_ROPE> ropeCosGm_;
    GlobalTensor<int32_t> sbtGm_;
    GlobalTensor<T> cmpKvOutGm_;

    TBuf<TPosition::VECCALC> apeBuf_;
    TBuf<TPosition::VECCALC> kvStageBuf0_;
    TBuf<TPosition::VECCALC> kvStageBuf1_;
    TBuf<TPosition::VECCALC> scoreStageBuf0_;
    TBuf<TPosition::VECCALC> scoreStageBuf1_;
    TBuf<TPosition::VECCALC> kvCurBuf0_;
    TBuf<TPosition::VECCALC> kvCurBuf1_;
    TBuf<TPosition::VECCALC> scoreCurBuf0_;
    TBuf<TPosition::VECCALC> scoreCurBuf1_;
    TBuf<TPosition::VECCALC> kvWinBuf_;
    TBuf<TPosition::VECCALC> scoreWinBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> cmpAccumBuf_;
    TBuf<TPosition::VECCALC> outStageBuf0_;
    TBuf<TPosition::VECCALC> outStageBuf1_;
    TBuf<TPosition::VECCALC> kvLeftBuf_;
    TBuf<TPosition::VECCALC> scoreLeftBuf_;
    TBuf<TPosition::VECCALC> gammaBuf_;
    TBuf<TPosition::VECCALC> ropeStageBuf_;
    TBuf<TPosition::VECCALC> gatherOffsetBuf_;

    event_t evtMte2ToV_;
    event_t evtVToMte3_;
    event_t eventIdVToMTE2_[2];
    event_t eventIdMTE3ToV_[2];
    event_t evtVToMte2Win_;
    event_t evtMte2ToVWin_;
    event_t evtSToV_;

    uint32_t batchSize_ = 0;
    uint32_t headDim_ = 0;
    uint32_t cmpRatio_ = 0;
    uint32_t blockSize_ = 0;
    uint32_t maxBlockNumPerBatch_ = 0;
    uint64_t stateStride0_ = 0;
    uint32_t mmKvStride0_ = 0;
    uint32_t mmScoreStride0_ = 0;
    uint32_t rowLen_ = 0;
    uint32_t stateRowLen_ = 0;
    uint32_t taskStart_ = 0;
    uint32_t taskEnd_ = 0;
    uint32_t compressedCnt_ = 0;
    uint32_t accSlots_ = 0;
    uint32_t ropeHeadDim_ = 0;
    uint32_t rotaryMode_ = 0;
    float normEps_ = 1e-6f;
};

} // namespace CompressNormRope

#endif // COMPRESS_NORM_ROPE_KERNEL_C4_H
