// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
// A paired AIV computes max and sum over up to three adjacent logical pages.
// Shared row maxima allow the AIC to accumulate all page contributions in L0C.
class GroupSoftmax {
    static constexpr uint32_t kRows = 16, kColumns = 128, kPageLogits = kRows * kColumns;
    static constexpr float kNegativeInfinity = -__builtin_inff();
    template <HardEvent event> __aicore__ inline void fence() {
        SetFlag<event>(EVENT_ID0);
        WaitFlag<event>(EVENT_ID0);
    }
    GlobalTensor<float> score_, stats_;
    GlobalTensor<bfloat16_t> weights_;
    TQue<QuePosition::VECIN, 1> scoreQueue_;
    TQue<QuePosition::VECOUT, 1> weightQueue_, statsQueue_;
    TBuf<QuePosition::VECCALC> rowReduceBuffer_, broadcastBuffer_, rowStatsBuffer_;

    TQue<QuePosition::VECOUT, 1> mapQueue_;

  public:
    __aicore__ inline void InitMap(TPipe &pipe) { pipe.InitBuffer(mapQueue_, 1, 32); }
    __aicore__ inline void WriteMap(GlobalTensor<int32_t> &map, uint32_t tile, const int32_t *rows) {
        auto record = mapQueue_.AllocTensor<int32_t>();
        fence<HardEvent::V_S>();
        for (uint32_t j = 0; j < 4; ++j)
            record.SetValue(j, rows[j]);
        fence<HardEvent::S_V>();
        mapQueue_.EnQue(record);
        record = mapQueue_.DeQue<int32_t>();
        DataCopy(map[static_cast<uint64_t>(tile) * 8], record, 8);
        mapQueue_.FreeTensor(record);
    }
    __aicore__ inline void Init(TPipe &p, GM_ADDR scores, GM_ADDR weights, GM_ADDR stats, uint32_t group) {
        score_.SetGlobalBuffer((__gm__ float *)scores);
        weights_.SetGlobalBuffer((__gm__ bfloat16_t *)weights);
        stats_.SetGlobalBuffer((__gm__ float *)stats);
        p.InitBuffer(scoreQueue_, 1, group * kPageLogits * 4);
        p.InitBuffer(weightQueue_, 1, group * kPageLogits * 2);
        p.InitBuffer(statsQueue_, 1, 2 * kRows * 4);
        p.InitBuffer(rowReduceBuffer_, kRows * 64 * 4);
        p.InitBuffer(broadcastBuffer_, kRows * 8 * 4);
        p.InitBuffer(rowStatsBuffer_, (group + 1) * kRows * 4);
    }
    __aicore__ inline void Run(uint64_t pageSlot, uint32_t first, uint32_t count, uint64_t outSlot,
                               uint32_t kvLength, uint32_t physicalQuery, uint32_t activeQuery, float scale,
                               bool causal, uint32_t firstInvalidToken) {
        auto src = scoreQueue_.AllocTensor<float>();
        DataCopy(src, score_[pageSlot * kPageLogits], count * kPageLogits);
        scoreQueue_.EnQue(src);
        src = scoreQueue_.DeQue<float>();
        Muls(src, src, scale, count * kPageLogits);
        PipeBarrier<PIPE_V>();
        auto info = statsQueue_.AllocTensor<float>();
        auto tmp = rowStatsBuffer_.Get<float>();
        auto fold = rowReduceBuffer_.Get<float>();
        auto broad = broadcastBuffer_.Get<float>();
        const BinaryRepeatParams interlace(1, 1, 1, 8, 16, 16);
        for (uint32_t p = 0; p < count; p++) {
            const uint32_t page = (first + p) * kColumns;
            auto x = src[p * kPageLogits];
            if (static_cast<uint64_t>(page) + kColumns + (causal && activeQuery ? activeQuery - 1 : 0) >
                kvLength) {
                fence<HardEvent::V_S>();
                for (uint32_t row = 0; row < kRows; row++) {
                    int64_t end = static_cast<int64_t>(kvLength) -
                                  (causal ? static_cast<int64_t>(activeQuery) - row / 4 - 1 : 0);
                    int64_t active = (end < kvLength ? end : kvLength) - page;
                    uint32_t begin = active <= 0 ? 0 : active < kColumns ? active : kColumns;
                    for (uint32_t col = begin; col < kColumns; col++)
                        x.SetValue(row * kColumns + col, kNegativeInfinity);
                }
                fence<HardEvent::S_V>();
            }
            Max(fold, x, x[64], 64, kRows, interlace);
            PipeBarrier<PIPE_V>();
            WholeReduceMax(tmp[p * kRows], fold, 64, kRows, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
            PipeBarrier<PIPE_V>();
        }
        Adds(info, tmp, 0.f, kRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t p = 1; p < count; p++) {
            Max(info, info, tmp[p * kRows], kRows);
            PipeBarrier<PIPE_V>();
        }
        Maxs(tmp, info, -__FLT_MAX__, kRows);
        PipeBarrier<PIPE_V>();
        Brcb(broad.ReinterpretCast<uint32_t>(), tmp.ReinterpretCast<uint32_t>(), 2, {1, 8});
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams sub(1, 1, 0, 16, 16, 1);
        auto weights = weightQueue_.AllocTensor<bfloat16_t>();
        for (uint32_t p = 0; p < count; p++) {
            auto x = src[p * kPageLogits];
            Sub(x, x, broad, 64, kRows, sub);
            Sub(x[64], x[64], broad, 64, kRows, sub);
            PipeBarrier<PIPE_V>();
            Exp(x, x, kPageLogits);
            PipeBarrier<PIPE_V>();
            Add(fold, x, x[64], 64, kRows, interlace);
            Cast(weights[p * kPageLogits], x, RoundMode::CAST_RINT, kPageLogits);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(tmp[p * kRows], fold, 64, kRows, 1, 1, 8);
            PipeBarrier<PIPE_V>();
        }
        Adds(info[kRows], tmp, 0.f, kRows);
        PipeBarrier<PIPE_V>();
        for (uint32_t p = 1; p < count; p++) {
            Add(info[kRows], info[kRows], tmp[p * kRows], kRows);
            PipeBarrier<PIPE_V>();
        }
        if (firstInvalidToken < kvLength || activeQuery < physicalQuery) {
            fence<HardEvent::V_S>();
            for (uint32_t r = 0; r < kRows; r++) {
                int64_t end = static_cast<int64_t>(kvLength) -
                              (causal ? static_cast<int64_t>(activeQuery) - r / 4 - 1 : 0);
                if (r / 4 >= activeQuery || end > firstInvalidToken)
                    info.SetValue(kRows + r, -1.f);
            }
            fence<HardEvent::S_V>();
        }
        weightQueue_.EnQue(weights);
        statsQueue_.EnQue(info);
        weights = weightQueue_.DeQue<bfloat16_t>();
        info = statsQueue_.DeQue<float>();
        DataCopy(weights_[pageSlot * kPageLogits], weights, count * kPageLogits);
        DataCopy(stats_[outSlot * 2 * kRows], info, 2 * kRows);
        weightQueue_.FreeTensor(weights);
        statsQueue_.FreeTensor(info);
        scoreQueue_.FreeTensor(src);
    }
};
