/**
 * Stage M: build row_map + token_batch in workspace.
 *
 * 对齐 CUDA k2q_build_row_map_kernel + Host token_batch（prepare_k2q_meta）。
 */
#ifndef K2Q_CSR_STAGE_META_H
#define K2Q_CSR_STAGE_META_H

#include "k2q_csr_common.h"
#include "k2q_csr_tiling.h"
#include "k2q_csr_scratch.h"

using namespace AscendC;

class K2qCsrStageMeta {
public:
    __aicore__ inline K2qCsrStageMeta() {}

    __aicore__ inline void Init(GM_ADDR cuQ, GM_ADDR cuB, GM_ADDR rowMap, GM_ADDR tokenBatch, GM_ADDR histScratch,
                                const K2qCsrTilingData *tiling, TPipe *pipe);

    __aicore__ inline void Process();

private:
    __aicore__ inline void BuildTokenBatch(LocalTensor<int32_t> &cuQLocal);
    __aicore__ inline void BuildRowMapConcat(LocalTensor<int32_t> &cuBLocal);
    __aicore__ inline void BuildRowMapRoundRobin(LocalTensor<int32_t> &cuBLocal);
    __aicore__ inline void ZeroRowCounts();

    const K2qCsrTilingData *tiling_;
    TPipe *pipe_;

    GlobalTensor<int32_t> cuQGm_;
    GlobalTensor<int32_t> cuBGm_;
    GlobalTensor<int32_t> rowMapGm_;
    GlobalTensor<int32_t> tokenBatchGm_;
    GlobalTensor<int32_t> rowCountsGm_;
    GM_ADDR rowMapAddr_;

    int64_t B_;
    int64_t T_;
    int64_t H_;
    int64_t totalRows_;
    int64_t maxKv_;
    int64_t rowMapElems_;
    int64_t rowCountElems_;
    int32_t orderMethod_;

    TBuf<TPosition::VECCALC> cuQBuf_;
    TBuf<TPosition::VECCALC> cuBBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;
};

__aicore__ inline void K2qCsrStageMeta::Init(GM_ADDR cuQ, GM_ADDR cuB, GM_ADDR rowMap, GM_ADDR tokenBatch,
                                             GM_ADDR histScratch, const K2qCsrTilingData *tiling, TPipe *pipe)
{
    tiling_ = tiling;
    pipe_ = pipe;
    B_ = tiling->B;
    T_ = tiling->T;
    H_ = tiling->H;
    totalRows_ = tiling->totalRows;
    maxKv_ = tiling->maxKv;
    rowMapElems_ = tiling->rowMapElems;
    orderMethod_ = tiling->orderMethod;
    rowMapAddr_ = rowMap;
    rowCountElems_ = H_ * (totalRows_ > 0 ? totalRows_ : 1);
    if (rowCountElems_ < 1) {
        rowCountElems_ = 1;
    }

    cuQGm_.SetGlobalBuffer((__gm__ int32_t *)cuQ, B_ + 1);
    cuBGm_.SetGlobalBuffer((__gm__ int32_t *)cuB, B_ + 1);
    rowMapGm_.SetGlobalBuffer((__gm__ int32_t *)rowMap, rowMapElems_ > 0 ? rowMapElems_ : 1);
    tokenBatchGm_.SetGlobalBuffer((__gm__ int32_t *)tokenBatch, T_ > 0 ? T_ : 1);
    rowCountsGm_.SetGlobalBuffer((__gm__ int32_t *)K2qCsrScratch::RowCountsAddr(histScratch, tiling), rowCountElems_);

    pipe_->InitBuffer(cuQBuf_, K2qCsrCommon::Align32((B_ + 1) * sizeof(int32_t)));
    pipe_->InitBuffer(cuBBuf_, K2qCsrCommon::Align32((B_ + 1) * sizeof(int32_t)));
    constexpr int64_t kChunk = 2048;
    pipe_->InitBuffer(chunkBuf_, K2qCsrCommon::Align32(kChunk * sizeof(int32_t)));
}

__aicore__ inline void K2qCsrStageMeta::BuildTokenBatch(LocalTensor<int32_t> &cuQLocal)
{
    constexpr int64_t kChunk = 2048;
    LocalTensor<int32_t> chunk = chunkBuf_.Get<int32_t>();
    for (int64_t b = 0; b < B_; ++b) {
        int32_t start = cuQLocal.GetValue(b);
        int32_t end = cuQLocal.GetValue(b + 1);
        for (int32_t t = start; t < end;) {
            int64_t n = end - t;
            if (n > kChunk) {
                n = kChunk;
            }
            // 向量填常量 batch id，避免逐元素 SetValue（T 大时显著降 Meta 时延）
            Duplicate(chunk, static_cast<int32_t>(b), static_cast<int32_t>(n));
            PipeBarrier<PIPE_ALL>();
            K2qCsrCommon::CopyOutInt32(tokenBatchGm_[t], chunk, n);
            t += static_cast<int32_t>(n);
        }
    }
}

__aicore__ inline void K2qCsrStageMeta::BuildRowMapConcat(LocalTensor<int32_t> &cuBLocal)
{
    constexpr int64_t kChunk = 2048;
    LocalTensor<int32_t> chunk = chunkBuf_.Get<int32_t>();
    int64_t filled = 0;
    int64_t gmOff = 0;
    for (int64_t b = 0; b < B_; ++b) {
        int32_t rb = cuBLocal.GetValue(b + 1) - cuBLocal.GetValue(b);
        int32_t base = cuBLocal.GetValue(b);
        for (int64_t level = 0; level < maxKv_; ++level) {
            int32_t v = (level < rb) ? (base + static_cast<int32_t>(level)) : K2qCsrCommon::INVALID;
            chunk.SetValue(filled, v);
            ++filled;
            if (filled == kChunk) {
                K2qCsrCommon::CopyOutInt32(rowMapGm_[gmOff], chunk, kChunk);
                gmOff += kChunk;
                filled = 0;
            }
        }
    }
    if (filled > 0) {
        K2qCsrCommon::CopyOutInt32(rowMapGm_[gmOff], chunk, filled);
    }
}

__aicore__ inline void K2qCsrStageMeta::BuildRowMapRoundRobin(LocalTensor<int32_t> &cuBLocal)
{
    // 对齐 CUDA: level 上 rows_before = Σ min(rb, level)，再按 batch 赋 active
    // 写出顺序与 [B, max_kv] 行主序一致
    constexpr int64_t kChunk = 2048;
    LocalTensor<int32_t> chunk = chunkBuf_.Get<int32_t>();
    int64_t filled = 0;
    int64_t gmOff = 0;
    for (int64_t b = 0; b < B_; ++b) {
        for (int64_t level = 0; level < maxKv_; ++level) {
            int32_t rowsBefore = 0;
            for (int64_t bb = 0; bb < B_; ++bb) {
                int32_t rb = cuBLocal.GetValue(bb + 1) - cuBLocal.GetValue(bb);
                rowsBefore += (rb < static_cast<int32_t>(level)) ? rb : static_cast<int32_t>(level);
            }
            int32_t active = 0;
            int32_t v = K2qCsrCommon::INVALID;
            for (int64_t bb = 0; bb < B_; ++bb) {
                int32_t rb = cuBLocal.GetValue(bb + 1) - cuBLocal.GetValue(bb);
                if (rb > static_cast<int32_t>(level)) {
                    if (bb == b) {
                        v = rowsBefore + active;
                        break;
                    }
                    ++active;
                }
            }
            chunk.SetValue(filled, v);
            ++filled;
            if (filled == kChunk) {
                K2qCsrCommon::CopyOutInt32(rowMapGm_[gmOff], chunk, kChunk);
                gmOff += kChunk;
                filled = 0;
            }
        }
    }
    if (filled > 0) {
        K2qCsrCommon::CopyOutInt32(rowMapGm_[gmOff], chunk, filled);
    }
}

__aicore__ inline void K2qCsrStageMeta::ZeroRowCounts()
{
    // 对齐 CUDA torch::zeros(row_counts)：供后续 Hist AtomicAdd 累加
    if (H_ <= 0 || rowCountElems_ <= 0) {
        return;
    }
    LocalTensor<int32_t> chunk = chunkBuf_.Get<int32_t>();
    constexpr int64_t kChunk = 2048;
    int64_t chunkN = kChunk < rowCountElems_ ? kChunk : rowCountElems_;
    Duplicate(chunk, static_cast<int32_t>(0), static_cast<int32_t>(chunkN));
    PipeBarrier<PIPE_ALL>();
    for (int64_t off = 0; off < rowCountElems_; off += chunkN) {
        int64_t n = chunkN;
        if (off + n > rowCountElems_) {
            n = rowCountElems_ - off;
        }
        K2qCsrCommon::CopyOutInt32(rowCountsGm_[off], chunk, n);
    }
}

__aicore__ inline void K2qCsrStageMeta::Process()
{
    if (B_ <= 0) {
        ZeroRowCounts();
        return;
    }

    LocalTensor<int32_t> cuQLocal = cuQBuf_.Get<int32_t>();
    LocalTensor<int32_t> cuBLocal = cuBBuf_.Get<int32_t>();
    K2qCsrCommon::CopyInInt32(cuQLocal, cuQGm_, B_ + 1);
    K2qCsrCommon::CopyInInt32(cuBLocal, cuBGm_, B_ + 1);
    PipeBarrier<PIPE_ALL>();

    // 设备侧自算 max_kv（算法真值）；Host attr 仅用于分配，二者应对齐
    int32_t maxKvDev = 0;
    for (int64_t b = 0; b < B_; ++b) {
        int32_t d = cuBLocal.GetValue(b + 1) - cuBLocal.GetValue(b);
        if (d > maxKvDev) {
            maxKvDev = d;
        }
    }
    maxKv_ = maxKvDev;
    rowMapElems_ = B_ * maxKv_;
    rowMapGm_.SetGlobalBuffer((__gm__ int32_t *)rowMapAddr_, rowMapElems_ > 0 ? rowMapElems_ : 1);

    if (T_ > 0) {
        BuildTokenBatch(cuQLocal);
    }
    if (rowMapElems_ > 0) {
        if (orderMethod_ == 0) {
            BuildRowMapConcat(cuBLocal);
        } else {
            BuildRowMapRoundRobin(cuBLocal);
        }
    }
    PipeBarrier<PIPE_ALL>();
    ZeroRowCounts();
}

#endif
