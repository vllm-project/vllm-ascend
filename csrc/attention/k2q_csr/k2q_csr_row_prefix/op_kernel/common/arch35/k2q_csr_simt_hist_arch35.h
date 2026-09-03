/**
 * SIMT Stage H（对齐 CUDA k2q_hist_kernel）。
 * 每核 = CTA，独占 q_per_cta；VF AtomicAdd → tile_counts；再 AtomicAddOut → row_counts。
 */
#ifndef K2Q_CSR_SIMT_HIST_ARCH35_H
#define K2Q_CSR_SIMT_HIST_ARCH35_H

#include "k2q_csr_simt_common_arch35.h"
#include "../k2q_csr_common.h"
#include "../k2q_csr_tiling.h"
#include "../k2q_csr_scratch.h"

using namespace AscendC;

namespace k2q_csr_simt {

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void HistEdgesVf(
    __gm__ int32_t *q2k, __gm__ int32_t *tokenBatch, __gm__ int32_t *rowMap, __gm__ int32_t *hist, int32_t hid,
    int32_t N, int32_t topk, int32_t q0, int32_t qLen, int32_t maxKv, int32_t totalRows, int32_t B)
{
    const uint64_t numEdges = static_cast<uint64_t>(qLen) * static_cast<uint64_t>(topk);
    for (uint64_t e = static_cast<uint64_t>(Simt::GetThreadIdx()); e < numEdges;
         e += static_cast<uint64_t>(Simt::GetThreadNum())) {
        const int32_t qi = static_cast<int32_t>(e / static_cast<uint64_t>(topk));
        const int32_t s = static_cast<int32_t>(e - static_cast<uint64_t>(qi) * static_cast<uint64_t>(topk));
        const int32_t qAbs = q0 + qi;
        const int32_t bi = tokenBatch[qAbs];
        if (bi < 0 || bi >= B) {
            continue;
        }
        const int32_t val = q2k[hid * N + qAbs * topk + s];
        if (val < 0 || val >= maxKv) {
            continue;
        }
        const int32_t row = rowMap[bi * maxKv + val];
        if (row >= 0 && row < totalRows) {
            (void)Simt::AtomicAdd(hist + row, static_cast<int32_t>(1));
        }
    }
}

} // namespace k2q_csr_simt

class K2qCsrSimtHist {
public:
    __aicore__ inline K2qCsrSimtHist() {}

    __aicore__ inline void Init(GM_ADDR q2k, GM_ADDR rowMap, GM_ADDR tokenBatch, GM_ADDR scratch,
                                const K2qCsrTilingData *tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        q2kAddr_ = q2k;
        rowMapAddr_ = rowMap;
        tokenBatchAddr_ = tokenBatch;
        H_ = tiling->H;
        T_ = tiling->T;
        topk_ = tiling->topk;
        N_ = tiling->N;
        B_ = tiling->B;
        totalRows_ = tiling->totalRows;
        maxKv_ = tiling->maxKv;
        G_ = tiling->numGroups > 0 ? tiling->numGroups : 1;
        qPerGroup_ = tiling->qPerGroup > 0 ? tiling->qPerGroup : (T_ > 0 ? T_ : 1);
        tileCountsAddr_ = scratch;
        int64_t tileElems = K2qCsrScratch::TileElems(tiling);
        tileCountsGm_.SetGlobalBuffer((__gm__ int32_t *)tileCountsAddr_, tileElems);
        int64_t rowCountElems = H_ * (totalRows_ > 0 ? totalRows_ : 1);
        if (rowCountElems < 1) {
            rowCountElems = 1;
        }
        rowCountsGm_.SetGlobalBuffer((__gm__ int32_t *)K2qCsrScratch::RowCountsAddr(scratch, tiling), rowCountElems);
        pipe_->InitBuffer(tileBuf_, K2qCsrCommon::Align32((totalRows_ > 0 ? totalRows_ : 1) * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        int64_t bid = GetBlockIdx();
        if (bid >= G_ || totalRows_ <= 0 || T_ <= 0 || topk_ <= 0) {
            return;
        }
        for (int64_t hid = 0; hid < H_; ++hid) {
            HistOneGroupHead(bid, hid);
        }
    }

private:
    __aicore__ inline void HistOneGroupHead(int64_t g, int64_t hid)
    {
        int64_t q0 = g * qPerGroup_;
        int64_t q1 = q0 + qPerGroup_;
        if (q1 > T_) {
            q1 = T_;
        }
        if (q0 >= q1) {
            return;
        }
        const int64_t histOff = (g * H_ + hid) * totalRows_;
        LocalTensor<int32_t> tile = tileBuf_.Get<int32_t>();
        Duplicate(tile, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
        PipeBarrier<PIPE_ALL>();
        K2qCsrCommon::CopyOutInt32(tileCountsGm_[histOff], tile, totalRows_);
        PipeBarrier<PIPE_ALL>();

        __gm__ int32_t *histGm =
            (__gm__ int32_t *)(tileCountsAddr_ + histOff * static_cast<int64_t>(sizeof(int32_t)));
        Simt::VF_CALL<k2q_csr_simt::HistEdgesVf>(
            Simt::Dim3(k2q_csr_simt::THREAD_NUM), (__gm__ int32_t *)q2kAddr_, (__gm__ int32_t *)tokenBatchAddr_,
            (__gm__ int32_t *)rowMapAddr_, histGm, static_cast<int32_t>(hid), static_cast<int32_t>(N_),
            static_cast<int32_t>(topk_), static_cast<int32_t>(q0), static_cast<int32_t>(q1 - q0),
            static_cast<int32_t>(maxKv_), static_cast<int32_t>(totalRows_), static_cast<int32_t>(B_));
        PipeBarrier<PIPE_ALL>();

        // 对齐 CUDA：把本 CTA 的 tile 累加进 row_counts[h]
        K2qCsrCommon::CopyInInt32(tile, tileCountsGm_[histOff], totalRows_);
        if (G_ == 1) {
            K2qCsrCommon::CopyOutInt32(rowCountsGm_[hid * totalRows_], tile, totalRows_);
        } else {
            K2qCsrCommon::AtomicAddOutInt32(rowCountsGm_[hid * totalRows_], tile, totalRows_);
        }
    }

    const K2qCsrTilingData *tiling_;
    TPipe *pipe_;
    GM_ADDR q2kAddr_;
    GM_ADDR rowMapAddr_;
    GM_ADDR tokenBatchAddr_;
    GM_ADDR tileCountsAddr_;
    GlobalTensor<int32_t> tileCountsGm_;
    GlobalTensor<int32_t> rowCountsGm_;
    int64_t H_, T_, topk_, N_, B_, totalRows_, maxKv_, G_, qPerGroup_;
    TBuf<TPosition::VECCALC> tileBuf_;
};

#endif
