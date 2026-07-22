/**
 * SIMT Stage S（对齐 CUDA k2q_scatter_kernel）。
 * 仅 ascend950 / use_simt=1（本文件位于 op_kernel/arch35/）。
 *
 * 每核 = CTA；UB cursor AtomicAdd；absBase / 输出直访 GM。
 * 入口先全核 grid-stride 将 q_ind/slot 填 INVALID，SyncAll 后再 scatter。
 * A2 / A5 MC 见上层 k2q_csr_mc.h（Host fill，无 SyncAll）。
 */
#ifndef K2Q_CSR_SIMT_SCATTER_ARCH35_H
#define K2Q_CSR_SIMT_SCATTER_ARCH35_H

#include "k2q_csr_simt_common_arch35.h"
#include "../k2q_csr_common.h"
#include "../k2q_csr_tiling.h"

using namespace AscendC;

namespace k2q_csr_simt {

/** 全核协作将 dst[0,n) 置为 INVALID（-1） */
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void FillInvalidVf(__gm__ int32_t *dst, uint64_t n)
{
    for (uint64_t i = static_cast<uint64_t>(Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
         i < n; i += static_cast<uint64_t>(Simt::GetThreadNum() * Simt::GetBlockNum())) {
        dst[i] = K2qCsrCommon::INVALID;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterEdgesVf(
    __gm__ int32_t *q2k, __gm__ int32_t *tokenBatch, __gm__ int32_t *rowMap, __gm__ int32_t *cuQ,
    __gm__ int32_t *qInd, __gm__ int32_t *slot, __gm__ int32_t *absBase, __ubuf__ int32_t *cursor, int32_t hid,
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
        const int32_t qStart = cuQ[bi];
        const int32_t localQ = qAbs - qStart;
        const int32_t val = q2k[hid * N + qAbs * topk + s];
        if (val < 0 || val >= maxKv) {
            continue;
        }
        const int32_t row = rowMap[bi * maxKv + val];
        if (row < 0 || row >= totalRows) {
            continue;
        }
        const int32_t localOff = Simt::AtomicAdd(cursor + row, static_cast<int32_t>(1));
        const int32_t pos = absBase[row] + localOff;
        if (pos < 0 || pos >= N) {
            continue;
        }
        qInd[hid * N + pos] = localQ;
        slot[hid * N + pos] = s;
    }
}

} // namespace k2q_csr_simt

class K2qCsrSimtScatter {
public:
    __aicore__ inline K2qCsrSimtScatter() {}

    __aicore__ inline void Init(GM_ADDR q2k, GM_ADDR cuQ, GM_ADDR rowMap, GM_ADDR tokenBatch, GM_ADDR qInd,
                                GM_ADDR slot, GM_ADDR scratch, const K2qCsrTilingData *tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        q2kAddr_ = q2k;
        cuQAddr_ = cuQ;
        rowMapAddr_ = rowMap;
        tokenBatchAddr_ = tokenBatch;
        qIndAddr_ = qInd;
        slotAddr_ = slot;
        H_ = tiling->H;
        T_ = tiling->T;
        topk_ = tiling->topk;
        N_ = tiling->N;
        B_ = tiling->B;
        totalRows_ = tiling->totalRows;
        maxKv_ = tiling->maxKv;
        G_ = tiling->numGroups > 0 ? tiling->numGroups : 1;
        qPerGroup_ = tiling->qPerGroup > 0 ? tiling->qPerGroup : (T_ > 0 ? T_ : 1);
        int64_t tileElems = G_ * H_ * (totalRows_ > 0 ? totalRows_ : 1);
        if (tileElems < 1) {
            tileElems = 1;
        }
        int64_t tileBytes = tileElems * static_cast<int64_t>(sizeof(int32_t));
        absBaseAddr_ = scratch + tileBytes;
        pipe_->InitBuffer(cursorBuf_, K2qCsrCommon::Align32((totalRows_ > 0 ? totalRows_ : 1) * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        FillOutputsInvalid();
        SyncAll();

        int64_t bid = GetBlockIdx();
        if (bid >= G_ || totalRows_ <= 0 || T_ <= 0) {
            return;
        }
        for (int64_t hid = 0; hid < H_; ++hid) {
            ScatterOneGroupHead(bid, hid);
        }
    }

private:
    __aicore__ inline void FillOutputsInvalid()
    {
        const uint64_t n = static_cast<uint64_t>(H_ > 0 && N_ > 0 ? H_ * N_ : 0);
        if (n == 0) {
            return;
        }
        Simt::VF_CALL<k2q_csr_simt::FillInvalidVf>(Simt::Dim3(k2q_csr_simt::THREAD_NUM), (__gm__ int32_t *)qIndAddr_,
                                                     n);
        Simt::VF_CALL<k2q_csr_simt::FillInvalidVf>(Simt::Dim3(k2q_csr_simt::THREAD_NUM), (__gm__ int32_t *)slotAddr_,
                                                     n);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ScatterOneGroupHead(int64_t g, int64_t hid)
    {
        int64_t q0 = g * qPerGroup_;
        int64_t q1 = q0 + qPerGroup_;
        if (q1 > T_) {
            q1 = T_;
        }
        if (q0 >= q1 || topk_ <= 0) {
            return;
        }

        LocalTensor<int32_t> cursor = cursorBuf_.Get<int32_t>();
        Duplicate(cursor, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
        PipeBarrier<PIPE_ALL>();

        const int64_t absOff = (g * H_ + hid) * totalRows_;
        __gm__ int32_t *absBaseGm =
            (__gm__ int32_t *)(absBaseAddr_ + absOff * static_cast<int64_t>(sizeof(int32_t)));
        Simt::VF_CALL<k2q_csr_simt::ScatterEdgesVf>(
            Simt::Dim3(k2q_csr_simt::THREAD_NUM), (__gm__ int32_t *)q2kAddr_, (__gm__ int32_t *)tokenBatchAddr_,
            (__gm__ int32_t *)rowMapAddr_, (__gm__ int32_t *)cuQAddr_, (__gm__ int32_t *)qIndAddr_,
            (__gm__ int32_t *)slotAddr_, absBaseGm, (__ubuf__ int32_t *)cursor.GetPhyAddr(),
            static_cast<int32_t>(hid), static_cast<int32_t>(N_), static_cast<int32_t>(topk_),
            static_cast<int32_t>(q0), static_cast<int32_t>(q1 - q0), static_cast<int32_t>(maxKv_),
            static_cast<int32_t>(totalRows_), static_cast<int32_t>(B_));
        PipeBarrier<PIPE_ALL>();
    }

    const K2qCsrTilingData *tiling_;
    TPipe *pipe_;
    GM_ADDR q2kAddr_;
    GM_ADDR cuQAddr_;
    GM_ADDR rowMapAddr_;
    GM_ADDR tokenBatchAddr_;
    GM_ADDR qIndAddr_;
    GM_ADDR slotAddr_;
    GM_ADDR absBaseAddr_;
    int64_t H_, T_, topk_, N_, B_, totalRows_, maxKv_, G_, qPerGroup_;
    TBuf<TPosition::VECCALC> cursorBuf_;
};

#endif
