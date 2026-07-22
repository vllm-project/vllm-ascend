/**
 * k2q_csr MC Hist / Scatter（A2 / A5 MC 共用，放在 op_kernel/ 上层）。
 *
 * Prefix 由独立算子 RowPrefix / TilePrefix 完成；本类仅：
 *   ProcessPhase1Hist    → tile_counts + AtomicAdd row_counts
 *   ProcessPhase3Scatter → 读 abs_base，写 q_ind/slot（多核按 group，无 SyncAll）
 *
 * q_ind/slot=-1：由 Host fill_ 与 TilePrefix 异步重叠完成（对齐 CUDA memset）。
 * A2 上 SyncAll 不可用，故 MC 路径禁止核内 Fill+SyncAll。
 * A5 SIMT 的 Fill+SyncAll 见 arch35/k2q_csr_simt_scatter_arch35.h。
 *
 * scratch（相对 histScratch）：tile_counts | abs_base | row_counts
 */
#ifndef K2Q_CSR_MC_H
#define K2Q_CSR_MC_H

#include "k2q_csr_common.h"
#include "k2q_csr_tiling.h"

using namespace AscendC;

class K2qCsrPipelineMc {
public:
    __aicore__ inline K2qCsrPipelineMc() {}

    __aicore__ inline void Init(GM_ADDR q2k, GM_ADDR cuQ, GM_ADDR rowMap, GM_ADDR tokenBatch, GM_ADDR rowPtr,
                                GM_ADDR qInd, GM_ADDR slot, GM_ADDR scratch, const K2qCsrTilingData *tiling,
                                TPipe *pipe);

    __aicore__ inline void ProcessPhase1Hist();
    __aicore__ inline void ProcessPhase3Scatter();

private:
    __aicore__ inline void HistOneGroupHead(int64_t g, int64_t hid);
    __aicore__ inline void ScatterOneGroupHead(int64_t g, int64_t hid);

    const K2qCsrTilingData *tiling_;
    TPipe *pipe_;

    GlobalTensor<int32_t> q2kGm_;
    GlobalTensor<int32_t> cuQGm_;
    GlobalTensor<int32_t> rowMapGm_;
    GlobalTensor<int32_t> tokenBatchGm_;
    GlobalTensor<int32_t> qIndGm_;
    GlobalTensor<int32_t> slotGm_;
    GlobalTensor<int32_t> tileCountsGm_;
    GlobalTensor<int32_t> absBaseGm_;
    GlobalTensor<int32_t> rowCountsGm_;

    int64_t H_;
    int64_t T_;
    int64_t topk_;
    int64_t N_;
    int64_t B_;
    int64_t totalRows_;
    int64_t maxKv_;
    int64_t G_;
    int64_t qPerGroup_;
    int32_t useGather_;

    TBuf<TPosition::VECCALC> histBuf_;
    TBuf<TPosition::VECCALC> cursorBuf_;
    TBuf<TPosition::VECCALC> rowMapBuf_;
    TBuf<TPosition::VECCALC> tokenBuf_;
    TBuf<TPosition::VECCALC> cuQBuf_;
    TBuf<TPosition::VECCALC> qTileBuf_;
    TBuf<TPosition::VECCALC> scalarBuf_;
};

__aicore__ inline void K2qCsrPipelineMc::Init(GM_ADDR q2k, GM_ADDR cuQ, GM_ADDR rowMap, GM_ADDR tokenBatch,
                                               GM_ADDR rowPtr, GM_ADDR qInd, GM_ADDR slot, GM_ADDR scratch,
                                               const K2qCsrTilingData *tiling, TPipe *pipe)
{
    (void)rowPtr;
    tiling_ = tiling;
    pipe_ = pipe;
    H_ = tiling->H;
    T_ = tiling->T;
    topk_ = tiling->topk;
    N_ = tiling->N;
    B_ = tiling->B;
    totalRows_ = tiling->totalRows;
    maxKv_ = tiling->maxKv;
    G_ = tiling->numGroups > 0 ? tiling->numGroups : 1;
    qPerGroup_ = tiling->qPerGroup > 0 ? tiling->qPerGroup : (T_ > 0 ? T_ : 1);
    useGather_ = tiling->useGather;

    int64_t tileElems = G_ * H_ * totalRows_;
    if (tileElems < 1) {
        tileElems = 1;
    }
    int64_t tileBytes = tileElems * static_cast<int64_t>(sizeof(int32_t));

    q2kGm_.SetGlobalBuffer((__gm__ int32_t *)q2k, H_ * N_ > 0 ? H_ * N_ : 1);
    if (cuQ != nullptr) {
        cuQGm_.SetGlobalBuffer((__gm__ int32_t *)cuQ, B_ + 1);
    }
    rowMapGm_.SetGlobalBuffer((__gm__ int32_t *)rowMap, tiling->rowMapElems > 0 ? tiling->rowMapElems : 1);
    tokenBatchGm_.SetGlobalBuffer((__gm__ int32_t *)tokenBatch, T_ > 0 ? T_ : 1);
    if (qInd != nullptr) {
        qIndGm_.SetGlobalBuffer((__gm__ int32_t *)qInd, H_ * N_ > 0 ? H_ * N_ : 1);
    }
    if (slot != nullptr) {
        slotGm_.SetGlobalBuffer((__gm__ int32_t *)slot, H_ * N_ > 0 ? H_ * N_ : 1);
    }
    tileCountsGm_.SetGlobalBuffer((__gm__ int32_t *)scratch, tileElems);
    absBaseGm_.SetGlobalBuffer((__gm__ int32_t *)(scratch + tileBytes), tileElems);
    int64_t rowCountElems = H_ * (totalRows_ > 0 ? totalRows_ : 1);
    if (rowCountElems < 1) {
        rowCountElems = 1;
    }
    rowCountsGm_.SetGlobalBuffer((__gm__ int32_t *)(scratch + tileBytes * 2), rowCountElems);

    pipe_->InitBuffer(histBuf_, K2qCsrCommon::Align32((totalRows_ > 0 ? totalRows_ : 1) * sizeof(int32_t)));
    pipe_->InitBuffer(cursorBuf_, K2qCsrCommon::Align32((totalRows_ > 0 ? totalRows_ : 1) * sizeof(int32_t)));
    pipe_->InitBuffer(tokenBuf_, K2qCsrCommon::Align32((T_ > 0 ? T_ : 1) * sizeof(int32_t)));
    if (cuQ != nullptr) {
        pipe_->InitBuffer(cuQBuf_, K2qCsrCommon::Align32((B_ + 1) * sizeof(int32_t)));
    }
    pipe_->InitBuffer(qTileBuf_, K2qCsrCommon::Align32(K2qCsrCommon::DEFAULT_TILE_EDGES * sizeof(int32_t)));
    pipe_->InitBuffer(scalarBuf_, K2qCsrCommon::Align32(8 * sizeof(int32_t)));
    if (useGather_) {
        pipe_->InitBuffer(rowMapBuf_, K2qCsrCommon::Align32(tiling->rowMapElems * sizeof(int32_t)));
    }
}

__aicore__ inline void K2qCsrPipelineMc::HistOneGroupHead(int64_t g, int64_t hid)
{
    int64_t q0 = g * qPerGroup_;
    int64_t q1 = q0 + qPerGroup_;
    if (q1 > T_) {
        q1 = T_;
    }
    if (q0 >= q1 || totalRows_ <= 0) {
        return;
    }

    LocalTensor<int32_t> hist = histBuf_.Get<int32_t>();
    LocalTensor<int32_t> tokenLocal = tokenBuf_.Get<int32_t>();
    Duplicate(hist, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<int32_t> qBuf = qTileBuf_.Get<int32_t>();
    int64_t maxQTile = K2qCsrCommon::DEFAULT_TILE_EDGES / (topk_ > 0 ? topk_ : 1);
    if (maxQTile < 1) {
        maxQTile = 1;
    }
    int64_t qTileEff = 128 < maxQTile ? 128 : maxQTile;

    for (int64_t qs = q0; qs < q1; qs += qTileEff) {
        int64_t qLen = qTileEff;
        if (qs + qLen > q1) {
            qLen = q1 - qs;
        }
        K2qCsrCommon::CopyInInt32(qBuf, q2kGm_[hid * N_ + qs * topk_], qLen * topk_);
        for (int64_t qi = 0; qi < qLen; ++qi) {
            int64_t qAbs = qs + qi;
            int32_t bi = tokenLocal.GetValue(qAbs);
            for (int64_t s = 0; s < topk_; ++s) {
                int32_t val = qBuf.GetValue(qi * topk_ + s);
                int32_t row = K2qCsrCommon::INVALID;
                if (val >= 0 && val < maxKv_) {
                    if (useGather_) {
                        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
                        row = rm.GetValue(static_cast<int64_t>(bi) * maxKv_ + val);
                    } else {
                        row = rowMapGm_.GetValue(static_cast<int64_t>(bi) * maxKv_ + val);
                    }
                }
                if (row >= 0 && row < totalRows_) {
                    hist.SetValue(row, hist.GetValue(row) + 1);
                }
            }
        }
    }
    K2qCsrCommon::CopyOutInt32(tileCountsGm_[(g * H_ + hid) * totalRows_], hist, totalRows_);
    // 对齐 CUDA Hist：atomicAdd(row_counts[h], tile) 供 PR 直接扫
    if (G_ == 1) {
        K2qCsrCommon::CopyOutInt32(rowCountsGm_[hid * totalRows_], hist, totalRows_);
    } else {
        K2qCsrCommon::AtomicAddOutInt32(rowCountsGm_[hid * totalRows_], hist, totalRows_);
    }
}

__aicore__ inline void K2qCsrPipelineMc::ProcessPhase1Hist()
{
    int64_t bid = GetBlockIdx();
    if (bid >= G_ || totalRows_ <= 0 || T_ <= 0) {
        return;
    }
    LocalTensor<int32_t> tokenLocal = tokenBuf_.Get<int32_t>();
    K2qCsrCommon::CopyInInt32(tokenLocal, tokenBatchGm_, T_);
    if (useGather_) {
        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(rm, rowMapGm_, tiling_->rowMapElems);
    }
    for (int64_t hid = 0; hid < H_; ++hid) {
        HistOneGroupHead(bid, hid);
    }
}

__aicore__ inline void K2qCsrPipelineMc::ScatterOneGroupHead(int64_t g, int64_t hid)
{
    int64_t q0 = g * qPerGroup_;
    int64_t q1 = q0 + qPerGroup_;
    if (q1 > T_) {
        q1 = T_;
    }
    if (q0 >= q1 || totalRows_ <= 0) {
        return;
    }

    LocalTensor<int32_t> cursor = cursorBuf_.Get<int32_t>();
    LocalTensor<int32_t> absBase = histBuf_.Get<int32_t>();
    LocalTensor<int32_t> tokenLocal = tokenBuf_.Get<int32_t>();
    LocalTensor<int32_t> cuQLocal = cuQBuf_.Get<int32_t>();
    LocalTensor<int32_t> scalarTmp = scalarBuf_.Get<int32_t>();
    Duplicate(cursor, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
    PipeBarrier<PIPE_ALL>();
    K2qCsrCommon::CopyInInt32(absBase, absBaseGm_[(g * H_ + hid) * totalRows_], totalRows_);

    LocalTensor<int32_t> qBuf = qTileBuf_.Get<int32_t>();
    int64_t maxQTile = K2qCsrCommon::DEFAULT_TILE_EDGES / (topk_ > 0 ? topk_ : 1);
    if (maxQTile < 1) {
        maxQTile = 1;
    }
    int64_t qTileEff = 128 < maxQTile ? 128 : maxQTile;

    for (int64_t qs = q0; qs < q1; qs += qTileEff) {
        int64_t qLen = qTileEff;
        if (qs + qLen > q1) {
            qLen = q1 - qs;
        }
        K2qCsrCommon::CopyInInt32(qBuf, q2kGm_[hid * N_ + qs * topk_], qLen * topk_);
        for (int64_t qi = 0; qi < qLen; ++qi) {
            int64_t qAbs = qs + qi;
            int32_t bi = tokenLocal.GetValue(qAbs);
            int32_t qStart = cuQLocal.GetValue(bi);
            int32_t localQ = static_cast<int32_t>(qAbs - qStart);
            for (int64_t s = 0; s < topk_; ++s) {
                int32_t val = qBuf.GetValue(qi * topk_ + s);
                int32_t row = K2qCsrCommon::INVALID;
                if (val >= 0 && val < maxKv_) {
                    if (useGather_) {
                        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
                        row = rm.GetValue(static_cast<int64_t>(bi) * maxKv_ + val);
                    } else {
                        row = rowMapGm_.GetValue(static_cast<int64_t>(bi) * maxKv_ + val);
                    }
                }
                if (row < 0 || row >= totalRows_) {
                    continue;
                }
                int32_t localOff = cursor.GetValue(row);
                cursor.SetValue(row, localOff + 1);
                int32_t pos = absBase.GetValue(row) + localOff;
                if (pos < 0 || pos >= N_) {
                    continue;
                }
                K2qCsrCommon::StoreScalarInt32(qIndGm_, hid * N_ + pos, localQ, scalarTmp);
                K2qCsrCommon::StoreScalarInt32(slotGm_, hid * N_ + pos, static_cast<int32_t>(s), scalarTmp);
            }
        }
    }
}

__aicore__ inline void K2qCsrPipelineMc::ProcessPhase3Scatter()
{
    // 多核按 group 划分写盘，段互不重叠；-1 已由 Host 预填，无需 SyncAll
    int64_t bid = GetBlockIdx();
    if (bid >= G_ || totalRows_ <= 0 || T_ <= 0) {
        return;
    }
    LocalTensor<int32_t> tokenLocal = tokenBuf_.Get<int32_t>();
    LocalTensor<int32_t> cuQLocal = cuQBuf_.Get<int32_t>();
    K2qCsrCommon::CopyInInt32(tokenLocal, tokenBatchGm_, T_);
    K2qCsrCommon::CopyInInt32(cuQLocal, cuQGm_, B_ + 1);
    if (useGather_) {
        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(rm, rowMapGm_, tiling_->rowMapElems);
    }
    for (int64_t hid = 0; hid < H_; ++hid) {
        ScatterOneGroupHead(bid, hid);
    }
}

#endif
