/**
 * k2q_csr MC Hist / Scatter（A2 / A5 MC 共用，放在 op_kernel/ 上层）。
 *
 * 禁止 SIMT API。优化要点（相对初版 scalar）：
 *   - B==1：qTile=8192 + ping-pong CopyIn∥边循环
 *   - B==1 + Concat：Hist/Scatter FastB1（row=val，省 rowMap/token/cuQ）
 *   - Scatter：32B 对齐写槽 FlushStoreBatch（一批两次 barrier）
 *   - 通用路径：qTile=DEFAULT_TILE_EDGES，仍走 MapRow + token
 *
 * q_ind/slot=-1：Host fill_；多核段不重叠，无 SyncAll。
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
    __aicore__ inline void HistOneGroupHeadFastB1(int64_t g, int64_t hid);
    __aicore__ inline void ScatterOneGroupHead(int64_t g, int64_t hid);
    __aicore__ inline void ScatterOneGroupHeadFastB1(int64_t g, int64_t hid);
    __aicore__ inline int64_t QTileEff() const;
    __aicore__ inline int32_t MapRow(int32_t bi, int32_t val);

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
    int32_t qGlobalOffset_;
    int32_t orderMethod_;
    int32_t fastConcatB1_; // B==1 && Concat：row==val

    TBuf<TPosition::VECCALC> histBuf_;
    TBuf<TPosition::VECCALC> cursorBuf_;
    TBuf<TPosition::VECCALC> rowMapBuf_;
    TBuf<TPosition::VECCALC> tokenBuf_;
    TBuf<TPosition::VECCALC> cuQBuf_;
    TBuf<TPosition::VECCALC> qTileBuf_;
    TBuf<TPosition::VECCALC> qTileBufB_; // ping-pong 第二槽（仅 B==1 分配）
    TBuf<TPosition::VECCALC> scalarBuf_;
    TBuf<TPosition::VECCALC> storePosBuf_;
    TBuf<TPosition::VECCALC> storeQBuf_;
    TBuf<TPosition::VECCALC> storeSBuf_;

    static constexpr int32_t kStoreBatch = 64;
    // B==1 时 UB 富余，放大 qTile 减少外层循环 / CopyIn 次数
    static constexpr int64_t kTileEdgesB1 = 8192;
    int64_t tileEdgesCap_;
    int32_t useQPingPong_;
};

__aicore__ inline int64_t K2qCsrPipelineMc::QTileEff() const
{
    int64_t maxQTile = tileEdgesCap_ / (topk_ > 0 ? topk_ : 1);
    if (maxQTile < 1) {
        maxQTile = 1;
    }
    return maxQTile;
}

__aicore__ inline int32_t K2qCsrPipelineMc::MapRow(int32_t bi, int32_t val)
{
    if (val < 0 || val >= maxKv_) {
        return K2qCsrCommon::INVALID;
    }
    if (fastConcatB1_ != 0) {
        return val;
    }
    if (useGather_) {
        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
        return rm.GetValue(static_cast<int64_t>(bi) * maxKv_ + val);
    }
    return rowMapGm_.GetValue(static_cast<int64_t>(bi) * maxKv_ + val);
}

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
    qGlobalOffset_ = tiling->qGlobalOffset;
    orderMethod_ = tiling->orderMethod;
    fastConcatB1_ = (B_ == 1 && orderMethod_ == 0) ? 1 : 0;
    tileEdgesCap_ = (B_ == 1) ? kTileEdgesB1 : K2qCsrCommon::DEFAULT_TILE_EDGES;
    useQPingPong_ = (B_ == 1) ? 1 : 0;

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
    // B==1 时 bi 恒 0，token 缓冲缩到 1，避免 T 很大时占满 UB
    int64_t tokenElems = (B_ == 1) ? 1 : (T_ > 0 ? T_ : 1);
    pipe_->InitBuffer(tokenBuf_, K2qCsrCommon::Align32(tokenElems * sizeof(int32_t)));
    if (cuQ != nullptr) {
        pipe_->InitBuffer(cuQBuf_, K2qCsrCommon::Align32((B_ + 1) * sizeof(int32_t)));
    }
    pipe_->InitBuffer(qTileBuf_, K2qCsrCommon::Align32(tileEdgesCap_ * sizeof(int32_t)));
    if (useQPingPong_ != 0) {
        pipe_->InitBuffer(qTileBufB_, K2qCsrCommon::Align32(tileEdgesCap_ * sizeof(int32_t)));
    }
    pipe_->InitBuffer(scalarBuf_, K2qCsrCommon::Align32(8 * sizeof(int32_t)));
    // q/s 每槽 32B 对齐（STORE_SLOT_ALIGN_INTS），供 FlushStoreBatch 作 DataCopyPad 源
    pipe_->InitBuffer(storePosBuf_, K2qCsrCommon::Align32(kStoreBatch * sizeof(int32_t)));
    pipe_->InitBuffer(storeQBuf_,
                      K2qCsrCommon::Align32(kStoreBatch * K2qCsrCommon::STORE_SLOT_ALIGN_INTS * sizeof(int32_t)));
    pipe_->InitBuffer(storeSBuf_,
                      K2qCsrCommon::Align32(kStoreBatch * K2qCsrCommon::STORE_SLOT_ALIGN_INTS * sizeof(int32_t)));
    if (useGather_ && fastConcatB1_ == 0) {
        pipe_->InitBuffer(rowMapBuf_, K2qCsrCommon::Align32(tiling->rowMapElems * sizeof(int32_t)));
    }
}

/** B==1 + Concat：row==val，无 token/rowMap；紧凑边循环降 scalar。 */
__aicore__ inline void K2qCsrPipelineMc::HistOneGroupHeadFastB1(int64_t g, int64_t hid)
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
    Duplicate(hist, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<int32_t> qBufA = qTileBuf_.Get<int32_t>();
    LocalTensor<int32_t> qBufB = (useQPingPong_ != 0) ? qTileBufB_.Get<int32_t>() : qBufA;
    int64_t qTileEff = QTileEff();
    // Concat B1: row=val，合法域为 [0, min(totalRows,maxKv))
    int32_t lim = static_cast<int32_t>(totalRows_ < maxKv_ ? totalRows_ : maxKv_);
    const int64_t base = hid * N_;

    // ping-pong：发下一 tile 的 MTE2 同时标量处理当前 tile
    int64_t qs = q0;
    if (qs < q1) {
        int64_t qLen0 = qTileEff;
        if (qs + qLen0 > q1) {
            qLen0 = q1 - qs;
        }
        K2qCsrCommon::CopyInInt32(qBufA, q2kGm_[base + qs * topk_], qLen0 * topk_);
        LocalTensor<int32_t> cur = qBufA;
        LocalTensor<int32_t> nxt = qBufB;
        while (qs < q1) {
            int64_t qLen = qTileEff;
            if (qs + qLen > q1) {
                qLen = q1 - qs;
            }
            int64_t nEdge = qLen * topk_;
            int64_t qsNext = qs + qLen;
            if (useQPingPong_ != 0 && qsNext < q1) {
                int64_t qLenN = qTileEff;
                if (qsNext + qLenN > q1) {
                    qLenN = q1 - qsNext;
                }
                K2qCsrCommon::CopyInInt32NoBarrier(nxt, q2kGm_[base + qsNext * topk_], qLenN * topk_);
            }
            for (int64_t e = 0; e < nEdge; ++e) {
                int32_t val = cur.GetValue(e);
                if (static_cast<uint32_t>(val) < static_cast<uint32_t>(lim)) {
                    hist.SetValue(val, hist.GetValue(val) + 1);
                }
            }
            if (useQPingPong_ != 0 && qsNext < q1) {
                PipeBarrier<PIPE_ALL>();
                LocalTensor<int32_t> tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
            qs = qsNext;
        }
    }
    K2qCsrCommon::CopyOutInt32(tileCountsGm_[(g * H_ + hid) * totalRows_], hist, totalRows_);
    if (G_ == 1) {
        K2qCsrCommon::CopyOutInt32(rowCountsGm_[hid * totalRows_], hist, totalRows_);
    } else {
        K2qCsrCommon::AtomicAddOutInt32(rowCountsGm_[hid * totalRows_], hist, totalRows_);
    }
}

__aicore__ inline void K2qCsrPipelineMc::HistOneGroupHead(int64_t g, int64_t hid)
{
    if (fastConcatB1_ != 0) {
        HistOneGroupHeadFastB1(g, hid);
        return;
    }
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
    int64_t qTileEff = QTileEff();

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
                int32_t row = MapRow(bi, val);
                if (row >= 0 && row < totalRows_) {
                    hist.SetValue(row, hist.GetValue(row) + 1);
                }
            }
        }
    }
    K2qCsrCommon::CopyOutInt32(tileCountsGm_[(g * H_ + hid) * totalRows_], hist, totalRows_);
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
    if (B_ != 1) {
        LocalTensor<int32_t> tokenLocal = tokenBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(tokenLocal, tokenBatchGm_, T_);
    }
    if (useGather_ && fastConcatB1_ == 0) {
        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(rm, rowMapGm_, tiling_->rowMapElems);
    }
    for (int64_t hid = 0; hid < H_; ++hid) {
        HistOneGroupHead(bid, hid);
    }
}

/** B==1 + Concat 热路径：qVal=qAbs，row=val，批量 MTE3。 */
__aicore__ inline void K2qCsrPipelineMc::ScatterOneGroupHeadFastB1(int64_t g, int64_t hid)
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
    LocalTensor<int32_t> qBufA = qTileBuf_.Get<int32_t>();
    LocalTensor<int32_t> qBufB = (useQPingPong_ != 0) ? qTileBufB_.Get<int32_t>() : qBufA;
    LocalTensor<int32_t> posB = storePosBuf_.Get<int32_t>();
    LocalTensor<int32_t> qB = storeQBuf_.Get<int32_t>();
    LocalTensor<int32_t> sB = storeSBuf_.Get<int32_t>();
    Duplicate(cursor, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
    PipeBarrier<PIPE_ALL>();
    K2qCsrCommon::CopyInInt32(absBase, absBaseGm_[(g * H_ + hid) * totalRows_], totalRows_);

    int64_t qTileEff = QTileEff();
    int32_t batchN = 0;
    int32_t lim = static_cast<int32_t>(totalRows_ < maxKv_ ? totalRows_ : maxKv_);
    const int32_t nU = static_cast<int32_t>(N_);
    const int64_t base = hid * N_;

    int64_t qs = q0;
    if (qs < q1) {
        int64_t qLen0 = qTileEff;
        if (qs + qLen0 > q1) {
            qLen0 = q1 - qs;
        }
        K2qCsrCommon::CopyInInt32(qBufA, q2kGm_[base + qs * topk_], qLen0 * topk_);
        LocalTensor<int32_t> cur = qBufA;
        LocalTensor<int32_t> nxt = qBufB;
        while (qs < q1) {
            int64_t qLen = qTileEff;
            if (qs + qLen > q1) {
                qLen = q1 - qs;
            }
            int64_t qsNext = qs + qLen;
            if (useQPingPong_ != 0 && qsNext < q1) {
                int64_t qLenN = qTileEff;
                if (qsNext + qLenN > q1) {
                    qLenN = q1 - qsNext;
                }
                K2qCsrCommon::CopyInInt32NoBarrier(nxt, q2kGm_[base + qsNext * topk_], qLenN * topk_);
            }
            for (int64_t qi = 0; qi < qLen; ++qi) {
                int32_t qVal = static_cast<int32_t>(qs + qi);
                for (int64_t s = 0; s < topk_; ++s) {
                    int32_t val = cur.GetValue(qi * topk_ + s);
                    if (static_cast<uint32_t>(val) >= static_cast<uint32_t>(lim)) {
                        continue;
                    }
                    int32_t localOff = cursor.GetValue(val);
                    cursor.SetValue(val, localOff + 1);
                    int32_t pos = absBase.GetValue(val) + localOff;
                    if (static_cast<uint32_t>(pos) >= static_cast<uint32_t>(nU)) {
                        continue;
                    }
                    posB.SetValue(batchN, pos);
                    int32_t slotOff = batchN * K2qCsrCommon::STORE_SLOT_ALIGN_INTS;
                    qB.SetValue(slotOff, qVal);
                    sB.SetValue(slotOff, static_cast<int32_t>(s));
                    ++batchN;
                    if (batchN >= kStoreBatch) {
                        K2qCsrCommon::FlushStoreBatch(qIndGm_, slotGm_, base, posB, qB, sB, batchN);
                        batchN = 0;
                    }
                }
            }
            if (useQPingPong_ != 0 && qsNext < q1) {
                PipeBarrier<PIPE_ALL>();
                LocalTensor<int32_t> tmp = cur;
                cur = nxt;
                nxt = tmp;
            }
            qs = qsNext;
        }
    }
    if (batchN > 0) {
        K2qCsrCommon::FlushStoreBatch(qIndGm_, slotGm_, base, posB, qB, sB, batchN);
    }
}

__aicore__ inline void K2qCsrPipelineMc::ScatterOneGroupHead(int64_t g, int64_t hid)
{
    if (fastConcatB1_ != 0) {
        ScatterOneGroupHeadFastB1(g, hid);
        return;
    }
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
    LocalTensor<int32_t> qBuf = qTileBuf_.Get<int32_t>();
    LocalTensor<int32_t> posB = storePosBuf_.Get<int32_t>();
    LocalTensor<int32_t> qB = storeQBuf_.Get<int32_t>();
    LocalTensor<int32_t> sB = storeSBuf_.Get<int32_t>();
    Duplicate(cursor, static_cast<int32_t>(0), static_cast<int32_t>(totalRows_));
    PipeBarrier<PIPE_ALL>();
    K2qCsrCommon::CopyInInt32(absBase, absBaseGm_[(g * H_ + hid) * totalRows_], totalRows_);

    int64_t qTileEff = QTileEff();
    int32_t batchN = 0;
    const bool needCuQ = (qGlobalOffset_ == 0 && B_ > 1);
    LocalTensor<int32_t> cuQLocal = cuQBuf_.Get<int32_t>();
    const int64_t base = hid * N_;

    for (int64_t qs = q0; qs < q1; qs += qTileEff) {
        int64_t qLen = qTileEff;
        if (qs + qLen > q1) {
            qLen = q1 - qs;
        }
        K2qCsrCommon::CopyInInt32(qBuf, q2kGm_[base + qs * topk_], qLen * topk_);
        for (int64_t qi = 0; qi < qLen; ++qi) {
            int64_t qAbs = qs + qi;
            int32_t bi = tokenLocal.GetValue(qAbs);
            int32_t qVal = static_cast<int32_t>(qAbs);
            if (needCuQ) {
                qVal = static_cast<int32_t>(qAbs - cuQLocal.GetValue(bi));
            }
            for (int64_t s = 0; s < topk_; ++s) {
                int32_t val = qBuf.GetValue(qi * topk_ + s);
                int32_t row = MapRow(bi, val);
                if (row < 0 || row >= totalRows_) {
                    continue;
                }
                int32_t localOff = cursor.GetValue(row);
                cursor.SetValue(row, localOff + 1);
                int32_t pos = absBase.GetValue(row) + localOff;
                if (pos < 0 || pos >= N_) {
                    continue;
                }
                posB.SetValue(batchN, pos);
                int32_t slotOff = batchN * K2qCsrCommon::STORE_SLOT_ALIGN_INTS;
                qB.SetValue(slotOff, qVal);
                sB.SetValue(slotOff, static_cast<int32_t>(s));
                ++batchN;
                if (batchN >= kStoreBatch) {
                    K2qCsrCommon::FlushStoreBatch(qIndGm_, slotGm_, base, posB, qB, sB, batchN);
                    batchN = 0;
                }
            }
        }
    }
    if (batchN > 0) {
        K2qCsrCommon::FlushStoreBatch(qIndGm_, slotGm_, base, posB, qB, sB, batchN);
    }
}

__aicore__ inline void K2qCsrPipelineMc::ProcessPhase3Scatter()
{
    int64_t bid = GetBlockIdx();
    if (bid >= G_ || totalRows_ <= 0 || T_ <= 0) {
        return;
    }
    if (B_ != 1) {
        LocalTensor<int32_t> tokenLocal = tokenBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(tokenLocal, tokenBatchGm_, T_);
    }
    if (qGlobalOffset_ == 0 && B_ > 1) {
        LocalTensor<int32_t> cuQLocal = cuQBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(cuQLocal, cuQGm_, B_ + 1);
    }
    if (useGather_ && fastConcatB1_ == 0) {
        LocalTensor<int32_t> rm = rowMapBuf_.Get<int32_t>();
        K2qCsrCommon::CopyInInt32(rm, rowMapGm_, tiling_->rowMapElems);
    }
    for (int64_t hid = 0; hid < H_; ++hid) {
        ScatterOneGroupHead(bid, hid);
    }
}

#endif
