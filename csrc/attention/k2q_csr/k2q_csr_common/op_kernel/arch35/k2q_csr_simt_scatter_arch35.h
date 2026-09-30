/**
 * SIMT Stage S（对齐 CUDA k2q_scatter_kernel 的「分区 + 前缀预留」思路）。
 * 仅 ascend950 / use_simt=1（本文件位于 op_kernel/arch35/）。
 *
 * 竞态消除（相对旧版「全线程对共享 cursor AtomicAdd 抢偏移」）：
 *   1. 将本 CTA 的 q 区间切成 kScatterWarps 段（对齐 CUDA warp 分区）；
 *   2. 各 warp 独立直方图计数（计数可 AtomicAdd，交换律，不影响顺序）；
 *   3. 对每行做跨 warp exclusive scan → 每 warp 独占连续写槽；
 *   4. warp 内按 q 升序写；同 q 多 slot 命中同行时用 ballot/lane-rank
 *      确定偏移（不依赖 AtomicAdd 返回次序）。
 *
 * 跨 CTA 仍由 TilePrefix 的 abs_base[G,H,R] 保证互不重叠。
 * q_ind/slot=-1 由 Host fill_ 预填（对齐 MC / CUDA memset）；本核不再 Fill+SyncAll，
 * 以避免对 H*N 的二次全量写与核间屏障（Dump case 可省 ~半个 Scatter 时延）。
 */
#ifndef K2Q_CSR_SIMT_SCATTER_ARCH35_H
#define K2Q_CSR_SIMT_SCATTER_ARCH35_H

#include "k2q_csr_simt_common_arch35.h"
#include "../k2q_csr_common.h"
#include "../k2q_csr_tiling.h"
#include "simt_api/cpp/kernel_simt_warp_level_intf.h"
#include "simt_api/cpp/kernel_simt_math_intf.h"

using namespace AscendC;

namespace k2q_csr_simt {

/** 对齐 CUDA kWarps∈{1,2,4}：核内最多 4 路 q 分区 */
constexpr int32_t kScatterWarps = 4;
constexpr int32_t kWarpSize = 32;

/**
 * 无竞态 scatter：warp 分区 + exclusive 预留 + warp 内确定性 rank。
 *
 * partBase 布局 [numWarps, totalRows]：
 *   Phase1 写入 counts；Phase2 原地改为 exclusive base；Phase3 作 running cursor。
 */
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterEdgesVf(
    __gm__ int32_t *q2k, __gm__ int32_t *tokenBatch, __gm__ int32_t *rowMap, __gm__ int32_t *cuQ,
    __gm__ int32_t *qInd, __gm__ int32_t *slot, __gm__ int32_t *absBase, __ubuf__ int32_t *partBase, int32_t hid,
    int32_t N, int32_t topk, int32_t q0, int32_t qLen, int32_t maxKv, int32_t totalRows, int32_t B,
    int32_t numWarps, int32_t qGlobalOffset)
{
    const int32_t tid = static_cast<int32_t>(Simt::GetThreadIdx());
    const int32_t nthreads = static_cast<int32_t>(Simt::GetThreadNum());
    if (numWarps < 1) {
        numWarps = 1;
    }
    if (numWarps > kScatterWarps) {
        numWarps = kScatterWarps;
    }

    const int32_t activeThreads = numWarps * kWarpSize;
    const int32_t warpId = tid / kWarpSize;
    const int32_t lane = tid - warpId * kWarpSize;
    const bool inActiveWarp = (warpId < numWarps);

    // ---- Phase0: 清零 partBase[numWarps, R] ----
    const int32_t histElems = numWarps * totalRows;
    for (int32_t i = tid; i < histElems; i += nthreads) {
        partBase[i] = 0;
    }
    Simt::ThreadBarrier();

    // ---- Phase1: 各 warp 统计本 q 子区间对每行的 nnz（AtomicAdd 仅用于计数）----
    const int32_t qPerWarp = (qLen + numWarps - 1) / numWarps;
    const int32_t wq0 = inActiveWarp ? warpId * qPerWarp : qLen;
    int32_t wq1 = inActiveWarp ? wq0 + qPerWarp : qLen;
    if (wq1 > qLen) {
        wq1 = qLen;
    }
    __ubuf__ int32_t *warpCounts = partBase + warpId * totalRows;

    if (inActiveWarp && wq0 < wq1 && topk > 0) {
        const int32_t warpQLen = wq1 - wq0;
        const uint64_t numEdges = static_cast<uint64_t>(warpQLen) * static_cast<uint64_t>(topk);
        for (uint64_t e = static_cast<uint64_t>(lane); e < numEdges; e += static_cast<uint64_t>(kWarpSize)) {
            const int32_t qi = static_cast<int32_t>(e / static_cast<uint64_t>(topk));
            const int32_t s = static_cast<int32_t>(e - static_cast<uint64_t>(qi) * static_cast<uint64_t>(topk));
            const int32_t qAbs = q0 + wq0 + qi;
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
                (void)Simt::AtomicAdd(warpCounts + row, static_cast<int32_t>(1));
            }
        }
    }
    Simt::ThreadBarrier();

    // ---- Phase2: 每行跨 warp exclusive scan → partBase[w,r] = Σ_{w'<w} counts[w',r] ----
    for (int32_t r = tid; r < totalRows; r += nthreads) {
        int32_t sum = 0;
        for (int32_t w = 0; w < numWarps; ++w) {
            const int32_t c = partBase[w * totalRows + r];
            partBase[w * totalRows + r] = sum;
            sum += c;
        }
    }
    Simt::ThreadBarrier();

    // ---- Phase3: 各 warp 按 q 升序写；同行并发用 lane-rank，不用 AtomicAdd 定序 ----
    if (!inActiveWarp || wq0 >= wq1 || topk <= 0) {
        return;
    }
    __ubuf__ int32_t *warpCursor = partBase + warpId * totalRows; // 已是 exclusive base，作 running cursor

    // topk<=32：每 q 用一个 warp 的 lane 并行 slot；否则 lane0 串行 slot。
    // 注意：WarpShflSync 要求 warp 内所有 lane 同等参与，不可按 valid 分支跳过。
    if (topk <= kWarpSize) {
        for (int32_t qi = wq0; qi < wq1; ++qi) {
            const int32_t qAbs = q0 + qi;
            const int32_t bi = tokenBatch[qAbs];
            int32_t qVal = 0;
            int32_t row = K2qCsrCommon::INVALID;
            int32_t validI = 0;
            const int32_t s = lane;
            if (bi >= 0 && bi < B && s < topk) {
                qVal = (qGlobalOffset != 0) ? qAbs : (qAbs - cuQ[bi]);
                const int32_t val = q2k[hid * N + qAbs * topk + s];
                if (val >= 0 && val < maxKv) {
                    row = rowMap[bi * maxKv + val];
                    if (row >= 0 && row < totalRows) {
                        validI = 1;
                    }
                }
            }

            // 收集 lane0..topk-1 的 (valid,row)；全体 lane 都执行相同次数的 shfl
            uint32_t sameMask = 0u;
            for (int32_t L = 0; L < topk; ++L) {
                const int32_t rowL = Simt::WarpShflSync(row, L);
                const int32_t validL = Simt::WarpShflSync(validI, L);
                if (validI != 0 && validL != 0 && rowL == row) {
                    sameMask |= (1u << L);
                }
            }
            const int32_t rank = (validI != 0) ? Simt::Popc(sameMask & ((1u << lane) - 1u)) : 0;
            const int32_t nSame = (validI != 0) ? Simt::Popc(sameMask) : 0;
            // 无效 lane：leader=自己，shfl 自取 0；有效 lane：leader=同行最低 lane
            int32_t leader = lane;
            int32_t baseOff = 0;
            if (validI != 0 && sameMask != 0u) {
                leader = 0;
                uint32_t m = sameMask;
                while ((m & 1u) == 0u) {
                    ++leader;
                    m >>= 1u;
                }
                if (lane == leader) {
                    baseOff = warpCursor[row];
                    warpCursor[row] = baseOff + nSame;
                }
            }
            baseOff = Simt::WarpShflSync(baseOff, leader);
            if (validI != 0) {
                const int32_t pos = absBase[row] + baseOff + rank;
                if (pos >= 0 && pos < N) {
                    qInd[hid * N + pos] = qVal;
                    slot[hid * N + pos] = s;
                }
            }
        }
    } else {
        // topk > warpSize：仅 lane0 按 (qi,s) 串行写，保证升序、无竞态
        if (lane == 0) {
            for (int32_t qi = wq0; qi < wq1; ++qi) {
                const int32_t qAbs = q0 + qi;
                const int32_t bi = tokenBatch[qAbs];
                if (bi < 0 || bi >= B) {
                    continue;
                }
                const int32_t qVal = (qGlobalOffset != 0) ? qAbs : (qAbs - cuQ[bi]);
                for (int32_t s = 0; s < topk; ++s) {
                    const int32_t val = q2k[hid * N + qAbs * topk + s];
                    if (val < 0 || val >= maxKv) {
                        continue;
                    }
                    const int32_t row = rowMap[bi * maxKv + val];
                    if (row < 0 || row >= totalRows) {
                        continue;
                    }
                    const int32_t localOff = warpCursor[row];
                    warpCursor[row] = localOff + 1;
                    const int32_t pos = absBase[row] + localOff;
                    if (pos >= 0 && pos < N) {
                        qInd[hid * N + pos] = qVal;
                        slot[hid * N + pos] = s;
                    }
                }
            }
        }
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
        qGlobalOffset_ = tiling->qGlobalOffset;
        int64_t tileElems = G_ * H_ * (totalRows_ > 0 ? totalRows_ : 1);
        if (tileElems < 1) {
            tileElems = 1;
        }
        int64_t tileBytes = tileElems * static_cast<int64_t>(sizeof(int32_t));
        absBaseAddr_ = scratch + tileBytes;

        // partBase[kScatterWarps, R]；R 很大时降到 1/2 warp，避免撑爆 UB
        numWarps_ = k2q_csr_simt::kScatterWarps;
        const int64_t maxHistBytes = 96LL * 1024LL; // 给 hist 留上限，其余留给 DCache/管道
        while (numWarps_ > 1 &&
               numWarps_ * (totalRows_ > 0 ? totalRows_ : 1) * static_cast<int64_t>(sizeof(int32_t)) > maxHistBytes) {
            numWarps_ >>= 1;
        }
        if (numWarps_ < 1) {
            numWarps_ = 1;
        }
        const int64_t histElems = numWarps_ * (totalRows_ > 0 ? totalRows_ : 1);
        pipe_->InitBuffer(partBuf_, K2qCsrCommon::Align32(histElems * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        // Host 已 fill_(-1)；段互不重叠，无需 SyncAll
        int64_t bid = GetBlockIdx();
        if (bid >= G_ || totalRows_ <= 0 || T_ <= 0) {
            return;
        }
        for (int64_t hid = 0; hid < H_; ++hid) {
            ScatterOneGroupHead(bid, hid);
        }
    }

private:
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

        LocalTensor<int32_t> part = partBuf_.Get<int32_t>();
        const int64_t absOff = (g * H_ + hid) * totalRows_;
        __gm__ int32_t *absBaseGm =
            (__gm__ int32_t *)(absBaseAddr_ + absOff * static_cast<int64_t>(sizeof(int32_t)));
        Simt::VF_CALL<k2q_csr_simt::ScatterEdgesVf>(
            Simt::Dim3(k2q_csr_simt::THREAD_NUM), (__gm__ int32_t *)q2kAddr_, (__gm__ int32_t *)tokenBatchAddr_,
            (__gm__ int32_t *)rowMapAddr_, (__gm__ int32_t *)cuQAddr_, (__gm__ int32_t *)qIndAddr_,
            (__gm__ int32_t *)slotAddr_, absBaseGm, (__ubuf__ int32_t *)part.GetPhyAddr(),
            static_cast<int32_t>(hid), static_cast<int32_t>(N_), static_cast<int32_t>(topk_),
            static_cast<int32_t>(q0), static_cast<int32_t>(q1 - q0), static_cast<int32_t>(maxKv_),
            static_cast<int32_t>(totalRows_), static_cast<int32_t>(B_), static_cast<int32_t>(numWarps_),
            static_cast<int32_t>(qGlobalOffset_));
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
    int64_t numWarps_;
    int32_t qGlobalOffset_;
    TBuf<TPosition::VECCALC> partBuf_;
};

#endif
