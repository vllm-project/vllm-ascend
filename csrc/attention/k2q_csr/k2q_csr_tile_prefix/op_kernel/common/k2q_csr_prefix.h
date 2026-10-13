/**
 * Stage PR / PT（A2 / A5 共用，放在 op_kernel/ 上层）。
 *
 * 命名历史：曾叫 simt_prefix；实为通用多核 Prefix，非 SIMT 专属。
 * A5 SIMT 专有逻辑仅在 op_kernel/arch35/。
 *
 * PR 对齐 CUDA k2q_row_prefix_kernel：
 *   - 输入已是 Hist 归约好的 row_counts[H,R]（不再 Σ_g tile_counts）
 *   - 每 head 一核 exclusive scan → row_ptr
 *   - q_ind/slot 的 -1 预填由 Host 完成（对齐 CUDA memset），本阶段不做
 *
 * PT：沿 g 轴 exclusive → abs_base；按 row 多核
 */
#ifndef K2Q_CSR_PREFIX_H
#define K2Q_CSR_PREFIX_H

#include "k2q_csr_common.h"
#include "k2q_csr_tiling.h"
#include "k2q_csr_scratch.h"

using namespace AscendC;

class K2qCsrRowPrefixKernel {
public:
    __aicore__ inline K2qCsrRowPrefixKernel() {}

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR qInd, GM_ADDR slot, GM_ADDR scratch,
                                const K2qCsrTilingData *tiling, TPipe *pipe)
    {
        (void)qInd;
        (void)slot;
        tiling_ = tiling;
        pipe_ = pipe;
        H_ = tiling->H;
        totalRows_ = tiling->totalRows;
        bins_ = totalRows_ + 1;
        usedCores_ = tiling->usedCores > 0 ? tiling->usedCores : 1;
        int64_t rowCountElems = H_ * (totalRows_ > 0 ? totalRows_ : 1);
        if (rowCountElems < 1) {
            rowCountElems = 1;
        }
        rowCountsGm_.SetGlobalBuffer((__gm__ int32_t *)K2qCsrScratch::RowCountsAddr(scratch, tiling), rowCountElems);
        rowPtrGm_.SetGlobalBuffer((__gm__ int32_t *)rowPtr, H_ * bins_ > 0 ? H_ * bins_ : 1);
        pipe_->InitBuffer(cntBuf_, K2qCsrCommon::Align32((totalRows_ > 0 ? totalRows_ : 1) * sizeof(int32_t)));
        pipe_->InitBuffer(outBuf_, K2qCsrCommon::Align32((bins_ > 0 ? bins_ : 1) * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        int64_t bid = GetBlockIdx();
        if (bid >= usedCores_ || bid >= H_) {
            return;
        }
        // 对齐 CUDA：grid = H，每核一个 head
        int64_t headsPer = (H_ + usedCores_ - 1) / usedCores_;
        int64_t h0 = bid * headsPer;
        int64_t h1 = h0 + headsPer;
        if (h1 > H_) {
            h1 = H_;
        }
        if (h0 >= h1) {
            return;
        }

        LocalTensor<int32_t> cnt = cntBuf_.Get<int32_t>();
        LocalTensor<int32_t> out = outBuf_.Get<int32_t>();

        for (int64_t hid = h0; hid < h1; ++hid) {
            if (totalRows_ <= 0) {
                Duplicate(out, static_cast<int32_t>(0), static_cast<int32_t>(bins_ > 0 ? bins_ : 1));
                PipeBarrier<PIPE_ALL>();
                K2qCsrCommon::CopyOutInt32(rowPtrGm_[hid * bins_], out, bins_);
                continue;
            }
            K2qCsrCommon::CopyInInt32(cnt, rowCountsGm_[hid * totalRows_], totalRows_);
            // exclusive scan：row_ptr[0]=0；row_ptr[r+1]=sum_{i<=r} counts[i]
            int32_t run = 0;
            for (int64_t r = 0; r < bins_; ++r) {
                out.SetValue(r, run);
                if (r < totalRows_) {
                    run += cnt.GetValue(r);
                }
            }
            PipeBarrier<PIPE_ALL>();
            K2qCsrCommon::CopyOutInt32(rowPtrGm_[hid * bins_], out, bins_);
        }
    }

private:
    const K2qCsrTilingData *tiling_;
    TPipe *pipe_;
    GlobalTensor<int32_t> rowCountsGm_;
    GlobalTensor<int32_t> rowPtrGm_;
    int64_t H_, totalRows_, bins_, usedCores_;
    TBuf<TPosition::VECCALC> cntBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
};

/** PT：按 row 多核；abs_base[g,h,r] = row_ptr[h,r] + Σ_{g'<g} tile_counts[g',h,r] */
class K2qCsrTilePrefixKernel {
public:
    __aicore__ inline K2qCsrTilePrefixKernel() {}

    __aicore__ inline void Init(GM_ADDR rowPtr, GM_ADDR scratch, const K2qCsrTilingData *tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        pipe_ = pipe;
        H_ = tiling->H;
        totalRows_ = tiling->totalRows;
        bins_ = totalRows_ + 1;
        G_ = tiling->numGroups > 0 ? tiling->numGroups : 1;
        usedCores_ = tiling->usedCores > 0 ? tiling->usedCores : 1;
        int64_t tileElems = K2qCsrScratch::TileElems(tiling);
        int64_t tileBytes = tileElems * static_cast<int64_t>(sizeof(int32_t));
        tileCountsGm_.SetGlobalBuffer((__gm__ int32_t *)scratch, tileElems);
        absBaseGm_.SetGlobalBuffer((__gm__ int32_t *)(scratch + tileBytes), tileElems);
        rowPtrGm_.SetGlobalBuffer((__gm__ int32_t *)rowPtr, H_ * bins_ > 0 ? H_ * bins_ : 1);

        int64_t rowsPer = 1;
        if (totalRows_ > 0 && usedCores_ > 0) {
            rowsPer = (totalRows_ + usedCores_ - 1) / usedCores_;
        }
        if (rowsPer < 1) {
            rowsPer = 1;
        }
        rowsPer_ = rowsPer;
        // 向量 Add 需 32B 对齐长度；按 align8(rowsPer) 分配
        int64_t rowsPerAlign = (rowsPer + 7) / 8 * 8;
        if (rowsPerAlign < 8) {
            rowsPerAlign = 8;
        }
        rowsPerAlign_ = rowsPerAlign;
        pipe_->InitBuffer(tileSegBuf_, K2qCsrCommon::Align32(rowsPerAlign * sizeof(int32_t)));
        pipe_->InitBuffer(absSegBuf_, K2qCsrCommon::Align32(rowsPerAlign * sizeof(int32_t)));
        pipe_->InitBuffer(runBuf_, K2qCsrCommon::Align32(rowsPerAlign * sizeof(int32_t)));
    }

    __aicore__ inline void Process()
    {
        if (totalRows_ <= 0) {
            return;
        }
        int64_t bid = GetBlockIdx();
        if (bid >= usedCores_) {
            return;
        }
        int64_t r0 = bid * rowsPer_;
        int64_t r1 = r0 + rowsPer_;
        if (r1 > totalRows_) {
            r1 = totalRows_;
        }
        if (r0 >= r1) {
            return;
        }
        const int64_t nRow = r1 - r0;
        // 向量长度向上对齐到 8（32B）；尾部 pad 0，写出仍用 nRow
        int32_t nAlign = static_cast<int32_t>((nRow + 7) / 8 * 8);
        if (nAlign < 8) {
            nAlign = 8;
        }

        LocalTensor<int32_t> tileSeg = tileSegBuf_.Get<int32_t>();
        LocalTensor<int32_t> absSeg = absSegBuf_.Get<int32_t>();
        LocalTensor<int32_t> run = runBuf_.Get<int32_t>();

        for (int64_t hid = 0; hid < H_; ++hid) {
            Duplicate(run, static_cast<int32_t>(0), nAlign);
            PipeBarrier<PIPE_ALL>();
            K2qCsrCommon::CopyInInt32(run, rowPtrGm_[hid * bins_ + r0], nRow);

            // G==1：abs_base = row_ptr 段
            if (G_ == 1) {
                int64_t off = hid * totalRows_;
                K2qCsrCommon::CopyOutInt32(absBaseGm_[off + r0], run, nRow);
                continue;
            }

            for (int64_t g = 0; g < G_; ++g) {
                int64_t off = (g * H_ + hid) * totalRows_;
                DataCopy(absSeg, run, nAlign);
                PipeBarrier<PIPE_ALL>();
                K2qCsrCommon::CopyOutInt32(absBaseGm_[off + r0], absSeg, nRow);
                if (g + 1 < G_) {
                    Duplicate(tileSeg, static_cast<int32_t>(0), nAlign);
                    PipeBarrier<PIPE_ALL>();
                    K2qCsrCommon::CopyInInt32(tileSeg, tileCountsGm_[off + r0], nRow);
                    Add(run, run, tileSeg, nAlign);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

private:
    const K2qCsrTilingData *tiling_;
    TPipe *pipe_;
    GlobalTensor<int32_t> tileCountsGm_;
    GlobalTensor<int32_t> absBaseGm_;
    GlobalTensor<int32_t> rowPtrGm_;
    int64_t H_, totalRows_, bins_, G_, usedCores_, rowsPer_, rowsPerAlign_;
    TBuf<TPosition::VECCALC> tileSegBuf_;
    TBuf<TPosition::VECCALC> absSegBuf_;
    TBuf<TPosition::VECCALC> runBuf_;
};

#endif
