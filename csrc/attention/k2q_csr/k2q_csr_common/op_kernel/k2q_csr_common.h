/**
 * k2q_csr AscendC common helpers.
 */
#ifndef K2Q_CSR_COMMON_H
#define K2Q_CSR_COMMON_H

#include "kernel_operator.h"

namespace K2qCsrCommon {

constexpr int32_t INVALID = -1;
constexpr int64_t ROW_MAP_UB_MAX = 32768;
constexpr int64_t DEFAULT_TILE_EDGES = 2048;
constexpr int64_t EMIT_TILE_EDGES = 2048;

__aicore__ inline int64_t DivCeil(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline int64_t Align32(int64_t bytes)
{
    return (bytes + 31) / 32 * 32;
}

__aicore__ inline void CopyInInt32NoBarrier(const AscendC::LocalTensor<int32_t> &dst,
                                            const AscendC::GlobalTensor<int32_t> &src, int64_t nElems)
{
    if (nElems <= 0) {
        return;
    }
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(nElems * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params, pad);
}

__aicore__ inline void CopyInInt32(const AscendC::LocalTensor<int32_t> &dst,
                                   const AscendC::GlobalTensor<int32_t> &src, int64_t nElems)
{
    CopyInInt32NoBarrier(dst, src, nElems);
    if (nElems > 0) {
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

__aicore__ inline void CopyOutInt32(const AscendC::GlobalTensor<int32_t> &dst,
                                    const AscendC::LocalTensor<int32_t> &src, int64_t nElems)
{
    if (nElems <= 0) {
        return;
    }
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(nElems * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params);
    AscendC::PipeBarrier<PIPE_ALL>();
}

/** 整段 AtomicAdd 写 GM（对齐 CUDA Hist→row_counts） */
__aicore__ inline void AtomicAddOutInt32(const AscendC::GlobalTensor<int32_t> &dst,
                                         const AscendC::LocalTensor<int32_t> &src, int64_t nElems)
{
    if (nElems <= 0) {
        return;
    }
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(nElems * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params);
    AscendC::SetAtomicNone();
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void FlushGmInt32(const AscendC::GlobalTensor<int32_t> &gm)
{
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(gm);
}

/** 单点写 GM（MTE3），多核并发安全；避免 GlobalTensor.SetValue 写回丢失 */
__aicore__ inline void StoreScalarInt32(const AscendC::GlobalTensor<int32_t> &dst, int64_t index, int32_t value,
                                        const AscendC::LocalTensor<int32_t> &tmp)
{
    tmp.SetValue(0, value);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPad(dst[index], tmp, params);
}

/**
 * 同 pos 写 q_ind + slot：一次 barrier + 两次 4B MTE3。
 * tmpQ/tmpS 须各自 32B 对齐（不可用 tmp[1] 作源）。
 */
__aicore__ inline void StorePairInt32(const AscendC::GlobalTensor<int32_t> &qInd,
                                      const AscendC::GlobalTensor<int32_t> &slotGm, int64_t index, int32_t qVal,
                                      int32_t slotVal, const AscendC::LocalTensor<int32_t> &tmpQ,
                                      const AscendC::LocalTensor<int32_t> &tmpS)
{
    tmpQ.SetValue(0, qVal);
    tmpS.SetValue(0, slotVal);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPad(qInd[index], tmpQ, params);
    AscendC::DataCopyPad(slotGm[index], tmpS, params);
}

/**
 * 写缓冲批量刷：q/s 槽按 32B 步长排布（源 LocalTensor 必须 32B 对齐）。
 * 布局：qValBuf[i*kAlignInts]=q，sBuf[i*kAlignInts]=s；一批仅首尾各 1 次 barrier。
 */
constexpr int32_t STORE_SLOT_ALIGN_INTS = 8; // 32B / sizeof(int32)

__aicore__ inline void FlushStoreBatch(const AscendC::GlobalTensor<int32_t> &qInd,
                                       const AscendC::GlobalTensor<int32_t> &slotGm, int64_t base,
                                       const AscendC::LocalTensor<int32_t> &posBuf,
                                       const AscendC::LocalTensor<int32_t> &qValBuf,
                                       const AscendC::LocalTensor<int32_t> &sBuf, int32_t n)
{
    if (n <= 0) {
        return;
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
    for (int32_t i = 0; i < n; ++i) {
        int64_t idx = base + static_cast<int64_t>(posBuf.GetValue(i));
        int32_t off = i * STORE_SLOT_ALIGN_INTS;
        AscendC::DataCopyPad(qInd[idx], qValBuf[off], params);
        AscendC::DataCopyPad(slotGm[idx], sBuf[off], params);
    }
    // 等本批 MTE3 完成再复用写缓冲
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace K2qCsrCommon

#endif
