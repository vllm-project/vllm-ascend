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

__aicore__ inline void CopyInInt32(const AscendC::LocalTensor<int32_t> &dst,
                                   const AscendC::GlobalTensor<int32_t> &src, int64_t nElems)
{
    if (nElems <= 0) {
        return;
    }
    AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(nElems * sizeof(int32_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params, pad);
    AscendC::PipeBarrier<PIPE_ALL>();
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

} // namespace K2qCsrCommon

#endif
