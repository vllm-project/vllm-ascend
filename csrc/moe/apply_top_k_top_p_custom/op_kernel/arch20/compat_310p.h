/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * 310P compat helpers for apply_top_k_top_p_custom.
 *
 * Ascend 310P bans DataCopyPad; this header provides DataCopyPadCustom
 * (GM->UB, aligned copy + tail-mask contract on consumer) and
 * DataCopyCustom (UB->GM, with optional needBack read-modify-write for
 * unaligned tails). Both are drop-in replacements for the banned
 * DataCopyPad call signature. Ported verbatim from
 * csrc/attention/recurrent_gated_delta_rule_v310/op_kernel/recurrent_gated_delta_rule_v310.h
 * (which has been production-tested on 310P).
 *
 * CONSUMER CONTRACT for DataCopyPadCustom(): when `blockLen / sizeof(T)`
 * is not a multiple of BLOCK_BYTES/sizeof(T), the copy over-reads the
 * remainder of the last block. Consumers doing reduce / max / abs /
 * multiply over the buffer MUST mask the UB tail (Duplicate to identity
 * or SetVectorMask) before consumption.
 */
#ifndef APPLY_TOPKP_COMPAT_310P_H
#define APPLY_TOPKP_COMPAT_310P_H

#include "kernel_operator.h"

// Dummy bfloat16_t only needed on 310P (dav_m200) where the compiler
// doesn't provide a native bf16 type. 310P has no bf16 hardware support;
// the model dispatches to fp16 at runtime, but the AscendC build system
// still compiles the bf16 dtype variant of the op, and that variant
// references bfloat16_t. Provide a stub so the compile succeeds; the
// generated bf16 kernel binary is never loaded on 310P.
// Mirrored from
// csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/arch20/compat_310p.h.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200) && !defined(__bfloat16_t_defined)
#define __bfloat16_t_defined
#define __COMPAT_310P_ACTIVE__
struct bfloat16_t {
    uint16_t val;
    bfloat16_t() = default;
    bfloat16_t(float v) : val(0) { (void)v; }
    operator float() const { return 0.f; }
};
#endif

// 310P has no AscendC::ToFloat — the dummy bfloat16_t already has
// operator float(), so route ToFloat through it.
#ifdef __COMPAT_310P_ACTIVE__
namespace AscendC {
    inline float ToFloat(bfloat16_t v) { return (float)v; }
}
#endif

namespace ApplyTopKTopPCompat310P {

using namespace AscendC;

constexpr int64_t COMPAT_BLOCK_BYTES = 32;

template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

// GM -> UB. Replaces DataCopyPad(LocalTensor, GlobalTensor, DataCopyExtParams, DataCopyPadExtParams).
// Aligned fast path when blockLen and srcStride are 32B-aligned; otherwise
// per-row aligned reads. Ignores padParams (padValue) on the fallback path;
// consumer must mask the UB tail.
template <typename T>
__aicore__ inline void DataCopyPadCustom(LocalTensor<T> inLocal, GlobalTensor<T> srcGm,
                                        DataCopyExtParams tokenCopyParams,
                                        DataCopyPadExtParams<T> padParams)
{
    int64_t elem = tokenCopyParams.blockLen / sizeof(T);
    int64_t numPerBlock = COMPAT_BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);
    int64_t srcStrideElem = tokenCopyParams.srcStride / sizeof(T);
    int64_t gmStepPerRow = elem + srcStrideElem;

    if (likely(alignElem == elem && srcStrideElem == 0)) {
        DataCopyParams copyParams = {tokenCopyParams.blockCount,
                                     static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(inLocal, srcGm, copyParams);
    } else {
        DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        for (uint32_t i = 0; i < tokenCopyParams.blockCount; i++) {
            DataCopy(inLocal[i * alignElem], srcGm[i * gmStepPerRow], copyParams);
        }
    }
    (void)padParams;  // unused on 310P fallback
}

// UB -> GM. Replaces DataCopyPad(GlobalTensor, LocalTensor, DataCopyExtParams).
// Aligned fast path when elem is 32B-aligned; unaligned path has three
// variants:
//   - needBack=true:  read-modify-write to preserve out-of-range bytes
//                      (correct for real vocab-length final writes).
//   - isAtomic=true:  zero the UB tail before over-write (safe if the
//                      target GM was zeroed and we're accumulating).
//   - default:         over-write with garbage in the tail (safe only if
//                      the caller guarantees `elem == alignElem`).
template <typename T, bool needBack = false, bool isAtomic = false>
__aicore__ inline void DataCopyCustom(GlobalTensor<T> dstGm, LocalTensor<T> inLocal,
                                       DataCopyExtParams copyParamsIn)
{
    int64_t elem = copyParamsIn.blockLen / sizeof(T);
    int64_t numPerBlock = sizeof(T) == 0 ? 1 : COMPAT_BLOCK_BYTES / sizeof(T);
    int64_t alignElem = AlignUp(elem, numPerBlock);

    if (likely(alignElem == elem)) {
        DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                     static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
        DataCopy(dstGm, inLocal, copyParams);
    } else {
        if (copyParamsIn.blockCount == 1) {
            if constexpr (needBack) {
                int64_t elemAlignDown = numPerBlock == 0 ? 0 : elem / numPerBlock * numPerBlock;
                if (elemAlignDown != 0) {
                    DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                                 static_cast<uint16_t>(elemAlignDown / numPerBlock), 0, 0};
                    DataCopy(dstGm, inLocal, copyParams);
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    for (uint32_t i = 0; i < numPerBlock; i++) {
                        inLocal.SetValue(alignElem - 1 - i, inLocal.GetValue(elem - 1 - i));
                    }
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopyParams copyParamslast = {1, 1, 0, 0};
                    DataCopy(dstGm[elem - numPerBlock], inLocal[elemAlignDown], copyParamslast);
                } else {
                    T tmp[COMPAT_BLOCK_BYTES];
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    for (uint32_t i = 0; i < elem; i++) {
                        tmp[i] = inLocal.GetValue(elem - 1 - i);
                    }
                    DataCopyParams copyParamslast = {1, 1, 0, 0};
                    SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
                    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                    DataCopy(inLocal, dstGm[elem - numPerBlock], copyParamslast);
                    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                    for (uint32_t i = 0; i < elem; i++) {
                        inLocal.SetValue(numPerBlock - 1 - i, tmp[i]);
                    }
                    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                    DataCopy(dstGm[elem - numPerBlock], inLocal, copyParamslast);
                }
            } else if constexpr (isAtomic) {
                SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                for (uint32_t i = 0; i < alignElem - elem; i++) {
                    inLocal.SetValue(alignElem - 1 - i, T(0));
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                             static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
                DataCopy(dstGm, inLocal, copyParams);
            } else {
                DataCopyParams copyParams = {static_cast<uint16_t>(copyParamsIn.blockCount),
                                             static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
                DataCopy(dstGm, inLocal, copyParams);
            }
        } else {
            DataCopyParams copyParams = {1, static_cast<uint16_t>(alignElem / numPerBlock), 0, 0};
            for (uint32_t i = 0; i < copyParamsIn.blockCount; i++) {
                DataCopy(dstGm[i * elem], inLocal[i * alignElem], copyParams);
                PipeBarrier<PIPE_MTE3>();
            }
        }
    }
}

}  // namespace ApplyTopKTopPCompat310P

#endif  // APPLY_TOPKP_COMPAT_310P_H
