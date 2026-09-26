// SPDX-License-Identifier: Apache-2.0
#ifndef FLASH_ATTN_C8_QUERY_SCALE_H
#define FLASH_ATTN_C8_QUERY_SCALE_H

#include "../../a5_mla_common/op_kernel/memcopy/fa_gm_tensor.h"
#include "../../a5_mla_common/op_kernel/memcopy/attn_copy_gm_to_l1.h"
#include "../../a5_mla_common/op_kernel/memcopy/attn_copy_gm_to_ub.h"

// This specialization is visible only in the new FlashAttnC8 translation
// unit, before its vector block is instantiated. Existing MLA decode
// kernels retain the original common helper and their compiled pipeline.
// Expanded attention has N2=H, G=1: TNG scales for one head are strided
// across tokens, unlike the N2=1 compressed MLA decode case.
template <>
class CopyQueryScaleGmToUb<float, GmFormat::TNG> {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaUbTensor<float> &dstTensor,
        FaGmTensorType &srcTensor, GmCoordGs1Merge &gmCoord)
    {
        auto &offsetCalculator = srcTensor.offsetCalculator;
        const uint64_t gSize = offsetCalculator.GetDimG();
        if (offsetCalculator.GetDimN2() != 1) {
            const auto vectorToScalar = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vectorToScalar);
            WaitFlag<HardEvent::V_S>(vectorToScalar);
            for (uint32_t row = 0; row < gmCoord.gS1DealSize; ++row) {
                const uint64_t gs1 = static_cast<uint64_t>(gmCoord.gS1Idx) + row;
                const uint64_t offset = offsetCalculator.GetOffset(
                    gmCoord.bIdx, gmCoord.n2Idx, gs1 % gSize, gs1 / gSize);
                dstTensor.tensor.SetValue(row, srcTensor.gmTensor.GetValue(offset));
            }
            const auto scalarToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            return;
        }
        const uint64_t offset = offsetCalculator.GetOffset(gmCoord.bIdx,
            gmCoord.n2Idx, gmCoord.gS1Idx % gSize, gmCoord.gS1Idx / gSize);
        DataCopyExtParams params{1, static_cast<uint32_t>(gmCoord.gS1DealSize * sizeof(float)), 0, 0, 0};
        DataCopyPad(dstTensor.tensor, srcTensor.gmTensor[offset], params,
            DataCopyPadExtParams<float>{false, 0, 0, 0});
    }
};
#endif
