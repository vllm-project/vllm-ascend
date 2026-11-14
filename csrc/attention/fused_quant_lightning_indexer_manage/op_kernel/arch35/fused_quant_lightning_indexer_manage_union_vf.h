#ifndef FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_UNION_VF_H
#define FUSED_QUANT_LIGHTNING_INDEXER_MANAGE_UNION_VF_H

#include "kernel_operator.h"
#include "lim_request_limits.h"

namespace FusedQuantLiManageUnionVF {
using namespace AscendC;

constexpr uint32_t LANES = 64U;

// MrgSort emits sorted (key, source) pairs. Compare each source with its
// global predecessor, including at vector boundaries, then compact once.
// Invalid lanes gather pair zero, never beyond the last valid input pair.
// No packed GatherMask predicate/repeat geometry or scalar O(total) walk.
__simd_vf__ void DeduplicateSortedPairs(
    __ubuf__ uint32_t *dst, __ubuf__ uint32_t *pairs, uint32_t total)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> zero;
    MicroAPI::RegTensor<uint32_t> one;
    MicroAPI::RegTensor<uint32_t> limit;
    MicroAPI::RegTensor<uint32_t> sentinel;
    MicroAPI::Duplicate(zero, 0U);
    MicroAPI::Duplicate(one, 1U);
    MicroAPI::Duplicate(limit, total);
    MicroAPI::Duplicate(sentinel, 0xffffffffU); // outside the 17-bit source domain
    MicroAPI::ClearSpr<SpecialPurposeReg::AR>();
    MicroAPI::UnalignRegForStore alignOut;

    for (uint32_t base = 0U; base < total; base += LANES) {
        MicroAPI::RegTensor<int32_t> positions;
        MicroAPI::Arange(positions, static_cast<int32_t>(base));
        auto &position = (MicroAPI::RegTensor<uint32_t>&)positions;
        MicroAPI::MaskReg valid;
        MicroAPI::MaskReg first;
        MicroAPI::MaskReg hasPrevious;
        MicroAPI::Compare<uint32_t, CMPMODE::LT>(valid, position, limit, all);
        MicroAPI::Compare<uint32_t, CMPMODE::EQ>(first, position, zero, all);
        MicroAPI::Compare<uint32_t, CMPMODE::GT>(hasPrevious, position, zero, all);
        MicroAPI::And(hasPrevious, hasPrevious, valid, all);

        MicroAPI::RegTensor<uint32_t> currentOffset;
        MicroAPI::RegTensor<uint32_t> previousOffset;
        MicroAPI::Select(currentOffset, position, zero, valid);
        MicroAPI::Sub(previousOffset, position, one, all);
        MicroAPI::Select(previousOffset, previousOffset, zero, hasPrevious);
        MicroAPI::ShiftLefts(currentOffset, currentOffset, static_cast<int16_t>(1), all);
        MicroAPI::ShiftLefts(previousOffset, previousOffset, static_cast<int16_t>(1), all);
        MicroAPI::Add(currentOffset, currentOffset, one, all);
        MicroAPI::Add(previousOffset, previousOffset, one, all);

        MicroAPI::RegTensor<uint32_t> current;
        MicroAPI::RegTensor<uint32_t> previous;
        MicroAPI::Gather(current, pairs, currentOffset, all);
        MicroAPI::Gather(previous, pairs, previousOffset, all);
        MicroAPI::Select(previous, sentinel, previous, first);
        MicroAPI::MaskReg keep;
        MicroAPI::Compare<uint32_t, CMPMODE::NE>(keep, current, previous, all);
        MicroAPI::And(keep, keep, valid, all);
        MicroAPI::RegTensor<uint32_t> compact;
        MicroAPI::Squeeze<uint32_t, MicroAPI::GatherMaskMode::STORE_REG>(
            compact, current, keep);
        MicroAPI::StoreUnAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dst, compact, alignOut);
    }
    MicroAPI::StoreUnAlignPost(dst, alignOut);
}

} // namespace FusedQuantLiManageUnionVF

#endif
