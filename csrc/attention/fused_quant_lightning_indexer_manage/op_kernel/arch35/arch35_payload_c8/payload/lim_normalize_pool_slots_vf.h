#ifndef LIM_NORMALIZE_POOL_SLOTS_VF_H
#define LIM_NORMALIZE_POOL_SLOTS_VF_H

#include "kernel_operator.h"

namespace lim_normalize_pool_slots_vf {
using namespace AscendC;

// The A5 payload consumes int16 slots. Normalize negative int32 pool entries
// before narrowing: INT32_MIN must not become slot zero after conversion.
__simd_vf__ inline void Normalize(__ubuf__ uint32_t *slots, uint32_t loops)
{
    MicroAPI::MaskReg all =
        MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint32_t> invalid;
    MicroAPI::RegTensor<uint32_t> signBit;
    MicroAPI::Duplicate(invalid, 0xffffffffU);
    MicroAPI::Duplicate(signBit, 0x80000000U);
    for (uint32_t index = 0U; index < loops; ++index) {
        MicroAPI::RegTensor<uint32_t> value;
        MicroAPI::LoadAlign<uint32_t, MicroAPI::LoadDist::DIST_NORM>(
            value, slots + index * 64U);
        MicroAPI::MaskReg negative;
        MicroAPI::Compare<uint32_t, CMPMODE::GE>(
            negative, value, signBit, all);
        MicroAPI::RegTensor<uint32_t> normalized;
        MicroAPI::Select(normalized, invalid, value, negative);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(
            slots + index * 64U, normalized, all);
    }
}

} // namespace lim_normalize_pool_slots_vf

#endif // LIM_NORMALIZE_POOL_SLOTS_VF_H
