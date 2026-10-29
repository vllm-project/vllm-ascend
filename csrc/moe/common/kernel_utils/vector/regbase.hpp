
using namespace AscendC::MicroAPI;

#pragma once

constexpr static CastTrait ctHalf2Fp32Zero = {
    RegLayout::ZERO,
    SatMode::SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};
constexpr static CastTrait ctHalf2Fp32One = {
    RegLayout::ONE,
    SatMode::SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE,
};
constexpr static CastTrait ctFp322HalfZero = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::MERGING,
    AscendC::RoundMode::CAST_ROUND
};
constexpr static CastTrait ctFp322HalfOne = {
    RegLayout::ONE,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND
};

constexpr uint16_t PRELOAD_NUM = 2;

template <typename TType, bool BroatCast = false>
__simd_callee__ inline void LoadIn(RegTensor<TType>& dstReg, __ubuf__ TType* srcUb)
{
    if constexpr (BroatCast) {
        if constexpr (!std::is_same<TType, float>()) {
            LoadAlign<TType, LoadDist::DIST_BRC_B16>(dstReg, srcUb);
        } else {
            LoadAlign<TType, LoadDist::DIST_BRC_B32>(dstReg, srcUb);
        }
    } else {
        LoadAlign<TType, LoadDist::DIST_NORM>(dstReg, srcUb);
    }
}

template <typename TType>
__simd_callee__ inline void CastHalf2Float(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg, RegTensor<TType>& srcReg, MaskReg& mask)
{
    Cast<float, TType, ctHalf2Fp32Zero>(dstZeroReg, srcReg, mask);
    Cast<float, TType, ctHalf2Fp32One>(dstOneReg, srcReg, mask);
}

template <typename TType>
__simd_callee__ inline void CastFloat2Half(RegTensor<TType>& dstReg, RegTensor<float>& srcZeroReg, RegTensor<float>& srcOneReg, MaskReg& mask)
{
    Cast<TType, float, ctFp322HalfOne>(dstReg, srcOneReg, mask);
    Cast<TType, float, ctFp322HalfZero>(dstReg, srcZeroReg, mask);
}

template <typename TType>
__simd_callee__ inline void HalfOrFloat2Float(RegTensor<float>& dstReg, RegTensor<TType>& srcReg, MaskReg& maskHalf, MaskReg& maskFloat)
{
    if constexpr (!std::is_same<TType, float>()) {
        Cast<float, TType, ctHalf2Fp32Zero>(dstReg, srcReg, maskHalf);
    } else {
        Duplicate(dstReg, srcReg, maskFloat);
    }
}

__simd_callee__ inline void MulFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& src1ZeroReg, RegTensor<float>& src1OneReg,
                                           RegTensor<float>& src2ZeroReg, RegTensor<float>& src2OneReg,
                                           MaskReg& maskFloat)
{
    Mul(dstZeroReg, src1ZeroReg, src2ZeroReg, maskFloat);
    Mul(dstOneReg, src1OneReg, src2OneReg, maskFloat);
}

__simd_callee__ inline void MinsFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& srcZeroReg, RegTensor<float>& srcOneReg,
                                           float scalarValue,
                                           MaskReg& maskFloat)
{
    Mins(dstZeroReg, srcZeroReg, scalarValue, maskFloat);
    Mins(dstOneReg, srcOneReg, scalarValue, maskFloat);
}

__simd_callee__ inline void ExpFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& srcZeroReg, RegTensor<float>& srcOneReg,
                                           MaskReg& maskFloat)
{
    Exp(dstZeroReg, srcZeroReg, maskFloat);
    Exp(dstOneReg, srcOneReg, maskFloat);
}

__simd_callee__ inline void SubFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& src1ZeroReg, RegTensor<float>& src1OneReg,
                                           RegTensor<float>& src2ZeroReg, RegTensor<float>& src2OneReg,
                                           MaskReg& maskFloat)
{
    Sub(dstZeroReg, src1ZeroReg, src2ZeroReg, maskFloat);
    Sub(dstOneReg, src1OneReg, src2OneReg, maskFloat);
}

__simd_callee__ inline void AddFloatTwoReg(RegTensor<float>& dstZeroReg, RegTensor<float>& dstOneReg,
                                           RegTensor<float>& src1ZeroReg, RegTensor<float>& src1OneReg,
                                           RegTensor<float>& src2ZeroReg, RegTensor<float>& src2OneReg,
                                           MaskReg& maskFloat)
{
    Add(dstZeroReg, src1ZeroReg, src2ZeroReg, maskFloat);
    Add(dstOneReg, src1OneReg, src2OneReg, maskFloat);
}

template <typename TType = float, AscendC::CMPMODE mode = AscendC::CMPMODE::EQ>
__simd_callee__ inline void CompareTwoReg(MaskReg& maskZeroReg, MaskReg& maskOneReg,
                                           RegTensor<TType>& cmp1ZeroReg, RegTensor<TType>& cmp1OneReg,
                                           RegTensor<TType>& cmp2ZeroReg, RegTensor<TType>& cmp2OneReg,
                                           MaskReg& mask)
{
    Compare<TType, mode>(maskZeroReg, cmp1ZeroReg, cmp2ZeroReg, mask);
    Compare<TType, mode>(maskOneReg, cmp1OneReg, cmp2OneReg, mask);
}

template <typename TType = float>
__simd_callee__ inline void SelectTwoReg(RegTensor<TType>& dstZeroReg, RegTensor<TType>& dstOneReg,
                                           RegTensor<TType>& src1ZeroReg, RegTensor<TType>& src1OneReg,
                                           RegTensor<TType>& src2ZeroReg, RegTensor<TType>& src2OneReg,
                                           MaskReg& mask1, MaskReg& mask2)
{
    Select<TType>(dstZeroReg, src1ZeroReg, src2ZeroReg, mask1);
    Select<TType>(dstOneReg, src1OneReg, src2OneReg, mask2);
}

template <typename TType>
__simd_callee__ inline void StoreUnAlignOut(__ubuf__ TType* dstUb, RegTensor<float>& srcReg, MaskReg& mask, UnalignRegForStore& uStore, uint32_t copyNum)
{
    RegTensor<TType> outZeroReg;
    if constexpr (!std::is_same<TType, float32_t>()) {
        Cast<TType, float, ctFp322HalfZero>(outZeroReg, srcReg, mask);
        StoreUnAlign(dstUb, outZeroReg, uStore, copyNum);
        StoreUnAlignPost(dstUb, uStore, 0);
    } else {
        StoreUnAlign(dstUb, srcReg, uStore, copyNum);
        StoreUnAlignPost(dstUb, uStore, 0);
    }
}
