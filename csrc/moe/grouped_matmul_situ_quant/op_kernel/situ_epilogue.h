/*
 * SiTU + MXQuant epilogue numerics — VERBATIM adaptation of the production
 * situ_mx_quant arch35 regbase implementation (vllm-ascend csrc/moe/situ_mx_quant,
 * CANN Open Software License). Bit-exactness with the production kernel requires
 * the identical MicroAPI instruction sequence; do not "optimize" op order here.
 *
 * Adaptations vs the original:
 *  - namespace renamed (SituMxQuant -> GmsqEpilogue)
 *  - platform::GetVRegSize()/GetUbBlockSize() inlined as constants (256/32)
 *  - ops::CeilDivision -> local CeilDivU32
 */
#ifndef GMSQ_SITU_EPILOGUE_H
#define GMSQ_SITU_EPILOGUE_H

#include "kernel_operator.h"

// The kernel_macros.h MicroAPI alias is gated on __NPU_ARCH__ == 3510, which the
// precompile (-E) pass does not define. Provide a fallback alias for that pass.
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
namespace AscendC {
namespace MicroAPI = Reg;
} // namespace AscendC
#endif

namespace GmsqEpilogue {
namespace plat {
__aicore__ inline constexpr uint32_t GetVRegSize() { return 256U; }
__aicore__ inline constexpr uint32_t GetUbBlockSize() { return 32U; }
} // namespace plat
__aicore__ inline uint32_t CeilDivision(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

// ==================== Type Traits ====================
template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

// ==================== Constants ====================
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400;
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t QUANT_ONCE_NUM = 256;
constexpr int64_t X_ONCE_NUM = 512;
constexpr int64_t QUANT_ONCE_NUM_FP4 = 128;
constexpr int64_t SCALE_ONCE_NUM = 8;
constexpr int64_t CONST_64 = 64;
constexpr int64_t CONST_32 = 32;
constexpr int64_t CONST_2 = 2;
constexpr int64_t CONST_4 = 4;
constexpr uint32_t VF_LEN_T = plat::GetVRegSize() / sizeof(half);     // 128
constexpr uint32_t VF_LEN_FP32 = plat::GetVRegSize() / sizeof(float); // 64
constexpr uint32_t ONE_BLOCK_UB = plat::GetUbBlockSize();
constexpr uint32_t ONE_BLOCK_NUM = ONE_BLOCK_UB / sizeof(half); // 16

// ==================== Cast Traits ====================
static constexpr AscendC::MicroAPI::CastTrait CAST_ZERO = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_ONE = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
static constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_BF16 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_80 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_81 = {
    AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_82 = {
    AscendC::MicroAPI::RegLayout::TWO, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
static constexpr AscendC::MicroAPI::CastTrait CAST_32_TO_83 = {
    AscendC::MicroAPI::RegLayout::THREE, AscendC::MicroAPI::SatMode::SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

using namespace AscendC;

// ===================================================================
// MxQuant helper: Extract BF16 exponent and compute per-32-block max
// Adapted from swiglu_mx_quant_common.h (BF16-only path)
// ===================================================================
template <typename T>
__aicore__ inline void ComputeVfMaxExpVfLast(__ubuf__ T* srcAddr, __ubuf__ uint16_t* maxExpAddr, int64_t dim0OnceSize,
                                             int64_t alignDim1Size)
{
    uint32_t totalCountInUB = dim0OnceSize * alignDim1Size;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    uint16_t maxExpbf16 = MAX_EXP_FOR_BF16;
    int64_t onceNum = QUANT_ONCE_NUM;
    int64_t scaleNum = SCALE_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaskBF16, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMaskBF16, maxExpbf16);
        AscendC::MicroAPI::MaskReg scaleMask1;
        AscendC::MicroAPI::UnalignReg u1;
        for (uint16_t i = 0; i < loopNum; i++) {
            scaleMask1 = AscendC::MicroAPI::UpdateMask<T>(totalCountInUB);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr, onceNum);
            AscendC::MicroAPI::And(vdExpExtract0, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp0, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::And(vdExpExtract1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vdExp1, expMaskBF16,
                                   scaleMask1);
            AscendC::MicroAPI::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, scaleMask1);
            AscendC::MicroAPI::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, scaleMask1);
            AscendC::MicroAPI::DataCopyUnAlign<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, u1, scaleNum);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(maxExpAddr, u1, 0);
    }
}

// ===================================================================
// MxQuant helper: Compute E8M0 scale and reciprocal scale (OCP algorithm)
// Adapted from swiglu_mx_quant_common.h
// ===================================================================
template <typename T>
__aicore__ inline void ComputeScaleLast(uint16_t fEmax, __ubuf__ uint16_t* maxExpAddr,
                                        __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                        int64_t dim0OnceSize, int64_t alignDim1Size)
{
    uint32_t totalScaleInUB = dim0OnceSize * (alignDim1Size / CONST_32);
    uint16_t loopNumScale = CeilDivision(totalScaleInUB, QUANT_ONCE_NUM_FP4);
    uint16_t maxExpBf16 = MAX_EXP_FOR_BF16;
    int64_t onceNum = QUANT_ONCE_NUM_FP4;
    int64_t onceNumMxScale = CONST_64;
    uint16_t bf16ExpBias = BF16_EXP_BIAS;
    uint16_t maxExpFp8 = MAX_EXP_FOR_FP8;
    uint16_t nanCustomZation = NAN_CUSTOMIZATION;
    uint16_t specailExpThreshold = SPECIAL_EXP_THRESHOLD;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> expMask, vdMaxExp;
        AscendC::MicroAPI::Duplicate(expMask, maxExpBf16);
        AscendC::MicroAPI::MaskReg cmpResult, zeroMask, cmpResultSub, preMaskScale;
        AscendC::MicroAPI::RegTensor<uint16_t> maxExpValue, sharedExp, scaleValue, scaleBias, halfScale;
        AscendC::MicroAPI::Duplicate(maxExpValue, fEmax);
        AscendC::MicroAPI::Duplicate(scaleBias, bf16ExpBias);
        AscendC::MicroAPI::RegTensor<uint16_t> fp8NanRegTensor, zeroRegTensor, nanRegTensor;
        AscendC::MicroAPI::Duplicate(fp8NanRegTensor, maxExpFp8);
        AscendC::MicroAPI::Duplicate(zeroRegTensor, 0);
        AscendC::MicroAPI::Duplicate(nanRegTensor, nanCustomZation);
        AscendC::MicroAPI::MaskReg invalidDataMask, specialDataMask;
        AscendC::MicroAPI::RegTensor<uint16_t> specialExpRegTensor;
        AscendC::MicroAPI::Duplicate(specialExpRegTensor, specailExpThreshold);
        for (uint16_t i = 0; i < loopNumScale; i++) {
            preMaskScale = AscendC::MicroAPI::UpdateMask<uint16_t>(totalScaleInUB);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, onceNum);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);
            AscendC::MicroAPI::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            AscendC::MicroAPI::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_PACK_B16>(mxScaleLocalAddr, scaleValue,
                                                                                     onceNumMxScale, preMaskScale);
            AscendC::MicroAPI::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            AscendC::MicroAPI::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            AscendC::MicroAPI::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                halfScaleLocalAddr, halfScale, onceNum, preMaskScale);
        }
    }
}

// ===================================================================
// MxQuant helper: Quantize BF16 data to FP8 (multiply by reciprocal scale, then cast)
// Adapted from swiglu_mx_quant_common.h (BF16-only path)
// ===================================================================
template <typename T, typename U>
__aicore__ inline void ComputeDataF8Last(__ubuf__ T* srcAddr, __ubuf__ uint16_t* halfScaleLocalAddr,
                                         __ubuf__ int8_t* outLocalAddr, int64_t dim0OnceSize, int64_t dim1AlignSize)
{
    uint32_t totalCountInUB = dim0OnceSize * dim1AlignSize;
    uint16_t loopNum = CeilDivision(totalCountInUB, QUANT_ONCE_NUM);
    int64_t elementAfterReduce = SCALE_ONCE_NUM;
    int64_t onceXNum = QUANT_ONCE_NUM;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint16_t> halfScaleForMul;
        AscendC::MicroAPI::RegTensor<T> vdExp0, vdExp1;
        AscendC::MicroAPI::RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        AscendC::MicroAPI::RegTensor<float> vdExp1FP32Zero, vdExp1FP32One;
        AscendC::MicroAPI::RegTensor<U> vdExp0FP8Zero, vdExp0FP8One;
        AscendC::MicroAPI::RegTensor<U> vdExp1FP8Zero, vdExp1FP8One;
        AscendC::MicroAPI::MaskReg
            maskAll = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg
            maskAllB8 = AscendC::MicroAPI::CreateMask<uint8_t, AscendC::MicroAPI::MaskPattern::ALL>();
        for (uint16_t i = 0; i < loopNum; i++) {
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, srcAddr,
                                                                                      onceXNum);
            AscendC::MicroAPI::DataCopy<uint16_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::LoadDist::DIST_E2B_B16>(halfScaleForMul, halfScaleLocalAddr,
                                                                                   elementAfterReduce);
            // BF16 path: multiply in BF16 domain, then cast to FP32, then to FP8
            AscendC::MicroAPI::Mul(vdExp0, vdExp0, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Mul(vdExp1, vdExp1, (AscendC::MicroAPI::RegTensor<T>&)halfScaleForMul, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp0FP32Zero, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp0FP32One, vdExp0, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vdExp1FP32Zero, vdExp1, maskAll);
            AscendC::MicroAPI::Cast<float, T, CAST_ONE>(vdExp1FP32One, vdExp1, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_80>(vdExp0FP8Zero, vdExp0FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_82>(vdExp0FP8One, vdExp0FP32One, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_81>(vdExp1FP8Zero, vdExp1FP32Zero, maskAll);
            AscendC::MicroAPI::Cast<U, float, CAST_32_TO_83>(vdExp1FP8One, vdExp1FP32One, maskAll);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8One, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8Zero, maskAllB8);
            AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp0FP8Zero,
                                   (AscendC::MicroAPI::RegTensor<uint8_t>&)vdExp1FP8One, maskAllB8);
            AscendC::MicroAPI::DataCopy<int8_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE,
                                        AscendC::MicroAPI::StoreDist::DIST_NORM_B8>(
                outLocalAddr, (AscendC::MicroAPI::RegTensor<int8_t>&)vdExp0FP8Zero, onceXNum, maskAllB8);
        }
    }
}

// ===================================================================
// Situ activation: beta * tanh(gate / beta) * sigmoid(gate) * up
//                  (+ optional linear_beta * tanh(up / linear_beta) on up)
// Replaces ComputeVfSwigluV1 from swiglu_mx_quant
// ===================================================================
template <typename T, bool hasLinearBeta>
__aicore__ inline void ComputeVfSitu(__local_mem__ T* gateUbAddr, __local_mem__ T* upUbAddr,
                                     __local_mem__ T* situUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize,
                                     int64_t dim1AlignSize, float beta, float invBeta, float linearBeta,
                                     float invLinearBeta)
{
    uint16_t dim0VfTimes = dim0OnceSize;
    uint16_t dim1VfTimes = dim1OnceSize / VF_LEN_FP32;
    uint32_t dim1Tail = dim1OnceSize % VF_LEN_FP32;
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    uint32_t alignDim1In = ((dim1OnceSize + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM) * ONE_BLOCK_NUM;
    uint32_t alignDim1Out = dim1AlignSize;
    auto gateUbAddr1 = gateUbAddr;
    auto upUbAddr1 = upUbAddr;
    auto situUbAddr1 = situUbAddr;
    auto situUbAddr2 = situUbAddr;
    T numZero = 0;
    if (dim1Tail > 0) {
        mask1Num = dim1Tail;
        dim1TailTimes = 1;
        uint32_t padNum = alignDim1Out - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        int32_t offsetAlgin = dim1VfTimes * VF_LEN_FP32;
        gateUbAddr1 = gateUbAddr + offsetAlgin;
        upUbAddr1 = upUbAddr + offsetAlgin;
        situUbAddr1 = situUbAddr + offsetAlgin;
        situUbAddr2 = situUbAddr + offsetAlgin + dim1TailTimes * VF_LEN_FP32;
    }
    float scalarOne = 1.0f;
    float negScalarOne = -1.0f;
    float scalarTwo = 2.0f;
    float negTwo = -2.0f;
    // Two-path tanh (adapted from tanh.h reference):
    //   |x| < 0.6:  degree-9 polynomial, FMA Horner (matches tanh.h exactly)
    //   |x| >= 0.6: sigmoid decomposition, sign naturally preserved
    float tanhC1 = -0.333327681f;
    float tanhC2 = 0.133152977f;
    float tanhC3 = -0.0523039624f;
    float tanhC4 = 0.0157396831f;
    float tanhThreshold = 0.6f;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregGate;
        AscendC::MicroAPI::RegTensor<T> vregUp;
        AscendC::MicroAPI::RegTensor<float> gateF;
        AscendC::MicroAPI::RegTensor<float> upF;
        AscendC::MicroAPI::RegTensor<float> gateDivBeta;
        AscendC::MicroAPI::RegTensor<float> polyReg;     // sigmoid path result
        AscendC::MicroAPI::RegTensor<float> x2;          // x² for Horner / temp
        AscendC::MicroAPI::MaskReg cmpMask;              // comparison result for Select
        AscendC::MicroAPI::RegTensor<float> negGate;
        AscendC::MicroAPI::RegTensor<float> expReg;      // sigmoid Exp
        AscendC::MicroAPI::RegTensor<float> oneReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;  // sigmoid result / linear_beta work reg
        AscendC::MicroAPI::RegTensor<float> c1Reg;       // tanh polynomial coeff c1 (preloaded)
        AscendC::MicroAPI::RegTensor<float> c2Reg;       // tanh polynomial coeff c2 (preloaded)
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<T> outTReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<T>(mask3Num);
        AscendC::MicroAPI::Duplicate(oneReg, scalarOne);
        AscendC::MicroAPI::Duplicate(c1Reg, tanhC1);
        AscendC::MicroAPI::Duplicate(c2Reg, tanhC2);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset = AscendC::MicroAPI::CreateAddrReg<T>(
                    dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, 64);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGate, gateUbAddr,
                                                                                             srcIdxOffset);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregUp, upUbAddr,
                                                                                             srcIdxOffset);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(gateF, vregGate, mask);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(upF, vregUp, mask);

                // Two-path tanh(gate/beta) — adapted from tanh.h reference:
                //   small |x|: degree-7 polynomial (Horner)
                //   large |x|: sigmoid decomposition on |x|, sign restore
                AscendC::MicroAPI::Muls(gateDivBeta, gateF, invBeta, mask); // x = gate/beta

                // --- Polynomial path (all x, used for |x| < 0.6) ---
                // tanh(x) ≈ x * (1 + c1*x² + c2*x⁴ + c3*x⁶ + c4*x⁸)
                // FMA Horner (matches tanh.h reference: 7 ops, 7 roundings)
                AscendC::MicroAPI::Mul(x2, gateDivBeta, gateDivBeta, mask);     // x²
                AscendC::MicroAPI::Muls(sigmoidReg, x2, tanhC4, mask);          // c4*x²
                AscendC::MicroAPI::Adds(sigmoidReg, sigmoidReg, tanhC3, mask);  // +c3
                AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask); // *x²+c2
                AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask); // *x²+c1
                AscendC::MicroAPI::Mul(sigmoidReg, sigmoidReg, x2, mask);       // *x²
                AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, gateDivBeta, gateDivBeta, mask); // *x+x = x*(p+1)

                // --- Sigmoid path (used for |x| >= 0.6) ---
                // tanh(x) = 2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1, sign naturally preserved
                AscendC::MicroAPI::Muls(negGate, gateDivBeta, negTwo, mask); // -2x
                AscendC::MicroAPI::Exp(expReg, negGate, mask);
                AscendC::MicroAPI::Adds(expReg, expReg, scalarOne, mask);    // 1+exp(-2x)
                AscendC::MicroAPI::Div(polyReg, oneReg, expReg, mask);       // sigmoid = 1/(1+exp(-2x))
                AscendC::MicroAPI::Muls(polyReg, polyReg, scalarTwo, mask);  // 2*sigmoid
                AscendC::MicroAPI::Adds(polyReg, polyReg, negScalarOne, mask); // 2*sigmoid - 1

                // --- Path selection: sigmoid if |x| >= 0.6, else polynomial ---
                AscendC::MicroAPI::Muls(x2, gateDivBeta, negScalarOne, mask);
                AscendC::MicroAPI::Max(x2, gateDivBeta, x2, mask);           // |x|
                AscendC::MicroAPI::Duplicate(expReg, tanhThreshold);         // 0.6
                AscendC::MicroAPI::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask);
                AscendC::MicroAPI::Select(sigmoidReg, polyReg, sigmoidReg, cmpMask);
                // sigmoidReg = tanh(gate/beta) — save to negGate (free after |x|)
                AscendC::MicroAPI::Mul(negGate, sigmoidReg, oneReg, mask);   // negGate = tanh

                // sigmoid(gate) = 1 / (1 + exp(-gate))  → result in polyReg
                AscendC::MicroAPI::Muls(polyReg, gateF, negScalarOne, mask); // -gate
                AscendC::MicroAPI::Exp(expReg, polyReg, mask);
                AscendC::MicroAPI::Adds(expReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(polyReg, oneReg, expReg, mask);       // sigmoid(gate)

                // situ_a = beta * tanh * sigmoid
                AscendC::MicroAPI::Mul(polyReg, negGate, polyReg, mask);     // tanh * sigmoid
                AscendC::MicroAPI::Muls(polyReg, polyReg, beta, mask);       // * beta

                // Optional: up = linear_beta * tanh(up / linear_beta)
                // Uses sigmoidReg/negGate as work registers to preserve polyReg (situ_a)
                if constexpr (hasLinearBeta) {
                    AscendC::MicroAPI::Muls(upF, upF, invLinearBeta, mask); // x = up/lb

                    // Poly path → sigmoidReg (FMA Horner)
                    AscendC::MicroAPI::Mul(x2, upF, upF, mask);
                    AscendC::MicroAPI::Muls(sigmoidReg, x2, tanhC4, mask);
                    AscendC::MicroAPI::Adds(sigmoidReg, sigmoidReg, tanhC3, mask);
                    AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask);
                    AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask);
                    AscendC::MicroAPI::Mul(sigmoidReg, sigmoidReg, x2, mask);
                    AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, upF, upF, mask);

                    // Sigmoid path on x (sign naturally preserved) → negGate
                    AscendC::MicroAPI::Muls(expReg, upF, negTwo, mask);      // -2x
                    AscendC::MicroAPI::Exp(expReg, expReg, mask);
                    AscendC::MicroAPI::Adds(expReg, expReg, scalarOne, mask); // 1+exp(-2x)
                    AscendC::MicroAPI::Div(negGate, oneReg, expReg, mask);
                    AscendC::MicroAPI::Muls(negGate, negGate, scalarTwo, mask);
                    AscendC::MicroAPI::Adds(negGate, negGate, negScalarOne, mask); // 2*sig-1

                    // Path selection → sigmoidReg = tanh(up/lb)
                    AscendC::MicroAPI::Muls(x2, upF, negScalarOne, mask);
                    AscendC::MicroAPI::Max(x2, upF, x2, mask);               // |x|
                    AscendC::MicroAPI::Duplicate(expReg, tanhThreshold);
                    AscendC::MicroAPI::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask);
                    AscendC::MicroAPI::Select(sigmoidReg, negGate, sigmoidReg, cmpMask);

                    AscendC::MicroAPI::Muls(upF, sigmoidReg, linearBeta, mask);
                }

                // situOut = situ_a * up
                AscendC::MicroAPI::Mul(outFReg, polyReg, upF, mask);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out,
                                                                                           dim1vfLoopIdx, 64);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(situUbAddr, outTReg, outOffset, mask);
            }
            // Handle tail elements
            AscendC::MicroAPI::AddrReg srcIdxOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregGate, gateUbAddr1,
                                                                                             srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregUp, upUbAddr1,
                                                                                             srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(gateF, vregGate, mask1);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(upF, vregUp, mask1);

                // Two-path tanh(gate/beta) — tail path
                AscendC::MicroAPI::Muls(gateDivBeta, gateF, invBeta, mask1);

                // Poly path → sigmoidReg (FMA Horner)
                AscendC::MicroAPI::Mul(x2, gateDivBeta, gateDivBeta, mask1);
                AscendC::MicroAPI::Muls(sigmoidReg, x2, tanhC4, mask1);
                AscendC::MicroAPI::Adds(sigmoidReg, sigmoidReg, tanhC3, mask1);
                AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask1);
                AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask1);
                AscendC::MicroAPI::Mul(sigmoidReg, sigmoidReg, x2, mask1);
                AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, gateDivBeta, gateDivBeta, mask1);

                // Sigmoid path on x → polyReg: tanh(x) = 2/(1+exp(-2x)) - 1
                AscendC::MicroAPI::Muls(negGate, gateDivBeta, negTwo, mask1);
                AscendC::MicroAPI::Exp(expReg, negGate, mask1);
                AscendC::MicroAPI::Adds(expReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(polyReg, oneReg, expReg, mask1);
                AscendC::MicroAPI::Muls(polyReg, polyReg, scalarTwo, mask1);
                AscendC::MicroAPI::Adds(polyReg, polyReg, negScalarOne, mask1);

                // Path selection → sigmoidReg = tanh
                AscendC::MicroAPI::Muls(x2, gateDivBeta, negScalarOne, mask1);
                AscendC::MicroAPI::Max(x2, gateDivBeta, x2, mask1);           // |x|
                AscendC::MicroAPI::Duplicate(expReg, tanhThreshold);
                AscendC::MicroAPI::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask1);
                AscendC::MicroAPI::Select(sigmoidReg, polyReg, sigmoidReg, cmpMask);
                AscendC::MicroAPI::Mul(negGate, sigmoidReg, oneReg, mask1); // save tanh

                // sigmoid(gate) → polyReg
                AscendC::MicroAPI::Muls(polyReg, gateF, negScalarOne, mask1);
                AscendC::MicroAPI::Exp(expReg, polyReg, mask1);
                AscendC::MicroAPI::Adds(expReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(polyReg, oneReg, expReg, mask1);

                // situ_a = beta * tanh * sigmoid
                AscendC::MicroAPI::Mul(polyReg, negGate, polyReg, mask1);
                AscendC::MicroAPI::Muls(polyReg, polyReg, beta, mask1);

                // Optional: up = linear_beta * tanh(up / linear_beta)
                if constexpr (hasLinearBeta) {
                    AscendC::MicroAPI::Muls(upF, upF, invLinearBeta, mask1);

                    AscendC::MicroAPI::Mul(x2, upF, upF, mask1);
                    AscendC::MicroAPI::Muls(sigmoidReg, x2, tanhC4, mask1);
                    AscendC::MicroAPI::Adds(sigmoidReg, sigmoidReg, tanhC3, mask1);
                    AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c2Reg, mask1);
                    AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, x2, c1Reg, mask1);
                    AscendC::MicroAPI::Mul(sigmoidReg, sigmoidReg, x2, mask1);
                    AscendC::MicroAPI::FusedMulDstAdd(sigmoidReg, upF, upF, mask1);

                    // Sigmoid path on x → negGate: tanh(x) = 2/(1+exp(-2x)) - 1
                    AscendC::MicroAPI::Muls(expReg, upF, negTwo, mask1);
                    AscendC::MicroAPI::Exp(expReg, expReg, mask1);
                    AscendC::MicroAPI::Adds(expReg, expReg, scalarOne, mask1);
                    AscendC::MicroAPI::Div(negGate, oneReg, expReg, mask1);
                    AscendC::MicroAPI::Muls(negGate, negGate, scalarTwo, mask1);
                    AscendC::MicroAPI::Adds(negGate, negGate, negScalarOne, mask1);

                    // Path selection → sigmoidReg = tanh(up/lb)
                    AscendC::MicroAPI::Muls(x2, upF, negScalarOne, mask1);
                    AscendC::MicroAPI::Max(x2, upF, x2, mask1);               // |x|
                    AscendC::MicroAPI::Duplicate(expReg, tanhThreshold);
                    AscendC::MicroAPI::Compare<float, CMPMODE::GE>(cmpMask, x2, expReg, mask1);
                    AscendC::MicroAPI::Select(sigmoidReg, negGate, sigmoidReg, cmpMask);

                    AscendC::MicroAPI::Muls(upF, sigmoidReg, linearBeta, mask1);
                }

                // situOut = situ_a * up
                AscendC::MicroAPI::Mul(outFReg, polyReg, upF, mask1);

                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_BF16>(outTReg, outFReg, mask1);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(situUbAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                AscendC::MicroAPI::Duplicate<T>(vregGate, numZero);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(situUbAddr2, vregGate, outOffset1, mask3);
            }
        }
    }
}

} // namespace GmsqEpilogue

#endif // GMSQ_SITU_EPILOGUE_H
