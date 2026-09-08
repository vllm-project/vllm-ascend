/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef VLLM_ASCEND_RMS_NORM_CAST_ARCH35_KERNEL_H
#define VLLM_ASCEND_RMS_NORM_CAST_ARCH35_KERNEL_H

#include "kernel_operator.h"

namespace RmsNormCastArch35 {
using namespace AscendC;
using namespace AscendC::MicroAPI;

constexpr uint32_t B16_PER_BLOCK = 16;
constexpr uint32_t VECTOR_LENGTH = AscendC::VECTOR_REG_WIDTH / sizeof(float);

constexpr CastTrait CAST_B16_TO_FP32 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr CastTrait CAST_FP32_TO_B16 = {
    RegLayout::ZERO,
    SatMode::NO_SAT,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__simd_vf__ void RmsNormCastVF(__local_mem__ T* x, __local_mem__ T* gamma,
                               __local_mem__ T* y,
                               __local_mem__ float* yFp32,
                               uint32_t numCol, float invNumCol,
                               float epsilon)
{
    RegTensor<float> sum;
    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
    MaskReg firstMask = CreateMask<float, MaskPattern::VL1>();
    Duplicate(sum, 0.0f);

    uint32_t remaining = numCol;
    const uint16_t repeats = CeilDivision(numCol, VECTOR_LENGTH);
    for (uint16_t i = 0; i < repeats; ++i) {
        MaskReg mask = UpdateMask<float>(remaining);
        RegTensor<T> xB16;
        RegTensor<float> xFp32;
        RegTensor<float> square;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xB16,
                                               x + i * VECTOR_LENGTH);
        Cast<float, T, CAST_B16_TO_FP32>(xFp32, xB16, mask);
        Mul(square, xFp32, xFp32, mask);
        Add(sum, sum, square, fullMask);
    }

    ReduceSum(sum, sum, fullMask);
    Muls(sum, sum, invNumCol, firstMask);
    Adds(sum, sum, epsilon, firstMask);
    Sqrt(sum, sum, firstMask);
    RegTensor<float> one;
    RegTensor<float> rstdScalar;
    RegTensor<float> rstd;
    Duplicate(one, 1.0f, firstMask);
    Div(rstdScalar, one, sum, firstMask);
    Duplicate<float, HighLowPart::LOWEST, MaskMergeMode::ZEROING>(
        rstd, rstdScalar, fullMask);

    remaining = numCol;
    for (uint16_t i = 0; i < repeats; ++i) {
        MaskReg mask = UpdateMask<float>(remaining);
        RegTensor<T> xB16;
        RegTensor<T> gammaB16;
        RegTensor<T> yB16;
        RegTensor<float> xFp32;
        RegTensor<float> gammaFp32;
        RegTensor<float> normalized;
        RegTensor<float> widened;
        const uint32_t offset = i * VECTOR_LENGTH;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xB16, x + offset);
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(gammaB16,
                                               gamma + offset);
        Cast<float, T, CAST_B16_TO_FP32>(xFp32, xB16, mask);
        Cast<float, T, CAST_B16_TO_FP32>(gammaFp32, gammaB16, mask);
        Mul(normalized, xFp32, rstd, mask);
        Mul(normalized, normalized, gammaFp32, mask);
        Cast<T, float, CAST_FP32_TO_B16>(yB16, normalized, mask);
        DataCopy<T, StoreDist::DIST_PACK_B32>(y + offset, yB16, mask);

        // HashTopK must consume exactly the low-precision RMSNorm result
        // widened to FP32, rather than the unrounded FP32 intermediate.
        Cast<float, T, CAST_B16_TO_FP32>(widened, yB16, mask);
        DataCopy<float, StoreDist::DIST_NORM>(yFp32 + offset, widened,
                                              mask);
    }
}

template <typename T>
class KernelRmsNormCast {
public:
    __aicore__ inline explicit KernelRmsNormCast(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                GM_ADDR yFp32,
                                const RmsNormCastTilingData* tiling)
    {
        numRow_ = tiling->num_row;
        numCol_ = tiling->num_col;
        numColAligned_ = tiling->num_col_aligned;
        rowsPerCore_ = tiling->rows_per_core;
        invNumCol_ = tiling->inv_num_col;
        epsilon_ = tiling->epsilon;
        const uint32_t rowBegin = GetBlockIdx() * rowsPerCore_;
        rowBegin_ = rowBegin;
        rowEnd_ = rowBegin + rowsPerCore_ < numRow_
                      ? rowBegin + rowsPerCore_
                      : numRow_;

        xGm_.SetGlobalBuffer((__gm__ T*)x, numRow_ * numCol_);
        gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, numCol_);
        yGm_.SetGlobalBuffer((__gm__ T*)y, numRow_ * numCol_);
        yFp32Gm_.SetGlobalBuffer((__gm__ float*)yFp32,
                                 numRow_ * numCol_);

        pipe_->InitBuffer(xBuf_, numColAligned_ * sizeof(T));
        pipe_->InitBuffer(gammaBuf_, numColAligned_ * sizeof(T));
        pipe_->InitBuffer(yBuf_, numColAligned_ * sizeof(T));
        pipe_->InitBuffer(yFp32Buf_, numColAligned_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rowBegin_ >= numRow_) {
            return;
        }
        LocalTensor<T> gammaLocal = gammaBuf_.Get<T>();
        CopyIn(gammaLocal, gammaGm_, numCol_);
        SyncMte2ToVector();

        for (uint32_t row = rowBegin_; row < rowEnd_; ++row) {
            LocalTensor<T> xLocal = xBuf_.Get<T>();
            LocalTensor<T> yLocal = yBuf_.Get<T>();
            LocalTensor<float> yFp32Local = yFp32Buf_.Get<float>();
            const uint32_t offset = row * numCol_;
            CopyIn(xLocal, xGm_[offset], numCol_);
            SyncMte2ToVector();
            RmsNormCastVF<T>((__local_mem__ T*)xLocal.GetPhyAddr(),
                             (__local_mem__ T*)gammaLocal.GetPhyAddr(),
                             (__local_mem__ T*)yLocal.GetPhyAddr(),
                             (__local_mem__ float*)yFp32Local.GetPhyAddr(),
                             numCol_, invNumCol_, epsilon_);
            SyncVectorToMte3();
            CopyOut(yGm_[offset], yLocal, numCol_);
            CopyOut(yFp32Gm_[offset], yFp32Local, numCol_);
            SyncMte3ToMte2();
        }
    }

private:
    template <typename U>
    __aicore__ inline void CopyIn(LocalTensor<U> dst, GlobalTensor<U> src,
                                  uint32_t count)
    {
        DataCopyExtParams params{
            1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        const uint32_t aligned =
            CeilDivision(count, static_cast<uint32_t>(32 / sizeof(U))) *
            (32 / sizeof(U));
        DataCopyPadExtParams<U> pad{true, 0,
                                    static_cast<uint8_t>(aligned - count),
                                    static_cast<U>(0)};
        DataCopyPad(dst, src, params, pad);
    }

    template <typename U>
    __aicore__ inline void CopyOut(GlobalTensor<U> dst, LocalTensor<U> src,
                                   uint32_t count)
    {
        DataCopyExtParams params{
            1, static_cast<uint32_t>(count * sizeof(U)), 0, 0, 0};
        DataCopyPad(dst, src, params);
    }

    __aicore__ inline void SyncMte2ToVector()
    {
        event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void SyncVectorToMte3()
    {
        event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
    }

    __aicore__ inline void SyncMte3ToMte2()
    {
        event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
    }

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> gammaBuf_;
    TBuf<TPosition::VECCALC> yBuf_;
    TBuf<TPosition::VECCALC> yFp32Buf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> yFp32Gm_;
    uint32_t numRow_;
    uint32_t numCol_;
    uint32_t numColAligned_;
    uint32_t rowsPerCore_;
    uint32_t rowBegin_;
    uint32_t rowEnd_;
    float invNumCol_;
    float epsilon_;
};
}  // namespace RmsNormCastArch35
#endif
