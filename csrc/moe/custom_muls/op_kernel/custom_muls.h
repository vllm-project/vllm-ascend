#ifndef CUSTOM_MULS_H
#define CUSTOM_MULS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "custom_muls_tiling_data.h"

using optiling::CustomMulsTilingData;

namespace NsCustomMuls {
template <typename T, bool PROMOTE_FP32>
class CustomMuls {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const CustomMulsTilingData* tilingData)
    {
        if (AscendC::GetBlockIdx() >= tilingData->blockNum) {
            blockLength_ = 0;
            return;
        }
        const bool isTailBlock = (AscendC::GetBlockIdx() + 1) == tilingData->blockNum;
        blockLength_ = isTailBlock ? tilingData->blockTail : tilingData->blockFormer;
        tileCount_ = isTailBlock ? tilingData->ubLoopTail : tilingData->ubLoopFormer;
        tailCount_ = isTailBlock ? tilingData->ubTailTail : tilingData->ubTailFormer;
        ubLength_ = tilingData->ubFormer;
        scalarValue_ = tilingData->scalarValue;

        const int64_t offset = static_cast<int64_t>(AscendC::GetBlockIdx()) * tilingData->blockFormer;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + offset, blockLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + offset, blockLength_);
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, ubLength_ * sizeof(T));
        if constexpr (PROMOTE_FP32) {
            pipe_.InitBuffer(xFp32Buf_, ubLength_ * sizeof(float));
            pipe_.InitBuffer(yFp32Buf_, ubLength_ * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }
        for (uint32_t i = 0; i < tileCount_; ++i) {
            const uint32_t valid = ((i + 1) == tileCount_) ? tailCount_ : ubLength_;
            CopyIn(i, valid);
            Compute(valid);
            CopyOut(i, valid);
        }
    }

private:
    static constexpr uint32_t BUFFER_NUM = 2;

    __aicore__ inline void CopyIn(int64_t i, uint32_t n)
    {
        AscendC::LocalTensor<T> x = inQueueX_.AllocTensor<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{true, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(x, xGm_[i * static_cast<int64_t>(ubLength_)], params, pad);
        inQueueX_.EnQue(x);
    }

    __aicore__ inline void Compute(uint32_t n)
    {
        AscendC::LocalTensor<T> x = inQueueX_.DeQue<T>();
        AscendC::LocalTensor<T> y = outQueueY_.AllocTensor<T>();
        if constexpr (PROMOTE_FP32) {
            AscendC::LocalTensor<float> xf = xFp32Buf_.Get<float>();
            AscendC::LocalTensor<float> yf = yFp32Buf_.Get<float>();
            AscendC::Cast(xf, x, AscendC::RoundMode::CAST_NONE, n);
            AscendC::Muls(yf, xf, scalarValue_, n);
            AscendC::Cast(y, yf, AscendC::RoundMode::CAST_RINT, n);
        } else {
            AscendC::Muls(y, x, scalarValue_, n);
        }
        outQueueY_.EnQue(y);
        inQueueX_.FreeTensor(x);
    }

    __aicore__ inline void CopyOut(int64_t i, uint32_t n)
    {
        AscendC::LocalTensor<T> y = outQueueY_.DeQue<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[i * static_cast<int64_t>(ubLength_)], y, params);
        outQueueY_.FreeTensor(y);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFp32Buf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    int64_t blockLength_ = 0;
    uint32_t ubLength_ = 0;
    uint32_t tileCount_ = 0;
    uint32_t tailCount_ = 0;
    float scalarValue_ = 0.0F;
};
} // namespace NsCustomMuls

#endif // CUSTOM_MULS_H
