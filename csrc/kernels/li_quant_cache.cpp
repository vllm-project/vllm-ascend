// SPDX-License-Identifier: Apache-2.0
// Six-row specialization matching npu_dynamic_quant arithmetic and rounding.
#include "kernel_operator.h"

namespace {
constexpr int64_t ROWS = 6;
constexpr int64_t WIDTH = 128;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr float INV_INT8_MAX = 0.0078740157186985015869140625f;

template <typename Input, typename Slot>
class LIQuantCache {
public:
    __aicore__ inline explicit LIQuantCache(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(__gm__ void* x, __gm__ void* slots, __gm__ void* keyCache,
                                __gm__ void* scaleCache, __gm__ void* key, __gm__ void* scale,
                                int64_t capacity)
    {
        row_ = AscendC::GetBlockIdx();
        capacity_ = capacity;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ Input*>(x));
        slots_.SetGlobalBuffer(reinterpret_cast<__gm__ Slot*>(slots));
        keyCache_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(keyCache));
        scaleCache_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scaleCache));
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(key));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scale));
        pipe_->InitBuffer(inputQueue_, 1, WIDTH * sizeof(Input));
        pipe_->InitBuffer(slotQueue_, 1, 2 * BLOCK_BYTES);
        pipe_->InitBuffer(outputQueue_, 1, WIDTH + BLOCK_BYTES);
        pipe_->InitBuffer(workBuffer_, WIDTH * sizeof(float));
        pipe_->InitBuffer(absBuffer_, WIDTH * sizeof(float));
        pipe_->InitBuffer(ratioBuffer_, WIDTH * sizeof(float));
        pipe_->InitBuffer(integerBuffer_, WIDTH * sizeof(int32_t));
        pipe_->InitBuffer(halfBuffer_, WIDTH * sizeof(half));
        pipe_->InitBuffer(maxBuffer_, BLOCK_BYTES);
        pipe_->InitBuffer(reduceBuffer_, BLOCK_BYTES);
        pipe_->InitBuffer(maskBuffer_, BLOCK_BYTES);
    }

    __aicore__ inline void Process()
    {
        using namespace AscendC;
        LocalTensor<Input> input = inputQueue_.AllocTensor<Input>();
        DataCopyPad(input, x_[row_ * WIDTH],
                    DataCopyExtParams{1, WIDTH * sizeof(Input), 0, 0, 0},
                    DataCopyPadExtParams<Input>{false, 0, 0, static_cast<Input>(0)});
        inputQueue_.EnQue(input);
        LocalTensor<Slot> slotData = slotQueue_.AllocTensor<Slot>();
        DataCopyPad(slotData, slots_,
                    DataCopyExtParams{1, ROWS * sizeof(Slot), 0, 0, 0},
                    DataCopyPadExtParams<Slot>{false, 0, 0, 0});
        slotQueue_.EnQue(slotData);
        slotData = slotQueue_.DeQue<Slot>();
        event_t copyToScalar = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(copyToScalar);
        WaitFlag<HardEvent::MTE2_S>(copyToScalar);
        const int64_t slot = static_cast<int64_t>(slotData.GetValue(row_));
        slotQueue_.FreeTensor(slotData);

        input = inputQueue_.DeQue<Input>();
        LocalTensor<float> work = workBuffer_.Get<float>();
        LocalTensor<float> absolute = absBuffer_.Get<float>();
        LocalTensor<float> maximum = maxBuffer_.Get<float>();
        LocalTensor<float> scratch = reduceBuffer_.Get<float>();
        Cast(work, input, RoundMode::CAST_NONE, WIDTH);
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
        Abs(absolute, work, WIDTH);
        PipeBarrier<PIPE_V>();
        ReduceMax(maximum, absolute, scratch, WIDTH);
        event_t vectorToScalar = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        const float maxValue = maximum.GetValue(0);

        LocalTensor<int8_t> output = outputQueue_.AllocTensor<int8_t>();
        LocalTensor<half> scaleOutput = output[WIDTH].ReinterpretCast<half>();
        LocalTensor<half> halfValues = halfBuffer_.Get<half>();
        if (maxValue == 0.0f || maxValue != maxValue || maxValue > 3.40282346638528859812e+38f) {
            // Native oracle: +/-zero => (0, +0); NaN anywhere => all q=0,
            // scale-half bits 0x7fff; +/-Inf anywhere => all q=0, +Inf scale.
            Duplicate(halfValues, static_cast<half>(0), WIDTH);
            PipeBarrier<PIPE_V>();
            Cast(output, halfValues, RoundMode::CAST_TRUNC, WIDTH);
            uint16_t scaleBits = maxValue != maxValue ? 0x7fff : (maxValue == 0.0f ? 0 : 0x7c00);
            scaleOutput.ReinterpretCast<uint16_t>().SetValue(0, scaleBits);
        } else {
            LocalTensor<float> ratio = ratioBuffer_.Get<float>();
            // Do not replace the hardware vector division with host/CPU
            // arithmetic. Native halfway cases distinguish those results.
            Duplicate(absolute, maxValue, WIDTH);
            Duplicate(ratio, 127.0f, WIDTH);
            PipeBarrier<PIPE_V>();
            Div(ratio, ratio, absolute, WIDTH);
            PipeBarrier<PIPE_V>();
            Mul(work, work, ratio, WIDTH);
            PipeBarrier<PIPE_V>();
            if (maxValue < 4.0e-37f) {
                // Reciprocal overflow is observable for tiny BF16 rows:
                // native saturates +/-Inf to 127/-128 and maps 0*Inf to 0.
                LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
                Compare(mask, work, work, CMPMODE::EQ, WIDTH);
                PipeBarrier<PIPE_V>();
                Select(work, mask, work, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, WIDTH);
                PipeBarrier<PIPE_V>();
                Mins(work, work, 127.0f, WIDTH);
                PipeBarrier<PIPE_V>();
                Maxs(work, work, -128.0f, WIDTH);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<int32_t> rounded = integerBuffer_.Get<int32_t>();
            Cast(rounded, work, RoundMode::CAST_RINT, WIDTH);
            PipeBarrier<PIPE_V>();
            SetDeqScale(static_cast<half>(1.0f));
            PipeBarrier<PIPE_V>();
            Cast(halfValues, rounded, RoundMode::CAST_NONE, WIDTH);
            PipeBarrier<PIPE_V>();
            Cast(output, halfValues, RoundMode::CAST_TRUNC, WIDTH);
            // Raw native FP32 scales matched multiply-by-inv127 in every
            // tested finite row; dividing by 127 did not match their bits.
            Muls(maximum, maximum, INV_INT8_MAX, 1);
            PipeBarrier<PIPE_V>();
            Cast(scaleOutput, maximum, RoundMode::CAST_NONE, 1);
        }
        outputQueue_.EnQue(output);
        output = outputQueue_.DeQue<int8_t>();
        scaleOutput = output[WIDTH].ReinterpretCast<half>();
        event_t scalarToCopy = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarToCopy);
        WaitFlag<HardEvent::S_MTE3>(scalarToCopy);
        const DataCopyExtParams keyCopy{1, WIDTH, 0, 0, 0};
        const DataCopyExtParams scaleCopy{1, sizeof(half), 0, 0, 0};
        DataCopyPad(key_[row_ * WIDTH], output, keyCopy);
        DataCopyPad(scale_[row_], scaleOutput, scaleCopy);
        if (slot >= 0 && slot < capacity_) {
            DataCopyPad(keyCache_[slot * WIDTH], output, keyCopy);
            DataCopyPad(scaleCache_[slot], scaleOutput, scaleCopy);
        }
        outputQueue_.FreeTensor(output);
    }

private:
    AscendC::TPipe* pipe_;
    int64_t row_;
    int64_t capacity_;
    AscendC::GlobalTensor<Input> x_;
    AscendC::GlobalTensor<Slot> slots_;
    AscendC::GlobalTensor<int8_t> keyCache_, key_;
    AscendC::GlobalTensor<half> scaleCache_, scale_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inputQueue_, slotQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> workBuffer_, absBuffer_, ratioBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> integerBuffer_, halfBuffer_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> maxBuffer_, reduceBuffer_, maskBuffer_;
};
}  // namespace

#define LI_QUANT_ENTRY(NAME, INPUT, SLOT) \
extern "C" __global__ __aicore__ void NAME( \
    __gm__ void* x, __gm__ void* slots, __gm__ void* keyCache, __gm__ void* scaleCache, \
    __gm__ void* key, __gm__ void* scale, int64_t capacity) \
{ \
    AscendC::TPipe pipe; \
    LIQuantCache<INPUT, SLOT> kernel(&pipe); \
    kernel.Init(x, slots, keyCache, scaleCache, key, scale, capacity); \
    kernel.Process(); \
}

LI_QUANT_ENTRY(li_quant_cache_f16_i32, half, int32_t)
LI_QUANT_ENTRY(li_quant_cache_f16_i64, half, int64_t)
LI_QUANT_ENTRY(li_quant_cache_bf16_i32, bfloat16_t, int32_t)
LI_QUANT_ENTRY(li_quant_cache_bf16_i64, bfloat16_t, int64_t)

namespace vllm_ascend {
void li_quant_cache_impl(void* stream, void* x, void* slots, void* keyCache,
                         void* scaleCache, void* key, void* scale,
                         int64_t capacity, int64_t bf16, int64_t slots64)
{
    // ROWS is independent work, not a hardcoded device core count.
    if (bf16 && slots64) {
        li_quant_cache_bf16_i64<<<ROWS, nullptr, stream>>>(x, slots, keyCache, scaleCache, key, scale, capacity);
    } else if (bf16) {
        li_quant_cache_bf16_i32<<<ROWS, nullptr, stream>>>(x, slots, keyCache, scaleCache, key, scale, capacity);
    } else if (slots64) {
        li_quant_cache_f16_i64<<<ROWS, nullptr, stream>>>(x, slots, keyCache, scaleCache, key, scale, capacity);
    } else {
        li_quant_cache_f16_i32<<<ROWS, nullptr, stream>>>(x, slots, keyCache, scaleCache, key, scale, capacity);
    }
}
}  // namespace vllm_ascend
