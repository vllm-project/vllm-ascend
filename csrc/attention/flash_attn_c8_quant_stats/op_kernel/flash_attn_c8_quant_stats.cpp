// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "flash_attn_c8_quant_stats_tiling_data.h"

namespace FlashAttnC8Stats {
using namespace AscendC;
constexpr MicroAPI::CastTrait BF16_TO_FLOAT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
template <HardEvent Event> __aicore__ inline void Sync()
{
    SetFlag<Event>(0);
    WaitFlag<Event>(0);
}
__aicore__ inline void AbsMax(__local_mem__ bfloat16_t *input, __local_mem__ float *output, uint16_t count)
{
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<bfloat16_t> raw;
        MicroAPI::RegTensor<float> x, maximum;
        auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(maximum, 0.0f, mask);
        for (uint16_t offset = 0; offset < count; offset += 64) {
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(raw, input + offset);
            MicroAPI::Cast<float, bfloat16_t, BF16_TO_FLOAT>(x, raw, mask);
            MicroAPI::Abs(x, x, mask);
            MicroAPI::Max(maximum, maximum, x, mask);
        }
        MicroAPI::ReduceMax(maximum, maximum, mask);
        MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(output, maximum, mask);
    }
}
class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR partial,
        const optiling::FlashAttnC8QuantStatsTilingData &tiling)
    {
        data = tiling;
        keyGm.SetGlobalBuffer((__gm__ bfloat16_t *)key);
        valueGm.SetGlobalBuffer((__gm__ bfloat16_t *)value);
        partialGm.SetGlobalBuffer((__gm__ float *)partial);
        pipe.InitBuffer(inputBuffer, 256 * 128 * sizeof(bfloat16_t));
        pipe.InitBuffer(reductionBuffer, 32);
        pipe.InitBuffer(outputBuffer, 64);
    }
    __aicore__ inline float Reduce(GlobalTensor<bfloat16_t> &source, int64_t start, int64_t end,
        int64_t head, int64_t tokenStride, int64_t headStride)
    {
        auto input = inputBuffer.Get<bfloat16_t>();
        auto reduction = reductionBuffer.Get<float>();
        float maximum = 0.0f;
        for (int64_t token = start; token < end; token += 256) {
            const uint32_t rows = static_cast<uint32_t>(end - token < 256 ? end - token : 256);
            Sync<HardEvent::V_MTE2>();
            DataCopyExtParams copy{static_cast<uint16_t>(rows), 256,
                static_cast<uint32_t>((tokenStride - 128) * 2), 0, 0};
            DataCopyPad(input, source[token * tokenStride + head * headStride], copy,
                DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            Sync<HardEvent::MTE2_V>();
            AbsMax((__local_mem__ bfloat16_t *)input.GetPhyAddr(),
                (__local_mem__ float *)reduction.GetPhyAddr(), rows * 128);
            Sync<HardEvent::V_S>();
            const float tileMax = reduction.GetValue(0);
            maximum = maximum > tileMax ? maximum : tileMax;
        }
        return maximum;
    }
    __aicore__ inline void Process()
    {
        const int64_t parts = (data.tokens + FlashAttnC8Config::STATS_TOKENS - 1) / FlashAttnC8Config::STATS_TOKENS;
        auto output = outputBuffer.Get<float>();
        for (int64_t task = GetBlockIdx(); task < parts * data.heads; task += GetBlockNum()) {
            const int64_t part = task / data.heads, head = task % data.heads;
            const int64_t start = part * FlashAttnC8Config::STATS_TOKENS;
            const int64_t limit = start + FlashAttnC8Config::STATS_TOKENS;
            const int64_t end = limit < data.tokens ? limit : data.tokens;
            float keyMax = Reduce(keyGm, start, end, head, data.keyTokenStride, data.keyHeadStride);
            float valueMax = Reduce(valueGm, start, end, head, data.valueTokenStride, data.valueHeadStride);
            Sync<HardEvent::MTE3_S>();
            for (int i = 0; i < 8; ++i) {
                output.SetValue(i, keyMax);
                output.SetValue(8 + i, valueMax);
            }
            Sync<HardEvent::S_MTE3>();
            DataCopy(partialGm[task * 16], output, 16);
        }
    }
private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, reductionBuffer, outputBuffer;
    GlobalTensor<bfloat16_t> keyGm, valueGm;
    GlobalTensor<float> partialGm;
    optiling::FlashAttnC8QuantStatsTilingData data;
};
}
extern "C" __global__ __aicore__ void flash_attn_c8_quant_stats(GM_ADDR key, GM_ADDR value,
    GM_ADDR partial, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttnC8QuantStatsTilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    FlashAttnC8Stats::Kernel kernel;
    kernel.Init(key, value, partial, data);
    kernel.Process();
}
