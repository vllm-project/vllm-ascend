// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "flash_attn_c8_prepare_tiling_data.h"

namespace FlashAttnC8Prepare {
using namespace AscendC;
constexpr MicroAPI::CastTrait TO_FLOAT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr MicroAPI::CastTrait FROM_FLOAT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::DivSpecificMode DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, false};
template <HardEvent Event> __aicore__ inline void Sync()
{
    SetFlag<Event>(0);
    WaitFlag<Event>(0);
}

// Kind: 0=query192, 1=key128+rope64, 2=value128. Every VF writes
// materialized FP8 bytes. The fake-quant VF reads those bytes separately.
template <int Kind>
__aicore__ inline void Quantize(__local_mem__ bfloat16_t *input, __local_mem__ fp8_e4m3fn_t *output,
    __local_mem__ bfloat16_t *rope, __local_mem__ float *queryScale, __local_mem__ float *scales,
    __local_mem__ bfloat16_t *keyRope, uint16_t startHead, uint16_t rows, uint16_t heads, bool sharedRope)
{
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<bfloat16_t> raw0, raw1, rawRope, rope16;
        MicroAPI::RegTensor<fp8_e4m3fn_t> code;
        MicroAPI::RegTensor<float> x0, x1, maximum, abs1, scale, keyScale, one, normalized, r;
        auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg zeros;
        for (uint16_t row = 0; row < rows; ++row) {
            const uint16_t head = (startHead + row) % heads;
            constexpr uint32_t inputWidth = Kind == 0 ? 192 : 128;
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(raw0, input + row * inputWidth);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(raw1, input + row * inputWidth + 64);
            MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(x0, raw0, mask);
            MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(x1, raw1, mask);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(keyScale, scales + head);
            if constexpr (Kind == 0) {
                MicroAPI::Abs(maximum, x0, mask);
                MicroAPI::Abs(abs1, x1, mask);
                MicroAPI::Max(maximum, maximum, abs1, mask);
                MicroAPI::ReduceMax(maximum, maximum, mask);
                MicroAPI::Duplicate(scale, maximum, mask);
                MicroAPI::Muls(scale, scale, 1.0f / 448.0f, mask);
                MicroAPI::Duplicate(one, 1.0f, mask);
                MicroAPI::CompareScalar<float, CMPMODE::EQ>(zeros, scale, 0.0f, mask);
                MicroAPI::Select(scale, one, scale, zeros);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(queryScale + row, scale, mask);
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale,
                    scales + head + (Kind == 2 ? 256 : 0));
            }
            MicroAPI::Div<float, &DIV_MODE>(normalized, x0, scale, mask);
            MicroAPI::Mins(normalized, normalized, 448.0f, mask);
            MicroAPI::Maxs(normalized, normalized, -448.0f, mask);
            MicroAPI::Cast<fp8_e4m3fn_t, float, FROM_FLOAT>(code, normalized, mask);
            MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(output + row * 128, code, mask);
            MicroAPI::Div<float, &DIV_MODE>(normalized, x1, scale, mask);
            MicroAPI::Mins(normalized, normalized, 448.0f, mask);
            MicroAPI::Maxs(normalized, normalized, -448.0f, mask);
            MicroAPI::Cast<fp8_e4m3fn_t, float, FROM_FLOAT>(code, normalized, mask);
            MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::StoreDist::DIST_PACK4_B32>(output + row * 128 + 64, code, mask);
            if constexpr (Kind != 2) {
                if constexpr (Kind == 0) {
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawRope, input + row * 192 + 128);
                    MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(r, rawRope, mask);
                    MicroAPI::Div<float, &DIV_MODE>(r, r, scale, mask);
                    MicroAPI::Div<float, &DIV_MODE>(r, r, keyScale, mask);
                    MicroAPI::Cast<bfloat16_t, float, FROM_FLOAT>(rope16, r, mask);
                } else {
                    const uint16_t ropeRow = sharedRope ? (startHead + row) / heads : row;
                    MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawRope, keyRope + ropeRow * 64);
                    rope16 = rawRope;
                }
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(rope + row * 64, rope16, mask);
            }
        }
    }
}

template <int Kind>
__aicore__ inline void Reconstruct(__local_mem__ fp8_e4m3fn_t *input, __local_mem__ bfloat16_t *rope,
    __local_mem__ float *queryScale, __local_mem__ float *scales, __local_mem__ bfloat16_t *output,
    uint16_t startHead, uint16_t rows, uint16_t heads)
{
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<fp8_e4m3fn_t> code;
        MicroAPI::RegTensor<bfloat16_t> result, rawRope;
        MicroAPI::RegTensor<float> x, scale, keyScale;
        auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        for (uint16_t row = 0; row < rows; ++row) {
            const uint16_t head = (startHead + row) % heads;
            if constexpr (Kind == 0) {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale, queryScale + row);
            } else {
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(scale,
                    scales + head + (Kind == 2 ? 256 : 0));
            }
            constexpr uint32_t width = Kind == 2 ? 128 : 192;
            for (uint16_t col = 0; col < static_cast<uint16_t>(128); col += 64) {
                MicroAPI::DataCopy<fp8_e4m3fn_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(code, input + row * 128 + col);
                MicroAPI::Cast<float, fp8_e4m3fn_t, TO_FLOAT>(x, code, mask);
                MicroAPI::Mul(x, x, scale, mask);
                MicroAPI::Cast<bfloat16_t, float, FROM_FLOAT>(result, x, mask);
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(output + row * width + col, result, mask);
            }
            if constexpr (Kind != 2) {
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawRope, rope + row * 64);
                if constexpr (Kind == 0) {
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(keyScale, scales + head);
                    MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(x, rawRope, mask);
                    MicroAPI::Mul(x, x, scale, mask);
                    MicroAPI::Mul(x, x, keyScale, mask);
                    MicroAPI::Cast<bfloat16_t, float, FROM_FLOAT>(result, x, mask);
                } else {
                    result = rawRope;
                }
                MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(output + row * width + 128, result, mask);
            }
        }
    }
}

template <bool Fake> class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR rope, GM_ADDR partial,
        GM_ADDR queryOut, GM_ADDR keyOut, GM_ADDR valueOut, GM_ADDR queryRopeOut, GM_ADDR keyRopeOut,
        GM_ADDR queryScale, GM_ADDR keyScale, GM_ADDR valueScale, const optiling::FlashAttnC8PrepareTilingData &tiling)
    {
        data = tiling;
        inputQ.SetGlobalBuffer((__gm__ bfloat16_t *)query);
        inputK.SetGlobalBuffer((__gm__ bfloat16_t *)key);
        inputV.SetGlobalBuffer((__gm__ bfloat16_t *)value);
        inputRope.SetGlobalBuffer((__gm__ bfloat16_t *)rope);
        inputPartial.SetGlobalBuffer((__gm__ float *)partial);
        outputQ.SetGlobalBuffer((__gm__ uint8_t *)queryOut);
        outputK.SetGlobalBuffer((__gm__ uint8_t *)keyOut);
        outputV.SetGlobalBuffer((__gm__ uint8_t *)valueOut);
        outputQR.SetGlobalBuffer((__gm__ bfloat16_t *)queryRopeOut);
        outputKR.SetGlobalBuffer((__gm__ bfloat16_t *)keyRopeOut);
        outputSQ.SetGlobalBuffer((__gm__ float *)queryScale);
        outputSK.SetGlobalBuffer((__gm__ float *)keyScale);
        outputSV.SetGlobalBuffer((__gm__ float *)valueScale);
        pipe.InitBuffer(inputBuffer, FlashAttnC8Config::PREPARE_ROWS * 192 * 2);
        pipe.InitBuffer(codeBuffer, FlashAttnC8Config::PREPARE_ROWS * 128);
        pipe.InitBuffer(ropeBuffer, FlashAttnC8Config::PREPARE_ROWS * 64 * 2);
        pipe.InitBuffer(keyRopeBuffer, FlashAttnC8Config::PREPARE_ROWS * 64 * 2);
        pipe.InitBuffer(queryScaleBuffer, FlashAttnC8Config::PREPARE_ROWS * 4);
        pipe.InitBuffer(scaleBuffer, 512 * 4);
        pipe.InitBuffer(partialBuffer, 256 * 16 * 4);
        if constexpr (Fake) pipe.InitBuffer(reconstructedBuffer, FlashAttnC8Config::PREPARE_ROWS * 192 * 2);
    }
    __aicore__ inline void LoadScales()
    {
        auto scales = scaleBuffer.Get<float>();
        auto partial = partialBuffer.Get<float>();
        for (int64_t h = 0; h < data.heads; ++h) {
            scales.SetValue(h, 0.0f);
            scales.SetValue(256 + h, 0.0f);
        }
        for (int64_t part = 0; part <
            (data.keyTokens + FlashAttnC8Config::STATS_TOKENS - 1) / FlashAttnC8Config::STATS_TOKENS; ++part) {
            Sync<HardEvent::S_MTE2>();
            DataCopy(partial, inputPartial[part * data.heads * 16], data.heads * 16);
            Sync<HardEvent::MTE2_S>();
            for (int64_t h = 0; h < data.heads; ++h) {
                const float oldK = scales.GetValue(h), nextK = partial.GetValue(h * 16);
                const float oldV = scales.GetValue(256 + h), nextV = partial.GetValue(h * 16 + 8);
                scales.SetValue(h, oldK > nextK ? oldK : nextK);
                scales.SetValue(256 + h, oldV > nextV ? oldV : nextV);
            }
        }
        for (int64_t h = 0; h < data.heads; ++h) {
            float keyScale = scales.GetValue(h) * (1.0f / 448.0f);
            float valueScale = scales.GetValue(256 + h) * (1.0f / 448.0f);
            scales.SetValue(h, keyScale == 0.0f ? 1.0f : keyScale);
            scales.SetValue(256 + h, valueScale == 0.0f ? 1.0f : valueScale);
        }
        Sync<HardEvent::S_V>();
        if (GetBlockIdx() == 0) {
            Sync<HardEvent::S_MTE3>();
            DataCopyPad(outputSK, scales, DataCopyExtParams{1, static_cast<uint32_t>(data.heads * 4), 0, 0, 0});
            DataCopyPad(outputSV, scales[256],
                DataCopyExtParams{1, static_cast<uint32_t>(data.heads * 4), 0, 0, 0});
        }
    }
    template <int Kind> __aicore__ inline void ProcessRows(GlobalTensor<bfloat16_t> &source,
        GlobalTensor<uint8_t> &destination, int64_t tokens, int64_t tokenStride, int64_t headStride)
    {
        auto input = inputBuffer.Get<bfloat16_t>();
        auto codes = codeBuffer.Get<fp8_e4m3fn_t>();
        auto rope = ropeBuffer.Get<bfloat16_t>();
        auto keyRope = keyRopeBuffer.Get<bfloat16_t>();
        auto sq = queryScaleBuffer.Get<float>();
        auto scales = scaleBuffer.Get<float>();
        constexpr int64_t batchRows = FlashAttnC8Config::PREPARE_ROWS;
        for (int64_t start = GetBlockIdx() * batchRows; start < tokens * data.heads; start += GetBlockNum() * batchRows) {
            const int64_t remaining = tokens * data.heads - start;
            const uint32_t rows = static_cast<uint32_t>(remaining < batchRows ? remaining : batchRows);
            Sync<HardEvent::V_MTE2>();
            Sync<HardEvent::MTE3_V>();
            constexpr uint32_t width = Kind == 0 ? 192 : 128;
            if (tokenStride == data.heads * headStride) {
                const bool dense = headStride == width;
                DataCopyPad(input, source[(start / data.heads) * tokenStride + (start % data.heads) * headStride],
                    DataCopyExtParams{static_cast<uint16_t>(dense ? 1 : rows), (dense ? rows : 1) * width * 2,
                        static_cast<uint32_t>((headStride - width) * 2), 0, 0},
                    DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            } else {
                for (uint32_t row = 0; row < rows; ++row) {
                    const int64_t index = start + row;
                    DataCopyPad(input[row * width], source[(index / data.heads) * tokenStride + (index % data.heads) * headStride],
                        DataCopyExtParams{1, width * 2, 0, 0, 0}, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                }
            }
            if constexpr (Kind == 1) {
                const int64_t firstToken = start / data.heads, firstHead = start % data.heads;
                if (data.ropeHeadStride == 0) {
                    // Shared RoPE is read once per token and broadcast in VF,
                    // instead of issuing one DMA for every expanded head.
                    const uint16_t ropeRows = static_cast<uint16_t>((firstHead + rows + data.heads - 1) / data.heads);
                    const bool dense = data.ropeTokenStride == 64;
                    DataCopyPad(keyRope, inputRope[firstToken * data.ropeTokenStride],
                        DataCopyExtParams{static_cast<uint16_t>(dense ? 1 : ropeRows),
                            static_cast<uint32_t>(dense ? ropeRows * 128 : 128),
                            static_cast<uint32_t>((data.ropeTokenStride - 64) * 2), 0, 0},
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                } else if (data.ropeTokenStride == data.heads * data.ropeHeadStride) {
                    const bool dense = data.ropeHeadStride == 64;
                    DataCopyPad(keyRope, inputRope[firstToken * data.ropeTokenStride + firstHead * data.ropeHeadStride],
                        DataCopyExtParams{static_cast<uint16_t>(dense ? 1 : rows), (dense ? rows : 1) * 128,
                            static_cast<uint32_t>((data.ropeHeadStride - 64) * 2), 0, 0},
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                } else {
                    for (uint32_t row = 0; row < rows; ++row) {
                        const int64_t index = start + row;
                        DataCopyPad(keyRope[row * 64],
                            inputRope[(index / data.heads) * data.ropeTokenStride + (index % data.heads) * data.ropeHeadStride],
                            DataCopyExtParams{1, 128, 0, 0, 0}, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
                    }
                }
            }
            Sync<HardEvent::MTE2_V>();
            Quantize<Kind>((__local_mem__ bfloat16_t *)input.GetPhyAddr(),
                (__local_mem__ fp8_e4m3fn_t *)codes.GetPhyAddr(), (__local_mem__ bfloat16_t *)rope.GetPhyAddr(),
                (__local_mem__ float *)sq.GetPhyAddr(), (__local_mem__ float *)scales.GetPhyAddr(),
                (__local_mem__ bfloat16_t *)keyRope.GetPhyAddr(), static_cast<uint16_t>(start % data.heads),
                static_cast<uint16_t>(rows), static_cast<uint16_t>(data.heads), data.ropeHeadStride == 0);
            if constexpr (Fake) {
                PipeBarrier<PIPE_V>();
                auto reconstructed = reconstructedBuffer.Get<bfloat16_t>();
                Reconstruct<Kind>((__local_mem__ fp8_e4m3fn_t *)codes.GetPhyAddr(),
                    (__local_mem__ bfloat16_t *)rope.GetPhyAddr(), (__local_mem__ float *)sq.GetPhyAddr(),
                    (__local_mem__ float *)scales.GetPhyAddr(), (__local_mem__ bfloat16_t *)reconstructed.GetPhyAddr(),
                    static_cast<uint16_t>(start % data.heads), static_cast<uint16_t>(rows), static_cast<uint16_t>(data.heads));
                Sync<HardEvent::V_MTE3>();
                constexpr uint32_t outWidth = Kind == 2 ? 128 : 192;
                DataCopy(destination[start * outWidth * 2], reconstructed.template ReinterpretCast<uint8_t>(), rows * outWidth * 2);
            } else {
                Sync<HardEvent::V_MTE3>();
                DataCopy(destination[start * 128], codes.template ReinterpretCast<uint8_t>(), rows * 128);
            }
            if constexpr (Kind == 0) {
                DataCopy(outputQR[start * 64], rope, rows * 64);
                DataCopyPad(outputSQ[start], sq, DataCopyExtParams{1, rows * 4, 0, 0, 0});
            } else if constexpr (Kind == 1) {
                DataCopy(outputKR[start * 64], rope, rows * 64);
            }
        }
    }
    __aicore__ inline void Process()
    {
        LoadScales();
        ProcessRows<0>(inputQ, outputQ, data.queryTokens, data.queryTokenStride, data.queryHeadStride);
        ProcessRows<1>(inputK, outputK, data.keyTokens, data.keyTokenStride, data.keyHeadStride);
        ProcessRows<2>(inputV, outputV, data.keyTokens, data.valueTokenStride, data.valueHeadStride);
    }
private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> inputBuffer, codeBuffer, ropeBuffer, keyRopeBuffer, queryScaleBuffer;
    TBuf<TPosition::VECCALC> scaleBuffer, partialBuffer, reconstructedBuffer;
    GlobalTensor<bfloat16_t> inputQ, inputK, inputV, inputRope, outputQR, outputKR;
    GlobalTensor<uint8_t> outputQ, outputK, outputV;
    GlobalTensor<float> inputPartial, outputSQ, outputSK, outputSV;
    optiling::FlashAttnC8PrepareTilingData data;
};
}
extern "C" __global__ __aicore__ void flash_attn_c8_prepare(GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR keyRope, GM_ADDR partial, GM_ADDR queryOut, GM_ADDR keyOut, GM_ADDR valueOut,
    GM_ADDR queryRopeOut, GM_ADDR keyRopeOut, GM_ADDR queryScale, GM_ADDR keyScale, GM_ADDR valueScale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttnC8PrepareTilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        FlashAttnC8Prepare::Kernel<false> kernel;
        kernel.Init(query, key, value, keyRope, partial, queryOut, keyOut, valueOut, queryRopeOut,
            keyRopeOut, queryScale, keyScale, valueScale, data);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        FlashAttnC8Prepare::Kernel<true> kernel;
        kernel.Init(query, key, value, keyRope, partial, queryOut, keyOut, valueOut, queryRopeOut,
            keyRopeOut, queryScale, keyScale, valueScale, data);
        kernel.Process();
    }
}
