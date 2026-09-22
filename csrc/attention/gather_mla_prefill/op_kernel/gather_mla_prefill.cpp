// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t TILE_ROWS = 16;
constexpr uint32_t LATENT_DIM = 512;
constexpr uint32_t ROPE_DIM = 64;
constexpr MicroAPI::CastTrait UPCAST = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr MicroAPI::CastTrait DOWNCAST = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// FP8 dequantization stays in vector registers: no full FP32 UB intermediate.
__simd_vf__ void Dequantize(__ubuf__ bfloat16_t *dst, __ubuf__ fp8_e4m3fn_t *src,
                          float scale, uint16_t repeats)
{
    MicroAPI::RegTensor<fp8_e4m3fn_t> packed;
    MicroAPI::RegTensor<float> value;
    MicroAPI::RegTensor<bfloat16_t> output;
    auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < repeats; ++i) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            reinterpret_cast<MicroAPI::RegTensor<int8_t> &>(packed),
            reinterpret_cast<__ubuf__ int8_t *>(src) + i * 64);
        MicroAPI::Cast<float, fp8_e4m3fn_t, UPCAST>(value, packed, mask);
        MicroAPI::Muls(value, value, scale, mask);
        MicroAPI::Cast<bfloat16_t, float, DOWNCAST>(output, value, mask);
        MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * 64, output, mask);
    }
}

class GatherMlaPrefillKernel {
public:
    __aicore__ inline void Init(GM_ADDR latentCache, GM_ADDR ropeCache, GM_ADDR blockTable,
        GM_ADDR cumulativeLengths, GM_ADDR lengths, GM_ADDR starts, GM_ADDR scale,
        GM_ADDR latent, GM_ADDR rope, const GatherMlaPrefillTilingData &tiling, TPipe *pipe)
    {
        tiling_ = tiling;
        latentCache_.SetGlobalBuffer(reinterpret_cast<__gm__ fp8_e4m3fn_t *>(latentCache));
        ropeCache_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(ropeCache));
        blockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        cumulativeLengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cumulativeLengths));
        lengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        starts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(starts));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
        latent_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(latent));
        rope_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(rope));
        pipe->InitBuffer(metadataBuffer_, 5 * 32);
        pipe->InitBuffer(latentInputBuffer_, TILE_ROWS * LATENT_DIM);
        pipe->InitBuffer(latentOutputBuffer_, TILE_ROWS * LATENT_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(ropeBuffer_, TILE_ROWS * ROPE_DIM * sizeof(bfloat16_t));
        const DataCopyExtParams scalarCopy{1, sizeof(float), 0, 0, 0};
        DataCopyPad(metadataBuffer_.Get<float>()[24], scale_, scalarCopy, DataCopyPadExtParams<float>{});
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        scaleValue_ = metadataBuffer_.Get<float>().GetValue(24);
    }

    __aicore__ inline void Process()
    {
        const int64_t taskCount = tiling_.requests * tiling_.tilesPerRequest;
        int64_t activeRequest = -1;
        int64_t outputStart = 0;
        int64_t length = 0;
        int64_t sequenceStart = 0;
        for (int64_t task = GetBlockIdx(); task < taskCount; task += tiling_.usedCoreNum) {
            const int64_t request = task / tiling_.tilesPerRequest;
            const int64_t localStart = (task % tiling_.tilesPerRequest) * TILE_ROWS;
            auto metadata = metadataBuffer_.Get<int32_t>();
            const DataCopyExtParams scalarCopy{1, sizeof(int32_t), 0, 0, 0};
            const DataCopyPadExtParams<int32_t> noPadding{};
            if (request != activeRequest) {
                DataCopyPad(metadata, cumulativeLengths_[request], scalarCopy, noPadding);
                DataCopyPad(metadata[8], lengths_[request], scalarCopy, noPadding);
                DataCopyPad(metadata[16], starts_[request], scalarCopy, noPadding);
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                outputStart = metadata.GetValue(0);
                length = metadata.GetValue(8);
                sequenceStart = metadata.GetValue(16);
                activeRequest = request;
            }
            if (localStart >= length) continue;
            int64_t rows = length - localStart;
            if (rows > TILE_ROWS) rows = TILE_ROWS;
            if (sequenceStart < 0 || outputStart < 0 || outputStart + localStart + rows > tiling_.numTokens) continue;
            int64_t copied = 0;
            // A tile may straddle a page after a non-aligned history start.
            while (copied < rows) {
                const int64_t logical = sequenceStart + localStart + copied;
                const int64_t logicalPage = logical / 128;
                if (logicalPage >= tiling_.tableColumns) break;
                DataCopyPad(metadata[32], blockTable_[request * tiling_.tableColumns + logicalPage],
                    scalarCopy, noPadding);
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                const int64_t page = metadata.GetValue(32);
                if (page < 0 || page >= tiling_.pages) break;
                const int64_t pageRow = logical % 128;
                int64_t segment = rows - copied;
                if (segment > 128 - pageRow) segment = 128 - pageRow;
                CopyAndDequantize(page, pageRow, outputStart + localStart + copied,
                    static_cast<uint32_t>(segment));
                copied += segment;
            }
        }
    }

private:
    __aicore__ inline void CopyAndDequantize(int64_t page, int64_t pageRow,
                                            int64_t outputRow, uint32_t rows)
    {
        auto input = latentInputBuffer_.Get<fp8_e4m3fn_t>();
        auto output = latentOutputBuffer_.Get<bfloat16_t>();
        auto rope = ropeBuffer_.Get<bfloat16_t>();
        const DataCopyExtParams latentCopy{static_cast<uint16_t>(rows), LATENT_DIM,
            static_cast<uint32_t>(tiling_.latentRowStride - LATENT_DIM), 0, 0};
        const DataCopyExtParams ropeCopy{static_cast<uint16_t>(rows), ROPE_DIM * sizeof(bfloat16_t),
            static_cast<uint32_t>((tiling_.ropeRowStride - ROPE_DIM) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPad(input, latentCache_[page * tiling_.latentPageStride + pageRow * tiling_.latentRowStride],
            latentCopy, DataCopyPadExtParams<fp8_e4m3fn_t>{});
        DataCopyPad(rope, ropeCache_[page * tiling_.ropePageStride + pageRow * tiling_.ropeRowStride],
            ropeCopy, DataCopyPadExtParams<bfloat16_t>{});
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Dequantize(reinterpret_cast<__ubuf__ bfloat16_t *>(output.GetPhyAddr()),
            reinterpret_cast<__ubuf__ fp8_e4m3fn_t *>(input.GetPhyAddr()),
            scaleValue_, static_cast<uint16_t>(rows * LATENT_DIM / 64));
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
        // Every output row is 32-byte aligned, including page-boundary tails.
        DataCopy(latent_[outputRow * LATENT_DIM], output, rows * LATENT_DIM);
        DataCopy(rope_[outputRow * ROPE_DIM], rope, rows * ROPE_DIM);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    GatherMlaPrefillTilingData tiling_;
    GlobalTensor<fp8_e4m3fn_t> latentCache_;
    GlobalTensor<bfloat16_t> ropeCache_, latent_, rope_;
    GlobalTensor<int32_t> blockTable_, cumulativeLengths_, lengths_, starts_;
    GlobalTensor<float> scale_;
    TBuf<TPosition::VECCALC> metadataBuffer_, latentInputBuffer_, latentOutputBuffer_, ropeBuffer_;
    float scaleValue_ = 1.0f;
};
} // namespace

extern "C" __global__ __aicore__ void gather_mla_prefill(GM_ADDR latentCache, GM_ADDR ropeCache,
    GM_ADDR blockTable, GM_ADDR cumulativeLengths, GM_ADDR lengths, GM_ADDR starts, GM_ADDR scale,
    GM_ADDR latent, GM_ADDR rope, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GatherMlaPrefillKernel kernel;
    kernel.Init(latentCache, ropeCache, blockTable, cumulativeLengths, lengths, starts, scale,
        latent, rope, tilingData, &pipe);
    kernel.Process();
}
