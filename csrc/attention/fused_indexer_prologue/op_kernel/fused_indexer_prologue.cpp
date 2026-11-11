/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include "kernel_operator.h"

#if __has_include("../../mla_preprocess/op_kernel/mla_preprocess_mix_bf16_nq.hpp")
#include "../../mla_preprocess/op_kernel/mla_preprocess_mix_bf16_nq.hpp"
#else
#include "../../../mla_preprocess/op_kernel/mla_preprocess_mix_bf16_nq.hpp"
#endif
#include "fused_indexer_prologue_tiling_data.h"

namespace FusedIndexerPrologue {
using namespace AscendC;
using MlaDataFormat = ::DataFormat;

constexpr uint32_t HEAD_DIM = 128;
constexpr float RMS_AVG_FACTOR = 0.0078125F;  // 1 / HEAD_DIM
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t ROPE_START = HEAD_DIM - ROPE_DIM;
constexpr float QUANT_MAX = 127.0F;
constexpr uint32_t FP32_VECTOR_LEN = 64;
constexpr uint64_t INTERLEAVED_NEGATE_MASK = 0x5555555555555555ULL;

// Keep the fixed-shape vector helpers local. The standalone RMSNorm and RoPE
// kernel headers expose global symbols that conflict with MLA's kernel helpers.
template <typename T>
__aicore__ inline void CopyGmToLocal(
    const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint32_t len)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = len * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(dst, src, copyParams, {});
}

template <typename T>
__aicore__ inline void CopyLocalToGm(
    const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint32_t len)
{
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = len * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(dst, src, copyParams);
}

__aicore__ inline float ReduceSum128(const LocalTensor<float> &tensor)
{
    Add(tensor, tensor, tensor[FP32_VECTOR_LEN], FP32_VECTOR_LEN);
    PipeBarrier<PIPE_V>();
    AscendCUtils::SetMask<float>(FP32_VECTOR_LEN);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceSum<float, false>(tensor, tensor, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceSum<float, false>(tensor, tensor, MASK_PLACEHOLDER, 1, 1, 1, 8);
#endif
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event);
    WaitFlag<HardEvent::V_S>(event);
    return tensor.GetValue(0);
}

__aicore__ inline void ReduceMax128(const LocalTensor<float> &tensor)
{
    AscendCUtils::SetMask<float>(FP32_VECTOR_LEN);
    Max(tensor, tensor[FP32_VECTOR_LEN], tensor, MASK_PLACEHOLDER, 1, {1, 1, 1, 0, 8, 0});
    PipeBarrier<PIPE_V>();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        WholeReduceMax<float, false>(tensor, tensor, MASK_PLACEHOLDER, 1, 0, 1, 0);
    }
#else
    WholeReduceMax<float, false>(tensor, tensor, MASK_PLACEHOLDER, 1, 8, 1, 8);
#endif
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void QuantizeToInt8(
    LocalTensor<int8_t> &dst, LocalTensor<float> &src)
{
    Cast(src.ReinterpretCast<int32_t>(), src, RoundMode::CAST_RINT, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    SetDeqScale(static_cast<half>(1.0F));
    PipeBarrier<PIPE_V>();
    Cast(src.ReinterpretCast<half>(), src.ReinterpretCast<int32_t>(),
         RoundMode::CAST_NONE, HEAD_DIM);
    PipeBarrier<PIPE_V>();
    Cast(dst, src.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, HEAD_DIM);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void InterleavedInversion64(LocalTensor<float> &tensor)
{
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<float, AscendC::MaskMode::NORMAL>(
        0, INTERLEAVED_NEGATE_MASK);
    AscendC::Muls<float, false>(tensor, tensor, -1.0F, MASK_PLACEHOLDER, 1,
                                {1, 1, 8, 8});
    AscendC::ResetMask();
}

class VectorPostProcess {
public:
    __aicore__ explicit VectorPostProcess(TPipe *pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(
        GM_ADDR projection, GM_ADDR normWeight, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR slotMapping, GM_ADDR kCache, GM_ADDR scaleCache,
        const FusedIndexerPrologueTilingData &tiling)
    {
        projectionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(projection));
        normWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(normWeight));
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(cos));
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(sin));
        slotMappingGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(slotMapping));
        kCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(kCache));
        scaleCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scaleCache));

        numTokens_ = tiling.numTokens;
        vectorCoreNum_ = tiling.vectorCoreNum;
        cacheBlocks_ = tiling.cacheBlocks;
        cacheBlockSize_ = tiling.cacheBlockSize;
        epsilon_ = tiling.epsilon;
        kCacheStride0_ = tiling.kCacheStride0;
        kCacheStride1_ = tiling.kCacheStride1;
        scaleCacheStride0_ = tiling.scaleCacheStride0;
        scaleCacheStride1_ = tiling.scaleCacheStride1;

        pipe_->InitBuffer(inputBuf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe_->InitBuffer(weightBuf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe_->InitBuffer(roundBuf_, HEAD_DIM * sizeof(bfloat16_t));
        pipe_->InitBuffer(xFp32Buf_, HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(weightFp32Buf_, HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(tmpFp32Buf_, HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(swappedFp32Buf_, HEAD_DIM * sizeof(float));
        pipe_->InitBuffer(cosBuf_, ROPE_DIM * sizeof(float));
        pipe_->InitBuffer(sinBuf_, ROPE_DIM * sizeof(float));
        pipe_->InitBuffer(gatherOffsetBuf_, HEAD_DIM * sizeof(uint32_t));
        pipe_->InitBuffer(quantBuf_, HEAD_DIM * sizeof(int8_t));
        pipe_->InitBuffer(slotBuf_, 32);
        pipe_->InitBuffer(scaleBuf_, 32);
    }

    __aicore__ inline void Process()
    {
#ifdef __DAV_C220_VEC__
        const uint32_t vectorBlockIdx =
            (GetBlockIdx() / 2U) * 2U + static_cast<uint32_t>(GetSubBlockidx());
        if (vectorBlockIdx >= vectorCoreNum_) {
            return;
        }

        LocalTensor<int32_t> gatherOffsetsInt = gatherOffsetBuf_.Get<int32_t>();
        for (uint32_t i = 0; i < ROPE_DIM; ++i) {
            gatherOffsetsInt.SetValue(i, static_cast<int32_t>((i ^ 1U) * sizeof(float)));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        LocalTensor<uint32_t> gatherOffsets =
            gatherOffsetsInt.ReinterpretCast<uint32_t>();

        LocalTensor<bfloat16_t> normWeight = weightBuf_.Get<bfloat16_t>();
        CopyGmToLocal(normWeight, normWeightGm_, HEAD_DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t token = vectorBlockIdx; token < numTokens_; token += vectorCoreNum_) {
            ProcessToken(token, normWeight, gatherOffsets);
        }
#endif
    }

private:
    __aicore__ inline void ProcessToken(
        uint32_t token, LocalTensor<bfloat16_t> &normWeight,
        LocalTensor<uint32_t> &gatherOffsets)
    {
        LocalTensor<int32_t> slot = slotBuf_.Get<int32_t>();
        CopyGmToLocal(slot, slotMappingGm_[static_cast<uint64_t>(token) * 2U], 2U);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        const int32_t blockId = slot.GetValue(0);
        const int32_t blockOffset = slot.GetValue(1);
        if (blockId < 0 || blockOffset < 0 ||
            static_cast<uint32_t>(blockId) >= cacheBlocks_ ||
            static_cast<uint32_t>(blockOffset) >= cacheBlockSize_) {
            return;
        }

        LocalTensor<bfloat16_t> input = inputBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> rounded = roundBuf_.Get<bfloat16_t>();
        LocalTensor<float> x = xFp32Buf_.Get<float>();
        LocalTensor<float> weight = weightFp32Buf_.Get<float>();
        LocalTensor<float> tmp = tmpFp32Buf_.Get<float>();
        LocalTensor<float> swapped = swappedFp32Buf_.Get<float>();

        CopyGmToLocal(input, projectionGm_[static_cast<uint64_t>(token) * HEAD_DIM], HEAD_DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(x, input, RoundMode::CAST_NONE, HEAD_DIM);
        Cast(weight, normWeight, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        Mul(tmp, x, x, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        const float squareSum = ReduceSum128(tmp);
        const float reciprocalRms = 1.0F / sqrt(squareSum * RMS_AVG_FACTOR + epsilon_);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(x, x, reciprocalRms, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Mul(x, x, weight, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        // Preserve the BF16 boundary of npu_rms_norm before RoPE.
        // AscendC CAST_RINT performs round-to-nearest-even for FP32 -> BF16.
        Cast(rounded, x, RoundMode::CAST_RINT, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Cast(x, rounded, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> cos = cosBuf_.Get<float>();
        LocalTensor<float> sin = sinBuf_.Get<float>();
        CopyGmToLocal(cos, cosGm_[static_cast<uint64_t>(token) * ROPE_DIM], ROPE_DIM);
        CopyGmToLocal(sin, sinGm_[static_cast<uint64_t>(token) * ROPE_DIM], ROPE_DIM);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls(swapped, x[ROPE_START], 1.0F, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Gather(swapped, swapped, gatherOffsets, 0U, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        InterleavedInversion64(swapped);
        PipeBarrier<PIPE_V>();
        Mul(tmp, x[ROPE_START], cos, ROPE_DIM);
        Mul(swapped, swapped, sin, ROPE_DIM);
        PipeBarrier<PIPE_V>();
        Add(x[ROPE_START], tmp, swapped, ROPE_DIM);
        PipeBarrier<PIPE_V>();

        // Match the BF16 in-place RoPE result consumed by dynamic quant.
        // AscendC CAST_RINT performs round-to-nearest-even for FP32 -> BF16.
        Cast(rounded, x, RoundMode::CAST_RINT, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        Cast(x, rounded, RoundMode::CAST_NONE, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        Abs(tmp, x, HEAD_DIM);
        PipeBarrier<PIPE_V>();
        ReduceMax128(tmp);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const float absMax = tmp.GetValue(0);
        const float scale = absMax / QUANT_MAX;
        const float inverseScale = absMax > 0.0F ? QUANT_MAX / absMax : 0.0F;
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(x, x, inverseScale, HEAD_DIM);
        PipeBarrier<PIPE_V>();

        LocalTensor<int8_t> quantized = quantBuf_.Get<int8_t>();
        QuantizeToInt8(quantized, x);

        const uint64_t keyOffset = static_cast<uint64_t>(blockId) * kCacheStride0_ +
                                   static_cast<uint64_t>(blockOffset) * kCacheStride1_;
        const uint64_t scaleOffset = static_cast<uint64_t>(blockId) * scaleCacheStride0_ +
                                     static_cast<uint64_t>(blockOffset) * scaleCacheStride1_;
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyLocalToGm(kCacheGm_[keyOffset], quantized, HEAD_DIM);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);

        LocalTensor<half> scaleLocal = scaleBuf_.Get<half>();
        scaleLocal.SetValue(0, static_cast<half>(scale));
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        CopyLocalToGm(scaleCacheGm_[scaleOffset], scaleLocal, 1U);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    TPipe *pipe_ = nullptr;
    TBuf<TPosition::VECCALC> inputBuf_;
    TBuf<TPosition::VECCALC> weightBuf_;
    TBuf<TPosition::VECCALC> roundBuf_;
    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> weightFp32Buf_;
    TBuf<TPosition::VECCALC> tmpFp32Buf_;
    TBuf<TPosition::VECCALC> swappedFp32Buf_;
    TBuf<TPosition::VECCALC> cosBuf_;
    TBuf<TPosition::VECCALC> sinBuf_;
    TBuf<TPosition::VECCALC> gatherOffsetBuf_;
    TBuf<TPosition::VECCALC> quantBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    TBuf<TPosition::VECCALC> scaleBuf_;

    GlobalTensor<bfloat16_t> projectionGm_;
    GlobalTensor<bfloat16_t> normWeightGm_;
    GlobalTensor<float> cosGm_;
    GlobalTensor<float> sinGm_;
    GlobalTensor<int32_t> slotMappingGm_;
    GlobalTensor<int8_t> kCacheGm_;
    GlobalTensor<half> scaleCacheGm_;

    uint32_t numTokens_{0};
    uint32_t vectorCoreNum_{0};
    uint32_t cacheBlocks_{0};
    uint32_t cacheBlockSize_{0};
    float epsilon_{1e-6F};
    uint64_t kCacheStride0_{0};
    uint64_t kCacheStride1_{0};
    uint64_t scaleCacheStride0_{0};
    uint64_t scaleCacheStride1_{0};
};
}  // namespace FusedIndexerPrologue

extern "C" __global__ __aicore__ void fused_indexer_prologue(
    GM_ADDR latent, GM_ADDR wkWeight, GM_ADDR normWeight, GM_ADDR cos,
    GM_ADDR sin, GM_ADDR slotMapping, GM_ADDR kCache, GM_ADDR scaleCache,
    GM_ADDR kCacheOut, GM_ADDR scaleCacheOut, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)kCache;
    (void)scaleCache;
    REGISTER_TILING_DEFAULT(FusedIndexerPrologueTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    PRELOAD(2);
#endif
    SetAtomicnone();
    SetMasknorm();
#ifdef __DAV_C220_CUBE__
    SetPadding<uint64_t>(static_cast<uint64_t>(0));
    SetNdpara(1, 0, 0);
#endif
    GET_TILING_DATA_WITH_STRUCT(
        FusedIndexerPrologueTilingData, tilingData, tiling);

    GM_ADDR projection = AscendC::GetUserWorkspace(workspace);
#ifdef __DAV_C220_CUBE__
    if ASCEND_IS_AIC {
        MLAPO_BF16_NQ::PpMatmulEinSum<
            __bf16, __bf16, FusedIndexerPrologue::MlaDataFormat::ND,
            true, 0, 0, 0, false>
            matmul;
        matmul.Init(latent, wkWeight, projection, tilingData.matmul);
        matmul.Process();
        FftsCrossCoreSync<PIPE_FIX, 0>(MM1);
        AscendC::WaitEvent(MM1);
        FftsCrossCoreSync<PIPE_FIX, 0x2>(QUANT1);
    }
#endif
#ifdef __DAV_C220_VEC__
    if ASCEND_IS_AIV {
        AscendC::WaitEvent(QUANT1);
        AscendC::TPipe pipe;
        FusedIndexerPrologue::VectorPostProcess post(&pipe);
        post.Init(
            projection, normWeight, cos, sin, slotMapping, kCacheOut,
            scaleCacheOut, tilingData);
        post.Process();
    }
#endif
}
