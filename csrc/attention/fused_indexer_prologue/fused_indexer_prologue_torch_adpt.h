/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FUSED_INDEXER_PROLOGUE_TORCH_ADPT_H_
#define FUSED_INDEXER_PROLOGUE_TORCH_ADPT_H_

namespace vllm_ascend {
inline void fused_indexer_prologue_npu(
    const at::Tensor &latent, const at::Tensor &wkWeight,
    const at::Tensor &normWeight, const at::Tensor &cos,
    const at::Tensor &sin, const at::Tensor &slotMapping,
    at::Tensor &kCache, at::Tensor &scaleCache, double epsilon)
{
    constexpr int64_t HEAD_DIM = 128;
    constexpr int64_t ROPE_DIM = 64;
    constexpr int64_t INPUT_DIM = 512;
    TORCH_CHECK(
        latent.device().is_privateuseone(),
        "DeepSeek-V4.1 indexer prologue tensors must be on NPU.");
    TORCH_CHECK(
        latent.dim() == 2 && wkWeight.dim() == 2 && normWeight.dim() == 1,
        "latent/wk_weight/norm_weight must have ranks 2/2/1.");
    TORCH_CHECK(
        latent.size(1) == INPUT_DIM && wkWeight.size(0) == HEAD_DIM &&
            wkWeight.size(1) == INPUT_DIM &&
            normWeight.size(0) == HEAD_DIM,
        "DeepSeek-V4.1 indexer prologue requires a [T,512] x [128,512] projection.");
    TORCH_CHECK(
        latent.scalar_type() == at::kBFloat16 &&
            wkWeight.scalar_type() == at::kBFloat16 &&
            normWeight.scalar_type() == at::kBFloat16,
        "latent, wk_weight and norm_weight must be BF16.");
    TORCH_CHECK(
        cos.dim() > 0 && sin.dim() > 0 && cos.scalar_type() == at::kFloat &&
            sin.scalar_type() == at::kFloat &&
            cos.size(-1) == ROPE_DIM && sin.size(-1) == ROPE_DIM &&
            cos.numel() == latent.size(0) * ROPE_DIM &&
            sin.numel() == latent.size(0) * ROPE_DIM,
        "cos and sin must be FP32 [T, ..., 64] tensors with singleton middle dimensions.");
    TORCH_CHECK(
        slotMapping.dim() == 2 && slotMapping.size(0) == latent.size(0) &&
            slotMapping.size(1) == 2,
        "slot_mapping must provide [T, 2] block coordinates.");
    TORCH_CHECK(
        kCache.dim() == 4 && kCache.size(2) == 1 &&
            kCache.size(3) == HEAD_DIM && kCache.scalar_type() == at::kChar,
        "k_cache must be INT8 [blocks, block_size, 1, 128].");
    TORCH_CHECK(
        scaleCache.dim() == 4 && scaleCache.size(2) == 1 &&
            scaleCache.size(3) == 1 && scaleCache.scalar_type() == at::kHalf,
        "scale_cache must be FP16 [blocks, block_size, 1, 1].");
    TORCH_CHECK(
        kCache.size(0) == scaleCache.size(0) &&
            kCache.size(1) == scaleCache.size(1),
        "k_cache and scale_cache must have matching block geometry.");
    TORCH_CHECK(
        kCache.stride(3) == 1 && kCache.stride(1) >= HEAD_DIM &&
            scaleCache.stride(3) == 1 && scaleCache.stride(1) >= 1,
        "cache rows must be dense while block/page strides may contain gaps.");
    TORCH_CHECK(epsilon > 0.0, "epsilon must be positive.");

    const auto device = latent.device();
    const at::Tensor tensors[] = {
        wkWeight, normWeight, cos, sin, slotMapping, kCache, scaleCache};
    for (const auto &tensor : tensors) {
        TORCH_CHECK(
            tensor.device() == device,
            "all DeepSeek-V4.1 indexer prologue tensors must share one NPU.");
    }
    const at::Tensor slots =
        slotMapping.scalar_type() == at::kLong
            ? slotMapping.to(at::kInt)
            : slotMapping;
    TORCH_CHECK(
        slots.scalar_type() == at::kInt,
        "slot_mapping must be INT32 or INT64.");

    const int64_t cacheBlockSize = kCache.size(1);
    const int64_t kCacheStride0 = kCache.stride(0);
    const int64_t kCacheStride1 = kCache.stride(1);
    const int64_t scaleCacheStride0 = scaleCache.stride(0);
    const int64_t scaleCacheStride1 = scaleCache.stride(1);

    EXEC_NPU_CMD(
        aclnnFusedIndexerPrologue, latent, wkWeight, normWeight, cos, sin,
        slots, kCache, scaleCache, epsilon, cacheBlockSize, kCacheStride0,
        kCacheStride1, scaleCacheStride0, scaleCacheStride1, kCache, scaleCache);
}
}  // namespace vllm_ascend

#endif  // FUSED_INDEXER_PROLOGUE_TORCH_ADPT_H_
