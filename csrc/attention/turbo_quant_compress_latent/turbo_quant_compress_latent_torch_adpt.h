#ifndef TQ_COMPRESS_LATENT_TORCH_ADPT_H
#define TQ_COMPRESS_LATENT_TORCH_ADPT_H

namespace vllm_ascend {

// TurboQuant compress: latent [N,512] fp32 (post-rmsnorm, @ signed-Hadamard, NOT normalized)
// + centroids [16] fp32 -> legacy [N,320] or compact corrected [N,258] uint8 slot.
// Hadamard matmul + centroid prep stay in Python (tq_latent_store.compress_kernel); this op
// keeps output_mode=0 compatible with the GLM path.
at::Tensor turbo_quant_compress_latent(const at::Tensor &latent, const at::Tensor &centroids, int64_t output_mode)
{
    constexpr int64_t legacySlotBytes = 320;
    constexpr int64_t compactSlotBytes = 258;
    TORCH_CHECK(output_mode == 0 || output_mode == 1,
                "TurboQuant compression output_mode must be 0 or 1, got ", output_mode);
    int64_t N = latent.size(0);
    int64_t slotBytes = output_mode == 1 ? compactSlotBytes : legacySlotBytes;
    at::Tensor slot = at::empty({N, slotBytes}, latent.options().dtype(at::kByte));
    EXEC_NPU_CMD(aclnnTurboQuantCompressLatent, latent, centroids, output_mode, slot);
    return slot;
}

} // namespace vllm_ascend
#endif
