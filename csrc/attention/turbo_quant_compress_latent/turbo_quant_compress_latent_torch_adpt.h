/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TQ_COMPRESS_LATENT_TORCH_ADPT_H
#define TQ_COMPRESS_LATENT_TORCH_ADPT_H

namespace vllm_ascend {

// TurboQuant compress: latent [N, 512] fp32 (post-rmsnorm, already multiplied by the
// signed Hadamard matrix, NOT normalized) + centroids [16] fp32
//   -> slot [N, 320] uint8 (256 packed TQ4 nibble bytes + 2 B fp16 vecNorm + 62 B pad).
// The Hadamard matmul and centroid preparation stay in Python
// (tq_latent_store.compress_kernel); this only wraps the aclnn host launch.
// The kernel is specialized for kv_lora_rank=512, hence the fixed slot width.
constexpr int64_t TQ_COMPRESS_HEAD_DIM = 512;
constexpr int64_t TQ_COMPRESS_SLOT_BYTES = 320;
constexpr int64_t TQ_COMPRESS_CENTROID_COUNT = 16;

at::Tensor turbo_quant_compress_latent(const at::Tensor &latent, const at::Tensor &centroids)
{
    TORCH_CHECK(latent.dim() == 2 && latent.size(1) == TQ_COMPRESS_HEAD_DIM,
                "turbo_quant_compress_latent expects a [N, ", TQ_COMPRESS_HEAD_DIM,
                "] latent, but got shape ", latent.sizes());
    TORCH_CHECK(latent.scalar_type() == at::kFloat,
                "turbo_quant_compress_latent expects a float32 latent");
    TORCH_CHECK(centroids.scalar_type() == at::kFloat,
                "turbo_quant_compress_latent expects float32 centroids");
    TORCH_CHECK(centroids.numel() == TQ_COMPRESS_CENTROID_COUNT,
                "turbo_quant_compress_latent expects exactly ", TQ_COMPRESS_CENTROID_COUNT,
                " centroids, but got ", centroids.numel());
    at::Tensor slot = at::empty({latent.size(0), TQ_COMPRESS_SLOT_BYTES}, latent.options().dtype(at::kByte));
    EXEC_NPU_CMD(aclnnTurboQuantCompressLatent, latent, centroids, slot);
    return slot;
}

} // namespace vllm_ascend
#endif
