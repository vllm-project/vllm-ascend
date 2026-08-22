/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
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

at::Tensor turbo_quant_compress_latent(const at::Tensor &latent, const at::Tensor &centroids)
{
    TORCH_CHECK(latent.dim() == 2 && latent.size(1) == TQ_COMPRESS_HEAD_DIM,
                "turbo_quant_compress_latent expects a [N, ", TQ_COMPRESS_HEAD_DIM,
                "] latent, but got shape ", latent.sizes());
    at::Tensor slot = at::empty({latent.size(0), TQ_COMPRESS_SLOT_BYTES}, latent.options().dtype(at::kByte));
    EXEC_NPU_CMD(aclnnTurboQuantCompressLatent, latent, centroids, slot);
    return slot;
}

} // namespace vllm_ascend
#endif
