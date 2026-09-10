/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
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
#ifndef K2Q_CSR_TORCH_ADPT_H
#define K2Q_CSR_TORCH_ADPT_H

#include <algorithm>
#include <cstdint>
#include <tuple>

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include "torch_npu/csrc/core/npu/NPUFunctions.h"

namespace vllm_ascend {

inline int64_t k2q_csr_get_aiv_num_from_device()
{
    int32_t device_id = 0;
    aclError derr = c10_npu::GetDevice(&device_id);
    if (derr != ACL_SUCCESS) {
        device_id = static_cast<int32_t>(c10_npu::current_device());
    }

    int64_t aiv = 0;
    aclError ret =
        aclrtGetDeviceInfo(static_cast<uint32_t>(device_id), ACL_DEV_ATTR_VECTOR_CORE_NUM, &aiv);
    if (ret != ACL_SUCCESS || aiv <= 0) {
        ret = aclGetDeviceCapability(static_cast<uint32_t>(device_id), ACL_DEVICE_INFO_VECTOR_CORE_NUM, &aiv);
    }
    TORCH_CHECK(ret == ACL_SUCCESS && aiv > 0,
                "npu_k2q_csr: query VECTOR_CORE_NUM failed, device=", device_id,
                " ret=", static_cast<int>(ret), " aiv=", aiv);
    return aiv;
}

/** Align with op_host FillCudaLikeGroups (kMinQPerCta=256). */
inline int64_t k2q_csr_cuda_like_group_count(int64_t T, int64_t aiv)
{
    constexpr int64_t kMinQPerCta = 256;
    if (T <= 0) {
        return 1;
    }
    if (aiv < 1) {
        aiv = 1;
    }
    int64_t targetG = aiv * 2;
    if (targetG > aiv) {
        targetG = aiv;
    }
    int64_t maxGForQ = (T + kMinQPerCta - 1) / kMinQPerCta;
    if (maxGForQ < 1) {
        maxGForQ = 1;
    }
    int64_t G = targetG;
    if (G > maxGForQ) {
        G = maxGForQ;
    }
    if (G > T) {
        G = T;
    }
    if (G < 1) {
        G = 1;
    }
    int64_t qpc = (T + G - 1) / G;
    if (qpc < 1) {
        qpc = 1;
    }
    G = (T + qpc - 1) / qpc;
    if (G < 1) {
        G = 1;
    }
    if (G > aiv) {
        G = aiv;
    }
    return G;
}

inline void k2q_csr_calc_cu_block_stats(const at::Tensor &cu_block_lens, int64_t &total_rows,
                                        int64_t &max_kv)
{
    TORCH_CHECK(cu_block_lens.dim() == 1, "cu_block_lens must be 1-D");
    at::Tensor host = cu_block_lens.contiguous().to(at::kCPU);
    const int64_t n = host.numel();
    if (n <= 0) {
        total_rows = 0;
        max_kv = 0;
        return;
    }

    const int32_t *p = host.data_ptr<int32_t>();
    total_rows = static_cast<int64_t>(p[n - 1]);
    max_kv = 0;
    for (int64_t i = 0; i + 1 < n; ++i) {
        int64_t d = static_cast<int64_t>(p[i + 1]) - static_cast<int64_t>(p[i]);
        if (d > max_kv) {
            max_kv = d;
        }
    }
}

/**
 * q2k[H,T,topk] + cu_seqlens + cu_block_lens -> (row_ptr, q_ind, slot)
 *
 * Stages: Meta -> Hist -> RowPrefix -> TilePrefix -> Scatter
 * total_rows / max_kv: use CPU values when >= 0; otherwise derive via D2H.
 * use_simt: Hist/Scatter SIMT on ascend950 (ignored on other SoCs by tiling).
 * q_global_offset: 1 → q_ind is global Q token index; 0 → batch-local (default).
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_k2q_csr(
    const at::Tensor &q2k, const at::Tensor &cu_seqlens, const at::Tensor &cu_block_lens,
    int64_t order_method, int64_t total_rows, int64_t max_kv, int64_t use_simt,
    int64_t q_global_offset = 0)
{
    TORCH_CHECK(q2k.defined() && cu_seqlens.defined() && cu_block_lens.defined(),
                "npu_k2q_csr: inputs must be defined");
    TORCH_CHECK(q2k.scalar_type() == at::kInt, "q2k must be int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_block_lens.scalar_type() == at::kInt, "cu_block_lens must be int32");
    TORCH_CHECK(q2k.dim() == 3, "q2k must be 3-D [H, T, topk], got dim=", q2k.dim());
    TORCH_CHECK(order_method == 0 || order_method == 1, "order_method must be 0 or 1, got ",
                order_method);

    if (total_rows < 0 || max_kv < 0) {
        k2q_csr_calc_cu_block_stats(cu_block_lens, total_rows, max_kv);
    }
    TORCH_CHECK(total_rows >= 0, "total_rows must be >= 0, got ", total_rows);
    TORCH_CHECK(max_kv >= 0, "max_kv must be >= 0, got ", max_kv);

    const int64_t use_simt_i = (use_simt != 0) ? 1 : 0;
    const int64_t q_global_offset_i = (q_global_offset != 0) ? 1 : 0;

    const int64_t H = q2k.size(0);
    const int64_t T = q2k.size(1);
    const int64_t topk = q2k.size(2);
    const int64_t B = cu_block_lens.numel() > 0 ? cu_block_lens.numel() - 1 : 0;
    auto opts = q2k.options().dtype(at::kInt);

    at::Tensor q2k_c = q2k.contiguous();
    at::Tensor cu_q = cu_seqlens.contiguous();
    at::Tensor cu_b = cu_block_lens.contiguous();

    const int64_t aiv = k2q_csr_get_aiv_num_from_device();
    int64_t G = 1;
    if (use_simt_i != 0) {
        G = k2q_csr_cuda_like_group_count(T, aiv);
    } else {
        G = aiv;
        if (T > 0 && T < G) {
            G = T;
        }
        if (G < 1) {
            G = 1;
        }
    }

    // scratch: meta | tile_counts[G,H,R] | abs_base[G,H,R] | row_counts[H,R]
    int64_t meta = std::max<int64_t>(B, 0) * std::max<int64_t>(max_kv, 0) + std::max<int64_t>(T, 0);
    int64_t ghr = G * std::max<int64_t>(H, 1) * std::max<int64_t>(total_rows, 1);
    int64_t hist = 2 * ghr + std::max<int64_t>(H, 1) * std::max<int64_t>(total_rows, 1);
    int64_t scratch_elems = std::max<int64_t>(meta + hist, 1);
    at::Tensor scratch = at::empty({scratch_elems}, opts);

    EXEC_NPU_CMD(aclnnK2qCsrMeta, cu_q, cu_b, scratch, order_method, total_rows, max_kv, H, T, topk);

    at::Tensor row_ptr = at::empty({H, total_rows + 1}, opts);
    row_ptr.zero_();

    EXEC_NPU_CMD(aclnnK2qCsrHist, q2k_c, scratch, total_rows, max_kv, use_simt_i, B);

    EXEC_NPU_CMD(aclnnK2qCsrRowPrefix, scratch, total_rows, max_kv, use_simt_i, H, T, topk, B, row_ptr);
    EXEC_NPU_CMD(aclnnK2qCsrTilePrefix, scratch, row_ptr, total_rows, max_kv, use_simt_i, H, T, topk, B);

    at::Tensor q_ind = at::empty({H, T * topk}, opts);
    at::Tensor slot = at::empty({H, T * topk}, opts);
    q_ind.fill_(-1);
    slot.fill_(-1);

    EXEC_NPU_CMD(aclnnK2qCsrScatter, q2k_c, cu_q, scratch, total_rows, max_kv, use_simt_i,
                 q_global_offset_i, q_ind, slot);

    return std::make_tuple(row_ptr, q_ind, slot);
}

}  // namespace vllm_ascend

#endif  // K2Q_CSR_TORCH_ADPT_H
