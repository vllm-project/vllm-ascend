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

#ifndef SPARSE_KV_GATHER_TORCH_ADPT_H
#define SPARSE_KV_GATHER_TORCH_ADPT_H

namespace vllm_ascend {
namespace sparse_kv_gather_detail {

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t CTKV_DIM = 512;
constexpr int64_t KPE_DIM = 64;

inline bool is_index_dtype(const at::ScalarType dtype)
{
    return dtype == at::kInt || dtype == at::kLong;
}

inline bool is_cache_dtype(const at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16;
}

inline void check_same_device(const at::Tensor &reference,
                              const at::Tensor &tensor,
                              const char *tensor_name)
{
    TORCH_CHECK(tensor.device() == reference.device(), tensor_name,
                " must be on device ", reference.device(),
                ", got ", tensor.device());
}

inline void validate_inputs(
    const at::Tensor &paged_ctkv,
    const at::Tensor &paged_kpe,
    const at::Tensor &block_table,
    const at::Tensor &topk_indices,
    const at::Tensor &cur_pos,
    const int64_t block_size)
{
    TORCH_CHECK(block_size == BLOCK_SIZE,
                "block_size must be 128, got ", block_size);
    TORCH_CHECK(is_cache_dtype(paged_ctkv.scalar_type()),
                "paged_ctkv must be float16 or bfloat16, got ",
                paged_ctkv.scalar_type());
    TORCH_CHECK(paged_kpe.scalar_type() == paged_ctkv.scalar_type(),
                "paged_kpe dtype must match paged_ctkv dtype");
    TORCH_CHECK(is_index_dtype(block_table.scalar_type()),
                "block_table must be int32 or int64");
    TORCH_CHECK(is_index_dtype(topk_indices.scalar_type()),
                "topk_indices must be int32 or int64");
    TORCH_CHECK(is_index_dtype(cur_pos.scalar_type()),
                "cur_pos must be int32 or int64");
    TORCH_CHECK(block_table.scalar_type() == topk_indices.scalar_type() &&
                    block_table.scalar_type() == cur_pos.scalar_type(),
                "block_table, topk_indices and cur_pos must use the same dtype");
    TORCH_CHECK(paged_ctkv.scalar_type() != at::kHalf ||
                    block_table.scalar_type() == at::kInt,
                "float16 cache currently requires int32 indices");

    TORCH_CHECK(paged_ctkv.dim() == 4,
                "paged_ctkv must have shape [num_blocks, 128, 1, 512]");
    TORCH_CHECK(paged_kpe.dim() == 4,
                "paged_kpe must have shape [num_blocks, 128, 1, 64]");
    TORCH_CHECK(block_table.dim() == 2,
                "block_table must have shape [num_actual, max_blocks]");
    TORCH_CHECK(topk_indices.dim() == 2,
                "topk_indices must have shape [num_actual, topk_n]");
    TORCH_CHECK(cur_pos.dim() == 1,
                "cur_pos must have shape [num_actual]");

    TORCH_CHECK(paged_ctkv.size(0) > 0 && paged_ctkv.size(1) == BLOCK_SIZE &&
                    paged_ctkv.size(2) == 1 && paged_ctkv.size(3) == CTKV_DIM,
                "paged_ctkv must have shape [num_blocks, 128, 1, 512]");
    TORCH_CHECK(paged_kpe.size(0) == paged_ctkv.size(0) &&
                    paged_kpe.size(1) == BLOCK_SIZE && paged_kpe.size(2) == 1 &&
                    paged_kpe.size(3) == KPE_DIM,
                "paged_kpe must have shape [num_blocks, 128, 1, 64] and share "
                "num_blocks with paged_ctkv");

    const int64_t num_actual = topk_indices.size(0);
    TORCH_CHECK(num_actual > 0 && topk_indices.size(1) > 0,
                "topk_indices dimensions must be non-zero");
    TORCH_CHECK(block_table.size(0) == num_actual && block_table.size(1) > 0,
                "block_table must have shape [num_actual, max_blocks]");
    TORCH_CHECK(cur_pos.size(0) == num_actual,
                "cur_pos must contain one value per topk_indices row");

    check_same_device(paged_ctkv, paged_kpe, "paged_kpe");
    check_same_device(paged_ctkv, block_table, "block_table");
    check_same_device(paged_ctkv, topk_indices, "topk_indices");
    check_same_device(paged_ctkv, cur_pos, "cur_pos");

    TORCH_CHECK(paged_ctkv.is_contiguous(), "paged_ctkv must be contiguous");
    TORCH_CHECK(paged_kpe.is_contiguous(), "paged_kpe must be contiguous");
    TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
    TORCH_CHECK(topk_indices.is_contiguous(), "topk_indices must be contiguous");
    TORCH_CHECK(cur_pos.is_contiguous(), "cur_pos must be contiguous");
}

inline void validate_outputs(
    const at::Tensor &paged_ctkv,
    const at::Tensor &paged_kpe,
    const at::Tensor &topk_indices,
    const at::Tensor &out_ctkv,
    const at::Tensor &out_kpe)
{
    const int64_t num_actual = topk_indices.size(0);
    const int64_t topk_n = topk_indices.size(1);

    TORCH_CHECK(out_ctkv.dim() == 3 && out_ctkv.size(0) == num_actual &&
                    out_ctkv.size(1) == topk_n && out_ctkv.size(2) == CTKV_DIM,
                "out_ctkv must have shape [", num_actual, ", ", topk_n,
                ", 512], got ", out_ctkv.sizes());
    TORCH_CHECK(out_kpe.dim() == 3 && out_kpe.size(0) == num_actual &&
                    out_kpe.size(1) == topk_n && out_kpe.size(2) == KPE_DIM,
                "out_kpe must have shape [", num_actual, ", ", topk_n,
                ", 64], got ", out_kpe.sizes());
    TORCH_CHECK(out_ctkv.scalar_type() == paged_ctkv.scalar_type(),
                "out_ctkv dtype must match paged_ctkv dtype");
    TORCH_CHECK(out_kpe.scalar_type() == paged_kpe.scalar_type(),
                "out_kpe dtype must match paged_kpe dtype");
    check_same_device(paged_ctkv, out_ctkv, "out_ctkv");
    check_same_device(paged_ctkv, out_kpe, "out_kpe");
    TORCH_CHECK(out_ctkv.is_contiguous(), "out_ctkv must be contiguous");
    TORCH_CHECK(out_kpe.is_contiguous(), "out_kpe must be contiguous");
}

inline void run_unchecked(
    const at::Tensor &paged_ctkv,
    const at::Tensor &paged_kpe,
    const at::Tensor &block_table,
    const at::Tensor &topk_indices,
    const at::Tensor &cur_pos,
    const int64_t block_size,
    at::Tensor &out_ctkv,
    at::Tensor &out_kpe)
{
    EXEC_NPU_CMD(aclnnSparseKvGather,
                 paged_ctkv,
                 paged_kpe,
                 block_table,
                 topk_indices,
                 cur_pos,
                 block_size,
                 out_ctkv,
                 out_kpe);
}

}  // namespace sparse_kv_gather_detail

inline void npu_sparse_kv_gather_out(
    const at::Tensor &paged_ctkv,
    const at::Tensor &paged_kpe,
    const at::Tensor &block_table,
    const at::Tensor &topk_indices,
    const at::Tensor &cur_pos,
    at::Tensor &out_ctkv,
    at::Tensor &out_kpe,
    const int64_t block_size)
{
    sparse_kv_gather_detail::validate_inputs(
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos, block_size);
    sparse_kv_gather_detail::validate_outputs(
        paged_ctkv, paged_kpe, topk_indices, out_ctkv, out_kpe);

    sparse_kv_gather_detail::run_unchecked(
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos,
        block_size, out_ctkv, out_kpe);
}

inline std::tuple<at::Tensor, at::Tensor> npu_sparse_kv_gather(
    const at::Tensor &paged_ctkv,
    const at::Tensor &paged_kpe,
    const at::Tensor &block_table,
    const at::Tensor &topk_indices,
    const at::Tensor &cur_pos,
    const int64_t block_size)
{
    sparse_kv_gather_detail::validate_inputs(
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos, block_size);

    const int64_t num_actual = topk_indices.size(0);
    const int64_t topk_n = topk_indices.size(1);
    at::Tensor out_ctkv = at::empty(
        {num_actual, topk_n, sparse_kv_gather_detail::CTKV_DIM},
        paged_ctkv.options());
    at::Tensor out_kpe = at::empty(
        {num_actual, topk_n, sparse_kv_gather_detail::KPE_DIM},
        paged_kpe.options());

    sparse_kv_gather_detail::run_unchecked(
        paged_ctkv, paged_kpe, block_table, topk_indices, cur_pos,
        block_size, out_ctkv, out_kpe);
    return std::make_tuple(out_ctkv, out_kpe);
}

}  // namespace vllm_ascend

#endif  // SPARSE_KV_GATHER_TORCH_ADPT_H
