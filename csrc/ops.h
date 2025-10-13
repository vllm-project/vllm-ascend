/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>
#include "kernels/types.h"
#include "torch_npu/csrc/aten/common/from_blob.h"

namespace vllm_ascend {
  extern void rotary_embedding_impl(AscendType type, bool isNeox, void *stream, int64_t *positions, void *queryDst,
    void *keyDst, void *query, void *key, void *cosSinCache, const int rotDim,
    const int64_t queryStride, const int64_t keyStride, const int64_t dstQueryStride,
    const int64_t dstKeyStride, const int numHeads, const int numKvHeads,
    const int headSize, const int64_t numTokens, const uint32_t loopCnt,
    uint32_t aivNum);

  extern void get_masked_input_and_mask_impl(
    void* stream,
    void* input,
    void* masked_input,
    void* mask_out,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding, 
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index,
    const int64_t size,
    const uint32_t loop_cnt,
    const uint32_t aiv_num);
    
  torch::Tensor weak_ref_tensor(torch::Tensor& tensor) {
    if (!tensor.is_privateuseone()) {
      throw std::runtime_error("Tensor must be on NPU device");
    }
    // Get the raw data pointer
    void* data_ptr = tensor.data_ptr();
    // Get tensor sizes and strides
    std::vector<int64_t> sizes = tensor.sizes().vec();
    std::vector<int64_t> strides = tensor.strides().vec();
    // Get tensor options (dtype, device)
    auto options = tensor.options();
    // Create a new tensor from the raw data pointer
    auto new_tensor = at_npu::native::from_blob(data_ptr, sizes, strides, options);
    return new_tensor;
  }

  extern void bgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y, 
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void bgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *indices,
        uint32_t indicesSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);

    extern void sgmv_shrink_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t input_hidden_dim,
        uint32_t lora_rank,
        float scale);

    extern void sgmv_expand_impl(
        AscendType type,
        void *stream,
        void *x,
        void *weight,
        void *loraIndices,
        uint32_t loraIndicesSize,
        void *seqLen,
        uint32_t seqLenSize,
        void *y,
        void *y_out,
        uint32_t batch_size,
        uint32_t num_tokens_per_core,
        uint32_t lora_rank,
        uint32_t output_hidden_dim,
        uint32_t slice_offset,
        uint32_t output_full_dim);
}
