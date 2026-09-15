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
#ifndef BLASST_ATTENTION_SCORE_TORCH_ADPT_H
#define BLASST_ATTENTION_SCORE_TORCH_ADPT_H

#include <tuple>
#include <utility>
#include <cstdio>
#include <cstring>
#include <vector>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"

namespace vllm_ascend {
// Element count of the per-core sparse-stats output tensor ([sparseNum,
// countNum] pairs padded to a cache line; mirrors the op's stats layout).
static constexpr int64_t kSparseStatsElems = 16;


#define EXEC_NPU_CMD_WS(aclnn_api, ws_tensor, ...)                              \
  do {                                                                        \
    static const auto getWorkspaceSizeFuncAddr =                              \
        GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                      \
    static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);           \
    static const auto initMemAddr =                                           \
        GetOpApiFuncAddr("InitHugeMemThreadLocal");                           \
    static const auto unInitMemAddr =                                         \
        GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                         \
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");    \
    TORCH_CHECK(                                                              \
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,      \
        #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",        \
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");         \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);           \
    uint64_t workspace_size = 0;                                              \
    uint64_t *workspace_size_addr = &workspace_size;                          \
    aclOpExecutor *executor = nullptr;                                        \
    aclOpExecutor **executor_addr = &executor;                                \
    InitHugeMemThreadLocal initMemFunc =                                      \
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
    UnInitHugeMemThreadLocal unInitMemFunc =                                  \
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);            \
    if (initMemFunc) {                                                        \
      initMemFunc(nullptr, false);                                            \
    }                                                                         \
    auto converted_params =                                                   \
        ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);        \
    static auto getWorkspaceSizeFunc =                                        \
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);       \
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);     \
    TORCH_CHECK(workspace_status == 0,                                        \
                "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg()); \
    void *workspace_addr = nullptr;                                           \
    if (workspace_size != 0) {                                                \
      TORCH_CHECK((ws_tensor).numel() >= static_cast<int64_t>(workspace_size), \
                  #aclnn_api " provided workspace too small: have ",          \
                  (ws_tensor).numel(), " need ", workspace_size);             \
      workspace_addr = (ws_tensor).data_ptr();                                \
    }                                                                         \
    auto acl_call = [converted_params, workspace_addr, workspace_size,        \
                     acl_stream, executor]() -> int {                         \
      typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                               const aclrtStream);                            \
      OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
      auto api_ret =                                                          \
          opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
      TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                  aclGetRecentErrMsg());                                      \
      ReleaseConvertTypes(converted_params);                                  \
      ReleaseHugeMem releaseMemFunc =                                         \
          reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
      if (releaseMemFunc) {                                                   \
        releaseMemFunc(nullptr, false);                                       \
      }                                                                       \
      return api_ret;                                                         \
    };                                                                        \
    at_npu::native::OpCommand cmd;                                            \
    cmd.Name(#aclnn_api);                                                     \
    cmd.SetCustomHandler(acl_call);                                           \
    cmd.Run();                                                                \
    if (unInitMemFunc) {                                                      \
      unInitMemFunc(nullptr, false);                                          \
    }                                                                         \
  } while (false)

// chunked +14%、decode q=1 +373%）。
inline void fia_exec_common(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    const at::IntArrayRef &actual_seq_lengths,
    const at::IntArrayRef &actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &blocktable,
    int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise, int64_t block_size,
    int64_t antiquant_mode, double sparse_lambda, bool softmax_lse_flag,
    bool sparse_stats_flag, bool flash_decode,
    at::Tensor &attention_out, at::Tensor &softmax_lse, at::Tensor &sparse_stats,
    const c10::optional<at::Tensor> &workspace)
{
    TORCH_CHECK(query.dim() == 3, "query must be 3D TND layout");
    TORCH_CHECK(key.dim() == 3, "key must be 3D TND layout");
    TORCH_CHECK(value.dim() == 3, "value must be 3D TND layout");
    TORCH_CHECK(antiquant_mode == 0, "only antiquant_mode 0 is supported in this migration");

    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    // host-list-only: seq lens reach the kernel via the host IntArray attrs
    // embedded into TilingData by the op host (framework-managed upload,
    // task-update safe). The device seq inputs are passed as absent
    // optionals (nullptr aclTensor), mirroring the torch_npu builtin FIA
    // interface contract of host-list-only seq lengths.
    const c10::optional<at::Tensor> seqQAbsent;
    const c10::optional<at::Tensor> seqKvAbsent;

    if (workspace.has_value()) {
        EXEC_NPU_CMD_WS(
            aclnnVllmBlasstAttentionScore, *workspace,
            query, key, value, pse_shift, atten_mask,
            seqQAbsent, seqKvAbsent, blocktable,
            num_heads, scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size,
            antiquant_mode, sparse_lambda, softmax_lse_flag,
            actual_seq_lengths, actual_seq_lengths_kv, sparse_stats_flag,
            flash_decode,
            attention_out, softmax_lse, sparse_stats);
    } else {
        EXEC_NPU_CMD(
            aclnnVllmBlasstAttentionScore,
            query, key, value, pse_shift, atten_mask,
            seqQAbsent, seqKvAbsent, blocktable,
            num_heads, scale, pre_tokens, next_tokens, input_layout_ptr,
            num_key_value_heads, sparse_mode, inner_precise, block_size,
            antiquant_mode, sparse_lambda, softmax_lse_flag,
            actual_seq_lengths, actual_seq_lengths_kv, sparse_stats_flag,
            flash_decode,
            attention_out, softmax_lse, sparse_stats);
    }
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_blasst_attention_score(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &blocktable,
    int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise, int64_t block_size,
    int64_t antiquant_mode, double sparse_lambda, bool softmax_lse_flag,
    bool sparse_stats_flag, bool flash_decode)
{
    at::Tensor attention_out = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    at::Tensor softmax_lse = at::empty({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
    at::Tensor sparse_stats = at::empty({kSparseStatsElems}, query.options().dtype(at::kInt));

    fia_exec_common(query, key, value, pse_shift, atten_mask,
                    actual_seq_lengths, actual_seq_lengths_kv, blocktable,
                    num_heads, scale, pre_tokens, next_tokens, input_layout,
                    num_key_value_heads, sparse_mode, inner_precise, block_size,
                    antiquant_mode, sparse_lambda, softmax_lse_flag,
                    sparse_stats_flag, flash_decode,
                    attention_out, softmax_lse, sparse_stats, c10::nullopt);
    return std::make_tuple(attention_out, softmax_lse, sparse_stats);
}

inline void npu_blasst_attention_score_out(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    at::Tensor &attention_out, at::Tensor &softmax_lse, at::Tensor &sparse_stats,
    const c10::optional<at::Tensor> &workspace,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &blocktable,
    int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise, int64_t block_size,
    int64_t antiquant_mode, double sparse_lambda, bool softmax_lse_flag,
    bool sparse_stats_flag, bool flash_decode)
{
    fia_exec_common(query, key, value, pse_shift, atten_mask,
                    actual_seq_lengths, actual_seq_lengths_kv, blocktable,
                    num_heads, scale, pre_tokens, next_tokens, input_layout,
                    num_key_value_heads, sparse_mode, inner_precise, block_size,
                    antiquant_mode, sparse_lambda, softmax_lse_flag,
                    sparse_stats_flag, flash_decode,
                    attention_out, softmax_lse, sparse_stats, workspace);
}

inline int64_t npu_blasst_attention_score_get_workspace(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const c10::optional<at::Tensor> &blocktable,
    int64_t num_heads, double scale, int64_t pre_tokens, int64_t next_tokens,
    c10::string_view input_layout, int64_t num_key_value_heads,
    int64_t sparse_mode, int64_t inner_precise, int64_t block_size,
    int64_t antiquant_mode, double sparse_lambda, bool softmax_lse_flag,
    bool sparse_stats_flag, bool flash_decode)
{
    TORCH_CHECK(query.dim() == 3, "query must be 3D TND layout");
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    // host-list-only: seq lens reach the kernel via host IntArray attrs
    // embedded into TilingData; the device seq inputs stay absent (see
    // fia_exec_common).
    const c10::optional<at::Tensor> seqQAbsent;
    const c10::optional<at::Tensor> seqKvAbsent;
    at::Tensor attention_out = at::empty(query.sizes(), query.options().dtype(query.dtype()));
    at::Tensor softmax_lse = at::empty({query.size(0), query.size(1), 1}, query.options().dtype(at::kFloat));
    at::Tensor sparse_stats = at::empty({kSparseStatsElems}, query.options().dtype(at::kInt));

    static const auto getWorkspaceSizeFuncAddr =
        GetOpApiFuncAddr("aclnnVllmBlasstAttentionScoreGetWorkspaceSize");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr,
                "aclnnVllmBlasstAttentionScoreGetWorkspaceSize not in ",
                GetOpApiLibName());
    InitHugeMemThreadLocal initMemFunc =
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;
    auto converted_params = ConvertTypes(
        query, key, value, pse_shift, atten_mask,
        seqQAbsent, seqKvAbsent, blocktable,
        num_heads, scale, pre_tokens, next_tokens, input_layout_ptr,
        num_key_value_heads, sparse_mode, inner_precise, block_size,
        antiquant_mode, sparse_lambda, softmax_lse_flag,
        actual_seq_lengths, actual_seq_lengths_kv, sparse_stats_flag,
        flash_decode,
        attention_out, softmax_lse, sparse_stats,
        workspace_size_addr, executor_addr);
    static auto getWorkspaceSizeFunc =
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
    TORCH_CHECK(workspace_status == 0,
                "call aclnnVllmBlasstAttentionScoreGetWorkspaceSize failed, detail:",
                aclGetRecentErrMsg());
    ReleaseConvertTypes(converted_params);
    ReleaseHugeMem releaseMemFunc =
        reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
    if (releaseMemFunc) {
        releaseMemFunc(nullptr, false);
    }
    UnInitHugeMemThreadLocal unInitMemFunc =
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    return static_cast<int64_t>(workspace_size);
}

} // namespace vllm_ascend

#endif // BLASST_ATTENTION_SCORE_TORCH_ADPT_H
