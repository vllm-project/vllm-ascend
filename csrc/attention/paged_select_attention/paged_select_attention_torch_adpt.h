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
#ifndef PAGED_SELECT_ATTENTION_TORCH_ADPT_H
#define PAGED_SELECT_ATTENTION_TORCH_ADPT_H
namespace vllm_ascend {

namespace {
inline at::Tensor allocate_workspace_tensor(uint64_t workspace_size)
{
    at::TensorOptions options =
        at::TensorOptions(torch_npu::utils::get_npu_device_type());
    return at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(kByte));
}

// Host-side validation of the statically-knowable contract (shapes, dtypes,
// attrs). The device kernel re-checks the data-dependent invariants (e.g. that
// each selected logical page id is in range) which cannot be validated here
// without a device->host sync on the decode hot path. Keeping these checks on
// the host gives a clean, catchable error before the kernel ever launches.
inline void check_paged_select_attention_inputs(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const at::Tensor &block_table,
    const at::Tensor &selected_kv_indices,
    int64_t num_heads,
    int64_t num_key_value_heads,
    int64_t block_size)
{
    TORCH_CHECK(num_heads > 0 && num_key_value_heads > 0 && block_size > 0,
        "npu_paged_select_attention: num_heads, num_key_value_heads and block_size must be positive (got ",
        num_heads, ", ", num_key_value_heads, ", ", block_size, ").");
    TORCH_CHECK(num_heads % num_key_value_heads == 0,
        "npu_paged_select_attention: num_heads (", num_heads,
        ") must be a multiple of num_key_value_heads (", num_key_value_heads, ").");

    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
        "npu_paged_select_attention: query/key/value must be rank-3 (got dims ",
        query.dim(), "/", key.dim(), "/", value.dim(), ").");
    TORCH_CHECK(block_table.dim() == 2,
        "npu_paged_select_attention: block_table must be rank-2 (got dim ", block_table.dim(), ").");
    TORCH_CHECK(selected_kv_indices.dim() == 3,
        "npu_paged_select_attention: selected_kv_indices must be rank-3 (got dim ",
        selected_kv_indices.dim(), ").");

    const int64_t head_dim = query.size(2);
    TORCH_CHECK(query.size(1) == num_heads,
        "npu_paged_select_attention: query.size(1) (", query.size(1),
        ") must equal num_heads (", num_heads, ").");
    TORCH_CHECK(selected_kv_indices.size(1) == num_heads,
        "npu_paged_select_attention: selected_kv_indices.size(1) (", selected_kv_indices.size(1),
        ") must equal num_heads (", num_heads, ").");
    TORCH_CHECK(selected_kv_indices.size(2) > 0,
        "npu_paged_select_attention: selected_kv_indices.size(2) (k) must be > 0 (got ",
        selected_kv_indices.size(2), ").");

    TORCH_CHECK(key.size(1) == block_size && value.size(1) == block_size,
        "npu_paged_select_attention: key/value.size(1) must equal block_size (", block_size,
        ") (got ", key.size(1), "/", value.size(1), ").");
    TORCH_CHECK(key.size(2) == num_key_value_heads * head_dim && value.size(2) == num_key_value_heads * head_dim,
        "npu_paged_select_attention: key/value.size(2) must equal num_key_value_heads * head_dim (",
        num_key_value_heads * head_dim, ") (got ", key.size(2), "/", value.size(2), ").");

    TORCH_CHECK(selected_kv_indices.size(0) == block_table.size(0),
        "npu_paged_select_attention: selected_kv_indices.size(0) (", selected_kv_indices.size(0),
        ") must equal block_table.size(0) i.e. batch (", block_table.size(0), ").");

    TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
        "npu_paged_select_attention: query/key/value must share a dtype (got ",
        query.scalar_type(), "/", key.scalar_type(), "/", value.scalar_type(), ").");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
        "npu_paged_select_attention: query/key/value dtype must be float16 or bfloat16 (got ",
        query.scalar_type(), ").");
    TORCH_CHECK(block_table.scalar_type() == at::kInt,
        "npu_paged_select_attention: block_table must be int32 (got ", block_table.scalar_type(), ").");
    TORCH_CHECK(selected_kv_indices.scalar_type() == at::kInt,
        "npu_paged_select_attention: selected_kv_indices must be int32 (got ",
        selected_kv_indices.scalar_type(), ").");
}
} // namespace

at::Tensor npu_paged_select_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const at::Tensor &block_table,
    const at::Tensor &selected_kv_indices,
    int64_t num_heads,
    double scale_value,
    int64_t num_key_value_heads,
    int64_t block_size)
{
    check_paged_select_attention_inputs(
        query, key, value, block_table, selected_kv_indices,
        num_heads, num_key_value_heads, block_size);
    at::Tensor output = at::empty_like(query);

    EXEC_NPU_CMD(
        aclnnPagedSelectAttention,
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
        output);
    return output;
}

at::Tensor &npu_paged_select_attention_out(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const at::Tensor &block_table,
    const at::Tensor &selected_kv_indices,
    int64_t num_heads,
    double scale_value,
    int64_t num_key_value_heads,
    int64_t block_size,
    at::Tensor &output)
{
    check_paged_select_attention_inputs(
        query, key, value, block_table, selected_kv_indices,
        num_heads, num_key_value_heads, block_size);
    TORCH_CHECK(output.sizes() == query.sizes(),
        "npu_paged_select_attention_out: output shape must equal query shape (got ",
        output.sizes(), " vs ", query.sizes(), ").");
    EXEC_NPU_CMD(
        aclnnPagedSelectAttention,
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
        output);
    return output;
}

at::Tensor npu_paged_select_attention_get_workspace(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const at::Tensor &block_table,
    const at::Tensor &selected_kv_indices,
    int64_t num_heads,
    double scale_value,
    int64_t num_key_value_heads,
    int64_t block_size,
    const at::Tensor &output)
{
    check_paged_select_attention_inputs(
        query, key, value, block_table, selected_kv_indices,
        num_heads, num_key_value_heads, block_size);
    TORCH_CHECK(output.sizes() == query.sizes(),
        "npu_paged_select_attention_get_workspace: output shape must equal query shape (got ",
        output.sizes(), " vs ", query.sizes(), ").");
    static const auto getWorkspaceSizeFuncAddr =
        GetOpApiFuncAddr("aclnnPagedSelectAttentionGetWorkspaceSize");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr,
                "aclnnPagedSelectAttentionGetWorkspaceSize not in ", GetOpApiLibName(),
                ", or ", GetOpApiLibName(), "not found.");

    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;
    InitHugeMemThreadLocal initMemFunc =
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc =
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    auto converted_params = ConvertTypes(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
        output,
        workspace_size_addr,
        executor_addr);
    static auto getWorkspaceSizeFunc =
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
    TORCH_CHECK(workspace_status == 0,
                "call aclnnPagedSelectAttentionGetWorkspaceSize failed, detail:",
                aclGetRecentErrMsg());

    ReleaseConvertTypes(converted_params);
    ReleaseHugeMem releaseMemFunc =
        reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
    if (releaseMemFunc) {
        releaseMemFunc(nullptr, false);
    }
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }

    return allocate_workspace_tensor(workspace_size);
}

at::Tensor &npu_paged_select_attention_graph_out(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    const at::Tensor &block_table,
    const at::Tensor &selected_kv_indices,
    int64_t num_heads,
    double scale_value,
    int64_t num_key_value_heads,
    int64_t block_size,
    const at::Tensor &workspace,
    at::Tensor &output)
{
    check_paged_select_attention_inputs(
        query, key, value, block_table, selected_kv_indices,
        num_heads, num_key_value_heads, block_size);
    TORCH_CHECK(output.sizes() == query.sizes(),
        "npu_paged_select_attention_graph_out: output shape must equal query shape (got ",
        output.sizes(), " vs ", query.sizes(), ").");
    static const auto getWorkspaceSizeFuncAddr =
        GetOpApiFuncAddr("aclnnPagedSelectAttentionGetWorkspaceSize");
    static const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnPagedSelectAttention");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(
        getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
        "aclnnPagedSelectAttention or aclnnPagedSelectAttentionGetWorkspaceSize not in ",
        GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;
    InitHugeMemThreadLocal initMemFunc =
        reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc =
        reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }

    auto converted_params = ConvertTypes(
        query,
        key,
        value,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        block_table,
        selected_kv_indices,
        num_heads,
        scale_value,
        num_key_value_heads,
        block_size,
        output,
        workspace_size_addr,
        executor_addr);
    static auto getWorkspaceSizeFunc =
        ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
    TORCH_CHECK(workspace_status == 0,
                "call aclnnPagedSelectAttentionGetWorkspaceSize failed, detail:",
                aclGetRecentErrMsg());
    TORCH_CHECK(
        workspace.numel() >= static_cast<int64_t>(workspace_size),
        "paged_select_attention workspace tensor is too small. expected at least ",
        workspace_size, " bytes, got ", workspace.numel(), " bytes.");

    void *workspace_addr = nullptr;
    if (workspace_size != 0) {
        workspace_addr = const_cast<void *>(workspace.storage().data());
    }
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor,
                     opApiFuncAddr, releaseMemAddr]() -> int {
        typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,
                                 const aclrtStream);
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        TORCH_CHECK(api_ret == 0,
                    "call aclnnPagedSelectAttention failed, detail:",
                    aclGetRecentErrMsg());
        ReleaseConvertTypes(converted_params);
        ReleaseHugeMem releaseMemFunc =
            reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnPagedSelectAttention");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }
    return output;
}
}
#endif
