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
