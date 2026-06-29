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

#ifndef MEGA_MOE_COMM_CONTEXT_H
#define MEGA_MOE_COMM_CONTEXT_H

#include <string>
#include <cstdint>
#include <ATen/Tensor.h>
#include <torch/custom_class.h>

namespace vllm_ascend {

// ======================== Backend mode enum ========================
enum class BackendMode : uint8_t { UNINITIALIZED, KFC, CHANNEL };

// ======================== Communication context struct ========================
// This struct is serialized into an int32 NPU tensor and passed to the
// aclnnMegaMoe operator as the "context" input.

static constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSizePerServer = 0;
    uint64_t kfcContextAddr = 0;
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE] = {};
};

// ======================== Forward declarations ========================
class KfcContextBuilder;
class HcclChannelContextBuilder;

// ======================== CommContextManager ========================
// Creates and manages HCCL communication context tensors for MC2 operators.
// Supports automatic backend resolution and explicit backend override:
//   - "auto"    : Resolve from aclrtGetSocName()
//   - "kfc"     : KFC mode (Ascend910B / Ascend910_93)
//   - "channel" : HCCL Channel mode (Ascend950)
//
// Python usage via torch::class_:
//   ctx_mgr = torch.classes._C_ascend.CommContextManager(group_name, world_size, "auto")
//   context_tensor = ctx_mgr.create_context()
//   ccl_buf_size = ctx_mgr.ccl_buffer_size

class CommContextManager : public torch::CustomClassHolder {
public:
    CommContextManager(const std::string &group, int64_t worldSize,
                       const std::string &backend = "auto");

    // Create the communication context tensor on NPU device.
    // Returns an int32 1D tensor containing the serialized CommContext struct.
    at::Tensor create_context();

    // Update the HCCL group name and recreate the context tensor.
    void update_group(const std::string &group, at::Tensor &contextTensor);

    // The size of the HCCL communication buffer in bytes.
    int64_t ccl_buffer_size() const { return cclBufferSize_; }

private:
    static int64_t context_tensor_size();

    void EnsureResolved();
    void DispatchBuild(at::Tensor &tensor);

    std::string group_;
    int64_t worldSize_;
    std::string backend_;
    BackendMode mode_;
    int64_t cclBufferSize_ = 0;
};

// ======================== SoC name utility ========================
const char *GetSocName();

} // namespace vllm_ascend

#endif // MEGA_MOE_COMM_CONTEXT_H
