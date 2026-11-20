// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <acl/acl_rt.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// These offsets are for the CANN 9.1 A2/A3 FIA BF16 TND launch. The runtime
// prepends one argument to the public fused_infer_attention_score signature.
constexpr size_t kQuery = 1;
constexpr size_t kKeyList = 2;
constexpr size_t kValueList = 3;
constexpr size_t kQueryLengths = 6;
constexpr size_t kKvLengths = 7;
constexpr size_t kBlockTable = 15;
constexpr size_t kOutput = 32;
constexpr size_t kTiling = 35;
constexpr size_t kLaunchArgs = 37;
constexpr size_t kArgsCapacity = 4096;
constexpr std::string_view kFiaPrefix = "FusedInferAttentionScore_";
constexpr std::string_view kFiaDecodeSinglePass = "_5000000000010200200";

using Address = uintptr_t;

struct Rebind {
  aclmdlRITask kernel;
  aclmdlRITaskParams old_params;
  std::vector<Address> args;
  Address address;
};

Address capture_model(Address stream) {
  aclmdlRICaptureStatus status{};
  aclmdlRI model{};
  if (aclmdlRICaptureGetInfo(reinterpret_cast<aclrtStream>(stream), &status, &model) != ACL_SUCCESS) {
    return 0;
  }
  return reinterpret_cast<Address>(model);
}

void check_update(aclError result, const char* function) {
  if (result != ACL_SUCCESS) {
    throw std::runtime_error(std::string(function) + " failed: " + std::to_string(result));
  }
}

bool bind_lengths(Address model_address, const std::vector<std::array<Address, 2>>& query_outputs, Address block_table,
                  const std::vector<int64_t>& capture_lengths, Address kv_lengths, Address args_storage) {
  if (query_outputs.empty() || capture_lengths.empty()) {
    return false;
  }
  auto model = reinterpret_cast<aclmdlRI>(model_address);
  uint32_t stream_count = 0;
  if (aclmdlRIGetStreams(model, nullptr, &stream_count) != ACL_SUCCESS || stream_count != 1) {
    return false;
  }
  aclrtStream stream{};
  if (aclmdlRIGetStreams(model, &stream, &stream_count) != ACL_SUCCESS) {
    return false;
  }
  uint32_t task_count = 0;
  if (aclmdlRIGetTasksByStream(stream, nullptr, &task_count) != ACL_SUCCESS) {
    return false;
  }
  std::vector<aclmdlRITask> tasks(task_count);
  if (aclmdlRIGetTasksByStream(stream, tasks.data(), &task_count) != ACL_SUCCESS) {
    return false;
  }

  std::vector<Rebind> bindings;
  for (auto task : tasks) {
    aclmdlRITaskType type{};
    if (aclmdlRITaskGetType(task, &type) != ACL_SUCCESS) {
      return false;
    }
    if (type != ACL_MODEL_RI_TASK_KERNEL) {
      continue;
    }
    aclmdlRITaskParams params{};
    if (aclmdlRITaskGetParams(task, &params) != ACL_SUCCESS) {
      return false;
    }
    auto& kernel = params.kernelTaskParams;
    std::array<char, 256> name{};
    if (aclrtGetFunctionName(kernel.funcHandle, name.size(), name.data()) != ACL_SUCCESS) {
      return false;
    }
    const std::string_view function(name.data());
    if (function.substr(0, kFiaPrefix.size()) != kFiaPrefix) {
      continue;
    }
    if (function.size() < kFiaDecodeSinglePass.size() ||
        function.substr(function.size() - kFiaDecodeSinglePass.size()) != kFiaDecodeSinglePass ||
        bindings.size() >= query_outputs.size() || kernel.isHostArgs || kernel.argsSize % sizeof(Address) ||
        kernel.argsSize < kLaunchArgs * sizeof(Address) || kernel.argsSize > kArgsCapacity) {
      return false;
    }

    std::vector<Address> args(kernel.argsSize / sizeof(Address));
    if (aclrtMemcpy(args.data(), kernel.argsSize, kernel.args, kernel.argsSize, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
      return false;
    }
    const auto [query, output] = query_outputs[bindings.size()];
    if (args[kQuery] != query || args[kBlockTable] != block_table || args[kOutput] != output) {
      return false;
    }

    const auto old_address = reinterpret_cast<Address>(kernel.args);
    const auto new_address = args_storage + bindings.size() * kArgsCapacity;
    const auto is_local = [&](Address pointer, size_t size) {
      return pointer >= old_address + kLaunchArgs * sizeof(Address) && pointer % alignof(Address) == 0 &&
             size <= kernel.argsSize && pointer - old_address <= kernel.argsSize - size;
    };
    const auto length_bytes = capture_lengths.size() * sizeof(int64_t);
    if (!is_local(args[kKeyList], sizeof(Address)) || !is_local(args[kValueList], sizeof(Address)) ||
        !is_local(args[kQueryLengths], length_bytes) || !is_local(args[kKvLengths], length_bytes) ||
        !is_local(args[kTiling], sizeof(Address))) {
      return false;
    }
    const size_t query_offset = (args[kQueryLengths] - old_address) / sizeof(Address);
    const size_t kv_offset = (args[kKvLengths] - old_address) / sizeof(Address);
    for (size_t i = 0; i < capture_lengths.size(); ++i) {
      if (args[query_offset + i] != i + 1 || args[kv_offset + i] != Address(capture_lengths[i])) {
        return false;
      }
    }

    for (auto slot : {kKeyList, kValueList, kQueryLengths, kTiling}) {
      args[slot] = new_address + (args[slot] - old_address);
    }
    args[kKvLengths] = kv_lengths;
    bindings.push_back({task, params, std::move(args), new_address});
  }
  if (bindings.size() != query_outputs.size()) {
    return false;
  }

  for (const auto& binding : bindings) {
    const auto bytes = binding.old_params.kernelTaskParams.argsSize;
    if (aclrtMemcpy(reinterpret_cast<void*>(binding.address), kArgsCapacity, binding.args.data(), bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      return false;
    }
  }

  // The runtime can fault the graph after a failed mutation. Propagate any
  // failure from this point so the caller never replays a partially changed graph.
  for (const auto& binding : bindings) {
    const auto& old = binding.old_params.kernelTaskParams;
    aclmdlRITaskParams replacement{};
    replacement.type = ACL_MODEL_RI_TASK_KERNEL;
    replacement.kernelTaskParams.funcHandle = old.funcHandle;
    replacement.kernelTaskParams.cfg = old.cfg;
    replacement.kernelTaskParams.args = reinterpret_cast<void*>(binding.address);
    replacement.kernelTaskParams.argsSize = old.argsSize;
    replacement.kernelTaskParams.numBlocks = old.numBlocks;
    check_update(aclmdlRITaskSetParams(binding.kernel, &replacement), "aclmdlRITaskSetParams");
  }
  check_update(aclmdlRIUpdate(model), "aclmdlRIUpdate");
  return true;
}

}  // namespace

PYBIND11_MODULE(_experimental_fia_replay, module) {
  module.attr("ARGS_CAPACITY") = kArgsCapacity;
  module.def("capture_model", &capture_model);
  module.def("bind_lengths", &bind_lengths);
}
