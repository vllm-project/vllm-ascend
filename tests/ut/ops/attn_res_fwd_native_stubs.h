// SPDX-License-Identifier: Apache-2.0
// Host-metadata stand-ins only: this test does not emulate CANN or an NPU.
#pragma once
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <vector>
using aclnnStatus = int;
using aclrtStream = void*;
constexpr int ACLNN_SUCCESS = 0;
constexpr int ACLNN_ERR_PARAM_INVALID = 1;
constexpr int ACLNN_ERR_INNER_CREATE_EXECUTOR = 2;
#define ACLNN_API
namespace op {
enum class DataType { DT_BF16, DT_FLOAT };
}
struct TestShape {
  std::vector<int64_t> dims;
  TestShape(std::initializer_list<int64_t> values) : dims(values) {}
  size_t GetDimNum() const { return dims.size(); }
  int64_t GetDim(size_t index) const { return dims.at(index); }
};
struct aclTensor {
  TestShape shape;
  std::vector<int64_t> strides;
  op::DataType dtype = op::DataType::DT_BF16;
  aclTensor(std::initializer_list<int64_t> values) : shape(values), strides(values.size()) {
    int64_t stride = 1;
    for (size_t dim = values.size(); dim > 0; --dim) {
      strides[dim - 1] = stride;
      stride *= shape.GetDim(dim - 1);
    }
  }
  const TestShape& GetViewShape() const { return shape; }
  const std::vector<int64_t>& GetViewStrides() const { return strides; }
  op::DataType GetDataType() const { return dtype; }
  void SetOriginalShape(const TestShape&) const {}
  bool IsEmpty() const {
    for (auto dim : shape.dims)
      if (dim == 0) return true;
    return false;
  }
};
struct aclOpExecutor {
  uint64_t GetWorkspaceSize() const { return 0; }
};
namespace op {
inline unsigned launchCalls = 0;
struct TestExecutor {
  std::unique_ptr<aclOpExecutor> executor{new aclOpExecutor};
  aclOpExecutor* get() const { return executor.get(); }
  aclOpExecutor* operator->() const { return executor.get(); }
  void ReleaseTo(aclOpExecutor** output) { *output = executor.release(); }
};
inline TestExecutor CreateExecutor() { return {}; }
}  // namespace op
#define L0_DFX(...)
#define L2_DFX_PHASE_1(...)
#define L2_DFX_PHASE_2(...)
#define OP_TYPE_REGISTER(...)
#define ADD_TO_LAUNCHER_LIST_AICORE(...) (++op::launchCalls, ACLNN_SUCCESS)
#define CREATE_EXECUTOR() op::CreateExecutor()
inline aclnnStatus CommonOpExecutorRun(void*, uint64_t, aclOpExecutor*, aclrtStream) { return ACLNN_SUCCESS; }
