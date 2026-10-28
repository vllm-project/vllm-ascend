// SPDX-License-Identifier: Apache-2.0
#include "attn_res_fwd_native_stubs.h"
#include "csrc/attention/attn_res_fwd/op_host/op_api/aclnn_attn_res_fwd.cpp"
#include <cassert>
#include <limits>

aclnnStatus Invoke(const aclTensor* projection, const aclTensor* addend, const aclTensor* outputNorm, aclTensor* output,
                   double eps = 1e-5, double outputEps = 1e-4, bool nullWorkspace = false, bool nullExecutor = false, int64_t bankStride = 12) {
  aclTensor prefix{1, 4}, bank{1, 3, 4}, norm{4}, raw{1, 4}, materialized{1, 4};
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  const auto status = aclnnAttnResFwdGetWorkspaceSize(
      &prefix, &bank, projection, &norm, addend, outputNorm, eps, 2, bankStride, 2, outputEps, true, true, true, false, output,
      &raw, &materialized, nullWorkspace ? nullptr : &workspaceSize, nullExecutor ? nullptr : &executor);
  delete executor;
  return status;
}

int main() {
  aclTensor projection{1, 4}, addend{1, 4}, gamma{4}, output{1, 4};
  assert(Invoke(&projection, &addend, &gamma, &output) == ACLNN_SUCCESS);
  assert(op::launchCalls == 1);
  aclTensor emptyProjection{0, 4}, scalarProjection{}, shortAddend{1}, shortGamma{1}, shortOutput{1};
  assert(Invoke(&emptyProjection, &addend, &gamma, &output) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&scalarProjection, &addend, &gamma, &output) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&projection, &shortAddend, &gamma, &output) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&projection, &addend, &shortGamma, &output) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&projection, &addend, &gamma, &shortOutput) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&projection, &addend, &gamma, nullptr) == ACLNN_ERR_PARAM_INVALID);
  for (double eps :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 1e300, 1e-300}) {
    assert(Invoke(&projection, &addend, &gamma, &output, eps) == ACLNN_ERR_PARAM_INVALID);
  }
  for (double eps :
       {-1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 1e300, 1e-300}) {
    assert(Invoke(&projection, &addend, &gamma, &output, 1e-5, eps) == ACLNN_ERR_PARAM_INVALID);
  }
  assert(Invoke(&projection, &addend, &gamma, &output, 1e-5, 1e-4, true) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&projection, &addend, &gamma, &output, 1e-5, 1e-4, false, true) == ACLNN_ERR_PARAM_INVALID);
  assert(Invoke(&projection, &addend, &gamma, &output, 1e-5, 1e-4, false, false, int64_t{1} << 30) ==
         ACLNN_ERR_PARAM_INVALID);
  projection.strides[1] = 2;
  assert(Invoke(&projection, &addend, &gamma, &output) == ACLNN_ERR_PARAM_INVALID);
  // No rejected call may reach the fake backend launcher.
  assert(op::launchCalls == 1);
}
