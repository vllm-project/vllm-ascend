#include <ATen/ATen.h>
#include <torch/library.h>

#include "aclnn_torch_adapter/op_api_common.h"
#include "../dcut_causal_conv1d/dcut_causal_conv1d_torch_adpt.h"
#include "../dcut_recurrent_gated_delta_rule/dcut_recurrent_gated_delta_rule_torch_adpt.h"

namespace vllm_ascend::dcut_meta {

at::Tensor npu_dcut_causal_conv1d_meta(
    const at::Tensor& output,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& conv_state,
    const c10::optional<at::Tensor>& bias,
    const c10::optional<at::Tensor>& query_start_loc,
    const c10::optional<at::Tensor>& cache_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    int64_t activation_mode,
    int64_t pad_slot_id) {
  return output;
}

at::Tensor npu_dcut_recurrent_gated_delta_rule_meta(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& state,
    const c10::optional<at::Tensor>& beta,
    const c10::optional<double> scale,
    const c10::optional<at::Tensor>& query_start_loc,
    const c10::optional<at::Tensor>& ssm_state_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    const c10::optional<at::Tensor>& g,
    const c10::optional<at::Tensor>& gk,
    bool zero_padded_output) {
  auto options = value.options().dtype(at::ScalarType::BFloat16);
  return at::empty_symint(value.sym_sizes(), options);
}

}  // namespace vllm_ascend::dcut_meta

TORCH_LIBRARY_FRAGMENT(_C_ascend, ops) {
  ops.def(
      "npu_dcut_recurrent_gated_delta_rule(Tensor query, "
      "                                    Tensor key, "
      "                                    Tensor value, "
      "                                    Tensor(a!) state, "
      "                                    *, "
      "                                    Tensor? beta=None, "
      "                                    float? scale=None, "
      "                                    Tensor? query_start_loc=None, "
      "                                    Tensor? ssm_state_indices=None, "
      "                                    Tensor? num_accepted_tokens=None, "
      "                                    Tensor? g=None, "
      "                                    Tensor? gk=None, "
      "                                    bool zero_padded_output=False) -> Tensor");
  ops.def(
      "npu_dcut_causal_conv1d(Tensor(a!) output, Tensor x, "
      "                         Tensor weight, Tensor(b!) conv_state, "
      "                         Tensor? bias=None, "
      "                         Tensor? query_start_loc=None, "
      "                         Tensor? cache_indices=None, "
      "                         Tensor? num_accepted_tokens=None, "
      "                         int activation_mode=0, "
      "                         int pad_slot_id=-1) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, ops) {
  ops.impl("npu_dcut_recurrent_gated_delta_rule",
           &vllm_ascend::npu_dcut_recurrent_gated_delta_rule);
  ops.impl("npu_dcut_causal_conv1d", &vllm_ascend::npu_dcut_causal_conv1d);
}

TORCH_LIBRARY_IMPL(_C_ascend, Meta, ops) {
  ops.impl("npu_dcut_recurrent_gated_delta_rule",
           &vllm_ascend::dcut_meta::npu_dcut_recurrent_gated_delta_rule_meta);
  ops.impl("npu_dcut_causal_conv1d",
           &vllm_ascend::dcut_meta::npu_dcut_causal_conv1d_meta);
}
