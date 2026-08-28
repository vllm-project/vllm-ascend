#ifndef DCUT_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H
#define DCUT_RECURRENT_GATED_DELTA_RULE_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_dcut_recurrent_gated_delta_rule(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                                               at::Tensor& state, const c10::optional<at::Tensor>& beta,
                                               const c10::optional<double> scale,
                                               const c10::optional<at::Tensor>& query_start_loc,
                                               const c10::optional<at::Tensor>& ssm_state_indices,
                                               const c10::optional<at::Tensor>& num_accepted_tokens,
                                               const c10::optional<at::Tensor>& g,
                                               const c10::optional<at::Tensor>& gk,
                                               bool zero_padded_output) {
  TORCH_CHECK(scale.has_value(), "scale cannot be empty.");
  TORCH_CHECK(query_start_loc.has_value(), "query_start_loc cannot be empty.");
  TORCH_CHECK(num_accepted_tokens.has_value(), "num_accepted_tokens cannot be empty.");

  auto options = value.options().dtype(at::ScalarType::BFloat16);
  at::Tensor output =
      zero_padded_output ? at::zeros(value.sizes(), options) : at::empty(value.sizes(), options);
  float scale_real = static_cast<float>(scale.value());
  EXEC_NPU_CMD(aclnnDcutRecurrentGatedDeltaRule, query, key, value, beta, state, query_start_loc, ssm_state_indices, g,
               gk, num_accepted_tokens, scale_real, output);
  return output;
}

}  // namespace vllm_ascend
#endif
