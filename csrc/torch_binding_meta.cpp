#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include "utils.h"

namespace vllm_ascend {
namespace meta {

std::tuple<at::Tensor, at::Tensor> rotary_embedding_meta(
  at::Tensor &positions,
  at::Tensor &query,
  at::Tensor &key,
  int64_t head_size, 
  at::Tensor &cos_sin_cache,
  bool is_neox) {
    auto num_tokens = positions.sym_numel();
    auto query_hidden_size = query.sym_numel() / num_tokens;
    auto key_hidden_size = key.sym_numel() / num_tokens;

    auto num_heads = query_hidden_size / head_size;
    auto num_kv_heads = key_hidden_size / head_size;
    at::Tensor query_dst = at::empty_symint({num_tokens, num_heads, head_size}, query.options());
    at::Tensor key_dst = at::empty_symint({num_tokens, num_kv_heads, head_size}, key.options());

    return {query_dst, key_dst};
}

std::tuple<at::Tensor, at::Tensor> get_masked_input_and_mask_meta(
    at::Tensor &input,
    const int64_t org_vocab_start_index,
    const int64_t org_vocab_end_index,
    const int64_t num_org_vocab_padding,
    const int64_t added_vocab_start_index,
    const int64_t added_vocab_end_index) {

    at::Tensor masked_input = at::empty_like(input);
    at::Tensor mask = at::empty_like(input, input.options().dtype(at::kBool));

    return {masked_input, mask};
}


} // namespace meta
} // namespace vllm_ascend

namespace {
  // Register the meta implementations of the custom kernels for symbolic tracing, this will also 
  // the custom kernel been captured into aclgraph
  TORCH_LIBRARY_IMPL_EXPAND(_C, Meta, ops) {
    // Rotary embedding meta implementation
    ops.impl("rotary_embedding", &vllm_ascend::meta::rotary_embedding_meta);
    // Masked input and mask meta implementation
    ops.impl("get_masked_input_and_mask", &vllm_ascend::meta::get_masked_input_and_mask_meta);

}
}