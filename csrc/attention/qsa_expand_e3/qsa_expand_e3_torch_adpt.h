#ifndef QSA_EXPAND_E3_TORCH_ADPT_H
#define QSA_EXPAND_E3_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor& qsa_expand_e3_out(
    const at::Tensor& groups,
    const at::Tensor& complete_groups,
    const at::Tensor& tail_start,
    const at::Tensor& tail_count,
    const at::Tensor& sequence_lengths,
    const at::Tensor& token_to_req,
    at::Tensor& out)
{
    TORCH_CHECK(groups.dim() == 2 && groups.size(1) == 512,
                "QSA E3 groups must have shape [rows, 512]");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == groups.size(0) &&
                    out.size(1) == 2051,
                "QSA E3 output must have shape [rows, 2051]");
    EXEC_NPU_CMD(aclnnQsaExpandE3, groups, complete_groups, tail_start,
                 tail_count, sequence_lengths, token_to_req, out);
    return out;
}
}  // namespace vllm_ascend

#endif
