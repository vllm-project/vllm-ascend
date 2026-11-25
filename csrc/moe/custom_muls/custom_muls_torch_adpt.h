#ifndef CUSTOM_MULS_TORCH_ADPT_H
#define CUSTOM_MULS_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor custom_muls(const at::Tensor& x, double scalar)
{
    TORCH_CHECK(x.is_privateuseone(), "custom_muls: x must be an NPU tensor");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat,
                "custom_muls: x must be bfloat16, float16 or float32");
    TORCH_CHECK(x.dim() <= 8, "custom_muls: x rank must be <= 8, but got ", x.dim());
    TORCH_CHECK(x.is_contiguous(), "custom_muls: x must be contiguous");

    at::Tensor y = at::empty_like(x);
    if (x.numel() == 0) {
        return y;
    }

    EXEC_NPU_CMD(aclnncustomMuls, x, scalar, y);
    return y;
}

} // namespace vllm_ascend

#endif // CUSTOM_MULS_TORCH_ADPT_H
