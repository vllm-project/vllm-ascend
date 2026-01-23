import torch
import torch_npu

from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm


class AscendRMSNorm310(AscendRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            orig_dtype = residual.dtype
            if x is None or x.numel() == 0 or x.shape[-1] == 0:
                x = residual.to(dtype=residual.dtype)
            else:
                x = x + residual.to(x.dtype)

            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias is not None:
            x.add_(self.bias)
        return x


class AscendGemmaRMSNorm310(AscendGemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
        return x
