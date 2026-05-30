import torch
import torch.nn.functional as F
import torch_npu
from vllm import ir
from vllm.model_executor.layers.layernorm import RMSNormGated

from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm


class AscendRMSNorm310(AscendRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
            if self.bias is not None:
                x.add_(self.bias)
            return x, residual

        x = ir.ops.rms_norm(x, self.weight, self.variance_epsilon)
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
            x = ir.ops.rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x = ir.ops.rms_norm(x, 1.0 + self.weight, self.variance_epsilon)
        return x


class AscendRMSNormGated310(RMSNormGated):
    def _apply_activation(self, z: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(z)
        if self.activation in ("silu", "swish"):
            return F.silu(z)
        raise AssertionError(f"Unsupported activation: {self.activation}")

    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.group_size is not None:
            return super().forward_native(x, z)

        if z is not None and not self.norm_before_gate:
            x = torch.mul(x, self._apply_activation(z))

        x, _ = torch_npu.npu_rms_norm(x, self.weight, self.eps)

        if z is not None and self.norm_before_gate:
            x = torch.mul(x, self._apply_activation(z))

        return x
