import torch
import torch.nn.functional as F

from vllm_ascend.ops.activation import AscendSiluAndMul as _Base


class AscendSiluAndMul310(_Base):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.ops.vllm.maybe_prefetch_mlp_down_proj(x)
        h = x.shape[-1] // 2
        out = (F.silu(x[..., :h].to(torch.float32)) *
               x[..., h:].to(torch.float32)).to(torch.float16)
        torch.ops.vllm.maybe_wait_prefetch_done(out)
        return out
