import torch


class _MissingOp:
    def __init__(self, op_name: str):
        self.op_name = op_name
        self.default = self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"Missing upstream op `{self.op_name}` was invoked.")


def _set_missing(namespace, op_name: str, full_name: str) -> None:
    if not hasattr(namespace, op_name):
        setattr(namespace, op_name, _MissingOp(full_name))


_set_missing(torch.ops._C, "rms_norm", "torch.ops._C.rms_norm")
_set_missing(torch.ops._C, "fused_add_rms_norm", "torch.ops._C.fused_add_rms_norm")
_set_missing(torch.ops._C, "rotary_embedding", "torch.ops._C.rotary_embedding")
_set_missing(torch.ops._C, "static_scaled_fp8_quant", "torch.ops._C.static_scaled_fp8_quant")
_set_missing(torch.ops._C, "dynamic_scaled_fp8_quant", "torch.ops._C.dynamic_scaled_fp8_quant")
_set_missing(torch.ops._C, "dynamic_per_token_scaled_fp8_quant", "torch.ops._C.dynamic_per_token_scaled_fp8_quant")
_set_missing(torch.ops._C, "silu_and_mul", "torch.ops._C.silu_and_mul")
