import torch


class _MissingOp:
    def __init__(self, op_name: str):
        self.op_name = op_name
        self.default = self

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f"Missing upstream op `{self.op_name}` was invoked.")


class _NpuRmsNormOp:
    """NPU implementation for torch.ops._C.rms_norm.

    Calling convention matches the CUDA op:
        rms_norm(out, input, weight, epsilon) -> None
    Writes the result into `out` in-place.
    """

    def __init__(self):
        self.default = self

    def __call__(self, out, input, weight, epsilon):
        import torch_npu

        result, _ = torch_npu.npu_rms_norm(input, weight, epsilon)
        out.copy_(result)


class _NpuRotaryEmbeddingOp:
    """NPU implementation for torch.ops._C.rotary_embedding.

    Calling convention matches the CUDA op:
        rotary_embedding(positions, query, key, head_size,
                         cos_sin_cache, is_neox) -> None
    Modifies `query` (and `key` if not None) in-place.
    """

    def __init__(self):
        self.default = self

    def __call__(self, positions, query, key, head_size, cos_sin_cache, is_neox):
        from vllm.model_executor.layers.rotary_embedding import (
            RotaryEmbedding,
        )

        q_result, k_result = RotaryEmbedding.forward_static(
            positions=positions,
            query=query,
            key=key,
            head_size=head_size,
            rotary_dim=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox_style=is_neox,
        )
        query.copy_(q_result)
        if key is not None and k_result is not None:
            key.copy_(k_result)


def _set_missing(namespace, op_name: str, full_name: str) -> None:
    if not hasattr(namespace, op_name):
        setattr(namespace, op_name, _MissingOp(full_name))


# --- Register real NPU implementations for the two ops that
#     precompute_and_store_context_kv (and potentially others) call ---
torch.ops._C.rms_norm = _NpuRmsNormOp()
torch.ops._C.rotary_embedding = _NpuRotaryEmbeddingOp()


_set_missing(torch.ops._C, "fused_add_rms_norm", "torch.ops._C.fused_add_rms_norm")
_set_missing(torch.ops._C, "static_scaled_fp8_quant", "torch.ops._C.static_scaled_fp8_quant")
_set_missing(torch.ops._C, "dynamic_scaled_fp8_quant", "torch.ops._C.dynamic_scaled_fp8_quant")
_set_missing(torch.ops._C, "dynamic_per_token_scaled_fp8_quant", "torch.ops._C.dynamic_per_token_scaled_fp8_quant")
_set_missing(torch.ops._C, "silu_and_mul", "torch.ops._C.silu_and_mul")