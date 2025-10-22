from typing import Optional

import torch
import vllm
from vllm.forward_context import ForwardContext, get_forward_context


def forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    # For some alternate attention backends like MLA the attention output
    # shape does not match the query shape, so we optionally let the model
    # definition specify the output tensor shape.
    output_shape: Optional[torch.Size] = None,
) -> torch.Tensor:
    """
    The KV cache is stored inside this class and is accessed via
    `self.kv_cache`.

    Attention metadata (`attn_metadata`) is set using a context manager in
    the model runner's `execute_model` method. It is accessed via forward
    context using
    `vllm.forward_context.get_forward_context().attn_metadata`.
    """
    if self.calculate_kv_scales:
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata.enable_kv_scales_calculation:
            self.calc_kv_scales(query, key, value)

    output_dtype = query.dtype
    if self.query_quant is not None:
        # quantizing with a simple torch operation enables
        # torch.compile to fuse this into previous ops
        # which reduces overheads during decoding.
        # Otherwise queries are quantized using custom ops
        # which causes decoding overheads
        assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}
        query, _ = self.query_quant(query, self._q_scale)

    if self.use_output:
        output_shape = (output_shape
                        if output_shape is not None else query.shape)
        output = torch.empty(output_shape,
                                dtype=output_dtype,
                                device=query.device)
        hidden_size = output_shape[-1]
        # We skip reshaping query, key and value tensors for the MLA
        # backend since these tensors have different semantics and are
        # processed differently.
        if not self.use_mla:
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size)
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            self.impl.forward(self,
                                query,
                                key,
                                value,
                                self_kv_cache,
                                attn_metadata,
                                output=output)
        else:
            torch.ops.vllm.unified_attention_with_output(
                query, key, value, output, self.layer_name)
        return output.view(-1, hidden_size)
    else:
        if self.use_direct_call:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            return self.impl.forward(self, query, key, value,
                                        self_kv_cache, attn_metadata)
        else:
            return torch.ops.vllm.unified_attention(
                query, key, value, self.layer_name)

vllm.attention.layer.Attention.forward = forward
