"""Opt-in shared-rotation bundles; every projection retains independent clips."""
import types
import torch
import torch_npu


def prepare(layers):
    if any(getattr(layer, 'bias', None) is not None for layer in layers):
        raise ValueError('Dense  bundled projections currently require no bias')
    if any(not getattr(layer, '_splitquant_weight_packed', False) for layer in layers):
        raise RuntimeError('Dense  bundle must be prepared after weight packing')
    rotations = [layer.group_trans.to(torch.bfloat16) for layer in layers]
    if not all(torch.equal(rotations[0], other) for other in rotations[1:]):
        raise ValueError('Cannot share different projection rotations')
    if any(getattr(layer, name, None) is not None for layer in layers for name in ('right_trans', 'left_trans')):
        raise ValueError('Dense  shared rotation does not support extra right/left transforms')
    splits = [layer.weight.shape[0] for layer in layers]
    g = layers[0].weight.shape[1] // 64
    def cast_nz(t):
        return torch_npu.npu_format_cast(t.contiguous(), 2)
    w = torch.cat([layer.weight for layer in layers], dim=0)
    return dict(
        weight=cast_nz(w.reshape(sum(splits), g, 64).permute(1, 0, 2)),
        scale=cast_nz(torch.cat([layer.weight_scale.float() for layer in layers], dim=0).t()),
        offset=cast_nz(torch.cat([layer.weight_offset.to(torch.int32).float() for layer in layers], dim=0).t()),
        clip_max=cast_nz(torch.stack([layer.activation_clip_max.float().reshape(g) for layer in layers])),
        clip_min=cast_nz(torch.stack([layer.activation_clip_min.float().reshape(g) for layer in layers])),
        rotation=cast_nz(rotations[0].transpose(1, 2)), splits=splits)


def apply(owner, layers, x):
    from vllm_ascend.quantization.w4a4_splitquant import load_splitquant_operator

    load_splitquant_operator()
    if not hasattr(owner, '_dense__bundle'):
        owner._dense__bundle = prepare(layers)
        owner.dense__bundle_calls = 0
    d = owner._dense__bundle
    y = torch.ops.npu.splitquant_dense_fused_multi(
        x.contiguous(), d['weight'], d['scale'], d['offset'], d['clip_max'],
        d['clip_min'], d['rotation'], d['splits'])
    owner.dense__bundle_calls += 1
    for layer in layers:
        method = layer.quant_method.quant_method  # type: ignore[attr-defined]
        method.fused_calls = getattr(method, 'fused_calls', 0) + 1  # type: ignore[attr-defined]
        method.fused_operator_name = 'splitquant_dense_fused_multi'  # type: ignore[attr-defined]
    return y


def attention_forward(self, positions, hidden_states):
    qkv = apply(self, (self.q_proj, self.k_proj, self.v_proj), hidden_states)
    q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
    q = self.q_norm(q.reshape(-1, self.num_heads, self.head_dim)).reshape(q.shape)
    k = self.k_norm(k.reshape(-1, self.num_kv_heads, self.head_dim)).reshape(k.shape)
    q, k = self.rotary_emb(positions, q, k)
    output, _ = self.o_proj(self.attn(q, k, v))
    return output


def mlp_forward(self, x):
    gate_up = apply(self, (self.gate_proj, self.up_proj), x)
    output, _ = self.down_proj(self.act_fn(gate_up))
    return output


def qwen35_mlp_forward(self, x):
    gate_up = apply(self, (self.gate_proj, self.up_proj), x)
    gate, up = gate_up.chunk(2, dim=-1)
    # Preserve the export's BF16 SiLU rounding before multiplication. The 14B
    # fused SiluAndMul path is a different numerical contract.
    output, _ = self.down_proj(torch.nn.functional.silu(gate) * up)
    return output


def install_qwen35(model):
    from vllm.distributed import get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError('Qwen3.6  bundled MLP currently requires TP=1')
    from vllm_ascend.models.qwen3_next import SplitQuantQwen3NextMLP
    for module in model.modules():
        if isinstance(module, SplitQuantQwen3NextMLP):
            module.forward = types.MethodType(qwen35_mlp_forward, module)  # type: ignore[method-assign]


def install(model):
    for layer in model.model.layers:
        layer.self_attn.forward = types.MethodType(attention_forward, layer.self_attn)
        layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)
