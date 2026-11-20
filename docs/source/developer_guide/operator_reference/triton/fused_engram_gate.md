# fused_engram_gate

Apply the fused Engram gate and residual update.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/engram_gate.py#L51) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/deepseek_v41/engram/layer.py#L47)

## Calling convention

```text
fused_engram_gate(hidden, kv, q_weight, k_weight, token_mask, eps)
```

Fuse RMS/dot/gate/residual, keeping wkv projection outside this kernel.

See the [shared implementation and constraints](implementation_notes/engram.md).
