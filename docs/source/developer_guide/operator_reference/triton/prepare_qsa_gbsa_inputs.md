# prepare_qsa_gbsa_inputs

Prepare block-sparse attention inputs from selected QSA groups.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/qwen4_exp/gbsa.py#L84) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/qwen4_exp/gbsa.py#L58)

## Calling convention

```text
prepare_qsa_gbsa_inputs(compact_indices: torch.Tensor, logical_positions: torch.Tensor, seq_lens: torch.Tensor, token_to_req: torch.Tensor, block_table: torch.Tensor, num_kv_heads: int)
```

Produce the same integer tensors as the original GBSA torch pipeline.

The block table already addresses 128-token kernel pages. Residual blocks
remain explicit (residual_block_mode=0), including the 512-entry limit.
Every output element is written before use. No device-to-host reads or
process-global tensor cache are introduced; inputs can change every call.

See the [shared implementation and constraints](implementation_notes/qwen4_exp_qsa_dispatch.md).
