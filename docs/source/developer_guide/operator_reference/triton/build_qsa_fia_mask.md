# build_qsa_fia_mask

Build the FIA attention mask for selected QSA groups.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/qwen4_exp/fia.py#L157) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/qwen4_exp/fia.py#L153)

## Calling convention

```text
build_qsa_fia_mask(compact_indices: torch.Tensor, visible_lens: torch.Tensor, kv_length: int, mask_width: int, workspace: QSAFIAMaskWorkspace)
```

Fill and return a reusable [T, KV] mask plus nonempty-row flags.

See the [shared implementation and constraints](implementation_notes/qwen4_exp_qsa_dispatch.md).
