# copy_qsa_fia_output

Copy the FIA result to the QSA output buffer.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/ops/triton/qwen4_exp/fia.py#L198) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/models/qwen4_exp/fia.py#L183)

## Calling convention

```text
copy_qsa_fia_output(result: torch.Tensor, selected: torch.Tensor, output: torch.Tensor)
```

See the [shared implementation and constraints](implementation_notes/qwen4_exp_qsa_dispatch.md).
