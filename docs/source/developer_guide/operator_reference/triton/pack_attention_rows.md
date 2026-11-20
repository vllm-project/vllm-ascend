# pack_attention_rows

Pack attention rows for HiSparse staging.

[Wrapper implementation](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/hisparse/cache_ops.py#L503) · [Call or registration](https://github.com/vllm-project/vllm-ascend/blob/c3afc3faab78a012cd519aba0e8f3b618eea3ca3/vllm_ascend/attention/sfa_hisparse.py#L476)

## Calling convention

```text
pack_attention_rows(packed_cache, physical_topk, selected_latent_out, selected_rope_out, latent_dim)
```

Gather packed physical KV rows into the split, dense SFA input layout.

See the [shared implementation and constraints](implementation_notes/hisparse.md#parameters) and [test coverage](implementation_notes/hisparse.md#test-cases).
