# Aligned Mamba metadata

`aligned_mamba_block_table` in `models/glm5next/ops/metadata.py` gathers the
recurrent state window used by the GLM-5.3-Flash KDA metadata builder in
`mamba_cache_mode=align`. It combines the
sequence-length arithmetic and gather into one Triton launch on NPU, avoiding
the seven small PyTorch kernels before decode graph execution.

For request `r` and column `c`, the output is
`block_table[r, max((seq_lens[r] - 1) // block_size, 0) + c]`, where
`0 <= c <= num_speculative_blocks`. Sequence lengths come from the live device
tensor so asynchronous speculative corrections are preserved. Zero-length
graph rows start at column zero, matching the PyTorch reference.

Inputs are an INT32 or INT64 block table of shape `[requests, pages]` and an
INT32 or INT64 sequence-length tensor of shape `[requests]` on the same
device. Row, column and sequence strides are supported. The result has shape
`[requests, 1 + num_speculative_blocks]` and the block table's dtype. Empty
batches do not launch a kernel. Callers must provide nonnegative sequence
lengths and enough table columns for every selected window.

Only `Glm5NextLinearAttention` selects `Glm5NextKDABackend` and its dedicated
`Glm5NextKDAMetadataBuilder`. This builder selects the fused lookup on NPU;
its CPU lookup retains the reference path. Its pure speculative batches also
reuse their existing request order instead of transferring identity indices
and gathering state pages and accepted-token counts again. Mixed batches
retain their request partitioning, and full graph inputs still use the
existing persistent padding buffers.

Shared GDN builders default to the original lookup, index staging and gather
path. Qwen, Kimi and 310P do not select the GLM builder or import its fused
lookup through the shared metadata builder. Regression tests check both GLM
selection and the original calls for the other builders.

Run correctness, graph replay and host-submission benchmarks on Ascend with:

```bash
pytest -v tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_mamba_metadata.py \
  --junitxml=mamba-metadata-results.xml
```

The benchmark records baseline and fused milliseconds per metadata build for
1, 8 and 128 requests. End-to-end validation should compare the same GLM decode
workload before and after the change, including token accuracy, latency and
stream idle intervals; the microbenchmark does not establish model speedup.
