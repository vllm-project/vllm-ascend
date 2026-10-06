# CompressorV2

The host and kernel sources are ported from
[cann/ops-transformer at 50978ceeee11eb55f7c8d58c6f6ce5e81f0d879d](https://gitcode.com/cann/ops-transformer/tree/50978ceeee11eb55f7c8d58c6f6ce5e81f0d879d/experimental/attention/compressor_v2).
Their CANN Open Software License notices are preserved; the agreement is
included in [LICENSE](LICENSE).

`torch.ops._C_ascend.compressor_v2` projects values and gate scores, performs
per-channel softmax over each completed compression group, and updates an
FP32 ring state cache in place. It returns packed BF16/FP16 completed rows.
This interface does not include the normalization, RoPE, or APE operations
of the existing `compressor` operator.

For `[T,H]` inputs, the output capacity is `min(T, T // cmp_ratio + B)`;
only completed groups contain valid rows. `state_block_table` maps each
request to one ring block.

Accuracy and incremental-state tests are in
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_v2.py`.
