# Vocabulary-parallel logits packing

## Description

- **Function**: Pack gathered DeepSeek V4.1 Flash target logits into a contiguous vocabulary dimension for Model Runner V2.
- **Formula**: For an output element at row `r` and vocabulary index `v`, read
  `gathered[(v // shard_size) * num_rows + r, v % shard_size]` and convert to the output dtype.
- **Algorithm flow**: Each vector tile reads a contiguous span within one TP shard and writes it to its final location.
  The load and store masks omit padded vocabulary columns.
- **Supported modes**: Atlas A2 and A3 with standard TP and Triton available. Other paths use the existing logits processor.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `gathered` | Input | Rank-ordered result of a row-axis TP all-gather | BF16, FP16, FP32 | Contiguous `[tp_size * num_rows, shard_size]` |
| `num_rows` | Input | Number of sampled rows per TP rank | Integer | Scalar |
| `vocab_size` | Input | Number of vocabulary columns after padding is removed | Integer | Scalar |
| `out_dtype` | Attribute | Requested output dtype, or `None` to preserve the input dtype | FP32 or `None` | Scalar |
| Return value | Output | Packed logits | Input dtype or FP32 | Contiguous `[num_rows, vocab_size]` |

## Constraints

- Each TP rank contributes an equal-sized shard. The unpadded vocabulary must fit in the gathered shards.
- The model caller uses the stock gather when the shard or vocabulary width is unaligned for 32-byte vector stores.
- The wrapper allocates the output and leaves the input unchanged. It returns an empty output without a kernel launch for zero rows.
- The model calls the wrapper from sampling after model execution. Graph capture of the wrapper is outside this integration.
- The kernel keeps `num_rows` dynamic and specializes on shard and vocabulary sizes, tile geometry, and input and output dtypes.

## Origin and Differences

- **Origin**: [vLLM #65436](https://github.com/vllm-project/vllm/pull/65436).
- **Differences**: Uses the Ascend TP group and an Ascend Triton vector kernel for DeepSeek V4.1 Flash.

## Test Cases

The operator test compares layout and values exactly with the PyTorch reference, including the DeepSeek V4.1 Flash vocabulary at TP8,
empty and single-row inputs, padding, and FP32 output.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_vocab_parallel_logits.py
```
