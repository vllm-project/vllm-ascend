# _bad_words_kernel

## Description

- **Function**: Masks the final token of every bad-word sequence whose prefix matches the request history by setting the corresponding logit to negative infinity.
- **Formula**: For token row `t` and bad word `w=(w_0,...,w_n)`, `logits[t,w_n] = -inf` when the last `n` committed or speculative tokens equal `w_0,...,w_(n-1)`.
- **Algorithm flow** (processed token row by token row, independently): map the expanded token to its request, scan that request's bad words, compare each prefix against committed and speculative token buffers, and mask the final token on a match.
- **Supported modes**: Atlas A2, Atlas A3, and Ascend 950. Used by regular and speculative sampling; graph mode is N/A because sampling runs after the captured model-forward graph.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `logits_ptr` | Input/Output | Logits `[num_tokens, vocab_size]`; modified in place | fp32 | ND |
| `logits_stride` | Attribute | Row stride of `logits_ptr` | int | scalar |
| `expanded_idx_mapping_ptr` | Input | Token-to-request mapping `[num_tokens]` | int32 | ND |
| `bad_word_token_ids_ptr` | Input | Flattened bad-word tokens per request | int32 | ND |
| `bad_word_token_ids_stride` | Attribute | Request stride of bad-word tokens | int | scalar |
| `bad_word_offsets_ptr` | Input | Word boundaries `[max_num_reqs, max_num_bad_words + 1]` | int32 | ND |
| `bad_word_offsets_stride` | Attribute | Request stride of offsets | int | scalar |
| `num_bad_words_ptr` | Input | Bad-word count per request | int32 | ND |
| `all_token_ids_ptr` | Input | Committed token history | int32 | ND |
| `all_token_ids_stride` | Attribute | Request stride of token history | int | scalar |
| `prompt_len_ptr`, `total_len_ptr` | Input | Prompt and total lengths per request | int32 | ND |
| `input_ids_ptr` | Input | Current expanded input tokens | int32 | ND |
| `expanded_local_pos_ptr` | Input | Local speculative position per token | int32 | ND |
| `num_tokens`, `max_num_bad_words` | Attribute | Active token rows and scan bound | int | scalar |
| `MAX_PREFIX_LEN` | Attribute | Reserved compile-time value passed as 32; currently unused by the kernel body | int32 | scalar |

## Constraints

- Bad-word token storage is limited to 1024 tokens and 128 words per request. Offsets must remain within the 1024-token row; final token IDs must lie in `[0, vocab_size)`.
- There is no independently enforced 32-token prefix limit: `MAX_PREFIX_LEN=32` is passed by the wrapper but is not referenced in the kernel.
- Mapping indices, offsets, lengths, positions, and token IDs must be valid for their backing tensors. Inputs must be contiguous in their documented last dimensions.
- The kernel changes only matching logits and supports dynamic active-token counts without recompilation.

## Origin and Differences

- **Origin**: Adapted from vLLM's `vllm/v1/worker/gpu/sample/bad_words.py`.
- **Differences**:
    - NPU adaptation for performance: distributes token rows across Ascend vector cores and reuses request state inside each token loop.
    - Modified for vllm-ascend logic: handles committed and speculative token histories in the same kernel.

## Test Cases

The NPU test covers 512–2048 token rows, 16–64 requests, vocabulary size 50257, zero and maximum bad-word counts, and the 1024-token storage boundary. It checks whether logits are changed or preserved as expected, but does not yet compare every element against a reference; no numerical tolerance applies.

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bad_words.py
```
