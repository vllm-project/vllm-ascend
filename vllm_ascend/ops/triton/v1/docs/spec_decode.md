# V1 speculative decoding helpers

## NGram

`vllm_ascend.ops.triton.v1.spec_decode.ngram` provides
`triton_ngram_spec_decode` and `ngram_spec_decode_kernel` for
`worker/v1/model_runner.py`. It filters accepted sampled tokens, updates token
history, and finds the longest matching n-gram to produce draft tokens and their
valid counts. The current V1 token-history and discard-mask interfaces remain
unchanged.

## Proposer input preparation

`vllm_ascend.ops.triton.v1.spec_decode.utils` contains:

| Kernel | V1 caller |
| --- | --- |
| `prepare_inputs_padded_kernel` | `worker/v1/spec_decode/llm_base_proposer.py` |
| `copy_and_expand_dflash_and_dspark_inputs_kernel` | V1 DFlash and DSpark proposers |
| `dflash2_greedy_selector_walk_kernel` | `worker/v1/spec_decode/dflash2_proposer.py` |

V1 rejection-sampler warmup also compiles `prepare_inputs_padded_kernel`.
V2 proposer implementations are under `worker/v2/spec_decode/` and use their
own input-preparation paths.

Run the existing operator tests on NPU:

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_prepare_inputs_padded.py
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_copy_and_expand_dflash_dspark.py
```
