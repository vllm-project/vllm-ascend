# Model Runner V2 penalty bincount

This implementation adapts [PR #15270](https://github.com/vllm-project/vllm-ascend/pull/15270)
by [zihanlin-ai](https://github.com/zihanlin-ai), pinned at
`ef792533de1d4fcd25e458eda701fa105ffafeb4`.
It preserves the public `bincount` signature and persistent tensor layouts.

## Statistics used by sampling penalties

| Statistic | Storage | Consumer |
| --- | --- | --- |
| Token occurs in the prompt | One bit per vocabulary entry, packed in int32 words | Repetition penalty |
| Output token multiplicity | int32 count per vocabulary entry | Frequency penalty |
| Token occurs in the output | Derived from `count != 0` | Presence and repetition penalties |

`expanded_idx_mapping` selects persistent request rows. It does not necessarily
contain consecutive row numbers. Rows outside the mapping and tensor padding
remain untouched. Token IDs and mappings can use int32 or int64.

## Computation

1. Create a zero-initialized int8 scratch row per staged request using
   `torch.zeros`. Clear the mapped output-count rows using Torch.
2. In the statistics kernel, write `1` to prompt flag entries for observed
   tokens. Repeated tokens write the same byte value, avoiding packed-word
   `atomic_or`. Output multiplicities retain `atomic_add`.
3. Pack 32 prompt flags into each int32 bitmap word. Each active word is fully
   overwritten, including a masked final partial word, so the persistent prompt
   bitmap does not need a separate clear.

These are two Triton kernels. Torch initialization can launch additional device
tasks; two Triton kernels does not mean two total device tasks. There is no
custom initialization kernel.

Scratch usage is `staged_requests * ceil(vocab_size / 32) * 32` bytes. The
scratch rows follow staged-request order, while persistent masks/counts use
mapped request-state indices. Scratch storage ends with the call.

## Work mapping and compilation

Statistics use fixed 1024-token tiles. Request/tile work is flattened and split
into contiguous quotient/remainder intervals over
`min(actual Vector cores, total token tiles)` programs. The existing device
capability helper supplies the core count. This avoids the oversized launch at
the 64-request, 1M-token boundary described in
[issue #19958](https://github.com/vllm-project/vllm-ascend/issues/19958).

Request and token-block counts remain runtime scalars. Tile sizes, packed-word
width and packing width are fixed structural constants. Their naming introduces
no variable tuning table or blanket specialization override. Ordinary scalar
value/alignment categories can still select compiler variants; the prior
three-kernel cache audit is not evidence for this revision.

## Validation

CPU wrapper UTs check initialization, mapped-row/padding preservation, empty
mapping and actual-core launch bounds. They mock kernel launches and do not
establish NPU arithmetic correctness:

```bash
pytest -q tests/ut/worker/v2/test_bincount.py
```

NPU tests check exact bitmaps/counts, bit 31, tails, strides, duplicate mappings,
unchanged inputs, Vector partitions, 1M launch boundaries and the chained
repetition/frequency/presence result:

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py
```

Zero-prompt inputs retain correctness coverage but are outside performance
optimization scope. Fresh shape/1M profiling, graph checks, AISBench GSM8K and
GPQA measurements must identify this revision before supporting performance or
model-accuracy claims. Results for the discarded custom-zero implementation are
historical and do not validate this implementation.
