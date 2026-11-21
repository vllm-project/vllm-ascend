# A3 recurrent GDN integration

## Runtime contract

The A3 fusion path accepts the verified FLA wheel
`flash-linear-attention-npu-a3==26.10.0+main.dev0058bb4`.
The source commit is `0058bb4a98e3d1d50958cab9dba996dd1f1027f4`.
Version, installed-module location, selected backend signature, hardware
capability and tensor metadata are checked before enabling fusion. Unknown
versions or unsupported inputs use external preprocessing.

Eligible calls use BF16 QKV, FP32 recurrent state, head dimensions 128 and
supported gate/state strides. Recurrent GDN performs Q/K normalization,
gate calculation and beta sigmoid internally. Verified noncontiguous QKV
views avoid materialization. Cached FP32 gate bias is refreshed after weight
loading and transfer, preserving addresses used by captured graphs.
Chunk prefill keeps `use_beta_sigmoid_in_kernel=False`: its beta is already
activated. It retains internal Q/K normalization and the native state layout.

## Compilation cache behavior

The npugraph_ex cache stores an inner kernel, not the full AOT runtime wrapper.
`AscendCompiler.load` rebuilds the backend callable to retain input mutation,
alias and output semantics. This adds startup compilation work; it does not
recompile every decode step.

## Current hardware evidence

On A3 physical NPUs 0/2/3, two PP2 prefill services and one DP2 decode service
ran the same frozen GSM8K workload: input body 131072 tokens, output 1024,
concurrency 24, 96 requests, shared-prefix warmup only, end-to-end total metrics.
P cache hits were 89.6423% in both states; each run completed 96/96 requests.

| State | Output tokens/s | Average TTFT ms | Average TPOT ms | MTP acceptance |
| --- | ---: | ---: | ---: | ---: |
| GDN adapter rollback + FLA 26.9.1 | 781.3638 | 13910.9 | 14.5 | 77.2158% |
| Current adapters + FLA 0058bb4 | 808.9380 | 14121.6 | 13.5 | 81.1657% |

These are individual observations, not proof of stable gains or model accuracy.
The new-wheel run flagged four outputs with at least 50 occurrences of role
words; full quality evaluation remains pending. Both TTFT averages exceed the
10-second target. The source and wheel changed together, so this comparison
cannot isolate either one's contribution. No new profiling is claimed here.

Earlier numerical and profiling experiments are historical evidence only;
their full records remain outside this source tree. Merge requires model-quality
validation, controlled performance evidence and upstream CI.
