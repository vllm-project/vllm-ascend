# V2 penalty bincount

The V2 sampler builds a packed prompt-presence bitmap and output-token counts
when penalty-using requests are admitted. This implementation continues
[PR #15270](https://github.com/vllm-project/vllm-ascend/pull/15270) by
[zihanlin-ai](https://github.com/zihanlin-ai), pinned at
`ef792533de1d4fcd25e458eda701fa105ffafeb4`.
Commit `129aa539530513236225e6cc60f40077ae885304` migrates that operator source
and its tests verbatim; the optimization is a separate signed commit.

## Computation and ownership

The public `bincount` signature and persistent int32 layouts are unchanged.
Tokens are valid vocabulary IDs; prompt/output histories have
`0 <= prompt_len <= prefill_len <= max_prefill_len` within token storage.
The upstream caller stages request indices and supplies their maximum prefill
length. Selected output rows are overwritten; other rows and padding survive.

Torch initializes one byte per rounded vocabulary entry per staged request and
clears selected output-count rows. The first Triton kernel scans fixed
1024-token tiles, writing prompt flag bytes and retaining integer `atomic_add`
for output multiplicities. The second kernel packs flags into int32 words.
It overwrites every active word, so the persistent prompt mask needs no clear.

Each stage flattens its independent request/tile work into `T` tiles and uses
`P = min(actual Vector cores, T)` programs. Program `i` processes
`q + (i < r)` consecutive tiles starting at `i*q + min(i,r)`, where
`q = T // P` and `r = T % P`. These intervals cover every tile once and their
lengths differ by at most one. Equal tile counts do not guarantee equal time
for ragged requests. The actual device helper supplies the core count; no
physical core count is hard-coded. Empty mappings return before launch, and
zero histories skip statistics while still initializing outputs.

This bounds the launch at the
[issue #19958](https://github.com/vllm-project/vllm-ascend/issues/19958)
boundary. Its old grid had 65,472 / 65,536 / 65,600 programs for 64 requests
with 1,047,552 / 1,048,576 / 1,049,600 tokens. The new grid remains bounded
by the device's Vector count while programs loop over their assigned tiles.

### Request rows, programs and scattered writes

The temporary flags have shape `[R, 32 * num_words]`, where R is the number
of staged request-mapping entries. They are shared device memory, with one row
per staged request, rather than one vocabulary per history token or program.
For R=64 and vocab151936, this allocation is 9,723,904 bytes (about 9.27 MiB).

| Kernel variable | Meaning |
| --- | --- |
| `pid` | Program index in the bounded one-dimensional grid |
| `tile` | Flattened request/block task owned by that program |
| `token_idx` | Staged request row, obtained by dividing `tile` by blocks per request |
| `req_state_idx` | Persistent request-state row loaded through the mapping |
| `prompt_tokens` | Actual token IDs used as vocabulary column indices |

Here `token_idx` is the name inside the two bincount kernels; it does not
index an individual history token. A program can process tiles from several
requests, and several programs can process different history tiles of the
same request. No program owns a private vocabulary. The later logits-penalty
kernel has a different meaning for its own `token_idx`.

The first stage still scatters: for prompt token ID t it writes
`flags[staged_request, t] = 1`. For example, prompt `[7, 7, 35]` sets columns
7 and 35; the repeated 7 writes the same value. Presence needs one bit,
whereas output frequency needs multiplicity and retains integer atomic adds.
The second stage reads contiguous vocabulary flags and writes each packed
word to the mapped persistent request row. Kernel launch order supplies the
stage boundary before packing begins.

The optimization therefore retains the original PR's scattered prompt writes
and dense scratch initialization/packing. It bounds launch work and reduces
the packing intermediate width; it does not reduce the per-request scratch
size. Dense vocabulary work explains the fixed cost for short histories.

## Packing four flags per load

The scratch remains int8 and has `ceil(vocab_size / 32) * 32` bytes per row.
A zero-copy int32 view lets packing load four flag bytes per element without
an in-kernel pointer reinterpretation. For little-endian flag bytes
`a,b,c,d` in `{0,1}`, the high byte of the uint32 product with
`0x01020408` equals `a + 2*b + 4*c + 8*d`. The result is at most 15;
eight shifted nibbles sum to one 32-bit presence word, including bit31.
All unused vocabulary bytes were zeroed, so the final word's full loads are
in bounds and its unused bits stay zero.

Compared with expanding each flag to int32, the reduction width falls from
32 to 8. Scratch allocation and GM bytes read stay unchanged. The packing
tile remains 128 bitmap words: direct 256-word expansion was slower and
512 overflowed UB; compact 256/512/1024-word variants passed correctness
but lost single-request performance. A nested-loop metadata-hoisting control
showed no useful device-work improvement, so the simpler tile loops remain.

`BITS_PER_WORD`, `FLAGS_PER_LOAD` and the gather multiplier are fixed
`tl.constexpr` format constants. Only fixed tile sizes are kernel configuration
parameters. Request counts, work counts, strides and vocabulary-word counts
remain runtime scalars. Backend dtype/value/alignment specialization can still
produce variants; the measured compilation audit is recorded below.

### Triton 3.6 constant compatibility

The format constants are instantiated with `tl.constexpr(value)`. A Python
annotation such as `BITS_PER_WORD: tl.constexpr = 32` leaves an ordinary integer
global, which Triton 3.6 rejects inside JIT functions. Host scratch allocation
uses `BITS_PER_WORD.value` as its integer dimension. The algorithm, kernel
signatures, tile sizes and launch grids are unchanged.

With `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS` unset and a fresh task cache, the old
source reproduced the reported `NameError`; the corrected source passed all
27 standalone cases on Triton-Ascend 3.6.0 and 3.2.2, including bit31, strides,
int64 and all three 1M boundaries. Both runs used Ascend910_9362 (A3); they
establish compiler compatibility and A3 correctness, not fresh A5 execution.
The current source is SHA256
`f07e939bd612e813a2055736bd5c1c90eafbaafcb77d6447c27294391be7aca4`.
Earlier performance/model tables retain their frozen pre-fix source identity.

## Correctness and regression coverage

Task-local CPU validation exercised tensor initialization and typed-view
aliasing with mocked NPU launches, including device-derived launch bounds and
the 1M metadata case. The standalone CPU test file is not shipped in this PR.
NPU tests retain the original PR cases and add ragged and padded
strides, int32/int64 inputs, duplicate mappings, empty histories, all 16 flag-byte
patterns in every nibble, both stages below/at/above the Vector tile count,
and the exact issue geometry with vocab151936. Chained penalty cases compare
FP32 logits with the original independent reference at `rtol=atol=1e-5`.

```bash
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py
```

The final A3 operator acceptance ran the current product test file through
normal repository conftests and real framework imports: 50 cases passed.
This includes six added cases for same-word updates across program boundaries
and long-to-short request-row reuse. The original A2/A3 test guard is restored;
A5 is explicitly outside this PR's verified scope. Early component checks used
standalone source binding and remain distinct from this final test.
Fresh model evidence is recorded separately below. A2 device execution and an
isolated bincount capture/replay test are outside the measured scope.

## Profiling protocol

Single-operator measurements use an external msprof capture and a task-local,
source-identified driver. It selects one declared shape, warms up the wrapper,
runs a fixed number of calls and checks final outputs against an independent
CPU reference. Timing comes from the exported device-task CSVs, rather than
Python wall-clock measurements. Both sources receive identical inputs, layouts,
workspace policy and generation history. The cases and retained sample counts
are specified below.

The issue cases use exactly 64 backing rows, repeated token7 and no output
history. Use a fresh process and an external timeout when reproducing an old
launch-limit failure; a failed launch is not a zero-latency sample.

### Recorded scope

Main is frozen at `5bd7714b5ef39315cc26867bc67826d42ac0760a`, and baseline2
is PR15270 at `ef792533de1d4fcd25e458eda701fa105ffafeb4`.
Matched captures use one Ascend910_9362 (40 Vector cores), CANN9.2.0,
PyTorch2.10.0, torch_npu2.10.0.post4 and Triton-Ascend3.2.2.
Inputs/mappings are int32 and outputs are contiguous for these measurements.
Ordinary cases have two inactive backing rows and descending active mappings;
the exact issue cases have 64 backing rows.
Each ordinary capture has three warmups and ten retained complete invocations,
with a post-capture exact output check. The table sums device-task durations
per invocation, including Torch initialization; it is not host-call or model
latency. Raw per-invocation data and first-to-last device-task intervals are
retained separately. Source ordering rotates across cases.

| Case | Main ms | PR15270 ms | New ms | Main/new | PR/new |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 0.096 | 0.151 | 0.076 | 1.27x | 1.99x |
| medium | 1.104 | 0.273 | 0.199 | 5.54x | 1.37x |
| long | 9.788 | 0.332 | 0.257 | 38.13x | 1.29x |
| batch | 9.592 | 1.125 | 0.524 | 18.31x | 2.15x |
| batch64 | 83.427 | 7.600 | 3.002 | 27.79x | 2.53x |
| repeat | 1.120 | 0.372 | 0.284 | 3.95x | 1.31x |
| same_word | 2.114 | 0.281 | 0.205 | 10.33x | 1.37x |
| output_only | 0.222 | 0.311 | 0.254 | 0.87x | 1.22x |
| vocab_tail | 2.213 | 0.636 | 0.335 | 6.60x | 1.90x |

The output-only control regressed in summed device work relative to main.
The original capture measured 221.84 us versus 254.38 us (+14.7%). An A/B/A
repeat with five warmups and 30 measured calls per capture measured
main225.54 / new247.17 / main221.13 us (+9.6% to +11.8%). All nine ordinary
cases improved against PR15270. These are scoped results; they do not establish
a universal improvement over main or model latency.

Cases: short1/32/32, medium1/4096/32, long1/32768/4096,
batch8/4096/32 and batch64/4096/256 (requests/prompt/output), vocab151936;
repeat1/4096/4096 repeats token31; same_word8/1023/2 uses vocab4097;
output_only2/0/1025 uses vocab97; vocab_tail4/2049/33 uses vocab151000.
No workload weighting is assumed; per-case regressions must be read alongside
the wins. A measured operator improvement does not establish model E2E gains.

| Issue case | Main ms | PR15270 ms | New ms | New samples |
| --- | ---: | ---: | ---: | ---: |
| issue_legal | 25859.443 | 2866.678 | 245.206 | 3 |
| issue_limit | launch failure | launch failure | 246.632 | 3 |
| issue_above | not run | not run | 251.149 | 3 |

Issue candidate and PR captures use one warmup and three retained samples;
the serialized main legal case uses one warmup and two samples with a longer
external timeout. The raw records identify any failed comparator launches.

## Extended legal input range

This range checks 37 nonempty token-history shapes with
`0 < prompt_len <= prefill_len <= token row length`, valid vocabulary IDs and
valid request mappings. Every main/PR15270/new result matched the independent
CPU reference exactly: 111 source/case checks and 1,110 measured invocations.
These are valid operator inputs; the issue's 1M history is a separate operator
launch boundary, beyond the selected model server's 40,960-token context.

The new implementation is faster than PR15270 on all 37 cases and faster than
main on 31. Six legal short-prompt, multi-request cases regress against main;
they remain in the table and have dedicated correctness tests. The source is
frozen at the approved implementation rather than claiming a universal win.

The text decoder admission path rejects `prompt_len == 0` with
`VLLMValidationError` in `vllm/v1/engine/input_processor.py::_validate_prompt_len`.
For the text-only Qwen3 model the multimodal skip flag is false. Thus the earlier
zero-prompt/output-only input is a diagnostic guard, outside the text admission
performance range. Its measured regression remains documented; guard correctness
is preserved. This exclusion does not remove any of the six legal regressions.

All captures use a separately idle Ascend910_9362 device, the same CANN/compiler
and runtime as the earlier scope, three warmups and ten retained calls per case.
Multiple cases share a profiler session; each measured group begins after that
case's own warmup anchors, so input construction and compilation are excluded.
Values sum the device tasks per wrapper invocation, including initialization.

For the `rR_pP_oO` rows, R=requests, P=prompt length, O=output length;
vocab151936, int32, random token IDs and contiguous outputs. Additional cases:
tail151000=4/2049/33/vocab151000; tail5121=3/1025/33/vocab5121;
tail33=2/31/33/vocab33 within one word; repeated7 and one_word=8/4096/256;
int64=4/1025/33/vocab151000 with int64 inputs/mappings;
strided=8/4096/256/vocab151936 with row padding.

| Case | Main ms | PR15270 ms | New ms | Main/new | PR/new |
| --- | ---: | ---: | ---: | ---: | ---: |
| r1_p1_o0 | 0.083 | 0.141 | 0.066 | 1.26x | 2.13x |
| r1_p8_o4 | 0.085 | 0.142 | 0.066 | 1.28x | 2.15x |
| r1_p32_o32 | 0.096 | 0.150 | 0.074 | 1.29x | 2.02x |
| r1_p128_o32 | 0.119 | 0.162 | 0.087 | 1.38x | 1.87x |
| r1_p1023_o2 | 0.345 | 0.270 | 0.195 | 1.77x | 1.39x |
| r1_p1024_o1 | 0.343 | 0.271 | 0.194 | 1.77x | 1.40x |
| r1_p1025_o33 | 0.347 | 0.269 | 0.196 | 1.77x | 1.37x |
| r1_p4096_o256 | 1.117 | 0.273 | 0.198 | 5.66x | 1.38x |
| r1_p16384_o256 | 4.218 | 0.277 | 0.199 | 21.19x | 1.39x |
| r1_p32768_o4096 | 9.522 | 0.332 | 0.256 | 37.25x | 1.30x |
| r8_p1_o0 | 0.194 | 0.978 | 0.373 | 0.52x | 2.62x |
| r8_p8_o4 | 0.209 | 0.980 | 0.375 | 0.56x | 2.62x |
| r8_p32_o32 | 0.262 | 0.988 | 0.383 | 0.68x | 2.58x |
| r8_p128_o32 | 0.457 | 1.000 | 0.396 | 1.15x | 2.52x |
| r8_p1023_o2 | 2.295 | 1.115 | 0.509 | 4.51x | 2.19x |
| r8_p1024_o1 | 2.315 | 1.114 | 0.509 | 4.55x | 2.19x |
| r8_p1025_o33 | 2.308 | 1.116 | 0.509 | 4.53x | 2.19x |
| r8_p4096_o256 | 9.247 | 1.124 | 0.522 | 17.70x | 2.15x |
| r8_p16384_o256 | 39.896 | 1.506 | 0.923 | 43.22x | 1.63x |
| r8_p32768_o4096 | 87.592 | 2.114 | 1.630 | 53.73x | 1.30x |
| r64_p1_o0 | 0.491 | 6.607 | 2.015 | 0.24x | 3.28x |
| r64_p8_o4 | 0.623 | 6.613 | 2.014 | 0.31x | 3.28x |
| r64_p32_o32 | 1.100 | 6.626 | 2.031 | 0.54x | 3.26x |
| r64_p128_o32 | 2.969 | 6.654 | 2.059 | 1.44x | 3.23x |
| r64_p1023_o2 | 21.132 | 6.879 | 2.279 | 9.27x | 3.02x |
| r64_p1024_o1 | 21.208 | 6.876 | 2.280 | 9.30x | 3.02x |
| r64_p1025_o33 | 21.194 | 6.892 | 2.295 | 9.24x | 3.00x |
| r64_p4096_o256 | 84.869 | 7.602 | 3.002 | 28.27x | 2.53x |
| r64_p16384_o256 | 341.879 | 10.111 | 5.642 | 60.60x | 1.79x |
| r64_p32768_o4096 | 685.319 | 14.675 | 10.281 | 66.66x | 1.43x |
| tail151000 | 2.225 | 0.635 | 0.334 | 6.66x | 1.90x |
| tail5121 | 0.824 | 0.264 | 0.182 | 4.52x | 1.45x |
| tail33 | 0.085 | 0.132 | 0.057 | 1.51x | 2.33x |
| repeated7 | 8.590 | 1.227 | 0.625 | 13.75x | 1.96x |
| one_word | 8.679 | 1.228 | 0.608 | 14.27x | 2.02x |
| int64 | 1.231 | 0.633 | 0.332 | 3.71x | 1.91x |
| strided | 9.579 | 1.152 | 0.548 | 17.47x | 2.10x |

## Fresh MRV2 model validation

These are fresh Qwen3-8B BF16, TP1 runs on the same Ascend910_9362 device,
with Model Runner V2 confirmed in server logs. Serving used the default
`FULL_AND_PIECEWISE` graph configuration; both capture phases completed.
Max context40960, max sequences32, batched tokens4096, memory utilization0.85
and prefix caching disabled were identical. The 1M issue test remains a
standalone launch-boundary test, beyond this model's configured context.

Only the penalty module changes between the three runs. Common Python sources
are upstream vLLM `544f182d9f0c9bfe8bd7645525951d5e3692a542` and this PR's
plugin tree `c0e0e05b94830034e40270384002d5ba5a416821`; later commits change
tests/docs and remove the benchmark. Native dependencies are the same image
artifacts for all three, rather than a new native build of the PR:

- Image: `sha256:0c85b29e600f291bbe920c7a50a667c548bd4826fd79795283d66ea97417fac2`.
- Native plugin revision: `b64b4d714484feaa6ca71edc99b98318fe6d2f0d`.
- CANN9.2.0, Torch2.10.0, torch_npu2.10.0.post4, Triton-Ascend3.2.2.
- Frozen weight-file and native-library hashes are retained in the task evidence.

Generation uses temperature0, top_p1, top_k-1, explicit request seed42,
`repetition_penalty=1.03` and `enable_thinking=False`. Each source has a fresh
server and a separate bounded profiler request before unprofiled workloads.
Its 24-token prompt invokes the main bincount kernel once, or the original/new
statistics and packing kernels once each. Source hashes, responses, server
logs and kernel CSVs establish the enabled penalty path. Those short profiles
are dispatch evidence, not sustained performance measurements.

### GPQA Diamond accuracy

The unchanged official AISBench zero-shot direct-answer template evaluates
all198 Diamond items. AISBench source provenance is
`7bcb6440da11bef020943b2af9ed7c1ac8362306`; the dataset CSV SHA256 is
`41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305`.
Concurrency16 and max output4096 are identical across sources. A preliminary
512-token pilot was rejected because it truncated answers; its records remain
separate. The full runs retain every item, including length finishes, in the
official evaluator. The scores below apply to this finite decoding budget.

| Implementation | Correct / total | Accuracy | Failed requests | Length finishes |
| --- | ---: | ---: | ---: | ---: |
| Main | 97/198 | 48.99% | 0 | 9 |
| PR15270 | 93/198 | 46.97% | 0 | 12 |
| New | 93/198 | 46.97% | 0 | 12 |

Every prediction JSONL and captured response was parsed using its LF record
delimiter. There are198 successful predictions per run plus a separately
identified client warmup. Paired comparisons use identical prompts and gold
answers for all198 IDs:

| Compared with main | Raw text matches | Evaluated answer matches | Correctness matches |
| --- | ---: | ---: | ---: |
| PR15270 | 25/198 | 145/198 | 170/198 |
| New | 25/198 | 145/198 | 170/198 |

Concurrent outputs are not identical, so these runs do not establish exact
model-output equivalence or by themselves identify the cause of the difference.
New and PR15270 have identical normalized predictions, token counts and scores
for all 198 items. Per-item differences from main remain in the raw evidence.
The additional control below
uses one request at a time to evaluate precision under a stable batch size.

### Serial GPQA final-answer control

This separate full198-item run adds only the explicit instruction:
"Give only the final answer in the required format. Do not include any reasoning
or explanation." Dataset questions, choices and the official evaluator remain
unchanged. Concurrency1, max output512, request seed42 and all model/penalty
settings are the same across sources. This is a final-answer prompt variant;
its score must not be mixed with the concurrent official-template score above.

| Implementation | Correct / total | Accuracy | Failures | Length finishes | Raw text matches vs main | Evaluated answer matches vs main |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 73/198 | 36.87% | 0 | 0 | 198/198 | 198/198 |
| PR15270 | 73/198 | 36.87% | 0 | 0 | 198/198 | 198/198 |
| New | 73/198 | 36.87% | 0 | 0 | 198/198 | 198/198 |

### GSM8K sustained serving performance

The full1,319-item test split uses the official zero-shot CoT performance
template; dataset SHA256 is
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.
Streaming concurrency32, 32 warmups and fixed128-token outputs
(`ignore_eos=True`) are identical. Runs are sequential main/PR15270/new with
two repeats each on the same device. Each of the six runs completed1,319
requests with zero failures and168,832 generated tokens. GSM8K is the serving
performance workload here; GPQA supplies the model accuracy result.

Latency values below come from the official AISBench CSV, while throughput
comes from its common-metric JSON. This is unprofiled end-to-end serving.

| Implementation / repeat | Duration s | Output token/s | Requests/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | Mean E2EL ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main / 1 | 116.734 | 1446.3 | 11.2992 | 158.7 | 218.8 | 20.7 | 2788.7 |
| Main / 2 | 116.421 | 1450.2 | 11.3296 | 154.8 | 168.2 | 20.7 | 2780.7 |
| PR15270 / 1 | 114.668 | 1472.4 | 11.5028 | 162.6 | 194.7 | 20.3 | 2738.6 |
| PR15270 / 2 | 114.999 | 1468.1 | 11.4696 | 162.4 | 182.2 | 20.3 | 2745.6 |
| New / 1 | 115.783 | 1458.2 | 11.3920 | 175.7 | 341.5 | 20.4 | 2762.2 |
| New / 2 | 114.311 | 1476.9 | 11.5387 | 164.1 | 208.7 | 20.2 | 2729.8 |

Mean output throughput is main 1448.2, PR15270 1470.2 and new 1467.6 token/s.
The new mean differs from main by +1.33% and from PR15270 by -0.18%.
These are two sequential repeats on one device; small differences do not
establish a broad model-level speedup. Mean TTFT is worse for the new version
in these runs: main 156.75, PR15270 162.50 and new 169.90 ms. The new mean is
13.15 ms above main and 7.40 ms above PR15270. No reverse-order confirmation
was performed, and these aggregate differences cannot be attributed solely to
bincount. The operator is initialized when new
penalty requests enter, so its microbenchmark gains need not dominate decode.

### Validation boundaries

The model runs above used the frozen pre-compatibility operator SHA256
`83b871bc1f77319da7e6d9733c7393606a5ef56a24aabd68aa052baf837a01b3`.
All 50 product NPU tests passed through normal repository conftests and real
framework imports in this model runtime. The accepted compiler is the
project-pinned Triton-Ascend 3.2.2.
A separate3.6 image probe rejects annotated constexpr globals and is not a
passed environment; no permissive compiler override or kernel rewrite was
used. Native API-loading failures during preparation and the discarded short
GPQA pilot remain separate diagnostic evidence.

## Invocation frequency and captured model share

The fixed upstream `PenaltiesState.apply_staged_writes` initializes the
nonempty list of newly admitted penalty requests, then clears that list.
Subsequent decode steps maintain counts incrementally. A preempted/resumed
request can require another initialization episode; disabled penalties skip
this path. R is the newly staged count, not the total active decode batch.

Two fresh Qwen3-8B captures on the same frozen operator confirm the behavior:

| Window | Generated tokens | Logical bincount calls | R per call | Max prefill per call | Penalty-application launches |
| --- | ---: | ---: | --- | --- | ---: |
| One request | 64 | 1 | 1 | 24 | 64 |
| Eight concurrent GSM8K questions | 512 | 2 | 1,7 | 38,123 | 65 |

Logical bincount calls each launch one statistics and one pack kernel. The
penalty-application kernel is a separate operator. The full 1,319-request
GSM8K performance runs were not instrumented for call counts, so no exact
total bincount count is inferred for those runs.

The following denominator sums every device task duration in its capture.
The inclusive wrapper scope includes Torch initialization and was associated
with its device tasks by the profiler; host duration is not used as device
time. These are device-work shares, not end-to-end critical-path fractions:

| Window | Statistics + pack us | Wrapper inclusive device work us | All captured device work us | Wrapper share |
| --- | ---: | ---: | ---: | ---: |
| One request / 64 output | 43.60 | 79.90 | 956064.42 | 0.00836% |
| Eight requests / 512 output | 282.96 | 402.34 | 1101596.32 | 0.03652% |

Only existing host metadata was read by the diagnostic wrapper: no tensor
copies, `.item()`, `.cpu()` or stream synchronization were added. Its CPU
recording still makes profiled request wall time unsuitable as the sustained
benchmark result. The earlier GSM8K runs were uninstrumented. These captures
show why single-operator wins do not imply comparable model-throughput gains;
they do not establish a universal model share or a bound on wall-time gains.

## Memory-ordering scope

Prompt flags use non-atomic byte stores and duplicate token IDs can target the
same byte. The intended result relies on identical writes of one, followed by
packing after the statistics kernel. Passing the stated device tests does not
prove a general race-free guarantee for all backends. Output counts retain
atomic addition. Unique-writer packed alternatives are evaluated separately;
experimental results are not silently substituted for this frozen source.

## Compilation variants and startup warmup

The statistics and packing kernels use fixed constexpr tiles, with request
counts, work counts and strides supplied as runtime integers. On the pinned
Triton-Ascend 3.2.2 runtime, ordinary positive integers still specialize into
exact-one, generic and divisible-by-16 classes. Pointer dtype/alignment and
the compilation target are additional dimensions. A new concrete shape does
not necessarily compile another binary.

An isolated, compile-only probe with task-owned cold and persistent caches
observed the following on the frozen operator:

| Change | Observed binary behavior |
| --- | --- |
| Request count 2 to 8 | Reused |
| Request count 16 to 32 | Reused |
| Token blocks 2 to 3, or 16 to 32 to 1024 | Reused within the respective class |
| Grid 1 to 40 with unchanged arguments | Reused |
| Exact-one/divisibility class, pointer dtype or alignment | Additional variants |

Fifteen controlled cases produced eight statistics and five packing binaries.
These include dtype/alignment controls rather than the complete production
signature set. The first cold statistics call took 9.153 s including toolchain
discovery; later new statistics variants took 0.516–0.626 s and packing variants
1.282–1.349 s. Same-process hits took about 0.04–0.16 ms. A fresh process with
the persistent cache ran no actual kernel compilations: the first statistics
lookup including discovery took 0.218 s, and later first-per-key lookups about
1.6–3.4 ms. These are host compile/cache API durations, not device execution
times or full-model startup measurements. Compiler help queries are excluded
from the count of 13 actual cold compilations.

The pinned MRV2 caller creates int64 admission mappings with int32 token
storage. The standalone probe's primary signature used int32 for both, so its
hashes do not establish production warmup coverage. Ordinary dummy decode also
does not exercise the admission-only path. A compile-only adapter for the
existing warmup registry has been prepared separately: the configured
32-request/40960-context worker has nine reachable statistics keys and three
packing keys, using representatives 1, 2 and 16. Actual JIT hash equivalence,
real-dependency integration and the 12-key startup cost remain unvalidated.
That adapter is not part of this tested revision; lazy compilation remains.

## Packed-workspace experiments

Two alternatives were tested separately without changing the approved operator:

- **Private packed partials:** each request/part has one writer for its packed
  GM row. That writer updates prompt words sequentially; a final unique writer
  ORs the partials. Prompt updates need no atomic OR under this ownership.
  Output counts retain atomic adds. The tested implementation uses 256-token
  tiles and at most `max(1, Vector cores // R)` parts per request.
- **Vocabulary ownership:** each task owns disjoint 128-word output ranges,
  scans the prompt in 1024-token tiles and masks IDs outside its range. A local
  histogram builds presence before packing; only one task per request counts
  output history. It allocates no explicit prompt workspace in GM, but that
  alone does not establish the backend's complete working set.

On a separately idle A3 device with the same runtime, each source/shape passed
an exact reference check, then retained ten measured wrapper calls after three
warmups. Values below sum device-task work including initialization, in us.
R/P/O denote requests/prompt/output; vocabulary is 151936 except the last row.
The source order and all raw per-call results are retained in task evidence.

| R / P / O | Main | PR15270 | Approved | Private packed | Vocabulary owner |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 / 0 | 83.09 | 141.15 | 65.51 | 34.46 | 201.54 |
| 1 / 32 / 0 | 89.93 | 145.07 | 68.98 | 41.46 | 201.62 |
| 1 / 128 / 0 | 113.62 | 156.04 | 80.79 | 62.78 | 201.47 |
| 8 / 1 / 0 | 193.19 | 979.14 | 374.59 | 135.17 | 1489.19 |
| 8 / 32 / 0 | 256.02 | 982.83 | 378.17 | 142.53 | 1489.04 |
| 8 / 128 / 0 | 449.23 | 994.83 | 389.88 | 165.79 | 1489.75 |
| 64 / 1 / 0 | 488.36 | 6610.57 | 2016.27 | 182.50 | 10542.28 |
| 64 / 32 / 0 | 1121.48 | 6615.70 | 2024.48 | 192.90 | 10545.88 |
| 64 / 128 / 0 | 2947.84 | 6639.92 | 2047.92 | 242.23 | 10543.09 |
| 8 / 4096 / 32 | 9529.01 | 1127.22 | 520.74 | 408.72 | 3576.06 |
| 64 / 4096 / 256 | 84767.77 | 7599.52 | 3007.06 | 2261.46 | 26530.66 |
| 1 / 32768 / 4096 | 9751.16 | 331.60 | 255.96 | 298.10 | 3641.76 |
| 3 / 1025 / 33, V=5121 | 882.56 | 259.33 | 185.91 | 89.91 | 279.53 |

Private packed partials reduce device work on 12/13 measured shapes, but the
single-request long-history case regresses by 42.14 us (16.5%). If S is the
number of partials per request, its GM prompt workspace is S/8 of the current
int8 flags, not S/32. With 40 Vectors and R=1, the long case uses S=40:
759680 bytes versus 151936 bytes, or five times the workspace. With R=64,
S=1 and the workspace is 1215488 versus 9723904 bytes, or one eighth.
Thus private packing is not an unconditional memory reduction.

The private-packed variant also passed the existing standalone cases including
int64, strides, duplicate mapping and all three exact #19958 boundaries.
Its 1M performance and model integration were not measured. The histogram
vocabulary-owner variant lost on all 13 comparison shapes; this result applies
to that realization, not every possible vocabulary-owner algorithm. An earlier
sort/scan partial-packing trial hit a vector-core runtime exception and was
excluded from the timing table. None of these trials is promoted into this PR;
the approved operator and its model evidence remain unchanged.
