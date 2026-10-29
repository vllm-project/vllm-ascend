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
produce variants; this is a static review, not a measured cold-cache audit.

## Correctness and regression coverage

CPU wrapper tests exercise real tensor initialization and typed-view aliasing
with mocked NPU launches, including device-derived launch bounds and the 1M
metadata case. NPU tests retain the original PR cases and add ragged and padded
strides, int32/int64 inputs, duplicate mappings, empty histories, all 16 flag-byte
patterns in every nibble, both stages below/at/above the Vector tile count,
and the exact issue geometry with vocab151936. Chained penalty cases compare
FP32 logits with the original independent reference at `rtol=atol=1e-5`.

```bash
python -B -m pytest -q -p no:cacheprovider --confcutdir=tests/ut/worker/v2 \
  tests/ut/worker/v2/test_bincount.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py
```

The task executes the unchanged product NPU test file with explicit source
binding to the exact operator and device-helper snapshots. This proves the
operator component in the frozen runtime, not full-framework/model dispatch.
A2, model E2E, graph execution and empirical recompilation counts are outside
the measured scope.

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
