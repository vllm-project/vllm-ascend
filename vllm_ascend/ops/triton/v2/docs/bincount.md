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

## Reproducible performance cases

`benchmarks/benchmark_bincount.py` selects shape, distribution and optional
frozen source from the command line. It checks final outputs against a CPU
reference and reports source hashes and call counts. It does not use Python
wall-clock timing. `profile-all` warms up and runs a declared number of calls;
`profile-op` invokes the wrapper once for externally selected kernel profiling.

```bash
msprof --output=./bincount-profile \
  --application="python benchmarks/benchmark_bincount.py --requests 64 --prompt 4096 --output 256 --warmup-calls 3 --measured-calls 10"
```

The issue's legal stress case is selected with `--requests 64 --state-rows 64
--prompt 1047552 --output 0 --pattern repeat`; the next boundaries use 1048576
and 1049600. Use `--token 31` for the smaller repeated-token control.
Use a fresh process and an external timeout for old launch-limit failures.
Do not treat failed launches as zero latency.

### Recorded scope

Main is frozen at `5bd7714b5ef39315cc26867bc67826d42ac0760a`, and baseline2
is PR15270 at `ef792533de1d4fcd25e458eda701fa105ffafeb4`.
Matched captures use one Ascend910_9362 (40 Vector cores), CANN9.2.0,
PyTorch2.10.0, torch_npu2.10.0.post4 and Triton-Ascend3.2.2.
Inputs/mappings are int32 and outputs are contiguous for these measurements.
Ordinary cases have two inactive backing rows and descending active mappings;
the exact issue cases have 64 backing rows. The benchmark supports both.
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
