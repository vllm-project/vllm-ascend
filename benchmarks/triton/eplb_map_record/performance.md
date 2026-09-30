# EPLB map and valid-token record: validation and performance

## Scope and evidence boundary

The unchanged Router/CANN path produces logical TopK IDs and weights. This
patch replaces the EPLB mapping step with two Triton kernels: map IDs and
write grid-private valid-token physical-expert counts, then reduce those
counts into cumulative load. No gating or TopK time is credited to this patch.

Measurements below use one isolated Ascend 910B4-1 host, CANN 9.2,
torch 2.10.0+cpu, torch-npu 2.10.0.post4, Triton-Ascend 3.2.0, and
vLLM `ced6857afa0ea7b2e3f0846a62e1394e90f15607`. The baseline is branch
HEAD `f8c000227e5cb2fa69a435c8579be4907c1ae03d`; the candidate is the
uncommitted refactor of that HEAD, with exact source hashes recorded in the
task-owned validation directory. Both import their intended source tree via
`PYTHONPATH`, use the same read-only four-layer Kimi K3 W4A8 weights and
tokenizer, and run V2/TP8/EP8/DP1, ALLGATHER, eager mode, EPLB enabled
without redundant experts, and prefix caching off. The full 93-layer model
was not run; this experiment establishes call-path and limited serving
behavior, not full-model accuracy or throughput.

An initial candidate A/B/A preceded the final sequence-parallel valid-prefix
fix and is exploratory only: the then-current mapping path counted gathered
padding rows. A subsequent short profile on the corrected source confirms
that all eight ranks execute the new kernels. The final-source unprofiled
candidate repeat is reported separately below.

## Actual model call and short profile

After loading and two warmup requests, one prompt of 127 tokens generating
four tokens was enclosed by `/start_profile` and `/stop_profile`. Offline
`torch_npu.profiler.profiler.analyse` produced `operator_details.csv` and
`kernel_details.csv` for all eight ranks. Every rank has three prefill and
nine decode calls in both versions; the model's prepared rows have
`logical_ids` shapes `[128,16]` and `[8,16]` respectively. The candidate
actually executes both new kernels. Rank 0 details (device kernel time):

| Stage | Implementation/kernel | Calls | Input shape | Median/call (µs) | Sum (µs) |
|---|---|---:|---|---:|---:|
| Prefill | Both: CANN `MoeGatingTopK` | 3 | `[128,896]` | 17.38 / 16.98 | 54.56 / 51.50 |
| Prefill | Baseline map | 3 | IDs `[128,16]`, table `[1024,896]` | 57.40 | 171.78 |
| Prefill | Baseline downstream record | 3 | local count `112` | 4.90 | 14.58 |
| Prefill | Candidate grid map/record | 3 | IDs `[128,16]`, grid `[32,112]` | 23.26 | 69.42 |
| Prefill | Candidate reduce | 3 | grid `[32,112]` | 2.28 | 6.66 |
| Decode | Both: CANN `MoeGatingTopK` | 9 | `[8,896]` | 6.02 / 6.16 | 56.68 / 57.82 |
| Decode | Baseline map | 9 | IDs `[8,16]`, table `[1024,896]` | 35.44 | 319.96 |
| Decode | Baseline downstream record | 9 | local count `112` | 4.72 | 42.62 |
| Decode | Candidate grid map/record | 9 | IDs `[8,16]`, grid `[2,112]` | 17.96 | 160.62 |
| Decode | Candidate reduce | 9 | grid `[2,112]` | 1.78 | 15.98 |

The median of eight per-rank medians is 62.13 vs 25.01 µs for prefill and
40.19 vs 19.57 µs for decode when only target component medians are added.
These diagnostic sums are **not** request wall-clock times: baseline records
after MoE, whereas candidate records at mapping; kernels may overlap other
work. Call counts and input shapes match across versions.

## Independent `msprof op` on the model-observed shapes

One `msprof op` invocation per kernel/shape used device 0, internal warmup 5,
and 20 measured launches. Values are median Task Duration in microseconds;
the range across launches is in parentheses. Task Duration includes NPU task
head overhead. The baseline record receives a matching synthetic local count
vector; it is a component-cost comparison, not a contiguous model replay.

| Model shape | Baseline map | Baseline record | Grid map/record | Grid reduce | Baseline sum | Candidate sum |
|---|---:|---:|---:|---:|---:|---:|
| Decode: T=8, E=896, K=16, local=112 | 30.36 (30.06–31.74) | 2.06 (2.02–2.42) | 15.15 (14.80–16.56) | 1.96 (1.90–2.36) | 32.42 | 17.11 |
| Prefill: T=128, E=896, K=16, local=112 | 47.26 (46.70–48.38) | 2.01 (1.78–2.32) | 16.67 (16.34–17.22) | 2.17 (2.04–2.54) | 49.27 | 18.84 |

`PipeUtilization.csv` for the slowest vector program in each launch shows
that prefill map scalar time falls from 46.04 to 14.64 µs (median), while
vector time rises from 0.40 to 1.35 µs; MTE2 remains about 0.25 vs
0.20 µs. Decode map scalar time falls from 28.98 to 13.11 µs. These
components overlap and must not be summed. The old map uses a 256-assignment
program tile (one decode or eight prefill programs); the new 64-assignment
tile has two decode or 32 prefill programs. More independent programs and
shorter per-program scalar work plausibly explain the kernel reduction;
program IDs are not physical core IDs. The extra vector comparison work is
visible but is not the dominant cost. All 26 cached TTIR files for the two
new kernels contain no atomic operation.

## Unprofiled four-layer serving

The same container, weights, device set, launch arguments and requests were
used for baseline A, baseline B, then the corrected final-source candidate.
The initial candidate block between A and B is excluded because it preceded
the sequence-parallel padding fix. Each launch warmed a short and a long
request before measurement. Streaming curl used temperature 0, seed 0,
output length 128, concurrency 1, and input lengths 100/1097 tokens.
Each cell below is the per-block median; repeats were 3/2/5 respectively.

| Request | Block | TTFT (ms) | E2E (ms) | TPOT (ms) |
|---|---|---:|---:|---:|
| Short | Baseline A | 98.26 | 8034.33 | 62.48 |
| Short | Baseline B | 91.89 | 7908.39 | 61.53 |
| Short | Final candidate | 97.19 | 7836.81 | 60.93 |
| Long | Baseline A | 161.55 | 7829.87 | 60.39 |
| Long | Baseline B | 162.00 | 7933.07 | 61.17 |
| Long | Final candidate | 168.26 | 7679.10 | 59.08 |

AIS Bench used the same first 32 GSM8K prompts (input 114–195 tokens),
concurrency 4, output length 128, temperature 0, seed 0 and ignore-EOS.
The first dataset pass after a fresh source/cache was affected by compilation
(baseline A: 39.56; final candidate: 50.01 output tokens/s) and is treated
as warmup, not steady-state evidence. Later 32-request passes were:

| Block/pass | Output throughput (tokens/s) | Median E2E (ms) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) |
|---|---:|---:|---:|---:|---:|
| Baseline A / 2 | 62.17 | 8188.2 | 217.0 | 62.8 | 62.6 |
| Baseline B / 1 (warm cache) | 65.08 | 7810.6 | 237.0 | 59.6 | 59.2 |
| Final candidate / 2 | 64.36 | 7927.0 | 210.6 | 60.9 | 60.7 |

The candidate sits between the two steady baseline throughput runs. Curl
E2E medians are lower, but TTFT is not consistently lower, and the baseline
run-to-run spread is larger than the apparent GSM8K difference. **No stable
model-level speedup is established.** The truncated model's generated text
differs even across repeated requests within one version, so exact text
parity is not claimed;
the exact operator mapping/record tests are the correctness evidence.

## Correctness and retained artifacts

Final-source CPU integration UT: 39 passed. Final-source NPU map/record UT:
25 passed, covering exact mainline mapping, padding exclusion (including
the model-observed T=8/valid=1 and T=128/valid=127), nonzero load,
record-off, local physical range, E=128/896, K=16, noncontiguous IDs and
runtime table/flag/valid-count graph replay. Three common-domain comparisons
with PR #17574's atomic version matched physical IDs and expert loads exactly.

Raw short-profile CSVs, `msprof op` summaries and `PipeUtilization.csv`,
unprofiled request JSON/CSV, source hashes and test logs are preserved under
the task-owned blue-zone path
`/home/shy/moe-gating-topk-map-record/eplb-map-record-refactor-20260930`.
The full Kimi K3 model and graph-mode serving remain unverified.
