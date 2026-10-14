# GLM-5.3-Flash MTP graph metadata and padding

Captured draft operators still depend on host input/metadata preparation and
earlier autoregressive steps. A late graph submission on one EP rank appears
as Dispatch waiting on the other ranks.

## Changes and scope

- Preserve the first verify-sized draft forward; use request-sized later-step
  capacity `min(max_num_seqs, max(verify_bucket // (K + 1), 1))` and MC2 padding.
- Warm persistent per-step buffers before capture and capture later-step
  metadata updates with the model forward.
- Carry device active-request counts through query boundaries, sequence
  lengths, slots and current main's persistent NoPE valid-row mask. Preserve
  newer main's forward-time NoPE planning.
- Read persistent block tables and preallocate secondary-group tables for
  FULL buckets exceeding `max_num_seqs`.
- Replace GDN padding's synchronous CPU-boundary upload with device `clamp`.
  Its scalar threshold comes from the CPU mirror; shared MLA metadata remains
  unchanged. No device `.item()` or cross-rank barrier is added.
- Add profiler ranges for first-pass inputs, first-pass metadata and graph execution.

The smaller-step path is limited to GLM multi-KV-group MTP, A5 NoPE, TP=1,
PCP=DCP=1, uniform decode, FULL draft graphs and dynamic MC2 `global_bs`.
CP/SP, offload, M-RoPE, compressed attention, parallel drafting and uniform
MC2 keep their previous path. Recapture graphs after changing this code.

## Evidence from the tested integration branch

The A5 experiment used MTP=5, FP8 main/indexer caches, 32 requests with
262156-262158 input tokens and fixed DP8 counts `[5,2,2,4,2,5,7,5]`.
Prefill and KV reception completed before capture. Prompt counters did not
increase during capture, waiting was zero and no preemption occurred.

Padded ranks 0, 5 and 7 used to synchronously upload corrected query boundaries.
The tiny upload waited for queued device work while peers advanced to the next
target graph. Eager had the same roughly 4 ms first-layer skew, so the entire
delay cannot be attributed to enabling MTP graphs.

Align 42 target and 5 draft Dispatches per iteration across eight ranks and
exclude only the first two and final capture-boundary iterations. Arrival
spread is the difference between latest/earliest same-slot starts; iteration
period uses consecutive latest final-draft Dispatch completion times. These
are profiler metrics, not token TPOT or no-profiler throughput.

| Case | Steady iterations | Target first arrival P50/P99 (ms) | Iteration period P50/P99 (ms) |
| --- | ---: | ---: | ---: |
| Graph before padding fix, 5 s | 101 | 4.067 / 4.699 | 52.120 / 52.743 |
| Eager, 5 s | 100 | 4.015 / 7.913 | 52.222 / 56.034 |
| Graph after padding fix, 5 s | 108 | 0.639 / 1.141 | 48.472 / 49.150 |
| Graph after padding fix, 20 s | 411 | 0.671 / 1.794 | 48.697 / 49.725 |

The matched 5-second graph comparison reduces median arrival skew by about
84% and median iteration period by about 7%. Each 5-second case generated
1024 tokens per request; all 32 stability requests generated 4096. Standalone
A5 GDN padding validation passed 5 cases; the old upload fails a CPU negative control.

These measurements are from integration revision
`8d5969cc00db06a8cbf3e654c2358ef440435c82`. They support the diagnosis but are
not new measurements of this main-based PR. The extracted branch preserves
newer NoPE planning and still needs its own NPU/model acceptance.

## Remaining limits

Different request counts produce different first-layer work. The 20-second
trace's worst arrival skew was 3.098 ms: rank 7 submitted its target graph late
and ran Dispatch for about 23 us while rank 2 waited about 3.108 ms. That window
no longer contains the old 3-4 ms padding upload. Host scheduling, preparation
and profiler overhead still need to be distinguished.

The original 416.8 ms extreme tail did not recur in this finite run; this does
not prove elimination under every workload. Eager-worker shutdown malloc
corruption/SIGABRT remains unresolved. Output/acceptance, lifetime stability
and formal no-profiler throughput are separate acceptance items.
