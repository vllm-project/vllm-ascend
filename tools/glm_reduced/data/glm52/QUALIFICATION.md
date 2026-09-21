# GLM-5.2 preliminary qualification, 2026-09-21

This case is **not qualified as a nightly regression baseline**. No correctness or performance
baseline has been accepted. The aarch64 target runner is pending resource
coordination; x86 measurements must not be copied into its baseline.

## Real-weight and serving validation

The x86 A3 preliminary run used the pinned source in `source.json`, eight
visible Ascend910_9382 devices, TP8/DP1/EP, MRV1, retained MTP with three
draft tokens, FULL_DECODE_ONLY, and disabled prefix caching.

- Verified checkpoint identity:
  `eafe6ee02203907ef160fdfe19c92b748a340471d68856d07e67f649c06132b5`.
- 42,086,049,280 tensor bytes and 18,723 tensors; decoder layers 0–7;
  original MTP layer 78 mapped to layer 8.
- Production hidden size 6144, 256 routed experts, and index top-k 2048 retained.
- All eight workers loaded the main model and MTP weights, about 5.4128 GiB
  per worker. Logs confirmed that MTP drafted tokens during requests.
- The isolated runtime used vLLM commit
  `84030bbe3d74d99bad477a3d2e37a973ccd8865c`, Ascend base
  `cb79183b7c04467c7358fe568ee88e85603ee965`, CANN 9.1.0 and driver 26.1.1.
  Native libraries were reused from an existing compatible x86 build;
  this is not the target nightly image qualification.

## Failures that prevent a baseline

All eight fixed requests returned complete 64-token responses, but each
request produced five different normalized texts in the same server start.
Temperature was zero, seed 1024, and input IDs were identical. This repeatability
failure blocks output-regression calibration; it is not a semantic accuracy
evaluation of full GLM.

Both streaming workloads completed one warmup and five measured rounds.
Every request count and actual input/output length passed validation.
The following medians are observations from **one** server start, not baselines:

| Workload | Output tokens/s | Mean TTFT, ms | Mean TPOT, ms | Relative MAD: throughput / TTFT / TPOT |
| --- | ---: | ---: | ---: | --- |
| 128/128, 32 requests, concurrency 1 | 48.9184 | 42.4516 | 20.2666 | 0.705% / 1.483% / 0.688% |
| 4096/128, 64 requests, concurrency 8 | 249.6004 | 708.7483 | 26.6229 | 0.690% / 5.617% / 2.072% |

Long-input TTFT dispersion exceeded the 3% requirement and the calibration
command exited with status 1. No subsequent calibration starts or baseline
publication occurred. Output instability independently disqualifies the run.
The gate now reports within-start output instability before deciding whether
to continue to another server start, while still collecting both workloads.

Two subsequent bounded diagnostics enabled deterministic communication settings.
With FULL_DECODE_ONLY, each of the five short prompts became stable over five
repeats, while each 2047/2048/2049-token prompt still produced two distinct texts.
An eager-mode diagnostic had the same unique-output counts. Both retained MTP.
These observations do not establish graph execution as the cause, and neither
diagnostic produced a baseline.

The user subsequently requested a separate full-model prefix numerical
comparison, documented in [PREFIX_PROBE.md](../../PREFIX_PROBE.md). That work
compares hidden states and complete logits using identical fixed token histories;
it does not treat semantic answers from an untrained cropped model as accuracy.

The full source's 95 main shards and associated JSON/safetensors files (104
files total) matched the pinned provider hashes. The full reference executed
78 layers and the candidate eight, at TP8/EP, eager, PP1, block size 128, no MTP,
no prefix cache, no chunked prefill, and serial fixed-continuation requests.
The reference runtime was explicitly selected by the user; no separate full
semantic accuracy report was supplied.

In the first numerical comparison, all paired prefix hidden states, normalized
states and logits matched exactly. An extra candidate self-check differed by
up to 0.0625 for the three long prefill inputs when selecting a row before
normalization. Work stopped and the user approved one additional comparison
with full-batch normalization before row selection and an actual-norm capture.

That approved comparison matched **exactly in all five categories**: hidden
states, normalized states, complete vocabulary logits, candidate normal-norm
self-check, and candidate normal-final-logits self-check. Each category contains
256 rank observations (eight requests, four positions, eight TP ranks), covering
32 request positions, not 256 independent experiments. Maximum absolute and RMS
errors were zero. See `prefix-observation-lab-168.json` for the measured summary.
This is prefix numerical parity in the tested eager configuration, not MTP,
graph-mode or full-model semantic qualification, and not an accepted nightly
baseline. No tolerance was relaxed or failing input removed.

## CPU and integration checks

Linux: 165 CPU tests passed with no skips, including the new prefix probe.
Windows: 162 passed and three skipped (two symlink tests and the optional
Torch-backed worker probe, which passed on Linux).
The complete `format.sh ci` passed on Linux. Failure injection exercised
changed expected text, out-of-range performance, summary output, and exception
propagation; normal simulated candidates passed.

Raw HTTP responses, benchmark JSON, startup failures, configuration identities,
logs, native library hashes, and cleanup evidence are retained with the task's
validation report. Formal qualification still requires stable output and
three independent starts with five rounds each on the actual aarch64 runner.
