# GLM5.x 11-layer performance gate fixture

Offline single-round performance gate for the real GLM-5.2 W4A8 truncated
checkpoint (11 main layers, one retained MTP layer, MTP disabled), measured with
the same pinned runtime as the logits gate (vLLM 84030bbe, vllm-ascend cb79183).

## Comparison contract

- Compares the SAME checkpoint: a fresh candidate run against this baseline.
- Identity that must match exactly: model, hardware tag, engine settings
  (minus the checkpoint path), workload shapes, warmup and measured iteration counts.
- The runner architecture must also match the hardware tag, so a mismatched host
  fails closed instead of comparing across hardware.
- Statistics: the median of the measured iterations for each workload.
- Thresholds are supplied by the caller per hardware. The registered case uses
  output throughput >= 90% and TTFT/TPOT/whole-workload latency <= 120%, derived
  from the A/A calibration below, not chosen to make a candidate pass.
- Fail closed: a malformed record, an identity mismatch, or a missing threshold is
  a hard failure, never a skip. CI never writes or bootstraps a baseline.

## Baseline calibration (2026-09-22)

The baseline is the median of three independent launches on the same host, each
with one warmup plus three measured iterations. The earlier single-launch
baseline was an optimistic outlier (its throughput was 8.7% to 13.3% above the
three-start median) and is superseded.

Observed across-start dispersion (three fresh starts):

- relative MAD: 4.22% worst for throughput, 10.19% worst for TTFT, 4.42% for TPOT
- worst single-start deviation: 15.30% (throughput), 12.91% (TPOT), 13.27% (latency)

The Reduced8 3% relative-MAD stability cap is NOT met on this shared x86_64 lab
host. Thresholds therefore carry a margin of roughly twice the worst observed
one-sided ratio: 0.9578 low-side throughput and 1.1019 high-side latency.

## Host scope and boundaries

- The baseline is host-scoped (linux-x86_64-a3-800i-8-167) and was measured on the
  lab x86_64 A3 host from the frozen A1 environment snapshot image. A runner with a
  different hardware tag fails closed until it has its own calibration; this baseline
  cannot be reused on an aarch64 runner.
- enforce_eager=True: this is a regression baseline, not a deployment peak.
- These numbers are for an 11-layer reduced model. They do not extrapolate to the
  full 78-layer checkpoint and say nothing about the MTP speedup.

## Files

- baseline-167-eager.json: identity, per-workload median/min/max aggregates, and the
  three-start calibration samples and dispersion.
- case.json: registered model, hardware, pinned baseline checksum and threshold basis.
