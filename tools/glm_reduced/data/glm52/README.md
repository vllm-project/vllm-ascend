# GLM-5.2 reduced nightly gate

The `GLM-5.2-W4A8-MTP-A3-Reduced8.yaml` case uses real prefix-8 W4A8
weights, retained MTP, TP8/DP1/EP, MRV1 and FULL_DECODE_ONLY. Prefix caching
is disabled. It tests fixed generated content and serving performance;
it does not measure the full model's semantic task accuracy.

`source.json` pins ModelScope revision
`6360ba682fc8adfa251ade060635e85c7269dbf2` and the provider's SHA-256/size
inventory. `suite.json` defines eight prompts and two serving workloads.
Source verification and reduced-file checksums are mandatory. Model identity
excludes build timestamps and filesystem locations.

## Provisioning

Provision outside the offline nightly job, in a dedicated workspace:

```bash
python -m tools.glm_reduced.prepare \
  --source tools/glm_reduced/data/glm52/source.json \
  --source-dir /workspace/glm52-source \
  --cache-dir /workspace/glm52-reduced --download
```

The source directory may instead contain matching pre-existing weights.
Only required shards and verified auxiliaries are needed. Configure
`reduced_model_gate.source_dir` and `cache_dir` for custom locations.
Without `source_dir`, nightly resolves the pinned ModelScope cache using
the repository's existing offline downloader.

## Explicit calibration

Use an idle machine matching the final runner, the checked-out repository
and its pinned vLLM dependency. Record the image ID and native library
hashes alongside the calibration artifacts. Select devices before launch.

```bash
python -m tools.glm_reduced.nightly \
  --config tests/e2e/nightly/single_node/models/configs/GLM-5.2-W4A8-MTP-A3-Reduced8.yaml \
  --source-dir /workspace/glm52-source --cache-dir /workspace/glm52-reduced \
  --hardware linux-aarch64-a3-800i-8 --output /workspace/calibration-new
```

This starts three independent servers. Each prompt is sent five times per
server as a single curl request. All 15 normalized responses must match.
Only CRLF is normalized to LF; whitespace and punctuation remain significant.
The generated token count must be 64 and the response must finish by length.

Each server runs one full warmup and five measured rounds per workload.
Calibration records all samples and their medians. Relative median absolute
deviation must be at most 3% within each start and across all 15 rounds.
Output throughput, mean TTFT and mean TPOT are mandatory; detailed results
also preserve tail latency. The checker verifies every input/output length
and the number of successful requests.

The pinned vLLM revision defines `random_range_ratio` as a symmetric
deviation from the requested length, so it is **0** for fixed lengths.
Using 1 would sample variable lengths, rather than the intended constant
length multiplier of 1. Per-request token checks enforce the actual workload.

After reviewing calibration, copy its `requests.json` and `baseline.json`
to this directory, naming the latter `baseline-a3-800i-8.json`, and commit
them together. Neither file is fabricated or automatically bootstrapped.
Missing files fail the configured nightly job. Do not deploy the new matrix
entry until target-runner calibration artifacts have been committed.

For preliminary x86 A3 verification, use `--hardware lab-168-a3-x86_64`.
Those results are separate artifacts and cannot qualify the aarch64 runner.
The currently planned target-runner qualification is pending resource
coordination; the local work must not be described as a calibrated CI gate.
See [the preliminary qualification record](QUALIFICATION.md) for the real
x86 run's failures. `observation-lab-168.json` contains measurements with
`baseline_qualified: false`; its format is deliberately rejected as a baseline.

## Checking and failures

To verify a separately stored preliminary baseline on its matching host:

```bash
python -m tools.glm_reduced.nightly \
  --config tests/e2e/nightly/single_node/models/configs/GLM-5.2-W4A8-MTP-A3-Reduced8.yaml \
  --source-dir /workspace/glm52-source --cache-dir /workspace/glm52-reduced \
  --hardware lab-168-a3-x86_64 --output /workspace/check-new \
  --baseline /workspace/calibration-new/baseline.json \
  --requests /workspace/calibration-new/requests.json
```

Nightly checks every prompt twice, then runs one warmup plus three measured
rounds per performance workload. Median output throughput must be at least
90% of baseline; median mean TTFT and mean TPOT must be at most 110%.
Correctness errors do not suppress performance collection when the server
is usable. Any failure propagates to pytest and the GitHub job, emits an
error annotation and summary, and preserves raw requests/responses,
benchmark JSON, identity and errors under `benchmark_results/`.

Baseline comparison locks checkpoint contents, prompt tokens, workload,
server arguments, configured environment, hardware tag, CPU architecture,
visible NPU count, NPU model and memory,
Torch/TorchNPU/Transformers, CANN and driver versions. Runtime Git revisions are
recorded for attribution but may change in a candidate. Environment/model
changes require an explicit reviewed recalibration, not wider tolerances.
