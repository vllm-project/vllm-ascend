# Nightly CI Test

This document explains how to trigger nightly hardware CI tests against your own PR code
on Ascend NPU hardware (A2/A3), without waiting for the scheduled nightly run.

## Background

By default, nightly CI tests run on a fixed schedule using pre-built nightly images.
Contributors can self-service trigger these tests directly against their PR changes
by combining a GitHub label with a comment command.

!!! note

    The `schedule_nightly_test_*.yaml` workflows do not declare a GitHub Actions
    `schedule:` (cron) trigger; they are dispatched by external automation through
    `workflow_dispatch`. See
    [CI workflow triggers and the schedule_ prefix](./testing.md#ci-workflow-triggers-and-the-schedule_-prefix).

## How to Trigger

### 1. Post a comment

Post one of the following comments in the PR to specify which tests to run.
The comment itself triggers the workflow — no label is required.

| Comment | Effect |
|---------|--------|
| `/nightly` | Run **all** nightly tests |
| `/nightly all` | Run **all** nightly tests (same as above) |
| `/nightly test1 test2 ...` | Run only the **named** tests |
| `/nightly <tests> --aop_enabled` | Run named tests with AOP bisect / classify enabled |

!!! note

    Only repository **Contributors** (Triage role) and **Maintainers** (Write role) can
    trigger the `/nightly` command. If you do not have this permission, ask a maintainer
    to post the comment for you. You can find the list of maintainers and contributors in
    the project's [Governance](../../community/governance.md) page or by checking the
    [CODEOWNERS](https://github.com/vllm-project/vllm-ascend/blob/main/.github/CODEOWNERS)
    file.

### Maintenance window

Maintainers can pause PR-triggered commands during a maintenance window by setting
the repository variable `NIGHTLY_COMMAND_BLOCK_WINDOW`. The value is an optional
switch followed by an `HH:MM-HH:MM` range in Beijing time (UTC+8):

| Value | Behavior |
|-------|----------|
| `true 22:00-08:00` | Gate enabled; block inside the window |
| `22:00-08:00` | Same as above (switch omitted) |
| `false 22:00-08:00` | Gate explicitly disabled |
| unset / invalid | Gate disabled (fail-open) |

Switch aliases are `true/1/yes/on` and `false/0/no/off` (case-insensitive); a space
or comma may separate the switch from the range. Cross-midnight ranges are
supported. While the gate is enabled and the current Beijing time falls inside the
range, `/nightly` and `/weekly` comments do not dispatch tests; the bot replies with
the active window instead.

### 2. Wait for results

GitHub Actions will trigger the `Nightly-A2` or `Nightly-A3` workflow. Only tests
matching the filter will be dispatched, which saves hardware resources.

## AISBench Output Artifacts

YAML-driven nightly tests upload the complete AISBench `outputs/` directory to
the dispatched workflow run's **Artifacts** section, for all YAML-driven runs.
Look for `aisbench-outputs-<scene>-<job>-<run-id>-<attempt>`. It contains the experiments
from that test container, including configuration snapshots, logs, performance
details, and reports produced by AISBench. Multiple benchmark experiments retain
their original directory structure:

```text
# Single node
default/<experiment-timestamp>/...

# Multiple nodes
node-0/default/<experiment-timestamp>/...
```

Multi-node tests preserve outputs on shared storage after pytest finishes and
before AOP reruns or Pod deletion. Only nodes that produced outputs appear in the
artifact. Upload is attempted even when a benchmark or baseline check fails;
files already generated are retained. A cancelled job or lost runner/Pod may
prevent collection. These artifacts have a seven-day retention period and are
separate from benchmark summary JSON.

## Differences Between PR and Scheduled Runs

| | Scheduled / Manual Dispatch | PR-triggered |
|---|----------------------------|---|
| Trigger | Cron (daily) or `workflow_dispatch` | `/nightly` comment |
| Code tested | Pre-built nightly image | Your PR's HEAD commit (source installed fresh) |
| Test scope | All tests | Configurable via `/nightly <names>` |
| vLLM + vllm-ascend | From image | Checked out and installed from source |
| Test matrix | From main branch's matrix YAML | From PR branch's matrix YAML |

When a PR run is detected (`is_pr_test: true`), the workflow additionally:

1. Uninstalls any existing vllm packages in the container.
2. Checks out the specific vllm version and your PR's vllm-ascend commit from source.
3. Installs all dependencies from source.
4. Installs the `aisbench` benchmark suite.

## Test Matrix Data Source

The set of nightly test cases (their names, runners, test paths, model configs) is
declared in a single data file:

```text
.github/workflows/configs/nightly_config.yaml
```

The file is organized as `a2:` and `a3:` top-level keys (one per SoC). Under each
SoC, tests are grouped by execution shape (single-node, multi-node, double-node,
multi-card, accuracy) and each group holds a `test_config` (or `nightly` / `pr_only`
for accuracy) list whose entries carry a `name` plus the fields consumed by the
downstream reusable workflows (`os`, `tests`, `config_file_path`, `size`, etc.).

Both the `Nightly-A2` and `Nightly-A3` workflows dynamically read this file at run
time — there is no hardcoded test matrix in the workflow YAMLs. The
`/nightly <name>` slash command resolves names by walking the same file from the
PR branch, so newly added entries can be exercised on a PR before they land on
main.

## Adding a New Nightly Test Case

To add a new test case (no need to touch the workflow YAMLs):

1. Append an entry under the appropriate section in
   `.github/workflows/configs/nightly_config.yaml`. Each entry needs at least:
   - `name`: unique identifier used in `/nightly <name>` filters
   - `os` (for single-node / multi-card pytest+yaml tests) or `runner` is inferred
   - one of `tests:` (pytest directory) or `config_file_path:` (YAML-driven model config)
   - `size` (multi-node / double-node only)
2. Add the actual test files (pytest modules under `tests/e2e/nightly/...` or
   YAML model configs in `tests/e2e/nightly/.../configs/`).
3. Open a PR. Once CI is green, you can validate the new entry against real NPU
   hardware **without** merging the PR — see *Examples* below.

## Available Test Names

The test names you can pass to `/nightly` correspond to the `name` fields under
the matching section in `.github/workflows/configs/nightly_config.yaml`. The
tables below mirror the current contents of that file.

### A2 workflow (`.github/workflows/schedule_nightly_test_a2.yaml`)

**Single-node tests** (`a2.single_node.test_config`):

| Test name | Description |
|-----------|-------------|
| `test_custom_op_multi_card` | Custom operator tests (multi card) |
| `qwen3-vl-32b-instruct-w8a8` | Qwen3-VL-32B-Instruct W8A8 |
| `qwen3-32b-int8` | Qwen3-32B INT8 quantization |
| `Qwen3.5-27B-w8a8-A2` | Qwen3.5-27B W8A8 |
| `Qwen3.5-397B-A17B-w4a8-mtp` | Qwen3.5-397B-A17B W4A8 + MTP |

**Multi-node tests** (`a2.multi_node.test_config`):

| Test name | Description |
|-----------|-------------|
| `multi-node-qwen3-235b-dp` | Qwen3-235B-A22B, 2-node DP |
| `multi-node-GLM-5.1-w8a8-A2` | GLM-5.1 W8A8, 2 nodes |
| `multi-node-Kimi-K2.5-W4A8-A2` | Kimi-K2.5 W4A8, 2 nodes |

**Accuracy tests** (`a2.accuracy.nightly` and `a2.accuracy.pr_only`):

| Test name | Description | Scope |
|-----------|-------------|-------|
| `accuracy-group-1` | Qwen3-VL-8B, Qwen3-8B, Qwen2-Audio-7B, etc. | nightly |
| `accuracy-group-2` | ERNIE-4.5, Molmo-7B, Llama-3.2-3B, etc. | nightly |
| `accuracy-group-3` | Qwen3-30B-A3B, Qwen3-VL-30B-A3B, etc. | nightly |
| `accuracy-group-4` | Qwen3-Next-80B-A3B, Qwen3-Omni-30B-A3B, etc. | nightly |
| `pr-accuracy-group-1` | gemma-3-4b-it, internlm3-8b-instruct, etc. | pr_only |
| `pr-accuracy-group-2` | Qwen2.5-Math-RM-72B, Hunyuan-A13B-Instruct | pr_only |

The `pr-accuracy-group-*` entries only run on `/nightly` (PR-triggered) runs;
`/nightly all` on the schedule skips them.

### A3 workflow (`.github/workflows/schedule_nightly_test_a3.yaml`)

**Multi-node tests** (`a3.multi_node.test_config`, 4-node):

| Test name | Description |
|-----------|-------------|

**Double-node tests** (`a3.double_node.test_config`, 2-node, run after multi-node):

| Test name | Description |
|-----------|-------------|
| `multi-node-qwen-vl-disagg-pd` | Qwen3-VL-235B disaggregated PD, 2-node |
| `multi-node-deepseek-v3.1` | DeepSeek-V3.1-BF16, 2-node |
| `multi-node-glm-5.2` | GLM-5.1-W8A8, 2-node |

!!! warning

    The A3 resource pool has a maximum concurrency of **5×16 NPUs**. Multi-node tests
    run with `max-parallel: 2` to avoid resource exhaustion. Running `/nightly all` on
    A3 will queue a large number of jobs — prefer targeting specific test names when
    possible.

### A3-560T workflow (`.github/workflows/schedule_nightly_test_a3_560t.yaml`)

**Single-node tests** (`a3-560t.single_node.test_config`):

| Test name | Description |
|-----------|-------------|
| `Qwen3.8-27B-w8a8-A3` | Qwen3.8-27B W8A8 + MTP |
| `MiniMax-M3-W8A8-A3` | MiniMax-M3 W8A8 |
| `qwen_3_6_27b_w8a8_A3` | Qwen3.6-27B W8A8 + MTP |
| `DeepSeek-V4-Flash-W8A8-A3` | DeepSeek-V4-Flash W8A8 + MTP |
| `DeepSeek-V4-Flash-W8A8-A3-MRV2` | DeepSeek-V4-Flash W8A8, MRV2 + DSpark |
| `Qwen3.5-122B-A10B-W8A8-A3` | Qwen3.5-122B-A10B W8A8 + MTP |
| `kimi-k2.6_max_model_len` | Kimi-K2.6 W4A8, maximum model length |
| `Kimi-K2.6-w4a8-A3` | Kimi-K2.6 W4A8 + DFlash |
| `glm-5.1-w8a8-prefill-mc2` | GLM-5.1 W8A8, prefill MC2 |
| `MiniMax-M3-BF16-A3` | MiniMax-M3 BF16 |
| `Kimi-K2.6-W4A8-PCP-DCP` | Kimi-K2.6 W4A8 + PCP + DCP |
| `glm-5.2-W4A8C8-DSpark-PCP` | GLM-5.2 W4A8C8 + DSpark + PCP |
| `Qwen3.5-397B-A17B-w8a8-mtp` | Qwen3.5-397B-A17B W8A8 + MTP |
| `Qwen3-235B-A22B-W8A8_piecewise_fullgraph_A3` | Qwen3-235B-A22B W8A8, PIECEWISE and full graph |
| `acceptace_rate_deepseekv4-flash-w8a8-mtp` | DeepSeek-V4-Flash W8A8 + MTP acceptance rate |
| `glm-5.2-w4a8c8-sfa-dcp` | GLM-5.2 W4A8C8, DCP16 |
| `glm-5.2-w4a8-dspark` | GLM-5.2 W4A8 + DSpark |
| `glm-5.2-w4a8-mtp` | GLM-5.2 W4A8 + MTP |
| `glm-5.2-w4a8-dspark-v2` | GLM-5.2 W4A8, MRV2 + DSpark |
| `glm-5.2-w4a8-mtp-v2` | GLM-5.2 W4A8, MRV2 + MTP |

**Multi-card tests** (`a3-560t.multi_card.test_config`):

| Test name | Description |
|-----------|-------------|
| `qwen3-30b-acc` | Qwen3-30B accuracy test |
| `Qwen3.5-27B-w8a8-A3` | Qwen3.5-27B W8A8 + MTP |
| `qwen3-30b-a3b-w8a8` | Qwen3-30B-A3B W8A8 |
| `Qwen3-30B-A3B-W4A8-llm-compressor` | Qwen3-30B-A3B W4A8 via llm-compressor |
| `qwen3-32b-int8-prefix-cache` | Qwen3-32B W8A8 QuaRot prefix cache |
| `qwen3-32b-int8` | Qwen3-32B W8A8 QuaRot |
| `Qwen3-30B-QuaRot` | Qwen3-30B-A3B W8A8 QuaRot + Eagle3 |
| `Qwen3-32B-QuaRot` | Qwen3-32B W8A8 QuaRot + Eagle3 |
| `Qwen3-30B-A3B-VWN-eagle3-PCP` | Qwen3-30B-A3B BF16 + VWN-Eagle3 + PCP |

## Examples

Run all available nightly tests against your PR:

```text
/nightly
```

Run only the custom operator multi-card test:

```text
/nightly test_custom_op_multi_card
```

Run two specific tests at once (one per SoC):

```text
/nightly test_custom_op_multi_card mtpx-deepseek-r1-0528-w8a8
```

Run the MLA, GQA and SFA context-parallel tests on A3-560T:

```text
/nightly Kimi-K2.6-W4A8-PCP-DCP Qwen3-30B-A3B-VWN-eagle3-PCP glm-5.2-W4A8C8-DSpark-PCP --a3-560t
```

Run a single accuracy group (with all of its models):

```text
/nightly accuracy-group-1
```

Run a single accuracy model (only that model from a group):

```text
/nightly accuracy-group-1/Qwen3-8B
```

Re-trigger after fixing an issue: just push a new commit. The `synchronize` event
re-runs the workflow and picks up the existing `/nightly` comment automatically — no
need to post a new comment.

## Profiling from PR Comments

### Overview

Use `/nightly --profile` to capture Ascend PyTorch Profiler data during selected
YAML-driven AISBench performance benchmarks. Raw traces are uploaded to OBS;
offline CPU jobs can parse them and publish per-rank results. A GitHub Artifact
contains manifests, confirmed download links, and a download script.

This section describes the PR comment interface. Profiling is supported by the
`/nightly` command for YAML-driven AISBench performance benchmarks; it does not
add capture to arbitrary pytest tests or accuracy benchmarks. `/weekly --profile`
is not supported.

### Important Notes

!!! warning "Profiling affects benchmark performance"

    Active profiling consumes resources, and stopping the profiler can take
    additional time. Throughput can decrease and latency can increase enough to
    fail the existing performance baseline. Baseline checks remain enabled.
    Use results from a comparable run without profiling for performance comparisons. Check the
    benchmark verdict and the profiling manifest independently: a baseline failure
    can still leave useful profiling artifacts.

- Capture covers all serving instances recorded for the selected benchmark.
  There is no node or rank selection option. With internal DP, one API endpoint
  controls its backend engines, including engines on headless nodes. In P/D
  deployments, the separate serving groups are controlled through their own endpoints.
- Start with a short capture and leave stack collection disabled unless it is
  needed. Trace size and memory use depend on the model, worker count, workload,
  and capture duration. The workflow does not impose a fixed disk-space budget;
  resource exhaustion is reported as a failure.
- Failures that prevent an enabled profiling capture, upload, or parse from
  completing are reported as errors.
- `--profile-duration` is the requested capture interval. It does not include all
  start/stop, compression, transfer, or offline parsing time.

### Quick Start

Inspect a previous successful scheduled nightly or PR-triggered run with comparable
hardware, deployment, and workload settings. Use its AISBench timings or a
steady-state report, when available, to choose a capture window. If no suitable
reference exists, trigger a run without profiling:

```text
/nightly DeepSeek-V4-Flash-W8A8-A3
```

Choose `--profile-start-after` and `--profile-duration` from the report. The
following example collects the `perf` benchmark with a 60-second offset and an
eight-second capture. Adjust both values to fit the desired window; the start
can be earlier or later than the 60-second default:

```text
/nightly DeepSeek-V4-Flash-W8A8-A3 --profile --profile-benchmark-names perf --profile-start-after 60 --profile-duration 8
```

The test name and `perf` key in this example come from the nightly matrix and its
[model YAML](https://github.com/vllm-project/vllm-ascend/blob/main/tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Flash-W8A8-A3.yaml).
Choose the corresponding names from your PR when testing another configuration.

Wait for the capture job and its `Plan`, `Parse / node-N`, and `Finalize` jobs.
In the workflow run's **Artifacts** section, download `profile-artifact-<key>`
and extract it into `./profile_artifact`. Download the parsed archives with:

```bash
bash ./profile_artifact/download.sh parsed
```

The **Profiling summary** step in `Finalize` also contains grouped links and
download commands. For raw-only collection, add `--profile-output raw`; no offline
parse jobs are needed, and the summary is printed in the capture job.

### Select Tests and Benchmarks

There are three distinct names:

| Name | Source | Example |
|------|--------|---------|
| Nightly test name | `name` in `.github/workflows/configs/nightly_config.yaml` | `DeepSeek-V4-Flash-W8A8-A3` |
| Benchmark name | Key under `benchmarks:` in the model YAML | `perf` |
| Benchmark artifact name | Generated from the YAML test configuration name and benchmark name | `DeepSeek-V4-Flash-W8A8-A3__perf` |

For example, the relevant part of a model YAML can look like this:

```yaml
test_cases:
  - name: DeepSeek-V4-Flash-W8A8-A3
    # Model and server configuration omitted.
    benchmarks:
      acc-gpqa:
        case_type: accuracy
        # Accuracy settings omitted.
      perf:
        case_type: performance
        # Performance settings omitted.
```

`--profile-benchmark-names perf` selects `perf` for capture. It does not change
which benchmarks the YAML test executes: accuracy checks and other configured
benchmarks still run. Selection uses exact benchmark keys; use a comma-separated
list without spaces to select several, such as `perf_TPOT20,perf_TPOT50`.
These names must exist in the selected model configurations.

AISBench's internal warmup is excluded from capture timing. A separate
YAML benchmark named `warmup` is still an ordinary benchmark: if its `case_type`
is `performance`, the default selection includes it. Select the intended benchmark
keys explicitly when such a setup benchmark should run without capture.

### Profiling Options

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--profile` | Disabled | Enable collection for selected performance benchmarks. Other profiling options alone do not enable it. |
| `--profile-start-after <seconds>` | `60` when neither trigger is specified | Delay from the first formal AISBench request, excluding internal warmup. Accepts a non-negative integer. |
| `--profile-start-after-requests <count>` | Disabled | Start after this many successful formal requests complete, summed across AISBench workers. Excludes warmup and failed requests. Accepts a positive integer. Mutually exclusive with `--profile-start-after`. |
| `--profile-duration <seconds>` | `8` | Requested capture duration after the start calls return. Accepts a positive integer. |
| `--profile-output <mode>` | `parsed` | `parsed` uploads raw traces and runs offline parsing; `raw` uploads only raw traces and skips offline parsing. |
| `--profile-with-stack` | Disabled | Enable the profiler's stack/module attribution option. This can increase overhead and artifact size. |
| `--profile-benchmark-names <names>` | All performance benchmarks | Comma-separated YAML benchmark keys to capture. Does not filter which benchmarks execute. |

Triggers apply independently to each selected benchmark. Omitting both start
options uses 60 seconds from that benchmark's first formal request. Request-count
mode does not also wait for the default 60-second offset. Capture duration always
uses seconds.

For a capture after 100 successful formal requests:

```text
/nightly DeepSeek-V4-Flash-W8A8-A3 --profile --profile-benchmark-names perf --profile-start-after-requests 100 --profile-duration 8
```

Workers periodically publish AISBench's existing success counters outside the
request path, and the controller polls their sum. Both intervals are one second,
so capture can start roughly two seconds after the threshold is reached, plus
scheduling and API overhead; the observed count may exceed the threshold. The
manifest records `completed_requests` at the trigger and `actual_start_after`,
the time of the start calls relative to the first formal request marker. Request-count mode
is useful for workload progress, but long requests may produce no completions
during a useful capture window; use the time trigger in that case.

For a raw-only capture with a different offset and duration:

```text
/nightly DeepSeek-V4-Flash-W8A8-A3 --profile --profile-benchmark-names perf --profile-start-after 80 --profile-duration 5 --profile-output raw
```

### Choose a Capture Window

Use a successful reference run with complete request timings. Compare hardware,
model, deployment, concurrency, request count, and prompt/output lengths. Existing
scheduled nightly runs are suitable references; a fresh run is only needed when
those settings have changed or no comparable result exists.

Choose a time or successful-request count after ramp-up and leave enough work for
the full capture before drain. The default 60-second delay is adjustable. Long
requests may have no completions during useful decode activity, so use a time
trigger when a completion threshold would be too late.

The controller uses a marker written before the first formal request and does
not depend on plotting. If a steady-state report is available, its result-based
request timestamps provide a reference, not an exact synchronization guarantee.
Profiling itself can change workload timing; inspect the actual trace and capture
status after the run.

### Follow the Workflow

```text
1. PR comment -> Deploy services and record topology
2. AISBench warmup -> First formal request (t = 0)
3. Wait for time or completed-request threshold -> Start -> Wait duration -> Stop
4. Archive raw traces -> Upload to OBS
5. raw:    Publish final index and summary
   parsed: Plan -> Parse per node -> Finalize (index and summary)
```

Each unique serving endpoint receives start and stop requests. Internal DP
workers sharing an endpoint are controlled together. The benchmark's deployment
snapshot records the endpoint, node index, and trace directory for each target.

Offline parsing runs after raw upload, on separate CPU jobs. Nodes can parse in
parallel. Within each node, rank parsing overlaps with compression/upload of the
previous rank, with at most one completed rank waiting for publication. The
`Parse / node-N` job name refers to the source node, not to one particular DP rank.

The workflow attempts artifact publication and finalization even after a test or
parse failure. Cancellation, runner loss, or a failure before any artifact exists
can still prevent publication. Check the final manifest for completeness.

The capture job groups raw upload and index preparation under **Publish profiling
data**. **Upload profiling index** and **Profiling summary** remain separate so
index upload and download-link reporting can still run after publication errors.
Finalize similarly groups result merging and index preparation under **Finalize
profiling data**.

### Find and Download Artifacts

Open the dispatched `Nightly-*` workflow run, then its **Artifacts** section:

| Artifact | Use |
|----------|-----|
| `profile-artifact-<key>` | Final index with manifests, confirmed download links, and a download script. Start here. |
| `profile-capture-<key>` | Intermediate raw index for parsed-mode runs. Useful while parsing is pending or if finalization cannot finish. |
| `profile-shard-<key>-node-N` | Per-node parse status used by Finalize; useful for diagnosing a missing or failed shard. |

The key includes repository, run ID, run attempt, deployment scene, branch, and
job identity. Match it to the test and run attempt being investigated. The
index artifacts currently have a 90-day GitHub retention period. OBS data retention
depends on the bucket's lifecycle policy and is separate from GitHub retention.

Extract the final index into `profile_artifact/`:

```text
profile_artifact/
├── profile_manifest.json
├── benchmarks/
│   └── <benchmark_artifact_name>/
│       └── serve_manifest.json
├── raw/
│   └── downloads.md
├── parsed/
│   └── downloads.md
├── summary.txt
└── download.sh
```

`raw/` and `parsed/` contain download documents, not the large trace archives.
The archives live in OBS. Raw-only runs omit `parsed/`.
`profile_manifest.json` records requested settings, benchmark/target/rank status,
errors, and archive locations. Each benchmark has its own `serve_manifest.json`
snapshot so later deployments do not overwrite the topology associated with
earlier captures. `summary.txt` is a readable overview of the same index.

With Bash and curl available, download all confirmed archives, or select a stage:

```bash
bash ./profile_artifact/download.sh all
bash ./profile_artifact/download.sh raw
bash ./profile_artifact/download.sh parsed ./profile-download
```

The optional second argument is the destination; its default is
`./profile-download`. The script preserves the `raw/` and `parsed/` archive paths,
retries failed transfers, and downloads sequentially. It stops on an unsuccessful
download; incomplete files keep a `.part` suffix. It does not extract archives.
For an individual target or rank, use the link in its `downloads.md` entry.

The **Profiling summary** step prints an overview, quick-download commands, and
separate expandable **RAW download links** / **PARSED download links** groups.
In parsed mode, look in `Finalize`; raw-only runs and capture publication failures
also print a summary in the capture job when an index could be prepared. This log
provides a second access path if the GitHub index upload fails.

The printed `obsutil cp ... -r -j=4` commands download a whole run's data with up
to four concurrent files. They require OBS list/download permissions. The script
and individual HTTPS links use the objects' existing read permissions; possession
of an index does not grant additional access.

### Inspect Profiling Results

Raw archives group traces by benchmark and serving target. Parsed archives are
published per rank, so you can download just the ranks needed for analysis.
Unpack an archive into a chosen directory:

```bash
mkdir -p ./profile-view
tar -xzf <archive.tar.gz> -C ./profile-view
```

A parsed archive contains the rank's `ASCEND_PROFILER_OUTPUT/` directory and
available profiler metadata. Typical contents include:

| File | Purpose |
|------|---------|
| `trace_view.json` | Timeline of CPU/NPU activity for inspection in MindStudio Insight. |
| `kernel_details.csv` | Detailed kernel execution records. |
| `operator_details.csv` / `op_statistic.csv` | Operator details and aggregate statistics. |
| `communication.json` / `communication_matrix.json` | Communication analysis, when produced by the profiler. |
| `profiler_info*.json` / `profiler_metadata.json` | Capture metadata, when available. |

Open `trace_view.json` with MindStudio Insight and select the relevant rank and
time region. For the viewer and file descriptions, see
[View Results](../performance_and_debug/service_profiling_guide.md#5-view-results).
Available files depend on the profiler version and captured activity.

If you downloaded raw data, or need to repeat analysis, unpack the raw archive
and locate a `*_ascend_pt` directory. In an environment with a compatible
Ascend PyTorch Profiler installation, analyze that directory:

```python
from torch_npu.profiler.profiler import analyse

analyse("./profile-view/<target>/<path-to-rank>_ascend_pt")
```

Use a compatible environment, preferably the capture job's image/software
versions. Repeat for the required rank directories. The offline CI workflow uses
this analysis API with its default process setting. See
[Analyze Data](../performance_and_debug/service_profiling_guide.md#4-analyze-data)
for the underlying service profiling workflow.

### Profiling Troubleshooting

| Symptom | What to check or change |
|---------|-------------------------|
| Throughput drops or the baseline check fails | Compare against a separate run without profiling. Check profiling status independently; successful traces can still be useful. Shorten capture and leave stack collection off when possible. |
| No benchmarks were captured | Confirm `--profile` is present, the test uses the YAML/AISBench performance path, and selected names exactly match performance benchmark keys. |
| `benchmark_ended_before_start` | The benchmark finished before its offset elapsed. Reduce `start_after` or increase the workload duration. |
| `benchmark_ended_before_request_threshold` | Too few successful formal requests completed before the benchmark ended. Lower `start_after_requests`, increase the workload, or use a time trigger. This is an incomplete capture and fails profiling. |
| Capture ends with `benchmark_ended` | The benchmark ended before the requested interval completed. Choose an earlier/shorter window or run more requests. This is an incomplete capture. |
| Raw exists but parsed data is missing | In `raw` mode this is expected. Otherwise inspect Plan/Parse/Finalize and the manifest's per-target/per-rank errors; raw data can be downloaded for another analysis attempt. |
| Manifest status is `partial` | Inspect benchmark, target, and rank errors. Download links include confirmed uploads only and do not imply that all requested data was produced. |
| One internal DP API port produces many rank traces | The endpoint controls multiple backend engines. Collection covers the recorded deployment, including headless nodes. |
| Storage or memory is exhausted | Shorten subsequent captures, leave stack collection off, and choose fewer benchmarks. Download only needed ranks; use a destination with enough space for archives and extracted data. |
| Download returns an access error | Check object read permissions; `obsutil` recursive download also needs list permission. GitHub Artifact access alone does not grant OBS access. |

## AOP Hooks (Bisect)

Add `--aop_enabled` to any `/nightly` command to enable the AOP pipeline:

```text
/nightly all --aop_enabled
```

When enabled, the workflow will:

1. **Capture** the test result (pass / fail).
2. **Classify** the failure as environmental (network, infra) or code-related.
3. **Bisect** genuine code failures to pinpoint the offending commit.

This is useful for automated root-cause analysis of nightly regressions.

## Adding a New Test Case — Worked Example

To add `my-new-test` to the A3 multi-card section:

1. Edit `.github/workflows/configs/nightly_config.yaml`, append under
   `a3.multi_card.test_config`:

   ```yaml
     - name: my-new-test
       os: linux-aarch64-nightly-a3-4
       tests: tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_my_new.py
   ```

2. Commit the new pytest file (`test_my_new.py`) in the same PR.

3. Trigger from the PR:

   ```text
   /nightly my-new-test
   ```

The workflow will:

- `pr_nightly_command.yml` reads your PR's `nightly_config.yaml` and resolves
  `my-new-test` → dispatch A3 only.
- `Nightly-A3` is dispatched at `main`, but `setup-vars` checks out your
  PR commit and reads the new entry from the matrix.
- `multi-card-tests` runs one matrix job for `my-new-test`, with
  `should_run=true`. The reusable workflow checks out your PR code (via
  `vllm_ascend_ref`) and runs your pytest.

## Troubleshooting

**The workflow didn't start after I posted the comment.**

- Check that the comment starts exactly with `/nightly` with no leading spaces or
  extra characters before the slash.
- Confirm you have at least Triage permission on the repository; unauthorized
  users' comments are ignored.
- To re-trigger after fixing an issue, simply push a new commit — the workflow will
  reuse the existing `/nightly` comment automatically.

**My `/nightly` comment was ignored during the maintenance window.**

- If the repository variable `NIGHTLY_COMMAND_BLOCK_WINDOW` is set and the current
  Beijing time is inside the configured range, `/nightly` and `/weekly` are
  intentionally blocked to protect the third-party scheduling tests. The bot replies
  with the active window; retry after it ends.

**Only some tests ran, not the ones I expected.**

- Test names are case-sensitive and must match the `name` field in
  `.github/workflows/configs/nightly_config.yaml` exactly (see the tables above).
- For a PR-triggered run, the matrix is loaded from your PR's
   `nightly_config.yaml`, not main. If a name isn't in your PR's file, it won't
  be recognized and the dispatch will be skipped.
- Check the `parse-trigger` job output in GitHub Actions for the resolved
  `test_filter` value.

**The workflow ran with the scheduled image, not my PR code.**

- Confirm the workflow was triggered by `repository_dispatch` (slash command),
  not bare `workflow_dispatch`. The `pr_nightly_command.yml` workflow is what
  actually dispatches `schedule_nightly_test_a2.yaml` / `_a3.yaml` with
  `vllm_ascend_ref` pointing at your PR SHA.

**A new test I added isn't being recognized.**

- Confirm the entry is well-formed YAML under
  `.github/workflows/configs/nightly_config.yaml`. The `name` field is required
  and must be unique within the SoC's section.
- The matrix is loaded from your PR branch, so make sure the file is committed
  to the same branch the `/nightly` comment was posted on.

**How to obtain more detailed logs to pinpoint problems for multi-node tests**

- For most issues, the stdout pop-up logs from GitHub actions are sufficient (this log always represents the logs from the first node).
- If the logs from a first node are no longer sufficient to provide effective logging information, see the summary of your jobs to download log archive for the corresponding test, which includes the framework-side logs and plog information for each node, structured as follows:

  ```shell
  .
  ├── node0
  │   ├── root
  │   │   └── ascend
  │   │       └── log
  │   └── var
  │       └── log
  │           └── vllm-deepseek-v3-0f233d-0_logs.txt
  └── node1
      ├── root
      │   └── ascend
      │       └── log
      └── var
          └── log
              └── vllm-deepseek-v3-0f233d-0-1_logs.txt
  ```
