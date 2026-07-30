# Nightly/Weekly Auto-Bisect

Automatically locates the **first bad commit** (and its PR) when a scheduled E2E
case fails, by binary-searching the `vllm-ascend` history between the last
known-good commit and the failing commit. It reuses the existing nightly launch
entries so the bisect reproduces the real nightly environment.

> 中文功能总览见 [`AOP_BISECT_FEATURE_zh.md`](./AOP_BISECT_FEATURE_zh.md)，
> 操作步骤见 [`USAGE_zh.md`](./USAGE_zh.md)，参数扩展见
> [`BISECT_PARAMS.md`](./BISECT_PARAMS.md)，UT 设计与结果见
> [`UT_REPORT_zh.md`](./UT_REPORT_zh.md)。

## How it works

```text
trigger (case FAIL)
  -> resolve range: bad = current commit, good = latest success row in the status table
  -> candidate list = git log --first-parent good..bad   (commit-atomic)
  -> verify endpoints (good must PASS, bad must FAIL)
  -> binary search:
       for each midpoint commit:
         checkout  (+ pip install -e . ONLY if that commit touched native/cpp files)
         run the WHOLE yaml (all test_cases) via the nightly entry
         verdict from pytest rc + benchmark_results/*.json
         print [PASS]/[FAIL]/[SKIP] <PR/commit>
         shrink window
  -> report first bad commit + PR
```

* **Commit-atomic**: each candidate is one mainline commit; the PR number is
  parsed from the `(#NNNN)` subject trailer for display.
* **Whole-YAML granularity**: nightly cannot select a single case, so each trial
  runs the entire `CONFIG_YAML_PATH` file; FAIL if any case fails.
* **Compile only on C++ changes**: by default (`--native-check per-commit`) a
  rebuild happens only when that commit's own diff touches
  `*.cpp/*.cc/*.cu/*.h/*.hpp/*.cuh`, `csrc/**`, `CMakeLists.txt`, or `setup.py`.
  Pure `.py`/yaml changes are picked up live by the editable install.
  `--native-check since-build` widens the check to all changes since the last
  build (safer across bisect jumps).
* **Runtime env follows the status table**: the paired vLLM, CANN and torch-npu
  versions are read from `env_table.csv`. If the current container env differs,
  auto-bisect switches it at runtime before building/testing the candidate. vLLM
  is always installed from source (`/vllm-workspace/vllm` by default) so rc/dev
  refs can be tested.
* **SKIP semantics**: a flaky/unconfirmed FAIL, a build failure, or a collection
  error (pytest rc 2/3/4/5, e.g. a conftest ImportError) becomes `SKIP` instead
  of a misleading FAIL — like `git bisect skip`.

## Status tables (good source, read-only)

The good commit is read from the frequency-specific CSV produced by the
pipeline:

```csv
name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,soc,scene,time
```

Nightly and weekly do not share baselines:

```text
/root/.cache/vllm-ascend/<branch>/nightly/good_table.csv
/root/.cache/vllm-ascend/<branch>/weekly/good_table.csv
```

Formal scheduled and manually dispatched workflows that use a good table also
publish the current table as a GitHub Actions artifact named
`good-table-<frequency>-<platform>-<branch>`. The downloaded archive preserves
the cadence-specific path: `nightly/good_table.csv` or
`weekly/good_table.csv`. PR-command runs do not publish the formal baseline
artifact.

For the requested case, all supplied dimensions (`--name`, `--config-yaml`,
`--soc`, and `--scene`) are matched. The good commit is the
`vLLM-Ascend Git information` of the most recent successful row. Legacy
seven-column rows remain readable during migration.

## CI triggers

After opening a PR, an authorized contributor can comment:

```text
/nightly all --aop_enabled
/weekly all --aop_enabled
/weekly <case-name> --aop_enabled
```

The slash-command dispatcher starts the corresponding workflow on `main` and
passes the PR SHA for the tested code. Consequently, workflow-file changes in a
PR take effect only after they are merged; the PR's test configuration and code
are still checked out at the PR SHA.

Weekly scheduled workflows run at `0 2 * * 0` (Sunday 10:00 Beijing time).
Scheduled runs select `all`, test `main`, and enable AOP automatically for the
single/multi-node workflows that support bisect. The A2 accuracy-model workflow
selects the LM, ASR or RM accuracy entry from the model config and is
bisect-capable. Nightly workflows expose `workflow_dispatch` and are started by
the existing external/manual dispatch path.

## Runtime environment table

The environment table is a separate CSV, defaulting to `env_table.csv` beside
the good table (override with `--env-table` / `$BISECT_ENV_TABLE`):

```csv
name,yaml/path,link,status,vLLM Git information,VLLM-Ascend Git information,CANN Version,torch-npu Version,time
```

For each candidate commit, auto-bisect first looks for an exact yaml status row.
If none exists, it uses the closest preceding status row for the same yaml/name
in first-parent history. See `env_table.sample.csv`.

## Usage

Single-node:

```bash
python -m tools.bisect.auto_bisect \
    --scene single_node \
    --config-yaml DeepSeek-R1-0528-W8A8.yaml \
    --name DeepSeek-R1-0528-W8A8 \
    --soc a2 \
    --bad-commit HEAD \
    --good-table /path/to/nightly_status.csv
```

Multi-node — run on **every** node (master + workers) pointing at a shared
`--coord-dir`. The master (`LWS_WORKER_INDEX=0`) drives the search; other nodes
auto-enter the worker loop:

```bash
python -m tools.bisect.auto_bisect \
    --scene multi_node \
    --config-yaml Qwen3-235B-W8A8.yaml \
    --bad-commit "$VLLM_ASCEND_REF" \
    --num-nodes 2 \
    --coord-dir /shared/nightly_bisect/coord
```

Common flags: `--good-commit` (skip the table), `--soc`,
`--config-base-path`
(internal/external DP configs), `--native-check {per-commit,since-build}`,
`--force-initial-build`, `--fail-confirm-retries`, `--no-verify-good`,
`--no-verify-bad`, `--trial-timeout-s`. Full reference: see `USAGE_zh.md` §9.

## Outputs

Per run, under `$BISECT_WORK_DIR/<scene>__<config_yaml>/`:

* `logs/round<N>_<sha>.log` — build + pytest output per trial (`tail -f` for
  live progress; the build step is silent on the console)
* `state.json` — resumable search window + cached verdicts (rerun the same
  command to resume)
* `report.json` — final result (first bad commit/PR + full trial history)

Exit code: `0` first-bad found; `2` not found (endpoint check failed / invalid
range / environment error).
