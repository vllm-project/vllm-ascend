# Nightly Auto-Bisect

Automatically locates the **first bad commit** (and its PR) when a nightly E2E
case fails, by binary-searching the `vllm-ascend` history between the last
known-good commit and the failing commit. It reuses the existing nightly launch
entries so the bisect reproduces the real nightly environment.

> 中文使用指南见 [`USAGE_zh.md`](./USAGE_zh.md)。

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
  Pure `.py`/yaml changes are picked up live by the editable install (vLLM is
  never touched). `--native-check since-build` widens the check to all changes
  since the last build (safer across bisect jumps).
* **SKIP semantics**: a flaky/unconfirmed FAIL, a build failure, or a collection
  error (pytest rc 2/3/4/5, e.g. a conftest ImportError) becomes `SKIP` instead
  of a misleading FAIL — like `git bisect skip`.

## Status table (good source, read-only)

The good commit is read from the nightly status CSV (produced by the pipeline):

```csv
name,yaml/path,link,status,vLLM Git information,vLLM-Ascend Git information,time
```

For the requested case (matched by `--name` or by `--config-yaml` against
`yaml/path`), the good commit is the `vLLM-Ascend Git information` of the most
recent row whose `status` is `success`. Point `--good-table` at it (or
`$BISECT_GOOD_TABLE`). See `good_table.sample.csv`.

## Usage

Single-node:

```bash
python -m tools.bisect.auto_bisect \
    --scene single_node \
    --config-yaml DeepSeek-R1-0528-W8A8.yaml \
    --name DeepSeek-R1-0528-W8A8 \
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

Common flags: `--good-commit` (skip the table), `--config-base-path`
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
