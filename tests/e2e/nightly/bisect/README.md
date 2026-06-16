# Nightly Auto-Bisect

Automatically locates the **first bad commit** (and its PR) when a nightly E2E
case fails, by binary-searching the `vllm-ascend` history between the last
known-good commit and the failing commit. It reuses the existing nightly launch
entries so the bisect reproduces the real nightly environment.

## How it works

```
trigger (case FAIL)
  -> resolve range: bad = current commit, good = good-table lookup
  -> candidate list = git log --first-parent good..bad   (commit-atomic)
  -> verify endpoints (good must PASS, bad must FAIL)
  -> binary search:
       for each midpoint commit:
         checkout  (+ pip install only if native/build files changed)
         run the failing case via the nightly entry
         read pass_fail / exit code -> PASS | FAIL | SKIP
         print [PASS]/[FAIL]/[SKIP] <PR/commit>
         shrink window
  -> report first bad commit + PR
```

* **Commit-atomic**: each candidate is one mainline commit; the PR number is
  parsed from the `(#NNNN)` subject trailer for display.
* **Compile optimisation**: a rebuild is triggered only when the delta *since
  the last successfully built commit* touches native/build-definition files
  (`*.cpp/*.h/CMakeLists.txt/setup.py/...`). Pure `.py`/yaml changes are picked
  up live by the editable install — checkout only.
* **Flaky guard / SKIP**: a FAIL is re-confirmed; an unstable commit or a build
  failure becomes `SKIP` (like `git bisect skip`) instead of a misleading FAIL.

## Good table

A plain CSV at `$BISECT_GOOD_TABLE` (default
`/root/.cache/nightly_bisect/good_table.csv`), designed to be read at a glance.
See `good_table.sample.csv`:

```
case_key,scene,config_yaml,case_name,last_good_commit,last_good_pr,updated_at
single_node::DeepSeek-R1-0528-W8A8.yaml::DeepSeek-R1-0528-W8A8-aclgraph,single_node,DeepSeek-R1-0528-W8A8.yaml,DeepSeek-R1-0528-W8A8-aclgraph,<sha>,10442,2026-06-15T02:00:00Z
```

`case_key = scene::config_yaml::case_name` is the lookup key.

## Usage

Single-node:

```bash
python -m tests.e2e.nightly.bisect.auto_bisect \
    --scene single_node \
    --config-yaml DeepSeek-R1-0528-W8A8.yaml \
    --case-name DeepSeek-R1-0528-W8A8-aclgraph \
    --bad-commit HEAD
```

Multi-node — run on **every** node (master + workers) pointing at a shared
`--coord-dir`. The master (`LWS_WORKER_INDEX=0`) drives the search; other nodes
auto-enter the worker loop:

```bash
python -m tests.e2e.nightly.bisect.auto_bisect \
    --scene multi_node \
    --config-yaml Qwen3-235B-W8A8.yaml \
    --bad-commit "$VLLM_ASCEND_REF" \
    --num-nodes 2
```

Common flags: `--good-commit` (skip the table), `--config-base-path`
(internal/external DP configs), `--fail-confirm-retries`, `--no-verify-good`,
`--no-verify-bad`. Set `BISECT_UPDATE_GOOD_TABLE=1` to write the discovered
last-good commit back to the table.

## Outputs

Per run, under `$BISECT_WORK_DIR/<case_key>/`:
- `logs/round<N>_<sha>.log` — build + pytest output per trial
- `state.json` — resumable search window + cached verdicts
- `report.json` — final result (first bad commit/PR + full trial history)
