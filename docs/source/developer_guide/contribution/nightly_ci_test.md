# Nightly CI Test

This document explains how to trigger nightly hardware CI tests against your own PR code
on Ascend NPU hardware (A2/A3), without waiting for the scheduled nightly run.

## Background

By default, nightly CI tests run on a fixed schedule using pre-built nightly images.
Contributors can self-service trigger these tests directly against their PR changes
by combining a GitHub label with a comment command.

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

### 2. Wait for results

GitHub Actions will trigger the `Nightly-A2` or `Nightly-A3` workflow. Only tests
matching the filter will be dispatched, which saves hardware resources.

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
| `multi-node-deepseek-v3.2-W8A8-EP` | DeepSeek-V3.2-W8A8 with EP, 4-node |

**Double-node tests** (`a3.double_node.test_config`, 2-node, run after multi-node):

| Test name | Description |
|-----------|-------------|
| `multi-node-deepseek-r1-w8a8-longseq` | DeepSeek-R1-W8A8 long sequence, 2-node |
| `multi-node-qwen3-dp` | Qwen3-235B-A22B, 2-node DP |
| `multi-node-qwenw8a8-2node-eplb` | Qwen3-235B-W8A8 with EPLB, 2-node |
| `multi-node-dpsk3.2-2node` | DeepSeek-V3.2-W8A8, 2-node |
| `multi-node-qwenw8a8-2node-longseq` | Qwen3-235B-W8A8 long sequence, 2-node |
| `multi-node-qwen-disagg-pd` | Qwen3-235B disaggregated PD, 2-node |
| `multi-node-qwen-vl-disagg-pd` | Qwen3-VL-235B disaggregated PD, 2-node |
| `multi-node-deepseek-v3.1` | DeepSeek-V3.1-BF16, 2-node |
| `multi-node-deepseek-v3.2-W8A8-EP` | DeepSeek-V3.2-W8A8 with EP, 4-node |
| `multi-node-glm-5.2` | GLM-5.1-W8A8, 2-node |

**Single-node tests** (`a3.single_node.test_config`):

| Test name | Description |
|-----------|-------------|
| `mtpx-deepseek-r1-0528-w8a8` | MTP-X + DeepSeek-R1-0528-W8A8 |
| `deepseek-r1-0528-w8a8` | DeepSeek-R1-0528-W8A8 |
| `kimi-k2-thinking` | Kimi-K2-Thinking |
| `qwen3-vl-235b-a22b-instruct-w8a8` | Qwen3-VL-235B-A22B-Instruct-W8A8 |
| `deepseek-r1-0528-w8a8-prefix-cache` | DeepSeek-R1-0528-W8A8 prefix cache |
| `deepseek-v3-2-w8a8` | DeepSeek-V3.2-W8A8 |
| `glm-4.7-w8a8` | GLM-4.7 W8A8 |
| `kimi-k2.5` | Kimi-K2.5 |
| `qwen3-235b-a22b-w8a8` | Qwen3-235B-A22B-W8A8 |
| `Qwen3.5-397B-A17B-w8a8-mtp` | Qwen3.5-397B-A17B W8A8 + MTP |
| `MiniMax-M2.5-w8a8-QuaRot-A3` | MiniMax-M2.5 W8A8 + QuaRot |
| `Qwen3.5-27B-w8a8-A3` | Qwen3.5-27B W8A8 |
| `Qwen3.5-122B-A10B-W8A8-A3` | Qwen3.5-122B-A10B W8A8 |
| `DeepSeek-V4-Flash-W8A8-A3` | DeepSeek-V4-Flash W8A8 |

**Multi-card tests** (`a3.multi_card.test_config`):

| Test name | Description |
|-----------|-------------|
| `qwen3-30b-acc` | Qwen3-30B accuracy test |
| `qwen3-30b-a3b-w8a8` | Qwen3-30B-A3B-W8A8 |
| `qwen3-32b-int8` | Qwen3-32B-Int8 |
| `qwen3-32b-int8-prefix-cache` | Qwen3-32B-Int8 prefix cache |
| `Qwen3-30B-A3B-W4A8-llm-compressor` | Qwen3-30B-A3B W4A8 via llm-compressor |
| `Qwen3-30B-QuaRot` | Qwen3-30B QuaRot + eagle3 |
| `Qwen3-32B-QuaRot` | Qwen3-32B QuaRot + eagle3 |

!!! warning

    The A3 resource pool has a maximum concurrency of **5×16 NPUs**. Multi-node tests
    run with `max-parallel: 2` to avoid resource exhaustion. Running `/nightly all` on
    A3 will queue a large number of jobs — prefer targeting specific test names when
    possible.

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

To add `my-new-test` to the A2 single-node section:

1. Edit `.github/workflows/configs/nightly_config.yaml`, append under
   `a2.single_node.test_config`:

   ```yaml
     - name: my-new-test
       os: linux-aarch64-a2b3-4
       tests: tests/e2e/nightly/single_node/ops/multicard_ops_a2/test_my_new.py
   ```

2. Commit the new pytest file (`test_my_new.py`) in the same PR.

3. Trigger from the PR:

   ```text
   /nightly my-new-test
   ```

The workflow will:

- `pr_nightly_command.yml` reads your PR's `nightly_config.yaml` and resolves
  `my-new-test` → dispatch A2 only.
- `Nightly-A2` is dispatched at `main`, but `generate-a2-matrix` checks out your
  PR commit and reads the new entry from the matrix.
- `single-node-tests` runs one matrix job for `my-new-test`, with
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
