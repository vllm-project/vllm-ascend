# Nightly CI Test

This document explains how to trigger nightly hardware CI tests against your own PR code
on Ascend NPU hardware (A2/A3), without waiting for the scheduled nightly run.

## Background

By default, nightly CI tests run on a fixed schedule using pre-built nightly images.
Contributors can self-service trigger these tests directly against their PR changes
by combining a GitHub label with a comment command.

## How to Trigger

### 1. Post a comment

First, post one of the following comments in the PR to specify which tests to run:

| Comment | Effect |
|---------|--------|
| `/nightly` | Run **all** nightly tests |
| `/nightly all` | Run **all** nightly tests (same as above) |
| `/nightly test1 test2 ...` | Run only the **named** tests |

### 2. Add the label

After posting the comment, add the **`nightly-test`** label to your PR.
Adding the label is what actually **triggers** the workflow — at that point the workflow
reads the existing comments to find the `/nightly` command.

:::{note}
Only repository **Contributors** (Triage role) and **Maintainers** (Write role) can add
labels. If you do not have this permission, ask a maintainer to add the label for you.
You can find the list of maintainers and contributors in the project's
[Governance](../../community/governance.md) page or by checking the
[CODEOWNERS](https://github.com/vllm-project/vllm-ascend/blob/main/.github/CODEOWNERS)
file.
:::

:::{important}
The comment must be posted **before** the label is added. If you add the label first,
the workflow will find no `/nightly` comment and skip all tests.
:::

### 3. Wait for results

GitHub Actions will trigger the `Nightly-A2` or `Nightly-A3` workflow. Only tests
matching the filter will be dispatched, which saves hardware resources.

## Differences Between PR and Scheduled Runs

| | Scheduled / Manual Dispatch | PR-triggered |
|-|----------------------------|-|
| Trigger | Cron (daily) or `workflow_dispatch` | Label `nightly-test` + `/nightly` comment |
| Code tested | Pre-built nightly image | Your PR's HEAD commit (source installed fresh) |
| Test scope | All tests | Configurable via `/nightly <names>` |
| vLLM + vllm-ascend | From image | Checked out and installed from source |

When a PR run is detected (`is_pr_test: true`), the workflow additionally:

1. Uninstalls any existing vllm packages in the container.
2. Checks out the specific vllm version and your PR's vllm-ascend commit from source.
3. Installs all dependencies from source.
4. Installs the `aisbench` benchmark suite.

## Available Test Names

The test names you can pass to `/nightly` correspond to the `name` fields in the
workflow matrix.

### A2 workflow (`.github/workflows/schedule_nightly_test_a2.yaml`)

**Single-node tests**:

| Test name | Description |
|-----------|-------------|
| `test_custom_op` | Custom operator tests (single card) |
| `test_custom_op_multi_card` | Custom operator tests (multi card) |
| `qwen3-32b` | Qwen3-32B model test |
| `qwen3-next-80b-a3b-instruct` | Qwen3-Next-80B-A3B-Instruct model test |
| `qwen3-32b-int8` | Qwen3-32B INT8 quantization test |
| `accuracy-group-1` | Accuracy tests: Qwen3-VL-8B, Qwen3-8B, Qwen2-Audio-7B, etc. |
| `accuracy-group-2` | Accuracy tests: ERNIE-4.5, InternVL3_5-8B, Molmo-7B, Llama-3.2-3B, etc. |
| `accuracy-group-3` | Accuracy tests: Qwen3-30B-A3B, Qwen3-VL-30B-A3B, etc. |
| `accuracy-group-4` | Accuracy tests: Qwen3-Next-80B-A3B, Qwen3-Omni-30B-A3B, etc. |

**Multi-node tests**:

| Test name | Description |
|-----------|-------------|
| `multi-node-deepseek-dp` | DeepSeek-R1-W8A8, 2-node DP |
| `multi-node-qwen3-235b-dp` | Qwen3-235B-A22B, 2-node DP |

:::{note}
The `doc-test` job in the A2 workflow only runs on `schedule` or `workflow_dispatch`
events — it will **not** run on PR-triggered runs even with `/nightly all`.
:::

### A3 workflow (`.github/workflows/schedule_nightly_test_a3.yaml`)

**Multi-node tests** (run first, single-node tests wait for these to complete):

| Test name | Description |
|-----------|-------------|
| `multi-node-deepseek-pd` | DeepSeek-V3, 2-node PD disaggregation |
| `multi-node-qwen3-dp` | Qwen3-235B-A22B, 2-node DP |
| `multi-node-qwenw8a8-2node` | Qwen3-235B-W8A8, 2-node |
| `multi-node-qwenw8a8-2node-eplb` | Qwen3-235B-W8A8 with EPLB, 2-node |
| `multi-node-dpsk3.2-2node` | DeepSeek-V3.2-W8A8, 2-node |
| `multi-node-qwen3-dp-mooncake-layerwise` | Qwen3-235B-A22B with Mooncake layerwise, 2-node |
| `multi-node-deepseek-r1-w8a8-longseq` | DeepSeek-R1-W8A8 long sequence, 2-node |
| `multi-node-qwenw8a8-2node-longseq` | Qwen3-235B-W8A8 long sequence, 2-node |
| `multi-node-deepseek-V3_2-W8A8-cp` | DeepSeek-V3.2-W8A8 context parallel, 2-node |
| `multi-node-qwen-disagg-pd` | Qwen3-235B disaggregated PD, 2-node |
| `multi-node-qwen-vl-disagg-pd` | Qwen3-VL-235B disaggregated PD, 2-node |
| `multi-node-kimi-k2-instruct-w8a8` | Kimi-K2-Instruct-W8A8, 2-node |
| `multi-node-deepseek-v3.1` | DeepSeek-V3.1-BF16, 2-node |
| `multi-node-deepseek-v3.2-W8A8-EP` | DeepSeek-V3.2-W8A8 with EP, 4-node |

**Single-node tests** (run after multi-node tests complete):

| Test name | Description |
|-----------|-------------|
| `qwen3-30b-acc` | Qwen3-30B accuracy test |
| `deepseek-r1-0528-w8a8` | DeepSeek-R1-0528-W8A8 |
| `deepseek-r1-w8a8-hbm` | DeepSeek-R1-W8A8 HBM |
| `deepseek-v3-2-w8a8` | DeepSeek-V3.2-W8A8 |
| `glm-5-w4a8` | GLM-5-W4A8 |
| `glm-4.7-w8a8` | GLM-4.7-W8A8 |
| `kimi-k2-thinking` | Kimi-K2-Thinking |
| `kimi-k2.5` | Kimi-K2.5 |
| `minimax-m2-5` | MiniMax-M2.5 |
| `mtpx-deepseek-r1-0528-w8a8` | MTP-X + DeepSeek-R1-0528-W8A8 |
| `qwen3-235b-a22b-w8a8` | Qwen3-235B-A22B-W8A8 |
| `qwen3-30b-a3b-w8a8` | Qwen3-30B-A3B-W8A8 |
| `qwen3-next-80b-a3b-instruct-w8a8` | Qwen3-Next-80B-A3B-Instruct-W8A8 |
| `qwq-32b` | QwQ-32B |
| `qwen3-32b-int8` | Qwen3-32B-Int8 |
| `qwen2-5-vl-7b` | Qwen2.5-VL-7B-Instruct |
| `qwen2-5-vl-7b-epd` | Qwen2.5-VL-7B-Instruct EPD |
| `qwen2-5-vl-32b` | Qwen2.5-VL-32B-Instruct |
| `qwen3-32b-int8-a3-feature-stack3` | Qwen3-32B-Int8 feature stack3 |
| `qwen3-32b-int8-prefix-cache` | Qwen3-32B-Int8 prefix cache |
| `deepseek-r1-0528-w8a8-prefix-cache` | DeepSeek-R1-0528-W8A8 prefix cache |
| `custom-multi-ops` | Custom multi-card operator tests |

:::{warning}
The A3 resource pool has a maximum concurrency of **5×16 NPUs**. Multi-node tests
run with `max-parallel: 2` to avoid resource exhaustion. Running `/nightly all` on
A3 will queue a large number of jobs — prefer targeting specific test names when
possible.
:::

## Examples

Run all available nightly tests against your PR:

```text
/nightly
```

Run only the custom operator single-card test:

```text
/nightly test_custom_op
```

Run two specific tests at once:

```text
/nightly test_custom_op qwen3-32b
```

Re-trigger after fixing an issue: just push a new commit. The `synchronize` event
re-runs the workflow and picks up the existing `/nightly` comment automatically — no
need to post a new comment.

## Troubleshooting

**The workflow didn't start after I added the label.**

- Make sure the `/nightly` comment was posted **before** the label was added.
  If the label was added first, remove it and re-add it after posting the comment.
- Check that the comment starts exactly with `/nightly` with no leading spaces or
  extra characters before the slash.
- To re-trigger after fixing an issue, simply push a new commit — the workflow will
  reuse the existing `/nightly` comment automatically.

**Only some tests ran, not the ones I expected.**

- Test names are case-sensitive and must match the `name` field in the workflow matrix
  exactly (see the table above).
- Check the `parse-trigger` job output in GitHub Actions for the resolved `test_filter`
  value.

**The workflow ran with the scheduled image, not my PR code.**

- Confirm the workflow was triggered by a `pull_request` event (label or push), not
  `workflow_dispatch`.
- The `parse-trigger` job logs show `is_pr_event` — check its value.
