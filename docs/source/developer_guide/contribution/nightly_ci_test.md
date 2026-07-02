# Nightly CI Test

This document explains how to trigger nightly hardware CI tests against your PR code
on Ascend NPU hardware (A2/A3) via a slash command comment.

## Background

By default, nightly CI tests run on a fixed schedule using pre-built nightly images.
Authorized users can self-service trigger these tests directly against their PR changes
by posting a `/nightly` comment on the PR or issue.

## How to Trigger

Post a comment on the PR or issue:

```text
/nightly <test-names> [--branch <branch>]
```

- **`<test-names>`** is required. Use `all` to run all tests, or specify one or more
  test names separated by spaces or commas.
- **`--branch <branch>`** is optional. Without it, the branch defaults to `main`.
  When commenting on a PR, the workflow always tests against the PR's HEAD commit
  regardless of the branch value.

| Comment | Effect |
|---|---|
| `/nightly all` | Run all nightly tests on both A2 and A3 |
| `/nightly test_custom_op` | Run a single named test |
| `/nightly test_custom_op qwen3-32b` | Run multiple named tests |
| `/nightly all --branch releases/v0.21.0` | Run all tests on a specific branch |

:::{note}
Bare `/nightly` without any test names is not accepted and will result in a bot error.
Use `/nightly all` to run everything.
:::

Once the comment is posted, the workflow is triggered automatically. A 🚀 reaction is
added to your comment to confirm receipt, along with links to the dispatched A2 and A3
workflow runs.

:::{note}
Only users with **triage or higher** repository permission can trigger `/nightly`.
Unlike `/e2e`, the PR author alone is not sufficient. Ask a maintainer to post the
comment if you lack permission.
:::

## Differences Between PR and Scheduled Runs

| | Scheduled / Manual Dispatch | PR-triggered |
|---|---|---|
| Trigger | Cron (daily) or `workflow_dispatch` | `/nightly` comment on PR or issue |
| Code tested | Pre-built nightly image | PR's HEAD commit (installed from source) |
| Test scope | All tests | Configurable via test names or `all` |
| vLLM + vllm-ascend | From image | Checked out and installed from source |

When triggered from a PR, the workflow additionally:

1. Uninstalls any existing vllm packages in the container.
2. Checks out the specified vLLM version and the PR's vllm-ascend commit from source.
3. Installs all dependencies from source.

## Available Test Names

The test names passed to `/nightly` correspond to the `name` fields in the workflow
matrix of `schedule_nightly_test_a2.yaml` and `schedule_nightly_test_a3.yaml`.

### A2 workflow

**Single-node tests**:

| Test name | Description |
|---|---|
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
|---|---|
| `multi-node-deepseek-dp` | DeepSeek-R1-W8A8, 2-node DP |
| `multi-node-qwen3-235b-dp` | Qwen3-235B-A22B, 2-node DP |

:::{note}
The `doc-test` job in the A2 workflow only runs on `schedule` or `workflow_dispatch`
events — it will not run on PR-triggered runs even with `/nightly all`.
:::

### A3 workflow

**Multi-node tests** (run first, single-node tests wait for these to complete):

| Test name | Description |
|---|---|
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
| `multi-node-glm-5.2` | GLM-5.1-W8A8, 2-node |

**Single-node tests** (run after multi-node tests complete):

| Test name | Description |
|---|---|
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
| `qwen3-32b-int8` | Qwen3-32B-Int8 |
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

Run all available nightly tests:

```text
/nightly all
```

Run only the custom operator single-card test:

```text
/nightly test_custom_op
```

Run two specific tests at once:

```text
/nightly test_custom_op qwen3-32b
```

Run tests on a specific release branch:

```text
/nightly all --branch releases/v0.21.0
```

## Troubleshooting

**The bot replied with a permission error.**

- Only users with triage+ repository permission can trigger `/nightly`.
  Ask a maintainer to post the comment instead.

**The bot replied that test names are required.**

- Bare `/nightly` without arguments is not accepted. Use `/nightly all` or specify
  test names explicitly.

**Only some tests ran, not the ones I expected.**

- Test names are case-sensitive and must match the `name` field in the workflow matrix
  exactly (see the tables above).
- Comma-separated and space-separated names are both accepted.

**How to obtain detailed logs for multi-node tests.**

- For most issues, the stdout logs from GitHub Actions are sufficient (these always
  represent logs from the first node).
- If first-node logs are insufficient, download the log archive from the job summary.
  It includes framework-side logs and plog information for each node, structured as:

  ```text
  .
  ├── node0
  │   ├── root/ascend/log
  │   └── var/log/<job>-0_logs.txt
  └── node1
      ├── root/ascend/log
      └── var/log/<job>-0-1_logs.txt
  ```
