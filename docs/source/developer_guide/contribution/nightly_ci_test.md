# Nightly CI Test

This document explains how the periodic (nightly / weekly) hardware CI works on Ascend
NPU hardware (A2 / A3 / 310P), and how to add or manually trigger a periodic test.

## Background

All periodic hardware tests are driven by a single unified workflow,
[`.github/workflows/schedule_periodic_test.yaml`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/schedule_periodic_test.yaml)
(`Periodic-Test`). It covers four frameworks — **model** (single-node and multi-node),
**accuracy**, and **ops** — using pre-built nightly images.

Cases are **not** hard-coded in the workflow. They are registered as plain path strings
in [`.github/workflows/scripts/schedule_config.yaml`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/scripts/schedule_config.yaml).
The parser
[`parse_schedule_config.py`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/scripts/parse_schedule_config.py)
reads that registry and **infers** the framework, resource size, chip, runner, and route
from each path, then emits one job matrix per framework
(`single_node_matrix`, `multi_node_matrix`, `accuracy_matrix`, `ops_matrix`). Empty
matrices skip their jobs cleanly.

## How It Is Triggered

### Scheduled (automatic)

| Schedule | Cron (UTC) | Local time | `schedule_config.yaml` section |
|----------|------------|------------|--------------------------------|
| Nightly | `45 15 * * *` | 23:45 Beijing, daily | `nightly-main` |
| Weekly | `45 15 * * 1` | 23:45 Beijing, Monday | `weekly-main` |

The cron value selects the matching section in `schedule_config.yaml`, and every path in
that section's `files:` list is run.

### Manual (`workflow_dispatch`)

From the **Actions** tab, choose **Periodic-Test → Run workflow** (requires write access
to the repository). Inputs:

| Input | Default | Purpose |
|-------|---------|---------|
| `vllm_ascend_branch` | `main` | Branch to check out for the test runs (used for `actions/checkout`; a normalized form is used for image tags / artifact names). |
| `schedule_name` | `manual` | Which `schedule_config.yaml` section to run (`nightly-main`, `weekly-main`, or `manual`). |
| `test_filter` | `all` | Narrow the selected cases — see below. |

## Selecting Tests with `test_filter`

`test_filter` narrows the cases within the chosen section. It is matched in this order:

1. `all` — every case in the section.
2. Full path exact match — `tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml`
3. Filename exact match — `GLM5_1-W8A8-EP-external_dp.yaml`
4. Filename stem exact match — `GLM5_1-W8A8-EP-external_dp`
5. Path segment match — e.g. `Qwen`, `accuracy`, `ops`, `one_node`
6. Substring match — falls back to a substring of the case name or path.

Examples: `all`, `DeepSeek`, `Qwen`, `accuracy`, `ops`,
`GLM5_1-W8A8-EP-external_dp`.

Because `image_build_targets` is derived from the selected cases, a filtered manual run
only builds the images it actually needs (e.g. a2-only selection builds the a2 image
only).

## Adding a New Periodic Test

No workflow edits are required. Two steps:

1. Place the case file under `tests/e2e/schedule/` following the layout rules below.
2. Register its path in `schedule_config.yaml` under the relevant section.

Directory layout (framework is always the 4th path segment):

```text
tests/e2e/schedule/model/<Family>/<resource_dir>/*.yaml   -> model framework
tests/e2e/schedule/accuracy/<resource_dir>/<chip>/*.yaml  -> accuracy framework
tests/e2e/schedule/ops/<resource_dir>/*.py                -> ops framework (file or directory)
```

Supported `<resource_dir>` values (English form only; numeric forms such as `1_card`
are rejected): `one_card`, `two_card`, `four_card`, `eight_card`, `one_node`,
`two_node`, `four_node`.

Inference rules applied by the parser:

- **Route**: model `*_card` and `one_node` → single-node; model `two_node` / `four_node`
  → multi-node; accuracy `*_card` → accuracy; ops → ops.
- **Chip**: a separator-bounded `a2` / `A2` token → `a2`; `a3` / `A3` → `a3`;
  `310` / `310p` / `v310` → `310p`; otherwise defaults to `a3`. For accuracy, the chip is
  the explicit `a2` / `a3` / `310p` directory.
- **Runner**: resolved from `(chip, resource)` via
  [`runner_label.json`](https://github.com/vllm-project/vllm-ascend/blob/main/.github/workflows/scripts/runner_label.json)
  — never hard-coded in the workflow.
- **Multi-node type**: a model multi-node filename whose **stem** contains `external_dp`
  routes as external DP; otherwise internal.

Accuracy configs are real executable YAMLs; a directory entry is grouped into one job per
chip (a `config_paths` list). Ops directory entries are grouped by detected chip into one
job per chip.

To validate your registration locally before pushing:

```bash
pytest .github/workflows/scripts/tests/test_parse_schedule_config.py -q

python3 .github/workflows/scripts/parse_schedule_config.py \
  --config .github/workflows/scripts/schedule_config.yaml \
  --runner-label .github/workflows/scripts/runner_label.json \
  --event-name workflow_dispatch --schedule-name manual --test-filter all
```

The dry-run prints the selected cases (framework, route, chip, runner, path) and the
resulting matrices.

## Troubleshooting

**How to obtain more detailed logs to pinpoint problems for multi-node tests**

- For most issues, the stdout pop-up logs from GitHub Actions are sufficient (this log
  always represents the logs from the first node).
- If the logs from the first node are no longer sufficient to provide effective logging
  information, see the summary of your jobs to download the log archive for the
  corresponding test, which includes the framework-side logs and plog information for each
  node, structured as follows:

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
