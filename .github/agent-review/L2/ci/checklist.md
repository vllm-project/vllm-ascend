# L2 CI Domain Checklist

Rules for PRs touching `.github/`, CI scripts, and workflow files. Owned by
the CI maintainers (see CODEOWNERS). Entries cite stable IDs (`L2-ci#slug`).

## Workflow authoring

- `L2-ci#self-hosted-bot-jobs`: Bot/orchestration jobs run on the self-hosted
  CPU pool (`linux-amd64-cpu-2-hk` and friends). Never place bot jobs on
  GitHub-hosted runners, and never let them consume NPU machines.
- `L2-ci#container-contract`: Jobs that need tooling must declare a
  `container:` image; do not assume tools exist on the runner host. Images
  come from `.github/workflows/dockerfiles/` built by the `_schedule_*_image_build`
  workflows.
- `L2-ci#comment-body-not-argv`: Untrusted text (PR titles, comment bodies,
  branch names) travels through files or env, never interpolated into shell
  command lines.
- `L2-ci#persist-credentials-false`: `actions/checkout` uses
  `persist-credentials: false` unless the job explicitly needs to push.
- `L2-ci#untrusted-code-isolation`: Workflows triggered by fork PRs must not
  execute PR code in steps that hold credentials; privileged credentials only
  appear in workflows defined by main-branch code.
- `L2-ci#concurrency-group`: Flammable/comment-driven workflows define a
  `concurrency` group keyed by PR/issue number so bursts do not interleave.

## Flakiness and known traps

- `L2-ci#known-flaky`: sleep_wake tests (#19063/#19686) and test_d3
  (accelerate/import-order, #21410) are upstream-known flaky; do not block on
  them, but flag unexplained new failures.
- `L2-ci#main-snapshot-first`: When shared-lane CI fails, first suspect the
  "Rebase on main snapshot" breakage before blaming the PR.

## Test selection

- `L2-ci#select-tests-hints`: New/changed tests should be reachable by the
  select-tests machinery (merge_extra_recommended_tests.py /
  estimated_times.yaml) or CI coverage silently shrinks.
- `L2-ci#e2e-needs-npu`: E2E and performance changes must be validated by
  NPU CI lanes; never claim local NPU results without hardware.
