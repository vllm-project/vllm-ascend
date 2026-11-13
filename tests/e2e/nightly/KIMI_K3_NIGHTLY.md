# Kimi-K3 A5 PR nightly trigger

Community `/nightly` dispatches the upstream `main` workflow. PR workflow
changes do not become dispatch defaults before merge. Use the explicit
command below on this PR; maintainers must supply the required test labels.

```text
/nightly Kimi-K3-W4A8-E256-A5-GSM8K-First100 --branch main --aop_enabled --log-path '[{"node0":"log_archives"},{"node0":"outputs"},{"node0":"rescore"},{"node0":"benchmark_results"},{"node0":"/tmp/test-logs"},{"node0":"/tmp/aop-bisect-diagnostics"}]'
```

The command selects one K3 case, 20 questions, batch size 1, five rounds.
AOP can run additional tests for regression bisection after a failure.
The current accuracy tolerance accepts 0%–100%; accuracy changes alone do
not necessarily trigger AOP.

`aisbench-outputs` is an artifact label; its filesystem directory is `outputs`.
The four relative paths resolve below `/vllm-workspace/vllm-ascend/`.
`outputs/nightly_rounds/` retains all five rounds. `rescore` is collected
only when generated; this configuration does not run a rescoring tool.

The PR collector recognizes this exact six-path bundle as optional logs.
Missing directories are reported and skipped even when the upstream workflow
does not pass `--skip-missing`. At least one directory must exist. All other
requests remain strict unless `--skip-missing` is explicitly supplied.
Path validation and credential/symlink exclusions remain enabled.

Find the `nightly-download-*` artifact in the workflow run's Artifacts section;
retention is seven days. The actual upload still depends on the runner
reaching collection and GitHub artifact availability.
