# vLLM upstream interface compatibility

This directory is collected by vLLM's existing Ascend NPU job. In addition to the hardware sampler smoke test,
`test_upstream_interface_compatibility.py` performs a source-only compatibility check between the checked-out vLLM PR
and the vllm-ascend revision installed by that job. The analysis does not import either project and does not require NPU
execution.

## Overall flow

1. Detect the vLLM source checkout at `/workspace/vllm`.
2. Fetch upstream `main` and calculate the exact `merge-base -> HEAD` PR range.
3. Record the current vllm-ascend Git revision.
4. Run `tools.vllm_interface_contracts` with the `vllm-interface` analysis plan.
5. Print the executed capability states, phase timings, and generated Markdown summary to the pytest job log.
6. Fail the pytest case only when the analyzer reports an introduced break or cannot complete a valid analysis.

## Analysis phases

### Input verification

The analyzer verifies that the vLLM checkout is at the selected new SHA, the old SHA is an ancestor of the new SHA, and
the vllm-ascend checkout matches the recorded baseline SHA. Missing Git metadata or an invalid range is an analysis
failure rather than a compatibility result.

### Dependency discovery

The `vllm-interface` plan reads vllm-ascend first and discovers direct imports, verified overrides, and exact downstream
calls to vLLM. Inheritance and C3 MRO are used only to prove override ownership. Monkey patches, inheritance-only
findings, and broad generator reviews are intentionally outside this upstream PR plan.

### Old/new contract comparison

Each proven dependency is resolved independently against the old and new vLLM snapshots. The analyzer compares symbol
presence, callable argument binding, constrained return use, and replacement return contracts. A finding is actionable
only when it is newly introduced by the selected PR range and the downstream relationship is statically proven.

### Classification and result

New incompatibilities are reported as introduced breaks. Historical incompatibilities are not attributed to the PR,
and ambiguous bindings remain review or unresolved evidence. The pytest entry uses `--fail-on introduced`, so an
introduced break fails this test while a valid report with no introduced break passes.

### Current CI presentation

The existing upstream job renders the Markdown summary in its pytest log. It does not yet upload the JSON, CSV,
Markdown, or metadata files as Buildkite artifacts and does not create a separate Buildkite annotation. The upstream
Ascend NPU job is currently soft-fail, so this integration provides early awareness rather than a required merge gate.
The analysis itself is CPU-only, but its first upstream run must also confirm that the combined image-build, analysis,
and sampler duration fits the existing job timeout.

## Local commands

Run the source-only unit tests without NPU hardware:

```bash
pytest -q tests/ut/tools/vllm_interface_contracts
```

Running the E2E entry outside the upstream vLLM NPU image skips it because `/workspace/vllm` is not present:

```bash
pytest -q tests/e2e/vllm_interface/test_upstream_interface_compatibility.py
```
