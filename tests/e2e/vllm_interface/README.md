# vLLM upstream interface compatibility

This directory is collected by vLLM's existing Ascend NPU job. In addition to the hardware sampler smoke test,
`test_upstream_interface_compatibility.py` performs a source-only compatibility check between the checked-out vLLM PR
and the vllm-ascend revision installed by that job. The analysis does not import either project and does not require NPU
execution.

All implementation code for this check is kept in this directory:

```text
tests/e2e/vllm_interface/
├── vllm_interface_contracts/  # source analyzer and CLI
├── test_upstream_interface_compatibility.py
├── singlecard/                # existing NPU sampler test
└── README.md
```

## Overall flow

1. Detect the vLLM source checkout at `/workspace/vllm`.
2. Fetch upstream `main` and calculate the exact `merge-base -> HEAD` PR range.
3. Record the current vllm-ascend Git revision.
4. Run `python -m tests.e2e.vllm_interface.vllm_interface_contracts`; this package has one fixed `vllm-interface`
   analysis scope and does not expose a main2main scenario switch.
5. Parse upstream Python files in bounded batches with a process pool, build the downstream source index, and then run
   relation comparison, direct-import analysis, and direct-call analysis concurrently inside the same job.
6. Render the compatibility summary in memory and print it directly to the pytest job log, together with the selected
   revisions and phase timings. The CI path does not create report files.
7. Fail the pytest case only when the analyzer reports an introduced break or cannot complete a valid analysis.

## Analysis phases

### Input verification

The analyzer verifies that the vLLM checkout is at the selected new SHA, the old SHA is an ancestor of the new SHA, and
the vllm-ascend checkout matches the recorded baseline SHA. Missing Git metadata or an invalid range is an analysis
failure rather than a compatibility result.

### Dependency discovery

The analyzer reads vllm-ascend first and discovers direct imports, verified overrides, and exact downstream
calls to vLLM. Inheritance and C3 MRO are resolved lazily only to prove override ownership. Monkey patches,
inheritance-only findings, and broad generator reviews are intentionally outside this upstream PR check; their
collector and report implementations are not included in this directory.

Known wrapper semantics are matched only after the decorator resolves to an exact canonical symbol. They are not
pinned to whole-repository commits, and this check does not require or build an external PyTorch source index.

### Old/new contract comparison

Each proven dependency is resolved independently against the old and new vLLM snapshots. The analyzer compares symbol
presence, callable argument binding, constrained return use, and replacement return contracts. A finding is actionable
only when it is newly introduced by the selected PR range and the downstream relationship is statically proven.

### In-job parallel analysis

Inheritance/MRO discovery remains ahead of override discovery because override ownership depends on the completed MRO.
After relation generation finishes, the analyzer runs three independent branches in one Python process: relation
comparison, direct-import analysis, and direct-call discovery/comparison. The branches use isolated old/new Git snapshot
caches and their findings are merged in a fixed order before the existing deterministic finding sort. The upstream CI
entry uses three workers. Use `--analysis-workers 1` to reproduce the serial execution path.

### Process-parallel upstream indexing

Each run parses the complete vLLM source tree. Python files are grouped into bounded batches and analyzed with a
`ProcessPoolExecutor`; the upstream CI entry uses four index workers. The parent process merges fragments in sorted
source order and always runs global class-variant, star-import, dataclass, callable-alias, and consistency finalization.
This keeps cross-module results deterministic. Use `--index-workers 1` to use the serial indexing path.

The CI implementation does not write or restore persistent repository-index or file-fragment data. This avoids relying
on state that is not preserved by the job's ephemeral container.

### Classification and result

New incompatibilities are reported as introduced breaks. Historical incompatibilities are not attributed to the PR,
and ambiguous bindings remain review or unresolved evidence. The pytest entry uses `--fail-on introduced`, so an
introduced break fails this test while a valid report with no introduced break passes.

### Current CI presentation

The upstream CI entry renders the same Markdown content previously written to `vllm-interface-pr-summary.md` and sends
it directly to standard output. It does not create JSON, CSV, Markdown, or
metadata report files, upload Buildkite artifacts, or create a separate Buildkite annotation. The selected revisions
and phase timings appear before the summary in separate collapsible log sections. The upstream Ascend NPU job is
currently soft-fail, so this integration provides early awareness rather than a required merge gate. The analysis
itself is CPU-only, but its first upstream run must also confirm that the combined image-build, analysis, and sampler
duration fits the existing job timeout.

## Local commands

Running the E2E entry outside the upstream vLLM NPU image skips it because `/workspace/vllm` is not present:

```bash
pytest -q tests/e2e/vllm_interface/test_upstream_interface_compatibility.py
```

Run an exact range with the same in-job parallel settings used by upstream CI:

```bash
python -m tests.e2e.vllm_interface.vllm_interface_contracts analyze-range \
  --vllm-root /workspace/vllm \
  --ascend-root . \
  --old <old-sha> \
  --new <new-sha> \
  --expect-ascend-sha <ascend-sha> \
  --analysis-workers 3 \
  --index-workers 4
```
