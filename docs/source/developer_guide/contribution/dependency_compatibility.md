# Qualifying third-party dependency candidates

The **Dependency compatibility probe** workflow compares a baseline and a
candidate in separate clean Linux x86_64 environments. It is package-agnostic:
select a dependency source, optionally select direct packages from that source,
and supply candidate requirements as a JSON array. Python 3.10, 3.11 and 3.12
are available for manual runs.

## Sources and candidate semantics

Supported sources are `requirements.txt`, `requirements-dev.txt`,
`requirements-lint.txt`, and `pyproject.toml`'s `[build-system].requires`.
Requirements includes are resolved relative to their containing file; cycles
and paths outside the repository fail. Unsupported pip options also fail
explicitly, so the probe does not silently drop constraints or index settings.
The workflow configures the public PyPI, Ascend and PyTorch CPU indexes.

`packages_json` selects plain distribution names from the source; `[]` selects
its entire declaration set. Both profiles include their normally resolved
transitive dependencies. Selecting only a subset does not validate coexistence
with the packages excluded from that source.

`candidates_json` contains PEP 508 version requirements. An entry replaces
matching selected direct declarations in the temporary candidate file, or adds
a package absent from the baseline. Existing markers and extras must be
preserved. A single candidate cannot replace multiple distinct marker/extra
contexts for the same name; use a separately scoped experiment. Use one entry
per distribution; duplicate candidate names, direct
URLs and pip options are rejected. The repository files are never edited.

This replacement is an explicit hypothetical direct-dependency experiment.
It does not rewrite a third-party Wheel's `Requires-Dist`: transitive exact
pins remain enforced. For example, adding `numpy==2.0.2` to a selected
Triton-Ascend 3.2.2 baseline must still fail if that Wheel requires NumPy 1.26.4.

## Run and inspect a comparison

Choose the workflow in GitHub Actions and supply the source, selected names
and candidates. For example:

| Experiment | Source | packages_json | candidates_json |
| --- | --- | --- | --- |
| Packaging candidate | requirements.txt | `["packaging"]` | `["packaging==24.2"]` |
| Resource library candidate | requirements.txt | `["psutil"]` | `["psutil==6.0.0"]` |
| Complete lint environment | requirements-lint.txt | `[]` | `["mypy==1.11.1"]` |
| Build dependency subset | pyproject.toml | `["packaging"]` | `["packaging>=24.2,<27"]` |
| Triton transitive conflict | requirements.txt | `["triton-ascend"]` | `["numpy==2.0.2"]` |

The generic probe runs binary-only resolution and installation, then
`pip check`, for both profiles. A failed baseline does not prevent collecting
candidate diagnostics; either failure fails the job. The default PR probe
selects `packaging` and tests `packaging>=24.2,<27`, keeping unrelated CI and
production dependencies unchanged. These are experiment inputs, not new
production constraints.

Artifacts include the temporary declarations, `plan.json`, resolution and
installation reports/logs, installed metadata, Python/platform/revision context,
and `comparison.json`. The comparison uses the installed graph, including
bootstrap packages that may not appear in pip's installation report. It reports
added, removed and version-changed distributions only when both profiles
finish successfully. Otherwise its status is `incomplete`; partial installed
metadata is diagnostic evidence, not a passing candidate environment. The
comparison command returns a nonzero exit code for incomplete results.

A failed resolver may not produce a JSON resolution report; inspect its log.
The probe does not fall back to source builds when a binary Wheel is unavailable.
This documents binary availability, not a claim that every source build fails.

## Optional runtime checks and limits

Set `smoke` to `numpy-pyarrow` to run NumPy/Arrow array conversions and a
single-file Parquet roundtrip in the candidate environment. Add those packages
explicitly to its graph, for example by selecting `triton-ascend` and supplying
`["numpy==1.26.4", "pyarrow==16.1.0"]`. Missing imports fail rather than skip.
The smoke module is opt-in and is not collected by ordinary pytest discovery.

For local preparation and graph comparison, install the tooling dependency
`packaging` (also `tomli` on Python 3.10), then use a fresh output directory:

```bash
python -m tools.dependency_compatibility prepare \
  --source requirements.txt --packages-json '["psutil"]' \
  --candidates-json '["psutil==6.0.0"]' --output compatibility-results
# Resolve/install each generated file in its own clean environment.
# Save pip inspect output to <output>/<profile>/installed.json and create
# <output>/<profile>/success only after normal install, pip check and tests pass.
python -m tools.dependency_compatibility compare --output compatibility-results
```

The generic result verifies declared dependency compatibility and binary
installation. Optional smoke tests cover only their named behavior. Selecting
the build source does not execute PEP 517 isolation, CMake, or ACLNN builds;
selecting the runtime source does not install the vLLM-Ascend package itself.
Full-stack verification also requires the matched upstream vLLM installation,
CANN, torch-npu/native imports, NPU kernels and the supported hardware profiles.
Passing a candidate range does not prove every version in the range compatible
or authorize relaxing production pins. Keep exact tested versions in reports
and qualify native/ABI-sensitive changes separately.

See [the third-party compatibility task](https://github.com/vllm-project/vllm-ascend/issues/18802)
for remaining version-range and upstream metadata work.
