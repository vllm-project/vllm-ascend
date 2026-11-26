# EPLB Deployment Implementation: Local Validation

Validation date: 2026-11-09. Base: vLLM Ascend main `3725f6b36`.
The repository selects vLLM main `24420b7a341af37a9bc3186e6fc6062512e0aa58`
and release `v0.32.0`.

## Executed checks

The local host is Windows, with Python 3.14 and CPU torch 2.15 in the isolated
`.venv-eplb` environment. This environment is not a production dependency change.
No supported vLLM installation, Ascend driver or NPU is available on this host.

| Check | Observed result |
| --- | --- |
| Deployment-specific CPU tests | 27 passed, 25 subtests passed |
| Two-process CPU Gloo transfer and export | Passed as part of the CPU tests |
| Ruff check and format, changed files | Passed |
| Codespell and Markdown lint, changed files | Passed |
| Python package init, forbidden imports, boolean context managers, long functions and symbolic meta checks | Passed |
| Release-note draft YAML parsing | Passed |
| Python AST parsing of changed files | Passed |
| Git diff whitespace check | Passed |

For the Windows CPU run, a local bootstrap bypasses unavailable package/device
initialization. The deployment modules and Ascend state/controller methods
under test are the real repository code. `compute_logical_maps`,
`TransferMetadata`, `move_from_buffer`, and the upstream state classes are
loaded from the exact selected vLLM source. The NPU and vLLM process-boundary
imports are stubbed. The Gloo test actually starts two processes and exchanges
CPU weights through torch distributed, but uses a test CPU P2P communicator,
not Ascend's NZ CPU-staging or HIXL implementation.

The bootstrap and downloaded upstream reference files are local ignored
artifacts under `.venv-eplb`, not changes to production dependencies. To repeat
this host-specific run from the repository root:

```powershell
.venv-eplb/Scripts/python.exe .venv-eplb/run_placement_tests.py
```

`bash format.sh ci` was attempted. Initial invocation used a Bash environment
without the virtualenv's pre-commit executable. With Git Bash and the correct
PATH, the full run progressed through hook initialization but remained in the
actionlint Go environment installation and was stopped. The relevant changed-file
hooks listed above were subsequently run with an isolated local configuration;
the repository's `.pre-commit-config.yaml` was not changed. This is not a passed
full `format.sh ci` run.

The pre-existing user change in `eplb/adaptor/vllm_adaptor.py` was excluded
from focused formatting and its file hash was checked against the saved copy.

## Required before merge

- Run the configuration tests in `test_ascend_config.py` and `test_platform.py`,
  the existing EPLB controller/state regressions, and the new deployment tests
  in a supported Linux vLLM Ascend CPU-UT environment.
- Complete `bash format.sh ci` in the normal repository lint environment.
- Run the four-card `test_mrv2_eplb_placement.py` cases. The test requires
  calibration to change at least one layer before restarting statically.
  It covers explicit Gloo and automatic selection; if auto selects Gloo,
  separate validation on an HIXL-capable deployment is still needed.
- Validate actual NPU weights/scales, graph capture/replay and answers on the
  affected quantization/hardware combinations, including repeated dynamic
  rearrangements before export.
- Measure disabled/static/dynamic throughput, TTFT, TPOT, tail latency,
  initialization cost and memory with the same checkpoint and traffic.
- Replace the release-note template PR placeholders when actual PRs are opened.

No NPU correctness, HIXL/NZ correctness, supported-vLLM integration pass, or
end-to-end performance gain is claimed by the local CPU evidence.
