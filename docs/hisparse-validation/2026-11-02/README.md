# HiSparse operator validation: 2026-11-02

Final runs after Ruff formatting and switching to `vllm.triton_utils` imports:

| Validation | Result | Pytest time |
| --- | --- | --- |
| Ascend A3 / Ascend910_9382 | 145 passed, no failed/skipped/xfail/deselected | 164.16 s |
| CPU Triton interpreter | 140 passed, 8 NPU cases deselected | 23.03 s |

The eight CPU-deselected NPU cases passed in the hardware run. The hardware suite
covers 119 four-operator cases plus 26 existing invalidation cases. These are
ported operator scenarios; the original CUDA/ROCm-gated module, backend/model
integration, KV writeback and row-DMA integration, and performance were not run.

## Reproduce

From the vLLM-Ascend checkout, with the pinned vLLM checkout at `../vllm`:

```bash
bash tools/run_hisparse_ops_tests.sh cpu all
bash tools/run_hisparse_npu_tests.sh PHYSICAL_DEVICE RUNNING_CONTAINER
```

Inspect device occupancy and obtain access before choosing a runtime. The NPU
runner uses only the selected device and bounded subprocesses, preserves the
existing container, and does not broadcast. The CPU runner creates a container
without NPU mappings and cleans it up. See the reproduction documents in `docs/`
for requirements, test-port mapping, failure history, and pending integrations.

## Provenance

- Ascend base: `cae7ff0290dd6c6155f71d7223e0fd4aff95c9e9` plus this change.
- Reference vLLM: `d5301a4f5487db1d7e8d87daf7828225b95b1774`; unchanged.
- Runtime image: `sha256:8725ad72b9db273848f13894bad0e547a67567cccb2f8692b7e3588b2e959982`.
- Python 3.12.13; torch 2.10.0+cpu; torch-npu 2.10.0.post7;
  triton-ascend 3.6.0; pytest 8.4.2; CANN 9.2.0.
- Exact commands, pytest output, JUnit XML, source SHA-256, runtime/image,
  before/after device state, CPU isolation and cleanup are committed here.
- Workspace paths are normalized to `$WORKSPACE` and the borrowed runtime name
  to `$RUNTIME_CONTAINER`. The original artifacts remain in the development
  workspace. Command and cleanup trailing spaces are trimmed. Counts, assertions, timings, and source hash values are unchanged.

## Static checks

Ruff 0.14.0 check and format, Markdownlint 0.45.0, Python syntax, Bash syntax,
staged whitespace, forbidden imports, long-function comments, boolean context
managers, package `__init__.py`, logger and symbolic-meta checks passed for this
change (repository-wide checks where their scripts do not accept filenames).

`bash format.sh ci` was attempted; hook bootstrap failed fetching
`ruff-pre-commit` because of the runtime's self-signed certificate chain. The
complete pre-commit suite (including its secret scanner, typos, codespell and
ShellCheck hooks) has **not passed**. `format-ci.log` preserves that failure;
individual checks above are separate results. CI remains required.
