# Preload regression tests

All 23 preload cases live in `test_preload.py`: `TestPreloadPR` has 15 basic cases
and `TestPreloadNightly` has eight extended cases. `conftest.py` provides fixtures
and CI selection, `helpers.py` manages processes and assertions, and `runner.py`
executes models in fresh subprocesses and returns JSON measurements.

The tests use one NPU and a non-quantized dense text model. The default model is
`Qwen/Qwen3-0.6B`; pass `--preload-model /path/to/checkpoint` to use a local copy.
Run from the repository root with the supported vLLM/CANN/torch_npu environment.
Preload model tests skip when vLLM lacks the required memory accounting or
distributed release interfaces. A skipped case does not establish preload
functionality. Ordinary NZ disk loading does not depend on this interface check.

## PR coverage

File and directory targets select the 15 PR cases by default; the collection hook
in `conftest.py` deselects the nightly class unless it is explicitly requested.

```bash
pytest -sv --durations=0 tests/e2e/pull_request/one_card/model_runner_v2/preload/test_preload.py
```

`TestPreloadPR` covers:

- IPC layouts, repeated consumers, tied parameters and NZ rejection: eight cases.
- BF16 eager zero-copy and copy loading, inference equivalence, model memory
  accounting and producer release: two cases.
- Missing daemon and dtype mismatch, each with strict rejection and explicit
  fallback to disk: four cases.
- Public preload/serve CLI and repeated completions requests: one case.

The consolidated PR suite measured 1281.88 seconds including pytest startup and
teardown on Ascend 910B4 with a locally cached Qwen3-0.6B checkpoint. Its file-level
CI estimate is 1410 seconds, following the repository's 10% allowance and rounding
to the nearest ten seconds.

## Nightly coverage

The A2 nightly configuration selects `TestPreloadNightly` explicitly as
`preload-model-runner-v2`:

```bash
pytest -sv --durations=0 tests/e2e/pull_request/one_card/model_runner_v2/preload/test_preload.py::TestPreloadNightly
```

These eight cases use representative combinations instead of a full Cartesian
product. PR tests already cover BF16 eager loading in both IPC modes.

| Test | dtype | Execution | IPC mode |
| --- | --- | --- | --- |
| Loading | BF16 | graph | zero_copy |
| Loading | BF16 | graph | copy |
| Loading | FP16 | eager | copy |
| Loading | FP16 | graph | zero_copy |
| YaRN cache rebinding | BF16 | eager | zero_copy |
| YaRN cache rebinding | FP16 | graph | copy |

The other two cases cover five repeated zero-copy restarts and ordinary NZ disk
loading. Each preload test has an independent producer; disk baselines are reused
within the module for matching configurations.

To run all 23 cases, request both classes:

```bash
pytest -sv tests/e2e/pull_request/one_card/model_runner_v2/preload/test_preload.py::TestPreloadPR tests/e2e/pull_request/one_card/model_runner_v2/preload/test_preload.py::TestPreloadNightly
```

## Logs and device isolation

Engine and producer logs and JSON results are stored below pytest's temporary
directory; failures report the relevant log path. Shared fixtures clean up
process groups, including workers left behind after their parent exits.

Run on an otherwise idle device. Copy-release and restart-drift checks compare
device free memory and cannot distinguish allocations made by other workloads.
The copy-release check waits up to 30 seconds for driver memory reclamation after
the consumer exits, while still requiring an empty producer cache and at least
80% of exported weight storage to become available.
Preserve the inherited `ASCEND_RT_VISIBLE_DEVICES` selection when invoking pytest.

## Related production bug fix

The migration also includes a preload configuration bug fix: fine-grained TP size
`1` means no sharding and must be accepted; only values greater than `1` enable
unsupported sharding. This fix has a default-configuration regression test in
`tests/ut/model_loader/preload/test_preload.py` and is separate from the test
migration itself.
