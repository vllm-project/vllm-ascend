# AscendStore metadata validation

## Scope and environment

Base commit: `c9a826cd5594599f177719a26671be48c55160a9`.

Validation ran on Windows with Python 3.12 and CPU PyTorch 2.10.0, using an
external virtual environment. No Ascend hardware or installed vLLM runtime was
available. The standalone metadata tests use the existing `_mock_deps` helper.
Broader CPU checks use a local bootstrap that supplements missing upstream leaf
imports from the adjacent vLLM source and mirrors the attention-gate fixture.
Production classes under review are not mocked by that bootstrap.

## Results

- Original metadata baseline: 77 passed.
- Refactored metadata: 84 passed, including seven new group-selection,
  descriptor-addressing, save-decision, and copied-view ownership cases.
- Metadata, transfer, snapshots, real segmented copies, and PP keys: 308 passed.
- Broader runnable scope, including DCP and PP layer-count regressions:
  910 passed, 1 skipped, 40 pre-existing failing cases deselected.
- A full comparison before exclusions retained the same 40 baseline failures
  with no new failures in the AscendStore and PP-key scope.
- Ruff checking and formatting passed for the affected package and tests.
- `git diff --check` passed.

The broader command covers `tests/ut/distributed/ascend_store`,
`tests/ut/kv_transfer/test_layerwise_pp_global_layers.py`,
`tests/ut/kv_transfer/test_layerwise_dcp_sharding.py`, and
`tests/ut/kv_offload/test_pool_worker_pp_layer_count.py`.
`test_request_lifecycle.py` requires the unavailable real vLLM scheduler and is
outside this local bootstrap. The 40 baseline failures involve missing upstream
interfaces, simplified spec mocks, or Windows IPC support; their exact IDs are
preserved in the local validation evidence.

## CPU preparation comparison

The comparison loads the actual metadata implementation from the base commit
and the candidate, checks equal output, and measures repeated CPU calls. These
are local microbenchmarks, not NPU or serving-throughput evidence.

| Metadata blocks | Baseline | Candidate |
| --- | --- | --- |
| 32 | 5.33 microseconds | 5.32 microseconds |
| 512 | 23.91 microseconds | 14.30 microseconds |
| 4096 | 162.22 microseconds | 84.75 microseconds |
| Segmented address preparation | 1.18 microseconds | 1.07 microseconds |

The normal path reuses the group-0 ID array and avoids legacy list-normalization
work. Immutable save decisions use a named tuple to avoid frozen-dataclass
initialization overhead per request.

## Remaining validation

Run the normal Linux CPU-UT environment with real vLLM dependencies, the complete
repository formatting check, and selected Ascend transfer/E2E cases in CI.
