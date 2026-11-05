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
- Independent review found no critical, important, or minor defects. Its
  read-only comparisons covered 20,000 save scenarios and 10,000 segmented
  bulk/layer address scenarios, with matching baseline results. Pickle,
  deepcopy, and dataclass replacement retained payload values.
- An additional deterministic comparison matched 5,000 save decisions and
  tracker transitions against the original factory.
- Ruff checking and formatting passed for the affected package and tests.
- Affected-file pre-commit hooks passed: Ruff, codespell, typos, Markdown,
  filename/package checks, forbidden imports, boolean context managers, long
  functions, and symbolic meta checks. The logger script passed through Git
  Bash; its pre-commit launcher expects `/bin/bash`, unavailable on Windows.
- `git diff --check` passed.

`bash format.sh ci` was attempted twice. The first setup failed on the npm
certificate chain; using the existing Windows trust roots resolved that issue.
The retry could not finish downloading Go for actionlint (`WinError 10054`).
The complete repository check must therefore still run in supported CI.

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

## Decisions retained

- Transfer IDs and GVAs fall back together. Independent fallback could associate
  one group's block IDs with another group's addresses.
- Hardware, distributed transport/concurrency, and the real upstream scheduler
  remain explicit CI follow-up. CPU evidence cannot establish those contracts.

## Remaining validation

Run the normal Linux CPU-UT environment with real vLLM dependencies, the complete
repository formatting check, and selected Ascend transfer/E2E cases in CI.

## PR preparation on 2026-11-05

Rebased onto upstream `main` at
`75593e4662d7d45984d6a14df19e50fa57d68e7f`. The scheduler refactor merged
upstream required resolving imports in the two pool scheduler modules; its
new base class and transfer-mode behavior were retained.

After rebasing, the broader local CPU command again reported 910 passed,
1 skipped, and 40 deselected. The initial baseline exclusion list was retained.
The local bootstrap additionally loads the actual upstream `KVConnectorOutput`
dataclass: the generic dependency mock discards constructor fields used by the
updated scheduler test. The same failure was reproduced with the unchanged
upstream scheduler before adding that leaf dependency to the local bootstrap.
Ruff checking and formatting passed across 60 affected and adjacent Python files,
and `git diff --check` passed against the updated base.
A third `bash format.sh ci` attempt remained in actionlint dependency setup and
was stopped after 228 seconds without completing the checks.

## Baseline failure IDs

These cases failed before production changes and were excluded from the runnable
local suite. They are not claimed as passing.

```text
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnector::test_shutdown_closes_lookup_server
tests/ut/distributed/ascend_store/test_ascend_store_connector.py::TestAscendStoreConnectorLayerwise::test_mamba_state_copy_runs_after_layer_load
tests/ut/distributed/ascend_store/test_glm53_store.py::TestGLM53Store::test_hash_geometry_and_safe_full_hit
tests/ut/distributed/ascend_store/test_glm53_store.py::TestGLM53Store::test_layerwise_null_kda_slots_never_publish_readable_keys
tests/ut/distributed/ascend_store/test_glm53_store.py::TestGLM53Store::test_memcache_layerwise_round_trip_keeps_each_tp_state
tests/ut/distributed/ascend_store/test_glm53_store.py::TestGLM53Store::test_mooncake_layerwise_excludes_private_tail
tests/ut/distributed/ascend_store/test_glm53_store.py::TestGLM53Store::test_nonlayerwise_round_trip_keeps_each_tp_state_and_private_tail
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_no_reuse_skips_topology_validation
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_base_layers_are_merged_into_shared_slots
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_incompatible_cache_specs_use_separate_slots
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_main_spec_controls_reuse_regardless_of_indexer
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_cache_spec_roles_do_not_depend_on_order
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_single_indexer_spec_is_the_primary_spec
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_multi_main_spec_layer_selects_attn_as_main
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_multi_group_sfa_descriptors_are_merged_by_main_component
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_component_sharing_merges_main_across_a_and_b_layers
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_packed_cache_tensor_descriptors_are_rejected
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_standardized_multi_layer_descriptor_becomes_private_reuse_slots[0]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_standardized_multi_layer_descriptor_becomes_private_reuse_slots[64]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_out_of_bounds_contiguous_descriptor_is_rejected
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_mixed_indexers_share_max_capacity_without_changing_reuse_order[False]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_mixed_indexers_share_max_capacity_without_changing_reuse_order[True]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_reuse_rebuilds_generic_indexer_descriptors_with_physical_capacity[False-1]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_reuse_rebuilds_generic_indexer_descriptors_with_physical_capacity[False-2]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_reuse_rebuilds_generic_indexer_descriptors_with_physical_capacity[True-1]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_reuse_rebuilds_generic_indexer_descriptors_with_physical_capacity[True-2]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_generic_indexer_descriptors_unchanged_without_reuse[True]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_generic_indexer_descriptors_unchanged_without_reuse[False]
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_reuse_rejects_unknown_indexer_stride
tests/ut/distributed/ascend_store/test_layerwise_cache_layout.py::test_reuse_rejects_out_of_bounds_generic_indexer_descriptor
tests/ut/distributed/ascend_store/test_pool_scheduler.py::test_joint_prefix_group_ownership_survives_scheduler_conversion_and_empty_pp_layer_names[False]
tests/ut/distributed/ascend_store/test_pool_scheduler.py::test_joint_prefix_group_ownership_survives_scheduler_conversion_and_empty_pp_layer_names[True]
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestGetZmqRpcPathLookup::test_independent_dp_lookup_servers_bind_distinct_paths
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerUpdateConnectorOutput::test_finished_request_keeps_receive_registration_until_completion
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerUpdateConnectorOutput::test_receive_completion_cleans_only_reported_registrations
tests/ut/distributed/ascend_store/test_pool_scheduler.py::TestKVPoolSchedulerUpdateConnectorOutput::test_receive_completion_preserves_pending_send_events
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPPPoolWorker::test_pcp_registers_persistent_owner_layers_and_mtp
tests/ut/distributed/ascend_store/test_pool_worker.py::TestKVPPPoolWorker::test_registers_persistent_layers_and_mtp
tests/ut/distributed/ascend_store/test_qsa_basic_pooling.py::test_packed_qsa_gdn_pages_restore_disjoint_live_block_ids[memcache]
tests/ut/distributed/ascend_store/test_qsa_basic_pooling.py::test_packed_qsa_gdn_pages_restore_disjoint_live_block_ids[mooncake]
```
