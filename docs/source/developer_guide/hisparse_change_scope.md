# HiSparse change-scope audit

This feature PR is a platform-backend implementation, not the complete experimental
performance branch. It also includes the measured CPU page-descriptor optimization;
see [performance decisions](hisparse_performance_decisions.md).
Rebased onto Ascend main `24a76e84394ae3427195e3715fc08e90769d398c`.
The paired common vLLM HiSparse interface remains a prerequisite.
Historical integration runs do not qualify the rebased head.

The retained local experimental sources are also available as a non-installed
[experimental review snapshot](../../../benchmarks/hisparse/experimental/20261031/README.md).
Its manifest lists every retained source path and hash, plus original hashes for
files edited during cleanup. The exclusions below refer to the
active backend, not absence from that review snapshot. Frozen-source lint/test
exclusions preserve source bytes and prevent unrelated historical tests from
being collected as current-head tests; they do not establish a validation pass.

The unused custom BF16 Attention implementation, native registration and its
dedicated tests were removed from the snapshot on 2026-11-02. The active backend
already uses the existing SFA operator. Archived BF16 dispatch now also uses its
existing SFA bridge; C8 dispatch and shared layout helper definitions are unchanged.
The retained snapshot has 150 payload files. This cleanup changes no installed
backend or existing Attention kernel and makes no new performance claim.

## Operator boundary

The existing CANN attention, LightningIndexer, MoE and quantization kernels are
not changed by this PR. `sfa_hisparse.py` adapts HiSparse's physical row addresses
and cache layout to the existing SFA implementation; it is not a replacement
attention kernel.

Two new operator modules implement the missing NPU residency and payload-layout
operations. Upstream CUDA kernels cannot execute on Ascend. This establishes a
need for an NPU implementation, not a proof that this particular implementation
or its number of launches is optimal:

- `ops/hisparse.py`: mapped Host ownership, geometry checks and NPU dispatch.
- `ops/triton/hisparse.py`: Resident/Hot lookup, replacement and LRU, miss gather,
  invalidation, selected-row packing and packed-KV scatter. These operate on
  HiSparse metadata/payloads, retaining the upstream lifecycle.

The rebase does not change these two modules' algorithms relative to the previous
PR head `d8a268d94641b232bf42e0b34c96c2938a3e9fea`. Their necessity is at the
backend-operation level; kernel-by-kernel performance necessity remains unproven.
No claim is made that zero misses imply zero overhead or a TPOT regression below 3%.

## Per-file decisions

All paths below are relative to the repository root. Keep means required by this
implementation's feature or tests, not that no smaller design is possible.

| File | Decision and reason |
| --- | --- |
| `docs/source/user_guide/feature_guide/hisparse.md` | Keep: configuration, restrictions and evidence limits. |
| `mkdocs.yml` | Keep: exposes the user guide in navigation. |
| `docs/source/developer_guide/hisparse_change_scope.md` | Review evidence only; no runtime cost. |
| `release_notes/fragments/19109.yaml` | Keep: required release fragment for a user-facing feature. |
| `tests/e2e/pull_request/one_card/test_hisparse_attention.py` | Keep: existing attention path with physical Resident/Hot rows. |
| `tests/e2e/pull_request/one_card/test_hisparse_kv_preprocess.py` | Keep: packed-KV and selected-row payload regressions. |
| `tests/e2e/pull_request/one_card/test_hisparse_ops.py` | Keep: resolver/gather/LRU/invalidation and graph replay boundaries. |
| `tests/e2e/pull_request/one_card/test_hisparse_prefill_staging.py` | Keep: common Prefill staging and opaque-byte gather contract. |
| `tests/e2e/pull_request/one_card/test_hisparse_selection_order.py` | Keep: selected logical rows retain attention order. |
| `tests/ut/distributed/kv_transfer/test_hisparse.py` | Keep: backend signatures, ownership, aliases and unsupported-mode rejection. |
| `tests/ut/distributed/kv_transfer/test_hisparse_page_descriptors.py` | Keep: CPU descriptor byte/order/lifetime guards, fallback and cold installer coverage. |
| `tests/ut/worker/test_worker_v1.py` | Keep: HiSparse shared-pool memory accounting bypass regression. |
| `vllm_ascend/attention/indexer.py` | Keep: metadata-only page/slot stride translation for the shared allocation; no Indexer compute kernel change. |
| `vllm_ascend/attention/sfa_hisparse.py` | Keep: backend integration and layout bridge; current feature rejects attention C8. |
| `vllm_ascend/attention/sfa_v1.py` | Keep: select HiSparse classes only when enabled and avoid an unnecessary exact CPU length copy. |
| `vllm_ascend/distributed/kv_transfer/hisparse/__init__.py` | Keep: exports the platform backend. |
| `vllm_ascend/distributed/kv_transfer/hisparse/backend.py` | Keep: common-runtime backend contract for NPU streams, events, memory and payload operations. |
| `vllm_ascend/distributed/kv_transfer/hisparse/binding.py` | Keep: declare Indexer alignment before allocation and create same-storage views inside common rollback. |
| `vllm_ascend/distributed/kv_transfer/hisparse/page_dma_descriptors.py` | Include measured CPU vectorization; no DMA/kernel/event changes. See the performance decision record. |
| `vllm_ascend/ops/hisparse.py` | Keep as backend primitives; implementation/launch-count optimality is not established. |
| `vllm_ascend/ops/triton/hisparse.py` | Keep as NPU equivalents; optional fusion candidates are excluded. |
| `vllm_ascend/ops/mla.py` | Keep: forwards the common Indexer-group builder; no projection or attention math change. |
| `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py` | Keep: delegate HiSparse roles to the common coordinator rather than an ordinary Ascend hybrid manager. |
| `vllm_ascend/patch/__init__.py` | Keep: document the temporary CPU descriptor patch and removal plan. |
| `vllm_ascend/patch/worker/patch_hisparse_page_descriptors.py` | Include cold NPU-only, source-pinned installer; unknown versions retain original behavior. |
| `vllm_ascend/patch/worker/patch_deepseek_v2.py` | Keep: forwards group ownership from the model into the MLA wrapper. |
| `vllm_ascend/platform.py` | Keep: platform backend selection hook. |
| `vllm_ascend/worker/v2/attn_utils.py` | Keep: common role-aware specs, shared-slab alignment and groups without attention layers. |
| `vllm_ascend/worker/v2/model_runner.py` | Keep only dummy/profile preemption guard; preserve current upstream initialization and JIT warmup helper. |
| `vllm_ascend/worker/worker.py` | Keep: do not apply ordinary per-layer memory correction to the already-accounted common shared slab. |
| `docs/source/developer_guide/hisparse_performance_decisions.md` | Include per-candidate/operator decisions and measured evidence limits; no runtime cost. |

## Intentionally excluded from this upstream feature

The fork's broad integration snapshot is not copied wholesale:

- New AscendC attention source, C8 map/pack/fused resolver variants, grouped
  invalidator experiments, persistent gather and follower query-prepack.
- Mapped AIV Host-writeback, fused Resident updates,
  lazy-MTP/unused-staging wrappers and block-table view reuse candidates.
- Device-URMA P/D prototypes, export/import staging and transport control.
- Independent A5 W4A8/SwiGLU fixes and latest-common API compatibility changes.
- Machine configuration, profiles, model weights, binaries and private evidence.

Those are separate features or candidates, not established prerequisites for this
narrow backend PR. Removing them from the upstream submission does not establish
their absence from the measured integration runtime; its performance results must
not be attributed to this feature head.

## Rebase choices and remaining validation

The current upstream cache-initialization helper and its JIT warmup window were
preserved. The old direct `super().initialize_kv_cache` signature adaptation was
not carried over. Upstream's newer cache-allocation patch documentation was also
preserved; the obsolete HiSparse allocation monkey patch remains removed.
The dummy/profile preemption guard was retained separately.

Local CPU checks exercise real production definitions using import/AST isolation.
They do not execute NPU kernels or qualify framework installation. Normal pytest,
full lint, final-head NPU replay, full-model serving and final dataset accuracy
must be reported independently. No final-head speedup or complete C8/P/D support
is claimed.

Rebase checks on 2026-11-01: 12 selected existing CPU cases passed with the actual
backend class and test functions extracted by AST, a CPU torch implementation,
and substituted NPU event/stream interfaces. Coverage includes opaque bytes and
page gaps, invalid geometry, explicit registration release, shared workspace,
DMA-extension selection, current-stream lookup, and unsupported IPC/conversion.
The external isolation harness is not part of the serving implementation.
Ruff, formatting, Python syntax (22 files) and Git whitespace checks passed.
Normal pytest collection failed because the local common vLLM checkout does not
provide `vllm.v1.core.kv_cache_planning`; that failure is not a test pass.
The full `bash format.sh ci` attempt exceeded its 55-second bound while installing
the actionlint pre-commit environment. Standard CI and hardware checks remain open.
