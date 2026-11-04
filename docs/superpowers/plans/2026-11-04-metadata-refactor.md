# AscendStore metadata implementation plan

> **For agentic workers:** Execute inline using executing-plans. The user asked
> to proceed through all three stages in order. Request one fresh review at end.

**Goal:** Reduce AscendStore metadata coupling without changing cache behavior.

**Architecture:** Normalize group access, extract the token/addressing service,
then separate save decisions and worker preparation from scheduling payloads.

**Tech Stack:** Python dataclasses, NumPy, CPU PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-04-metadata-refactor-design.md`

## Global Constraints

- Preserve keys, transfer ordering, snapshot and speculative-token semantics.
- Preserve mask bounds and the group-0 compatibility path.
- Keep the three token-length fields distinct.
- No configuration, dependency, model-runner, or protocol changes.
- Signed-off Conventional Commits; no push or PR is requested.
- Follow AGENTS.md: run affected tests, broadening only for identified risks.

## Review Focus

- A suffix block table must retain its absolute token coordinates.
- Short lookup masks and short transfer masks have different tail behavior.
- Per-layer descriptor selection must retain broadcast lengths and strides.
- Snapshot identity must survive speculative rejection and request reuse.
- A backend-adjusted save range must not change the scheduled token length.

### Task 1: Normalize request group access and remove dead arguments

**Files:** `metadata.py`, its scheduler and transfer consumers under
`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/`, and
`tests/ut/distributed/ascend_store/test_metadata.py`.

**Interfaces:** Produce `ReqMeta.get_block_ids_array(group_id)` and
`ReqMeta.get_transfer_arrays(group_id, is_save=True)` with existing grouped-first,
group-0 fallback semantics. Keep all constructor fields declared consistently.

- [ ] Establish the existing affected CPU-test baseline.
- [ ] Test literal group selection and load/save array selection before changes.
- [ ] Implement accessors and migrate repeated selection branches.
- [ ] Remove unused parameters and their callers; declare dynamic payload fields.
- [ ] Run affected tests and Ruff; inspect the diff.
- [ ] Commit with sign-off and record validation in the ledger.

### Task 2: Consolidate layouts and segmented address calculation

**Files:** Create `token_database.py` and `cache_layout.py` in the AscendStore
package; update metadata imports and affected address-generation tests.

**Interfaces:** `GroupBufferLayout` owns the descriptors for one cache group.
`ChunkedTokenDatabase` retains its existing public methods and key outputs.
Bulk and layer addressing share one implementation over selected descriptors.

- [ ] Add tests for selecting a layer in a nonzero group and for segmented partial
  ranges, with hand-derived addresses and sizes.
- [ ] Extract the database, preserving a compatibility import in metadata.
- [ ] Normalize descriptor lengths/strides and layer boundaries at registration.
- [ ] Share address calculation and preserve the existing group-0 layer default.
- [ ] Run metadata, segmented-copy, worker, hybrid, layerwise, and PP tests.
- [ ] Commit with sign-off and record validation in the ledger.

### Task 3: Separate save decisions and worker preparation state

**Files:** Create `request_state.py` and `transfer_metadata.py`; update metadata,
scheduler, worker, backend, and transfer consumers and their affected fixtures.

**Interfaces:** A pure save-decision function returns immutable decisions.
Tracker state changes remain at the current scheduling point. A per-request
transfer state owns prepared arrays, keys, masks, and offsets. Existing metadata
entry points remain available for callers while production uses explicit state.

- [ ] Test decision computation without tracker mutation, hash-boundary saves,
  independent worker state, and group access after preparation.
- [ ] Extract decision computation and make tracker application explicit.
- [ ] Move worker-prepared fields into transfer state and migrate consumers.
- [ ] Keep group-0 aliases and constructor compatibility only where still needed.
- [ ] Run the complete affected CPU suite and formatting checks.
- [ ] Commit with sign-off, request a fresh whole-change review, and fix material
  findings with focused regression tests before final validation.
