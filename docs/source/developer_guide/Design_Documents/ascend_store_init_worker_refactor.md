# AscendStore initialization and worker separation plan

Status: draft plan; implementation and runtime validation are pending.

This work follows [#20909](https://github.com/vllm-project/vllm-ascend/pull/20909),
which shared scheduler backend resolution, and
[#20348](https://github.com/vllm-project/vllm-ascend/pull/20348), which separated
the whole-block and layerwise scheduler responsibilities.

## Shared immutable initialization configuration

- [ ] Extract common derivation of immutable configuration objects used during
  scheduler and worker initialization. Cover transfer permissions, model/cache
  properties, cache-group block sizes, hash granularity, transfer granularity,
  and backend capabilities where the semantics are shared.
- [ ] Preserve configuration defaults, validation, and mode-specific behavior.
  Resolve existing differences explicitly before sharing a calculation.
- [ ] Let each role create its own runtime resources and mutable state from the
  derived configuration. Coordinator instances, clients, tensors, threads, locks,
  and request/session tracking remain owned by their respective role.

The scheduler and worker run in separate processes; sharing derivation means
using the same calculation, not sharing a live configuration instance across
processes. Immutable objects must not expose mutable nested collections.

## Separate worker responsibilities

- [ ] Follow the scheduler separation in #20348 by introducing a common worker
  base and making the whole-block and layerwise workers sibling implementations:

  ```text
  BaseKVPoolWorker
  ├── KVPoolWorker
  └── LayerwiseKVPoolWorker
  ```

- [ ] Put genuinely shared cache registration, backend ownership, reporting,
  and lifecycle contracts in the base. Keep whole-block transfer scheduling,
  lookup service behavior, and TP-mismatch transfer handling in the whole-block
  implementation.
- [ ] Keep per-layer preparation, hooks, prefetch, sessions/leases, GVA handling,
  and transfer fences in the layerwise implementation.
- [ ] Update factory return types and test mocks for the new ownership. Avoid
  making the base constructor orchestrate layerwise-only initialization.

## Validation before requesting review

- [ ] Add focused tests for configuration defaults, supported modes, boundary
  values, invalid settings, and equivalent scheduler/worker derivation.
- [ ] Run affected AscendStore unit tests for both transfer modes, including
  registration, completion, load failures, preemption, reset, and shutdown.
- [ ] Validate affected whole-block and layerwise integration paths on Ascend
  hardware, including applicable hybrid caches and parallel configurations.
- [ ] Run `bash format.sh ci`, inspect formatter changes, and report commands,
  results, skips, and remaining hardware coverage in the PR description.

This is intended as an internal refactor preserving existing user-facing
behavior. No release note fragment is expected unless that scope changes.
