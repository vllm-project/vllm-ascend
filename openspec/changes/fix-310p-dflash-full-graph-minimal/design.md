## Context

See `proposal.md` for motivation and
`specs/310p-dflash-full-graph/spec.md` for normative behavior.  The frozen
implementation baseline is `1a8feb60d1d642c87feccdb9d1aee5d273f7197a`,
which already contains the accepted Piecewise, FULL_DECODE_ONLY, W8A8
de-chunking, and 310P DFlash discard-mask repairs.

Native non-DFlash FULL exists, but DFlash adds a second component and batch
state that the current descriptor-only graph cache does not distinguish.  The
310P GDN attention capability currently resolves native FULL to
FULL_DECODE_ONLY, and the previous experimental worktree found several
additional capture hazards.  Those findings are hypotheses to reproduce, not
code to restore: capture-time host transfers, boolean indexing lowered through
`NonzeroV2`, uninitialized dummy state, descriptor collisions, host-list GDN
metadata, and dense-prefill attention that was not recorded reliably.

## Goals / Non-Goals

**Goals:**

- Implement genuine FULL capture and replay for 310P DFlash target and draft.
- Keep the control plane, graph cache, inputs, diagnostics, and any optional
  operator private to the exact activation path.
- Fix only blockers freshly reproduced on the current baseline.
- Make graph authenticity, numerical correctness, compatibility, stability,
  and performance independently reviewable.

**Non-Goals:**

- Reworking generic ACL graph infrastructure or upstream vLLM.
- Changing Piecewise, FULL_DECODE_ONLY, Eager, or non-DFlash behavior.
- Copying or cherry-picking the abandoned FULL experiment.
- Adding an eager island and describing the result as FULL.
- Pre-approving a custom operator before the admission evidence exists.

## Decisions

### 1. Use an instance-owned FULL controller behind one exact predicate

Create `vllm_ascend/_310p/dflash_full.py` for the immutable activation
predicate, execution-signature classification, strict fallback policy, and an
instance-owned controller.  Construction occurs only in the 310P runner when
all activation inputs are exact matches.  No mutable module globals, import-time
monkey patches, or environment variable controls are introduced.

The current parent dispatcher remains authoritative.  The controller observes
the final parent result, validates it against the parent-owned capture coverage,
and records a closed reason for legitimate non-FULL contexts.  It does not
rebuild or mutate an upstream `BatchDescriptor`.

**Alternative rejected:** extend the existing FDO predicate and controller.
FDO intentionally graphs decode only; teaching it prefill and mixed semantics
would couple two accepted modes and reproduce the scope expansion we are
avoiding.

### 2. Override GDN graph capability only for exact native FULL

The 310P GDN attention subclass can inspect `vllm_config`.  Its capability hook
will return native FULL support only for the exact predicate; every other call
returns the baseline capability unchanged.  The hook is covered by positive
and negative unit tests before hardware capture.

**Alternative rejected:** change the parent GDN backend from `UNIFORM_BATCH` to
`ALWAYS`.  That would alter non-DFlash and other-platform dispatch.

### 3. Use a private qualified graph store instead of changing the shared cache

Create `vllm_ascend/_310p/dflash_full_graph.py`.  A graph key is:

```text
(component, tp_rank, upstream_batch_descriptor, execution_signature)
```

`component` is target or draft.  `execution_signature` is one of prefill,
chunked prefill, mixed, decode, speculative decode, or mixed with speculative
decode.  Each key owns its wrapper, retained tensors, input contract, capture
record, and replay counter.  Startup manifests enumerate keys per rank and
reject duplicates or missing target/draft coverage.

The generic `ACLGraphWrapper` and its descriptor-only dictionary are not
modified.  The private wrapper may compose the generic capture primitive, but
its lookup and lifecycle remain fully isolated.

**Alternative rejected:** add execution signature to the shared ACL graph key.
That changes cache behavior for every graph mode and is unnecessary for this
single feature.

### 4. Bind each entry to persistent, deterministic inputs

Create `vllm_ascend/_310p/dflash_full_inputs.py`. Each qualified entry owns an
independent immutable contract and stable references to explicitly owned
buffers. Runtime preparation reuses the existing runner/proposer persistent
buffers when they already satisfy the contract, applies deterministic padding,
and recursively checks structure, tensor order, shape, dtype, device, relevant
stride, address, bounded view, alias ownership, and version-sensitive metadata
before launch. A new private buffer is allocated only when a focused RED proves
that a required input has no stable existing owner. Capture and replay never
depend on a host value read from a device tensor.

A contract mismatch raises before graph launch and reports key, field, expected
value, and observed value.  It never silently chooses another graph or eager.

**Alternatives rejected:** allocating all inputs again per graph wastes scarce
310P memory, while one mutable global staging arena makes address ownership and
concurrent engines harder to reason about. Explicitly owned existing buffers
plus per-entry contracts provide the narrowest safe boundary.

### 5. Integrate through three narrow 310P hooks

The unconditional implementation surface is limited to:

- `vllm_ascend/_310p/model_runner_310p.py`: construct the controller, pass final
  dispatcher context, and route target capture/replay.
- `vllm_ascend/_310p/ops/gdn_attn_builder_310.py`: exact capability override.
- `vllm_ascend/_310p/spec_decode/llm_base_proposer_310.py`: route draft
  capture/replay only while DFlash proposal executes.

Files such as `attention_mask.py`, `attention_v1.py`, `gdn_310.py`, and
`chunk_gated_delta_rule.py` may be touched only after a fresh focused RED names
that file as the first failing boundary.  Their changes must sit behind the
same controller predicate or be implemented in a new FULL-only helper.

The generic worker, generic proposer, generic ACL graph wrapper, existing FDO
modules, and upstream vLLM remain unchanged.

### 6. Admit a dedicated operator only through a separate evidence gate

Python/plugin-side repair is attempted first.  If it cannot meet capture safety
without host synchronization, variable host lists, shared-path changes, or
numerical divergence, the corresponding OpenSpec task pauses.  Admission
requires all of:

1. A focused RED on the current baseline and exact first failing operation.
2. Evidence explaining why an existing operator or Python transformation is
   insufficient.
3. A fixed tensor-only ABI with shape, dtype, padding, address, and alias rules.
4. A new private operator name under a 310P DFlash FULL-only source/module.
5. Isolated eager-vs-operator numerical tests and repeated ACL graph replay.
6. Negative call-count tests for Eager, Piecewise, FDO, non-DFlash, and non-310P.
7. A new explicit review checkpoint before integration.

The operator is integrated only after its isolated capture/replay output is
stable.  Existing operator names, schemas, registrations, and call sites are
not edited.

**Alternative rejected:** modify the existing shared GDN or attention operator.
Even a correct implementation would broaden risk to accepted modes.

### 7. Use evidence-gated vertical slices

Implementation proceeds in small RED/GREEN commits:

1. Activation, dispatch, and negative isolation tests.
2. Qualified key, manifest, and replay-accounting tests.
3. Persistent input contract tests.
4. A 4B TP1 startup probe that stops at the first fresh capture blocker.
5. One blocker repair and focused hardware rerun per commit.
6. 4B TP1 C1 then C10; 4B TP2; 35B TP2; finally 35B TP4.
7. Frozen Eager, Piecewise, and FDO controls after every production slice.
8. The full formal matrix and AISBench only after all fast gates pass.

The default capture-size set is not hard-coded.  The acceptance command records
user-supplied sizes sufficient to cover decode and the chunked-prefill shapes
exercised by the fixed workloads.  Every required execution signature must show
a real replay; merely capturing decode descriptors does not validate FULL.

### 8. Stop instead of expanding scope

Implementation stops for a new design review if any of these occurs:

- upstream vLLM must change;
- a shared/public operator schema or generic graph cache must change;
- an accepted Piecewise or FDO contract must change;
- the third independent operator family needs modification;
- a dedicated operator fails isolated repeated replay or numerical comparison;
- graph resource limits require disabling or weakening an accepted mode;
- correctness requires a hidden eager island.

## Data Flow

1. The 310P runner evaluates the exact activation predicate once per engine.
2. The parent dispatcher produces its descriptor and final resolved mode.
3. The controller classifies execution state without changing that descriptor.
4. The private graph store resolves the qualified target or draft key.
5. The input manager validates and updates qualified views of owned persistent buffers.
6. Capture or replay runs; a successful replay increments only that entry's
   counter.
7. The manifest and DEBUG evidence expose requested/resolved/runtime modes,
   key, rank, addresses, contract version, and failure classification.

## Risks / Trade-offs

- **Capture coverage can consume excessive graph resources** -> add signatures
  incrementally, serialize capture, record memory per entry, and stop rather
  than weakening another mode.
- **Host-derived control can freeze dynamic values** -> prohibit device-to-host
  reads inside capture and keep dynamic metadata tensor-only.
- **A private operator increases maintenance cost** -> require admission
  evidence, a fixed ABI, isolated replay tests, and a separate review gate.
- **Small benchmark samples are noisy** -> use the fixed request sets, matched
  environment, warmup, and five-percent investigation threshold before judging
  regressions.
- **Asynchronous NPU errors can be attributed late** -> preserve the first
  device error, synchronize only at diagnostic boundaries, and never count a
  replay as successful before launch completion is known.

## Migration Plan

Develop on `fix/310p-dflash-full-graph-minimal` in the independent worktree
created from `1a8feb60`.  Each vertical slice is separately revertible.  The
feature has no migration or default behavior change because it activates only
for an explicit FULL configuration.  Rollback is the removal or revert of the
FULL-only modules and three narrow hooks; accepted baseline modes then execute
exactly as before.
