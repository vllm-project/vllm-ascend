## Purpose

Define the observable safety, isolation, authenticity, compatibility, and
acceptance contract for running DFlash with genuine `FULL` ACL graphs on
Ascend 310P without changing any other execution path.

## ADDED Requirements

### Requirement: Exact and isolated activation
The system SHALL activate the new behavior only when the platform is Ascend
310P, speculative decoding method is DFlash, the configured graph mode is
`FULL`, and the current runtime graph mode is `FULL`.

#### Scenario: Exact activation succeeds
- **WHEN** all four activation conditions are true
- **THEN** the 310P DFlash FULL controller owns graph policy and evidence for that engine instance

#### Scenario: Any activation condition is false
- **WHEN** at least one activation condition is false
- **THEN** execution SHALL use the behavior present at baseline `1a8feb60` without constructing or consulting the FULL controller

### Requirement: Native FULL dispatch remains authoritative
The system SHALL preserve the final dispatcher decision and SHALL classify
each graph invocation as prefill, chunked prefill, mixed, decode,
speculative decode, or mixed with speculative decode without changing the
parent descriptor.

#### Scenario: Eligible in-range batch
- **WHEN** the parent dispatcher selects `FULL` for an in-range batch
- **THEN** the selected target or draft component SHALL execute the matching FULL graph entry

#### Scenario: Legitimate non-FULL batch
- **WHEN** the parent dispatcher rejects FULL because the batch is outside configured capture coverage or an explicitly unsupported context
- **THEN** the system SHALL record the closed reason and SHALL NOT claim FULL replay for that invocation

#### Scenario: Unexpected in-range fallback
- **WHEN** an otherwise eligible in-range batch resolves to `NONE` or an incompatible mode
- **THEN** strict startup or acceptance execution SHALL fail closed with the requested mode, resolved mode, component, rank, descriptor, and execution signature

### Requirement: Graph entries have qualified identities
The system SHALL identify a FULL graph entry by component, TP rank, upstream
batch descriptor, and execution signature. Entries with different qualified
identities SHALL NOT share mutable entry state, input-contract instances,
capture records, or replay counters. They MAY reference an explicitly owned
runner or proposer buffer only when the alias is recorded, calls are serialized,
and the qualified contract validates its address and bounded view.

#### Scenario: Descriptor is reused by different batch classes
- **WHEN** two invocations have the same upstream batch descriptor but different execution signatures
- **THEN** they SHALL resolve to different graph entries

#### Scenario: Target and draft use the same descriptor
- **WHEN** target and DFlash draft invocations have otherwise identical descriptors
- **THEN** they SHALL resolve to different component graph entries

### Requirement: Graph inputs are persistent and validated
Every captured entry SHALL own a recursive input contract and stable references
to explicitly owned persistent device buffers. Existing runner or proposer
buffers SHALL be reused when they already satisfy the contract; a private
FULL-only buffer SHALL be allocated only for a proven missing stable input.
The contract SHALL cover tensor count, order, shape, dtype, device, stride where
relevant, address stability, bounded view, alias ownership, and
version-sensitive metadata. Runtime values SHALL be updated without replacing
the captured objects.

#### Scenario: Runtime inputs match the captured contract
- **WHEN** runtime inputs match the qualified entry contract
- **THEN** the system SHALL update persistent values and replay without changing captured tensor addresses

#### Scenario: Runtime inputs violate the captured contract
- **WHEN** tensor structure, metadata, address ownership, or execution signature differs from the captured contract
- **THEN** replay SHALL be rejected before graph launch with a structured contract error

### Requirement: Capture and replay are genuine
Startup SHALL build complete target and draft manifests for every configured
descriptor and participating TP rank that the workload requires.  A mode log,
successful capture without replay, target-only graph, draft-only graph, or
hidden eager execution SHALL NOT count as FULL success.

#### Scenario: Startup manifest is complete
- **WHEN** the service reports ready in strict acceptance mode
- **THEN** all required qualified target and draft entries SHALL have completed capture records on every TP rank

#### Scenario: Requests exercise FULL
- **WHEN** an accepted test request invokes a captured batch class
- **THEN** the corresponding per-entry replay counter SHALL increase only after graph launch succeeds

### Requirement: Capture blockers are fixed with the narrowest mechanism
A capture blocker SHALL be changed only after a fresh RED test reproduces it on
baseline `1a8feb60`.  A Python or existing plugin-side solution SHALL be used
when it can satisfy capture safety, numerical correctness, and performance
without changing shared behavior.

#### Scenario: Python-side repair is sufficient
- **WHEN** a focused test proves a plugin-side input or state transformation can make the operation capture-safe
- **THEN** no new custom operator SHALL be introduced for that blocker

#### Scenario: Blocker is not reproduced
- **WHEN** the current baseline does not reproduce a historical blocker
- **THEN** no historical repair for that blocker SHALL be copied into the change

### Requirement: Dedicated operator admission is evidence-gated
A new operator MAY be added only after a focused RED demonstrates that the
existing operator or Python path cannot participate safely in the required
FULL capture/replay contract.  Such an operator MUST have a new private name,
MUST be called only from the exact activation path, and MUST NOT modify an
existing schema or caller.

#### Scenario: Dedicated operator is admitted
- **WHEN** the RED evidence, attempted plugin-side alternative, required tensor ABI, and isolation test are recorded in OpenSpec
- **THEN** a separately named 310P DFlash FULL operator MAY be implemented and tested in isolation before integration

#### Scenario: Dedicated operator predicate is false
- **WHEN** execution is Eager, Piecewise, FULL_DECODE_ONLY, non-DFlash, or non-310P
- **THEN** the dedicated operator call count SHALL remain zero and the baseline operator path SHALL remain unchanged

### Requirement: Failure behavior is explicit
The system SHALL distinguish contract or implementation failures from external
device-resource failures.  It SHALL NOT convert a `507xxx`, AICore, HCCL,
contract, or numerical failure into a successful FULL result.

#### Scenario: Implementation failure occurs
- **WHEN** a graph contract, replay, output comparison, or dedicated-operator check fails
- **THEN** the affected test SHALL fail with preserved first-error evidence

#### Scenario: External device resource is unavailable
- **WHEN** the process cannot start because the selected cards are occupied or device resources are unavailable before the test reaches the graph path
- **THEN** the run SHALL be marked blocked rather than passed or counted as a code regression

### Requirement: Existing paths remain frozen
Eager, Piecewise, FULL_DECODE_ONLY, non-DFlash speculative decoding,
non-speculative decoding, other platforms, and upstream vLLM SHALL retain their
baseline control flow, operator selection, outputs, and graph contracts.

#### Scenario: Frozen-mode unit controls run
- **WHEN** the focused regression suite runs with any exact activation condition disabled
- **THEN** it SHALL observe baseline dispatch, inputs, operator calls, and outputs

#### Scenario: Frozen-mode hardware controls run
- **WHEN** matched Eager, Piecewise, and FULL_DECODE_ONLY controls are executed on the same model, topology, dataset, and request parameters
- **THEN** they SHALL complete without new errors and any throughput or acceptance-length change greater than five percent SHALL block acceptance pending explanation

### Requirement: Formal FULL acceptance is reproducible
Server-1 formal acceptance SHALL cover 4B TP1 and TP2 plus 35B W8A8 TP2 and
TP4. Each topology SHALL run GSM8K with output length 256 and random
input/output length 2048 at concurrency 1 with 4 requests and concurrency 10
with 20 requests. After the branch is pushed, server-2 independent pull
validation SHALL cover its available 2B TP1/TP2 and a representative 35B
topology without copying code or runtime files from server 1. The final report
SHALL include output throughput, acceptance length, success count, capture
manifest, replay evidence, code commit, environment versions, commands, and an
AISBench smoke result.

#### Scenario: Acceptance group passes
- **WHEN** every request in a group completes and required target and draft replay counters increase on every participating rank
- **THEN** the group SHALL be marked PASS and compared with its frozen Eager baseline

#### Scenario: Acceptance evidence is incomplete
- **WHEN** any required request, metric, manifest entry, rank, replay record, command, or environment record is missing
- **THEN** the group SHALL remain incomplete and SHALL NOT be reported as FULL support
