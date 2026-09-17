# Persistent Incremental Csrc Build Cache

## Overview

vLLM Ascend builds native C++ and AscendC artifacts in several CI paths. A
final artifact cache avoids rebuilding when the complete source and build
environment match exactly, but a miss previously forced every native action to
run again.

The persistent incremental csrc build cache adds a second cache level beneath
that final artifact cache. It stores the result of individual compiler actions,
persists those entries between CI jobs, and reuses only entries whose semantic
inputs, recipe, and compiler environment still match. The normal source build
remains authoritative: CMake still schedules every action, and the cache wrapper
either restores a valid result or runs the original command.

The implementation covers custom AscendC operators and selected third-party
native builds. It is used by the central csrc producer, direct source consumers,
historical-source workflows, Docker wheel and image builds, and node-scoped
multi-node builds.

## Goals and non-goals

### Goals

- Reuse native csrc work across CI jobs and workspaces.
- Invalidate only actions affected by a semantic input, recipe, or toolchain
  change.
- Make reuse safe when the same source is checked out under a different
  physical root.
- Persist the local action cache through the CI cache transport.
- Publish parallel custom-operator outputs without exposing partial results or
  assigning one artifact to multiple actions.
- Keep existing build paths operational when the persistent transport is
  unavailable or a historical source predates the cache engine.

### Non-goals

- Replace compiler dependency tracking or the ordinary CMake source build.
- Replace the exact final-artifact cache.
- Share filesystem locks between physical nodes.
- Require historical source snapshots to contain a newer cache engine.
- Migrate old on-disk entry schemas in place.
- Guarantee that every production caller can be allocated during validation;
  runner, credential, and release-context availability remain external CI
  concerns.

## System architecture

The final artifact cache is referred to as L0. The persistent fine-grained
cache is L1.

```text
                 +-------------------+
                 | L0 exact artifact |
                 +---------+---------+
                           | miss
                           v
                 +-------------------+
                 | persistent L1     |
                 | action entries    |
                 +---------+---------+
                           |
                           v
                  normal source build
                           |
                 +---------+---------+
                 | per-action cache  |
                 | lookup or compile |
                 +-------------------+
```

L0 contains a complete, ready-to-consume csrc result for one exact build. L1
contains content-addressed action entries and their artifacts. An L0 miss can
therefore restore a compatible L1 snapshot and run the normal source build;
unchanged actions become L1 hits while changed actions compile.

L1 is split into two domains:

- `custom_operator` entries contain one isolated AscendC action's outputs.
- `third_party` entries contain one third-party build unit's outputs.

## Build flow

### Cold build

```text
L0 miss
  -> L1 snapshot miss
  -> normal source build
  -> action identities calculated
  -> action MISS
  -> original compiler command
  -> verified entry save
  -> updated L1 snapshot publish
  -> optional L0 publish
```

### Warm build

```text
L0 miss
  -> compatible L1 snapshot restored
  -> normal source build starts
  -> action identity calculated
  -> valid entry HIT
  -> artifacts restored and published
  -> original compiler action skipped
```

The source build remains the control plane in both cases. L1 changes the
implementation of an equivalent action, not which actions the build system
requires.

## Persistent transport and trust model

The cache engine owns only the local content-addressed L1 directory. CI
persists snapshots of that directory through `runs-on/cache`, backed by the
project's Huawei OBS S3-compatible cache storage:

```text
local L1 directory
  -> runs-on/cache
  -> Huawei OBS
```

Bucket and endpoint selection are workflow concerns. Credentials and write
authorization are also kept at that layer: possession of the write-capable
`HW_OBS_AK` and `HW_OBS_SK` secrets authorizes publication to the shared cache.
Trusted producer and selected schedule or release paths pass those secrets to
save steps.
Workflows without both secrets, including fork and other untrusted source
paths, may restore through the runner's existing read-only OBS access but do
not invoke shared cache save.

This boundary prevents code from an untrusted source from publishing artifacts
into a namespace later consumed by trusted jobs. There is no independent
write-enable flag: the write credentials themselves are the authorization
boundary. OBS settings and credentials are deliberately absent from
`build_cache.py` and the reusable restore/save actions; cache identity remains
independent of the persistence backend.

An OBS restore or save failure is a performance degradation. The normal source
build remains authoritative, and a verified final L0 artifact remains usable.

The restore action is the single owner of the persistent snapshot key. Its
coarse compatibility descriptor contains only the normalized build
architecture, target SOC, and installed CANN metadata fingerprint; it never
contains a workflow role. When the action runs outside the compiler container,
as in wheel builds, the exact build image is used as the explicit toolchain
fallback. Producer and consumer jobs using the same compiler environment can
therefore discover the same snapshot namespace even when one uses a derived CI
image. The engine schema and csrc source hash rank compatible snapshots within
that namespace, while each inner action still validates its complete semantic
identity before a hit.

After restore, the action records the path and content hash of every entry
manifest in the snapshot. A trusted save publishes only when that manifest set
changes during the build. Diagnostic index updates from an all-hit build do not
republish an otherwise unchanged snapshot. Comparing entry manifests also
works across Docker builds, where compiler telemetry is produced inside the
image but the updated L1 directory is exported back to the host.

## Cache identity model

An action's final key is the canonical hash of three independent identities:

```text
prepared_input_hash
    = WHAT is compiled

recipe_hash
    = HOW it is compiled

compiler_environment_hash
    = WHAT TOOLCHAIN compiles it
```

`prepared_input_hash` covers generated compiler inputs, dependent generated
sources, shared compiler-visible content, and other files explicitly supplied
by CMake. Directory inputs are traversed in a stable logical order. The
manifest records the same normalized content hashes used by the final key.

`recipe_hash` covers the invoked command, recipe scripts and files, and
explicit recipe values. Known physical roots are normalized so an otherwise
identical command does not change identity merely because its checkout or build
directory moved.

`compiler_environment_hash` covers the selected environment profile, compiler
and toolkit versions, environment metadata files, explicit values, and tool
versions. It prevents an artifact produced by one effective toolchain from
being reused by another.

Custom operators also have an `operator_text_hash`. It identifies the tracked
operator source namespace and organizes entries and history. It is not a
fourth component of the final action key; the final key remains the canonical
hash of the three identities above.

Length-prefixed canonical records are sorted before hashing. This makes the
identity deterministic and prevents ambiguous concatenation.

## Prepared-input semantics

Prepared inputs describe the files the compiler actually consumes. For custom
operators they include:

- the operator's generated source;
- generated source from declared dependent operators;
- shared generated kernel content such as `ascendc/common`; and
- shared compiler-visible compatibility content such as `cann_compat.h`.

A physical checkout location is not semantic identity. For example, two
generated adapters may contain these otherwise equivalent compiler arguments:

```text
-include /tmp/work-a/csrc/common/include/cann_compat.h
-include /tmp/work-b/csrc/common/include/cann_compat.h
```

The known physical source root is normalized, while the referenced
`cann_compat.h` file is independently hashed as a prepared input. The resulting
contract is:

```text
workspace root moves     -> HIT
cann_compat.h changes    -> MISS
```

This is the safety condition for path normalization:

> A physical path may be normalized only when every semantic object referenced
> through that path is independently covered by cache identity.

Normalizing a path without covering its target content could create a false
hit, so CMake owns the prepared-input list and must add any newly embedded
semantic file to that list.

## Text and binary normalization contract

Prepared files are classified conservatively:

- UTF-8 text has only explicitly supplied normalize roots replaced with stable
  placeholders before hashing.
- Binary data is always hashed as raw bytes.
- A path outside the explicit normalize roots remains identity-sensitive.
- A file that cannot be decoded safely as UTF-8 remains raw-byte-sensitive.

The wrapper does not search for and rewrite arbitrary path-shaped strings.
Callers explicitly provide the source root, CMake binary root, toolkit root, or
another reviewed root. This keeps portability scoped and preserves false-hit
safety.

## Schema and compatibility

The entry schema is `SCHEMA_VERSION = 4`. Schema 4 introduced the prepared
text normalization contract and the associated semantic-input coverage. The
schema is part of the persistent L1 transport prefix and is also checked in
every entry manifest.

Schema 3 snapshots may remain in remote storage, but a schema 4 restore prefix
does not match them. The first schema 4 build is therefore cold and subsequent
compatible builds can be warm. No in-place migration is attempted.

`PUBLISH_STATE_SCHEMA` is versioned separately. Publish state describes
temporary ownership of artifacts inside one build tree; it is not a reusable
cache entry and does not determine cross-job compatibility.

## Entry and artifact lifecycle

For each action the wrapper performs the following sequence:

1. Hash prepared inputs, the recipe, and the compiler environment.
2. Derive the final action key and entry path.
3. Acquire the action and entry synchronization required by the domain.
4. Load and validate the manifest, schema, key, artifact model, and every
   artifact hash or symlink target.
5. On a valid hit, restore artifacts atomically. Custom-operator artifacts are
   then published from the private action stage into the shared operator output.
6. On a miss, run the original build command and discover the action's exact
   outputs.
7. Publish build outputs required by downstream steps.
8. Save the entry through a verified temporary directory and an atomic rename.
9. Update the best-effort observational index and emit telemetry.

The manifest is the correctness record for an entry. `cache_index.json` records
current and historical identities for diagnostics, but it is not consulted as
the source of truth for a hit.

## Concurrency model

Custom-operator actions compile into private stage directories. They never infer
artifact ownership by observing changes in a shared output directory.

The cache uses four synchronization scopes:

- The entry lock serializes validation and creation of one content-addressed
  entry.
- The action lock prevents concurrent invocations of the same action in one
  build tree from resetting the same private stage.
- The publish lock protects the short operation that merges private action
  outputs into the shared operator directory.
- The index lock protects diagnostic history and is non-blocking; a busy index
  is skipped rather than delaying the build.

Publish state records the action that owns each shared artifact. Publication
rejects cross-action ownership collisions, verifies an action's old output
before removing it, writes files through atomic replacement, and publishes
files before symlinks. Compilation remains parallel; only shared publication is
serialized.

Entry directories are prepared privately, verified, and atomically renamed.
An existing entry is moved aside only during replacement and is restored if
publication fails.

Multi-node jobs use a node-scoped cache directory:

```text
/root/.cache/vllm-ascend/csrc-build-cache/<soc>/node-<worker>
```

Consequently the design does not depend on `flock` working across physical
nodes or network filesystems.

## Failure behavior

The cache is an optimization around an authoritative build, with explicit
boundaries:

| Condition | Behavior |
|---|---|
| Historical source has no cache engine | Restore action reports `supported=false`; ordinary build continues. |
| L1 transport restore or save fails | Workflow continues; the source build or verified L0 artifact remains usable. |
| Entry is absent, corrupt, or has another schema/key/artifact model | Treat as MISS and compile. |
| Entry restore fails | Warn, reset a private custom-operator stage when applicable, and rebuild. |
| Entry lock is unavailable | BYPASS the persistent entry and run the original command. |
| Cache entry save fails | Warn and keep the successful build result. |
| Diagnostic index is busy or cannot be written | Skip or warn; never change cache correctness. |
| Custom-operator action or publish lock times out | Fail the wrapper instead of risking shared-output corruption. |
| Build command fails | Return the original failure. |
| Artifact ownership collides | Fail publication; do not overwrite another action's output. |

`HIT`, `MISS`, `SAVED`, and `BYPASS` events, phase timings, lock timeouts, and
index skips are emitted as JSON lines when an event log is configured.

## Integration architecture

The integrations use a small number of patterns rather than a separate cache
implementation per workflow.

### Direct source consumers

Selected tests, upstream E2E, doctest, and nightly jobs restore L1 before source
installation. Trusted callers with OBS write credentials save the updated
snapshot afterward; untrusted or credential-less callers remain restore-only.
CMake enters the cache wrapper for every configured native action.

### Central producer

The producer first checks the exact L0 artifact cache. For a missing target it
invokes the real source build with a target-compatible L1 snapshot, verifies the
final csrc artifacts, saves the updated L1 snapshot, and publishes the exact L0
artifact. L0 and L1 therefore remain separate: L0 avoids the build entirely;
L1 accelerates an L0 miss.

### Source replacement and historical builds

Main-to-main and bisect-style jobs keep workflow helpers in a separate checkout
from the selected historical source. The restore action reads the engine schema
from the selected source when it exists. If that source predates the engine it
returns `supported=false`, leaving the historical source build unchanged.

### Docker consumers

Wheel and image jobs cross a filesystem boundary:

```text
host L1 restore
  -> Docker build context
  -> source build inside image
  -> updated image-side L1
  -> docker create and docker cp
  -> host L1 directory
  -> persistent L1 save
```

The export helper checks both the built image and the expected image-side cache
directory before replacing the host directory. A Docker layer hit is not treated
as an L1 action hit; the build-cache telemetry remains the evidence for action
reuse.

### Multi-node consumers

Each worker restores and updates its node-scoped L1 directory. This provides
local reuse without cross-node lock or publication assumptions.

### Consumer matrix

| Consumer class | L0 | L1 boundary | Source/helper separation | Cache directory scope |
|---|---:|---|---:|---|
| Central csrc producer | Yes | Host source build | No | Per job/target |
| Selected tests and direct E2E | Optional upstream artifact | Host source build | No | Per job/target |
| Main-to-main and historical source | Optional | Host source build | Yes | Per job/target |
| Nightly source replacement | Optional | Host source build | Yes | Per job/target |
| Release wheel | Final wheel/image layers | Host to Docker and back | Yes | Per job/target |
| Image build | Final image | Host to Docker and back | Yes | Per job/target |
| Multi-node | Consumer-specific | Host source build | No | Per SOC and worker node |

## Validation summary

| Capability | Evidence |
|---|---|
| Cache-engine semantics and concurrency | 32 unit tests |
| Direct source reuse | A2, A3, and 310P cold/warm builds |
| A5 mechanism | Isolated cold/warm smoke build |
| Fine-grained invalidation | Exact operator-local misses with unrelated hits |
| Central producer | Persistent L1 cold/warm with L0 publication |
| Random workspace portability | Schema 4 doctest cold/warm |
| Docker boundary | Wheel and image L1 export/save/restore |
| Semantic header invalidation | Cross-root single-operator build plus `cann_compat.h` mutation |

Detailed run URLs, keys, telemetry, and the Stage 0-9 evidence remain in the
delivery evidence set under `summary/`; they are not duplicated in this design
document.

## Production workflow validation limitations

Some exact production callers require scheduled-event registration, repository
credentials, release context, multi-node allocation, or runner topology that is
not available on demand. Their production paths received structural review,
YAML parsing, actionlint, environment-propagation review, cache-directory
lifetime review, and source/helper separation review. Where an exact caller
could not run, it remains `BLOCKED-BY-INFRA`, not `VALIDATED`.

The per-caller blockers and the one remaining runtime step are recorded in
`summary/09-workflow-plumbing.md` in the delivery evidence set.

## Operational debugging

For an unexpected all-MISS build, inspect the schema and persistent
compatibility prefix, then compare `prepared_input_hash`, `recipe_hash`, and
`compiler_environment_hash` in the event log or manifests.

For unexpectedly broad selective invalidation, inspect the prepared-input
manifest and operator namespace. In particular, check whether a generated input
started embedding a new semantic file whose contents are not yet listed as a
prepared input.

For Docker reuse failures, check the host restore result, the cache path copied
into the build context, the image-side `VLLM_ASCEND_BUILD_CACHE_DIR`, export back
to the host, and the final persistent save.

For historical source, check the restore action's `supported` output. A false
value is the intended fail-open path for a source tree without the engine.

## Implementation map

| Concept | Implementation |
|---|---|
| Cache identity, entries, locking, publication, index, and telemetry | `csrc/scripts/build_cache.py` |
| CMake-to-wrapper command construction | `csrc/cmake/build_cache.cmake` |
| Custom-operator prepared inputs and action isolation | `csrc/cmake/func.cmake` |
| Third-party integration | `csrc/cmake/third_party/ascend_protobuf.cmake` |
| Persistent restore transport and historical-source fail-open | `.github/actions/csrc-l1-restore/action.yaml` |
| Persistent save transport | `.github/actions/csrc-l1-save/action.yaml` |
| Central L0/L1 producer | `.github/workflows/_build_csrc_cache.yaml` |
| Missing-L0 coordination | `.github/workflows/_ensure_csrc_cache.yaml` |
| Docker L1 extraction | `.github/workflows/scripts/export_csrc_l1_from_image.sh` |
| Cache-engine and concurrency tests | `tests/ut/_tools/test_build_cache.py`, `tests/ut/_tools/test_build_cache_concurrency.py` |
