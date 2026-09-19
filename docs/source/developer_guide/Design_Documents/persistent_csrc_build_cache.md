# Persistent Incremental Csrc Build Cache

## Overview

vLLM Ascend builds native third-party libraries and many AscendC custom
operators. An exact final-artifact cache is fast when it matches, but any
source change invalidates the complete artifact. The persistent incremental
csrc cache adds a second level that reuses individual compiler actions across
CI jobs, workspaces, and repeated local source builds.

The two cache levels have different roles:

```text
L0 exact final artifact
        |
        | miss
        v
L1 persistent action entries
        |
        v
ordinary source build
```

The source build remains authoritative. L1 only replaces an equivalent action
with previously verified artifacts.

## Goals and non-goals

Goals:

- reuse native build work across CI jobs;
- reuse native build work across repeated local source builds without remote
  transport;
- invalidate only actions whose semantic inputs changed;
- reuse entries across equivalent checkout and build roots;
- persist local entries through the repository's cache transport;
- publish artifacts safely under concurrent builds; and
- degrade to an ordinary build when an optional cache layer is unavailable.

Non-goals:

- replacing compiler dependency tracking;
- replacing the L0 final-artifact cache;
- sharing file locks across physical nodes;
- requiring historical source snapshots to contain the cache engine; or
- migrating older cache schemas in place.

## Architecture

```text
workflow
  |
  +-- restore L1 snapshot
  |
  +-- CMake adapter
  |     |
  |     +-- action identity
  |     +-- entry lookup/validation
  |     +-- restore or compile
  |     +-- local entry save
  |     `-- custom-operator publication
  |
  `-- publish changed L1 snapshot (writers only)
```

`build_cache.py` owns local action identity, entries, locking, restoration,
and publication. CMake declares semantic inputs and invokes the engine.
Composite actions own persistent snapshot keys and OBS transport. Workflows
only decide where to restore and which trusted build roles may publish.

## Build flow

### Cold build

```text
L0 miss
  -> L1 miss
  -> normal source build starts
  -> action MISS
  -> compiler runs
  -> verified local entry save
  -> .updated marker
  -> writer publishes L1
  -> optional L0 publication
```

### Warm build

```text
L0 miss
  -> compatible L1 snapshot restored
  -> .updated marker reset
  -> normal source build starts
  -> action identity calculated
  -> entry validated
  -> action HIT
  -> artifacts restored
  -> compiler action skipped
```

An all-HIT build does not recreate `.updated`, so it does not publish another
unique remote snapshot.

## Cache identity

Each action key is the canonical hash of three independent identities:

```text
prepared_input_hash
    what is compiled

recipe_hash
    how it is compiled

compiler_environment_hash
    which toolchain compiles it
```

For custom operators, `operator_text_hash` provides the operator namespace. It
does not replace any of the three action-key components.

The entry manifest stores the hashes and the exact artifact model used by the
key. A HIT requires the expected schema, domain, action key, artifact model,
and verified artifact content.

## Prepared-input contract

Prepared inputs include all compiler-visible semantic content, including:

- generated operator sources;
- dependent generated sources;
- shared kernel sources;
- compiler-visible recipe files; and
- shared compatibility headers such as `cann_compat.h`.

Physical checkout and build roots are not semantic identity. UTF-8 text may
therefore normalize explicitly declared roots. A root may be normalized only
when every semantic object referenced through it is independently covered by
the action identity.

For example, a generated adapter can contain:

```text
-include /temporary/root/csrc/common/include/cann_compat.h
```

The temporary root is normalized, while `cann_compat.h` content is hashed as a
prepared input. Moving the workspace remains a HIT; changing the header is a
MISS.

The normalization contract is deliberately narrow:

- UTF-8 text replaces only explicit normalize roots;
- binary inputs remain raw-byte-sensitive;
- paths outside explicit roots remain sensitive; and
- symlink identity and resolved semantic content both participate in hashing.

An unreadable explicit semantic input is never silently omitted from identity.

## Recipe and compiler environment

The recipe identity covers compiler commands, recipe files, and explicit
recipe values after the same root normalization contract.

The compiler-environment identity covers the host platform, selected compiler
profile, tool version output, CANN metadata, and explicit environment values.
Absolute compiler installation paths are not semantic when the reported tool
identity is equal.

## Schema and snapshot compatibility

Entry manifests use `SCHEMA_VERSION = 4`. Schema 4 introduced safe textual
prepared-input normalization and complete semantic coverage for normalized
paths. Older entries naturally miss because the persistent key contains the
schema and entry validation rejects a different schema. There is no in-place
migration: the first schema-4 build is cold and later builds are warm.

`PUBLISH_STATE_SCHEMA` is separate because build-tree artifact ownership is a
different format from persistent entry identity.

The snapshot key command has one compatibility model:

```text
schema
+ architecture
+ canonical SOC
+ explicit compiler-image identity for a nested container build
  or CANN metadata + runtime OS/libc identity for a direct build
```

The restore action adds the tracked csrc hash and a unique publication suffix.
Both producer and consumer keys therefore come from the same implementation.
An explicit compiler image takes precedence over the outer runner environment,
so a Docker build is keyed by the environment that actually compiles csrc.
Direct builds include the runtime OS/libc identity to keep host-built artifacts
from crossing incompatible system-header or ABI boundaries.

## Entry and artifact lifecycle

For each action the engine:

1. hashes prepared inputs, recipe, and compiler environment;
2. derives the content-addressed entry path;
3. acquires the entry and action locks;
4. validates and restores a matching entry;
5. otherwise runs the original build command;
6. discovers and verifies produced artifacts;
7. atomically saves the local entry;
8. marks the local L1 snapshot as updated; and
9. publishes custom-operator artifacts into the shared build output.

The entry manifest is the correctness record. No separate mutable index is
required for lookup or invalidation.

## Concurrency model

Three lock scopes protect distinct invariants:

- the entry lock permits one creator for a content-addressed entry;
- the action lock protects one action's private staging directory; and
- the publish lock serializes shared custom-operator output publication.

Custom operators compile into private staging directories. Publication tracks
artifact ownership and uses atomic replacement, preventing partial output from
becoming visible. A conflicting owner for the same relative artifact is a hard
error rather than an unsafe overwrite.

Multi-node jobs use a node-scoped cache directory:

```text
/root/.cache/vllm-ascend/csrc-build-cache/<soc>/node-<worker>
```

The design does not depend on cross-node `flock` behavior.

## Persistent transport and trust

The restore and save composite actions own the fixed Huawei OBS transport
configuration. The cache engine has no storage API or credential knowledge.

Read access follows the runner's existing OBS authorization. Shared writes
require both `HW_OBS_AK` and `HW_OBS_SK`; possession of those secrets is the
only write-authorization boundary. The save action receives the credentials
explicitly from trusted callers and skips publication when either is absent.

Only roles that create canonical reusable domains publish L1:

- the central csrc producer;
- release wheel builds for their container toolchains; and
- image builds for their container toolchains.

Doctest, nightly, and historical-source jobs are restore-only L1 consumers.
Selected tests consume the central producer's exact L0 output. On an L0 miss,
they continue directly to the ordinary source build without making persistent
L1 transport a prerequisite. Local compilation remains correct but does not
create a shared snapshot.

A writer publishes only after a successful build creates or replaces a local
entry. OBS restore and save failures remain performance degradations.

## Failure behavior

| Condition | Behavior |
| --- | --- |
| Historical source has no cache engine | Report unsupported and continue with the ordinary build. |
| Snapshot environment cannot be fingerprinted | Skip L1 restore and continue with the ordinary build. |
| Snapshot restore fails | Continue with an empty local L1. |
| Entry is absent, invalid, or corrupt | Compile and replace it. |
| Entry lock is unavailable | Bypass the entry and compile. |
| Local entry save fails | Warn and keep the successful build output. |
| Snapshot save fails | Keep the successful build or verified L0 artifact. |
| Compiler command fails | Fail the build. |
| Artifact ownership collides | Fail rather than overwrite another action's output. |
| Action or publish synchronization fails | Fail when continuing could corrupt shared output. |

Operational JSONL events are limited to cache results, warnings, and lock
contention. Observability is best effort and never serializes compilation.

## Integration patterns

### Direct source consumers

Doctest and nightly source replacement restore L1 before their existing source
installation. They do not publish. Selected tests and scheduled upstream E2E
use the canonical producer's exact L0 output; an L0 miss proceeds through their
existing source build without a second direct L1 fallback.

### Local source builds

Ordinary local source builds use the same action cache in the repository's
ignored `build_cache` directory. This is local-only acceleration: no OBS
credentials or remote snapshot transport are involved.

### Central producer

The reusable producer restores L1, performs the ordinary source build, verifies
final native artifacts, publishes changed L1 state, and then publishes the
exact L0 artifact.

### Historical sources

Main2Main and bisect-style builds keep current workflow helpers separate from
the selected source tree. A historical tree without `build_cache.py` reports
`supported=false` and builds normally.

### Docker consumers

Wheel and image builds cross a container boundary:

```text
host restore
  -> Docker build context
  -> source compile in image
  -> image-local L1
  -> export to host
  -> trusted save
```

Docker layer reuse is not an L1 action HIT; local entry validation remains the
source of truth.

## Validation summary

| Capability | Evidence |
| --- | --- |
| Action identity, entries, and failure behavior | cache-engine unit tests |
| Concurrent ownership and lock behavior | concurrency unit tests |
| Direct source reuse | A2, A3, and 310P cold/warm runs |
| Selective invalidation | operator-local mutation with unrelated HITs |
| Root-independent semantic identity | schema-4 CASE-C regression |
| Persistent producer | cold/warm producer runs |
| Docker boundary | wheel and image export/save/restore runs |

Some exact production callers still require release credentials, caller
registration, or multi-node allocation. Structural and component evidence does
not replace those caller-specific runtime gates.

## Operational debugging

For an unexpected full MISS, compare schema, snapshot compatibility,
`prepared_input_hash`, `recipe_hash`, and `compiler_environment_hash` in entry
manifests. For broad selective invalidation, inspect the prepared-input set and
operator namespace. For Docker reuse, verify host restore, image cache path,
export, and host save. For historical source, check the restore action's
`supported` output.

## Implementation map

| Responsibility | Source |
| --- | --- |
| Cache identity, entries, locking, and publication | `csrc/scripts/build_cache.py` |
| CMake-to-engine adapter | `csrc/cmake/build_cache.cmake` |
| Operator semantic inputs | `csrc/cmake/func.cmake` |
| Third-party semantic inputs | `csrc/cmake/third_party/ascend_protobuf.cmake` |
| Persistent restore and key orchestration | `.github/actions/csrc-l1-restore/action.yaml` |
| Persistent changed-snapshot publication | `.github/actions/csrc-l1-save/action.yaml` |
| Docker image-to-host export | `.github/workflows/scripts/export_csrc_l1_from_image.sh` |
| Central producer | `.github/workflows/_build_csrc_cache.yaml` |
