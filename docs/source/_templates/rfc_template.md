# [RFC] xx

*A proposal for a significant architectural or design change. Copy this template to write the design document for an RFC, and open the RFC issue first with the [`750-RFC` issue template](../../../.github/ISSUE_TEMPLATE/750-RFC.yml) so the community can follow and comment on the discussion. For substantial changes, an accepted RFC is a prerequisite for creating a `rfc/<feature-name>` feature branch (see the [versioning policy](../../community/versioning_policy.md)).*

> **Document Metadata**:
>
> - **Status**: Draft | Under review | Accepted | Implemented | Rejected | Superseded. Update it as the RFC moves; link the superseding RFC when it is superseded.
> - **Authors**: Names and GitHub handles of the authors.
> - **RFC issue**: Link to the RFC issue (e.g., `#12345`). Keep the issue and this document in sync; the issue holds the discussion, this document holds the design.
> - **Feature branch**: `rfc/<feature-name>`, if the change needs a long-lived branch.
> - **Target release**: The vLLM Ascend version this change is planned for, or `TBD`.
> - **Related issues / PRs**: Upstream vLLM issues, prior art, prototypes, and follow-up work.

## TL;DR (Optional)

*A self-contained summary for readers who will not read the whole document: what is broken, what you propose, what changes for users, what it costs. Three to five sentences.*

**Note: Recommended for any RFC that is longer than two screens. Reviewers triage RFCs by this section; keep it free of new terminology that is only defined later.**

> **Writing Requirements**:
>
> - State the problem in one or two sentences, in terms a user or operator would recognize.
> - State the proposed change and the mechanism, without implementation detail.
> - State the main benefit and the main cost (performance, memory, compatibility, or maintenance).
> - Do not repeat the whole Motivation or Design sections here, and do not hide caveats at the bottom of the document.

**Example**:

> Hybrid models must give every KV cache group the same page size in bytes, so core re-cuts groups by layer count, pads pages, and raises block sizes. This RFC removes the rule: the KV cache pool becomes a set of equal large pages, and each page belongs to one group. Grouping becomes one group per `KVCacheSpec`, and kernels keep their upstream addressing. Capacity stays within 1% of main; the motivation is simpler code, not memory.

## Motivation (Required)

*Why this RFC exists. Describe the current state, what is wrong with it, and why it is worth solving now.*

**Note: This section maps to the `Motivation.` field of the RFC issue template. A reviewer who disagrees with the problem will disagree with every design decision that follows, so make the problem concrete and evidenced.**

> **Writing Requirements**:
>
> - Describe the **current state**: what the code does today, with links to the relevant files, classes, or upstream code.
> - Describe the **problem**: the concrete limitation, failure mode, or cost. Give a reproducible example when possible (model, hardware, batch size, command).
> - Quantify the cost: benchmark numbers, memory overhead, unsupported model count, maintenance effort, or user-reported failures with links.
> - Explain **why now**: which incoming requirement, upstream change, or roadmap item makes the status quo unacceptable.
> - Separate facts from hypotheses. Mark unverified assumptions explicitly; they become open questions later.
> - Do not propose a solution in this section.

**Example**:

**Current state.** `KVCacheManager` asserts that all KV cache groups share one page size in bytes (`assert len(page_sizes) == 1`), and models satisfy the rule by grouping layers of the same type, raising the block size, or padding pages.

**Problem.** The grouping is byte arithmetic rather than model structure. On Kimi-Linear-48B at TP4 (20 KDA layers, 7 MLA layers), a KDA state is 561,152 bytes; to match the MLA page the block size is raised to 512 tokens and every KDA state is padded by 5.1%. The model has two kinds of KV cache but runs four groups.

| Evidence | Value | Source |
|------|------|------|
| Padded KDA state | 5.1% per state | Config dump of `Kimi-Linear-48B` at `TP4` |
| Group plan change on DSV4.1-Flash | 7 groups → 9 groups | Indexer precision fp8 → mxfp4, see #62100 |
| New concepts required by the workaround | `UniformTypeKVCacheSpecs`, `hash_block_size`, `prefix_match_unit` | `vllm/v1/core/kv_cache_utils.py` |

**Why now.** Two hybrid models are being brought up this release, and each one needs another special case in the grouping heuristic. The heuristic is also being made pluggable (#62100), which moves it out of core but keeps the equal-page rule that forces a heuristic to exist.

## Goals and Non-Goals (Required)

*The boundary of the proposal. Goals are what the RFC promises and will be reviewed against; non-goals are what it deliberately leaves out.*

> **Writing Requirements**:
>
> - Write goals as checkable statements ("one block size for all prefix-cacheable groups"), not as activities ("improve the design").
> - List non-goals explicitly, especially the ones a reviewer would otherwise assume are included.
> - If a goal is only partially met, say which part is met and which is deferred, with the follow-up issue.
> - Keep the list short. A goal that no design decision in this RFC serves should be removed.

**Example**:

| Goals | Non-Goals |
|------|------|
| One KV cache group per `KVCacheSpec`, with same-spec drafter layers joining the target model's group | Higher KV cache capacity; capacity may stay within a bounded loss |
| One block size for all prefix-cacheable groups, so a block hash means the same tokens in every group | A new eviction policy; LRU over cached blocks stays |
| No page padding, no block-size raise for byte reasons, no sub-block hashing | Attention kernel changes, apart from 64-bit offsets |
| Kernels, slot mapping, and block tables keep their upstream form | Multi-node or multi-tenant cache sharing |

## Proposed Change (Required)

*The design itself. This is the main body of the RFC and the section reviewers will implement against.*

**Note: This section maps to the `Proposed Change.` field of the RFC issue template. Start from the overall shape, then go down to interfaces and algorithms. A design that cannot be reviewed as a sequence of concrete decisions is not ready for an RFC.**

> **Writing Requirements**:
>
> - **Overview first**: an end-to-end diagram (Mermaid or ASCII) of the changed flow, plus numbered explanations of each step.
> - **Detailed design**: cover each affected module, its responsibility, and what changes in it. Name real files, classes, and functions.
> - **Interfaces and configuration**: give the exact public surface (function signatures, request fields, CLI flags, configuration keys). For vLLM Ascend, user-facing configuration belongs in a typed key under `--additional-config`; do not add a new environment variable unless it is genuinely unavoidable, and justify it if so.
> - **Data structures**: give a field table for every structure or interface this RFC introduces.
> - **Key algorithm**: give pseudocode or the step list for any non-obvious computation, and state its complexity and edge cases.
> - **Interaction with existing features**: ACLGraph, quantization, expert parallelism, PD disaggregation, speculative decoding, and prefix caching all constrain designs. Say what happens when they are combined with this change.
> - **Upstream alignment**: state whether this is a patch, an inheritance-based extension, or an upstream contribution, and what the long-term plan is for a patch.

**Example**:

**Overview.** The KV cache pool becomes `N` pages of `S` bytes. A page belongs to exactly one group at a time, and the group cuts it into blocks of its own size.

```text
      request ──► planner ──► group per KVCacheSpec
                     │
                     ├── block size  B (common to all prefix-cacheable groups)
                     └── page size   S (multiple of every group's block size)

      pool: [ page 0 ][ page 1 ][ page 2 ] ...
               │           │          │
             KDA grp     MLA grp    free ──► can be claimed by any group
```

1. **Plan**: the planner builds one group per distinct `KVCacheSpec` and computes the common block size `B` and page size `S`.
2. **Allocate**: a group takes a free block from a page it already owns, else a free uncached page, else evicts its LRU cached block or the oldest free page.
3. **Address**: block `b` of a group starts at byte `b × block_bytes`, so the kernel row is `b × R`; no lookup table is involved.

**Interfaces.**

| Interface | Change | Notes |
|------|------|------|
| `KVCacheSpec` | unchanged | The grain of grouping moves from page size to spec identity |
| `KVCacheManager.allocate_slots` | new `(group_id, block_id)` addressing | Block ids are unique only within a group |
| `--block-size` | reused | An explicit value is used as given if feasible |

**Interaction.** Prefix caching hashes at block granularity, so `hash_block_size` and `prefix_match_unit` are removed; ACLGraph keeps one capture per block size; PD disaggregation requires rank-uniform geometry, which startup asserts.

## Impact and Risks (Required)

*What this change costs and what can go wrong. Reviewers use this section to decide whether the benefit is worth the risk.*

> **Writing Requirements**:
>
> - Cover the dimensions that apply: performance, memory, compatibility (models, hardware, configuration, saved state), maintenance and review cost, testing effort, and security.
> - For each risk, give the likelihood, the observable symptom, and the mitigation. "Monitor it" is not a mitigation.
> - Quantify performance and memory impact when you have measurements, and state the hardware and workload they were measured on.
> - If the change is user-visible, say whether a release note fragment is required and which behavior changes for existing users.

**Example**:

| Dimension | Impact | Risk | Mitigation |
|------|------|------|------|
| Performance | Kernel addressing and slot mapping unchanged | Low | No change in the hot path; verified with the ACLGraph benchmark |
| Memory | Capacity within 1% of main on the verified models | Medium: tail waste on mixed page sizes | Page size bounded by 2% demand-weighted waste, asserted at startup |
| Compatibility | Configuration and saved KV cache state are untouched | Medium: custom `SupportsKVCacheGrouping` implementations break | Keep the old grouping available behind a migration flag for one release |
| Maintenance | Removes four special cases from core | Low | One group per spec is the only rule left to maintain |
| Testing | New unit tests for planner and allocator; E2E on hybrid models | Medium: hybrid model coverage is thin | Add Kimi-Linear and DSV4.1-Flash to `tests/e2e/models/configs/` |

## Alternatives Considered (Required)

*The options you rejected, and why. This section is what distinguishes an RFC from a patch.*

> **Writing Requirements**:
>
> - Include the **status quo** as one alternative, with the reason it is no longer acceptable.
> - For each alternative, state the idea, its advantages, its disadvantages, and the concrete reason it was rejected.
> - Do not include straw men. If an alternative would plausibly work, say what evidence rules it out.
> - If an alternative is deferred rather than rejected, say what would make it viable later.

**Example**:

| Alternative | Advantages | Disadvantages | Why it was not chosen |
|------|------|------|------|
| Keep the equal-page rule, make grouping pluggable (#62100) | Smaller change, no allocator work | Every model or platform owns its own heuristic, and the padding remains | Moves the complexity instead of removing it |
| One group with per-layer page sizes | Trivial plan | Every component must handle heterogeneous pages, including kernels and hashing | Blast radius far larger than the grouping code it replaces |
| Separate KV cache pool per group | Simple ownership | Fragmentation across pools, and capacity drift under mixed models | Rejected by measurement: capacity loss exceeded 5% on Kimi-Linear |

## Testing and Validation Plan (Required)

*How the change will be proven correct, and what evidence the PR will carry.*

**Note: The [PR testing requirements](../../developer_guide/contribution/testing.md#pr-testing-requirements) define the evidence a PR needs. This section plans that evidence before implementation starts.**

> **Writing Requirements**:
>
> - **Unit tests**: name the test files under `tests/ut/` and the cases they cover, including boundary cases (empty input, single element, min/max, sequence-length limits).
> - **E2E tests**: name the config YAML under `tests/e2e/models/configs/`, the models, the hardware (`A2` / `A3` / `A5`), and the accuracy metrics with thresholds.
> - **Performance validation**: give baseline and target, model, NPU type, card count, deployment mode, input/output length, and concurrency. Never leave the metric implicit.
> - **What CI can verify**: identify checks that run without NPU hardware (`tests/ut` on CPU, `bash format.sh ci`) so reviewers know what is pre-verified.
> - **Definition of done**: the exact statements that must be true before the RFC is marked `Implemented`.

**Example**:

| Layer | Coverage | Evidence |
|------|------|------|
| Unit | planner grouping, page sizing, allocate/evict order, hole handling | `tests/ut/core/test_kv_cache_planner.py`, added with the first PR |
| Unit (boundary) | zero-length request, request larger than a page, single group, all groups prefix-cached | Same file, marked with `pytest.mark.parametrize` |
| E2E | Kimi-Linear-48B and DSV4.1-Flash accuracy and throughput | New YAML under `tests/e2e/models/configs/`, `A2` 8-card, GPQA and the standard accuracy set |
| CI (no NPU) | unit suite and lint | `TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -sv tests/ut`, `bash format.sh ci` |

## Rollout and Migration (Optional)

*How the change reaches users without breaking them: implementation phases, flags, branch plan, deprecation.*

**Note: Required when the change is large, breaks existing behavior, or spans more than one release.**

> **Writing Requirements**:
>
> - Split the implementation into phases that can each be reviewed and merged independently, with the RFC section each phase implements.
> - State the default state of any new flag, and how a user opts in and opts out.
> - If behavior changes for existing users, give the migration steps and what happens if they do nothing.
> - State the branch plan: whether a `rfc/<feature-name>` branch is needed, which maintainers approved it, and when it merges back.
> - State the removal or follow-up plan for any temporary compatibility code.

**Example**:

| Phase | Content | Default | Merge condition |
|------|------|------|------|
| 1 | Pluggable grouping, old allocator kept | Old behavior | Unit coverage for every existing model plan |
| 2 | Large-page allocator, opt-in via `additional_config` | Off | Accuracy parity on 4 verified hybrid models |
| 3 | Large pages become the default, remove the old path | On | One release of opt-in use without regression |

The work lands on `rfc/large-page-kv-cache` (approved by two maintainers), one PR per phase. The old path is removed one release after phase 3, and the release note fragment records the default change.

## Open Questions (Optional)

*Decisions that are not settled yet, ideally with a resolution path.*

> **Writing Requirements**:
>
> - One numbered question per item, with the options, the default if nobody answers, and who should answer.
> - Do not use this section to hide missing design work. If a question blocks the design, the RFC is not ready.
> - Remove questions as they are resolved, and record the resolution in the relevant section.

**Example**:

1. **Should an explicit `--block-size` be allowed below 64 tokens?** Option A: allow it when feasible; option B: clamp to the kernel minimum. Default: A, unless a kernel cannot express the smaller block. Owner: attention kernel maintainers.
2. **Do we keep `hash_block_size` for one release as an alias?** Default: yes, with a deprecation warning.

## Feedback Period (Optional)

*How long the RFC stays open for comment, so that interested parties can plan their review.*

**Note: This section maps to the `Feedback Period.` field of the RFC issue template. The community convention is at least one week.**

> **Writing Requirements**:
>
> - Give the start and end dates, in UTC.
> - Name the SIG, team, or reviewers whose feedback is required before acceptance.
> - State whether silence means acceptance.

**Example**:

> Feedback period: 2026-11-24 to 2026-12-01 (UTC). Required reviewers: the vLLM Ascend KV cache maintainers and the upstream vLLM core maintainers named in the CC list. The RFC is accepted when no blocking comment is open at the end of the period.

## CC List (Optional)

*The people whose review or awareness this RFC needs, copied from the issue template.*

> **Writing Requirements**:
>
> - Use GitHub handles. Say why each person is on the list (owner of the affected code, dependent feature owner, upstream reviewer).
> - Keep the list short enough that every name is likely to respond.

**Example**:

> @xxx (KV cache and scheduler), @yyy (hybrid model bring-up), @zzz (upstream vLLM core, `v1/core/kv_cache_utils.py`)

## References (Optional)

*Source material a reviewer can use to check the claims in this RFC.*

> **Writing Requirements**:
>
> - Link the RFC issue, the tracking issue, related upstream RFCs and PRs, design documents, and benchmark reports.
> - Link with a short label that says what the reader will find, not a bare URL.

**Example**:

- [Hardware pluggable RFC](https://github.com/vllm-project/vllm/issues/11162): the plugin interface this design must keep.
- [Pluggable KV cache planning (#62100)](https://github.com/vllm-project/vllm/issues/62100): prior work that makes grouping replaceable.
- [Large-page allocator RFC (#64401)](https://github.com/vllm-project/vllm/issues/64401): the upstream discussion this template was modeled on.
