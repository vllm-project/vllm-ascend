# Release Notes Fragments

This directory holds release notes fragments. Everyone adds a fragment in the same PR that
introduces a user-facing change, so the release manager can assemble the release notes from
fresh and accurate context instead of re-investigating dozens of old commits.

- `fragments/` — one YAML file per PR, named after the PR number
- `archive/<version>/` — fragments already consumed by a released version
- `TEMPLATE.yaml` — copy it to create a new fragment

## Does my PR need a fragment?

One root principle: **can the user of the released artifact (wheel or Docker image) notice
the change?** If you cannot complete a sentence like "Users can now ...", "Users no longer
need to ...", or "Users should be aware that ...", your PR does not need a fragment.
"The code is cleaner", "it prepares X", and "it fixed a test" are not user sentences —
they describe the diff, not a difference users can observe.

Note that the decision does not depend on the type or the size of the PR, but on whether
users can observe the difference. A `refactor MoE` PR that also fixes a user-reported crash
needs a fragment; a pure code move does not.

### Write a fragment for

| Change | Example |
| --- | --- |
| New model, hardware, or operator support | DeepSeek V4.1 support on Model Runner V2 |
| New features, parameters, environment variables, or APIs | MooncakeConnector V2, `sp_min_token_num` |
| Changes to user-facing APIs, parameters, or configuration | an env var that now takes effect in more cases, a field added to API responses |
| Breaking changes, deprecations, default behavior changes | a feature switch enabled by default, a renamed argument |
| Dependency and toolchain updates | CANN, torch_npu, triton-ascend, vLLM alignment |
| Performance improvements (the entry must state the scenario and the effect) | FLA NPU backend, fewer H2D syncs |
| Fixes for user-reported bugs or issues affecting correctness and stability | hangs, OOM, wrong outputs |
| New or substantially expanded guides for brand-new scenarios (category `documentation`) | a new model deployment guide |

When an API or configuration change breaks existing usage, use `breaking`; when it adds a
new capability, use `feature`.

### Do not write a fragment for

- Internal refactors, logging, typing, readability cleanups
- Tests, CI, build scripts, development and debugging tools
- Documentation typos and small edits (complete guides for new scenarios are the exception)
- Plumbing that prepares a follow-up feature but is still invisible to users

### When in doubt, write one

A fragment costs a few lines of YAML and captures the context you have right now. A missing
entry is almost impossible to recover at release time, while an extra entry can simply be
dropped or merged by the release manager. The cost is asymmetric: prefer writing.

Scope note: "in doubt" means genuinely unsure whether users can observe the change. If you
know the change is internal, the decision is already made — do not write a fragment, and do
not write one whose only content is the implementation.

### Changes that cancel out before a release

The unit of a release note is the **net change relative to the previous release**, not the
individual PR. Two cases follow from this:

**A bugfix immediately following the bug it fixes:**

- **The buggy PR was already included in a released version.** The fix must have a
  `bugfix` fragment: users upgrading from that release carry the bug, and the entry states
  the symptom they may have hit. Note that at merge time you usually cannot know whether a
  release will be cut in between — this is why "prefer writing" wins for bugfixes.
- **The bug and the fix both land between two releases** (the bug never reached any wheel
  or image). The fix introduces no net change, so it needs no fragment of its own — but
  the buggy PR's **not-yet-archived fragment must be removed or amended in the fix PR**.
  Otherwise the release would announce a bug that no released artifact ever contained.

In short: a fix for a shipped bug writes a fragment; a fix for an unshipped bug deletes the
bug's fragment. Both actions happen in the fix PR.

**A revert of a PR merged after the last release:**

- The net change is zero, so the revert PR needs **no fragment** — a revert is not a
  bugfix, there is no symptom left for any user to hit.
- The reverted PR's **not-yet-archived fragment must be removed in the revert PR**,
  together with any docs or config-table entries it added. If the reverted PR shipped in a
  released version, the revert is instead a behavior change users will see when upgrading
  — write a fragment (usually `breaking`, or `bugfix` if it reverts a regression) stating
  what disappears and what to do.

## Fragment format

Copy `TEMPLATE.yaml` to `fragments/<PR number>.yaml` (for example `fragments/18123.yaml`) and
fill in the entries. Prefer one file per PR, with as many entries as needed inside. If a PR
genuinely needs a second file, name it `fragments/<PR number>-<slug>.yaml`. Split unrelated
changes into separate entries: each entry is categorized independently, and the release
manager may drop or merge entries at assembly time. If the PR's user-visible scope grows or
changes during review, update the fragment in the same PR.

```yaml
pr: 18123
entries:
  - category: feature
    text: Added DeepSeek V4.1 support on Model Runner V2.
  - category: performance
    text: Enabled torch.compile for DeepSeek V4.1.
    highlight: true
    model: deepseek-v4.1
    hardware: [A2, A3]
```

Field rules:

- `pr` (required): the PR number, matching the file name
- `entries` (required): one or more entries
- `category` (required): `feature`, `hardware`, `performance`, `dependency`, `breaking`,
  `config`, `documentation`, `bugfix`, or `other`
- `text` (required): one to three short English sentences written for users of the released
  artifact, not for reviewers — what changed for them and what they must do. Do not append
  the PR link; the release tooling adds it automatically. See
  [Writing the text](#writing-the-text) for the quality bar and per-category requirements.
- `highlight` (optional): `true` marks the entry as a candidate for the Highlights section.
  Use it sparingly — only for entries with the widest user impact (new model or hardware
  support, headline features); when in doubt, leave it unset
- `model` and `hardware` (optional): metadata used when building the hardware support matrix

The `text` is raw material for the release notes, not the final copy: the release manager may
merge, split, or polish entries while assembling the release. But raw material still has a
quality bar — vague fragments force the release manager to re-investigate the PR, which is
exactly what fragments exist to avoid.

## Writing the text

Write every entry for a reader who never sees the diff — a user deciding whether to upgrade
and whether the release matters for their deployment. A good entry answers three questions:

1. What changed for me?
2. Do I need to do anything (upgrade, set a flag, migrate)?
3. Is this relevant to my workload?

If only someone who read the diff can understand the entry, rewrite it from the user's side.

### Voice

- English, one to three short sentences, from the user's point of view ("Added ...",
  "Users can now ...", "Fixed ..."), not "We ..." or a commit log line.
- Name user-facing switches exactly: environment variables, `additional_config` keys, CLI
  flags, metric names. These are the user's interface; exact names make entries greppable
  and save a docs hunt. State defaults and escape hatches when behavior changes.
- Avoid implementation narration: file paths, class and function names, internal
  codenames, kernel-level mechanics. Mention a mechanism only when the user can act on it
  or it defines the boundary of the effect.
- No unquantified hype. "Significantly faster" says nothing; state the effect or the
  concrete mechanism instead. Never invent numbers.

### Per-category requirements

- `feature`: what the user can do now, how to enable it (exact switch), and whether it is
  on by default.
- `breaking`: what stops working; old behavior vs new behavior; the exact migration step;
  and what happens if the user does nothing (fails at startup, silently ignored, different
  output). See `fragments/20097.yaml` for a worked example.
- `config`: the exact name, the default value, and what it changes for the user.
- `performance`: the scenario — model, workload, hardware, and any switch required to get
  the improvement — and the effect. With a benchmark, give the metric, the magnitude, and
  the conditions the number depends on (e.g. "decode throughput +15% on A3 at TP4, 2K/4K
  input/output"); without one, describe precisely what is eliminated or reduced per step
  (e.g. "removes the per-step H2D sync that blocked AsyncScheduler"). Never invent
  numbers: run the benchmark or stay qualitative.
- `bugfix`: the symptom the user saw (hang, OOM, wrong output, stuck metric), the
  conditions that trigger it, and the affected models or features. Explain the root cause
  only if it helps the user recognize the bug. See `fragments/20207.yaml`.
- `dependency`: what moved from which version to which, and what the user must do (e.g.
  use the new release image).
- `documentation`: what the docs now cover, so users can find the guide.
- `hardware`: which products are newly supported, for which models and workloads, and any
  limitations or known gaps.
- `other`: avoid when any other category fits. If used, say plainly what the user notices.

Security fixes always need a fragment: state the impact and that users must upgrade.

### Examples

Weak → better pairs, reworded from real entries in this repository:

Performance entry that narrates the implementation instead of the outcome:

> Weak: `Reduced QLI sorting and merge work with an adaptive Top-K accumulator width, and batched candidate vector operations with AscendC Level-0 repeats on Ascend A3.`
>
> Better: `Improved DeepSeek V4.1 sparse-attention indexing performance on A3 long-context decode: redundant Top-K sorting and candidate vector work per step are reduced.` (Add the measured delta if the PR has a benchmark.)

Bugfix entry that names the internal mechanism instead of the symptom:

> Weak: `Fixed explicit and startup-plan KV-cache budgets bypassing multi-group allocation scaling.`
>
> Better: `Fixed KV-cache allocations that could exceed the configured memory budget in multi-group setups, which could cause unexpected OOM at startup or during memory replanning.`

Feature entry with a vague scope:

> Weak: `Enabled FlashMLA for GLM DSpark MLA drafts on supported hardware.`
>
> Better: `Enabled FlashMLA for GLM DSpark MLA drafts on A3, avoiding a sequence-length copy to the CPU on every drafting step.` (Name the hardware explicitly, or put it in the `hardware` field — "supported hardware" is not actionable.)

### Reviewer checklist

Reviewers confirm fragments together with the code, checking the same bar:

- Can a reader without the diff understand what changed and what to do?
- Performance entries: scenario and effect both present?
- Breaking/config entries: exact switch name and migration step present?
- Category correct? A behavior switch that is now on by default is `breaking`, not
  `feature`; a tuning knob shipped with its default changed is `config` (state the old
  default) or `breaking`.

## Review and release

- Reviewers confirm whether the fragment is needed and whether the wording meets the bar
  in [Writing the text](#writing-the-text) (see the reviewer checklist there), the same way
  they review any other part of the PR. There is no CI gate for fragments; the rules are
  enforced by review.
- At release time the release manager runs the aggregation script, reviews the audit list of
  code changes without fragments, and archives the consumed fragments into
  `release_notes/archive/<version>/` in the release PR.

```bash
python .agents/skills/vllm-ascend-release/scripts/collect_release_notes.py \
  --version v0.31.0 \
  --raw-commits output/v0.31.0/0-current-raw-commits.md \
  --output output/v0.31.0/1-fragments-draft.md \
  --archive
```

`--fragments` defaults to `release_notes/fragments`, `--archive-dir` to `archive/` next to the
fragments directory (so `release_notes/archive` by default), and `--repo` to
`vllm-project/vllm-ascend`.

Fragments are archived, never deleted: they are the raw material behind the final notes, so
the history of what each PR promised at the time keeps its value. Archiving is also the safe
choice: if the archive step is forgotten, the next release's audit flags those fragments as
"not found in the commit list" instead of silently losing them. The one exception is a
fragment for a change that never reached a released artifact (an unreleased bug's fix, or
a revert of an unreleased PR) — deleting that unarchived fragment is correct (see
[Changes that cancel out before a release](#changes-that-cancel-out-before-a-release)).
