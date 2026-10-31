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
| Measurable or clearly described performance improvements | FLA NPU backend, fewer H2D syncs |
| Fixes for user-reported bugs or issues affecting correctness and stability | hangs, OOM, wrong outputs |

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

## Fragment format

Copy `TEMPLATE.yaml` to `fragments/<PR number>.yaml` (for example `fragments/18123.yaml`) and
fill in the entries. Prefer one file per PR, with as many entries as needed inside. If a PR
genuinely needs a second file, name it `fragments/<PR number>-<slug>.yaml`.

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
- `text` (required): one or two English sentences from the user's point of view. Do not
  append the PR link; the release tooling adds it automatically
- `highlight` (optional): `true` marks the entry as a candidate for the Highlights section
- `model` and `hardware` (optional): metadata used when building the hardware support matrix

The `text` is raw material for the release notes, not the final copy: the release manager may
merge, split, or polish entries while assembling the release.

## Review and release

- Reviewers confirm whether the fragment decision and wording are right, the same way they
  review any other part of the PR. There is no CI gate for fragments; the rules above are
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
"not found in the commit list" instead of silently losing them.
