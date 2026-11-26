# L2 Release Note Checklist

Rules for anything touching `release_notes/` and for judging whether a PR
needs a fragment at all. Owned by the release maintainers. Entries cite
stable IDs (`L2-rn#slug`); the audience is the user of a released wheel or
Docker image, never the PR reviewer.

## Does this PR need a fragment?

- `L2-rn#fragment-required`: A PR with a user-facing change ships
  `release_notes/fragments/<PR number>.yaml` in the same PR. User-facing =
  new models/hardware/operators, new features, changed user-facing
  APIs/env vars/config, breaking changes or default behavior changes,
  dependency/toolchain updates (CANN, torch_npu, triton-ascend, vLLM),
  performance changes, and bug fixes a user could have hit.
- `L2-rn#no-fragment-for-internal`: Internal refactors, code moves,
  logging/typing cleanups, tests, CI/build scripts, dev tooling and doc
  typo fixes need NO fragment. A fragment whose honest summary is
  "refactored X" or "prepared Y for Z" must not exist — flag it for
  removal, the release manager drops extras but a wrong one ships.
- `L2-rn#cancelled-pr-fragment`: A fix for a bug that never shipped in a
  released artifact needs no fragment of its own, and a fix/revert of an
  unreleased PR must remove that PR's not-yet-archived fragment.

## What the text must say

- `L2-rn#user-language`: Say what the user gets, not what the code does.
  Name user-facing switches exactly (env vars, config keys, CLI flags);
  skip file, class and kernel-level narration.
- `L2-rn#perf-scenario`: Performance entries state the scenario (model,
  workload, hardware) and the measured effect. Never invent numbers; if
  the PR shows no measurement, flag it.
- `L2-rn#breaking-migration`: Breaking/config entries give old behavior →
  new behavior, the exact switch, the migration step, and what happens if
  the user does nothing.
- `L2-rn#bugfix-symptom`: Bug fixes describe the user-visible symptom and
  when it occurs, not the internal root cause.
- `L2-rn#yaml-format`: The fragment follows `release_notes/TEMPLATE.yaml`
  (categories and required fields); a malformed fragment fails aggregation.
