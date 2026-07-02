# Deliverables

## Required outputs in the current repo

1. One final signed commit containing the adaptation changes.
2. A concise but complete Chinese adaptation report at `docs/source/tutorials/models/<ModelName>-adaptation-report.md`.
3. A compact Chinese runbook that explains service startup and OpenAI-compatible validation.
4. A test-config YAML at `tests/e2e/models/configs/<ModelName>.yaml`.
5. A tutorial document at `docs/source/tutorials/models/<ModelName>.md`.
6. An update to `docs/source/tutorials/models/index.md`.
7. If needed, an exclusion entry for the adaptation report in `pyproject.toml`.
8. A GitHub issue comment containing the SKILL.md workflow or AI-assisted summary.

## Validation discipline

- Always provide log paths for key claims.
- Keep docs aligned with the latest known-good launch path.
- Report pass or fail for ACLGraph, EP, flashcomm1, MTP, and multimodal.
- Mark EP and flashcomm1 as not applicable for non-MoE models.
- Include the `128k + bs16` baseline result or an explicit reason if it is not feasible.
- Name the ModelScope reference dataset used for any accuracy statement.
- Do not sign off with dummy-only evidence.
- Startup alone is insufficient; include first-request smoke evidence.

## Suggested final response structure

- What changed
- What went well or badly
- Validation performed
- Commit hash and changed files
- Optional next step
