# Deliverables

## Required outputs in current repo

1. One final signed commit (`git commit -sm ...`) containing the adaptation changes.
2. Chinese analysis report doc at `docs/source/tutorials/models/<ModelName>-adaptation-report.md`（精简但完整）:
   - model architecture summary
   - incompatibility root causes
   - code changes and rationale
   - startup and inference verification evidence
   - accuracy baseline source and dataset choice（prefer ModelScope `gsm8k`; if absent, state which fallback dataset score was used）
   - feature status matrix（supported / unsupported / checkpoint-missing / not-applicable）
   - max model len: config theoretical vs runtime practical
   - dummy-vs-real validation matrix（what dummy proved / what only real proved）
   - false-ready cases and final resolution path（if any）
   - fallback ladder evidence（which fallback was tried, what changed）
3. Chinese compact runbook:
   - how to start server in `/workspace` (direct command, default `:8000`)
   - how to run OpenAI-compatible validation
   - optional eager fallback command
   - optional `TORCHDYNAMO_DISABLE=1` fallback command (if relevant)
4. Test config YAML at `tests/e2e/models/configs/<ModelName>.yaml` — must include `model_name`, `hardware`, `tasks` with accuracy metrics (name + value), and `num_fewshot`. Use accuracy results from evaluation to populate metric values. Follow the schema of existing configs (e.g. `Qwen3-8B.yaml`). When choosing the reference task, prefer the ModelScope `gsm8k` score for the matching model size; if unavailable, use another available ModelScope dataset score and record that fallback.
5. Tutorial doc at `docs/source/tutorials/models/<ModelName>.md` — must follow the standard template: Introduction, Supported Features, Environment Preparation (with docker tabs for A2/A3), Deployment (with serve script), Functional Verification (with curl example), Accuracy Evaluation, Performance. Fill in model-specific details (HF path, hardware requirements, TP size, max-model-len, served-model-name, sample curl, accuracy table). The accuracy table should use the same baseline rule: prefer ModelScope `gsm8k`, otherwise clearly label the fallback dataset.
6. Update `docs/source/tutorials/models/index.md` to include the tutorial entry and the adaptation report entry.
7. If the adaptation report is stored under `docs/source/tutorials/models/` and is not tutorial-style, add it to `[tool.check_docs_yaml_sync].exclude` in `pyproject.toml`.
8. Post SKILL.md content or AI-assisted workflow summary as a comment on the originating GitHub issue.

## Commit discipline

- Keep one signed commit for code changes in the current working repo.
- If implementation occurred in `/vllm-workspace/*`, backport minimal final diff to current repo before commit.
- Keep diff scoped to target model adaptation.

## Validation discipline

- Always provide log file paths for key claims.
- Keep docs synchronized with latest successful test mode (do not leave stale command variants as default).
- Final report must include pass/fail reason for each key feature attempt: ACLGraph / EP / flashcomm1 / MTP / multimodal.
- EP and flashcomm1 are MoE-only checks; for non-MoE models mark as not-applicable with evidence.
- Final report should include baseline capacity result (`128k + bs16`) or explicit reason if not feasible.
- Any accuracy claim, table, or acceptance statement must name the ModelScope reference dataset used. `gsm8k` is the default baseline; if unavailable for the matching model size, explicitly state the fallback dataset.
- Dummy-first can be used to speed up iterations, but real-weight gate is mandatory before final sign-off.
- Startup-only evidence is insufficient; include first-request smoke results.

## Suggested final response structure

- What changed
- What went well / what went wrong
- Validation performed
- Commit hash and changed files
- Optional next step
