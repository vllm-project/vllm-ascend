# AI Review Knowledge Base (vllm-ascend, RFC #21451)

Review rules fed to the AI review agent. Structure mirrors the domain
ownership already encoded in `.github/CODEOWNERS`, so the people who own a
module own the review rules for it.

## Layout

```text
instructions.md        # output protocol; maintained by all maintainers
allowlist.txt          # operators allowed to apply the label; maintainers
L1/checklist.md        # cross-domain global rules; all maintainers
domains.yaml           # coarse registry: path globs -> domain; maintainers
L2/<domain>/           # ONE directory per domain; owned by that domain's
   checklist.md        #   owners (same owners as the code it governs)
   lessons/*.md        #   optional experience cards, auto-collected by the
                       #   knowledge-base map as long as entries carry IDs
```

## How it is consumed

The orchestrator resolves the PR's changed files against `domains.yaml` and
tells the agent which domains the PR touches. The agent gets only a
knowledge-base map (file paths + rule IDs) per round and reads the full
checklists itself — progressive disclosure, so adding domains does not
inflate the prompt.

## Maintenance rules

- **Owner-gated by CODEOWNERS, for free.** Adding `/vllm_ascend/foo`-governing
  rules means: create `L2/<domain>/checklist.md`, add a `domains:` entry in
  `domains.yaml`, and append a CODEOWNERS line
  `/.github/agent-review/L2/<domain>/ @<owners>` (later entries win). From
  then on, editing that domain's rules requires those owners' approval —
  the same gate as editing the code.
- **Rule IDs are stable.** Checklist entries cite IDs like `L1#slug` /
  `L2-<domain>#slug`; agent findings must quote them. Renaming an ID is a
  breaking change for anything that cited it.
- **Entries must be checkable from a diff.** A rule the agent cannot verify
  by reading the PR diff belongs in human review docs, not here.
- Coarse domains only. Do not add per-file L2 directories; adjust the globs
  in `domains.yaml` instead.
