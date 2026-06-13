# LLM GitHub Action Skill

Create GitHub Actions workflows that call an LLM (vLLM/OpenAI-compatible endpoint) to review issues, PRs, or other GitHub events, then post the AI-generated feedback as a comment.

## When to Use

- User asks to create a bot that reviews issues or PRs with AI
- User wants an automated LLM-based workflow triggered by GitHub events
- User asks to adapt the issue review bot pattern to a new trigger (PR review, discussion review, etc.)

## Architecture: 5-Step Pipeline

Every LLM-based review workflow follows this pattern. Steps communicate via files (not `GITHUB_OUTPUT`), because LLM output can be multi-line and large.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ Extract  │───▶│ Prepare  │───▶│ Prepare  │───▶│ Call LLM │───▶│  Post    │
│  Input   │    │ Template │    │  System  │    │          │    │ Handler  │
│          │    │          │    │  Prompt  │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
 issue_type      template.txt    system_prompt   review_output     GitHub
   .txt                            .txt            .md           comment
```

### Step 1: Extract Input
Parse the event payload to extract structured data (issue type, PR metadata, etc.). Write to a plain text file.

**Script pattern**: `extract_input.py`
- Reads from `github.event.*` via environment variables
- Parses/extracts relevant fields
- Writes result to `--output` file (default: `input_type.txt`)

### Step 2: Prepare Template
Load the matching template (YAML issue template, PR template, etc.) based on the extracted type.

**Script pattern**: `prepare_template.py`
- Reads the type from `--input` file (from step 1)
- Maps type to template file path
- Parses template (YAML or markdown) into human-readable form
- Writes to `--output` file (default: `template.txt`)

### Step 3: Prepare System Prompt
Load the LLM system prompt from a file in the prompts directory.

**Script pattern**: `prepare_system_prompt.py`
- Reads prompt from a `prompts/` directory relative to the script
- Writes to `--output` file (default: `system_prompt.txt`)

### Step 4: Call LLM
Assemble the full prompt (system + template + event content) and call the LLM API.

**Script pattern**: `call_llm.py`
- Reads system prompt and template from files
- Reads event content from environment variables
- Calls OpenAI-compatible `/v1/chat/completions` endpoint
- Writes raw LLM output to `--output` file (default: `review_output.md`)

### Step 5: Post Handler
Post the LLM output as a GitHub comment (issue, PR, etc.).

**Script pattern**: `handle_review.py`
- Reads review from `--input` file
- Posts via GitHub API (`POST /repos/{owner}/{repo}/issues/{number}/comments`)
- Uses `GITHUB_TOKEN` for auth

## File Structure

```
.github/workflows/
├── bot_<name>_review.yaml          # Workflow definition
└── scripts/robot/
    ├── extract_input.py            # Step 1
    ├── prepare_template.py         # Step 2
    ├── prepare_system_prompt.py    # Step 3
    ├── call_llm.py                 # Step 4
    ├── handle_review.py            # Step 5
    ├── .gitignore                  # Contains: .env
    └── prompts/                    # System prompts directory
        └── system_prompt.txt
```

## Workflow YAML Template

```yaml
name: "<Name> Review Bot"
on:
  <trigger>:
    types: [<event_types>]

permissions:
  <resource>: write
  contents: read

jobs:
  review:
    runs-on: ubuntu-latest
    if: <guard_condition>
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install requests pyyaml

      - name: Extract input
        id: extract
        env:
          <event_env_vars>
        run: python .github/workflows/scripts/robot/extract_input.py --output input_type.txt

      - name: Prepare template
        id: template
        if: success()
        run: python .github/workflows/scripts/robot/prepare_template.py --input input_type.txt --output template.txt

      - name: Prepare system prompt
        id: sysprompt
        if: success()
        run: python .github/workflows/scripts/robot/prepare_system_prompt.py --output system_prompt.txt

      - name: Call LLM
        id: llm
        if: success()
        env:
          VLLM_BASE_URL: ${{ secrets.VLLM_BASE_URL }}
          VLLM_API_KEY: ${{ secrets.VLLM_API_KEY }}
          <event_env_vars>
        run: |
          python .github/workflows/scripts/robot/call_llm.py \
            --system-prompt system_prompt.txt \
            --template template.txt \
            --output review_output.md

      - name: Post handler
        id: post
        if: success()
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          <resource_id_env_vars>
          REPO: ${{ github.repository }}
        run: python .github/workflows/scripts/robot/handle_review.py --input review_output.md
```

## Key Design Decisions

1. **File-based step communication**: LLM output is multi-line and can be large. `GITHUB_OUTPUT` has size limits and escaping issues. Use files instead.

2. **Modular scripts**: Each step is a standalone Python script with `--input`/`--output` flags. This makes testing and debugging easy — you can run any step independently.

3. **Environment variables for secrets**: `VLLM_BASE_URL` and `VLLM_API_KEY` come from GitHub Secrets. Never hardcode API keys.

4. **`if: success()` guards**: Each step after step 1 uses `if: success()` to stop the pipeline if any previous step fails.

5. **Trigger guard**: The workflow-level `if:` condition filters events (e.g., only issues with recognized prefixes) to avoid unnecessary runs.

## Adapting for Different Triggers

### Issue Review → PR Review
| Aspect | Issue Review | PR Review |
|--------|-------------|-----------|
| Trigger | `issues: [opened]` | `pull_request: [opened]` |
| Guard | Check title prefix | Always run (or check draft) |
| Content | `github.event.issue.title` + `body` | `github.event.pull_request.title` + `body` + diff |
| Template | `.github/ISSUE_TEMPLATE/*.yml` | `.github/PULL_REQUEST_TEMPLATE.md` |
| Post to | `/issues/{number}/comments` | `/issues/{number}/comments` (PRs are also issues) |
| Permissions | `issues: write` | `pull-requests: write` |

### Key Changes for PR Review
1. **extract_input.py**: No type extraction needed (PRs don't have type prefixes). Can extract PR metadata (files changed, base branch, etc.)
2. **prepare_template.py**: Load the single PR template markdown file instead of mapping types to YAML templates
3. **call_llm.py**: Include PR diff in the user prompt (fetch via `GET /repos/{owner}/{repo}/pulls/{number}/files` or include `github.event.pull_request` body)
4. **System prompt**: Focus on code review criteria (correctness, style, testing, NPU considerations per AGENTS.md)

## Secrets Required

| Secret | Purpose |
|--------|---------|
| `VLLM_BASE_URL` | LLM API base URL (e.g., `https://api.deepseek.com`) |
| `VLLM_API_KEY` | API key for the LLM service |
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions for posting comments |

## Testing

Test scripts live in `tests/`:
- `collect_issues.py` — collect real issues for testing
- `test_issues_csv.py` — batch-test the pipeline against collected issues
- `test_<name>_review_prompt.py` — unit tests with curated test cases
