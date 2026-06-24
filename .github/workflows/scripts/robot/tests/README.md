# Robot Tests

Tests for the Issue/PR review bot pipeline (`.github/workflows/scripts/robot/`).

## Files

| File | Purpose |
| ------ | --------- |
| `test_scenarios.py` | E2E lifecycle tests — creates real issues/PRs, polls bot, verifies state transitions |
| `test_csv.py` | Batch LLM evaluation & judge — calls the LLM with production prompts on CSV datasets, supports evaluate and judge modes |
| `collect_issues.py` | Data collection — fetches issues from GitHub, stratified by state (open/solved/closed) |
| `collect_prs.py` | Data collection — fetches PRs from GitHub, stratified by time range (open/merged/closed) |
| `SCENARIOS.md` | Formal scenario spec — all issue (I1-I6) and PR (P1-P12) state transitions |
| `issues.csv` | Dataset — ~100 real issues for LLM evaluation |
| `prs.csv` | Dataset — ~60 real PRs for LLM evaluation |

## Prerequisites

```bash
export GITHUB_TOKEN=$(gh auth token)      # for E2E and collection scripts
export VLLM_BASE_URL=http://localhost:8000 # for LLM evaluation
export VLLM_API_KEY=EMPTY
```

## Quick Start

### Collect data

```bash
# Regenerate issue dataset (balanced open/solved/closed)
python collect_issues.py --output issues.csv

# Regenerate PR dataset (open/merged/closed)
python collect_prs.py --output prs.csv
```

State mapping for issues:

- `open` — issue is open
- `solved` — closed with `state_reason: completed`
- `closed` — closed with `state_reason: not_planned` or null (older issues)

State mapping for PRs:

- `open` — PR is open
- `merged` — PR was merged
- `closed` — PR was closed without merging

### Run E2E lifecycle tests

```bash
# Issue scenarios only (I1-I6)
python test_scenarios.py --repo owner/repo --mode issue

# PR scenarios only (P1-P12)
python test_scenarios.py --repo owner/repo --mode pr

# All scenarios
python test_scenarios.py --repo owner/repo --mode all
```

Refer to `SCENARIOS.md` for the full scenario matrix.

### Run LLM evaluation

```bash
# Evaluate issues (calls LLM, writes results back to CSV)
python test_csv.py --mode issue --input issues.csv

# Evaluate PRs
python test_csv.py --mode pr --input prs.csv

# Judge evaluation correctness (audits LLM output)
python test_csv.py --mode issue_judge --input issues.csv
python test_csv.py --mode pr_judge --input prs.csv

# Resume from a specific index
python test_csv.py --mode issue --input issues.csv --start 10 --limit 20

# Re-process all rows (skip existing results off)
python test_csv.py --mode issue --input issues.csv --skip-existing

# Retry rows with malformed eval output or error judge results
python test_csv.py --mode issue --input issues.csv --retry-errors
python test_csv.py --mode issue_judge --input issues.csv --retry-errors

# Use a different output file
python test_csv.py --mode issue --input issues.csv --output results.csv
```

The script writes `expected_ok_dsv4` (bool) and `deepseek_v4_flash_output` (raw LLM response) back to the CSV.
