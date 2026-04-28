---
name: vllm-ascend-lint
description: "Lint and format check skill for vLLM Ascend project. Handles ruff checks, format fixes, and SSL certificate issues in corporate network environments."
---

# vLLM Ascend Lint Skill

## Overview

This skill provides a practical workflow for running lint checks on the vLLM Ascend project, handling common issues like SSL certificate errors in corporate networks.

## When to Use This Skill

Use this skill when:
- Running lint checks before committing changes
- Fixing code style issues in vllm_ascend directory
- Encountering SSL certificate errors with pre-commit
- Need to quickly check and fix lint issues

## Prerequisites

- Python environment with ruff installed
- Pre-commit (optional, for full lint suite)

### Install Lint Dependencies

```bash
# Install minimal lint tools
pip install ruff pre-commit

# Or install all lint dependencies from requirements-lint.txt
pip install -r requirements-lint.txt
```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Lint Check Process                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Run ruff check                                      │
│  ├── python -m ruff check vllm_ascend/                       │
│  └── Check for errors (E501, etc.)                           │
│                                                              │
│  Step 2: Auto-fix with ruff                                  │
│  ├── python -m ruff check vllm_ascend/ --fix                 │
│  └── Fix auto-fixable issues                                 │
│                                                              │
│  Step 3: Format code                                         │
│  ├── python -m ruff format vllm_ascend/                      │
│  └── Reformat code style                                     │
│                                                              │
│  Step 4: Verify                                              │
│  └── python -m ruff check vllm_ascend/                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Commands

### Option 1: Minimal Lint (Recommended for SSL Issues)

When SSL certificate errors occur with pre-commit, use ruff directly:

```bash
cd vllm-ascend

# Check for lint errors
python -m ruff check vllm_ascend/

# Auto-fix lint errors
python -m ruff check vllm_ascend/ --fix

# Format code
python -m ruff format vllm_ascend/

# Verify all checks passed
python -m ruff check vllm_ascend/
```

### Option 2: Full Pre-commit Suite

For complete lint suite including codespell, typos, etc.:

```bash
cd vllm-ascend

# Standard run (requires SSL for downloading hooks)
pre-commit run --all-files

# CI mode (manual stage hooks)
pre-commit run --all-files --hook-stage manual
```

### Option 3: Workaround SSL Certificate Issues

If SSL certificate verification fails in corporate network:

```bash
# Method 1: Use ruff directly (bypasses pre-commit SSL issues)
python -m ruff check vllm_ascend/ --fix
python -m ruff format vllm_ascend/

# Method 2: Disable Python SSL verification (less secure)
$env:PYTHONHTTPSVERIFY=0; $env:CURL_CA_BUNDLE=""; python -m pre_commit run --all-files

# Method 3: Use NODE_TLS workaround (PowerShell)
$env:NODE_TLS_REJECT_UNAUTHORIZED=0; python -m pre_commit run --all-files
```

## Common Lint Rules

### Line Length (E501)

Maximum line length is **120 characters** (defined in pyproject.toml).

**Common Fix:**
```python
# Before (line too long)
time_budget -= self.profiling_chunk_manager.predict_time(num_new_tokens, request.num_computed_tokens)

# After (split across lines)
time_budget -= self.profiling_chunk_manager.predict_time(
    num_new_tokens, request.num_computed_tokens
)
```

### Ruff Configuration

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
exclude = [
    "tests/e2e/nightly/single_node/",
]

[tool.ruff.lint]
select = [
    "E",   # pycodestyle
    "F",   # Pyflakes
    "UP",  # pyupgrade
    "B",   # flake8-bugbear
    "SIM", # flake8-simplify
    "I",   # isort
    "G",   # flake8-logging-format
]
ignore = [
    "F405", "F403",  # star imports
    "E731",          # lambda expression assignment
    "B905",          # zip without `strict=`
    "B007",          # Loop control variable not used
    "UP032",         # f-string format
    "G004",          # logging format
    "B904",          # raise without from
    "SIM108",        # ternary operator
    "SIM102",        # nested if
]
```

## Lint Tools in Pre-commit

| Tool | Purpose | Notes |
|------|---------|-------|
| ruff-check | Python lint | Main lint tool |
| ruff-format | Python format | Code formatter |
| codespell | Spell check | Checks for typos |
| typos | Spell check | Additional typo checker |
| clang-format | C++ format | For C++ files (excludes csrc/) |
| markdownlint | Markdown lint | Only in CI (manual stage) |
| actionlint | GitHub Actions lint | Validates workflow files |

## Troubleshooting

### SSL Certificate Error

```
URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed>
```

**Solution:** Use ruff directly instead of pre-commit:
```bash
python -m ruff check vllm_ascend/ --fix
python -m ruff format vllm_ascend/
```

### Ruff Not Found

```
ruff: command not found
```

**Solution:** Use python module syntax:
```bash
python -m ruff check vllm_ascend/
```

### Pre-commit Hook Installation Failed

**Solution:** Install pre-commit hooks manually:
```bash
pre-commit install
```

## Expected Output

### Successful Lint Check

```
All checks passed!
```

### Successful Format

```
2 files reformatted, 332 files left unchanged
```

### Lint Error Example

```
E501 Line too long (121 > 120)
   --> vllm_ascend/core/scheduler_profiling_chunk.py:643:121
    |
643 |     time_budget -= self.profiling_chunk_manager.predict_time(num_new_tokens, request.num_computed_tokens)
    |                                                                                                                         ^
    |
```

## Integration with AGENTS.md

After lint fixes, follow AGENTS.md guidelines:
1. Run tests: `pytest tests/`
2. Commit with sign-off: `git commit -s`
3. Run full lint check before pushing: `bash format.sh ci`

## References

- AGENTS.md: Section "Quick Start for Contributors"
- pyproject.toml: Ruff configuration
- format.sh: Pre-commit wrapper script
- .pre-commit-config.yaml: Hook definitions

## Summary

For quick lint check in vLLM Ascend:

```bash
cd vllm-ascend
python -m ruff check vllm_ascend/ --fix && python -m ruff format vllm_ascend/
```

This handles most lint issues efficiently, especially in environments with SSL certificate restrictions.