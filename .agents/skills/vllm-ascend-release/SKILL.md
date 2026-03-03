---
name: vllm-ascend-release
description: "End-to-end release management skill for vLLM Ascend. Creates release checklist issues, identifies critical bugs, runs functional tests, invokes release note generation, and guides through the complete release process."
---

# vLLM Ascend Release Skill

## Overview

This skill manages the complete end-to-end release process for vLLM Ascend, from creating the release checklist issue to final release announcement. It automates repetitive tasks while ensuring human oversight at critical decision points.

## When to Use This Skill

Use this skill when:
- Starting a new release cycle (RC or stable)
- The release manager needs to track release progress
- Preparing release artifacts (notes, documentation, tests)

## Prerequisites

- GitHub CLI (`gh`) authenticated with write access to `vllm-project/vllm-ascend`
- Access to Ascend NPU hardware for functional testing (or CI infrastructure)
- Python environment with `uv` for running scripts

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         vLLM Ascend Release Process                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 1: Initialization                                                    │
│  ├── Determine version & branch                                             │
│  ├── Create feedback issue                                                  │
│  └── Create release checklist issue                                         │
│                                                                             │
│  Phase 2: Bug Triage                                                        │
│  ├── Scan open bugs                                                         │
│  ├── Identify release-blocking bugs                                         │
│  └── Update checklist with bug list                                         │
│                                                                             │
│  Phase 3: PR Management                                                     │
│  ├── Identify must-merge PRs                                                │
│  └── Update checklist with PR list                                          │
│                                                                             │
│  Phase 4: Functional Testing                                                │
│  ├── Select representative models                                           │
│  ├── Run automated tests                                                    │
│  └── Update checklist with test results                                     │
│                                                                             │
│  Phase 5: Release Notes (invoke existing skill)                             │
│  ├── Generate release notes via vllm-ascend-release-note-writer             │
│  └── Create release notes PR                                                │
│                                                                             │
│  Phase 6: Documentation & Artifacts                                         │
│  ├── Update version references                                              │
│  ├── Prepare Docker image                                                   │
│  └── Prepare wheel package                                                  │
│                                                                             │
│  Phase 7: Release Execution                                                 │
│  ├── Merge release notes PR                                                 │
│  ├── Create GitHub release                                                  │
│  ├── Verify PyPI & Docker availability                                      │
│  └── Broadcast release                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Initialization

### 1.1 Gather Release Information

Prompt the user for:
- **Release Version**: e.g., `v0.15.0rc1`, `v0.15.0`
- **Release Branch**: typically `main`
- **Target Release Date**: e.g., `2026.03.15`
- **Release Manager**: GitHub username

### 1.2 Determine Previous Version

```bash
# Get the latest release tag
gh release list --repo vllm-project/vllm-ascend --limit 5

# Or check existing tags
git tag --sort=-creatordate | head -10
```

### 1.3 Create Feedback Issue

Create a community feedback issue for the release:

```bash
gh issue create --repo vllm-project/vllm-ascend \
  --title "[Feedback]: v${VERSION} Release Feedback" \
  --body "$(cat templates/feedback-issue-template.md)" \
  --label "feedback"
```

### 1.4 Create Release Checklist Issue

Use the template in `templates/release-checklist-template.md`:

```bash
# Generate the checklist from template
python scripts/generate_checklist.py \
  --version ${VERSION} \
  --branch ${BRANCH} \
  --date ${DATE} \
  --manager ${MANAGER} \
  --feedback-issue ${FEEDBACK_ISSUE_NUMBER} \
  --output release-checklist.md

# Create the issue
gh issue create --repo vllm-project/vllm-ascend \
  --title "[Release]: Release checklist for ${VERSION}" \
  --body-file release-checklist.md \
  --label "release"
```

## Phase 2: Bug Triage

### 2.1 Scan Open Bugs

Run the bug scanning script to identify potential release-blocking bugs:

```bash
python scripts/scan_release_bugs.py \
  --repo vllm-project/vllm-ascend \
  --output bug-analysis.md
```

The script analyzes bugs based on:
- **Severity indicators**: keywords like "crash", "data loss", "security", "regression"
- **User impact**: number of reactions, comments, linked issues
- **Recency**: bugs reported in the current release cycle
- **Labels**: `priority:high`, `regression`, `blocker`

### 2.2 Bug Prioritization Criteria

| Priority | Criteria | Action |
|----------|----------|--------|
| P0 - Blocker | Crashes, data corruption, security issues | Must fix before release |
| P1 - Critical | Major feature broken, significant regression | Should fix, may delay release |
| P2 - Important | Notable bugs affecting common workflows | Fix if possible |
| P3 - Normal | Minor issues, edge cases | Document as known issues |

### 2.3 Update Checklist

After review, update the release checklist issue with the identified bugs:

```bash
python scripts/update_checklist_section.py \
  --issue-number ${CHECKLIST_ISSUE} \
  --section "Bug need Solve" \
  --content-file bug-list.md
```

## Phase 3: PR Management

### 3.1 Identify Must-Merge PRs

Scan for PRs that should be included in the release:

```bash
# List open PRs with release-related labels
gh pr list --repo vllm-project/vllm-ascend \
  --state open \
  --label "release-blocker" \
  --json number,title,url

# List PRs merged since last release
gh pr list --repo vllm-project/vllm-ascend \
  --state merged \
  --search "merged:>${LAST_RELEASE_DATE}" \
  --json number,title,mergedAt
```

### 3.2 Update Checklist

Update the checklist with PRs that need to be merged:

```bash
python scripts/update_checklist_section.py \
  --issue-number ${CHECKLIST_ISSUE} \
  --section "PR need Merge" \
  --content-file pr-list.md
```

## Phase 4: Functional Testing

### 4.1 Select Representative Models

The skill automatically selects models that cover core functionalities:

| Model Category | Representative Model | Features Covered |
|----------------|---------------------|------------------|
| Dense LLM | Qwen3-8B | Basic inference, attention |
| MoE LLM | Qwen3-MoE-7B | Expert parallelism, MoE routing |
| VL Model | Qwen3-VL-7B | Multimodal, image processing |
| Long Context | Qwen3-32B-128k | Long context, memory management |
| Speculative | DeepSeek-V3 + MTP | Speculative decoding |

Model selection criteria:
1. **Coverage**: Each model tests different code paths
2. **Availability**: Models accessible in test environment
3. **Stability**: Models with known-good baselines
4. **Popularity**: Commonly used by the community

### 4.2 Run Automated Tests

```bash
python scripts/run_functional_tests.py \
  --config references/test-models.yaml \
  --output test-results.md \
  --hardware 910B \
  --timeout 3600
```

Test categories:
- **Startup Test**: Model loads successfully
- **Inference Test**: Basic generation works
- **Accuracy Test**: Output quality meets baseline
- **Performance Test**: Throughput within expected range
- **Feature Test**: Specific features (EP, graph mode, etc.)

### 4.3 Test Result Format

```markdown
### Functional Test Results

| Model | Startup | Inference | Accuracy | Performance | Features | Status |
|-------|---------|-----------|----------|-------------|----------|--------|
| Qwen3-8B | ✅ | ✅ | 98.5% | 1200 tok/s | Graph ✅ | PASS |
| Qwen3-MoE | ✅ | ✅ | 97.2% | 800 tok/s | EP ✅ | PASS |
| Qwen3-VL | ✅ | ✅ | 96.8% | 600 tok/s | MM ✅ | PASS |
| DeepSeek-V3 | ✅ | ✅ | 99.1% | 1500 tok/s | MTP ✅ | PASS |
```

### 4.4 Update Checklist with Results

```bash
python scripts/update_checklist_section.py \
  --issue-number ${CHECKLIST_ISSUE} \
  --section "Functional Test" \
  --content-file test-results.md
```

## Phase 5: Release Notes

### 5.1 Invoke Release Note Skill

This phase invokes the existing `vllm-ascend-release-note-writer` skill:

```
/release-note --base-tag ${LAST_VERSION} --head-tag ${NEW_VERSION}
```

Or manually:

```bash
cd .agents/skills/vllm-ascend-release-note-writer

# Fetch commits
uv run python scripts/fetch_commits-optimize.py \
  --base-tag ${LAST_VERSION} \
  --head-tag ${NEW_VERSION} \
  --output output/${VERSION}/0-current-raw-commits.md

# Follow the release-note-writer SKILL.md workflow
```

### 5.2 Create Release Notes PR

After release notes are finalized:

```bash
# Create branch
git checkout -b release/${VERSION}

# Make changes (see Phase 6 for full list)
# ...

# Create PR
gh pr create --repo vllm-project/vllm-ascend \
  --title "Release ${VERSION}" \
  --body "Release notes and version updates for ${VERSION}" \
  --label "release"
```

## Phase 6: Documentation & Artifacts

### 6.1 Files to Update

| File | Update Required |
|------|-----------------|
| `README.md` | Getting Started version, Branch section |
| `README.zh.md` | Same as above (Chinese) |
| `docs/source/faqs.md` | Feedback issue link |
| `docs/source/user_guide/release_notes.md` | Add new release notes |
| `docs/source/community/versioning_policy.md` | Compatibility matrix, release window |
| `docs/source/community/contributors.md` | New contributors |
| `docs/conf.py` | Package version |

### 6.2 Version Update Script

```bash
python scripts/update_version_references.py \
  --version ${VERSION} \
  --vllm-version ${VLLM_VERSION} \
  --feedback-issue ${FEEDBACK_ISSUE_URL}
```

### 6.3 Docker Image Preparation

Verify Docker build and push:

```bash
# Check CI workflow for Docker build
gh workflow view docker-build --repo vllm-project/vllm-ascend

# Verify image availability (after CI completes)
# Image will be at: quay.io/ascend/vllm-ascend:${VERSION}
```

### 6.4 Wheel Package Preparation

Verify wheel build:

```bash
# Check CI workflow for wheel build
gh workflow view wheel-build --repo vllm-project/vllm-ascend

# Verify package availability (after CI completes)
# Package will be at: https://pypi.org/project/vllm-ascend/${VERSION}
```

## Phase 7: Release Execution

### 7.1 Pre-Release Checklist

Before executing the release, verify:

- [ ] All P0/P1 bugs resolved or documented as known issues
- [ ] All must-merge PRs merged
- [ ] Functional tests passing
- [ ] Release notes reviewed and approved
- [ ] Documentation updated
- [ ] CI passing on release branch

### 7.2 Execute Release

```bash
# 1. Merge release notes PR
gh pr merge ${RELEASE_PR_NUMBER} --repo vllm-project/vllm-ascend --squash

# 2. Create GitHub release
gh release create ${VERSION} \
  --repo vllm-project/vllm-ascend \
  --title "vLLM Ascend ${VERSION}" \
  --notes-file release-notes.md \
  --target main

# 3. Generate docs on ReadTheDocs
# (Triggered automatically by release, verify at https://app.readthedocs.org/dashboard/)

# 4. Wait for PyPI availability
# Check: https://pypi.org/project/vllm-ascend/${VERSION}

# 5. Wait for Docker availability
# Check: https://quay.io/ascend/vllm-ascend:${VERSION}

# 6. Upload 310P wheel if applicable
gh release upload ${VERSION} \
  --repo vllm-project/vllm-ascend \
  vllm_ascend-${VERSION}-310p-*.whl
```

### 7.3 Post-Release

```bash
# 1. Broadcast release (prepare announcement)
python scripts/generate_announcement.py \
  --version ${VERSION} \
  --release-notes release-notes.md \
  --output announcement.md

# 2. Close release checklist issue
gh issue close ${CHECKLIST_ISSUE} \
  --repo vllm-project/vllm-ascend \
  --comment "Release ${VERSION} completed successfully!"
```

## Script Reference

### scripts/generate_checklist.py

Generates the release checklist issue body from template.

**Arguments:**
- `--version`: Release version (e.g., v0.15.0rc1)
- `--branch`: Release branch (default: main)
- `--date`: Target release date
- `--manager`: Release manager GitHub username
- `--feedback-issue`: Feedback issue number
- `--output`: Output file path

### scripts/scan_release_bugs.py

Scans GitHub issues to identify release-blocking bugs.

**Arguments:**
- `--repo`: Repository (default: vllm-project/vllm-ascend)
- `--days`: Look back period in days (default: 30)
- `--output`: Output file path

**Output:** Markdown file with prioritized bug list

### scripts/run_functional_tests.py

Runs functional tests on representative models.

**Arguments:**
- `--config`: Test configuration YAML file
- `--output`: Output file for results
- `--hardware`: Hardware type (910B, 310P, etc.)
- `--timeout`: Test timeout in seconds
- `--models`: Specific models to test (optional)

**Output:** Markdown table with test results

### scripts/update_checklist_section.py

Updates a specific section of the release checklist issue.

**Arguments:**
- `--issue-number`: Release checklist issue number
- `--section`: Section name to update
- `--content-file`: File containing new content
- `--append`: Append to section instead of replace

### scripts/update_version_references.py

Updates version references across documentation files.

**Arguments:**
- `--version`: New version
- `--vllm-version`: Compatible vLLM version
- `--feedback-issue`: Feedback issue URL

### scripts/generate_announcement.py

Generates release announcement for broadcasting.

**Arguments:**
- `--version`: Release version
- `--release-notes`: Release notes file
- `--output`: Output file path

## Templates

### templates/release-checklist-template.md

The release checklist issue template (see file for full template).

### templates/feedback-issue-template.md

The feedback collection issue template.

## References

### references/test-models.yaml

Configuration for functional test models, including:
- Model paths
- Expected performance baselines
- Feature requirements
- Test commands

### references/version-files.yaml

List of files that need version updates and their update patterns.

## Error Handling

### Common Issues

| Issue | Solution |
|-------|----------|
| GitHub API rate limit | Use authenticated requests, implement backoff |
| Test timeout | Increase timeout, check hardware availability |
| Model not found | Verify model path, check storage |
| CI failure | Check CI logs, retry or fix |

### Recovery Procedures

If the release process fails midway:

1. Check the release checklist issue for current state
2. Resume from the last incomplete step
3. Update checklist with failure notes
4. Notify release manager

## Important Notes

1. **Human Oversight**: This skill automates tasks but requires human approval at key decision points (bug prioritization, test results review, release approval).

2. **Idempotency**: Most scripts can be re-run safely. Issue updates use section replacement.

3. **Rollback**: If a release needs to be rolled back:
   - Delete the GitHub release
   - Revert the release notes PR
   - Update checklist issue with rollback notes

4. **Communication**: Keep the community informed through the feedback issue and release checklist.

5. **Testing**: Always run functional tests before release, even for RC versions.
