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
│  Phase 4: Test Coverage Analysis                                            │
│  ├── Scan PRs for features/models without tests                             │
│  ├── Check previous feedback issue status                                   │
│  └── Update checklist with items needing manual testing                     │
│                                                                             │
│  Phase 5: Nightly Status                                                    │
│  ├── Get latest Nightly-A3 and Nightly-A2 runs                              │
│  ├── Analyze failures with extract_and_analyze.py                           │
│  └── Update checklist with nightly status table                             │
│                                                                             │
│  Phase 6: Release Notes (invoke existing skill)                             │
│  ├── Generate release notes via vllm-ascend-release-note-writer             │
│  └── Create release notes PR                                                │
│                                                                             │
│  Phase 7: Documentation & Artifacts                                         │
│  ├── Update version references                                              │
│  ├── Prepare Docker image                                                   │
│  └── Prepare wheel package                                                  │
│                                                                             │
│  Phase 8: Release Execution                                                 │
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

### 2.1 Scan Issues Since Last Release

Run the issue scanning script to browse all issues since the last release:

```bash
python scripts/scan_release_bugs.py \
  --repo vllm-project/vllm-ascend \
  --since-tag ${LAST_VERSION} \
  --output issue-scan.md
```

The script:
1. Gets the release date of the previous version (including rc versions)
2. Fetches all issues created since that date
3. Generates a report with:
   - **Flagged issues**: Automatically flagged based on engagement or keywords
   - **All open issues**: Quick browse table with titles
   - **Recently closed issues**: May be relevant for release notes

### 2.2 Human Review Process

The output is designed for quick human review:

1. **Check flagged issues first** - these have high engagement or concerning keywords
2. **Browse the open issues table** - scan titles, click to investigate if needed
3. **Review closed issues** - identify fixes that should be highlighted in release notes

### 2.3 Issue Flagging Criteria

Issues are automatically flagged when they have:
- High reactions (≥5) or many comments (≥5)
- Labels: `bug`, `regression`, `blocker`, `priority:high`, `critical`
- Keywords in title: crash, hang, freeze, oom, error, fail, etc.

### 2.4 Update Checklist

After manual review, add important bugs to the release checklist:

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

## Phase 4: Test Coverage Analysis

### 4.1 Identify Features/Models Needing Testing

CI already covers most test cases. Manual testing is only needed for:
- **New features** merged without test cases
- **New models** added due to environment constraints (e.g., CI doesn't have the model)
- **Issues** reported in the previous release's feedback

Run the test coverage scanner:

```bash
python scripts/scan_test_coverage.py \
  --repo vllm-project/vllm-ascend \
  --since-tag ${LAST_VERSION} \
  --feedback-issue ${PREVIOUS_FEEDBACK_ISSUE} \
  --output test-coverage-analysis.md
```

This script:
1. Scans PRs merged since the last release
2. Identifies features/models without corresponding test files
3. Checks the previous feedback issue for unresolved problems

### 4.2 Review the Analysis

The output categorizes items:

**Features/Models Needing Manual Testing:**
- New model support (e.g., Kimi K2.5, GLM-5)
- Features that couldn't be tested in CI

**Previous Feedback Status:**
- Unresolved issues from the feedback thread
- Items that need manual verification

### 4.3 Manual Testing Checklist

For items identified above, perform manual testing:

```markdown
#### Manual Testing Required

- [ ] Model: Kimi K2.5 - Basic inference works
- [ ] Model: GLM-5 - Multimodal features work
- [ ] Feature: Expert parallel with 8 GPUs
- [ ] Feedback: User reported slow startup (verify fixed)
```

### 4.4 Update Checklist with Results

```bash
python scripts/update_checklist_section.py \
  --issue-number ${CHECKLIST_ISSUE} \
  --section "Functional Test" \
  --content-file test-results.md
```

## Phase 5: Nightly Status

### 5.1 Analyze Nightly CI Runs

Get the latest Nightly-A3 and Nightly-A2 CI runs and analyze failures:

```bash
python scripts/scan_nightly_status.py \
  --repo vllm-project/vllm-ascend \
  --output nightly-status.md
```

This script:
1. Fetches the latest Nightly-A3 and Nightly-A2 workflow runs
2. Calls `extract_and_analyze.py` (from main2main-error-analysis skill) for failed runs
3. Extracts and categorizes errors:
   - **Code Bugs**: Real issues that need fixing
   - **Environment Flakes**: Transient issues (network, disk, etc.)

### 5.2 Review Output

The output includes:

| Workflow | Status | Failed Jobs | Code Bugs | Env Flakes | Run |
|----------|--------|-------------|-----------|------------|-----|
| Nightly-A3 | ✅ success | 0/15 | 0 | 0 | [#123](url) |
| Nightly-A2 | ❌ failure | 3/12 | 2 | 1 | [#124](url) |

For failed runs, it also shows:
- Code bugs that need fixing before release
- Failed test cases
- Environment flakes (informational)

### 5.3 Update Checklist

```bash
python scripts/update_checklist_section.py \
  --issue-number ${CHECKLIST_ISSUE} \
  --section "Nightly Status" \
  --content-file nightly-status.md
```

## Phase 6: Release Notes

### 6.1 Invoke Release Note Skill

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

### 6.2 Create Release Notes PR

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

## Phase 7: Documentation & Artifacts

### 7.1 Files to Update

| File | Update Required |
|------|-----------------|
| `README.md` | Getting Started version, Branch section |
| `README.zh.md` | Same as above (Chinese) |
| `docs/source/faqs.md` | Feedback issue link |
| `docs/source/user_guide/release_notes.md` | Add new release notes |
| `docs/source/community/versioning_policy.md` | Compatibility matrix, release window |
| `docs/source/community/contributors.md` | New contributors |
| `docs/conf.py` | Package version |

### 7.2 Version Update Script

```bash
python scripts/update_version_references.py \
  --version ${VERSION} \
  --vllm-version ${VLLM_VERSION} \
  --feedback-issue ${FEEDBACK_ISSUE_URL}
```

### 7.3 Docker Image Preparation

Verify Docker build and push:

```bash
# Check CI workflow for Docker build
gh workflow view docker-build --repo vllm-project/vllm-ascend

# Verify image availability (after CI completes)
# Image will be at: quay.io/ascend/vllm-ascend:${VERSION}
```

### 7.4 Wheel Package Preparation

Verify wheel build:

```bash
# Check CI workflow for wheel build
gh workflow view wheel-build --repo vllm-project/vllm-ascend

# Verify package availability (after CI completes)
# Package will be at: https://pypi.org/project/vllm-ascend/${VERSION}
```

## Phase 8: Release Execution

### 8.1 Pre-Release Checklist

Before executing the release, verify:

- [ ] All P0/P1 bugs resolved or documented as known issues
- [ ] All must-merge PRs merged
- [ ] Functional tests passing
- [ ] Release notes reviewed and approved
- [ ] Documentation updated
- [ ] CI passing on release branch

### 8.2 Execute Release

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

### 8.3 Post-Release

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

Scans GitHub issues since the last release for human review.

**Arguments:**
- `--repo`: Repository (default: vllm-project/vllm-ascend)
- `--since-tag`: Previous release tag (including rc versions)
- `--state`: Issue state filter (open, closed, all; default: all)
- `--output`: Output file path

**Output:** Markdown report with:
- Flagged issues (auto-detected as important)
- All open issues table for quick browsing
- Recently closed issues summary

### scripts/scan_test_coverage.py

Identifies features/models that need manual testing.

**Arguments:**
- `--repo`: Repository (default: vllm-project/vllm-ascend)
- `--since-tag`: Previous release tag
- `--feedback-issue`: Previous release feedback issue number (optional)
- `--output`: Output file path

**Output:** Markdown report with:
- Features/models merged without test coverage
- Previous feedback issue status (resolved/unresolved)

### scripts/scan_nightly_status.py

Scans Nightly CI status for release readiness.

**Arguments:**
- `--repo`: Repository (default: vllm-project/vllm-ascend)
- `--output`: Output file path

**Output:** Markdown report with:
- Summary table of Nightly-A3 and Nightly-A2 status
- Code bugs that need fixing (from extract_and_analyze.py)
- Environment flakes (informational)
- Failed test cases

**Dependencies:**
- Calls `main2main-error-analysis/scripts/extract_and_analyze.py` for detailed analysis

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
