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

### Verify GitHub CLI Installation

Before starting the release process, verify that `gh` CLI is installed and authenticated:

```bash
# Check if gh is installed
gh --version

# If not installed, install gh CLI:
# Ubuntu/Debian
apt install gh -y

# macOS
brew install gh

# OpenEuler
yum install gh -y

# Check authentication status
gh auth status

# If not authenticated, login with:
gh auth login
```

Expected output for `gh auth status`:
```
github.com
  ✓ Logged in to github.com account <username> (keyring)
  - Active account: true
  - Git operations protocol: https
  - Token: gho_****
  - Token scopes: 'gist', 'read:org', 'repo', 'workflow'
```

**Required scopes**: `repo` (for creating issues, PRs, releases) and `workflow` (for triggering CI workflows).

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
│  └── Update version references (Docker/wheel built by CI automatically)     │
│                                                                             │
│  Phase 8: Release Execution (requires human review)                         │
│  ├── Human review & approval                                                │
│  ├── Merge release notes PR                                                 │
│  ├── Create GitHub release                                                  │
│  └── Verify automated pipelines (PyPI, Docker, ReadTheDocs)                 │
│                                                                             │
│  Phase 9: WeChat Article (微信公众号推文)                                     │
│  ├── Collect release statistics (commits, contributors)                     │
│  ├── Generate WeChat article from template                                  │
│  └── Review and publish to WeChat official account                          │
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
# 1. [Priority] List open PRs/issues in the release milestone
gh pr list --repo vllm-project/vllm-ascend \
  --state open \
  --search "milestone:${VERSION}" \
  --json number,title,url,labels

gh issue list --repo vllm-project/vllm-ascend \
  --state open \
  --search "milestone:${VERSION}" \
  --json number,title,url,labels

# 2. List open PRs with release-related labels
gh pr list --repo vllm-project/vllm-ascend \
  --state open \
  --label "release-blocker" \
  --json number,title,url

# 3. List PRs merged since last release
gh pr list --repo vllm-project/vllm-ascend \
  --state merged \
  --search "merged:>${LAST_RELEASE_DATE}" \
  --json number,title,mergedAt
```

**Priority Order:**
1. PRs/Issues in the release milestone - these are explicitly targeted for this release
2. PRs with `release-blocker` label - critical items that must be merged
3. Recently merged PRs - for tracking what's already included

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

This phase handles the complete release notes writing process, from fetching commits to producing the final release notes.

### 6.1 Fetch Commits

Fetch all commits between the previous and current version:

```bash
# Create output directory
mkdir -p output/${VERSION}

# Fetch commits with contributor statistics
uv run python scripts/fetch_commits.py \
  --owner vllm-project \
  --repo vllm-ascend \
  --base-tag ${LAST_VERSION} \
  --head-tag ${NEW_VERSION} \
  --stats \
  --output output/${VERSION}/0-current-raw-commits.md \
  --stats-output output/${VERSION}/0-contributor-stats.md
```

The script outputs:
- `0-current-raw-commits.md`: Raw commit list for analysis
- `0-contributor-stats.md`: Contributor statistics including new contributors

### 6.2 Analyze Commits

Create a CSV file to analyze each commit:

```bash
# Create analysis workspace
touch output/${VERSION}/1-commit-analysis-draft.csv
```

The CSV should have headers:
| Column | Description |
|--------|-------------|
| `title` | Commit title |
| `pr number` | PR number |
| `user facing impact/summary` | What users should know |
| `category` | Highlights/Features/Performance/etc. |
| `decision` | include/exclude/merge |
| `reason` | Why this decision |

### 6.3 Draft Release Notes

Create the initial draft following the category order:

```markdown
## v${VERSION} - ${DATE}

This is the first release candidate of v${VERSION} for vLLM Ascend.
Please follow the [official doc](https://docs.vllm.ai/projects/ascend/en/latest) to get started.

### Highlights
(Top 3-5 most impactful changes)

### Features
(New functionality)

### Hardware and Operator Support
(New hardware/operators)

### Performance
(Performance improvements)

### Dependencies
(Version upgrades)

### Deprecation & Breaking Changes
(Breaking changes)

### Documentation
(Doc updates)

### Others
(Bug fixes, misc)

### Known Issue
(Known limitations)
```

Save drafts to:
- `output/${VERSION}/2-highlights-note-draft.md` - Initial draft
- `output/${VERSION}/3-highlights-note-edit.md` - Reviewed/edited version

### 6.4 Release Notes Writing Guidelines

**Inclusion Criteria:**
- User experience improvements (CLI, error messages)
- Core features (PD Disaggregation, KVCache, Graph mode, CP/SP, quantization)
- Breaking changes and deprecations (always include)
- Significant infrastructure changes
- Major dependency updates (CANN/torch_npu/triton-ascend)
- Hardware compatibility expansions (310P, A2, A3)

**Writing Tips:**
- Focus on what users should know, not internal details
- Look up PR descriptions when uncertain: `gh pr view <number> --repo vllm-project/vllm-ascend`
- Group related changes together
- Include PR links: `[#12345](https://github.com/vllm-project/vllm-ascend/pull/12345)`

**Reference:**
- See `references/ref-past-release-notes-highlight.md` for style examples

### 6.5 Create Release Notes PR

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
| `.github/workflows/schedule_image_build_and_push.yaml` | Config |
| `.github/workflows/schedule_update_estimated_time.yaml` | Config |

### 7.2 Version Update Script

```bash
python scripts/update_version_references.py \
  --version ${VERSION} \
  --vllm-version ${VLLM_VERSION} \
  --feedback-issue ${FEEDBACK_ISSUE_URL}
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

**⚠️ Human Review Required**: Before executing the release, ensure all previous phases have been reviewed and approved by the release manager. This step requires explicit human confirmation.

**Current Approach (Manual)**:
For now, execute release steps manually through GitHub UI or CLI after human review:

1. **Merge release notes PR** - Review PR, ensure CI passes, then merge via GitHub UI
2. **Create GitHub release** - Go to GitHub Releases page, create new release with tag
3. **Verify automated pipelines** - Docker image and wheel package are built automatically by CI

**Future Approach (Automated)**:
Once the release process is mature and well-tested, consider:
- Adding a GitHub Actions workflow with manual trigger (`workflow_dispatch`)
- Requiring approval from release manager before workflow proceeds
- Automating all steps below with proper guards

**Manual Execution Commands** (for reference):

```bash
# 1. Merge release notes PR (after human review)
gh pr merge ${RELEASE_PR_NUMBER} --repo vllm-project/vllm-ascend --squash

# 2. Create GitHub release
gh release create ${VERSION} \
  --repo vllm-project/vllm-ascend \
  --title "vLLM Ascend ${VERSION}" \
  --notes-file release-notes.md \
  --target main

# 3. Verify automated pipelines (no action needed - CI handles these)
# - Docker image: quay.io/ascend/vllm-ascend:${VERSION}
# - PyPI package: https://pypi.org/project/vllm-ascend/${VERSION}
# - ReadTheDocs: https://app.readthedocs.org/dashboard/

# 4. Upload 310P wheel if applicable
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

## Phase 9: WeChat Article (微信公众号推文)

After release notes are finalized and the release is completed, generate a WeChat article for community broadcast.

### 9.1 Article Structure Template

The WeChat article follows a structured format with emojis for visual appeal:

| Section | Emoji | Description | Recommended Items |
|---------|-------|-------------|-------------------|
| **Opening Paragraph** | 🎉 | Version announcement + positioning + core highlights summary | 1 paragraph |
| **Statistics** | 🥳 | Number of commits, new contributors | 1 line |
| **Core Highlights** | 💥 | Top 2-3 most important features/optimizations | 2-3 items |
| **New Features** | 🆕 | New functionality, models, operators | 3-5 items |
| **Performance** | 🚀 | Performance improvements (include metrics when available) | 2-4 items |
| **Refactoring** | 🔨 | Code refactoring, dependency upgrades | 1-3 items |
| **Bug Fixes** | 🐞 | Important bug fixes | 3-5 items |
| **Quality/Testing** | 🛡️ | Test coverage, CI/CD improvements | 0-2 items |
| **Documentation** | 📄 | Documentation updates (can combine into 1 item) | 1 item |
| **Links** | ➡️ | Source code, quick start, installation guide | 3 links |

### 9.2 Article Template

```markdown
vLLM Ascend ${VERSION}版本发布🎉 此版本是针对vLLM v${VLLM_VERSION}系列版本首个RC版本，[1-2句核心亮点描述]。

🥳 本版本共计${COMMITS_COUNT}个commits，新增${NEW_CONTRIBUTORS_COUNT}位新开发者！
💥 [核心亮点1]
💥 [核心亮点2]
🆕 [新特性1]
🆕 [新特性2]
🆕 [新特性3]
🚀 [性能优化1，最好包含具体数据如"提升X%"]
🚀 [性能优化2]
🔨 [重构/依赖升级1]
🔨 [重构/依赖升级2]
🐞 修复 [重要bug1]
🐞 修复 [重要bug2]
🐞 修复 [重要bug3]
🛡️ [质量/测试改进]
📄 [文档更新汇总]

➡️ 源码地址：https://github.com/vllm-project/vllm-ascend/releases/tag/${VERSION}
➡️ 快速体验：https://vllm-ascend.readthedocs.io/en/latest/quick_start.html
➡️ 安装指南：https://vllm-ascend.readthedocs.io/en/latest/installation.html
```

### 9.3 Fetch Release Note from Release Tag

**Important**: WeChat articles are typically published after the release is complete. Always fetch the release note directly from the release tag, as it contains the most accurate and up-to-date information including the precise new contributor count.

```bash
# Fetch release note from release tag (recommended - most accurate source)
gh release view ${VERSION} --repo vllm-project/vllm-ascend --json body,name,tagName

# The release body contains:
# - Highlights, Features, Performance, Documentation sections
# - Bug fixes (Others section)
# - Dependencies and Known Issues
# - New Contributors list with exact count
```

**Why use release tag instead of other sources:**
- The release tag's "New Contributors" section is auto-generated by GitHub and is the most accurate
- Release notes in the tag may have last-minute updates not in the PR
- Dependencies and Known Issues sections are finalized at release time

### 9.4 Writing Guidelines

1. **Opening Paragraph**:
   - Start with version number and 🎉
   - Describe version positioning (RC/stable, which vLLM version)
   - Highlight 1-2 core themes of this release

2. **Content Selection**:
   - Prioritize user-facing features over internal refactoring
   - Include specific performance numbers when available
   - Group related items (e.g., multiple bug fixes for one feature)
   - Highlight breaking changes or dependency upgrades

3. **Language Style**:
   - Use concise, active voice
   - Avoid overly technical jargon
   - Keep each item to one line when possible
   - Use "完成支持/适配" for new features, "优化/提升" for performance

4. **Statistics from Release Tag**:
   - New contributor count: Count entries in "New Contributors" section of release body
   - For commits count (if needed): `git rev-list --count ${LAST_VERSION}..${VERSION}`

### 9.5 Example: v0.18.0rc1

```
vLLM Ascend v0.18.0rc1版本发布🎉 此版本是针对vLLM v0.18.0系列版本首个RC版本，重点完成了C8(INT8 KV cache)对GQA attention模型的支持，以及性能优化、问题修复等。

🥳 本版本新增9位新开发者，感谢社区开发者的持续贡献！
💥 C8(INT8 KV cache)支持GQA attention模型，同时适配DeepSeek-V3.1 PD分离场景
💥 DeepSeek模型通过新MLA算子支持A5硬件
🆕 Flash Comm V1支持VL模型的MLA，解除多模态服务限制
🆕 支持speculative decoding中target和draft模型使用不同attention backend
🆕 VL MoE模型支持SP，`sp_threshold`替换为vLLM原生`sp_min_token_num`
🆕 Qwen VL模型支持`w8a8_mxfp8`量化
🚀 Triton算子重编译优化，提升算子性能
🚀 Qwen3.5/Qwen3-Next GDN prefill路径优化，预构建chunk metadata减少h2d同步开销
🚀 FIA prefill context merge路径简化，提升运行时效率
🐞 torch-npu 和 triton-ascend 依赖版本更新，请参考官方release note
🐞 修复PD分离场景decode节点因DP节点shape不对齐导致卡住的问题
🐞 修复单卡部署多实例显存 OOM 问题
🐞 修复 DeepSeek v3.1 C8在MTP + full decode + full graph模式下的问题
🐞 修复`AscendModelSlimConfig`中量化配置key映射导致的权重加载报错问题

📄 更新Kimi-K2.5、GLM-4.7、DeepSeek-V3.2、MiniMax-M2.5及PD分离部署文档

➡️ 源码地址：
https://github.com/vllm-project/vllm-ascend/releases/tag/v0.18.0rc1
➡️ 快速体验：
https://vllm-ascend.readthedocs.io/en/v0.18.0/quick_start.html
➡️ 安装指南：
https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html
```

## Script Reference

### scripts/fetch_commits.py

Fetches all commits between two tags and generates contributor statistics.

**Arguments:**
- `--owner`: Repository owner (default: vllm-project)
- `--repo`: Repository name (default: vllm-ascend)
- `--base-tag`: Base tag (older version, e.g., v0.14.0)
- `--head-tag`: Head tag (newer version, e.g., v0.15.0rc1)
- `--output`: Output file for commits (default: 0-current-raw-commits.md)
- `--stats`: Generate contributor statistics
- `--stats-output`: Output file for statistics (default: 0-contributor-stats.md)
- `--sort`: Sort mode (chronological/alphabetical/reverse)
- `--include-date`: Include commit date in output
- `--token`: GitHub token (or use GITHUB_TOKEN env var)

**Output:**
- Commit list in markdown format with PR links
- Contributor statistics including new contributors

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

### references/ref-past-release-notes-highlight.md

Past release notes examples for style and category reference. Use this as a guide when writing new release notes to maintain consistency in:
- Section ordering and naming
- Writing style and tone
- Level of detail for different categories
- How to describe features, bug fixes, and breaking changes

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
