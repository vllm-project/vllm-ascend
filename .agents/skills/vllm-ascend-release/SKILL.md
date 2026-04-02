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

### 9.3 Writing Guidelines

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

4. **Statistics Collection**:
   ```bash
   # Count commits since last release
   git rev-list --count ${LAST_VERSION}..${VERSION}

   # Count new contributors
   git log ${LAST_VERSION}..${VERSION} --format='%aN' | sort -u > all_contributors.txt
   git log ..${LAST_VERSION} --format='%aN' | sort -u > prev_contributors.txt
   comm -23 all_contributors.txt prev_contributors.txt | wc -l
   ```

### 9.4 Example: v0.16.0rc1

```
vLLM Ascend 0.16.0rc1版本发布🎉 此版本是针对vLLM v0.16.0系列版本首个RC版本，重点完成了Qwen3-Omni量化适配优化和GLM5-W8A8量化支持，同时新增多个AscendC自定义算子并持续优化MoE模型性能。

🥳 欢迎社区开发者持续贡献！
💥 Qwen3-Omni量化适配及优化完成，推理性能显著提升
💥 GLM5-W8A8量化支持，通过参数化MLA维度实现
🆕 实验性支持 FabricMem Mode，提供ADXL/HIXL互联支持
🆕 310P新增 w8a8sc 量化方法支持
🆕 Mooncake Layerwise Connector 新增 kv_pool 支持
🆕 Eagle3 新增 QuaRot 量化支持（无embedding）
🚀 Qwen3-VL 卷积计算优化，TTFT提升0.95%，吞吐提升0.59%
🚀 MTP执行优化，重排状态更新操作提升性能
🚀 全局CPU分片及IRQ绑定优化，改善资源管理
🔨 EPLB profiling增强，支持专家热度对比和调整时间显示
🔨 CANN升级至8.5.1，请手动升级或使用官方镜像
🐞 修复 Eagle + Context Parallel 组合使用问题
🐞 修复 LoRA 精度问题（由上游vLLM变更引入）
🐞 修复多个 Qwen-Omni 量化相关问题
🐞 修复 triton rope_siso 实现bug
🛡️ 完善310P max-model-len参数说明及部署文档
📄 新增CPU绑定开发指南、Metrics使用文档及GLM4.x多节点部署教程

➡️ 源码地址：https://github.com/vllm-project/vllm-ascend/releases/tag/v0.16.0rc1
➡️ 快速体验：https://vllm-ascend.readthedocs.io/en/latest/quick_start.html
➡️ 安装指南：https://vllm-ascend.readthedocs.io/en/latest/installation.html
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
