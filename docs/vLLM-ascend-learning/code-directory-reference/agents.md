# .agents：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **20 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `.agents/`

开发辅助技能与操作说明，不参与模型前向执行。

| 子目录 | 职责 |
| --- | --- |
| [skills/](../../../.agents/skills) | 开发辅助 skills 指引与参考资源；不进入推理热路径。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../.agents/README.md) | 开发辅助的 README 指引、模板或参考资源。 | 标题：vLLM Ascend skills |

## `.agents/skills/`

开发辅助 skills 指引与参考资源；不进入推理热路径。

| 子目录 | 职责 |
| --- | --- |
| [vllm-ascend-model-adapter/](../../../.agents/skills/vllm-ascend-model-adapter) | 开发辅助 vllm-ascend-model-adapter 指引与参考资源；不进入推理热路径。 |
| [vllm-ascend-release/](../../../.agents/skills/vllm-ascend-release) | 开发辅助 vllm-ascend-release 指引与参考资源；不进入推理热路径。 |

## `.agents/skills/vllm-ascend-model-adapter/`

开发辅助 vllm-ascend-model-adapter 指引与参考资源；不进入推理热路径。

| 子目录 | 职责 |
| --- | --- |
| [references/](../../../.agents/skills/vllm-ascend-model-adapter/references) | 开发辅助 references 指引与参考资源；不进入推理热路径。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [SKILL.md](../../../.agents/skills/vllm-ascend-model-adapter/SKILL.md) | 描述该开发技能的触发、步骤和交付要求；不会在推理服务中自动运行。 | 标题：vLLM Ascend Model Adapter |

## `.agents/skills/vllm-ascend-model-adapter/references/`

开发辅助 references 指引与参考资源；不进入推理热路径。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [deliverables.md](../../../.agents/skills/vllm-ascend-model-adapter/references/deliverables.md) | 开发辅助的 deliverables 指引、模板或参考资源。 | 标题：Deliverables |
| [fp8-on-npu-lessons.md](../../../.agents/skills/vllm-ascend-model-adapter/references/fp8-on-npu-lessons.md) | 开发辅助的 fp8-on-npu-lessons 指引、模板或参考资源。 | 标题：FP8-on-NPU Lessons |
| [multimodal-ep-aclgraph-lessons.md](../../../.agents/skills/vllm-ascend-model-adapter/references/multimodal-ep-aclgraph-lessons.md) | 开发辅助的 多模态、设备图 指引、模板或参考资源。 | 标题：Multimodal + EP + ACLGraph Lessons |
| [troubleshooting.md](../../../.agents/skills/vllm-ascend-model-adapter/references/troubleshooting.md) | 开发辅助的 troubleshooting 指引、模板或参考资源。 | 标题：Troubleshooting |
| [workflow-checklist.md](../../../.agents/skills/vllm-ascend-model-adapter/references/workflow-checklist.md) | 开发辅助的 检查 指引、模板或参考资源。 | 标题：Workflow Checklist |

## `.agents/skills/vllm-ascend-release/`

开发辅助 vllm-ascend-release 指引与参考资源；不进入推理热路径。

| 子目录 | 职责 |
| --- | --- |
| [references/](../../../.agents/skills/vllm-ascend-release/references) | 开发辅助 references 指引与参考资源；不进入推理热路径。 |
| [scripts/](../../../.agents/skills/vllm-ascend-release/scripts) | 开发辅助 scripts 指引与参考资源；不进入推理热路径。 |
| [templates/](../../../.agents/skills/vllm-ascend-release/templates) | 开发辅助 templates 指引与参考资源；不进入推理热路径。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [SKILL.md](../../../.agents/skills/vllm-ascend-release/SKILL.md) | 描述该开发技能的触发、步骤和交付要求；不会在推理服务中自动运行。 | 标题：vLLM Ascend Release Skill |

## `.agents/skills/vllm-ascend-release/references/`

开发辅助 references 指引与参考资源；不进入推理热路径。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ref-past-release-notes-highlight.md](../../../.agents/skills/vllm-ascend-release/references/ref-past-release-notes-highlight.md) | 开发辅助的 ref-past-release-notes-highlight 指引、模板或参考资源。 | 标题：v0.14.0rc1 - 2026.01.26 |
| [version-files.yaml](../../../.agents/skills/vllm-ascend-release/references/version-files.yaml) | 开发辅助的 version-files 指引、模板或参考资源。 | 顶层项：`files`, `update_order`, `validation` |

## `.agents/skills/vllm-ascend-release/scripts/`

开发辅助 scripts 指引与参考资源；不进入推理热路径。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [fetch_commits.py](../../../.agents/skills/vllm-ascend-release/scripts/fetch_commits.py) | 开发辅助的 fetch_commits 指引、模板或参考资源。 | 注释：Fetch all commits between two tags from a GitHub repository.；定义：`get_github_token`, `resolve_tag_to_sha`, `resolve_commit_sha` |
| [generate_announcement.py](../../../.agents/skills/vllm-ascend-release/scripts/generate_announcement.py) | 开发辅助的 generate_announcement 指引、模板或参考资源。 | 注释：Generate release announcement for broadcasting.；定义：`extract_highlights`, `generate_github_announcement`, `generate_short_announcement` |
| [generate_checklist.py](../../../.agents/skills/vllm-ascend-release/scripts/generate_checklist.py) | 开发辅助的 检查 指引、模板或参考资源。 | 注释：Generate release checklist issue body from template.；定义：`load_template`, `substitute_variables`, `main` |
| [scan_nightly_status.py](../../../.agents/skills/vllm-ascend-release/scripts/scan_nightly_status.py) | 开发辅助的 scan_nightly_status 指引、模板或参考资源。 | 注释：Scan Nightly CI status for release readiness.；定义：`NightlyRun`, `gh_api_json`, `get_latest_workflow_run`, `run_extract_and_analyze` |
| [scan_release_bugs.py](../../../.agents/skills/vllm-ascend-release/scripts/scan_release_bugs.py) | 开发辅助的 scan_release_bugs 指引、模板或参考资源。 | 注释：Scan GitHub issues since the last release to identify bugs and problems.；定义：`Issue`, `run_gh_command`, `get_release_date`, `fetch_issues_since` |
| [scan_test_coverage.py](../../../.agents/skills/vllm-ascend-release/scripts/scan_test_coverage.py) | 开发辅助的 scan_test_coverage 指引、模板或参考资源。 | 注释：Scan for features and new models that need manual testing.；定义：`FeatureItem`, `FeedbackItem`, `run_gh_command`, `get_release_date`, `fetch_merged_prs` |
| [update_checklist_section.py](../../../.agents/skills/vllm-ascend-release/scripts/update_checklist_section.py) | 开发辅助的 检查 指引、模板或参考资源。 | 注释：Update a specific section of the release checklist issue.；定义：`run_gh_command`, `get_issue_body`, `update_issue_body` |
| [update_version_references.py](../../../.agents/skills/vllm-ascend-release/scripts/update_version_references.py) | 开发辅助的 update_version_references 指引、模板或参考资源。 | 注释：Update version references across documentation files.；定义：`load_config`, `expand_glob_pattern`, `build_regex_pattern` |

## `.agents/skills/vllm-ascend-release/templates/`

开发辅助 templates 指引与参考资源；不进入推理热路径。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [feedback-issue-template.md](../../../.agents/skills/vllm-ascend-release/templates/feedback-issue-template.md) | 开发辅助的 feedback-issue-template 指引、模板或参考资源。 | 标题：vLLM Ascend ${VERSION} Release Feedback |
| [release-checklist-template.md](../../../.agents/skills/vllm-ascend-release/templates/release-checklist-template.md) | 开发辅助的 检查 指引、模板或参考资源。 | 标题：Release Checklist |
