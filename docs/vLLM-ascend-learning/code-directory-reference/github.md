# .github：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **119 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `.github/`

GitHub 协作与自动化：CI、工作流、Issue/PR 模板和版本参考。

| 子目录 | 职责 |
| --- | --- |
| [ISSUE_TEMPLATE/](../../../.github/ISSUE_TEMPLATE) | 提交问题、功能建议等使用的结构化模板。 |
| [actions/](../../../.github/actions) | 可复用的 GitHub Actions 步骤和环境准备。 |
| [workflows/](../../../.github/workflows) | 定义测试、构建、发布、文档和其他仓库自动化工作流。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CODEOWNERS](../../../.github/CODEOWNERS) | CODEOWNERS 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：PULL_REQUEST_TEMPLATE。 | 标题：What this PR does / why we need it? |
| [TEST_README.md](../../../.github/TEST_README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：TEST_README。 | 标题：CI Workflow Guide |
| [actionlint.yaml](../../../.github/actionlint.yaml) | 所在模块的结构化配置或参数模板。 | 顶层项：`self-hosted-runner`, `paths` |
| [dependabot.yml](../../../.github/dependabot.yml) | 所在模块的结构化配置或参数模板。 | 顶层项：`version`, `updates` |
| [issue-labeler.yml](../../../.github/issue-labeler.yml) | 所在模块的结构化配置或参数模板。 | 顶层项：`core-features`, `pd-disaggregation`, `kv-cache-pool`, `aclgraph`, `cpu-binding` |
| [labeler.yml](../../../.github/labeler.yml) | 所在模块的结构化配置或参数模板。 | 顶层项：`documentation` |
| [vllm-main-verified.commit](../../../.github/vllm-main-verified.commit) | 记录兼容、发布或测试依赖的版本标签/提交，不是代码实现。 | 依据文件格式及所在目录归类。 |
| [vllm-release-tag.commit](../../../.github/vllm-release-tag.commit) | 记录兼容、发布或测试依赖的版本标签/提交，不是代码实现。 | 依据文件格式及所在目录归类。 |

## `.github/ISSUE_TEMPLATE/`

提交问题、功能建议等使用的结构化模板。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [100-documentation.yml](../../../.github/ISSUE_TEMPLATE/100-documentation.yml) | GitHub Issue 提交流程或表单模板：100-documentation。 | 配置名称：📚 Documentation |
| [110-user-story.yml](../../../.github/ISSUE_TEMPLATE/110-user-story.yml) | GitHub Issue 提交流程或表单模板：110-user-story。 | 配置名称：📚 User Story |
| [200-installation.yml](../../../.github/ISSUE_TEMPLATE/200-installation.yml) | GitHub Issue 提交流程或表单模板：200-installation。 | 配置名称：🛠️ Installation |
| [300-usage.yml](../../../.github/ISSUE_TEMPLATE/300-usage.yml) | GitHub Issue 提交流程或表单模板：300-usage。 | 配置名称：💻 Usage |
| [400-bug-report.yml](../../../.github/ISSUE_TEMPLATE/400-bug-report.yml) | GitHub Issue 提交流程或表单模板：400-bug-report。 | 配置名称：🐛 Bug report |
| [500-feature-request.yml](../../../.github/ISSUE_TEMPLATE/500-feature-request.yml) | GitHub Issue 提交流程或表单模板：500-feature-request。 | 配置名称：🚀 Feature request |
| [600-new-model.yml](../../../.github/ISSUE_TEMPLATE/600-new-model.yml) | GitHub Issue 提交流程或表单模板：600-new-model。 | 配置名称：🤗 Support request for new model supported from huggingface/modelscope/modelers on Ascend |
| [700-performance-discussion.yml](../../../.github/ISSUE_TEMPLATE/700-performance-discussion.yml) | GitHub Issue 提交流程或表单模板：700-performance-discussion。 | 配置名称：⚡ Discussion on the performance of vllm-ascend |
| [750-RFC.yml](../../../.github/ISSUE_TEMPLATE/750-RFC.yml) | GitHub Issue 提交流程或表单模板：750-RFC。 | 配置名称：💬 Request for comments (RFC). |
| [800-others.yml](../../../.github/ISSUE_TEMPLATE/800-others.yml) | GitHub Issue 提交流程或表单模板：800-others。 | 配置名称：🎲 Others |
| [850-cut-branch-checklist.yml](../../../.github/ISSUE_TEMPLATE/850-cut-branch-checklist.yml) | GitHub Issue 提交流程或表单模板：850-cut-branch-checklist。 | 配置名称：Cut Branch Checklist |
| [900-release-checklist.yml](../../../.github/ISSUE_TEMPLATE/900-release-checklist.yml) | GitHub Issue 提交流程或表单模板：900-release-checklist。 | 配置名称：Release Checklist |
| [config.yml](../../../.github/ISSUE_TEMPLATE/config.yml) | GitHub Issue 提交流程或表单模板：config。 | 顶层项：`blank_issues_enabled` |

## `.github/actions/`

可复用的 GitHub Actions 步骤和环境准备。

| 子目录 | 职责 |
| --- | --- |
| [csrc-l1-restore/](../../../.github/actions/csrc-l1-restore) | 存储 子模块；所属范围：可复用的 GitHub Actions 步骤和环境准备。 |
| [csrc-l1-save/](../../../.github/actions/csrc-l1-save) | csrc-l1-save 子模块；所属范围：可复用的 GitHub Actions 步骤和环境准备。 |
| [read-vllm-release-tag/](../../../.github/actions/read-vllm-release-tag) | read-vllm-release-tag 子模块；所属范围：可复用的 GitHub Actions 步骤和环境准备。 |

## `.github/actions/csrc-l1-restore/`

存储 子模块；所属范围：可复用的 GitHub Actions 步骤和环境准备。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [action.yaml](../../../.github/actions/csrc-l1-restore/action.yaml) | 仓库工作流复用的 action 定义或执行辅助。 | 配置名称：Restore csrc incremental cache |

## `.github/actions/csrc-l1-save/`

csrc-l1-save 子模块；所属范围：可复用的 GitHub Actions 步骤和环境准备。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [action.yaml](../../../.github/actions/csrc-l1-save/action.yaml) | 仓库工作流复用的 action 定义或执行辅助。 | 配置名称：Save csrc incremental cache |

## `.github/actions/read-vllm-release-tag/`

read-vllm-release-tag 子模块；所属范围：可复用的 GitHub Actions 步骤和环境准备。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [action.yml](../../../.github/actions/read-vllm-release-tag/action.yml) | 仓库工作流复用的 action 定义或执行辅助。 | 配置名称：Read vLLM release tag |

## `.github/workflows/`

定义测试、构建、发布、文档和其他仓库自动化工作流。

| 子目录 | 职责 |
| --- | --- |
| [configs/](../../../.github/workflows/configs) | 配置 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。 |
| [dockerfiles/](../../../.github/workflows/dockerfiles) | dockerfiles 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。 |
| [matchers/](../../../.github/workflows/matchers) | matchers 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。 |
| [misc/](../../../.github/workflows/misc) | misc 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。 |
| [scripts/](../../../.github/workflows/scripts) | scripts 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [_build_csrc_cache.yaml](../../../.github/workflows/_build_csrc_cache.yaml) | GitHub 自动化工作流：Build csrc Cache；触发条件与 job 在文件中定义。 | 配置名称：Build csrc Cache |
| [_e2e_nightly_multi_node.yaml](../../../.github/workflows/_e2e_nightly_multi_node.yaml) | GitHub 自动化工作流：'e2e nightly test multi_node'；触发条件与 job 在文件中定义。 | 配置名称：'e2e nightly test multi_node' |
| [_e2e_nightly_single_node.yaml](../../../.github/workflows/_e2e_nightly_single_node.yaml) | GitHub 自动化工作流：'e2e nightly test'；触发条件与 job 在文件中定义。 | 配置名称：'e2e nightly test' |
| [_e2e_nightly_single_node_560t.yaml](../../../.github/workflows/_e2e_nightly_single_node_560t.yaml) | GitHub 自动化工作流：'e2e nightly test 560t'；触发条件与 job 在文件中定义。 | 配置名称：'e2e nightly test 560t' |
| [_e2e_nightly_single_node_models.yaml](../../../.github/workflows/_e2e_nightly_single_node_models.yaml) | GitHub 自动化工作流：'e2e nightly models test'；触发条件与 job 在文件中定义。 | 配置名称：'e2e nightly models test' |
| [_ensure_csrc_cache.yaml](../../../.github/workflows/_ensure_csrc_cache.yaml) | GitHub 自动化工作流：Ensure csrc Cache；触发条件与 job 在文件中定义。 | 配置名称：Ensure csrc Cache |
| [_nightly_image_build.yaml](../../../.github/workflows/_nightly_image_build.yaml) | GitHub 自动化工作流：'Nightly image build'；触发条件与 job 在文件中定义。 | 配置名称：'Nightly image build' |
| [_nightly_wait_for_pods_ready.yaml](../../../.github/workflows/_nightly_wait_for_pods_ready.yaml) | GitHub 自动化工作流：'Wait for pods launched signal'；触发条件与 job 在文件中定义。 | 配置名称：'Wait for pods launched signal' |
| [_schedule_image_build.yaml](../../../.github/workflows/_schedule_image_build.yaml) | GitHub 自动化工作流：Image_oncall；触发条件与 job 在文件中定义。 | 配置名称：Image_oncall |
| [_selected_tests.yaml](../../../.github/workflows/_selected_tests.yaml) | GitHub 自动化工作流：Selected Tests；触发条件与 job 在文件中定义。 | 配置名称：Selected Tests |
| [_selected_tests_upstream.yaml](../../../.github/workflows/_selected_tests_upstream.yaml) | GitHub 自动化工作流：Selected Tests Upstream；触发条件与 job 在文件中定义。 | 配置名称：Selected Tests Upstream |
| [bot_issue_manage.yaml](../../../.github/workflows/bot_issue_manage.yaml) | GitHub 自动化工作流："Issue Create/Update Labeler"；触发条件与 job 在文件中定义。 | 配置名称："Issue Create/Update Labeler" |
| [bot_merge_conflict.yaml](../../../.github/workflows/bot_merge_conflict.yaml) | GitHub 自动化工作流：Merge Conflict Labeler；触发条件与 job 在文件中定义。 | 配置名称：Merge Conflict Labeler |
| [bot_pr_create.yaml](../../../.github/workflows/bot_pr_create.yaml) | GitHub 自动化工作流：PR Create；触发条件与 job 在文件中定义。 | 配置名称：PR Create |
| [issue_wait_feedback_command.yml](../../../.github/workflows/issue_wait_feedback_command.yml) | GitHub 自动化工作流：Handle /wait-feedback Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /wait-feedback Command |
| [labeled_download_model_dataset.yaml](../../../.github/workflows/labeled_download_model_dataset.yaml) | GitHub 自动化工作流：'model downloader'；触发条件与 job 在文件中定义。 | 配置名称：'model downloader' |
| [nightly_image_build.yaml](../../../.github/workflows/nightly_image_build.yaml) | GitHub 自动化工作流：Nightly Image Build Schedule；触发条件与 job 在文件中定义。 | 配置名称：Nightly Image Build Schedule |
| [pr_cancel_command.yml](../../../.github/workflows/pr_cancel_command.yml) | GitHub 自动化工作流：Handle /cancel Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /cancel Command |
| [pr_cherry_pick_command.yml](../../../.github/workflows/pr_cherry_pick_command.yml) | GitHub 自动化工作流：Handle /cherry-pick Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /cherry-pick Command |
| [pr_close_job.yaml](../../../.github/workflows/pr_close_job.yaml) | GitHub 自动化工作流：Cancel runs on PR close；触发条件与 job 在文件中定义。 | 配置名称：Cancel runs on PR close |
| [pr_e2e_command.yml](../../../.github/workflows/pr_e2e_command.yml) | GitHub 自动化工作流：Handle /e2e Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /e2e Command |
| [pr_nightly_command.yml](../../../.github/workflows/pr_nightly_command.yml) | GitHub 自动化工作流：Handle /nightly Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /nightly Command |
| [pr_rerun_command.yml](../../../.github/workflows/pr_rerun_command.yml) | GitHub 自动化工作流：Handle /rerun Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /rerun Command |
| [pr_revert_command.yml](../../../.github/workflows/pr_revert_command.yml) | GitHub 自动化工作流：Handle /revert Command；触发条件与 job 在文件中定义。 | 配置名称：Handle /revert Command |
| [pr_supplement_new_tests.yaml](../../../.github/workflows/pr_supplement_new_tests.yaml) | GitHub 自动化工作流：Supplement New Tests；触发条件与 job 在文件中定义。 | 配置名称：Supplement New Tests |
| [pr_test.yaml](../../../.github/workflows/pr_test.yaml) | GitHub 自动化工作流：E2E；触发条件与 job 在文件中定义。 | 配置名称：E2E |
| [push_build_csrc_cache.yaml](../../../.github/workflows/push_build_csrc_cache.yaml) | GitHub 自动化工作流：Cache csrc Build Artifacts；触发条件与 job 在文件中定义。 | 配置名称：Cache csrc Build Artifacts |
| [push_build_precommit_cache.yaml](../../../.github/workflows/push_build_precommit_cache.yaml) | GitHub 自动化工作流：'Cache pre-commit'；触发条件与 job 在文件中定义。 | 配置名称：'Cache pre-commit' |
| [schedule_daily_test.yaml](../../../.github/workflows/schedule_daily_test.yaml) | GitHub 自动化工作流：Schedule Daily Tests；触发条件与 job 在文件中定义。 | 配置名称：Schedule Daily Tests |
| [schedule_doc_getting_started_test.yaml](../../../.github/workflows/schedule_doc_getting_started_test.yaml) | GitHub 自动化工作流：Doc Test；触发条件与 job 在文件中定义。 | 配置名称：Doc Test |
| [schedule_doc_linkcheck.yaml](../../../.github/workflows/schedule_doc_linkcheck.yaml) | GitHub 自动化工作流：Docs link check；触发条件与 job 在文件中定义。 | 配置名称：Docs link check |
| [schedule_doc_translate.yaml](../../../.github/workflows/schedule_doc_translate.yaml) | GitHub 自动化工作流：Auto Doc Translate；触发条件与 job 在文件中定义。 | 配置名称：Auto Doc Translate |
| [schedule_e2e_upstream_test.yaml](../../../.github/workflows/schedule_e2e_upstream_test.yaml) | GitHub 自动化工作流：E2E-upstream；触发条件与 job 在文件中定义。 | 配置名称：E2E-upstream |
| [schedule_image_build_and_push.yaml](../../../.github/workflows/schedule_image_build_and_push.yaml) | GitHub 自动化工作流：Image Build and Push；触发条件与 job 在文件中定义。 | 配置名称：Image Build and Push |
| [schedule_lint_image_build.yaml](../../../.github/workflows/schedule_lint_image_build.yaml) | GitHub 自动化工作流：'Image build lint'；触发条件与 job 在文件中定义。 | 配置名称：'Image build lint' |
| [schedule_main2main.yaml](../../../.github/workflows/schedule_main2main.yaml) | GitHub 自动化工作流：Main2Main；触发条件与 job 在文件中定义。 | 配置名称：Main2Main |
| [schedule_nightly_test_310p.yaml](../../../.github/workflows/schedule_nightly_test_310p.yaml) | GitHub 自动化工作流：Nightly-310P；触发条件与 job 在文件中定义。 | 配置名称：Nightly-310P |
| [schedule_nightly_test_a2.yaml](../../../.github/workflows/schedule_nightly_test_a2.yaml) | GitHub 自动化工作流：Nightly-A2；触发条件与 job 在文件中定义。 | 配置名称：Nightly-A2 |
| [schedule_nightly_test_a3.yaml](../../../.github/workflows/schedule_nightly_test_a3.yaml) | GitHub 自动化工作流：Nightly-A3；触发条件与 job 在文件中定义。 | 配置名称：Nightly-A3 |
| [schedule_nightly_test_a3_560t.yaml](../../../.github/workflows/schedule_nightly_test_a3_560t.yaml) | GitHub 自动化工作流：Nightly-A3-560T；触发条件与 job 在文件中定义。 | 配置名称：Nightly-A3-560T |
| [schedule_nightly_test_a5.yaml](../../../.github/workflows/schedule_nightly_test_a5.yaml) | GitHub 自动化工作流：Nightly-A5；触发条件与 job 在文件中定义。 | 配置名称：Nightly-A5 |
| [schedule_release_code_and_wheel.yml](../../../.github/workflows/schedule_release_code_and_wheel.yml) | GitHub 自动化工作流：Release Code and Wheel；触发条件与 job 在文件中定义。 | 配置名称：Release Code and Wheel |
| [schedule_stale_manage.yaml](../../../.github/workflows/schedule_stale_manage.yaml) | GitHub 自动化工作流："Close stale resolved/wait-feedback issues"；触发条件与 job 在文件中定义。 | 配置名称："Close stale resolved/wait-feedback issues" |
| [schedule_test_coverage.yaml](../../../.github/workflows/schedule_test_coverage.yaml) | GitHub 自动化工作流：Schedule Test Coverage；触发条件与 job 在文件中定义。 | 配置名称：Schedule Test Coverage |
| [schedule_weekly_test_310p.yaml](../../../.github/workflows/schedule_weekly_test_310p.yaml) | GitHub 自动化工作流：Weekly-310P；触发条件与 job 在文件中定义。 | 配置名称：Weekly-310P |
| [schedule_weekly_test_a2.yaml](../../../.github/workflows/schedule_weekly_test_a2.yaml) | GitHub 自动化工作流：Weekly-A2；触发条件与 job 在文件中定义。 | 配置名称：Weekly-A2 |
| [schedule_weekly_test_a3.yaml](../../../.github/workflows/schedule_weekly_test_a3.yaml) | GitHub 自动化工作流：Weekly-A3；触发条件与 job 在文件中定义。 | 配置名称：Weekly-A3 |
| [schedule_weekly_test_a3_560t.yaml](../../../.github/workflows/schedule_weekly_test_a3_560t.yaml) | GitHub 自动化工作流：Weekly-A3-560T；触发条件与 job 在文件中定义。 | 配置名称：Weekly-A3-560T |
| [slash_command_dispatch.yml](../../../.github/workflows/slash_command_dispatch.yml) | GitHub 自动化工作流：Slash Command Dispatch；触发条件与 job 在文件中定义。 | 配置名称：Slash Command Dispatch |

## `.github/workflows/configs/`

配置 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [daily_config.yaml](../../../.github/workflows/configs/daily_config.yaml) | GitHub 自动化工作流：daily_config；触发条件与 job 在文件中定义。 | 顶层项：`a3-560t` |
| [nightly_config.yaml](../../../.github/workflows/configs/nightly_config.yaml) | GitHub 自动化工作流：nightly_config；触发条件与 job 在文件中定义。 | 顶层项：`a2`, `a3`, `a3-560t`, `310p`, `a5` |
| [weekly_config.yaml](../../../.github/workflows/configs/weekly_config.yaml) | GitHub 自动化工作流：weekly_config；触发条件与 job 在文件中定义。 | 顶层项：`a2`, `a3`, `310p`, `a3-560t` |

## `.github/workflows/dockerfiles/`

dockerfiles 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Dockerfile.buildwheel.310p](../../../.github/workflows/dockerfiles/Dockerfile.buildwheel.310p) | 构建 buildwheel.310p 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.buildwheel.a2](../../../.github/workflows/dockerfiles/Dockerfile.buildwheel.a2) | 构建 buildwheel.a2 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.buildwheel.a3](../../../.github/workflows/dockerfiles/Dockerfile.buildwheel.a3) | 构建 buildwheel.a3 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.buildwheel.a5](../../../.github/workflows/dockerfiles/Dockerfile.buildwheel.a5) | 构建 buildwheel.a5 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.lint](../../../.github/workflows/dockerfiles/Dockerfile.lint) | 构建 lint 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.nightly.310p](../../../.github/workflows/dockerfiles/Dockerfile.nightly.310p) | 构建 nightly.310p 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.nightly.a2](../../../.github/workflows/dockerfiles/Dockerfile.nightly.a2) | 构建 nightly.a2 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.nightly.a3](../../../.github/workflows/dockerfiles/Dockerfile.nightly.a3) | 构建 nightly.a3 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |
| [Dockerfile.nightly.a5](../../../.github/workflows/dockerfiles/Dockerfile.nightly.a5) | 构建 nightly.a5 目标的镜像环境；依赖版本与安装步骤以文件内容为准。 | 依据文件格式及所在目录归类。 |

## `.github/workflows/matchers/`

matchers 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [actionlint.json](../../../.github/workflows/matchers/actionlint.json) | GitHub 自动化工作流：actionlint；触发条件与 job 在文件中定义。 | 顶层项：`problemMatcher` |
| [markdownlint.json](../../../.github/workflows/matchers/markdownlint.json) | GitHub 自动化工作流：markdownlint；触发条件与 job 在文件中定义。 | 顶层项：`problemMatcher` |
| [mypy.json](../../../.github/workflows/matchers/mypy.json) | GitHub 自动化工作流：mypy；触发条件与 job 在文件中定义。 | 顶层项：`problemMatcher` |

## `.github/workflows/misc/`

misc 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [model_dataset_list.json](../../../.github/workflows/misc/model_dataset_list.json) | GitHub 自动化工作流：model_dataset_list；触发条件与 job 在文件中定义。 | 顶层项：`models`, `datasets` |

## `.github/workflows/scripts/`

scripts 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。

| 子目录 | 职责 |
| --- | --- |
| [wheel/](../../../.github/workflows/scripts/wheel) | wheel 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [assemble_coverage.py](../../../.github/workflows/scripts/assemble_coverage.py) | GitHub 自动化工作流：assemble_coverage；触发条件与 job 在文件中定义。 | 注释：Check and backfill per-test coverage data for full coverage runs.；定义：`coverage_key`, `expected_coverage_keys`, `missing_coverage_keys` |
| [check_md_links.py](../../../.github/workflows/scripts/check_md_links.py) | GitHub 自动化工作流：check_md_links；触发条件与 job 在文件中定义。 | 注释：Check HTTP links in Markdown files.；定义：`extract_urls`, `check_url`, `collect_files` |
| [collect_merged_new_tests.py](../../../.github/workflows/scripts/collect_merged_new_tests.py) | GitHub 自动化工作流：collect_merged_new_tests；触发条件与 job 在文件中定义。 | 注释：Collect new or moved tests from a merged PR and write them as extra-yaml paths.；定义：`collect_nodeids`, `collect_from_changes`, `resolve_pr_diff_refs` |
| [conftest.py](../../../.github/workflows/scripts/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 注释：Pytest configuration for ``scripts/`` unit tests.；定义：`_load_runner_mapping` |
| [coverage.py](../../../.github/workflows/scripts/coverage.py) | GitHub 自动化工作流：coverage；触发条件与 job 在文件中定义。 | 定义：`pytest_node_file_path` |
| [csrc_cache_targets.json](../../../.github/workflows/scripts/csrc_cache_targets.json) | GitHub 自动化工作流：csrc_cache_targets；触发条件与 job 在文件中定义。 | 顶层项：`a2-arm64-ubuntu`, `a3-arm64-ubuntu`, `310p-arm64-ubuntu`, `a2-arm64-openeuler`, `a3-arm64-openeuler` |
| [detect_po_changes.py](../../../.github/workflows/scripts/detect_po_changes.py) | GitHub 自动化工作流：detect_po_changes；触发条件与 job 在文件中定义。 | 注释：Detect new or changed English paragraphs and update .po skeleton files.；定义：`process_file`, `main` |
| [estimated_times.yaml](../../../.github/workflows/scripts/estimated_times.yaml) | GitHub 自动化工作流：estimated_times；触发条件与 job 在文件中定义。 | 顶层项：`estimated_times` |
| [gitleaks.sh](../../../.github/workflows/scripts/gitleaks.sh) | GitHub 自动化工作流：gitleaks；触发条件与 job 在文件中定义。 | Shell 脚本；按工程阶段和文件名归类。 |
| [install_daily_deps.sh](../../../.github/workflows/scripts/install_daily_deps.sh) | GitHub 自动化工作流：install_daily_deps；触发条件与 job 在文件中定义。 | 命令类别：`python3` |
| [main2main_tests.json](../../../.github/workflows/scripts/main2main_tests.json) | GitHub 自动化工作流：main2main_tests；触发条件与 job 在文件中定义。 | 顶层项：`cases` |
| [merge_extra_recommended_tests.py](../../../.github/workflows/scripts/merge_extra_recommended_tests.py) | GitHub 自动化工作流：merge_extra_recommended_tests；触发条件与 job 在文件中定义。 | 注释：Append extra pytest targets from a YAML file onto a recommended test list.；定义：`load_extra_tests`, `load_existing_tests`, `merge_extra_tests` |
| [po_translate.py](../../../.github/workflows/scripts/po_translate.py) | GitHub 自动化工作流：po_translate；触发条件与 job 在文件中定义。 | 定义：`POTranslator`, `validate_coverage`, `async_main` |
| [prepare_csrc_l1_restore.py](../../../.github/workflows/scripts/prepare_csrc_l1_restore.py) | GitHub 自动化工作流：prepare_csrc_l1_restore；触发条件与 job 在文件中定义。 | 定义：`prepare`, `main` |
| [resolve_csrc_cache_targets.py](../../../.github/workflows/scripts/resolve_csrc_cache_targets.py) | GitHub 自动化工作流：resolve_csrc_cache_targets；触发条件与 job 在文件中定义。 | 定义：`load_targets`, `resolve_targets`, `write_outputs` |
| [resolve_daily_tests.py](../../../.github/workflows/scripts/resolve_daily_tests.py) | GitHub 自动化工作流：resolve_daily_tests；触发条件与 job 在文件中定义。 | 注释：Resolve daily nightly test cases for daytime regression runs.；定义：`main` |
| [resolve_nightly_tests.py](../../../.github/workflows/scripts/resolve_nightly_tests.py) | GitHub 自动化工作流：resolve_nightly_tests；触发条件与 job 在文件中定义。 | 定义：`parse_nightly_matrix`, `cmd_dispatch`, `cmd_matrix` |
| [run_selected_tests.sh](../../../.github/workflows/scripts/run_selected_tests.sh) | GitHub 自动化工作流：run_selected_tests；触发条件与 job 在文件中定义。 | 函数：`setup_coverage`, `setup_vllm_cache_root`, `terminate_process_group` |
| [run_suite.py](../../../.github/workflows/scripts/run_suite.py) | GitHub 自动化工作流：run_suite；触发条件与 job 在文件中定义。 | 定义：`_Color`, `TestFile`, `run_tests`, `load_suites`, `partition` |
| [runner_label.json](../../../.github/workflows/scripts/runner_label.json) | GitHub 自动化工作流：runner_label；触发条件与 job 在文件中定义。 | 顶层项：`linux-aarch64-310p-1`, `linux-aarch64-310p-2`, `linux-aarch64-310p-4`, `linux-aarch64-a3-800t-16`, `linux-aarch64-a3-2` |
| [select_tests.py](../../../.github/workflows/scripts/select_tests.py) | GitHub 自动化工作流：select_tests；触发条件与 job 在文件中定义。 | 注释：Determine which tests to run.；定义：`NpuType`, `RunnerInfo`, `main` |
| [test_config.yaml](../../../.github/workflows/scripts/test_config.yaml) | GitHub 自动化工作流：test_config；触发条件与 job 在文件中定义。 | 顶层项：`curated_tests`, `skip_tests`, `accuracy_tests`, `runner_mapping`, `partition` |
| [test_selector.py](../../../.github/workflows/scripts/test_selector.py) | GitHub 自动化工作流：test_selector；触发条件与 job 在文件中定义。 | 注释：Test Selector - Precision test selector based on coverage data (line, function, file granularity)；定义：`CoverageSelector`, `CodeChangeDetector`, `is_selectable_test`, `main` |
| [update_estimated_times.py](../../../.github/workflows/scripts/update_estimated_times.py) | GitHub 自动化工作流：update_estimated_times；触发条件与 job 在文件中定义。 | 注释：Update estimated_times.yaml from CI timing data.；定义：`collect_timings`, `compute_median`, `update_config` |
| [upstream_config.yaml](../../../.github/workflows/scripts/upstream_config.yaml) | GitHub 自动化工作流：upstream_config；触发条件与 job 在文件中定义。 | 顶层项：`e2e-upstream_pr`, `e2e-upstream_singlecard`, `e2e-upstream_online_1`, `e2e-upstream_a2_2`, `e2e-upstream_a2_4` |

## `.github/workflows/scripts/wheel/`

wheel 子模块；所属范围：定义测试、构建、发布、文档和其他仓库自动化工作流。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [auto_exclude.py](../../../.github/workflows/scripts/wheel/auto_exclude.py) | GitHub 自动化工作流：auto_exclude；触发条件与 job 在文件中定义。 | 定义：`repair_wheels`, `main` |
| [config.json](../../../.github/workflows/scripts/wheel/config.json) | GitHub 自动化工作流：config；触发条件与 job 在文件中定义。 | 顶层项：`variables`, `jobs` |
| [make_variant.py](../../../.github/workflows/scripts/wheel/make_variant.py) | GitHub 自动化工作流：make_variant；触发条件与 job 在文件中定义。 | 定义：`Job`, `load_jobs`, `load_variant_label_aliases`, `apply_variant_label_aliases` |
| [pyproject.toml](../../../.github/workflows/scripts/wheel/pyproject.toml) | GitHub 自动化工作流：pyproject；触发条件与 job 在文件中定义。 | 配置条目与规则；按实际读取者生效。 |
