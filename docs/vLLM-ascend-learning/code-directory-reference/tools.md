# tools：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **54 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tools/`

开发检查、调试、请求发送、文档生成与性能工具。

| 子目录 | 职责 |
| --- | --- |
| [ai_qos/](../../../tools/ai_qos) | ai_qos 子模块；所属范围：开发检查、调试、请求发送、文档生成与性能工具。 |
| [bisect/](../../../tools/bisect) | 构建、运行和裁决历史版本的自动二分定位工具。 |
| [docs_codegen/](../../../tools/docs_codegen) | 扫描、转换并生成文档的工具模块。 |
| [trace/](../../../tools/trace) | 解析运行日志和 trace 结果的工具。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [actionlint.sh](../../../tools/actionlint.sh) | actionlint 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | Shell 脚本；按工程阶段和文件名归类。 |
| [ai_qos.py](../../../tools/ai_qos.py) | ai qos 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`AiqosConfig`, `run_unset`, `main` |
| [aisbench.py](../../../tools/aisbench.py) | aisbench 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`AisbenchRunner`, `_MetricsServer`, `run_aisbench_cases`, `get_TTFT`, `get_lock` |
| [check_boolean_context_manager.py](../../../tools/check_boolean_context_manager.py) | 管理器、检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Lint: detect `with a() and b():` (boolean op in with-statement context).；定义：`check_file`, `main` |
| [check_forbidden_imports.py](../../../tools/check_forbidden_imports.py) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`ForbiddenImport`, `check_file`, `main` |
| [check_logger.sh](../../../tools/check_logger.sh) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | Shell 脚本；按工程阶段和文件名归类。 |
| [check_long_functions.py](../../../tools/check_long_functions.py) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`check_file`, `main` |
| [check_python_src_init.py](../../../tools/check_python_src_init.py) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`check_init_file_in_package`, `find_missing_init_dirs`, `main` |
| [check_repo.sh](../../../tools/check_repo.sh) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | Shell 脚本；按工程阶段和文件名归类。 |
| [check_symbolic_meta.py](../../../tools/check_symbolic_meta.py) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`has_exemption`, `check_file`, `main` |
| [collect_user_first_contribution.sh](../../../tools/collect_user_first_contribution.sh) | collect user first contribution 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 函数：`get_last_commit_hash`, `get_current_contributor_count`, `extract_login_from_noreply_email` |
| [format_contributors.py](../../../tools/format_contributors.py) | 格式化 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`parse_lines`, `main` |
| [generate_zh_docs.py](../../../tools/generate_zh_docs.py) | generate zh docs 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Generate Chinese (zh) markdown files from existing .po translation files.；定义：`parse_po_file`, `get_relative_path`, `apply_translations` |
| [mooncake_installer.sh](../../../tools/mooncake_installer.sh) | Mooncake 传输/存储 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 函数：`print_section`, `print_success`, `print_error` |
| [mypy.sh](../../../tools/mypy.sh) | mypy 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 函数：`run_mypy` |
| [png-lint.sh](../../../tools/png-lint.sh) | png-lint 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | Shell 脚本；按工程阶段和文件名归类。 |
| [rtd_build.sh](../../../tools/rtd_build.sh) | rtd build 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 命令类别：`vllm`, `python` |
| [send_mm_request.py](../../../tools/send_mm_request.py) | 请求 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`send_image_request` |
| [send_request.py](../../../tools/send_request.py) | 请求 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`resolve_prompt`, `validate_response`, `send_v1_completions` |
| [set_release_flag.py](../../../tools/set_release_flag.py) | set release flag 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Set is_release flag in mkdocs config for tagged releases. |
| [shellcheck.sh](../../../tools/shellcheck.sh) | 检查 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | Shell 脚本；按工程阶段和文件名归类。 |
| [spec_decode_metrics.py](../../../tools/spec_decode_metrics.py) | 投机解码、指标 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`fetch_metrics`, `analysis_metrics`, `capture_baseline` |
| [test_memfabric_pd_read.py](../../../tools/test_memfabric_pd_read.py) | test memfabric pd read 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Minimal two-node MemFabric pull-read diagnostic.；定义：`main` |
| [upload_to_openlibing.py](../../../tools/upload_to_openlibing.py) | 加载 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`upload_data_to_openlibing`, `main` |
| [vllm_bench.py](../../../tools/vllm_bench.py) | vllm bench 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`VllmbenchRunner`, `run_vllm_bench_case` |

## `tools/ai_qos/`

ai_qos 子模块；所属范围：开发检查、调试、请求发送、文档生成与性能工具。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [CMakeLists.txt](../../../tools/ai_qos/CMakeLists.txt) | 定义本目录源码、依赖、目标和安装/构建步骤。 | 构建项：`ai_qos` |
| [ai_qos.cpp](../../../tools/ai_qos/ai_qos.cpp) | ai qos 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 类型：`qos_gbl_config`, `qos_master_config`, `qos_mata_config` |

## `tools/bisect/`

构建、运行和裁决历史版本的自动二分定位工具。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [README.md](../../../tools/bisect/README.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：README。 | 标题：Nightly/Weekly Auto-Bisect |
| [USAGE_zh.md](../../../tools/bisect/USAGE_zh.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：USAGE_zh。 | 标题：Nightly Auto-Bisect 工具使用指导 |
| [__init__.py](../../../tools/bisect/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 注释：Auto-Bisect tooling for nightly E2E test failures. |
| [auto_bisect.py](../../../tools/bisect/auto_bisect.py) | auto bisect 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Auto-Bisect orchestrator: find the first bad commit for a nightly failure.；定义：`Bisector`, `main` |
| [build_manager.py](../../../tools/bisect/build_manager.py) | 管理器 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Decide whether a checkout needs recompilation and perform the deploy.；定义：`BuildDecision`, `BuildManager` |
| [config.py](../../../tools/bisect/config.py) | 配置 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Constants and data models shared across the bisect tool.；定义：`Candidate`, `TrialResult` |
| [coordinator.py](../../../tools/bisect/coordinator.py) | coordinator 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Filesystem barrier used to keep multi-node bisect rounds in lockstep.；定义：`Coordinator` |
| [git_ops.py](../../../tools/bisect/git_ops.py) | git ops 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Git helpers: resolve commits, build the candidate list, checkout, diff.；定义：`GitError`, `ensure_full_history`, `resolve_commit`, `is_ancestor` |
| [good_table.py](../../../tools/bisect/good_table.py) | good table 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Read a scheduled-test status table to find a case's last-known-good commit.；定义：`GoodEntry`, `GoodTable`, `valid_soc`, `valid_scene` |
| [good_table.sample.csv](../../../tools/bisect/good_table.sample.csv) | 采样 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 依据文件格式及所在目录归类。 |
| [report.py](../../../tools/bisect/report.py) | report 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Console markers and the final bisect report.；定义：`print_verdict`, `print_progress`, `print_endpoint_check` |
| [runner.py](../../../tools/bisect/runner.py) | 模型执行 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Launch abstraction: deploy a commit and run the failing case.；定义：`BisectFatalError`, `BaseRunner`, `kill_stray_servers`, `build_runner` |
| [state.py](../../../tools/bisect/state.py) | 状态 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Persist bisect progress so a preempted/timed-out run can resume.；定义：`BisectState` |
| [verdict.py](../../../tools/bisect/verdict.py) | verdict 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Normalise a trial run into PASS / FAIL / SKIP.；定义：`RunOutcome`, `evaluate` |
| [version_compat.py](../../../tools/bisect/version_compat.py) | version compat 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Version discovery and dependency adaptation for bisect trials.；定义：`PackageVersions`, `VersionPolicy`, `versions_equal`, `expected_versions`, `installed_package_version` |
| [worker_agent.py](../../../tools/bisect/worker_agent.py) | 设备执行 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 注释：Non-master node agent for multi-node bisect.；定义：`run_worker` |

## `tools/docs_codegen/`

扫描、转换并生成文档的工具模块。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tools/docs_codegen/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [cli.py](../../../tools/docs_codegen/cli.py) | cli 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`build_arg_parser`, `main` |
| [converters.py](../../../tools/docs_codegen/converters.py) | 卷积 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`GeneratedScript`, `BaseConverter`, `build_default_converters`, `get_converter` |
| [errors.py](../../../tools/docs_codegen/errors.py) | errors 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`DocsCodegenError`, `make_docs_codegen_error` |
| [generator.py](../../../tools/docs_codegen/generator.py) | generator 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`GeneratorService`, `create_default_generator_service` |
| [scanner.py](../../../tools/docs_codegen/scanner.py) | scanner 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`ModelCodeBlock`, `BlockScanner` |
| [sphinx_extension.py](../../../tools/docs_codegen/sphinx_extension.py) | sphinx extension 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`ModelCodeDirective`, `build_block_from_options`, `render_generated_script`, `on_builder_inited` |
| [utils.py](../../../tools/docs_codegen/utils.py) | 辅助函数 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`trim_blank_edges`, `require_mapping`, `require_mapping_list` |
| [yaml_loader.py](../../../tools/docs_codegen/yaml_loader.py) | 加载器 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`LoadedYaml`, `YamlLoader` |

## `tools/trace/`

解析运行日志和 trace 结果的工具。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [parse_logs.py](../../../tools/trace/parse_logs.py) | parse logs 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 定义：`parse_trace_logs` |
| [process_trace_model.py](../../../tools/trace/process_trace_model.py) | 模型、时间线跟踪 开发/诊断工具；右栏提供入口，运行前应阅读参数及外部操作。 | 导入：`os`, `pandas` |
