# tests/ut/tools：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **13 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/tools/`

验证 tools 对应组件的单元/回归行为；子目录按生产模块组织。

| 子目录 | 职责 |
| --- | --- |
| [bisect/](../../../tests/ut/tools/bisect) | 验证 tools/bisect 对应组件的单元/回归行为；子目录按生产模块组织。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/tools/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [test_aisbench.py](../../../tests/ut/tools/test_aisbench.py) | 验证 aisbench 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_request_config_reasoning_effort` |

## `tests/ut/tools/bisect/`

验证 tools/bisect 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/ut/tools/bisect/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/ut/tools/bisect/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 导入：`importlib.util`, `sys` |
| [test_auto_bisect.py](../../../tests/ut/tools/bisect/test_auto_bisect.py) | 验证 auto bisect 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_pick_mid_prefers_midpoint_then_nearest_unskipped_index`, `test_parse_args_maps_no_assume_built_head_flag`, `test_parse_args_requires_soc`, `test_parse_args_rejects_empty_or_placeholder_soc` |
| [test_build_manager.py](../../../tests/ut/tools/bisect/test_build_manager.py) | 验证 管理器 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_decide_rebuilds_without_established_baseline`, `test_decide_skips_when_target_is_already_built`, `test_decide_per_commit_file_categories`, `test_decide_since_build_uses_delta_from_last_built_commit` |
| [test_coordinator.py](../../../tests/ut/tools/bisect/test_coordinator.py) | 验证 coordinator 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_publish_command_and_wait_command_round_trip`, `test_wait_command_ignores_stale_done_when_since_ts_is_newer`, `test_wait_command_returns_none_for_release_file`, `test_wait_all_ready_rejects_split_commit` |
| [test_git_ops.py](../../../tests/ut/tools/bisect/test_git_ops.py) | 验证 git ops 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_candidate_list_parses_first_parent_log`, `test_candidate_list_rejects_shallow_repo`, `test_candidate_list_rejects_invalid_range`, `test_matches_any_returns_files_matching_configured_globs` |
| [test_report.py](../../../tests/ut/tools/bisect/test_report.py) | 验证 report 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_write_report_json_serializes_candidates_and_trials` |
| [test_runner.py](../../../tests/ut/tools/bisect/test_runner.py) | 验证 模型执行 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_safe_name_replaces_path_and_space_separators`, `test_base_env_includes_case_and_config_base`, `test_bisect_selects_current_common_test_entries`, `test_multi_node_runner_selects_external_dp_test_path` |
| [test_state.py](../../../tests/ut/tools/bisect/test_state.py) | 验证 状态 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_bisect_state_save_and_load_round_trips`, `test_bisect_state_load_ignores_stale_good_bad_range`, `test_bisect_state_load_missing_file_returns_none` |
| [test_verdict.py](../../../tests/ut/tools/bisect/test_verdict.py) | 验证 verdict 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_evaluate_zero_exit_checks_benchmark_json`, `test_evaluate_zero_exit_without_benchmark_failures_is_pass`, `test_evaluate_infra_exit_codes_are_skip`, `test_evaluate_nonzero_runtime_failure_is_fail` |
| [test_version_compat.py](../../../tests/ut/tools/bisect/test_version_compat.py) | 验证 version compat 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_expected_versions_read_nightly_source_files`, `test_expected_versions_falls_back_to_pyproject`, `test_expected_versions_at_commit_reads_all_files_without_checkout`, `test_version_policy_only_checks_switchable_endpoint_changes` |
