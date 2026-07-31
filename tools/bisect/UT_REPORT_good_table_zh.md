# UT 报告：good table 频率拆分

## 1. 范围

覆盖 good table 按 nightly / weekly 拆分后读侧、写侧与 CLI 透传三部分：

- `tests/ut/tools/bisect/test_good_table.py`（读侧）
- `tests/ut/tools/bisect/test_update_good_table.py`（写侧）
- `tests/ut/tools/bisect/test_auto_bisect.py`（`--soc` 端到端，本次新增 3 个）

## 2. 用例清单

### 2.1 `test_good_table.py`（14 个）

| 用例 | 覆盖点 |
| --- | --- |
| `test_lookup_last_good_by_name_uses_latest_success` | 按 name 匹配，最新成功胜出 |
| `test_lookup_last_good_by_yaml_basename` | 按 yaml/path 文件名匹配 |
| `test_lookup_uses_all_supplied_case_dimensions` | name + config_yaml + soc + scene 全维度匹配 |
| `test_lookup_legacy_row_without_dimensions_remains_compatible` | 旧七列行兼容 |
| `test_lookup_filters_by_soc` | soc 过滤（a2/a3 各取各的行，未知 soc 返回 None） |
| `test_lookup_filters_by_scene` | scene 过滤（single/multi 互不串扰） |
| `test_lookup_without_dimensions_matches_new_schema_rows` | 不传维度时新表行仍可匹配（向后兼容） |
| `test_lookup_requires_all_supplied_dimensions_to_match` | 维度都提供时必须全部匹配 |
| `test_lookup_skips_success_rows_without_commit` | 成功行缺 commit 时回退更早成功行 |
| `test_lookup_accepts_status_spelling_variants` | status 大小写/拼写变体（PASS/ok） |
| `test_lookup_tolerates_mixed_time_formats` | 混合时区/无法解析时间不崩溃、按 UTC 归一化排序 |
| `test_lookup_missing_table_returns_none` | 表文件不存在返回 None |
| `test_lookup_last_good_returns_none_for_missing_or_failed_case` | 缺失/失败用例返回 None |
| `test_norm_detects_surplus_csv_columns` | 超额列 CSV 容错 |

### 2.2 `test_update_good_table.py`（10 个）

| 用例 | 覆盖点 |
| --- | --- |
| `test_update_replaces_only_same_composite_key` | 仅替换同 `soc+scene+path` 复合键 |
| `test_update_migrates_matching_legacy_row` | 旧七列行迁移为新格式 |
| `test_update_creates_table_with_header_and_reports_new` | 首次更新建表 + 表头 + `is_new=True` |
| `test_update_reports_false_when_table_exists` | 已存在时 `is_new=False` |
| `test_update_keeps_rows_for_different_scene` | 同 soc 不同 scene 的行各自保留 |
| `test_update_preserves_unrelated_rows` | 无关行不被覆盖 |
| `test_update_normalises_path_separators` | 反斜杠/尾斜杠路径归一化 |
| `test_load_rows_returns_empty_for_missing_file` | 文件缺失读取为空 |
| `test_resolve_test_path_joins_bare_filename_with_config_base` | 裸文件名拼接 config base |
| `test_resolve_test_path_keeps_explicit_directory_unchanged` | 完整路径原样保留 |

### 2.3 `test_auto_bisect.py` 本次新增（3 个）

| 用例 | 覆盖点 |
| --- | --- |
| `test_resolve_good_selects_row_matching_soc_and_records_paired_vllm` | `--soc` 选中正确行并记录配对 vLLM commit |
| `test_resolve_good_explicit_commit_skips_table_lookup` | 显式 `--good-commit` 跳过查表 |
| `test_resolve_good_raises_when_no_matching_success_row` | 无匹配成功行时报错 |

## 3. 结果

| 分支/环境 | 结果 |
| --- | --- |
| 本地工作区（main + 改动） | **71 passed**（基线 52 + 新增 19） |
| PR 分支 `codex/split-good-table-frequency` | **75 passed**（PR 基线 56 + 新增 19） |
| UT 分支 `codex/split-good-table-frequency_ut` | **89 passed**（UT 基线 70 + 新增 19） |

全部通过，无失败、无跳过。新增 19 个用例：读侧 8 + 写侧 8 + CLI 端到端 3。

## 4. 运行方式

```powershell
python -m pytest -q `
  --confcutdir=tests/ut/tools/bisect `
  --basetemp <临时目录> `
  -p no:cacheprovider `
  tests/ut/tools/bisect
```

## 5. 环境

- Windows 11 / PowerShell；Python 3.12.13；pytest 9.1.1。
- 依赖：`pytest`、`regex`、`pyyaml`（`tools/bisect/git_ops.py` 与
  multi-node yaml 解析需要）。
- 静态检查：`git diff --check` 通过；改动文件 `compile()` 语法检查通过。
