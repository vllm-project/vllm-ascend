# tests/e2e/doctests：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **5 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/doctests/`

端到端 doctests 场景、运行脚本和配置。

| 子目录 | 职责 |
| --- | --- |
| [scripts/](../../../tests/e2e/doctests/scripts) | 端到端 scripts 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [001-quickstart-test.sh](../../../tests/e2e/doctests/001-quickstart-test.sh) | 001-quickstart-test 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 命令类别：`python3` |
| [002-installation-test.sh](../../../tests/e2e/doctests/002-installation-test.sh) | 002-installation-test 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 命令类别：`git`, `python3`, `pip` |

## `tests/e2e/doctests/scripts/`

端到端 scripts 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [common.sh](../../../tests/e2e/doctests/scripts/common.sh) | 公共逻辑 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | 函数：`log_info`, `die` |
| [doctest_helper.py](../../../tests/e2e/doctests/scripts/doctest_helper.py) | doctest helper 的测试辅助、测试入口或共享数据结构。 | 注释：Extract registered documentation blocks or preview the fixed CI plan.；定义：`DoctestError`, `parse_manual_selection`, `extract_doctest_block`, `parse_mkdocs_extra` |
| [run_doctests.sh](../../../tests/e2e/doctests/scripts/run_doctests.sh) | run doctests 测试启动与环境组织脚本；可能调用真实服务或多设备任务。 | Shell 脚本；按工程阶段和文件名归类。 |
