# tests/e2e：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **7 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/`

模型、部署和多设备场景的端到端验证。

| 子目录 | 职责 |
| --- | --- |
| [cases/](tests--e2e--cases.md) | 按模型或功能组织的测试场景。 |
| [common/](tests--e2e--common.md) | 端到端场景共用的部署与断言辅助。 |
| [doctests/](tests--e2e--doctests.md) | 端到端 doctests 场景、运行脚本和配置。 |
| [models/](tests--e2e--models.md) | 模型级测试配置及辅助定义。 |
| [nightly/](tests--e2e--nightly.md) | 夜间模型、精度、稳定性与性能场景。 |
| [prompts/](tests--e2e--prompts.md) | 端到端 prompts 场景、运行脚本和配置。 |
| [pull_request/](tests--e2e--pull_request.md) | 用于 PR 场景的分设备数量测试和运行脚本。 |
| [vllm_interface/](tests--e2e--vllm_interface.md) | 端到端 vllm_interface 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [__init__.py](../../../tests/e2e/__init__.py) | 标记 Python 包边界；本文件没有顶层类或函数实现。 | 无顶层类/函数；包标记、常量或数据定义。 |
| [conftest.py](../../../tests/e2e/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`ModelName`, `MooncakeLauncher`, `pytest_addoption`, `wait_until_npu_memory_free`, `cleanup_dist_env_and_memory` |
| [coverage.html](../../../tests/e2e/coverage.html) | coverage.html 附属文件；当前按文件格式及目录定位用途。 | 依据文件格式及所在目录归类。 |
| [coverage_taxonomy.py](../../../tests/e2e/coverage_taxonomy.py) | coverage taxonomy 的测试辅助、测试入口或共享数据结构。 | 注释：Single source of truth for the ``@pytest.mark.e2e_coverage`` taxonomy.；定义：`split_multi`, `is_invalid_combo`, `validate_coverage_marker` |
| [generate_coverage_html.py](../../../tests/e2e/generate_coverage_html.py) | generate coverage html 的测试辅助、测试入口或共享数据结构。 | 注释：Generate a self-contained interactive HTML coverage report by scanning；定义：`TestRecord`, `Summary`, `main` |
| [model_utils.py](../../../tests/e2e/model_utils.py) | 模型、辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`check_outputs_equal` |
| [utils.py](../../../tests/e2e/utils.py) | 辅助函数 的测试辅助、测试入口或共享数据结构。 | 定义：`fork_new_process_for_each_test`, `matryoshka_fy`, `check_embeddings_close` |
