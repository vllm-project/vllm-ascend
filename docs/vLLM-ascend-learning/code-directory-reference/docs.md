# docs：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **3 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `docs/`

项目文档、站点构建、主题覆盖与当前学习资料。

| 子目录 | 职责 |
| --- | --- |
| [hooks/](docs--hooks.md) | 父主题的 hooks 文档、模板或站点资源。 |
| [overrides/](docs--overrides.md) | 父主题的 overrides 文档、模板或站点资源。 |
| [source/](docs--source.md) | 用户与开发文档正文、图片、模板及翻译资源。 |
| [vLLM-ascend-learning/](docs--vLLM-ascend-learning.md) | 本次源码学习、详细设计、PR 分析与适配说明。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [Makefile](../../Makefile) | 文档站点/主题的配置或附属资源。 | 依据文件格式及所在目录归类。 |
| [README.md](../../README.md) | 说明 vLLM Ascend Plugin documents。 | 标题：vLLM Ascend Plugin documents |
| [requirements-docs.txt](../../requirements-docs.txt) | 该目录运行、测试或构建所需的依赖清单。 | 依据文件格式及所在目录归类。 |
