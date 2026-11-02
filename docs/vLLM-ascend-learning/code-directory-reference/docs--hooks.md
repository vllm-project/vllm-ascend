# docs/hooks：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **3 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `docs/hooks/`

父主题的 hooks 文档、模板或站点资源。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [macros_main.py](../../hooks/macros_main.py) | 文档构建、代码引用、主题或内容生成辅助。 | 注释：MkDocs macros plugin main module.；定义：`define_env` |
| [model_code.py](../../hooks/model_code.py) | 文档构建、代码引用、主题或内容生成辅助。 | 注释：MkDocs hook: process model-code blocks before build.；定义：`on_page_markdown`, `on_pre_build` |
| [nav_titles.py](../../hooks/nav_titles.py) | 文档构建、代码引用、主题或内容生成辅助。 | 注释：MkDocs hook: set per-page and per-section titles from a bilingual mapping.；定义：`on_nav`, `on_page_markdown` |
