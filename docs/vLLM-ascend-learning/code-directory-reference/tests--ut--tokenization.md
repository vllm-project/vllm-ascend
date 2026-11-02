# tests/ut/tokenization：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **1 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/ut/tokenization/`

验证 tokenization 对应组件的单元/回归行为；子目录按生产模块组织。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [test_lopt_tokenizer.py](../../../tests/ut/tokenization/test_lopt_tokenizer.py) | 验证 分词器 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_split_overlapping_uses_global_character_coordinates`, `test_encode_chunk_converts_unicode_offsets_to_global_coordinates`, `test_position_match_uses_offsets_instead_of_repeated_token_ids`, `test_position_match_supports_multiple_byte_tokens_with_the_same_character_offset` |
