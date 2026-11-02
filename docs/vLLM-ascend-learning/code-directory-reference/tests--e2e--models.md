# tests/e2e/models：目录与文件功能

[返回总索引](../code-directory-guide.md) · [学习文档入口](../README.md)

本分册覆盖 **28 个条目**。说明对应当前本地源码；右栏中的定义、注释或配置项用于追溯依据。

用途说明不等于运行验证；按文件角色归类的条目不保证已逐行确认全部算法。

## `tests/e2e/models/`

模型级测试配置及辅助定义。

| 子目录 | 职责 |
| --- | --- |
| [configs/](../../../tests/e2e/models/configs) | 端到端 配置 场景、运行脚本和配置。 |

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [conftest.py](../../../tests/e2e/models/conftest.py) | 为当前测试范围提供 pytest fixture、mock、环境或插件准备。 | 定义：`pytest_addoption`, `config_list_file`, `tp_size` |
| [report_template.md](../../../tests/e2e/models/report_template.md) | 当前目录的介绍、使用步骤、设计说明或操作参考：report_template。 | 标题：{{ model_name }} |
| [test_asr_eval_correctness.py](../../../tests/e2e/models/test_asr_eval_correctness.py) | 验证 asr eval correctness 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_asr_eval_param` |
| [test_lm_eval_correctness.py](../../../tests/e2e/models/test_lm_eval_correctness.py) | 验证 lm eval correctness 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_lm_eval_correctness_param` |
| [test_rm_eval_correctness.py](../../../tests/e2e/models/test_rm_eval_correctness.py) | 验证 rm eval correctness 的测试；具体断言场景由右栏测试符号定位。 | 定义：`test_rm_eval_param` |

## `tests/e2e/models/configs/`

端到端 配置 场景、运行脚本和配置。

| 文件 | 功能 | 源码依据 / 阅读入口 |
| --- | --- | --- |
| [ERNIE-4.5-21B-A3B-PT.yaml](../../../tests/e2e/models/configs/ERNIE-4.5-21B-A3B-PT.yaml) | ERNIE-4.5-21B-A3B-PT 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Hunyuan-A13B-Instruct.yaml](../../../tests/e2e/models/configs/Hunyuan-A13B-Instruct.yaml) | Hunyuan-A13B-Instruct 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [InternVL3_5-8B-hf.yaml](../../../tests/e2e/models/configs/InternVL3_5-8B-hf.yaml) | InternVL3 5-8B-hf 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Llama-3.2-3B-Instruct.yaml](../../../tests/e2e/models/configs/Llama-3.2-3B-Instruct.yaml) | Llama 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Minitron-8B-Base.yaml](../../../tests/e2e/models/configs/Minitron-8B-Base.yaml) | Minitron-8B-Base 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Mixtral-8x7B-Instruct-v0.1.yaml](../../../tests/e2e/models/configs/Mixtral-8x7B-Instruct-v0.1.yaml) | Mixtral-8x7B-Instruct-v0.1 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `hardware`, `serve`, `envs`, `tasks` |
| [Molmo-7B-D-0924.yaml](../../../tests/e2e/models/configs/Molmo-7B-D-0924.yaml) | Molmo-7B-D-0924 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen2-Audio-7B-Instruct.yaml](../../../tests/e2e/models/configs/Qwen2-Audio-7B-Instruct.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen2.5-Math-RM-72B.yaml](../../../tests/e2e/models/configs/Qwen2.5-Math-RM-72B.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `system_prompt` |
| [Qwen3-30B-A3B-W8A8.yaml](../../../tests/e2e/models/configs/Qwen3-30B-A3B-W8A8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-30B-A3B.yaml](../../../tests/e2e/models/configs/Qwen3-30B-A3B.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-8B-W8A8.yaml](../../../tests/e2e/models/configs/Qwen3-8B-W8A8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-8B.yaml](../../../tests/e2e/models/configs/Qwen3-8B.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-ASR-1.7B.yaml](../../../tests/e2e/models/configs/Qwen3-ASR-1.7B.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-Next-80B-A3B-Instruct.yaml](../../../tests/e2e/models/configs/Qwen3-Next-80B-A3B-Instruct.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-Omni-30B-A3B-Instruct.yaml](../../../tests/e2e/models/configs/Qwen3-Omni-30B-A3B-Instruct.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-VL-30B-A3B-Instruct.yaml](../../../tests/e2e/models/configs/Qwen3-VL-30B-A3B-Instruct.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-VL-8B-Instruct-W8A8.yaml](../../../tests/e2e/models/configs/Qwen3-VL-8B-Instruct-W8A8.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [Qwen3-VL-8B-Instruct.yaml](../../../tests/e2e/models/configs/Qwen3-VL-8B-Instruct.yaml) | Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [accuracy.txt](../../../tests/e2e/models/configs/accuracy.txt) | 精度 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 依据文件格式及所在目录归类。 |
| [gemma-3-4b-it.yaml](../../../tests/e2e/models/configs/gemma-3-4b-it.yaml) | Gemma 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [internlm3-8b-instruct.yaml](../../../tests/e2e/models/configs/internlm3-8b-instruct.yaml) | internlm3-8b-instruct 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
| [llava-onevision-qwen2-0.5b-ov-hf.yaml](../../../tests/e2e/models/configs/llava-onevision-qwen2-0.5b-ov-hf.yaml) | 视觉编码、Qwen 模型 测试输入、运行参数或预期数据；由相邻测试/脚本读取。 | 顶层项：`model_name`, `model_type`, `hardware`, `serve`, `tasks` |
