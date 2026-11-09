---
name: minicpm3-4b-ascend-adaptation
description: "MiniCPM3-4B 在 vLLM Ascend 上的适配经验：上游模型复用、trust_remote_code、MLA/LongRoPE、BF16 测试矩阵、故障排查与验证命令。"
disable-model-invocation: true
---

# MiniCPM3-4B Ascend NPU Adaptation

## 1. 目标与结论

为 OpenBMB `MiniCPM3-4B` 补齐 vLLM Ascend 的 E2E 配置、教程和支持矩阵。结论如下：

- `MiniCPM3ForCausalLM` 已在 upstream vLLM 注册，Ascend 无需复制模型代码或新增 `vllm_ascend/models` 适配层。
- 必须设置 `trust_remote_code=True`，因为 checkpoint 自带 `configuration_minicpm.py` 和 `modeling_minicpm.py`。
- 当前验证边界为 **BF16 + Atlas A2 + 单卡**；TP2 用例已补齐，需在 CI 硬件上确认。
- 310P、W8A8/W4A8、FP16、LoRA、MTP/投机解码不在本次支持范围。

## 2. 模型与运行路径

| 项 | 值 |
| --- | --- |
| architecture | `MiniCPM3ForCausalLM` |
| 参数量 | 约 4B，62 层，hidden size 2560 |
| Attention | MLA，`qk_nope_head_dim=64`、`qk_rope_head_dim=32`，每头 96 维 |
| KV 压缩 | `kv_lora_rank=256`，`q_lora_rank=768` |
| 位置编码 | LongRoPE，`max_position_embeddings=32768` |
| 权重 | BF16，约 7.6 GiB |
| 特殊配置 | `scale_emb=12`、`scale_depth=1.4`、`dim_model_base=256` |
| E2E 长度 | `max_model_len=4096`，用于覆盖 2K 中英文长提示 |

上游实现路径为 `vllm/model_executor/models/minicpm3.py`，它复用 `MiniCPMModel`，只替换 MLA attention 和 decoder layer。权重加载、TP 切分和采样走 vLLM 通用路径，因此不应在 vllm-ascend 重复实现，否则容易与上游 checkpoint 映射漂移。

## 3. 关键问题与处理

### 3.1 缺少 trust_remote_code

**现象：**

```text
ImportError: cannot import name 'MiniCPM3Config'
```

**原因：** checkpoint 的 `config.json` 使用 `AutoConfig`/`AutoModelForCausalLM` 自定义映射，仓库内自带配置与建模代码。

**处理：** 离线 `LLM(...)`、`vllm serve` 和 E2E YAML 全部设置 `trust_remote_code=True` / `--trust-remote-code`。这是硬要求，不是可选优化。

### 3.2 LongRoPE 配置归一化告警

**现象：**

```text
`rope_parameters`'s short_factor field must have length 48, got 16
```

**原因：** 原始 `config.json` 提供的是 `rope_scaling`（LongRoPE），部分 vLLM 版本会将其归一化为 `rope_parameters` 并校验 factor 长度。MiniCPM3 的 `qk_rope_head_dim=32`，按旋转维度推导出的因子数量为 16。

**处理：** 若服务能完成启动且中英文输出正确，将该告警视为兼容性提示，不要通过改写 checkpoint 配置绕过。若启动失败，记录 vLLM commit、`config.json` 的 `rope_scaling` 段和完整 traceback，再判断是 vLLM 归一化问题还是 Ascend RoPE kernel 问题。

### 3.3 triton-ascend 预编译头失败

**现象：**

```text
RuntimeError: Failed to compile ... precompiled.h.gch ...
clang++: error: cannot specify -o when generating multiple output files
```

**原因：** 旧版 `triton/backends/ascend/utils.py` 在生成 `.gch` 时同时传入 `-shared` 和 `-o`，clang 不接受该组合。这是 triton-ascend 工具链问题，不是 MiniCPM3 模型代码问题。

**处理：** 优先升级到已修复的 `triton-ascend`，不要在本仓库 patch 第三方包。排查时可清理缓存：

```bash
rm -rf ~/.triton/cache/* /tmp/triton/*
```

并用 `--enforce-eager` 判断故障是否只发生在 ACLGraph capture 阶段。

### 3.4 性能与显存

- BF16 权重约 7.6 GiB，32 GB 单卡即可运行。E2E 配置用 `gpu_memory_utilization=0.8` + `enforce_eager: true`：eager 避开 aclgraph 捕获的内存峰值（NPU + 宿主机），是 32 GB 卡/小内存容器稳定运行的关键；`batch_size: 8` 替代 lm-eval 的 `auto` 探测，避免批探测把内存顶满。
- 单卡 4B 模型不需要 TP。仅在显存不足或吞吐实验时使用 TP2；TP 必须整除 40 个 attention heads。
- 32K 是模型理论上限，不应直接作为服务默认值。先用 4096 验证，再按显存和长文本准确率逐步提高。

### 3.5 离线/在线对比的两个易错点

写「离线结果 == 在线结果」时最容易写出永远不会通过的断言，实测踩到两处：

1. **`VllmRunner.generate` / `generate_greedy` 返回的是 prompt + completion。** `_finalize_generate_outputs` 里执行的是 `prompt_ids + output_ids` 和 `prompt_str + output_str`，而 OpenAI `/v1/completions` 只返回 completion。直接比较文本或 token 必然不等。改用 `generate_w_logprobs` / `generate_greedy_logprobs`，它们返回 `sample.token_ids` 和 `sample.text`，即纯 completion。
2. **采样参数不会自动带到 HTTP 请求里。** 若在线请求固定写 `temperature=<配置值>`，那么「greedy 对比」实际仍在按 0.7 采样。greedy 必须显式传 `temperature=0`，并且只在 `temperature > 0` 时才附带 `top_p`。

另外 `tests/e2e/models/conftest.py` 的 `tp_size` fixture 是字符串（`--tp-size` 默认 `"1"`），传给 `tensor_parallel_size` 前要 `int()`；`config_filename` 是 `Path`，不是字符串。

## 4. 有效提示词（可复用）

以下提示词适合让 AI 编程助手快速定位模型适配边界，避免它直接生成重复模型代码：

```text
在 vllm-ascend 仓库中检查 MiniCPM3ForCausalLM 是否已被 upstream vLLM 注册。
只阅读注册表、模型实现和已有 E2E 配置，先判断是否需要 Ascend 自定义模型。
输出：注册位置、Attention 类型、权重映射风险、必需启动参数，以及最小改动文件列表。
不要修改文件。
```

```text
为 MiniCPM3-4B 设计 pytest E2E，参数化 tp_size=1/2。
覆盖离线 LLM 与 OpenAI /v1/completions，greedy 与固定 seed 的 temperature/top_p 采样，
中英文短提示和 2K token 长提示；greedy 结果与 Transformers 对比。
要求检查空输出、重复运行内存增长、非 greed 采样可复现，并在非 MiniCPM 配置下 skip。
沿用 tests/e2e/models/conftest.py 的 --config/--tp-size fixture。
```

```text
服务启动成功但 MiniCPM3 首个请求崩溃时，按以下顺序隔离：
1) 同参数复现；2) --enforce-eager；3) 清理 triton 缓存；
4) 检查 rope_parameters/LongRoPE 告警；5) 对比 upstream vLLM 的 minicpm3.py。
输出每一步的结论和日志签名，不要把 startup complete 当作通过。
```

```text
请只修改模型适配所需的文件：E2E YAML、生成测试、教程、支持矩阵。
不要引入 transformers 升级，不要复制 upstream minicpm3.py，除非能证明 Ascend 侧存在
已复现的 kernel/op 缺陷。最后运行 ruff 和 pytest --collect-only。
```

```text
在写任何断言前，先读 tests/e2e/conftest.py 里 VllmRunner 的实现，确认：
1) generate / generate_greedy 返回的 token_ids 与 text 是否包含 prompt；
2) generate_w_logprobs / generate_greedy_logprobs 返回什么；
3) RemoteOpenAIServer 的构造参数哪些是 keyword-only，返回体里 token_ids 字段叫什么。
再据此决定离线与在线两侧如何对齐，不要假设两边格式一致。
```

## 5. 最佳实践

1. **先查注册表，再写代码。** MiniCPM3 的 MLA 和通用 Attention 已在上游；真正的适配点是运行参数、dtype、设备能力和测试。
2. **`trust_remote_code` 必须显式写入配置和文档。** 仅在本机命令里加参数，CI 仍会失败。
3. **验收证据分层。** `Application startup complete` 只证明进程存活；必须追加 `/v1/models`、首个 greedy 请求、非 greedy 请求和长提示请求。
4. **区分理论长度和实测长度。** 支持矩阵写 32768 时，必须同时注明 E2E 使用 4096。
5. **TP 用例独立参数化。** 通过 `--tp-size` 复跑同一配置，不复制 YAML；TP2 需要至少两张空闲 NPU。
6. **确定性比较优先。** greedy 输出可逐 token 与 Transformers 对齐；非 greedy 只断言固定 seed 下可复现和非空，避免跨后端 RNG 误报。
7. **内存回归看趋势。** 先 warmup，再重置统计或记录基线，重复相同 batch 多轮；阈值用绝对值（如 512 MiB）而不是不稳定的百分比。
8. **不要修改第三方 triton 源码作为交付。** 记录版本和上游修复链接，仓库内提供升级/fallback 指引。
9. **支持矩阵不夸大。** 未在 CI 跑过的 TP2、A3、量化项使用 🟡/❌ 或留空，并在 Note 中说明边界。

## 6. 复现命令

```bash
# 单卡：精度 + 离线/在线生成矩阵
python -m pytest -sv tests/e2e/models \
  --config tests/e2e/models/configs/MiniCPM3-4B.yaml \
  --tp-size 1

# 双卡：同一套用例验证 TP2
python -m pytest -sv tests/e2e/models \
  --config tests/e2e/models/configs/MiniCPM3-4B.yaml \
  --tp-size 2

# 只收集，检查配置和 skip 逻辑
python -m pytest --collect-only -q tests/e2e/models \
  --config tests/e2e/models/configs/MiniCPM3-4B.yaml
```

保存日志：

```bash
mkdir -p tests/e2e/models/logs
python -m pytest -sv tests/e2e/models \
  --config tests/e2e/models/configs/MiniCPM3-4B.yaml \
  --tp-size 1 2>&1 | tee tests/e2e/models/logs/minicpm3-4b-tp1.log
```

### 6.1 本地静态验证记录

在没有 NPU 的机器上可以完成以下检查，结果均为通过：

| 检查 | 命令 | 结果 |
| --- | --- | --- |
| 语法 | `python -m py_compile tests/e2e/models/test_minicpm3_generation.py` | 通过 |
| lint | `uvx ruff check tests/e2e/models/test_minicpm3_generation.py` | `All checks passed!` |
| 格式 | `uvx ruff format --check tests/e2e/models/test_minicpm3_generation.py` | `1 file already formatted` |
| YAML 解析 | `yaml.safe_load` 读取 `MiniCPM3-4B.yaml` | 通过，顶层键 = `apply_chat_template/batch_size/fewshot_as_multiturn/generation/hardware/model_name/model_type/num_fewshot/serve/tasks` |
| 矩阵列数 | 解析表头与新增行单元格数量 | 21 == 21，各单元格与表头一一对应 |
| 文档链接 | 逐条检查相对路径目标文件 | 4 个链接目标均存在 |
| 空白错误 | `git diff --check` | 无输出 |

必须注意：`test_lm_eval_correctness.py` 用 `eval_config.get()` 逐项取字段，因此 YAML 里多出的 `generation:` 块不会影响它；但 `serve` 下只有 `trust_remote_code`、`max_model_len`、`dtype`、`gpu_memory_utilization`、`enforce_eager`、`enable_expert_parallel`、`enable_thinking`、`quantization`、`max_images`、`tensor_parallel_size` 会被转成 lm-eval 的 `model_args`，新增其他 `serve` 字段是无效的。

## 7. 交付文件

| 文件 | 作用 |
| --- | --- |
| `tests/e2e/models/configs/MiniCPM3-4B.yaml` | GSM8K 5-shot 与生成矩阵参数 |
| `tests/e2e/models/test_minicpm3_generation.py` | 离线/在线、greedy/非 greedy、中英文长短提示、TP、内存、Transformers 对比 |
| `tests/e2e/models/configs/accuracy.txt` | 纳入精度测试清单 |
| `.github/workflows/configs/nightly_config.yaml` | 纳入 A2 PR 精度组 |
| `docs/source/tutorials/models/MiniCPM3-4B.md` | 下载、启动、参数、验证、限制教程 |
| `docs/source/user_guide/support_matrix/supported_models.md` | dtype、设备、TP 与限制矩阵 |

## 8. 已知边界

- 本记录不替代真实 NPU 验收。若测试环境没有 Ascend 设备，只能完成静态检查和用例收集，必须在 PR 中明确标注未跑项。
- 本次未修改 upstream vLLM 或 `vllm_ascend/models`；如果后续 CI 暴露 MLA head size / LongRoPE 的 Ascend kernel 限制，应附最小复现和 profiler 证据后再加最小 patch。
- 支持状态保持 🔵 Experimental，直到 TP1/TP2 的 CI 日志和 GSM8K 实测结果齐全。
