---
name: vllm-ascend
description: Adapt, test, document, and review large MoE models and NVFP4 checkpoints for vLLM Ascend. Use when implementing an Ascend model adapter, mapping native Mistral or DeepSeek-style weights, registering a quantization backend, adding Linear or fused-MoE NVFP4 loading, creating e2e model YAML and deployment tutorials, or preparing an AI-assisted development report for a vllm-ascend issue.
---

# vLLM Ascend 大模型适配

## 目标

按最小改动原则完成新模型在 vLLM Ascend 上的架构注册、权重映射、量化后端、
测试配置和部 署文档。对 dummy、静态检查、真实权重和 NPU 内核验证分别给出
结论，不要用前一阶段替代后一阶段。

本技能来源于
[Issue #7338](https://github.com/vllm-project/vllm-ascend/issues/7338) 的
Mistral Large 3 675B NVFP4 适配过程，正文可以直接作为该 Issue 的 AI 辅助
开发记录粘贴。

## 本次使用的提示词

### 1. 模型架构适配

```text
项目路径 C:\Users\wangc\vllm-ascend，进入
vllm_ascend/model_executor/models 目录；参考同目录现有模型
（deepseek_v4.py 等）的代码结构、模型装饰器注册、权重加载、MoE 前向传播、
昇腾适配逻辑，新建 mistral_large3.py，实现
MistralLarge3ForCausalLM 类；适配 Mistral Large3 MoE 架构，预留 NVFP4
量化权重加载接口，代码补充基础注释，符合仓库原有编码规范。取消补丁导入
方式，直接在对应目录新建目标 py、yaml、md 文件，不使用临时目录、patch
脚本。
```

### 2. NVFP4 量化适配

```text
项目路径 C:\Users\wangc\vllm-ascend，进入
vllm_ascend/model_executor/quantization 目录；参考目录内 AWQ、GPTQ 量化
实现，新建 nvfp4.py；完成量化类注册、NVFP4 权重反量化、对接昇腾 NPU
算子的基础代码骨架，适配 Mistral-Large-3-675B 的 NVFP4 量化权重。
```

### 3. 端到端测试配置

```text
项目路径 C:\Users\wangc\vllm-ascend，进入 tests/e2e/models/configs
目录；复制仓库现有模型 yaml 测试配置格式，新建
mistral_large3_675b_nvfp4.yaml；填入 HuggingFace 模型路径
mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4、合理张量并行参数、
多组测试 prompt（复杂推理、多语言、企业场景）、推理采样参数，预留服务器
本地权重路径字段，满足端到端测试要求。
```

### 4. 部署教程

```text
项目路径 C:\Users\wangc\vllm-ascend，进入 docs/source/tutorials/models
目录；参考现有模型教程格式，新建 mistral_large3_675b_nvfp4.md；编写内容
包含：环境依赖准备、HuggingFace 模型下载命令、vLLM-Ascend 多卡推理启动
命令、常见报错排查模板，适配 675B NVFP4 模型部署场景。
```

### 5. AI 辅助开发记录

```text
在项目根目录新建 SKILL.md，严格参照 https://agentskills.io/specification
规范撰写；记录本次 AI 辅助开发全流程：使用的提示词、代码适配工作流、
MoE+NVFP4 量化适配的解决思路、可复用开发经验；内容后续可直接粘贴到
Issue #7338 评论区提交。
```

## 代码适配工作流

### 1. 先确认仓库事实

1. 读取根目录 `AGENTS.md`、适用技能和贡献规范。
2. 检查工作树，保留用户已有修改，不重置或覆盖无关文件。
3. 使用仓库搜索确认真实目录、注册入口、配置 schema 和现有测试。
4. 当提示词路径与当前仓库布局不一致时，以可被运行时导入的真实路径为准，
   并在交付说明中记录偏差。

本次发现仓库没有 `vllm_ascend/model_executor/models` 和
`vllm_ascend/model_executor/quantization` 下的对应实现体系。最终使用：

- `vllm_ascend/models/mistral_large3.py`；
- `vllm_ascend/quantization/nvfp4.py`；
- `vllm_ascend/models/__init__.py` 的 `ModelRegistry.register_model`；
- vLLM 的 `register_quantization_config` 和插件启动注册。

不要为了逐字匹配提示词而创建运行时不会导入的重复目录。

### 2. 分析模型和检查点

1. 查看官方 `params.json`、模型卡和权重命名。
2. 确认模型拓扑：MLA、61 层、128 个路由专家、每 token 选择 4 个专家、
   1 个共享专家，以及前三层 dense MLP。
3. 对比上游 vLLM 的 DeepSeek-V3、ModelOpt NVFP4 和 compressed-tensors
   实现，不凭记忆猜测张量格式。
4. 确认 Mistral 原生配置把 NVFP4 描述放在 `quantization_config`，格式为
   `nvfp4-pack-quantized`，但 `quant_method` 标记为 `compressed-tensors`。
5. 先列出权重名称、shape、dtype、分片维度和缩放因子，再写 loader。

### 3. 复用兼容架构

1. 优先复用已经验证的上游模型拓扑，不复制整套 forward。
2. 让 `MistralLarge3ForCausalLM` 继承 `DeepseekV3ForCausalLM`，保留 MLA、
   MoE、专家并行和 Ascend FusedMoE 路径。
3. 使用锚定正则的 `WeightsMapper` 显式映射：
   - attention norm、MLA Q/KV LoRA 投影和 output projection；
   - dense `w1/w2/w3`；
   - routed expert `w1/w2/w3`；
   - shared expert `w1/w2/w3`；
   - embedding、final norm 和 lm head。
4. 映射量化后缀：
   - `.qscale_act` 到 `.input_scale`；
   - `.qscale_weight` 到 `.weight_scale`；
   - `.qscale_weight_2` 到 `.weight_scale_2`。
5. 使用 `AutoWeightsLoader` 统一加载，避免为每种精度复制 loader。

### 4. 实现 NVFP4 量化后端

1. 注册 `nvfp4` 配置，并在插件发现阶段加入平台支持列表。
2. 同时识别显式 `--quantization nvfp4` 和原生
   `nvfp4-pack-quantized` 配置，不能只检查 `quant_method=nvfp4`。
3. 固定并校验 Mistral 检查点使用的 `group_size=16`。
4. 为 Linear 分配：
   - `weight`: `uint8[out, in/2]`；
   - `weight_scale`: `float8_e4m3fn[out, in/16]`；
   - `weight_scale_2`: FP32 全局权重 scale；
   - `input_scale`: FP32 全局激活 scale。
5. 为 MoE 分配：
   - `w13_weight`: `uint8[experts, 2*intermediate, hidden/2]`；
   - `w2_weight`: `uint8[experts, hidden, intermediate/2]`；
   - 对应的 16 元素分组 scale、二级权重 scale 和激活 scale。
6. 支持 `re:` 忽略规则，避免错误量化 attention、embedding、vision 或
   lm head。
7. 将 NPU 快路径接到 `_C_ascend.nvfp4_linear` 和
   `_C_ascend.nvfp4_moe`。算子缺失时在 NPU 上快速报错，不要静默展开 675B
   权重。
8. 仅提供 CPU Linear 参考路径用于验证反量化数学；不要把它当作生产后端。

## NVFP4 反量化思路

每个 `uint8` 沿输入维打包两个 E2M1 FP4 值，低四位在前，高四位在后。
使用下表解码 nibble：

```text
[0, 0.5, 1, 1.5, 2, 3, 4, 6,
 -0, -0.5, -1, -1.5, -2, -3, -4, -6]
```

将 FP8 E4M3 分组 scale 沿最后一维每 16 个元素展开，再乘 FP32 全局 scale：

```text
W_dequant = E2M1(unpack(weight)) * repeat(weight_scale, 16) * weight_scale_2
```

ModelOpt 风格的 `weight_scale_2` 是乘数 `amax / (6 * 448)`，不是倒数。
compressed-tensors 的某些导出格式可能保存倒数；接入新检查点时必须先确认
导出约定，不能复用名称后直接假设公式相同。

避免在热路径调用 `tensor.item()`。shape、dtype 和 group size 校验应在创建
权重或加载阶段完成。

## 测试与文档工作流

### 单元测试

至少覆盖：

- nibble 低位优先的解包顺序；
- E2M1 正负数值表；
- 分组 scale 和全局 scale 的组合；
- 错误 scale shape；
- 原生 Mistral `quantization_config` 自动识别；
- `re:` ignore 规则；
- 模型权重名称映射。

先运行目标用例，再运行格式和静态检查：

```shell
python -m pytest -sv tests/ut/quantization/test_nvfp4.py
python -m pytest -sv tests/ut/quantization/test_nvfp4_mistral_config.py
python -m pytest -sv tests/ut/models/test_mistral_large3.py
python -m py_compile vllm_ascend/models/mistral_large3.py
python -m py_compile vllm_ascend/quantization/nvfp4.py
git diff --check
```

如果本地缺少 `torch`、vLLM 或 NPU，不要声称 pytest 或真实推理通过。记录
阻塞依赖，并继续执行可以完成的 YAML、Markdown、AST、编译和 diff 检查。

### 端到端配置

保持 `tests/e2e/models/configs` 的标准字段：`model_name`、`model_type`、
`hardware`、`serve`、`tasks`、`num_fewshot`。为 675B NVFP4 使用 A3
TP16+EP、131072 初始上下文和 16 并发，并保留本地模型路径。

将服务 smoke prompt 放在独立扩展段，覆盖复杂推理、多语言和企业事故响应；
不要破坏 lm-eval 对标准字段的读取。

### 部署教程

至少写明：

- A3 16 NPU、磁盘和版本依赖；
- `hf download --local-dir` 下载及分片检查；
- `/workspace` 下直接执行的 TP16+EP 启动命令；
- readiness 与首个真实权重请求；
- Eager 和低上下文隔离命令；
- NVFP4 算子缺失、OOM、HCCL、ACLGraph、权重键异常模板；
- dummy 与真实权重不等价；
- 实测前不得将参考精度写成 Ascend 结果。

## 两阶段 NPU 验证门禁

### 阶段 A：dummy 快速门禁

验证架构构建、注册、基础算子路径和 API。必须同时满足 readiness 和至少一个
非空文本响应。dummy 不能验证权重键、FP4 反量化或真实显存占用。

### 阶段 B：真实权重强制门禁

使用完整约 403 GB 检查点，验证：

1. 所有分片完整加载；
2. 无未处理的权重键或 scale；
3. Linear 与 routed/shared expert 的 NVFP4 路径可执行；
4. `GET /v1/models` 返回 200；
5. 首个请求返回 200 且输出非空；
6. TP16+EP、FlashComm1 和 ACLGraph 有运行证据；
7. 记录 TTFT、TPOT、吞吐量及内存峰值。

`Application startup complete` 不是通过条件。readiness 正常但首请求崩溃属于
false-ready，必须按运行时失败处理。

## 常见问题与可复用经验

1. **目录名不等于运行时入口。** 先搜索注册链，再决定文件位置。
2. **架构相似时继承优于复制。** 复用 DeepSeek-V3 forward，只维护明确的
   Mistral 权重映射。
3. **量化方法名不可靠。** 同时检查 format、config group、dtype、group size
   和实际权重字段。
4. **MoE scale 必须区分 w1/w3。** `w13_weight_scale_2` 通常为
   `[experts, 2]`，`w2_weight_scale_2` 为 `[experts]`。
5. **融合层 scale 可能不一致。** 合并 q/k/v 或 gate/up 前先定义保守策略并
   输出警告，不能静默假设完全相等。
6. **不要在 NPU 上使用巨型反量化 fallback。** 缺少内核时快速失败，避免
   OOM 或长时间假运行。
7. **不要从 dummy 推断真实权重正确。** 权重命名、scale 配对和分片问题只在
   阶段 B 暴露。
8. **新增配置必须被现有 runner 消费。** 未被 runner 使用的 prompt 扩展应
   与标准 lm-eval 字段隔离。
9. **文档必须暴露限制。** 教程应明确内核依赖和未验证状态，而不是只给一个
   看似可运行的命令。
10. **每步形成单一、可审计提交。** 使用 Conventional Commit 和 sign-off，
    保持代码、测试、配置和文档差异清晰。

## 本次交付记录

| 提交 | 内容 | 验证状态 |
| --- | --- | --- |
| `efce799c7e30be5c6c70ab6dde9f38043e60caf9` | 模型注册、权重映射、模型 UT、初版 YAML/教程 | `py_compile` 和映射静态检查通过 |
| `afc8f833d38296c16bcafe750aad5d15c7e004ee` | NVFP4 配置、反量化、Linear/MoE NPU 骨架及 UT | 静态检查通过；本机缺少 torch，pytest 未执行 |
| `2e6015594749716355f5299ea818b73a92741a5c` | NVFP4 e2e YAML 与三类 prompt | YAML 解析和字段断言通过 |
| `0ae1a2a159000a195dfb5e19cd985ebf76cfd03c` | 中文下载、部署和排障教程 | Markdown 结构、引用和 diff 检查通过 |

主要文件：

- `vllm_ascend/models/mistral_large3.py`
- `vllm_ascend/quantization/nvfp4.py`
- `tests/ut/models/test_mistral_large3.py`
- `tests/ut/quantization/test_nvfp4.py`
- `tests/ut/quantization/test_nvfp4_mistral_config.py`
- `tests/e2e/models/configs/mistral_large3_675b_nvfp4.yaml`
- `docs/source/tutorials/models/mistral_large3_675b_nvfp4.md`

## 当前结论

已完成模型适配、权重映射、NVFP4 配置和反量化参考实现、Linear/MoE 权重
分配、NPU 算子接口、单元测试源码、e2e 配置及中文教程。

尚未完成真实 Ascend 服务器上的 403 GB 权重加载和首请求验证，也未在当前
环境证明 `_C_ascend.nvfp4_linear` 与 `_C_ascend.nvfp4_moe` 已有可用实现。
因此当前交付应表述为“适配代码与算子接口骨架已完成，等待真实 NPU 内核和
权重门禁”，不能表述为 Issue #7338 已完全验证通过。

## 后续执行清单

1. 在匹配版本的 A3 镜像中确认 vLLM 和 vLLM Ascend import 路径。
2. 下载并核对全部 Mistral Large 3 NVFP4 分片。
3. 确认两个 `_C_ascend` NVFP4 算子已注册且 schema 与 Python 调用一致。
4. 先以低上下文 Eager 模式验证真实权重加载和首请求。
5. 再验证 TP16+EP、FlashComm1、ACLGraph、131072 上下文和 16 并发。
6. 用实测结果替换参考精度和性能数据。
7. 将本文件内容粘贴到 Issue #7338 评论区，附上真实验证日志或明确阻塞项。
