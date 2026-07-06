# Processor And Multimodal Baseline

本文档定义 **processor、多模态输入接线、vision-language 入口** 这一层的当前能力基线。

## 1. 这一层解决什么问题

这一层关注：

- tokenizer / processor 是否与当前 `transformers` API 兼容；
- multimodal 输入是如何被整理成 vLLM 可消费格式的；
- vision encoder / image token / embedding 对齐是否进入了正确路径；
- 失败是在 processor 层、模型接线层，还是 attention/backend 层。

## 2. 当前能力基线

当前 skill 默认已有以下经验基线：

- VLM 需要额外分析 processor，不可只看语言模型 adapter；
- `skip_tensor_conversion` 这类 signature mismatch 是高频 processor 层失败；
- text-only isolation 只能用于分层定位，不可默认当最终修复；
- 多模态模型需要分别验证 text request 和 text+image request；
- MM 模型的 attention/backend 问题经常要与 processor 层问题分开分析。

## 3. 当前实现倾向

优先策略是：

- 尽量复用 vLLM 现有 processor 路径；
- 如果 remote processor 与当前 transformers API 不兼容，优先在 `vllm` 侧做兼容；
- 不把 processor 兼容问题直接伪装成 backend 问题处理。

## 4. 典型输入证据

- `config.json`
- `processor_config.json`
- `preprocessor_config.json`
- `auto_map`
- `processing_*.py`
- `tokenizer_config.json`
- text-only 能否启动
- text+image 第一请求错误栈

## 5. 典型失败信号

- `skip_tensor_conversion` 参数不兼容
- processor 导入失败
- image/video/audio prompt item 解析失败
- text-only 成功但 text+image 失败
- 绕过 processor 后暴露出新的 meta tensor / MM encoder 问题

## 6. 适配判断原则

### 6.1 先判断失败层级

优先区分：

1. processor 参数 / API 不兼容
2. multimodal embedding / encoder 接线问题
3. attention/backend 执行问题

### 6.2 文本隔离仅用于定位

如果加了：

- `--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`

只能说明你暂时绕过了 MM 入口，不能说明 VL 模型已经适配成功。

### 6.3 先修 vLLM 入口，再看 Ascend backend

如果失败发生在 processor / processor dispatch / MM input mapping 层，优先修 `vllm`。

## 7. 固定输出模板

```markdown
## Processor And Multimodal Gap Analysis

### 1. Current Capability
- Existing processor path:
- Existing multimodal support assumption:
- Known-good request types:
- Existing transformers compatibility assumptions:

### 2. Model Requirement
- Processor classes from config:
- Remote/local processing behavior:
- Modalities required:
- MM encoder / embedding path requirements:

### 3. Gap
- Processor API mismatch:
- Multimodal dispatch mismatch:
- Input formatting mismatch:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal files to touch:
- Validation focus:
- Stop / escalate condition:
```

## 8. 最常见的适配动作

- 修 processor 兼容层；
- 调整 vLLM multimodal dispatch；
- 修 text+image 请求路径；
- 将 processor 问题与后续 attention/backend 问题分层验证。
