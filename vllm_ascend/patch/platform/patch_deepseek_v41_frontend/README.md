# DeepSeek V4.1 前端 monkey patch

这是对 vLLM 前端的 day0 临时适配，放在 `patch/platform/`，随现有
`adapt_patch(is_global_patch=True)` 加载，注册独立的 `deepseek_v41`
tokenizer、renderer、reasoning parser 和 tool parser。原有 `deepseek_v4`
注册和编码保持不变。待 vLLM 上游支持相同协议后移除该 patch。

## 格式来源

`encoding.py`、编码测试和 fixtures 来自正式 V4.1 权重仓库的 `encoding/`
及 `inference/examples/`，保留 DeepSeek MIT 许可。编码器仅调整 Python
格式、类型注解、异常捕获及使用仓库要求的 `regex`，格式语义由原始 golden
outputs 验证；运行时不读取权重目录中的 Python 文件。

- thinking 默认开启；`low/high/xhigh/max` 对应 `25/50/75/100`，默认 50。
- `reasoning_effort="none"` 或显式关闭 thinking 切换至 chat 模式。
- 数字预算 1–100 通过 `chat_template_kwargs.reasoning_effort` 传入。
  `minimal/medium` 不是参考格式的有效预算，返回参数错误。
- 显式编码 `<｜System｜>`，保留中途 system、工具结果顺序和历史 thinking
  的参考语义；接受 `reasoning_content`，兼容 vLLM 的 `reasoning` 字段。
- 工具使用带空格的 `calls`、`invoke`、`parameter` 标签；解析保留
  `string="true|false"` 的类型语义及 reasoning/content 空白。
- 图片使用 `<｜deepseek_image｜>` 占位符。renderer 保留原始内容块的
  双换行和图片顺序，使用 vLLM 的媒体加载、安全限制和 UUID 通道。
- `response_format=json_schema` 的 schema 同时写入参考 system 提示。

## 使用

在 day0 的正常模型启动命令后添加：

```bash
--tokenizer-mode deepseek_v41 \
--reasoning-parser deepseek_v41 \
--tool-call-parser deepseek_v41 \
--enable-auto-tool-choice
```

数字预算请求示例：

```json
{
  "model": "v41",
  "messages": [{"role": "user", "content": "计算 17×23"}],
  "chat_template_kwargs": {"thinking": true, "reasoning_effort": 42}
}
```

与 vLLM 一致，`tool_choice=none` 是否从提示中移除工具由
`--exclude-tools-when-tool-choice-none` 控制。

`required`、指定函数及 strict auto tools 使用 V4.1 structural tag。
支持声明顺序的必选/可选参数、字符串、数值、布尔、null、数组、JSON 对象、
字符串 enum/pattern 以及类型 union。顶层跨参数条件和开放属性 schema
显式报错；不会退回旧版 DSML grammar。保留 vLLM 的
`VLLM_ENFORCE_STRICT_TOOL_CALLING` 开关语义。

## 验证与边界

```bash
python -m pytest -q tests/ut/patch/platform/deepseek_v41 \
  tests/ut/patch/platform/test_deepseek_v4_thinking.py
```

测试涵盖参考编码、参数映射、原始请求不被修改、同步/异步图片加载顺序、
完整/分块工具解析、类型语义及 grammar 接受/拒绝。另使用权重目录真实
tokenizer 验证 prompt token IDs 和逐 token 解析，并通过 CPU-only
`vllm launch render` 检查 HTTP render/derender。

这是前端协议适配。day0 当前 V4.1 模型类仍是文本执行路径，视觉 processor、
视觉权重、Engram、整模精度及性能验收属于模型适配；图片编码测试不代表
图片端到端推理通过。

本地 HTTP 验证使用 vLLM `6e448d0ea9`。day0 的固定 vLLM revision
`ba07e4a48f` 的全局 patch 加载和完整模型检查会因基线引用已迁移的
`vllm.model_executor.layers.attention.pcp` 而失败；需由 day0 基线另行对齐。
