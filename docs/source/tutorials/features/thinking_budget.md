# Thinking token budget

Thinking budget 为每个请求限制推理阶段的 token 数。达到预算后，模型进入答案阶段；
`max_tokens` 仍限制整个输出，因此应为答案和结束标记预留空间。
预算限制 reasoning 标记之间的 token；答案部分仍可包含解释。
该功能适用于配置了 reasoning 标记的生成任务，不限制答案阶段的 token 数。
未配置 reasoning 时，预算状态不启用；普通模型使用 `max_tokens` 控制总输出长度。
V1 未配置 reasoning 时设置非负预算，会在请求校验阶段报错；未设置、`null` 或 `-1` 可用于普通生成。

## 请求参数

| `thinking_token_budget` | 行为 |
| --- | --- |
| 未设置、`null`、`-1` | 不限制 thinking token 数，仍受最大输出长度约束 |
| `0` | 已进入 thinking 状态时直接生成结束标记，继续生成答案 |
| 非负整数 | 达到预算后强制输出 reasoning 结束序列 |
| 小于 `-1`、布尔、浮点或字符串 | 请求校验阶段报错 |

预算按 token ID 及 reasoning 标记计算，而非字符数。起始标记后的 prompt 尾部也属于
已有 reasoning 内容；预算控制不能用对展示文本重新分词的方式复核。
多请求可以设置不同预算，预算状态跟随请求的加入、重排、抢占恢复和移除。
V2 在设备端使用 int32 保存预算，超过 `2147483647` 的值按该上限保存；
实际输出长度仍由 `max_tokens` 和模型上下文长度约束。

## 离线推理

以下配置使用 Model Runner V2，以 Qwen3 的 `<think>` / `</think>` 标记为例。
先设置：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

再运行：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-1.7B",
    dtype="bfloat16",
    max_model_len=4096,
    max_num_seqs=4,
    max_num_batched_tokens=512,
    kv_cache_memory_bytes=2 * 1024**3,
    reasoning_config={
        "reasoning_parser": "qwen3",
        "reasoning_start_str": "<think>",
        "reasoning_end_str": "</think>",
    },
)
tokenizer = llm.get_tokenizer()
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 2 + 3?"}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
outputs = llm.generate(
    [prompt, prompt],
    [
        SamplingParams(temperature=0, max_tokens=512, thinking_token_budget=0),
        SamplingParams(temperature=0.6, max_tokens=512, thinking_token_budget=64),
    ],
)
for output in outputs:
    print(output.outputs[0].text)
```

## OpenAI 服务

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3-1.7B \
    --dtype bfloat16 --max-model-len 4096 --max-num-seqs 4 \
    --max-num-batched-tokens 512 --kv-cache-memory-bytes 2147483648 \
    --reasoning-parser qwen3 \
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="Qwen/Qwen3-1.7B",
    messages=[{"role": "user", "content": "What is 2 + 3?"}],
    max_tokens=512,
    extra_body={
        "thinking_token_budget": 64,
        "chat_template_kwargs": {"enable_thinking": True},
        "return_token_ids": True,
    },
)
print(response.choices[0].message.content)
```

`return_token_ids` 用于保存实际生成 token，便于按标记边界核对预算。流式响应需要按顺序
拼接各 chunk 的 token ID。对非法预算的检查应发生在请求进入执行队列前。

## N-gram speculative decoding

在支持该路径的 Model Runner V2 上可增加以下参数，无需额外 draft 模型：

```bash
--speculative-config '{"method":"ngram_gpu","num_speculative_tokens":3,"prompt_lookup_min":2,"prompt_lookup_max":5}'
```

接受的草稿才进入已提交历史；拒绝的草稿不能消耗下一轮的预算。启用投机后，应通过
草稿数量、接受数量和实际输出确认投机路径生效。没有命中的普通生成不能代替拒绝采样测试。

## 验证配置

| 项目 | 配置 |
| --- | --- |
| 硬件与 dtype | 单卡 Ascend 910B3，BF16，TP=1 |
| 模型 | Qwen3-1.7B；普通模型回归使用 Qwen2.5-0.5B-Instruct |
| 执行与采样 | eager / ACL Graph，greedy / temperature=0.6 |
| 投机 | 关闭 / ngram_gpu，3 个 draft token |
| 请求预算 | 未设置、null、-1、0、1、2、3、4、16、64、128 |
| 混合 batch | 最多 4 个同时运行的请求，独立预算 |

## 测试入口

```bash
python -m pytest -sv tests/ut/worker/test_thinking_budget_batch.py \
    tests/ut/sample/test_thinking_budget_rejection.py \
    tests/ut/sample/test_thinking_budget_validation.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_generation.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_serving.py
```

## 排查

- 没有进入答案阶段：核对 reasoning 标记及 parser，检查 `max_tokens` 是否留有答案空间。
- 数量比预算少：模型可以自然结束思考；prompt 的起始标记后内容也可能已经占用预算。
- 未观察到投机加速：先确认草稿命中与接受数据，再比较相同负载；不同输出长度不能直接作为加速证据。
- 预算关闭后行为变化：固定权重、prompt token、seed、batch 和执行模式比较，检查请求槽位复用后状态是否已清理。

## 输出一致性与性能对照

Greedy 在当前 logits 上选择最大值；默认 BF16 前向计算可能随 batch 布局产生数值差异，
接近或并列的候选会因此生成不同后续文本。比较预算开关时，固定 prompt token ID、请求行、
执行模式与缓存条件，并加入大于最大输出长度的预算，区分预算处理与强制结束的影响。
不要把预算值拼进 prompt，否则比较的输入内容也会改变。

需要跨 batch 的确定性输出时，可使用已有的 batch-invariant 配置进行验证：

```bash
--additional-config '{"rl_config":{"enabled":true,"enable_batch_invariant":true}}'
```

该配置需要包含 batch-invariant 算子的环境，详细说明见
[RL 配置](../../user_guide/configuration/additional_config.md)。它会使用确定性计算路径，
并将权重布局设为 ND；应按部署负载重新测量性能。

性能对照应使用完全相同的 prompt，并记录实际输出 token、EOS、草稿数量和接受数量。
正常服务评测遵守 EOS；`ignore_eos=True` 的固定长度微基准会继续生成 EOS 后的内容，
重复内容可能提高 n-gram 命中率，不能直接代表正常答案生成的加速比。

## V1 兼容适配

V1 的 `NPUInputBatch` 使用 `NPUThinkingBudgetStateHolder`，继承上游请求状态管理，
补齐开启投机但当前步骤无草稿时的独立请求行映射，以及多 token reasoning 结束序列的
接受／拒绝进度。结束序列按已提交输出推进，对验证窗口中的每个结束 token 强制采样。
V2 使用原有内核和状态路径。

适配对应上游 [#62970](https://github.com/vllm-project/vllm/pull/62970) 和
[#61005](https://github.com/vllm-project/vllm/pull/61005)。所有受支持的 vLLM 版本包含
这两项修复后，移除该子类并恢复上游 factory。

```bash
python -m pytest -sv tests/ut/sample/test_thinking_budget_v1.py
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_v1.py
```

V1 自然语言与 OpenAI 服务回归可单独运行：

```bash
python -m pytest -sv tests/e2e/pull_request/one_card/test_thinking_budget_v1.py -k natural
```

这组检查使用自然输出，包含 greedy/non-greedy、混合预算、HTTP 参数校验及流式输出。
无限预算请求仍可能以 `length` 结束；有限预算测试同时检查预算、reasoning 结束标记和答案。
