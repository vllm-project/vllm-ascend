# GumbelSample

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Ascend 910B 系列（`ascend910b`） | √ |
| Ascend 910_93 系列（`ascend910_93`） | √ |
| Ascend 310P | × |

## 功能说明

`GumbelSample` 用于 vLLM Model Runner V2 的随机采样。算子根据请求的
temperature、seed 和当前位置生成与 Triton 实现一致的
Philox4x32-10 随机数及 Gumbel 噪声，然后从每一行 logits 中选出 token。
当 temperature 为 0 时，算子执行 greedy argmax，不生成 Gumbel 噪声。

对于第 \(i\) 个 token，令请求状态索引
\(r=\mathrm{idx\_mapping}[i]\)，词表索引为 \(j\)，则：

$$
\hat{L}_{i,j} =
\begin{cases}
L_{i,j}/T_r, & T_r \ne 0\ \text{且}\ \mathrm{apply\_temperature=True} \\
L_{i,j}, & \text{其他情况}
\end{cases}
$$

$$
G_{i,j}=-\ln\left(-\ln\left(U_{i,j}+\epsilon\right)+\epsilon\right),
\quad \epsilon=10^{-20}
$$

$$
\mathrm{sampled}_i =
\begin{cases}
\arg\max_j L_{i,j}, & T_r=0 \\
\arg\max_j\left(\hat{L}_{i,j}+G_{i,j}\right), & T_r\ne0
\end{cases}
$$

其中，\(U_{i,j}\) 由 `seeds[r]`、`pos[i]` 和词表索引 \(j\) 通过
Philox4x32-10 生成。相同输入、seed 和 pos 会产生确定性相同的结果。

算子还可以将加 Gumbel 噪声之前的 \(\hat{L}\) 写入可选的
`output_processed_logits`。该写回仅在调用方显式传入输出缓冲区时发生，
常规 Eagle3 + Qwen3 推理路径传入 `None` 时不会写回。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|:---|:---:|:---|:---:|:---:|
| `logits` | 输入 | 二维张量，shape 为 `[num_tokens, vocab_size]`。每行是一个 token 的词表 logits。 | FLOAT | ND |
| `idx_mapping` | 输入 | 一维张量，shape 为 `[num_tokens]`。`idx_mapping[i]` 表示第 `i` 行对应的请求状态索引，取值范围为 `[0, num_req_states)`。 | INT32 | ND |
| `temperature` | 输入 | 一维张量，shape 为 `[num_req_states]`。值为 0 时使用 greedy sampling；非 0 时启用 Gumbel sampling。 | FLOAT | ND |
| `seeds` | 输入 | 一维张量，shape 为 `[num_req_states]`。每个请求状态使用的随机种子。 | INT64 | ND |
| `pos` | 输入 | 一维张量，shape 为 `[num_tokens]`。每个 token 的随机数位置计数。 | INT64 | ND |
| `output_processed_logits_col` | 可选输入 | 标量或 shape 为 `[1]`。指定三维 `output_processed_logits` 的写回列；未传入时默认为 0。 | INT64 | ND |
| `apply_temperature` | 属性 | 是否在生成 Gumbel 噪声前执行 `logits / temperature`，默认值为 `True`。greedy sampling 不执行温度缩放。 | BOOL | - |
| `sampled` | 输出 | 一维张量，shape 为 `[num_tokens]`，保存每一行选中的 token ID。 | INT64 | ND |
| `output_processed_logits` | 可选输出 | 调用方提供的原地写回缓冲区。支持 `[max_num_reqs, vocab_size]` 或 `[max_num_reqs, num_speculative_steps, vocab_size]`。 | FLOAT | ND |

## `output_processed_logits` 写回规则

- 只有传入 `output_processed_logits` 且其 shape 合法时才会写回。
- 二维缓冲区按
  `output_processed_logits[idx_mapping[i], :]` 写回；此时
  `output_processed_logits_col` 应省略或为 0。
- 三维缓冲区按
  `output_processed_logits[idx_mapping[i], output_processed_logits_col, :]`
  写回。
- ACLGraph padding 请求的 `idx_mapping[i]` 为 `-1`。算子会跳过该行的
  temperature、seed 和 processed logits 访问，并将 sampled 占位值写为
  `0`；该值会被上层忽略。
- 当 `output_processed_logits_col` 超出三维缓冲区的 speculative step
  范围时，算子不执行 processed logits 写回，避免越界访问。
- 写回内容不包含 Gumbel 噪声。
- 当 `temperature[idx_mapping[i]] != 0` 且
  `apply_temperature=True` 时，写回 `logits[i] / temperature`；其他情况写回
  原始 `logits[i]`。
- Model Runner V2 仅在上层采样流程需要保存 processed logits 时提供该缓冲区。
  Eagle3 默认 greedy draft 路径中该参数为 `None`；当
  `draft_sample_method="probabilistic"` 时会为每个 draft step 提供该缓冲区。

## 约束说明

- 支持动态图和 ACLGraph。
- 所有输入张量应位于同一 NPU 设备。
- `logits` 必须为二维 FLOAT 张量，且 `num_tokens` 和 `vocab_size` 均大于
  0。
- `idx_mapping` 和 `pos` 的长度必须等于 `num_tokens`。
- `temperature` 和 `seeds` 使用相同的 `num_req_states`，且
  `idx_mapping` 中的值不能越界。
- 当前 AscendC kernel 按 4096 个词表元素分块。
- 当前 kernel 使用
  `min(num_tokens, vector_core_count)` 个 Vector Core，并让每个 core 处理一行
  或多行。小 `num_tokens` 场景无法充分利用所有 core，性能可能低于 Triton
  实现。
- 不支持 FP64 Gumbel sampling。
- Philox 的有符号高位乘法、均匀分布转换、Gumbel 变换和相同最大值时取首个
  索引的行为均与现有 Triton 实现保持一致。

## 调用示例

```python
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

device = "npu"
num_tokens = 4
num_req_states = 2
vocab_size = 151936

logits = torch.randn(
    num_tokens, vocab_size, dtype=torch.float32, device=device
)
idx_mapping = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
temperature = torch.tensor([0.6, 0.8], dtype=torch.float32, device=device)
seeds = torch.tensor([1234, 5678], dtype=torch.int64, device=device)
pos = torch.arange(num_tokens, dtype=torch.int64, device=device)

sampled = torch.ops._C_ascend.npu_gumbel_sample(
    logits,
    idx_mapping,
    temperature,
    seeds,
    pos,
    True,
    None,
    None,
)
print(sampled)
```

需要保存第 1 个 speculative step 的 processed logits 时，可以传入三维缓冲区：

```python
processed_logits = torch.empty(
    num_req_states,
    3,
    vocab_size,
    dtype=torch.float32,
    device=device,
)
processed_logits_col = torch.tensor([1], dtype=torch.int64, device=device)

sampled = torch.ops._C_ascend.npu_gumbel_sample(
    logits,
    idx_mapping,
    temperature,
    seeds,
    pos,
    True,
    processed_logits,
    processed_logits_col,
)
```

## 测试

算子精度测试使用 CPU Philox4x32-10 golden 逐元素校验采样结果，并覆盖
temperature、greedy sampling、多个请求状态和 processed logits 写回：

```bash
pytest -v -s \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_gumbel_sample.py
```

## Eagle3 probabilistic 性能复测

测试配置为 Qwen3-8B + Eagle3、`draft_sample_method="probabilistic"`、
40 并发、40 请求、输入 1024 tokens、输出 100 tokens、temperature 0.01。
以下数据为三轮复测均值，两种实现的接受率均为 100%，平均接受长度均为 4：

| 实现 | 平均 TTFT (ms) | 平均 TPOT (ms) | 输出吞吐 (tok/s) | 总吞吐 (tok/s) |
|---|---:|---:|---:|---:|
| Triton | 2929.477 | 65.460 | 417.203 | 4722.753 |
| AscendC | 2937.597 | 37.050 | 593.793 | 6721.757 |

相较 Triton，AscendC 输出吞吐提升 42.33%，平均 TPOT 降低 43.40%。
TTFT 主要由 prefill 和排队开销决定，两者基本持平。
