# MTP (Multi-Token Prediction) 性能分析

## 表面上看起来的"浪费"

确实如你所想：
- 主模型要计算 1 + mtp 个 hidden states
- 要计算 1 + mtp 个 logits
- draft tokens 不一定全部被接受

但关键在于 **计算是如何进行的**。

---

## 真正的性能优势：矩阵乘法的批处理

### 传统 decode（无 speculative decoding）

```
Step 1: 主模型 forward [token_0]          → 1 个 token
Step 2: 主模型 forward [token_1]          → 1 个 token
Step 3: 主模型 forward [token_2]          → 1 个 token
...
```

### 有 MTP 的 decode

```
Step 1: 主模型 forward [token_0, draft_0, draft_1, draft_2, draft_3]
                                    ↓ (一次矩阵乘法)
        一次计算 [hs_0, hs_1, hs_2, hs_3, hs_4] + [logits_0, logits_1, ..., logits_4]
                                    ↓
        如果接受率 = 100% → 一步得到 4 个新 token！
        如果接受率 = 75%  → 一步得到 3 个新 token（平均）
```

**关键点**：
- `[token_0, draft_0, draft_1, draft_2, draft_3]` 的 forward 是**一次矩阵乘法**完成的
- 这 5 个 token 的 hidden states 是同时计算的，不是串行的

---

## 性能分析

### 计算角度

假设主模型计算 1 个 token 需要时间 T_main：

```
无 speculative:
  T_total = n * T_main  (n 步)

有 MTP (mtp=4, 接受率=80%):
  理想情况: 平均每步生成 1 + 4*0.8 = 4.2 个 tokens
  T_total = n / 4.2 * (T_main + T_draft) ≈ n / 4.2 * T_main

  加速比 ≈ 4.2x
```

### Memory-bound 角度（更重要的因素）

LLM 的主要瓶颈是 **Memory-bound** 而不是 **Compute-bound**：

```
计算一个 token 的 FLOPs:     O(hidden_dim^2)
读取一个 token 的 KV cache:  O(hidden_dim * seq_len)

对于长序列，读取 KV cache 的时间 >> 计算矩阵乘法的时间
```

MTP 的优势：
- Attention 操作对所有 1 + mtp 个 tokens **一起做**
- KV cache 的读取是一次性完成的（批量读取）
- Draft model 比主模型小得多（T_draft << T_main）

---

## Draft Model 的作用

Draft model 不是免费的，但它是**轻量级的**：

```python
# 主模型: 几十层 Transformer，参数巨大
# Draft model: 只有 1 层或几层，参数少很多

# Draft model 的 forward:
draft_output = draft_model(hidden_states_from_target)
# 这个 draft_output 是主模型 hidden_states 的函数，计算量很小
```

---

## 接受率的影响

| 接受率 | 有效加速 |
|--------|----------|
| 100% | ~mtp+1 x |
| 80% | ~1 + mtp*0.8 x |
| 50% | ~1 + mtp*0.5 x |
| 0% | ~1 x (和普通 decode 一样) |

**关键洞察**：
- 即使接受率不高，只要 **draft model 计算足够轻量 + 主模型的批处理优势**，仍能获得加速
- Draft model 的计算量通常只有主模型的 **5-10%**

---

## 举例

假设：
- T_main = 100ms（主模型计算 1 token）
- T_draft = 5ms（draft 模型计算 1 token，约为主模型的 5%）
- mtp = 4
- 接受率 = 80%

```
无 speculative:
  每步生成 1 token, T = 100ms

有 MTP:
  主模型一次计算 5 tokens: T_main * 1 = 100ms (不是 500ms!)
  Draft 模型: T_draft * 4 = 20ms
  理想情况（100%接受）: 5 tokens / 120ms = 4.17 tokens/100ms → 4.17x 加速

  实际情况（80%接受）:
    有效 tokens = 1 + 4*0.8 = 4.2
    T_total = 120ms
    等效加速 = 4.2 / 1 = 4.2x vs 无 speculative 的 1x
```

---

## 总结

| 因素 | 解释 |
|------|------|
| **批处理矩阵乘法** | 1+mtp 个 tokens 一次计算，而非串行 |
| **KV cache 批量读取** | Attention 一次性处理所有 tokens 的 KV |
| **Draft model 很轻量** | 只有几层，计算量 << 主模型 |
| **接受率** | 决定有效 token 生成速率 |

**核心洞察**：MTP 不是让主模型"多做工作"，而是让主模型在**一次计算中并行生成多个 tokens**，配合轻量级的 draft model 来"预测"可能的下几个 tokens，从而实现 **每步平均生成 >1 个 token** 的效果。

---

## MTP Spec Decode 校验流程

### 校验的整体流程

```
上一轮:
  Draft Model 提出了 [tok1, tok2, tok3, tok4] (4个draft tokens)

这一轮 Target Model Forward:
  input_ids = [real_token] + [tok1, tok2, tok3, tok4]  (5个tokens)
              ↓
  Target Model 对每个位置做 forward
              ↓
  logits[i] = target_model.lm_head(hidden_states[i])  (5个logits向量)
              ↓
  target_argmax = argmax(logits)  (5个预测token)
              ↓
  rejection_sampler 校验
              ↓
  输出: accepted_tokens + bonus_token
```

### 校验的具体逻辑

假设 batch_size=1, mtp=4:

#### 1. Target Model Forward

```python
# input_ids 包含 1 个 real token + 4 个 draft tokens
input_ids = [real_tok, draft_tok1, draft_tok2, draft_tok3, draft_tok4]
#            position 0    position 1  position 2  position 3  position 4

# Target model forward 后得到:
logits = model.lm_head(hidden_states)
# logits shape: [5, vocab_size]
# logits[0] = target 对 real_tok 位置的真实预测
# logits[1] = target 对 draft_tok1 位置的预测
# logits[2] = target 对 draft_tok2 位置的预测
# ...

target_argmax = logits.argmax(dim=-1)
# target_argmax = [pred_real, pred_1, pred_2, pred_3, pred_4]
```

#### 2. Rejection Sampling 校验

**Greedy 情况** (最简单):

```python
# 比较 draft tokens 和 target predictions
draft_tokens = [draft_tok1, draft_tok2, draft_tok3, draft_tok4]
target_preds = [pred_1, pred_2, pred_3, pred_4]

# 逐个比较，找到第一个不匹配的位置
mismatch = draft_tokens != target_preds
# 比如: [False, False, True, False]  →  在 position 2 处不匹配

first_mismatch = mismatch.argmax()  # = 2

# accepted: draft_tok1, draft_tok2 (2个tokens)
# rejected: draft_tok3, draft_tok4 (2个tokens)
# bonus_token: 真实采样自 logits[0] 的 token
```

**输出格式**:

```python
# output_token_ids shape: [batch_size, max_spec_len + 1] = [1, 5]
output_token_ids = [
    [pred_1, pred_2, bonus_token, PLACEHOLDER, PLACEHOLDER]
    #  前2个是accepted的draft, 第3个是bonus(真实采样的token)
]
```

### Shape 总结

| 阶段 | Shape (batch=B, mtp=4) |
|------|------------------------|
| input_ids (这一轮) | `[B * (1+4),]` = `[5B]` |
| hidden_states | `[5B, hidden_dim]` |
| logits | `[5B, vocab_size]` |
| target_argmax | `[5B]` |
| draft_token_ids (从input取出) | `[4B]` |
| output_token_ids | `[B, 5]` |

---

## MTP 工作流程

### 核心概念

MTP 是 **自回归式的多层 draft 模型**，每一层 draft layer 预测 **1个 token**。如果有 `mtp=4`，意味着有 4 层 draft layers，连续预测 4 个 token。

### 数据流图

```
                    Target Model
                         │
input_ids (1 token) ────► │  forward ──► hidden_states ──► logits ──► sampled_token
                         │                              (last position)
                         │                              ↑
                         │                              │
                         └──────────────────────────────┘
                                            │
                              (hidden_states 传给 draft model)
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                   Draft Model                    │
                    │                                               │
                    │  ┌─────────────┐    ┌─────────────┐           │
                    │  │  Draft L0   │───►│  Draft L1   │───► ...   │
                    │  │ (1 token)   │    │ (1 token)   │           │
                    │  └─────────────┘    └─────────────┘           │
                    │                                               │
                    │  hidden_states[0]    hidden_states[1]         │
                    │       ↓                   ↓                   │
                    │   logits[0]           logits[1]               │
                    │       ↓                   ↓                   │
                    │   draft_tok[0]        draft_tok[1]            │
                    └───────────────────────────────────────────────┘
                                        │
                                        ▼
                         [draft_tok[0], draft_tok[1], draft_tok[2], draft_tok[3]]
```

### 详细步骤 (以 mtp=4 为例)

#### 1. Target Model Forward

```
input_ids: [token_0]                    # 1 个 token (当前最新生成的)
           ↓
Target Model Forward
           ↓
hidden_states: [hs_0]                   # shape: [1, hidden_dim]
           ↓
logits = model.lm_head(hidden_states)   # shape: [1, vocab_size]
           ↓
sampled_token = argmax(logits)          # 1 个 token
```

#### 2. Draft Model Forward (循环 4 次)

**Step 0 - Draft L0:**

```python
# 第一次 forward
input_ids = [sampled_token]              # 来自 target model 的采样
hidden_states = target_model.hidden_states  # 复用 target 的 hidden_states

draft_output = draft_model(input_ids, hidden_states, positions)
# draft_output[0] = last_hidden_states  # shape: [1, hidden_dim]
# draft_output[1] = all_hidden_states   # shape: [1, num_layers, hidden_dim]

draft_tok[0] = argmax(draft_model.lm_head(draft_output[0]))
# draft_tok[0]: 1 个 token
```

**Step 1 - Draft L1:**

```python
# 第二次 forward
input_ids = [draft_tok[0]]              # 用上一步的输出
hidden_states = draft_output[1][0]      # 用第一层 draft 的 hidden_states
positions = positions + 1

draft_output = draft_model(input_ids, hidden_states, positions)
draft_tok[1] = argmax(draft_model.lm_head(draft_output[0]))
```

**Step 2 - Draft L2 & Step 3 - Draft L3:**

重复上述过程...

#### 3. 最终 Output

```python
# draft_token_ids shape: [batch_size, num_speculative_tokens] = [B, 4]
draft_token_ids = [draft_tok[0], draft_tok[1], draft_tok[2], draft_tok[3]]
```

### Target Model 如何使用这些 Draft Tokens

在 EagleProposer 中：
```python
# draft_token_ids: [batch_size, num_speculative_tokens]
# 返回后，spec_decode_metadata 会包含这些 draft tokens
# Target model 会在下次 forward 时把这些 tokens 当作 spec_decode_tokens 一起处理
```

---

## MTP 校验流程图

```
                    Target Model
                         │
input_ids: [real_tok, draft_0, draft_1, draft_2, draft_3]
           (5 tokens)
              ↓
         Forward All
              ↓
logits: [logit_0, logit_1, logit_2, logit_3, logit_4]
              ↓
      argmax = [pred_real, pred_0, pred_1, pred_2, pred_3]
              ↓
      Rejection Sampler
              ↓
         Compare:
         draft_0 vs pred_0
         draft_1 vs pred_1
         draft_2 vs pred_2
         draft_3 vs pred_3
              ↓
         Output:
         [accepted_0, accepted_1, ..., bonus_token]
```
