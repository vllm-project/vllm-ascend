# Positional / RoPE Analysis

本文档分析 vLLM Ascend 当前在位置编码与 RoPE 层的实现能力，聚焦 rotary embedding 的类型覆盖、cos/sin cache 管理、partial rope、mrope，以及与 model runner / attention backend 的协作关系。

## 1. 这一层解决什么问题

RoPE 层当前不只是“对 q/k 做旋转”，还承担了：

- 选择具体 rotary 实现
- 管理 cos/sin cache
- 处理 partial rotary / interleaved rotary
- 为 MLA / MRoPE / XD-RoPE 等路径准备位置数据
- 与 model runner 的 positions 同步

这一层连接的是“模型结构里的位置编码定义”和“运行时 attention 所需的输入形态”。

## 2. 当前能力总览

Ascend 当前已将多种 rotary 实现注册为 OOT custom op：

- `RotaryEmbedding -> AscendRotaryEmbedding`
- `MRotaryEmbedding -> AscendMRotaryEmbedding`
- `YaRNScalingRotaryEmbedding -> AscendYaRNRotaryEmbedding`
- `DeepseekScalingRotaryEmbedding -> AscendDeepseekScalingRotaryEmbedding`
- `ApplyRotaryEmb -> AscendApplyRotaryEmb`

注册入口见：

- [utils.py](/home/cmq/code/vllm-ascend/vllm_ascend/utils.py:688)

这说明 RoPE 不是单一实现，而是一个已经按模型类型分叉的能力簇。

## 3. 当前实现的关键能力

### 3.1 cos/sin cache 已被显式拆分管理

`vllm_ascend/ops/rotary_embedding.py` 中维护了：

- `_cos_sin_cache`
- `_cos_cache`
- `_sin_cache`
- `_cos_slice`
- `_sin_slice`
- `_cos_mla`
- `_sin_mla`

并提供：

- `set_cos_and_sin(...)`
- `update_cos_sin(...)`
- `get_cos_and_sin_slice()`
- `get_cos_and_sin_mla(...)`

见：

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:42)

这反映了当前实现的一个核心事实：Ascend 侧很多算子更适合直接消费拆分后的 cos/sin，而不是复用 upstream 的统一 cache 表达。

### 3.2 标准 GQA 路径与 MLA 路径是分开的

当前实现显式区分：

- 标准 GQA/普通 decoder-only 模型
- MLA 模型

在 `set_cos_and_sin(...)` 中：

- MLA 走 `_cos_mla/_sin_mla`
- 非 VL 且有 rope 的普通路径走 `_cos/_sin`

见：

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:62)

因此当前 RoPE 层不是统一抽象；不同 attention 子类型已经在这里分流。

### 3.3 partial rope 已被考虑

对于普通 RoPE 路径，当前实现会根据模型配置调整 rotary_dim：

- `partial_rotary_factor`
- `rotary_dim`

见：

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:79)

这说明当前实现已承认“rope_dim 可以小于 head_dim”，并在 `rope_forward_oot(...)` 中专门处理只旋转前半部分、后半部分 passthrough 的场景。

### 3.4 Triton 与 NPU 原生 rotary 共存

`rope_forward_oot(...)` 的执行路径是：

- 有 Triton 时优先 `rope_forward_triton(...)`
- 否则走 `torch_npu._npu_rotary_embedding`

见：

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:153)
- [ops/triton/rope.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton/rope.py)

当前 Triton rope 已明确支持：

- `rope_dim != head_dim`
- neox / non-neox 两类风格

但当进入 NPU fallback 时，`offsets` 仍不支持 batched rotary。

### 3.5 YaRN、DeepSeek scaling、MRoPE 均已进入正式路径

当前仓库除了普通 `RotaryEmbedding` 外，还显式支持：

- `YaRNScalingRotaryEmbedding`
- `DeepseekScalingRotaryEmbedding`
- `MRotaryEmbedding`
- `rope_dsv4.py` 中更特化的 DSV4 rope 状态管理

相关入口：

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py:253)
- [rope_dsv4.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rope_dsv4.py)

因此当前 RoPE 能力边界并不只覆盖“标准 Llama 式 rotary”，而是已经延伸到更复杂的 scaling 与 grouped rope 形式。

### 3.6 model runner 已负责位置同步

RoPE 不只是 op 层问题，`model_runner_v1.py` 已经接管了：

- `uses_mrope`
- `uses_xdrope_dim`
- `_calc_mrope_positions`
- `_calc_xdrope_positions`
- positions 拷贝到 GPU

见：

- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:964)

这说明当前实现已经把“位置元数据”视作 runner 级状态，而不是 layer 内局部状态。

## 4. 当前结构假设

当前 RoPE 层隐含的结构假设包括：

- rotary 仍然主要作用于 q/k
- cos/sin cache 可以提前准备并重用
- partial rope 可以通过 rotary_dim/head_dim 分离处理
- MRoPE/XD-RoPE 需要 runner 预处理 positions
- MLA 与普通 GQA/decoder path 的 rope 数据通路不同

## 5. 已知边界与风险

当前明确可见的边界有：

- batched rotary `offsets` 在 NPU 路径下仍未支持
- 不同模型的 rope cache 表达可能不同，Ascend 侧大量依赖 cache 预处理
- RoPE 层与 attention backend 有较强耦合，尤其是 MLA / SFA / DSA
- 图模式下需要显式同步 positions/cos/sin，运行时 metadata 不能随意变化

## 6. 分析这一层时应该看什么

建议优先看：

- `rope_type`
- `rope_theta`
- `partial_rotary_factor`
- `rotary_dim`
- `qk_rope_head_dim`
- 是否使用 `mrope` / `xdrope`
- positions 是 layer 内生成还是 runner 预生成

## 7. 相关代码

- [rotary_embedding.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rotary_embedding.py)
- [rope_dsv4.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/rope_dsv4.py)
- [ops/triton/rope.py](/home/cmq/code/vllm-ascend/vllm_ascend/ops/triton/rope.py)
- [worker/model_runner_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)
- [attention/dsa_v1.py](/home/cmq/code/vllm-ascend/vllm_ascend/attention/dsa_v1.py)
