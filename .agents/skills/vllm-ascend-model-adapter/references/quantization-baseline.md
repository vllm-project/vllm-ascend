# Quantization Baseline

本文档定义 **量化相关适配** 的当前能力基线，覆盖 fp8 checkpoint、KV quant、W8A8、C8 等高风险路径。

## 1. 这一层解决什么问题

量化层关注：

- checkpoint 是不是 fp8 / int8 / compressed-tensors / 其他量化格式；
- 这是“权重量化加载问题”还是“运行时 kernel/算子问题”；
- 当前 Ascend 后端已有哪条量化执行路径；
- 是否应该退回 bf16 加载或 fallback。

## 2. 当前能力基线

结合现有 skill 与仓库能力，当前高频基线包括：

- fp8-on-NPU 往往需要 load-time dequant 到 bf16；
- KV quant，尤其 C8 KV cache，有专门 attention 路径；
- W8A8 / compressed-tensors 存在 Ascend 侧量化实现，但不能假设所有模型格式都直接可用；
- dummy 不能替代 real-weight 验证，尤其在量化路径上。

## 3. 当前判断原则

### 3.1 先区分“checkpoint 量化”与“运行时量化”

要先确认：

- checkpoint 本身是什么格式；
- vLLM / vllm-ascend 现在是否已有对应 quant config / method；
- attention/KV cache 是否走了特殊量化分支；
- 失败发生在 load 阶段还是 first request 阶段。

### 3.2 fp8 优先走安全路径

如果是 fp8 checkpoint：

- 优先考虑 weight + scale pairing；
- 优先验证 load-time dequant / bf16 执行路径；
- 不要默认强推 fp8 runtime kernels。

### 3.3 KV quant 与普通 weight quant 分开判断

KV quant 往往直接影响 attention backend 路径，不能只在 linear / quant config 层看。

## 4. 典型输入证据

- `config.json` 中 quantization 相关字段
- checkpoint 中 `weight_scale_inv`、KV cache scale/offset 等键
- `quantization_config`
- `compressed_tensors` 配置
- load 阶段和 first request 阶段的错误栈

## 5. 固定输出模板

```markdown
## Quantization Gap Analysis

### 1. Current Capability
- Existing supported quant path:
- Existing safe fallback path:
- Existing KV quant / attention quant support:

### 2. Model Requirement
- Checkpoint quant format:
- Runtime quant expectations:
- KV/cache quant traits:
- Scale / shard / dequant requirements:

### 3. Gap
- Loader quant gap:
- Runtime kernel gap:
- KV quant gap:
- Unknowns to verify:

### 4. Adaptation Plan
- Fix location:
- Minimal quant handling change:
- Validation focus:
- Stop / escalate condition:
```

## 6. 最常见的适配动作

- fp8 -> bf16 dequant load；
- scale pairing 检查；
- TP / KV replication 下 norm/scale shard 修正；
- 走现有 C8 KV path；
- 若 quant kernel 不稳定，则退回更保守执行路径并报告限制。
