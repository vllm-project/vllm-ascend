# Gemma4-E4B-it Ascend NPU 适配报告

## 基本信息

| 项目 | 内容 |
|------|------|
| 模型 | `google/gemma-4-E4B-it` |
| checkpoint 路径 | `/home/cmq/cache/modelscope/models/google/gemma-4-E4B-it` |
| served-model-name | `gemma-4-E4B-it` |
| TP size | 4 |
| vLLM 源码 | `/home/cmq/code/vllm-cpu/vllm` |
| vllm-ascend 源码 | `/home/cmq/code/vllm-ascend` |
| 工作目录 | `/home/cmq/code/vllm-ascend/scripts` |
| 适配日期 | 2026-05-19 |

---

## 模型分析

### 架构结论

- 类型：VLM（`Gemma4ForConditionalGeneration`）
- 文本核心：Gemma4 decoder，混合 `sliding_attention` 与 `full_attention`
- 关键特征：
  - 全注意力层 `global_head_dim=512`
  - 支持 cross-layer KV sharing（`kv_sharing_target_layer_name`）
  - 理论最大上下文：`131072`
  - 非 MoE
  - 无 MTP checkpoint 信号
  - 无 remote code 依赖

### 适配风险点

本模型不是简单的纯文本 dense LLM，真正影响 Ascend 适配的是两点：

1. Gemma4 全注意力层的 head dim 为 512，而 Ascend FIA TND 路径对 `D=512` 有额外 RoPE 约束。
2. Gemma4 使用 cross-layer KV sharing，后半段共享层在 prefill 阶段不应重写本层 KV cache，而应读取目标层 cache。

---

## 适配过程

### 1. 初始分析

- 先完成环境检查、checkpoint 画像和上游代码排查。
- 确认上游 vLLM 已包含 Gemma4 的模型定义与 registry 注册。
- 确认问题主要落在 `vllm-ascend` attention/backend 路径，而不是缺少上游模型文件。

### 2. 启动阶段阻塞：MLA prefill patch 兼容性

最早的 dummy 启动先卡在 `vllm_ascend/patch/platform/patch_mla_prefill_backend.py` 对上游模块路径的硬依赖。当前使用的 vLLM 源码中不存在该模块路径，导致启动前 import 失败。

修复方式：

- 对缺失模块路径改为安全导入；
- 仅在符号存在时再应用 patch。

这一步解决的是**启动兼容性**，还没有进入 Gemma4 本身的算子问题。

### 3. 第一阶段真实算子问题：`D=512` 的 Ascend FIA 限制

dummy 权重启动后，首个请求在 `torch_npu.npu_fused_infer_attention_score` 崩溃，报错核心信息为：

```text
When D is 512, inputlayout TND q_rope and k_rope should not be null
```

按 skill 规则，对卡在 `torch_npu` 算子的问题先查询 HiAscend 官方文档。参考页面：

- HiAscend: `aclnnFusedInferAttentionScoreV3`
  https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha003/API/aolapi/context/aclnnFusedInferAttentionScoreV3.md

从官方文档得到的关键约束是：

- 当 `queryRope` / `keyRope` 为空时，TND 场景不支持 `Q_D=K_D=V_D=512`
- `D=512` 场景需要显式 RoPE 相关输入

而 Gemma4 全注意力层在进入 Ascend backend 前，RoPE 已经折叠进 `q/k`，无法直接满足该接口约束。

修复方式：

- 在 `vllm_ascend/attention/attention_v1.py` 中为 Gemma4 这类 `decoder + head_size=512 + TND` 路径增加 fallback；
- 避开 `npu_fused_infer_attention_score`，改走 `torch_npu.npu_fusion_attention`；
- 对 paged KV 场景先 materialize 成 dense KV，再以 TND 方式执行。

这一步修复后，dummy 权重的 text / multimodal smoke 都能通过，说明 **D=512 算子阻塞已解除**。

### 4. 第二阶段 false-ready：服务可用但真实权重输出乱码

在第一阶段修复后，真实权重服务可以启动，首个请求也不再 500，但输出是明显错误的乱码，典型样例如：

```text
"$}/**жете-a($\\mathrm"
"**D 성}\\of an a3"
```

这属于典型的 **false-ready**：

- 服务启动成功
- 请求也返回 200
- 但语义完全错误，不能算适配完成

进一步对照上游 Gemma4 和 vLLM KV sharing 语义后，定位到真正根因：

- Gemma4 的后半段共享层带有 `kv_sharing_target_layer_name`
- 这些层在 prefill 阶段应该读取目标层 KV cache
- 共享层本身不应再次把自己的 K/V 写回 cache

但当时 Ascend backend 没有接住这条语义：

1. 共享层仍会执行 `reshape_and_cache`
2. `PrefillNoCache` 路径仍直接使用当前层的 `key/value`

这会破坏 Gemma4 shared-KV full-attention 层的行为，最终表现为真实权重语义错误。

### 5. 最终修复：补齐 shared-KV 语义

在 `vllm_ascend/attention/attention_v1.py` 中补齐了共享 KV 层语义：

- 保存 `kv_sharing_target_layer_name`
- 增加 `uses_shared_kv_cache` 判断
- 共享层跳过 `reshape_and_cache`
- 共享层跳过 `do_kv_cache_update`
- 在 `PrefillNoCache` 下，若为共享层，则改为读取目标层 paged KV cache，而不是使用本层直传 `key/value`

修复后，真实权重文本和图文输出都恢复正常。

---

## 代码变更

| 文件 | 变更 | 目的 |
|------|------|------|
| `vllm_ascend/patch/platform/patch_mla_prefill_backend.py` | 安全导入 / 条件 patch | 解决启动阶段 import 路径不兼容 |
| `vllm_ascend/attention/attention_v1.py` | Gemma4 `D=512` TND fallback + shared-KV 语义修复 | 解决算子约束与真实权重乱码 |
| `tests/ut/attention/test_attention_v1.py` | 新增 shared-KV 回归用例 | 固化“不重写 cache / prefill 读取共享 cache” |

---

## 验证结果

### Stage A：dummy 权重

| 项目 | 结果 |
|------|------|
| Engine 启动 | ✅ |
| text smoke | ✅ |
| multimodal smoke | ✅ |
| 结论 | 证明 `D=512` 的 Ascend 算子阻塞已被绕过 |

### Stage B：真实权重（`--enforce-eager`）

服务命令：

```bash
export VLLM_VERSION=0.20.1
vllm serve /home/cmq/cache/modelscope/models/google/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --tensor-parallel-size 4 \
  --port 8002 \
  --max-model-len 4096 \
  --enforce-eager
```

验证结果：

| 输入 | 输出 | 结果 |
|------|------|------|
| `Answer with exactly one word: pong` | `Pong` | ✅ |
| `What is 2+2? Answer with exactly one token.` | `4` | ✅ |
| `Repeat exactly: hello` | `hello` | ✅ |
| 图像 `qwen.png` OCR | `The text in the image is "TONGYI Qwen".` | ✅ |

### Stage C：真实权重（默认图模式 / ACLGraph）

服务命令：

```bash
export VLLM_VERSION=0.20.1
vllm serve /home/cmq/cache/modelscope/models/google/gemma-4-E4B-it \
  --served-model-name gemma-4-E4B-it \
  --tensor-parallel-size 4 \
  --port 8003 \
  --max-model-len 4096
```

验证结果：

| 输入 | 输出 | 结果 |
|------|------|------|
| `Answer with exactly one word: pong` | `Pong` | ✅ |
| 图像 `qwen.png` OCR | `The text in the image is "TONGYI Qwen".` | ✅ |

补充观察：

- 默认图模式首次编译与 warmup 约 83.5s
- 其中 `torch.compile` 约 53.2s
- ACL graph capture 约 19s

---

## 特性矩阵

| 特性 | 状态 | 说明 |
|------|------|------|
| 文本推理 | ✅ | eager / 默认图模式都通过 |
| 多模态图片输入 | ✅ | eager / 默认图模式都通过 |
| ACLGraph | ✅ | 默认模式验证通过 |
| eager fallback | ✅ | 用于隔离图模式影响并最终通过 |
| EP | N/A | 非 MoE 模型 |
| flashcomm1 | N/A | 非 MoE 模型 |
| MTP | N/A | checkpoint 未体现 MTP 支持 |
| `128k + bs16` baseline | 未验证 | 本次优先完成功能适配与真实权重正确性修复 |

---

## Dummy 与真实权重的结论差异

| 阶段 | 能证明什么 | 不能证明什么 |
|------|------------|--------------|
| dummy 权重 | 启动链路、D=512 算子 fallback、首个请求路径可达 | 不能证明语义正确性 |
| 真实权重 | 文本语义正确、多模态路径正确、shared-KV 修复有效 | 才能作为最终签收依据 |

本次适配的关键经验是：**dummy 通过并不代表 Gemma4 已适配成功**。真正的 shared-KV 语义问题只会在真实权重下暴露为乱码输出。

---

## 已知说明

### 1. `VLLM_VERSION=0.20.1` 不影响当前运行时逻辑

按适配要求，运行时设置了：

```bash
export VLLM_VERSION=0.20.1
```

但当前 vLLM 运行日志会提示：

```text
Unknown vLLM environment variable detected: VLLM_VERSION
```

说明该变量在当前代码基线上不会被 vLLM 运行时消费，但不影响本次适配结果。

### 2. 当前实测最大长度仅覆盖到 4096

Gemma4 配置中的理论长度是 `131072`，但本次适配过程以“修通真实权重正确性”为主，最终验证口径固定在 `4096`。更大长度和 `128k + bs16` 容量基线建议在后续专门的容量/性能验证中补测。

---

## 结论

`google/gemma-4-E4B-it` 已在 Ascend NPU 上完成功能适配，关键问题共两类：

1. Ascend FIA TND 对 `D=512` 的接口约束
2. Gemma4 shared-KV 层语义在 Ascend backend 中缺失

修复后，真实权重在 eager 与默认图模式下都能稳定返回正确文本与图文结果，之前的启动失败和乱码问题均已消除。
