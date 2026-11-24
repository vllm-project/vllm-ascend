# M3.1 BF16 LoRA 全语言模块验证

本轮从个人分支 `87f5f67bdda60b5bf6c6785909730d0f9bf9a43f` 扩展，保持配套 vLLM main
`c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15` 不变。原 Q/K/V/O 的 P0–P9 合同和历史结果保留，
本页的独立入口补充 indexer、MLP、专家、混批及并行验证。实现和限制见[验证报告](VALIDATION-full-bf16.md)。

基座 checkpoint 重载的独立冷启动对照另见 [重载验证](BASE_WEIGHT_RELOAD.md)。

## 验收口径

- 真实 NPU 低秩执行对独立 CPU BF16 存储边界参考，`atol=rtol=1/64`；同时报告理想 FP64 差异。
- 磁盘 A/B、缩放后的实际 buffer、TP/EP 切片精确匹配；非目标片段精确不变；非零探针至少有十倍容差信号。
- base/A/B 的单独和混批请求，greedy token ID 完全一致；包含 prefill、decode 和真实同批探针。
- 独立 merged 模型关闭 LoRA，严格 token 对照独立记录。用户在 2026-11-24 确认：BF16 舍入导致的
  merged 差异单列，不改写原始 FAIL；按独立 BF16 参考和服务一致性验收后推送。

## 范围

| 目标 | 标准磁盘名称 | 实际执行与分布 |
| --- | --- | --- |
| Attention | `q_proj/k_proj/v_proj/o_proj` | Q 按 TP 分片；K/V 随 KV 头分片或复制；O 沿输入分片 |
| Indexer | `index_q_proj/index_k_proj` | 五段物理 QKV；index_q 随 KV 分片或复制；index_k 全量复制 |
| 普通 MLP | `mlp.gate_proj/up_proj/down_proj` | gate/up 合并列投影，down 行投影 |
| 共享专家 | `block_sparse_moe.shared_experts.gate_proj/up_proj/down_proj` | 普通线性 LoRA；区别于 shared-outer 格式 |
| 路由专家 | `block_sparse_moe.experts.<id>.w1/w2/w3` | 原 w13/w2 GMM 两处加入标准低秩增量；保留激活和路由 |
| 部分专家/投影 | 上表专家目标的子集 | 缺失整个目标表示零增量；缺 A/B 一半拒绝；无本地目标的 EP rank 使用基座 |

保持 router 权重和修正 bias 的原始 FP32 精度。仅验证主模型的语言侧；视觉/投影器、embedding、
LM head、router LoRA、MTP、DoRA、3D 混合格式不包含在通过声明中。
`fully_sharded_loras` 和 MoE `enable_moe_shared_loras` 在 M3 入口明确拒绝。
后者的共享因子专家轴为 1，与现有 NPU adapter×expert 展平索引不兼容。

DP 下 `--max-loras` 必须容纳**同一执行步所有 DP rank 的 adapter 并集**。
例如 A/B 分别在两个 DP 上执行，仍需要 `--max-loras 2`。超过并集容量或同 ID 对应不同 adapter 时明确报错。
本轮主闭环使用 eager；图模式、PP/PCP 需要另外验证。跨 DP 的路由专家 LoRA 明确拒绝 pin：
现有 pin 只约束本地 LRU，不能保证全局槽位重排后继续驻留。普通单 DP pin 沿用上游。

## 运行入口

在配套容器的 uv 管理环境中运行。以下路径由操作者指定；新目录不得已存在，以免覆盖证据。
脚本不会下载模型或训练 adapter；从实际运行模型生成标准低秩诊断 A/B/zero。

```bash
export CASE_ROOT=/path/to/new-case
export MODEL=/path/to/local-bf16-model
export PYTHON=/path/to/uv-venv/bin/python
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_BATCH_INVARIANT=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

本次容器 wheel 元数据为 `0.33.0`，配套源码为 main API，沿用环境既有的
`VLLM_VERSION=0.33.0.dev0` 覆盖以选择 main 兼容路径；其他环境应按真实配套设置。

### 模块、分片和数值参考

```bash
ASCEND_RT_VISIBLE_DEVICES=0 "$PYTHON" \
  tests/e2e/models/minimax_m31_lora/validate_bf16_profiles.py \
  --model "$MODEL" --directory "$CASE_ROOT/core" \
  --profiles attention indexer mlp shared experts all \
  expert_w1 expert_w2 expert_w3 expert-subset
```

追加 `--max-loras 2` 覆盖多槽 kernel；`--tp 2/4/8` 覆盖实际头分片和复制；
`--ep --tp 4 --profiles expert-subset` 覆盖无目标的 EP rank；对应设备列表由操作者分配。
追加 `--rank 16`、`--rank 128` 或 `--alpha 24` 创建独立资产和报告。
rank128 本轮仅签收 attention/indexer 代表项，不能推出所有模块和更大 rank 均已验证。

报告包含真实模块清单、每个 rank 的实际 buffer/primitive 比较、未活动片段、信号裕量和 FP64 诊断。
`SMOKE_PASS` 表示这些加载/数值项通过，不单独代表整模或服务验收。

### 独立 merged 对照

```bash
ASCEND_RT_VISIBLE_DEVICES=0 "$PYTHON" \
  tests/e2e/models/minimax_m31_lora/validate_bf16_merged.py \
  --model "$MODEL" --online "$CASE_ROOT/core" \
  --profile mlp --variant b --output "$CASE_ROOT/merged-mlp-b.json"
```

合并读取原始 checkpoint 和独立磁盘 A/B，计算
`Wmerged = BF16(FP32(W) + FP32(B_eff) @ FP32(A_eff))`，再交给常规 loader。
没有在线模型权重 reload、LoRA 缓存或从在线输出反推参考。报告验证 `enable_lora=False`。
严格 token 不一致继续返回非零退出码，不以用户接受数值差异为由修改测试。

### 混批、槽位复用和 indexer 长上下文

```bash
ASCEND_RT_VISIBLE_DEVICES=0 "$PYTHON" \
  tests/e2e/models/minimax_m31_lora/validate_bf16_service.py \
  --model "$MODEL" --online "$CASE_ROOT/core" --output "$CASE_ROOT/service.json"

ASCEND_RT_VISIBLE_DEVICES=0 "$PYTHON" \
  tests/e2e/models/minimax_m31_lora/validate_bf16_service.py \
  --model "$MODEL" --online "$CASE_ROOT/core" --profile indexer \
  --indexer-long --output "$CASE_ROOT/indexer-long.json"
```

短请求使用 3/17/129 token prompt，生成 8 个 token。服务用例验证 base/A/B 同批、A/B 切换、
C 引发槽位复用后 A 恢复；长用例使用 4097 token，超过本次配置的 16×128 选块容量，
检查实际 query 和 top-k 选择变化及无 hook 控制。其他模型配置需确认该长度同样超过其选择容量。

### HTTP TP+DP+EP

配套 main 的普通 `LLM` 入口拒绝直接启动 DP>1，因此 DP 使用真实 HTTP server。
测试专用具名 RPC 与动态 LoRA 接口仅在 loopback 服务启用；没有开放 callable 反序列化。

```bash
VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" \
  tests/e2e/models/minimax_m31_lora/serve_bf16.py \
  --model "$MODEL" --adapters "$CASE_ROOT/core" --tp 2 --dp 2 --ep --port 8005
```

另一个终端运行客户端，C 来自前述服务用例生成的目录：

```bash
"$PYTHON" tests/e2e/models/minimax_m31_lora/validate_bf16_http.py \
  --port 8005 --output "$CASE_ROOT/http.json" \
  --third-adapter "$CASE_ROOT/service-c" --adapter-a "$CASE_ROOT/core/all-a"
```

9 个单独基线、3 轮各 12 个并发请求逐项比较原始 token ID，并要求真实 base/A/B 同批证据。
可选 C 验证额外 12 个 base/B/C 请求、A 恢复、API 卸载/重载 A。
使用新端口避免将另一个尚未退出的服务误认成待测服务；保存 JSON、日志和源码哈希。

### CPU 回归与仓库检查

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON" -m pytest -q \
  tests/ut/lora tests/ut/models/minimax_m3 tests/ut/ops/test_prepare_finalize.py \
  tests/ut/worker/test_model_runner_v2.py tests/ut/worker/test_v2_dummy_padding.py \
  tests/ut/worker/test_model_runner_v2_finegrained_tp.py \
  tests/ut/worker/test_minimax_m3_kv_cache.py tests/ut/test_minimax_m3_kv_block_size.py
bash format.sh ci
```

硬件测试为显式命令行入口，避免默认 CPU UT 或 CI 在没有本地权重时隐式占用多张卡。
CPU 回归按生产模块归属组织；没有为测试辅助代码增加镜像实现式 UT。
