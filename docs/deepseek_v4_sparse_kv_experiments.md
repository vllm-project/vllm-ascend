# DeepSeek V4 Sparse KV LRU Mooncake Experiments

本文档用于在真实 Ascend 服务器上验证“DeepSeek V4 稀疏注意力 + Mooncake 后端 + NPU 本地小型 LRU KV 工作集”的可行性。

本仓库当前本地环境没有 Ascend/NPU/Mooncake 运行环境，因此这里准备的是服务器侧实验脚本、trace 钩子和离线分析工具。实验目标不是先改推理语义，而是先回答：相邻 decode step 的 DSA top-k 历史 token 是否足够稳定，足以支撑 2K/4K/8K 级别的本地 LRU 工作集。

## 实验产物

- `tools/dsv4_sparse_kv/run_dsv4_mooncake.sh`
  - 启动 `mooncake_master` 和 `vllm serve`。
  - 默认参数对齐用户已验证能跑通的 DeepSeek V4 + AscendStoreConnector 配置。
  - 支持 `DSV4_EXPERIMENT_MODE=baseline|eager_baseline|topk_trace`。
- `tools/dsv4_sparse_kv/send_dsv4_probe_requests.py`
  - 向 OpenAI-compatible `/v1/chat/completions` 发请求。
  - 支持 `smoke`、`shared-prefix`、`long-decode`、`mixed` 场景。
- `tools/dsv4_sparse_kv/analyze_dsv4_topk_lru.py`
  - 读取 `topk_trace.jsonl`，计算相邻 decode top-k overlap。
  - 离线模拟不同容量的 LRU 命中率。
- `tools/dsv4_sparse_kv/summarize_dsv4_experiment.py`
  - 汇总 vLLM 日志、Mooncake 日志、请求结果和 LRU 分析结果。

## 实验 E0：基线启动与 smoke

目的：确认当前代码分支仍能跑通用户已有的 DeepSeek V4 + Mooncake 流程。

终端 1：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e0_baseline
export MODEL_PATH=/data/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,12,13
bash tools/dsv4_sparse_kv/run_dsv4_mooncake.sh
```

终端 2：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e0_baseline
python tools/dsv4_sparse_kv/send_dsv4_probe_requests.py \
  --base-url http://127.0.0.1:8900 \
  --model dsv4 \
  --out-dir "$OUT_DIR" \
  --scenario smoke \
  --rounds 1 \
  --concurrency 1 \
  --max-tokens 64
python tools/dsv4_sparse_kv/summarize_dsv4_experiment.py --out-dir "$OUT_DIR"
```

通过标准：

- `probe_summary.json` 中 `failed=0`。
- `experiment_summary.md` 中 `startup_complete >= 1`。
- `run_dsv4.log` 没有致命 Traceback/OOM。

## 实验 E1：Mooncake / KV Pool 复用链路

目的：确认 AscendStoreConnector + Mooncake 的现有 KV Pool 路径在 shared-prefix 场景下可用，并收集是否存在 load/store/recompute 异常。

终端 1 保持 `DSV4_EXPERIMENT_MODE=baseline`，重新设置输出目录：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e1_kv_pool_shared_prefix
export MODEL_PATH=/data/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,12,13
bash tools/dsv4_sparse_kv/run_dsv4_mooncake.sh
```

终端 2：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e1_kv_pool_shared_prefix
python tools/dsv4_sparse_kv/send_dsv4_probe_requests.py \
  --base-url http://127.0.0.1:8900 \
  --model dsv4 \
  --out-dir "$OUT_DIR" \
  --scenario shared-prefix \
  --rounds 3 \
  --concurrency 1 \
  --max-tokens 128 \
  --prompt-repeat 80
python tools/dsv4_sparse_kv/summarize_dsv4_experiment.py --out-dir "$OUT_DIR"
```

观察点：

- `run_dsv4.log` 中是否出现 KV Pool load/store 相关日志。
- `experiment_summary.md` 中 `kv_load_failure` 是否为 0。
- 如果 shared-prefix 请求稳定，说明当前块级/prefix 级 Mooncake 链路可作为后续 token/mini-block 级方案的底座。

## 实验 E2：DSA top-k trace

目的：记录 DeepSeek V4 DSA decode 阶段每层 top-k 稀疏索引，验证相邻 decode step 重合度。

注意：top-k trace 会把 NPU tensor 拷到 CPU 并写 JSONL，会显著影响性能。这个实验只看局部性，不看吞吐。默认使用 eager，降低图捕获和动态文件写的干扰。

终端 1：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e2_topk_trace
export MODEL_PATH=/data/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,12,13
export DSV4_EXPERIMENT_MODE=topk_trace
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_MODEL_LEN=8192
export TOPK_TRACE_SAMPLE_ROWS=1
export TOPK_TRACE_MAX_ROWS=50000
bash tools/dsv4_sparse_kv/run_dsv4_mooncake.sh
```

终端 2：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e2_topk_trace
python tools/dsv4_sparse_kv/send_dsv4_probe_requests.py \
  --base-url http://127.0.0.1:8900 \
  --model dsv4 \
  --out-dir "$OUT_DIR" \
  --scenario long-decode \
  --rounds 1 \
  --concurrency 1 \
  --max-tokens 256 \
  --prompt-repeat 120
python tools/dsv4_sparse_kv/analyze_dsv4_topk_lru.py \
  --trace "$OUT_DIR/topk_trace.jsonl" \
  --out-dir "$OUT_DIR" \
  --phase decode \
  --capacities 512,1024,2048,4096,8192 \
  --compress-ratio 4
python tools/dsv4_sparse_kv/summarize_dsv4_experiment.py --out-dir "$OUT_DIR"
```

通过标准：

- `$OUT_DIR/topk_trace.jsonl` 非空。
- `topk_lru_summary.md` 中 `records` 和 `pairs` 足够大，至少覆盖数十个 decode step 的多层记录。
- 2K/4K/8K 稀疏容量的 LRU hit rate 有明显差异，用来判断本地窗口大小。

## 实验 E3：更贴近并发的 top-k 局部性

目的：在低并发下看 batch 中多请求是否会使 row-level trace 变复杂，以及 LRU 命中是否显著下降。

终端 1：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e3_topk_trace_concurrency2
export MODEL_PATH=/data/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,12,13
export DSV4_EXPERIMENT_MODE=topk_trace
export MAX_NUM_SEQS=4
export MAX_NUM_BATCHED_TOKENS=8192
export MAX_MODEL_LEN=8192
export TOPK_TRACE_SAMPLE_ROWS=4
export TOPK_TRACE_MAX_ROWS=100000
bash tools/dsv4_sparse_kv/run_dsv4_mooncake.sh
```

终端 2：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e3_topk_trace_concurrency2
python tools/dsv4_sparse_kv/send_dsv4_probe_requests.py \
  --base-url http://127.0.0.1:8900 \
  --model dsv4 \
  --out-dir "$OUT_DIR" \
  --scenario mixed \
  --rounds 2 \
  --concurrency 2 \
  --max-tokens 192 \
  --prompt-repeat 80
python tools/dsv4_sparse_kv/analyze_dsv4_topk_lru.py \
  --trace "$OUT_DIR/topk_trace.jsonl" \
  --out-dir "$OUT_DIR" \
  --phase decode \
  --capacities 512,1024,2048,4096,8192 \
  --compress-ratio 4
python tools/dsv4_sparse_kv/summarize_dsv4_experiment.py --out-dir "$OUT_DIR"
```

观察点：

- 并发后 overlap 是否仍高。
- row-level 分组是否足够稳定。当前 trace 还没有 request id，只能按 `pid/layer/row_idx` 近似分组；如果结果混乱，后续开发前需要把 request/slot 映射也打进 trace。

## 实验 E4：对照实验，关闭 KV Pool

目的：区分 Mooncake/KV Pool 本身造成的问题和 DSA top-k trace/LRU 假设造成的问题。

终端 1：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e4_no_kv_pool
export MODEL_PATH=/data/models/Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,12,13
export ENABLE_KV_POOL=0
export START_MOONCAKE=0
bash tools/dsv4_sparse_kv/run_dsv4_mooncake.sh
```

终端 2：

```bash
cd /vllm-workspace/vllm-ascend
export OUT_DIR=/vllm-workspace/mydata/dsv4_sparse_kv_exp/e4_no_kv_pool
python tools/dsv4_sparse_kv/send_dsv4_probe_requests.py \
  --base-url http://127.0.0.1:8900 \
  --model dsv4 \
  --out-dir "$OUT_DIR" \
  --scenario smoke \
  --rounds 1 \
  --concurrency 1 \
  --max-tokens 64
python tools/dsv4_sparse_kv/summarize_dsv4_experiment.py --out-dir "$OUT_DIR"
```

## 建议回传给 Codex 的文件

每个实验目录请优先回传：

- `experiment_summary.md`
- `experiment_summary.json`
- `probe_summary.json`
- `topk_lru_summary.md`，仅 E2/E3 有
- `topk_lru_summary.json`，仅 E2/E3 有
- `serve_command.txt`

如果失败，请额外回传：

- `run_dsv4.log` 中最后 300 行
- `run_mooncake.log` 中最后 200 行
- 完整 Python traceback 或 NPU/CANN 报错片段

如果 `topk_trace.jsonl` 不大，也可以回传；如果很大，先回传 `topk_lru_summary.json` 即可。

## 初步判定规则

- 如果 E2 的 2048/4096/8192 sparse-index 容量命中率都很低，说明“只靠小 LRU 承接 top-k KV”的收益可能不足，需要先研究预取或更大粒度。
- 如果 4096 或 8192 sparse-index 容量命中率很高，且 consecutive overlap 明显高，下一步可以进入 prototype。
- 如果 E3 并发后命中率大幅下降，后续实现必须把 request id、slot mapping 和 scheduler 层状态绑定好，不能只做 layer-global LRU。
- 如果 E1 的 KV Pool 日志大量失败或回退 recompute，需要先修 Mooncake/AscendStoreConnector 稳定性，再做稀疏 KV 细粒度加载。
