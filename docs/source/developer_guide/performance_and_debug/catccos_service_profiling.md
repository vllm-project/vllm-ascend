# CatCCOS 与 native MC2 服务内 A/B profiling

> **当前分支的选路说明**：本文件保留早期大 M CatCCOS 实验的 trace 分析流程。
> 从生产接入修改起，A5 CatCCOS 只替换原生 MC2；超过原生 MC2 容量的请求
> 即使调高 `CATCCOS_MAX_TOKENS_PER_RANK`，仍走原生 ALLGATHER/ALLTOALL。
> 新的无 profiler 服务 A/B 方法见 [AISBench 测试计划](catccos_aisbench_ab.md)。

## 1. 分支和实现范围

本 profiling 分支基于 CatCCOS A5 正式接入提交：

```text
0a9b67667 feat(moe): integrate CatCCOS A5 backend
```

分支名称：

```text
codex/megamoe-a5-vllm-v023-service-profiling
```

本轮配套的 CatCCOS 算子基线位于 CatCCOS 分支：

```text
codex/megamoe-vllm-kernel-reset (351af32)
```

该 CatCCOS 分支包含已解决连续 `M=1` launch 卡死的修改。每次 launch
仍然完整执行 metadata 清零、workspace full-clean、symmetric-A full-clean 和
rank barrier；profiling 只在这些操作外增加 Host user-scope，不改变同步、清零、
buffer 或 kernel launch 语义。

该分支不包含后续 CatCCOS 输入 dump 提交。它复用 vLLM 自带的
`--profiler-config`、`/start_profile` 和 `/stop_profile`，仅增加以下
shape-qualified profiler ranges：

```text
vllm_ascend.moe.catccos.host_moe_body[M=2,H=2048,topK=8]
└── vllm_ascend.moe.catccos.adapter_host_scope[...]
    ├── vllm_ascend.moe.catccos.pre_sync_host_wait[...]
    ├── vllm_ascend.moe.catccos.input_prepare_host[...]
    ├── vllm_ascend.moe.catccos.kernel_enqueue[...]
    └── vllm_ascend.moe.catccos.post_sync_host_wait[...]  # sync=true 时存在

vllm_ascend.moe.native_<实际通信路径>.host_moe_body[M=2,H=2048,topK=8]
└── vllm_ascend.moe.native_<实际通信路径>.pipeline_host_scope[...]
    ├── vllm_ascend.moe.native_mc2.dispatch_enqueue[...]
    ├── vllm_ascend.moe.native_mc2.mlp_enqueue[...]
    └── vllm_ascend.moe.native_mc2.combine_enqueue[...]

catccos.a5.binding.binding_host_scope[M=2,H=2048,N=1536,topK=8]
├── catccos.a5.binding.metadata_memset_host[...]
├── catccos.a5.binding.rank_barrier_host[...]
├── catccos.a5.binding.workspace_full_clean_host[...]
├── catccos.a5.binding.symmetric_a_full_clean_host[...]
├── catccos.a5.binding.dynamic_tiling_host[...]
└── catccos.a5.binding.kernel_launch_host[...]
```

`kernel_launch_host` 只统计异步 launch 的 Host 调用；融合 kernel 的 Device
duration 仍然从 `kernel_details.csv` 或 `trace_view.json` 的 NPU 行读取。

环境变量不开时，这些 profiler ranges 不会建立：

```bash
export VLLM_ASCEND_MOE_PROFILE_RANGES=1
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=0
```

默认的 `host_moe_body` 从两条路径共同的 `fused_experts` 调用点开始和结束，
代码边界一致，但它是 Host range。CatCCOS 内有显式同步，而 native 默认是异步
launch，因此两个 `host_moe_body` 的 duration 不能直接相减。

本次脚本中的真实路径是：

| 轮次 | `moe_comm_type` | `fused_experts` 内部实现 |
| --- | --- | --- |
| CatCCOS | `FUSED_MC2` | `apply_catccos` → `ascend950_dispatch_ffn_combine` |
| native default decode | 通常为 `MC2` | `token_dispatch` → `_apply_mlp` → `token_combine` |
| native default prefill | 由 token capacity 决定 | 可能为 `ALLGATHER` 或 `ALLTOALL` |

原始 native 脚本不传 additional config，而 CatCCOS 脚本设置了
`enable_prefill_mc2=true`。该配置参与计算 MC2 token capacity，因此原始两轮是
“CatCCOS 整套服务配置 vs native 默认服务配置”的端到端对照。它们的 decode
通常会形成 CatCCOS vs native MC2 对照；prefill 不保证走同一通信路径，必须以
trace 中的 `native_mc2/native_allgather/native_alltoall` 名称为准。

两边共同的 `fused_experts` range 对应同一个 routed-expert 替换接口，输入均为
router 选出的 `hidden_states/topk_ids/topk_weights` 和本层专家权重，输出均为
合并后的 routed expert hidden states。它们在语义上对应，但内部工作不是逐行等价：

- CatCCOS 包含 adapter 校验、初始化、显式 pre-sync、输入 dtype/layout 转换、
  融合算子 launch，以及配置开启时的 post-sync。
- native MC2 包含 dispatch 元数据构造与 launch、专家 MLP launch、combine
  元数据构造与 launch。
- 两边都不包含上游的 MC2 prepare/padding、router top-k，也不包含下游
  finalize；`build_fused_experts_input` 也在共同 range 之外。

因此要在相同通信路径下比较“服务采用这套后端付出的完整 routed-expert
成本”，看 CatCCOS 与 `native_mc2` 的 `synchronized_moe_body`；要比较
“CatCCOS 融合 Device kernel 本身”，则在
NPU timeline 中将它与 native dispatch、MLP、combine 的 Device critical-path
span 对齐。native 各 Device kernel 可能重叠，不能直接把 CSV 中每个 kernel 的
duration 无条件相加。

需要直接比较隔离后的 routed-expert 墙钟耗时时，再单独运行一轮：

```bash
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=1
```

该模式在进入共同 range 前执行一次 `torch.npu.synchronize()`，清空此前工作；
在 range 内的 MoE 调用后再次同步，再结束 range。range 名称变为
`synchronized_moe_body`。两边的开始和结束 Device 状态相同，因此该 range
才能直接比较。额外同步会改变 native 的正常 overlap，只用于诊断，不能用它
产生正式服务吞吐、TTFT 或 TPOT 数据。

旧版范围名中的 `M` 需要结合范围位置解释：`host_moe_body`、`prepare` 和
`native_allgather` 的 pipeline 输入是全局 token；CatCCOS binding、native MC2
或 ALLTOALL pipeline 输入是 TP 切分后的 rank-local token。本分支新增的
`global_M`/`rank_M` 会同时写出两者。在 TP4 下，切分路径的对应关系是：

```text
全局 decode T=8     → rank-local M=2
全局 prefill T=552  → rank-local M=138
全局 prefill T=2048 → rank-local M=512
```

## 2. 在 A5 测试机器拉取分支

```bash
cd /home/z00956592/vllm-ascend-catccos

git fetch git@github.com:WanhaoZhang/vllm-ascend.git \
  codex/megamoe-a5-vllm-v023-service-profiling

git switch -C codex/megamoe-a5-vllm-v023-service-profiling FETCH_HEAD

git rev-parse HEAD
```

同时拉取并重新编译带细分打点的 CatCCOS：

```bash
cd /home/z00956592/catccos

git fetch git@github.com:WanhaoZhang/CATCCOS.git \
  codex/megamoe-vllm-kernel-reset
git switch -C codex/megamoe-vllm-kernel-reset FETCH_HEAD

bash examples/ascend950_dispatch_ffn_combine/scripts/build_python.sh
git rev-parse HEAD
sha256sum build_torch_a5/lib/libcatccos_torch.so
```

构建后必须先执行已有的连续 launch/正确性测试。full-clean 是 repeat-launch
正确性保护，不能为了 profile 直接删除。

如果使用 editable install 或者启动脚本已经设置：

```bash
export PYTHONPATH=/home/z00956592/vllm-ascend-catccos:${PYTHONPATH:-}
```

切换分支后重启服务即可，不需要重新编译 CatCCOS 动态库。

## 3. 修改 CatCCOS 和 native 启动脚本

仓库已提供严格对齐启动脚本：

```text
tools/catccos_profiling/start_aligned_service.sh
```

它强制两边使用 `OMP_NUM_THREADS=1`、`enable_prefill_mc2=true`、相同 TP/EP、
batch capacity 和 profiler 配置。默认关闭 `SYNC_BOUNDARIES` 与
`catccos_sync_after_launch`，以保留真实异步流水。CatCCOS 轮：

```bash
cd /home/z00956592/vllm-ascend-catccos

PROFILE_TAG=catccos_aligned_m2_r1 \
PROFILE_ITERS=20 \
CATCCOS_ROOT=/home/z00956592/catccos \
bash tools/catccos_profiling/start_aligned_service.sh catccos
```

停止服务后启动 native 轮：

```bash
PROFILE_TAG=native_aligned_m2_r1 \
PROFILE_ITERS=20 \
bash tools/catccos_profiling/start_aligned_service.sh native
```

每个 `PROFILE_TAG` 必须唯一；脚本发现目录已存在时会退出，避免覆盖原始数据。
只有诊断同步完成边界时，才另起一轮显式设置 `SYNC_BOUNDARIES=1`；需要检查
CatCCOS launch 后等待时，再设置 `CATCCOS_SYNC_AFTER_LAUNCH=true`。这两类 trace
不能和默认异步 trace 混合计算吞吐。正式吞吐、TTFT 和 TPOT 必须关闭 profiler。

`CATCCOS_MAX_TOKENS_PER_RANK` 默认 512，`MAX_NUM_BATCHED_TOKENS` 默认 4096。
CatCCOS 的实际替换上限是其配置容量与原生 MC2 容量的较小值；在 TP4 下
原生 MC2 最多为 global M=2048。rank-local M=1024/2048/4096 的新实验
会走原生回退路径，早期 trace 中的 CatCCOS 大 M 样本仅用于历史分析。查看 trace 中的
`vllm_ascend.moe.<backend>.prepare/finalize[global_M=...,rank_M=...]`，按实际
backend 配对样本；finalize 内的 TP all_gather 用于还原 CatCCOS 的 token 切片。

接入侧验收按以下顺序进行，保持 CatCCOS/native 的 TP4、EP4、模型、调度器
budget 和请求输入相同：

1. 预热并检查 decode、global M 非 4 的整倍数、rank-local M=512/1024/2048/4096。
   确认两侧输出形状与顺序一致，padding 未进入有效输出，各 rank 均选到同一 backend。
2. 对 CatCCOS 的 FUSED_MC2 样本检查 `prepare` 切成 rank-local M，`finalize` 中
   一次 TP all_gather 恢复 global M，`outer_reduce_check` 内没有额外 TP all-reduce。
   高于原生 MC2 每 rank 512 的样本应在两侧走相同的原生回退路径。
3. 对照逐层输出误差、GSM8K 200 题准确率与整网吞吐；导出每档
   prepare、kernel、finalize 的 host/device 耗时及 gather/reduce 次数。
   未满足正确性和选路检查的样本不用于 kernel 性能归因。

以下手工配置用于理解脚本内容或适配已有启动脚本。

在原来的 `/home/z00956592/start_m1test.sh` 和
`/home/z00956592/start_m1test_native.sh` 中，在 `vllm serve` 之前加入：

```bash
export MSMONITOR_USE_DAEMON=0
export VLLM_ASCEND_MOE_PROFILE_RANGES=1
export VLLM_ASCEND_MOE_PROFILE_SYNC_BOUNDARIES=0

PROFILE_TAG="${PROFILE_TAG:?set PROFILE_TAG}"
PROFILE_ITERS="${PROFILE_ITERS:-20}"
PROFILE_DIR="/home/z00956592/profiles/${PROFILE_TAG}"

rm -rf "$PROFILE_DIR"
mkdir -p "$PROFILE_DIR"

PROFILER_CONFIG="{
  \"profiler\":\"torch\",
  \"torch_profiler_dir\":\"${PROFILE_DIR}\",
  \"torch_profiler_with_stack\":false,
  \"torch_profiler_with_memory\":false,
  \"ignore_frontend\":true,
  \"max_iterations\":${PROFILE_ITERS}
}"
```

在两个脚本的 `vllm serve` 参数中都加入：

```bash
--profiler-config "$PROFILER_CONFIG"
```

CatCCOS 启动脚本继续保留原有的：

```bash
--additional-config "$ADDCONF"
```

native 启动脚本仍然不传 `--additional-config`。因此 native 对照走的是：

```text
npu_moe_distribute_dispatch[_v2]
→ GMM1/SwiGLU
→ GMM2
→ npu_moe_distribute_combine[_v2]
```

decode 通常走以上 MC2 路径；prefill 可能因 MC2 capacity 回退到 ALLGATHER 或
ALLTOALL。它不是 CANN fused `_C_ascend.dispatch_ffn_combine`。

若本轮目的是做 CatCCOS/native MC2 的严格同路径 profiling，native 启动脚本
应显式传入：

```bash
NATIVE_ADDCONF='{
  "enable_fused_mc2":0,
  "fused_mc2_backend":"auto",
  "enable_prefill_mc2":true
}'
```

并在 `vllm serve` 中加入：

```bash
--additional-config "$NATIVE_ADDCONF"
```

这会关闭 CANN fused MC2，同时让 native 使用与 CatCCOS 相同的 prefill MC2
capacity。若 trace 中同一个 `M` 仍显示 `native_allgather` 或
`native_alltoall`，该样本不能与 CatCCOS 的 `synchronized_moe_body` 直接配对。

服务启动后先正常预热。只有调用 `/start_profile` 后才开始采集。不要在
profiling 同一轮开启输入 dump，否则 device-to-host 拷贝会污染时间线。

## 4. 使用原 AISBench 参数抓 CatCCOS

按照原复测报告的参数启动 CatCCOS 服务：

```bash
docker exec -d megamoe-vllm bash -lc '
  PROFILE_TAG=catccos_load_r1 PROFILE_ITERS=20 \
  bash /home/z00956592/start_m1test.sh \
  > /home/z00956592/serve_catccos_profile.log 2>&1
'
```

等待服务 `/health` 正常，然后后台启动原来的并发 8、200 题 perf 压测：

```bash
docker exec megamoe-vllm bash -lc '
  bash /home/z00956592/ab_test/run_aisbench_ab.sh \
    catccos perf 28001 200 load \
  > /home/z00956592/catccos_profile_load.log 2>&1 &
'
```

等待运行请求数达到 8：

```bash
docker exec megamoe-vllm bash -lc '
while true; do
  running=$(curl -s http://127.0.0.1:28001/metrics |
    awk "/^vllm:num_requests_running/ {sum += \$NF}
         END {print int(sum)}")
  echo "running=${running}"
  [ "$running" -ge 8 ] && break
  sleep 0.2
done
'
```

开始采集：

```bash
docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/start_profile
```

`max_iterations=20` 会把 worker 采集限制为 20 个模型 iteration。等待数秒
后调用 stop，确保已经结束并落盘：

```bash
sleep 5

docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

结果目录：

```text
/home/z00956592/profiles/catccos_load_r1
```

## 5. 使用相同请求抓 native

先停止 CatCCOS 服务并清理残留：

```bash
docker exec megamoe-vllm bash /home/z00956592/kill_vllm.sh
```

启动 native 服务：

```bash
docker exec -d megamoe-vllm bash -lc '
  PROFILE_TAG=native_load_r1 PROFILE_ITERS=20 \
  bash /home/z00956592/start_m1test_native.sh \
  > /home/z00956592/serve_native_profile.log 2>&1
'
```

发送相同的并发 8、200 题请求：

```bash
docker exec megamoe-vllm bash -lc '
  bash /home/z00956592/ab_test/run_aisbench_ab.sh \
    baseline perf 28001 200 load \
  > /home/z00956592/native_profile_load.log 2>&1 &
'
```

同样等 `vllm:num_requests_running` 达到 8，然后执行：

```bash
docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/start_profile

sleep 5

docker exec megamoe-vllm \
  curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

结果目录：

```text
/home/z00956592/profiles/native_load_r1
```

## 6. 受控 decode 和 prefill profile

原 AISBench 请求可以得到真实负载下的混合时间线。profiler range 名称中的
`M` 可以区分纯 decode 与 prefill/decode 混批：

- `M=2`：TP4、并发 8 下的纯 decode。
- `M>2`：prefill 或 prefill/decode 混批。

如果需要干净的纯 decode trace，可以启动恰好 8 个长输出请求并设置
`ignore_eos=true`。等首 token 已经返回、8 个请求全部进入 decode 后，再调用
`/start_profile`，用 `max_iterations=10` 抓 10 个 iteration。

如果需要受控 prefill，重启服务并设置：

```text
PROFILE_ITERS=1
```

在服务空闲时先调用 `/start_profile`，再发送一个精确 552-token prompt：

```bash
curl -fsS -X POST http://127.0.0.1:28001/start_profile
python3 /home/z00956592/bigm_perf_test.py 1 552 1 1
curl -fsS -X POST http://127.0.0.1:28001/stop_profile
```

该请求在 TP4 下应显示 `M=138`。使用 2048-token prompt 可以测默认容量边界
`M=512`。测试 4096-token prefill 时，若 budget 为 4096，当前生产选路在两侧
均回退到原生 ALLGATHER；调高 CatCCOS 容量也不会改变这一点。

## 7. 解析 profiling 文件

每个 worker 会生成独立的 `*_ascend_pt` 目录。容器内执行：

```python
from pathlib import Path

from torch_npu.profiler.profiler import analyse

for tag in ("catccos_load_r1", "native_load_r1"):
    root = Path("/home/z00956592/profiles") / tag
    for profile in root.rglob("*_ascend_pt"):
        print("analyse:", profile)
        analyse(str(profile))
```

主要查看：

```text
ASCEND_PROFILER_OUTPUT/trace_view.json
ASCEND_PROFILER_OUTPUT/operator_details.csv
ASCEND_PROFILER_OUTPUT/kernel_details.csv
ASCEND_PROFILER_OUTPUT/op_statistic.csv
ASCEND_PROFILER_OUTPUT/step_trace_time.csv
```

`trace_view.json` 可以使用 MindStudio Insight 打开。

仓库内可以直接执行：

```bash
python tools/catccos_profiling/analyse_profiles.py \
  /home/z00956592/profiles/catccos_aligned_m2_r1 \
  /home/z00956592/profiles/native_aligned_m2_r1

python tools/catccos_profiling/extract_ranges.py \
  /home/z00956592/profiles/catccos_aligned_m2_r1 \
  /home/z00956592/profiles/native_aligned_m2_r1 \
  --output /home/z00956592/profiles/aligned_m2_ranges.csv
```

`extract_ranges.py` 同时提取 `vllm_ascend.moe.*` 和
`catccos.a5.binding.*`，完整保留 `M/H/N/topK`，按每个 rank 的
`operator_details.csv` 输出 count、Host P50、P95 和最大值。

## 8. 大 M 分段统计与占比

本分支额外记录以下非重叠组件范围。范围名中同时保留全局 M 和 rank-local M：

```text
vllm_ascend.moe.catccos.prepare.padding
vllm_ascend.moe.catccos.prepare.tensor_split
vllm_ascend.moe.catccos.prepare.mc2_mask_split
vllm_ascend.moe.catccos.finalize.output_alloc
vllm_ascend.moe.catccos.finalize.tensor_split
vllm_ascend.moe.catccos.finalize.tp_all_gather
vllm_ascend.moe.catccos.finalize.unpad
vllm_ascend.moe.native_allgather.outer_reduce.tp_all_reduce
vllm_ascend.moe.native_allgather.outer_reduce.skip
```

例如：

```text
vllm_ascend.moe.catccos.finalize.tp_all_gather[
  global_M=4096,rank_M=1024,H=2048,topK=8]
```

在容器内生成汇总表：

```bash
cd /home/z00956592/vllm-ascend-catccos

python3 tools/catccos_profiling/summarize_moe_stats.py \
  /home/z00956592/profiles/catccos_m1024 \
  --tp-size 4 \
  --output /home/z00956592/profiles/catccos_m1024/moe_stats.csv
```

同时比较 CatCCOS 和 native：

```bash
python3 tools/catccos_profiling/summarize_moe_stats.py \
  /home/z00956592/profiles/catccos_m1024 \
  /home/z00956592/profiles/native_m1024 \
  --tp-size 4 \
  --output /home/z00956592/profiles/m1024_moe_stats.csv
```

输出中的字段含义：

| 字段 | 含义 |
| --- | --- |
| `host_p50_us` / `host_p95_us` | Host 范围耗时 |
| `device_p50_us` / `device_p95_us` / `device_max_us` | Device 范围耗时 |
| `host_share_of_body_pct` | 该范围 P50 Host 时间占 `host_moe_body` 的比例 |
| `device_share_of_body_pct` | 该范围 P50 Device 时间占 MoE body 的比例 |

share 是单个范围相对 body 的占比，嵌套范围之间不能直接相加。判断大 M 根因时优先看：

1. `catccos.finalize.tp_all_gather`：TP gather 是否占主要时间；
2. `catccos.a5.binding.*`：CatCCOS kernel enqueue、动态 tiling 和 device kernel 时间；
3. `native_allgather.outer_reduce.tp_all_reduce`：native 的真实外层归约成本；
4. `global_M` 与 `rank_M`：确认比较的是相同的 rank-local M。

## 9. 同步诊断与正式吞吐的区别

CatCCOS 默认异步提交，`kernel_enqueue` 很快返回，后续 collective 可能承担等待时间。每个 M 建议额外抓一轮：

```bash
CATCCOS_SYNC_AFTER_LAUNCH=true \
SYNC_BOUNDARIES=1 \
PROFILE_TAG=catccos_m512_sync \
bash tools/catccos_profiling/start_aligned_service.sh catccos
```

该轮只用于判断 kernel wait 是否被归入 finalize；正式吞吐仍使用：

```bash
CATCCOS_SYNC_AFTER_LAUNCH=false
SYNC_BOUNDARIES=0
```

如果同步后 `finalize.tp_all_gather` 缩短而总时间不变，说明原 trace 中包含了
CatCCOS kernel 的隐式等待。如果同步后 CatCCOS body 仍明显慢，则继续检查
kernel tiling 和通信本身。

## 10. trace 中搜索的名称

CatCCOS：

```text
vllm_ascend.moe.catccos.host_moe_body
vllm_ascend.moe.catccos.synchronized_moe_body
vllm_ascend.moe.catccos.adapter_host_scope
vllm_ascend.moe.catccos.pre_sync_host_wait
vllm_ascend.moe.catccos.input_prepare_host
vllm_ascend.moe.catccos.kernel_enqueue
vllm_ascend.moe.catccos.post_sync_host_wait
catccos.a5.binding.binding_host_scope
catccos.a5.binding.metadata_memset_host
catccos.a5.binding.rank_barrier_host
catccos.a5.binding.workspace_full_clean_host
catccos.a5.binding.symmetric_a_full_clean_host
catccos.a5.binding.dynamic_tiling_host
catccos.a5.binding.kernel_launch_host
```

native：

```text
vllm_ascend.moe.native_mc2.host_moe_body
vllm_ascend.moe.native_mc2.synchronized_moe_body
vllm_ascend.moe.native_mc2.pipeline_host_scope
vllm_ascend.moe.native_mc2.dispatch_enqueue
vllm_ascend.moe.native_mc2.mlp_enqueue
vllm_ascend.moe.native_mc2.combine_enqueue
vllm_ascend.moe.native_allgather.host_moe_body
vllm_ascend.moe.native_alltoall.host_moe_body
```

完整名称会带真实输入 shape：

```text
vllm_ascend.moe.catccos.kernel_enqueue[M=2,H=2048,topK=8]
vllm_ascend.moe.native_mc2.dispatch_enqueue[M=2,H=2048,topK=8]
```

## 11. A/B 判读方式

同一 workload、同一 `M` 下比较：

1. 自然服务轮比较整个 worker iteration、Device critical-path span 和 NPU 空洞；
   不直接比较两个 `host_moe_body` duration。
2. 对齐边界轮只配对相同 `M` 的 CatCCOS 与 `native_mc2`
   `synchronized_moe_body`；出现 `native_allgather/native_alltoall` 时不配对。
3. CatCCOS `pre_sync_host_wait`、`post_sync_host_wait` 的 Host 等待时间。
4. CatCCOS 融合 kernel 与 native dispatch、MLP、combine 所关联的 Device
   Kernel 总时间。
5. NPU 时间线中的空闲区间和算子 launch 间隔。
6. 四个 rank 的最大耗时。集合通信路径由最慢 rank 决定，不能只看平均值。
7. 多个 iteration 的中位数和 P95，避免由单个 iteration 下结论。

若 CatCCOS kernel 本身明显长于 native 整条链，应继续使用真实输入进行 EP4
standalone replay，并在 CatCCOS Device Kernel 内增加 routing、MXFP8、GMM、
通信和 unpermute 的阶段时间戳。普通服务内 profile 只能看到融合 kernel 的总
Device 时间，无法自动拆开这些内部阶段。

profiling 会影响服务速度，因此 Output Token Throughput、TTFT 和 TPOT 的正式
A/B 数值仍以关闭 profiler 的原始 200 题复测为准。服务内 profiling 用来解释
性能差距来自 kernel、同步、launch、通信还是 rank 长尾。

## 12. 本轮应执行的收敛顺序

### 12.1 先复用已有 trace

先对已有 `catccos_load_r1` 和 `native_load_r1` 执行第 7 节两个工具。原始 trace
中的 range 名保留了真实 `M`，不要再使用将 `M` 统一替换成占位符的旧脚本。

对每个 rank 先确认：

1. `M=2` 是否同时存在 CatCCOS 和 `native_mc2` 样本。
2. native 的 816 个 MC2 range 和 144 个 fallback range 分别对应哪些 `M`。
3. CatCCOS 融合 kernel 在 `kernel_details.csv` 中的准确名称和 Device duration。
4. `trace_view.json` 中 native 从第一个 dispatch Device kernel 开始，到最后一个
   combine Device kernel 结束的 critical-path span。

已有自然服务轮可以显示真实流水，但不能直接相减 CatCCOS/native 的
`host_moe_body`，因为 CatCCOS 包含同步完成等待，而 native 主要是异步下发。

### 12.2 再跑严格对齐轮

使用第 3 节统一启动脚本，依次抓：

| 场景 | rank-local M | `PROFILE_ITERS` | 目标 |
| --- | ---: | ---: | --- |
| 稳态 decode，并发 8 | 2 | 10 或 20 | 固定调用、barrier 和小 M kernel |
| 552-token prefill | 138 | 1 | 中等 M 的清零与 kernel |
| 2048-token prefill | 512 | 1 | capacity 边界的大 M 行为 |

每个场景分别运行 CatCCOS 和 native。只配对 backend、M、H、topK、rank 都一致
的样本，并同时记录 P50、P95 和四个 rank 的最大值。

### 12.3 用新增细分 range 判断下一步

按下面顺序判读：

1. `metadata_memset_host`、两个 `full_clean_host` 明显随 M 增长：继续定位哪些
   workspace/symmetric buffer 区域会在写前读取，在保持连续 launch 正确性的前提
   下缩小清零范围。不得直接删除 full-clean。
2. `rank_barrier_host` 长且四卡差异大：定位 rank 到达时间和 barrier 前的长尾。
3. `dynamic_tiling_host` 稳定占用明显：缓存相同 shape 的 tiling 结果，再做 A/B。
4. CatCCOS Device kernel 长于 native critical-path span：使用已 dump 的相同真实
   输入做 EP4 standalone replay，继续拆 routing、MXFP8、通信、GMM 和 unpermute。
5. Device kernel 不慢但自然服务仍慢：逐项验证 pre-sync、post-sync 和 Host/Device
   overlap。每次只改一个同步点，并执行连续 launch 正确性压测。

最终结果至少保留下列字段，避免只看一个总时间：

```text
M, rank, Cat synchronized body, metadata memset, barrier,
workspace clean, symmetric clean, tiling, launch Host,
Cat Device kernel, native Device span
```
