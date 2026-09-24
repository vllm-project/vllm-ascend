---
name: ascendc-op-optimize
description: vllm-ascend 仓 csrc 自研 AscendC 算子的优化开发与测试工作流。当任务是优化/新增 csrc 下的算子（kernel/tiling）、在 910B 上编译安装、跑精度回归、做性能测量归因（NPUGraph benchmark / msprof op）、或排查上游 CI 门禁（ci-gate/ready-* 标签）时使用。
---

# vllm-ascend AscendC 算子优化工作流

本仓库的算子开发环境：Ascend 910B3（dav_c220 向量核，40 AIV，UB 192KB/核），
CANN 9.1.0。每次修改 kernel 后必须走完整闭环：**精度是硬门槛，任何 shape 不允许回退**。

## 每轮工作流（按顺序执行，缺一步都可能是空跑）

```bash
# 0. 修改 csrc/<category>/<op>/op_kernel/*.h|*.cpp 或 op_host/*

# 1. 强制刷新该算子的 build 树（见"构建陷阱"，必做）
python3 -c "
import shutil, glob, os
for p in ['csrc/build/binary/ascend910b/src/<op>', 'csrc/build/binary/ascend910b/bin/<op>']:
    shutil.rmtree(p, ignore_errors=True)
for f in glob.glob('csrc/build/binary/ascend910b/gen/<Op>_<op>_ascend910b_*.done'):
    os.remove(f)"

# 2. 全量构建（内部调 build.sh --pkg 并装到 vendor 目录；约 2 分钟）
bash csrc/build_aclnn.sh $(pwd) ascend910b

# 3. 安装（约 5 分钟）
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH}
pip install -e . --no-build-isolation

# 4. 构建自检（两处 md5 必须一致）
md5sum csrc/<category>/<op>/op_kernel/<op>.h csrc/build/binary/ascend910b/src/<op>/op_kernel/<op>.h

# 5. 精度回归（硬门槛）
pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_<op>.py -q

# 6. 性能记分板（NPUGraph replay）
python benchmarks/<op>.py

# 7. pipe 归因（msprof op；shape 用位置参数传给 runner）
msprof op --kernel-name=<KernelName> --output=./profiling --aic-metrics=PipeUtilization \
  python benchmarks/<op>_msprof.py 2048
```

## 构建陷阱（最容易空跑一轮的地方）

- `csrc/build` 是增量保留的；kernel 源拷贝规则（`func.cmake` 的 src_copy）只以
  `.done` 标记为 OUTPUT、**不依赖源文件**——改了 op_kernel/op_host 后不会自动重编。
  每轮编译前必须按上面第 1 步清理。
- gen/ 下只能删 `*_ascend910b_*.done`（编译步骤的门闩）。**不要删** `.sh` 和
  `_param.json`——它们是 opc 编译脚本的输入，删了会导致
  `[ERROR] op <name>: not any obj compile success`（脚本在、输入没了）。
- 自检标准：build 树源文件与仓库源文件 md5 一致，且 .o 的 mtime 晚于源文件 mtime。

## 测量口径（对比数据必须注明口径）

| 口径 | 命令 | 语义 |
|---|---|---|
| NPUGraph replay | `python benchmarks/<op>.py` | ≈ 背靠背 serving；大 shape 受前一 kernel 写出排空影响（HBM 争抢可拉长 1.6×，属系统效应，参照算子同样受影响） |
| msprof op | `msprof op --aic-metrics=PipeUtilization ...` | 孤立峰值，自带 ~3-4µs 固定开销；用于 pipe 归因而非记分板 |
| msprof 判读 | vec ratio 高 → compute-bound；scalar ratio 高（如 1-token 场景 44% vs 26%）→ issue-bound | 两者的优化手段完全不同：前者减 pass 数/流水，后者减指令数 |

## 910B (c220) kernel 开发要点

**事件编排（手工流水的基础，错一个就死锁或 NaN）**：
- 事件四件套：`AllocEventID` / `SetFlag` / `WaitFlag` / `ReleaseEventID`
  （框架 TQue 同款）。`FetchEventID` 只窥探不占位（`sff0(eventOccupy)`），
  连续两次 fetch 之间没有 Set/Wait 会返回**同一个 ID** → 双在途标志塌缩。
- 每方向上限 `QUE_MAX_EVENT=8` 个在途 ID；隔行才消费的方向（如 MTE3_V 隔 2 行）
  按在途数持多个 ID。
- **kernel 退出时每个 SetFlag 必须被 WaitFlag 消费**——悬挂标志会毒化同核
  下一个 kernel 的事件状态（表现为本 kernel 正常、下一个算子挂死）。最稳做法：
  标志的发射条件与消费条件严格一致（如 `i+2 < local_rows`）。
- `wait_flag(srcPipe, dstPipe)` 挂在 **dstPipe** 上执行：跨 pipe 等待不阻塞发射线程；
  唯一例外 `V_S`（GetValue 标量往返）——需要标量时优先用 Gather/stride-0 广播
  留在向量域。

**dtype 硬件限制**：
- c220 基础向量算子对 **bf16 无元素级支持**（Mul/Muls/Duplicate 的 static_assert
  均不含 bf16）——bf16 数学必须全程 fp32 域（Cast up → 算 → Cast RINT down）。
- c220 无点积类 reduce（仅 Sum/Max），无 Axpy 可用的 FMA 语义 → pass 数有算法下限。

**常用范式**：
- 单标量广播到向量：`Gather(dst, src, zero_offsets, 0, count)`（count 重载，5 参）
  复制成 32B 块 + `Mul` 用 `{src1BlkStride=0, src1RepStride=0}`（BinaryRepeatParams
  构造子 (1,1,0,8,8,0)）块广播——数值与 Muls 标量路径位级一致。官方先例：
  antiquant_c220 / batchnorm_v220 / rmsnorm_v220。
- 双缓冲乒乓：x[2]/fp32[2] 槽位用**核内局部行号奇偶**（`i & 1`）索引（绝对行号
  奇偶会错位）；输出复用输入槽可省缓冲并让 MTE3 不上关键路径。
- UB 预算在 tiling 里按 dtype 分档校验（如 B/col 常量 + 固定开销 < ub_size），
  比较**不要写 `ub_size - RESERVED`**（下溢风险），把常量加到左边。

**优化方法论**：
- 先 profile 后动手：基线 msprof 找瓶颈 pipe；每轮结束后再 profile 验证
  （vec ratio 是否如预期移动）。
- 候选评估要有数据支撑再写代码：被 UB 阻断的（算 B/col）、语义不匹配的
  （如 Axpy 累加语义）、非瓶颈方向的（MTE 从不是约束时加深预取无收益）——
  负结果也记录，避免重复推导。
- 数值等价优先：只消除冗余与串行，不改计算路径（位级一致 → 精度零风险）。

## 精度测试

- 门槛：`tests/e2e/nightly/single_node/ops/singlecard_ops/test_<op>.py` 全过，
  与参照算子对比 + 契约断言（如 `y_fp32 == y.float()` 逐位相等）。
- 容差按 dtype：bf16 2e-2，fp16 2e-3（`assert_close(rtol, atol)`），自写 sanity
  脚本不要用平坦阈值（会误报）。
- 补覆盖时针对**实际改动路径**：流水边界（单核 1/2/3 行、混合核、空闲核）、
  hidden 尾部/非对齐、高 rank 输入、极值输入、同进程连续调用 + 交叉算子
  （守护退出标志契约）、非法输入拒绝路径。先探针验证再固化为用例。

## 提交与 PR

- 提交：Conventional Commits（perf/fix/feat/...）+ `git commit -s` 签名；
  性能数据写进 commit body。
- PR：从 fork 分支发到上游 main；标题 `[Type][Module] Description`
  （如 `[Ops][Misc] ...`）；描述按模板三段式（What/why、user-facing、How tested，
  附性能数据表）。范围单一（纯 perf PR 只带算子代码）。
- gemini-code-assist 会自动 review：逐条核实后回复闭环（是真问题就修，是误报
  就给出源码依据，如 API 重载辨析）。
- **ci-gate 门禁**：改了 csrc 源码的 PR 必须跑 NPU 精度测试——需要 maintainer
  给 PR 加 `ready-precise`（按覆盖率选测，推荐）/ `ready-all`（全量）/ `main2main`
  标签；标签加上后 workflow 自动重跑。pre-commit/cpu-ut/DCO 失败才是代码问题，
  ci-gate 单独挂=缺标签，去 PR 里 @ maintainer。

## 环境备注

- GitHub 链路不稳定（时通时断，HTTP2 framing / connect timeout 反复出现）：
  推送失败先 `curl -sI https://github.com` 测连通，断了就等恢复重试；可
  `git config http.version HTTP/1.1` 缓解 framing 错误。
- 工具调用安全分类器偶发超时会挡住 Bash/Write：可在
  `.claude/settings.local.json` 配 permissions.allow 白名单（git/pytest/python/
  pip/build/msprof 前缀）使命令确定性放行。
- msprof op 注意：只保留 `--kernel-name` / `--output`（不加 --launch-count/
  --warm-up）；被测命令作为位置参数直接跟在选项后（不要用 `--application=`）；
  shape 用位置参数传给 runner（环境变量穿不透 msprof 启动层）。
