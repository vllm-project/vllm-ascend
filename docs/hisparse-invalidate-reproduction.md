# HiSparse written-slot invalidation：实现与复现记录
>
> 提交前复验（2026-11-02）：统一导入入口及 Ruff 整理后，A3 145 passed（164.16s），
> CPU 140 passed、8 个 NPU 用例 deselected（23.03s）。最终日志、XML、源码哈希及
> 检查边界见 [validation evidence](hisparse-validation/2026-11-02/README.md)。下文保留各阶段执行记录。
>
> 后续阶段（2026-11-02）：四算子、统一 CPU 入口和联合回归见
> [四算子复现记录](hisparse-ops-reproduction.md)。下文保留 invalidation 初次实现的
> 历史范围、结果与独立复现入口；后续记录不把 CPU 通过解释为 A3 通过。

## 范围与当前状态

本次只实现 `hisparse_invalidate_written_slots_kernel` 及其六参数 Python 入口，
不接入 HiSparse backend、resolver、gather 或模型。参考当前工作区的
`vllm/csrc/hisparse_kernels.cu:945-1060`，不使用 Ascend PR #19109。

2026-11-02 用户要求先完成代码、暂不占卡。因此本轮只进行静态检查和无 NPU
设备映射的 CPU Triton 解释器测试。**A3 编译、执行、图重放和 stream 顺序均未验证。**
下文 NPU 命令是后续获得设备释放确认后执行的步骤，不是已完成结果。

锁定源码：

| 仓库 | 分支 | 提交 |
| --- | --- | --- |
| vLLM | HiSparse-Ascend-opsWorkWithCodex | d5301a4f5487db1d7e8d87daf7828225b95b1774 |
| vLLM-Ascend | HiSparse-Ascend-opsWorkWithCodex | cae7ff0290dd6c6155f71d7223e0fd4aff95c9e9 |

## 实现及 CUDA 对照

生产文件：`vllm_ascend/ops/triton/hisparse.py`。

```python
hisparse_invalidate_written_slots(
    device_global_indices_ptrs, request_state_indices,
    req_id_per_token, written_slots, num_state_rows, region_stride,
)
```

- `device_global_indices_ptrs`：连续一维 int64 NPU Tensor，保存多组 int32 索引表的设备地址。
  被引用表必须是连续的 `[num_state_rows, region_stride]`；调用者保留表与地址 Tensor
  的生命周期，直至使用它们的流和图执行完成。地址值及被引用对象的 shape 无法通过
  指针 Tensor 的元数据校验。
- `request_state_indices`、`req_id_per_token`：连续一维 int32 NPU Tensor。
- `written_slots`：连续一维 int64 NPU Tensor，与 `req_id_per_token` 等长。
- 对有效 request，根据 `request_state_indices[req]` 找到 state 行，将该行内所有
  等于 written slot 的条目原地改为 `-1`。其他行、其他组和不匹配条目保持原值。
- request 越界、state 越界、负 slot、超过 `INT32_MAX` 的 slot 都不失效。
  先检查 int64 范围，再收窄，避免 `2**32 + 7` 被错误当成 slot 7。
- 零 token、零 group、零 state 或零 region 是空操作；所有输入在同一 NPU；
  wrapper 不读取设备标量、不搬运 KV、不做设备同步。调用者负责与其他表写入排序。

CUDA 使用一个 block 处理一个 `(written token, index group)`。
本实现将一个 `(group, state, 256 元素行分片)` 分配给一个 Triton program，
在分片内依次处理所有写入，再写回一次。这样同一行多 token 重写不会产生
并行向量读改写的覆盖风险，也覆盖多个 request ID 指向同一 state 的情况。
无效 request 使用标量分支保护间接读取，不依赖仅对读取结果生效的 mask。

初版优先保证语义：每个行分片扫描 token 列表，工作量随 state 行数和 token 数增长。
没有性能提升结论；后续性能优化须保留本次状态表回归测试。

## UT 复用与覆盖

测试文件：`tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py`。
当前 vLLM 提交没有单独的 `test_hisparse_kernels.py`；可复用的最小回归位于
`tests/v1/attention/test_sparse_mla_backends.py:3344` 的
`test_hisparse_invalidation_covers_every_index_group`。

移植保留原来的三组表、数据、request/state 重排、padding token 和断言：只向算子
传入前两组表，验证它们的指定行失效，第三组及其他行不变。调用改为直接构造真实
设备地址 Tensor 并调用六参数入口。原 vLLM 测试文件保持不变。
`test_hisparse_newest_write_and_recycled_slot_invalidation` 还依赖 resolver/gather
和完整 runtime，超出本次单 kernel 范围，没有宣称复用了完整链路。

新增测试覆盖：

- 行宽 1、7、8、255、256、257、2048；跨 tile 边界及非 16 字节对齐表地址。
- 同 state 多 slot、重复 token、两个 request 共用 state、无效 request/state。
- int64 极值、`INT32_MAX` 合法值及大于 `2**24` 的精确整数比较。
- 表前后保护值，验证子视图偏移和尾部 mask 不破坏相邻存储。
- 空输入、CPU Tensor 拒绝；NPU 模式额外检查 dtype、维度、连续性、长度和混合设备。
- NPU 模式额外验证更改 slot、request、state 和地址表内容后的图重放，以及非默认流
  的事件完成顺序；这些用例不调用其他 HiSparse 算子。

测试通过 `importlib` 加载真实生产源文件，避免 `ops/__init__.py` 提前导入其他 serving
算子；没有复制/改写生产函数，也没有 mock kernel。它验证独立算子，不验证 backend
注册和 serving 导入链。CPU 解释器路径直接启动同一个 kernel，绕过 NPU-only wrapper；
空维度中不启动 kernel 的分支及 NPU wrapper 的完整行为需要后续硬件测试。

## 复现方式：130 宿主机

```bash
ssh vllm-ascend-130
cd /mnt/share/h00954440/HiSparse-Ascend-opsWorkWithCodex/vllm-ascend
git status --short --branch
git rev-parse HEAD
bash tools/run_hisparse_invalidate_tests.sh cpu
```

脚本使用本机已有镜像的不可变 ID：

```text
sha256:8725ad72b9db273848f13894bad0e547a67567cccb2f8692b7e3588b2e959982
```

对应仓库 digest：
`harbor.computing.lab.rnd.huawei.com/hisecr/vllm-ascend@sha256:eb30579f3fff978b200c52560ded96dad03751854afe0e7110f02a0c43e474fb`。
本轮探测为 Python 3.12.13、torch 2.10.0+cpu、torch-npu 2.10.0.post7、
triton-ascend 3.6.0、pytest 8.4.2。`torch` 显示 `+cpu` 是该镜像的基础 wheel 信息，
不说明 NPU 扩展可用；CPU 模式禁用自动加载，NPU 模式需显式导入 `torch_npu` 并验证。

CPU 模式创建无设备映射、非 privileged、无网络的临时容器。仅创建继承镜像包的 venv，
不下载/安装依赖。容器退出即清理，同时注册 15 分钟兜底清理。
每次运行会打印独立的 `.artifacts/hisparse-invalidate-时间-cpu` 目录，记录镜像、源码
哈希、分支、环境版本、pytest 日志、JUnit XML 和清理结果。

### 后续 A3 测试：本轮未执行

先从设备使用者处确认释放，再检查占用。物理设备 8 是此前建议，不是自动分配结果；
按实际释放设备替换以下数字。不得停止其他人的服务来腾卡。

```bash
npu-smi info
# --device-released 表示操作者已确认该物理设备释放且立即复查占用。
bash tools/run_hisparse_invalidate_tests.sh npu 8 --device-released
```

脚本只映射指定 `/dev/davinciN` 和必要管理设备，设置
`ASCEND_RT_VISIBLE_DEVICES=N`，不使用 privileged；预期测试进程使用逻辑 NPU 0，
需核对输出的可见设备数和名称。脚本广播资源与 15 分钟上限，注册自动清理，并在
结束时保存 `npu-after.txt`。若自动清理注册失败，会立即移除本任务容器。
NPU 模式应收集全部用例，不能把 CPU 模式的 `-k 'not npu_'` 带入硬件验收。

容器内的实际测试入口为：

```bash
"$ARTIFACTS/venv/bin/python" -m pytest \
  --confcutdir=tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_hisparse_invalidate_written_slots.py \
  -vv --junitxml="$ARTIFACTS/results.xml"
```

`--confcutdir` 排除无关模型下载和 serving conftest。用例在没有可用 NPU 时应失败，
不会 skip/xfail 制造通过结果。图测试改变设备 Tensor 内容，不能用一次 capture 成功
替代 replay 正确性。NPU 模式的镜像/驱动/源码组合及编译器生成代码尚待验证，首次
运行可能暴露环境或 lowering 问题；保留原始错误再修复，不能降格为 CPU 通过。

## 本轮过程记录

原始日志目录：工作区根下 `.artifacts/2026-11-02-hisparse-invalidate`。

1. 核对两个仓库分支、HEAD、无既有源码修改，以及本地/远端 coding guide SHA-256
   一致。只读检查显示设备 0–7 有已有进程；用户随后明确要求暂不占卡。
2. 创建无设备映射的 CPU 容器并记录依赖版本。首次探测的嵌套 shell 引号有误，
   改用独立脚本后成功；该错误没有启动 NPU。
3. 先上传新测试、未提供实现，收集失败：`before-implementation.log`，退出码 2，
   错误为缺少 `vllm_ascend/ops/triton/hisparse.py`。这是缺失功能的收集基线，
   不是已存在 CUDA/NPU kernel 的错误复现。
4. 添加 Triton 实现，首次 CPU 解释器运行 `cpu-interpreter.log`：
   **15 passed、8 deselected**。98 条警告来自 Triton 解释器与 NumPy 的标量转换弃用提示。
5. 增加子视图保护值回归并通过统一复现脚本重跑，结果为 **18 passed、8 deselected**，
   全部 26 个用例可收集，没有 skip/xfail。日志保留于
   `.artifacts/hisparse-invalidate-20261102-093535-cpu`，包含 `pytest.log`、
   `results.xml`、`container-scope.txt`、`source-sha256.txt` 和 `cleanup.log`。
6. Python AST、Bash 语法及新增文件空白检查单独记录；基础离线镜像没有 Ruff，
   本机下载 Ruff wheel 未成功，未宣称 Ruff 或全量 `format.sh ci` 通过。
7. 最终格式修正与脚本收紧可见设备数检查后，再次通过统一脚本执行：
   `.artifacts/hisparse-invalidate-20261102-093814-cpu`，
   **18 passed、8 deselected、106 warnings，2.21 秒**。
   `container-scope.txt` 明确记录 `devices=[]`、`privileged=false`、`network=none`；
   `cleanup.log` 记录容器移除，宿主机复查没有本任务遗留容器。

最终生产实现 SHA-256：
`5a31c2bcdbc1898acd2943d0146c7b079f55c4f5049297207ec9a4230fee2aac`。
最终测试文件 SHA-256：
`fe75afeb209384a9275ffa621958e85a12dda677251b30af02b74960adee9b08`。
脚本与各次运行完整哈希见对应目录的 `source-sha256.txt`。

CPU 用例收集基线失败及最终解释器通过均可从上述日志复核；它们不替代 A3 测试。

所有通过结果只代表明确标注的验证层。没有 A3 硬件、backend 集成、模型精度或性能结论。
