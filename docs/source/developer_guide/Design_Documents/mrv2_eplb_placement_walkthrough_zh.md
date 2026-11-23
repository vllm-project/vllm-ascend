# MRv2 部署表导出与静态 EPLB：代码学习说明

这次实现包含两个功能：动态运行结束后导出已生效部署，以及在 MRv2
启动阶段安装该部署并固定运行。没有实现 Auto EPLB。

## 1. 先理解三个必须一致的对象

1. 物理槽位实际存放的专家权重，包括计算使用的量化 scale 等附属数据。
2. `physical_to_logical_map`：每个物理槽位存放哪个逻辑专家。
3. 路由表：router 选中逻辑专家后，可以使用哪些物理副本。

例如，4 个逻辑专家、2 个冗余专家、EP=2，每个 rank 有 3 个槽位：

```text
physical_to_logical = [0, 1, 2, 0, 1, 3]

rank 0: [专家0, 专家1, 专家2]
rank 1: [专家0, 专家1, 专家3]

逻辑专家0 -> 物理槽位[0, 3]
逻辑专家1 -> 物理槽位[1, 4]
逻辑专家2 -> 物理槽位[2]
逻辑专家3 -> 物理槽位[5]
```

只修改映射会让 router 把 token 发给保存错误权重的位置；只移动权重
又会让旧路由继续把 token 发到原位置。代码必须把两者作为同一次部署提交。

## 2. 部署文件为什么保存物理到逻辑的映射

入口是 `vllm_ascend/distributed/eplb/placement.py`。

- `EplbPlacement` 定义文件中的部署和元信息。
- `validate()` 检查专家覆盖、槽位数量、rank 内重复、层顺序和运行配置。
- `build_placement_metadata()` 根据实际模型、配置和模块路径生成校验信息。
- `load_placement()` 解析文件并验证 schema。
- `save_placement()` 使用临时文件和原子替换，避免读到半个文件。

文件保存一份权威映射，反向映射和副本数可以据此重新推导。因此不必
保存三份可能互相矛盾的表。文件也不保存权重，静态启动仍然需要原模型。

模型结构摘要不能证明 checkpoint 内容相同。如果你覆盖了同一路径中的
权重，需要自己保证校准和静态运行使用同一份 checkpoint。

先读 `test_placement.py`。尝试手工构造缺失专家、错误 EP 数量、同 rank
重复专家等输入，再对照错误信息，就能掌握部署表的约束。

## 3. 动态导出为什么要先完成提交

入口依次是：

```text
LLM.collective_rpc("export_eplb_placement", ...)
  -> NPUWorker.export_eplb_placement()
  -> AscendEPLBController.export_placement()
  -> export_committed_placement()
  -> AscendEplbState.finish_pending_commits()
```

在离线调用中，上一轮生成返回后再调用 RPC，期间不要提交新请求。
worker 在前向之间执行这个操作，因此不会有前向线程同时发布新规划。

后台计算出新方案、把权重搬到接收缓冲，都还不意味着新部署已经生效。
`finish_pending_commits()` 会继续推进已有结果，通过现有的上游提交入口：

```text
把缓冲中的权重安装到计算槽位
  -> 提交物理和逻辑映射
  -> 原地刷新路由表
  -> 确认缓冲已经消费，允许后台复用
```

它没有调用普通 `step()`，所以不会为了导出再启动一个新周期。
它也没有复用 `drain_async()`：后者服务于关闭或取消流程，可以消费结果
但不安装这些结果。

提交结束后，各 rank 对快照计算相同摘要。只有一致时，EP rank 0 才写文件。
文件解析或写入失败也会通过 CPU 进程组通知其他 rank。

注意，导出只保证部署已经生效，不保证调优已经收敛。流量太少时，得到的
可能仍然是初始部署。`workload` 用来记录文件产生于什么数据和请求条件。

## 4. 静态启动为什么采用一次迁移

核心实现在 `placement_runtime.py` 的 `install_placement()`。

直接按目标表从 checkpoint 加载，要求适配模型各自的权重加载入口、每层
映射和不同量化布局。这次首先复用已经存在的加载与迁移机制：

```text
现有方式加载模型
  -> 校验目标文件和所有 rank 的初始部署
  -> 确定每个目标槽位的准确来源
  -> 搬运到缓冲
  -> 安装权重、提交映射、刷新路由
  -> 固定运行
```

来源由 `plan_placement_sources()` 确定。它优先使用原位置，其次使用同 rank
的其他副本，最后选择远端副本。这里执行的是文件指定的准确部署，不能
重新调用负载均衡算法计算另一个方案。

举例说明缓冲的重要性：

```text
旧槽位：[专家0, 专家1]
新槽位：[专家1, 专家0]
```

如果先直接覆盖槽位0，专家0的旧权重就丢了。正确做法是把两个来源先保存
到缓冲，之后再覆盖目标。跨 rank 时还需要确保所有读取已经完成，所以
代码在覆盖前同步并进行 barrier。

`stage_explicit_layer_transfer()` 只负责准备缓冲；真正的安装由上游
`move_from_buffer()` 完成。随后 `_commit_layer_mapping()` 推导并写回逻辑
映射和副本数，`refresh_placement_layer()` 刷新路由。

量化权重通过现有 `get_eplb_weight_views()` 暴露的视图移动。控制器不必
知道某种量化使用几个 scale、是普通 Tensor 还是 Tensor List；它移动的
正是计算核已经使用的存储。

## 5. 静态为什么仍然需要 enable_eplb

`enable_eplb` 参与专家槽位分配和路由构造。静态模式仍然需要这些能力，
因此新增 `mode` 来控制运行生命周期，而不修改这个开关。

```text
mode=dynamic: 使用原动态生命周期
mode=static: 初始化安装一次，保留映射，关闭持续动态工作
```

静态控制分布在三个位置：

- 控制器跳过 `prepare_forward()` 和 `step()` 的动态工作。
- 状态层跳过动态 step、profile rearrange 和后台 worker 启动。
- 层状态的 `load_recording_enabled=False` 直接跳过采集，包括 AllGather
  的统计辅助流程。

第三项在图捕获前设置，服务期间不变，因此采集工作不会被捕获到静态图中。
这与 Auto 的运行时采集切换不同；不能直接用它设计 Auto 状态机。

路由表仍然原地更新，保持 Tensor 对象稳定，避免图或长期存活的 router
继续引用旧对象。

## 6. 你可以怎样验证和继续学习

建议依次阅读并运行：

1. `test_placement.py`：格式、模型匹配、源槽位选择、文件原子写入。
2. `test_placement_runtime.py`：实际 CPU Tensor 的交换、复制、映射提交。
3. `test_placement_gloo.py`：两个进程通过真实 CPU Gloo 搬运并导出。
4. `test_placement_state.py`：完成提交与取消的区别、静态采集旁路、路由身份。
5. `test_eplb_static.py`：控制器的静态生命周期。
6. 四卡 `test_mrv2_eplb_placement.py`：动态校准、导出、静态重启、输出检查。

在准备好的 vLLM Ascend Linux 测试环境中：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q \
  tests/ut/distributed/eplb/test_placement*.py \
  tests/ut/worker/v2/test_eplb_static.py

pytest -sv tests/e2e/pull_request/four_card/test_mrv2_eplb_placement.py
```

CPU 验证不能证明 HIXL、HCCL、NZ 格式和 NPU 图执行正确。合并前还需要
在真机覆盖已有支持的通信路径、实际发生专家变更的部署，以及关闭、静态、
动态模式的性能对照。

首版要求 PP=1、关闭推测解码且只有一个目标 MoE 模型；保留上游兼容缓冲，
没有宣称最小显存占用。用户命令见 EPLB 用户指南，完整事务设计见同目录
`mrv2_eplb_placement.md`。
